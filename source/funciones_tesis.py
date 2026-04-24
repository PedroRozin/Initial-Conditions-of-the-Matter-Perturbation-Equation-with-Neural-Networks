import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classy import Class
from itertools import product
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import torch
import torch.nn as nn

#%% Red Neuronal

class RegressionNN(nn.Module):
    """ Neural network for regression with 4 input features and 2 output targets.
        Está pensada originalmente para que los features sean: a, Omega_m, kh (o k solo), h y los targets delta_m y delta_prime_m.
        Architecture:
        - Input layer: 4 neurons (features)
        - Hidden layers: 4 layers with 128 neurons each, ReLU activation
        - Hidden layer: 1 layer with 64 neurons, ReLU activation
        - Output layer: 2 neurons (targets)
        """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # salida de 2 targets
        )
        
    def forward(self, x):
        return self.network(x)
    
class ImprovedRegressionNN(nn.Module):
    def __init__(self, activation='tanh'):
        super().__init__()
        
        if activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            act_fn,
            nn.Linear(128, 128),
            act_fn,
            nn.Linear(128, 128),
            act_fn,
            nn.Linear(128, 128),
            act_fn,
            nn.Linear(128, 64),
            act_fn,
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.network(x)



#%% Func. generales

def compute_delta_m(delta_cdm, delta_b, omega_cdm, omega_b):
  """
  Compute the total matter density perturbation from CDM and baryonic perturbations.

  Args:
    delta_cdm (float): CDM density perturbation.
    delta_b (float): Baryonic density perturbation.
    omega_cdm (float): Omega_cdm value.
    omega_b (float): Omega_b value.

  Returns:
    array or float128: Total matter density perturbation.
  """
  result = (omega_cdm * delta_cdm + omega_b * delta_b) / (omega_cdm + omega_b)
  if isinstance(result, (int, float)):
    result = np.float128(result)
  else:
    result = np.array(result, dtype='float128')
  return result

def k_horizon(a_ini=.01, omega_m=0.3, omega_r=9.1e-5, c=299792458):
  """Calculate the comoving horizon scale dados los omegas que le pongamos y el a_ini.
  Returns:
    float: The comoving horizon scale in h/Mpc (multiply by h in case its needed).
  """
  omega_l= 1-omega_m-omega_r
  k_val = 2 * np.pi * a_ini * 100 / c * np.sqrt(omega_m / a_ini**3 + omega_r / a_ini**4 + omega_l)
  return k_val

def deriv_tau_to_a(df: pd.DataFrame, column_name='delta_dot_cdm')-> pd.DataFrame:
    """
    apply chain rule to convert tau to a.
    d delta/da = d delta/dtau * d tau/da = dot(delta)/(H a)
    Args:
        df (DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to convert. Default is 'delta_dot_cdm'.
    Returns:
        DataFrame: DataFrame with the new column added.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if column_name == 'delta_dot_cdm':
      df['delta_prime_cdm'] = (df[column_name] / (df['H'] * df['a'])).astype('float128')
      return df
    if column_name == 'delta_dot_b':
      df['delta_prime_b'] = (df[column_name] / (df['H'] * df['a'])).astype('float128')
      return df
    
def percent_diff_vs_class(a_vec, delta_m, a_class, delta_m_class):
    """
    Interpola la curva de CLASS (o cualquiera) sobre los puntos de integración y calcula el porcentaje de diferencia relativa.
    Si hay valores duplicados en a_class, solo toma el primero.
    """
    a_class = np.asarray(a_class)
    delta_m_class = np.asarray(delta_m_class)
    # eliminar duplicados si existen
    if len(np.unique(a_class)) < len(a_class):
        _, unique_indices = np.unique(a_class, return_index=True)
        a_class = a_class[unique_indices]
        delta_m_class = delta_m_class[unique_indices]
    interp_class = interp1d(a_class, delta_m_class, kind='quadratic',
                            bounds_error=False, fill_value= "extrapolate")
    delta_m_class_interp = interp_class(a_vec)
    # percent_diff = 100 * (delta_m - delta_m_class_interp) / delta_m_class_interp
    percent_diff = np.array(100 * (delta_m - delta_m_class_interp) / delta_m)
    return percent_diff

def percent_dif_matter_cdm(a_vec, delta_m, a_class, delta_m_class):
    """
    Interpola la curva de CLASS (o cualquiera) sobre los puntos de integración y calcula el porcentaje de diferencia relativa.
    Si hay valores duplicados en a_class, solo toma el primero.
    """
    a_class = np.asarray(a_class)
    delta_m_class = np.asarray(delta_m_class)
    # Solo eliminar duplicados si existen
    if len(np.unique(a_class)) < len(a_class):
        _, unique_indices = np.unique(a_class, return_index=True)
        a_class = a_class[unique_indices]
        delta_m_class = delta_m_class[unique_indices]
    interp_class = interp1d(a_class, delta_m_class, kind='quadratic',
                            bounds_error=False, fill_value="extrapolate")
    delta_m_class_interp = interp_class(a_vec)
    # percent_diff = 100 * (delta_m - delta_m_class_interp) / delta_m_class_interp
    percent_diff = np.array(100 * (delta_m - delta_m_class_interp) / delta_m)
    return percent_diff

#%%  Func. de CLASS

def common_settings(k=0.01, omega_m=.3, A_s=2.e-9, h=0.68): 
  """
  Set common settings for the CLASS simulation.
  Esto hace basicamente lo mismo que el diccionario de common_settings del principio (el de Julien),
    pero con los valores de k, omega_cdm, A_s y h como argumentos.
  Args:
    k (float): Value of k for the simulation.
    omega_cdm (float): Omega_cdm value for the simulation.
    A_s (float): A_s value for the simulation.
    h (float): h value for the simulation.

  Returns:
    A dictionary containing the common settings for the simulation.
  """
  _common_settings = {
    'output': 'mPk',
    'k_output_values': k,
    'h': h,
    # 'Omega_b': omega_m/6,
    # 'Omega_cdm': omega_cdm,
    'Omega_m': omega_m,
    'A_s': A_s,
    'n_s': 0.965,
    'tau_reio': 0.05430842,
    'YHe': 0.2454,
    'compute damping scale': 'yes',
    'gauge': 'newtonian'
    }
  M = Class()
  M.set(_common_settings)
  M.compute()
  return M

def get_df_from_M(M): #tal vez lo modifique para obtener las derivadas directamente acá. sería útil para la grilla.
  """
  Extracts 'delta_cdm', 'a', and 'tau [Mpc]' vectors from the perturbations.
  It does not extract inputs of the perturbation dictionary (like A_s, h, omega...)

  Args:
    class: M - (example: common_settings(k=1)).

  Returns:
    A Pandas DataFrame containing the desired vectors.
  """
  all_k = M.get_perturbations()
  one_k = all_k['scalar'][0]
  a = one_k['a']
  tau = one_k['tau [Mpc]']
  delta_cdm = one_k['delta_cdm']
  delta_b = one_k['delta_b']
  df = pd.DataFrame({'tau [Mpc]': tau, 'a': a, 'delta_cdm': delta_cdm, 'delta_b': delta_b})
  #rename tau
  df.rename(columns={'tau [Mpc]': 'tau'}, inplace=True)
  return df

def get_sigma8(M):
  """
  Extracts the current value of sigma8 from the perturbations dictionary.

  Args:
    dicc: The perturbations dictionary (default: all_k['scalar'][0]).

  Returns:
    The current value of sigma8.
  """

  _sigma= M.get_current_derived_parameters(['sigma8'])
  return _sigma['sigma8']
#%% Func. de file adhoc que sale de perturbations

def read_adhoc_txt(file_path = '/home/pedrorozin/scripts/delta_prime_cdm.txt'):
  """
  Función que parece simple, pero es clave. Trae todo lo que queremos a menos de la derivada respecto a 'a'.
  
  Args:
    file_path (str): Path to the text file.
  
  Returns:
    DataFrame: df con deltas y deltas dot (cdm y barionica) y a, k y H. NO incluye las derivadas respecto a 'a'; somente a tau.
  """
  # Especificar dtype para mayor precisión en las perturbaciones
  dtype_dict = {
    'delta_cdm': 'float128',
    'delta_dot_cdm': 'float128', 
    'delta_b': 'float128',
    'delta_dot_b': 'float128',
    'a': 'float64',
    'k': 'float64',
    'H': 'float64'
  }
  
  df = pd.read_csv(file_path, sep=' ', names=['delta_cdm', 'delta_dot_cdm', 'delta_b', 'delta_dot_b', 'a', 'k', 'H'], dtype=dtype_dict)
  return df

def filter_adhoc_by_k(df, k_value):
    """
    Filtra el DataFrame por un valor específico de k.

    Args:
        df (DataFrame): El DataFrame a filtrar.
        k_value (float): El valor de k por el cual filtrar.

    Returns:
        DataFrame: El DataFrame filtrado.
    """
    df_filtered = df[df['k'] >= k_value].copy()
    df_filtered.sort_values(by='k', inplace=True)
    #min k
    if not df_filtered.empty:
        min_k = df_filtered['k'].min()
        df_filtered = df[df['k'] == min_k]

    return df_filtered

#%% Funciones para obtener resultados desde el output de CLASS 

def set_column_names(df, path):
    """ Set column names for the DataFrame based on the first comment line in the file.
    Renombra a 'tau' if 'tau[Mpc]' is present.
	Args:
		df (DataFrame): The DataFrame to set column names for.
		path (str): Path to the file from which to read the column names.
	Returns:
		DataFrame: The DataFrame with updated column names.
    """
    with open(path) as f:
        for line in f:
            if line.startswith('#') and ':' in line:
                columns = [col.split(':')[1].strip() for col in line[1:].split('  ') if ':' in col]
                df.columns = columns
                break
    if 'tau[Mpc]' in df.columns:
        df.rename(columns={'tau[Mpc]': 'tau'}, inplace=True)
    return df

def read_ini_params(filepath):
    '''
    Obtiene los parameters k, omega_b y omega cdm from an .ini file and returns them as a dictionary.
    Args:
        filepath (str): Path to the .ini file.
    Returns:
        dict: Dictionary containing the parameters.
    '''
    params = {
        "k_output_values": None,
        "Omega_b": None,
        "Omega_cdm": None,
        'h': None,
        'A_s': None,
        'n_s': None,
        'tau_reio': None,
        'YHe': None
    }
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith(";") or line == "": #evito los comentarios y líneas vacías
                continue
            if "=" in line:
                key, value = [x.strip() for x in line.split("=", 1)] #separo valores
                if key in params:
                    params[key] = float(value) #agrego al dicc # type: ignore
    return params

#%% Clase para integraciones numéricas
class NumericalIntegrator:
    """
    Clase para realizar integraciones numéricas de las perturbaciones cosmológicas.
    
    Attributes:
        Om_m_0 (float): Matter density parameter
        sigma8 (float): Sigma8 parameter (opcional, usado para futuras extensiones)
        Om_r (float): Radiation density parameter
        a_0 (float): Initial scale factor
        a_f (float): Final scale factor
        method (str): Integration method for solve_ivp
        atol (float): Absolute tolerance
        rtol (float): Relative tolerance
        n_points (int): Number of points for integration
        
        # Results (populated after calling integrate())
        a_vec (array): Scale factor array
        delta (array): Delta values
        delta_prime (array): Delta prime values
    """
    
    def __init__(self, Om_m_0, sigma8=None, Om_r=9.045385269436243e-05, 
                 a_0=0.01, a_f=1.0, method='RK45', atol=1e-10, rtol=1e-8, n_points=20000):
        """
        Initialize the numerical integrator.
        
        Args:
            Om_m_0 (float): Matter density parameter
            sigma8 (float): Sigma8 parameter (opcional)
            Om_r (float): Radiation density parameter
            a_0 (float): Initial scale factor
            a_f (float): Final scale factor
            method (str): Integration method
            atol (float): Absolute tolerance
            rtol (float): Relative tolerance
            n_points (int): Number of integration points
        """
        self.Om_m_0 = Om_m_0
        self.sigma8 = sigma8 if sigma8 is not None else 0.8  # Default value
        self.Om_r = Om_r
        self.Om_L = 1 - Om_m_0 - Om_r
        self.a_0 = a_0
        self.a_f = a_f
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.n_points = n_points

        # declaro variables. se calculan después de la integración
        self.a_vec = None
        self.delta = None
        self.delta_prime = None
        
    def Hh(self, a):
        """
        Calcula el Hubble parameter dado 'a'. Está normalizado a 1 en a=1 (H_0=1).
        
        Args:
            a (float or array-like): The scale factor
            
        Returns:
            float or array-like: The Hubble parameter at the given scale factor 'a'
        """
        return np.sqrt(self.Om_L + self.Om_m_0/a**3 + self.Om_r/a**4)
    
    def Hh_prime(self, a):
        """
        Calcula la derivada del Hubble parameter con respecto a 'a'.
        
        Args:
            a (float or array-like): The scale factor
            
        Returns:
            float or array-like: The derivative of the Hubble parameter with respect to 'a'
        """
        num = (3*self.Om_m_0/a**4 + 4*self.Om_r/a**5)
        den = 2*np.sqrt(self.Om_L + self.Om_m_0/a**3 + self.Om_r/a**4)
        return -num/den
    
    def _differential_equation(self, a, X):
        """
        Define la ecuación diferencial para delta_m.
        
        Args:
            a (float): Scale factor
            X (array): [delta, delta_prime]
            
        Returns:
            array: [delta_prime, delta_double_prime]
        """
        f1 = X[1]  # delta_prime
        term1 = X[0] * 3 * self.Om_m_0 / (2 * (self.Hh(a) ** 2) * (a ** 5))
        term2 = -X[1] * ((3 / a) + (self.Hh_prime(a) / self.Hh(a)))
        f2 = term1 + term2  # delta_double_prime
        return np.array([f1, f2])
    
    def integrate(self, delta_0, delta_prime_0):
        """
        Integra la ecuación diferencial para delta_m.
        
        Args:
            delta_0 (float): Valor inicial de delta_m
            delta_prime_0 (float): Valor inicial de la derivada de delta_m
            
        Returns:
            tuple: (a_vec, delta_values, delta_prime_values)
        """
        self.a_vec = np.linspace(self.a_0, self.a_f, self.n_points)
        
        solution = solve_ivp(
            fun=self._differential_equation,
            t_span=[self.a_0, self.a_f],
            y0=np.array([delta_0, delta_prime_0]),
            t_eval=self.a_vec,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol
        )
        
        if not solution.success:
            print(f"Warning: solve_ivp did not converge with method {self.method}: {solution.message}")
        
        self.delta = solution.y[0]
        self.delta_prime = solution.y[1]
        
        return self.a_vec, self.delta, self.delta_prime
    
    def get_a(self):
        """
        Returnea el array de scale factors.
        
        Returns:
            array: Scale factor values
        """
        if self.a_vec is None:
            raise ValueError("Must call integrate() first")
        return self.a_vec
    
    def get_delta(self):
        """
        Returnea el array de delta values.
        
        Returns:
            array: Delta values
        """
        if self.delta is None:
            raise ValueError("Must call integrate() first")
        return self.delta
    
    def get_delta_prime(self):
        """
        Returnea el array de delta_prime values.
        
        Returns:
            array: Delta prime values
        """
        if self.delta_prime is None:
            raise ValueError("Must call integrate() first")
        return self.delta_prime
    
    def get_results_df(self):
        """
        Returnea los resultados como DataFrame.
        
        Returns:
            DataFrame: DataFrame con columnas 'a', 'delta', 'delta_prime'
        """
        if self.a_vec is None or self.delta is None or self.delta_prime is None:
            raise ValueError("Must call integrate() first")
        
        return pd.DataFrame({
            'a': self.a_vec,
            'delta': self.delta,
            'delta_prime': self.delta_prime
        })

## esto ya no es más de la clase
def get_delta_cdm_vs_a(params, delta_0, delta_prima_0, a_0=0.01, a_f=1, method='RK45',
                      atol=1e-10, rtol=1e-8, omr=9.045385269436243e-05):
    """
    Función de compatibilidad - usa la nueva clase internamente.
    
    Args:
        params: Tuple (Om_m, sigma8)
        delta_0: Valor inicial de delta_m
        delta_prima_0: Valor inicial de la derivada de delta_m
        a_0, a_f: Scale factors inicial y final
        method: Método de integración
        atol, rtol: Tolerancias
        omr: Parámetro de radiación
        
    Returns:
        tuple: (a_vec, delta_num)
    """
    Om_m_0, sigma8 = params
    integrator = NumericalIntegrator(Om_m_0, sigma8, omr, a_0, a_f, method, atol, rtol)
    a_vec, delta_num, _ = integrator.integrate(delta_0, delta_prima_0)
    return a_vec, delta_num

#%%