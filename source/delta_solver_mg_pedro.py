from multiprocessing.pool import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, simpson
from scipy.constants import c as c_ms
import torch
import joblib
import pandas as pd
from funciones_tesis import ImprovedRegressionNN



# Constantes cosmológicas por defecto
DEFAULT_OM_R_0 = 9.4e-5  # Omega_radiación hoy
# DEFAULT_OM_R_0 = 5.38e-5  # Omega_radiación hoy
DEFAULT_OM_M_0 = 0.305
DEFAULT_C = c_ms/1000  # Velocidad de la luz
DEFAULT_Z_F = 0  # Redshift final
DEFAULT_Z_INI_HS = 20  # Redshift inicial para el modelo HS
DEFAULT_Z_0 = 30  # Redshift inicial para la integración delta (cambiar a 25)
default_b=0.1
default_k=0.01
default_h = .68

# funciones globales para paralelización
# por alguna razón, para paralelizar no puedo tener las integraciones como métodos de la clase
# se usan para calcular las deltas de manera paralela y sigma8 (se integra en k para cada z)
def _process_k_parallel(args):
    """
    Función global para procesar cada k en paralelo.
    Necesita ser global para que multiprocessing pueda serializar (no termino de entender por qué).
    """
    i, k_val, h, Om_m_0, b, z_ini_HS, z_0, delta_ini_i, delta_p_ini_i, a_vec = args
    try:
        from scipy.integrate import solve_ivp
        solver_k = DeltaSolver(k=k_val, h=h, Om_m_0=Om_m_0, 
                              b=b, z_ini_HS=z_ini_HS, z_0=z_0)
        y0 = [delta_ini_i, delta_p_ini_i]
        
        # Resolver usando exactamente la misma grilla a_vec para todos los k's
        H_num, H_prime_num, r_num = solver_k.H_HS()
        
        def delta_wrapper(a, y):
            return solver_k.delta_mg(a, y, H_num, H_prime_num, r_num)
        
        a_ini = a_vec[0]
        a_fin = a_vec[-1]
        
        # Resolver con la grilla común a_vec
        sol_MG = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                          t_eval=a_vec, method='RK45',
                          atol=1e-12, rtol=1e-10)
        
        delta_result = sol_MG.y[0]
        delta_p_result = sol_MG.y[1]
        
        return i, delta_result, delta_p_result
    except Exception as e:
        print(f"Error en k={k_val}: {e}")
        return i, np.full_like(a_vec, np.nan), np.full_like(a_vec, np.nan)


def _process_k_parallel_lcdm(args):
    """
    Función global para procesar cada k en paralelo para LCDM.
    """
    i, k_val, h, Om_m_0, z_0, delta_ini_i, delta_p_ini_i, a_vec = args
    try:
        from scipy.integrate import solve_ivp
        solver_k = DeltaSolver(k=k_val, h=h, Om_m_0=Om_m_0, z_0=z_0)
        y0 = [delta_ini_i, delta_p_ini_i]
        
        def delta_wrapper(a, y):
            return solver_k.delta_lcdm(a, y)
        
        a_ini = a_vec[0]
        a_fin = a_vec[-1]
        
        # Resolver con la grilla común a_vec
        sol_LCDM = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                            t_eval=a_vec, method='RK45',
                            atol=1e-12, rtol=1e-10)
        
        delta_result = sol_LCDM.y[0]
        delta_p_result = sol_LCDM.y[1]
        
        return i, delta_result, delta_p_result
    except Exception as e:
        print(f"Error en k={k_val} (LCDM): {e}")
        return i, np.full_like(a_vec, np.nan), np.full_like(a_vec, np.nan)


def _simpson_chunk(args):
    """Función auxiliar para paralelizar simpson
    Args:
         args: tupla (y_chunk, x_chunk). pensada para recibir de multiprocessing
         

    """

    y_chunk, x_chunk = args
    return simpson(y_chunk, x=x_chunk)


class DeltaSolver:
    def __init__(self, Om_m_0=DEFAULT_OM_M_0, Om_r_0=DEFAULT_OM_R_0, z_f=DEFAULT_Z_F,
                z_ini_HS=DEFAULT_Z_INI_HS, z_0=DEFAULT_Z_0, c=DEFAULT_C,b=default_b,k=default_k, h=default_h):
        """
        Inicializa el solucionador con los parámetros cosmológicos
        """
        self.Om_m_0 = Om_m_0
        self.Om_r_0 = Om_r_0
        self.Om_l_0 = 1 - Om_m_0 - Om_r_0
        self.z_f = z_f
        self.z_ini_HS = z_ini_HS
        self.z_0 = z_0
        self.c = c
        self.b = b
        self.k = k
        self.h = h
        self.a_0 = 1/(1 + z_0)  

    def H_LCDM(self,z):#H_cdm/H0
        return np.sqrt(self.Om_r_0 * (1+z)**4 + self.Om_m_0 * (1+z)**3 + self.Om_l_0)


    def dH_dz_lcdm(self,z):#dH_cdm/dz/H0
        H_z_over_H0=self.H_LCDM(z)
        dH=1/2*(3*self.Om_m_0*(1+z)**2 + 4*self.Om_r_0*(1+z)**3)/H_z_over_H0
        return dH
    
# Friedmann equations for f(R)

    @staticmethod
    def Gamma(r, b):
        return (r + b) * ((r + b)**2 - 2*b) / (4*b*r)
    

    
    def F(self, z, X):
        x, y, v, Om, r, Or = X
        denom = (z + 1)
        g = self.Gamma(r, self.b)
        f1 = (-Om - 2*v + x + 4*y + x*v + x**2) / denom
        f2 = -(v*x*g - x*y + 4*y - 2*y*v) / denom
        f3 = -v * (x*g + 4 - 2*v) / denom
        f4 = Om * (-1 + 2*v + x) / denom
        f5 = -r * g * x / denom
        f6 = Or*(2*v+x)/denom
        return np.array([f1, f2, f3, f4, f5, f6])
    
    def condiciones_iniciales(self):
        """Condiciones iniciales en z0"""
        eta = self.Om_m_0 * (1 + self.z_ini_HS)**3 + self.Om_r_0 * (1 + self.z_ini_HS)**4 + self.Om_l_0
        y0 = np.zeros(6)
        y0[0] = 0.0
        y0[1] = (self.Om_m_0 * (1 + self.z_ini_HS)**3 + 2 * self.Om_l_0) / (2 * eta)
        y0[2] = (self.Om_m_0 * (1 + self.z_ini_HS)**3 + 4 * self.Om_l_0) / (2 * eta)
        y0[3] = self.Om_m_0 * (1 + self.z_ini_HS)**3 / eta
        y0[4] = (self.Om_m_0 * (1 + self.z_ini_HS)**3 + 4 * self.Om_l_0) / self.Om_l_0
        y0[5] = self.Om_r_0 * (1 + self.z_ini_HS)**4 / eta
        return y0
    
    def integ_HS(self):
        """Integra las ecuaciones de Friedmann de HS"""
        M = 10000
        z_array = np.linspace(self.z_ini_HS, self.z_f, M)
        y0 = self.condiciones_iniciales()
        
        # Necesitamos pasar self.F como una función que solo toma z, X y b
        def f_wrapper(z, X):
            return self.F(z, X)
        
        #args = (b,)
        sol = solve_ivp(f_wrapper, [self.z_ini_HS, self.z_f], y0, t_eval=z_array, 
                        method='RK45', rtol=1e-12, atol=1e-10)
        
        z_vec = sol.t
        x_vec = sol.y[0]
        v_vec = sol.y[2]
        r_vec = sol.y[4]

        return z_vec, x_vec, v_vec, r_vec
    
    def dH_dz(self, z, r, v, x):
        denom = (z + 1)
        g = self.Gamma(r, self.b)
        dv_dz = -v * (x*g + 4 - 2*v) / denom
        dr_dz = -r * g * x / denom
        coeff1 = 1/4 * (1 - self.Om_m_0)
        coeff2 = np.sqrt(r/(2*v) * (1 - self.Om_m_0))
        dH = coeff1/coeff2 * (dr_dz/v - r/v**2 * dv_dz)
        return dH
    
    def H_HS(self,z_ini_HS=None):
        """Calcula H_HS, dH_HS/dz, R_HS. Devuelve H(z)/H0"""
        if z_ini_HS is None:
            z_ini_HS = self.z_ini_HS
            
        z_vec, x_vec, v_vec, r_vec = self.integ_HS()
        H_HS_2 = np.sqrt((r_vec/(2*v_vec)) * (1 - self.Om_m_0))
        dHdz = self.dH_dz(z_vec, r_vec, v_vec, x_vec)
        H_prime = -dHdz * (1+z_vec)**2
        
        # Usar extrapolate en lugar de bounds_error=True
        H_interp = interp1d(z_vec, H_HS_2)
        Hprime_interp = interp1d(z_vec, H_prime)
        r_interp = interp1d(z_vec, r_vec)
        
        return H_interp, Hprime_interp, r_interp

    def Geff(self, z, r_val):
        """Calcula G_efectivo"""
        c = self.c
        h= self.h
        is_scalar = np.isscalar(z)
        z_array = np.atleast_1d(z)
        lamb = 3 * (h*100)**2 * (1 - self.Om_m_0) / c**2  # aca va H0**2 (ahora esta puesto =100 porque k se entra en esas unidades) CAMBIADO
        result = np.zeros_like(z_array)

        for i, z_val in enumerate(z_array):
            r = r_val(z_val) if callable(r_val) else r_val
            FR_num = 1 - 2/(self.b * (1 + r/self.b)**2)
            FRR_num = 4/(lamb * self.b**2 * (1 + r/self.b)**3)
            m = FRR_num/FR_num
            eps = m * self.k**2 * (1+z_val)**2
            result[i] = 1/FR_num * (1 + 1/(3 + 1/eps))

        return result[0] if is_scalar else result
    
    def delta_mg(self, a, y, H_num, H_prime_num, r_num):
        """Ecuación para evolución de delta"""
        z = 1/a - 1
        
        if z > self.z_ini_HS:
            H = self.H_LCDM(z)
            dH = -self.dH_dz_lcdm(z) * (1+z)**2
            term1 = dH/H + 3/a
            term2 = (3 * self.Om_m_0) / (2 * H**2 * a**5)
        else:
            H = H_num(z)
            dH = H_prime_num(z)
            r_val=r_num(z)
            term1 = dH/H + 3/a
            term2 = (3 * self.Om_m_0) / (2 * H**2 * a**5) * self.Geff(z, r_val)
        
        res1 = y[1]
        res2 = -term1 * y[1] + term2 * y[0]
        
        return [res1, res2]
    
    # terminos de la ecuacion
    
    def terminos(self, a, H_num, H_prime_num, r_num):

        term1 = []
        term2 = []
        term3 = []
        term4=  []
        for ai in a:
            zi = 1/ai - 1
            nini=abs(np.log(1e-3))
            if zi > self.z_ini_HS:
                H = self.H_LCDM(zi)
                dH = -self.dH_dz_lcdm(zi) * (1+zi)**2
                term1.append(dH/H + 3/ai)
                term2.append((3 * self.Om_m_0) / (2 * H**2 * ai**5)) 
                term3.append(nini*(dH/H*ai))
                term4.append((nini**2*3*self.Om_m_0)/(2*H**2*ai**3))
            else:
                H = H_num(zi)
                dH = H_prime_num(zi)
                r_val=r_num(zi)
                term1.append(dH/H + 3/ai)
                term2.append((3 * self.Om_m_0) / (2 * H**2 * ai**5) * self.Geff(zi, r_val))
                term3.append(nini*(dH/H*ai))
                term4.append((nini**2*3*self.Om_m_0)/(2*H**2*ai**3)*self.Geff(zi, r_val))
        return np.array(term1), np.array(term2),np.array(term3), np.array(term4)

    def red_para_condiciones_iniciales(self, parametros = [0.03, 0.1, 0.68, 0.3], name ='tanh_buena_v2'):
        """ condiciones iniciales a partir de la red neuronal entrenada"""
        device = torch.device('cpu')
        torch.manual_seed(0)
        model = ImprovedRegressionNN(activation='tanh')
        folder_path = str(name)
        network_name = f'_{folder_path}'
        path_model = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/regression_model{network_name}.pth'
        path_scaler_X = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/scaler_X{network_name}.pkl'
        path_scaler_y = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/scaler_y{network_name}.pkl'
        model.load_state_dict(torch.load(path_model))  
        model.eval()
        scaler_X = joblib.load(path_scaler_X)
        scaler_y = joblib.load(path_scaler_y)
        if type(parametros) is dict:
            a = np.array([parametros['a']])
            kh = np.array([parametros['k h']])
            h = np.array([parametros['h']])
            Omega_m = np.array([parametros['Omega_m']])
        elif type(parametros) is list or type(parametros) is np.ndarray:
            a = np.array([parametros[0]])
            kh = np.array([parametros[1]])
            h = np.array([parametros[2]])
            Omega_m = np.array([parametros[3]])
        elif type(parametros) is pd.DataFrame:
            a = np.array([parametros['a'].values[0]])
            kh = np.array([parametros['k h'].values[0]])
            h = np.array([parametros['h'].values[0]])
            Omega_m = np.array([parametros['Omega_m'].values[0]])
        X = np.column_stack((a, kh, h, Omega_m))
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_scaled = model(X_tensor).numpy()
        y = scaler_y.inverse_transform(y_scaled)
        delta_ini, delta_p_ini = y[0]
        return [delta_ini, delta_p_ini]
    

    def delta_lcdm(self, a, y):
        z = 1/a - 1
        """Ecuación para evolución de delta"""
        
        H = self.H_LCDM(z)
        dH = -self.dH_dz_lcdm(z) * (1+z)**2
        term1 = dH/H + 3/a
        term2 = (3 * self.Om_m_0) / (2 * H**2 * a**5)

        res1 = y[1]
        res2 = -term1 * y[1] + term2 * y[0]
        
        return [res1, res2]
    
    def solve_delta_lcdm(self, num_points=1000):
        """Resuelve la ecuación delta para parámetros dados"""
        # Preparar vector de factor de escala
        a_ini = self.a_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)

        # Condiciones iniciales
        # y0 = [a_ini, 1.0]  # Modo creciente en régimen de materia: δ ∝ a
        y0 = self.red_para_condiciones_iniciales(parametros = [a_ini, self.k, self.h, self.Om_m_0], name ='tanh')
        
        # Obtener interpoladores H
        H_num, H_prime_num, r_num = self.H_HS()
        
        # Función wrapper para delta_mg
        def delta_wrapper(a, y):
            return self.delta_lcdm(a, y)
        
        # Resolver la ecuación
        sol_lcdm = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                            t_eval=a_vec, method='RK45',
                            atol=1e-12, rtol=1e-10)
        
        a_num_mg = sol_lcdm.t
        delta_num_mg, delta_p_num_mg = sol_lcdm.y
        
        return a_num_mg, delta_num_mg, delta_p_num_mg
    
    def solve_delta_mg(self, num_points=1000):
        """Resuelve la ecuación delta para parámetros dados"""
        # Preparar vector de factor de escala
        a_ini = self.a_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)

        # Condiciones iniciales  (pedro modificar)
        # y0 = [a_ini, 1.0]  # Modo creciente en régimen de materia: δ ∝ a
        y0 = self.red_para_condiciones_iniciales(parametros = [a_ini, self.k, self.h, self.Om_m_0], name ='tanh')
        # Obtener interpoladores H
        H_num, H_prime_num, r_num = self.H_HS()
        
        # Función wrapper para delta_mg
        def delta_wrapper(a, y):
            return self.delta_mg(a, y, H_num, H_prime_num, r_num)
        
        # Resolver la ecuación
        sol_MG = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                            t_eval=a_vec, method='RK45',
                            atol=1e-12, rtol=1e-10)
        
        a_num_mg = sol_MG.t
        delta_num_mg, delta_p_num_mg = sol_MG.y
        
        return a_num_mg, delta_num_mg, delta_p_num_mg
    

        """Grafica los resultados de delta"""
        plt.figure(figsize=(10, 6))
        plt.plot(a_num_mg, delta_num_mg, label='Delta MG')
        plt.plot(a_num_mg, delta_p_num_mg, label='Delta prime MG')
        plt.xscale('log')
        plt.xlabel('Factor de escala (a)')
        plt.ylabel('Delta')
        plt.title(f'Evolución de delta con Om_m_0={self.Om_m_0}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def compute_f_with_f_k(self, z_vec, f_k_array, k_array=None, use_mg_hubble=True):
        """
        Calcula f(z) a partir de la transformada de Fourier de f(k,z).
        
        Basado en la fórmula de la imagen:
        f(z) = (1/2π²) ∫ k² f(k,z) sinc(k·x(z)) dk
        
        donde x(z) es la distancia comoving:
        x(z) = ∫_0^z c dz'/H(z')
        
        Para gravedad modificada, se usa H_HS(z) en lugar de H_LCDM(z).
        
        Args:
        -----
        z_vec : array
            Array de valores z correspondientes a las columnas de f_k_array
        f_k_array : array (n_k x n_z)
            Array de valores f(k,z) para todos los k's y z
        k_array : array (opcional)
            Array de valores k. Si None, usa un array default
        use_mg_hubble : bool
            Si True, usa H_HS para gravedad modificada. Si False, usa H_LCDM.

        Returns:
        --------
        f_z : array (n_z,)
            Array 1D de valores f(z) para cada redshift en z_vec
        """
        if k_array is None:
            k_array = np.logspace(-3, 1, f_k_array.shape[0])  # Default k range
        if f_k_array.shape[0] != len(k_array):
            raise ValueError("f_k_array debe tener la misma cantidad de filas que k_array.")
        
        # Calcular las distancias comoving x para cada z
        if use_mg_hubble:
            try:
                H_num, H_prime_num, r_num = self.H_HS()
                def H_function(z):
                    if np.isscalar(z):
                        return H_num(z) if z <= self.z_ini_HS else self.H_LCDM(z)
                    else:
                        result = np.zeros_like(z)
                        mask_mg = z <= self.z_ini_HS
                        result[mask_mg] = H_num(z[mask_mg])
                        result[~mask_mg] = self.H_LCDM(z[~mask_mg])
                        return result
            except:
                print("Warning: No se pudo calcular H_HS, usando H_LCDM")
                H_function = self.H_LCDM
        else:
            H_function = self.H_LCDM
        
        # Calcular distancias comoving x(z) = ∫_0^z c/H(z') dz' para cada z
        x_comoving = np.zeros_like(z_vec)
        c = self.c  
        for i, z in enumerate(z_vec):
            if z > 0:
                z_int = np.linspace(0, z, 1000)
                H_int = H_function(z_int)
                # Integrar c/H(z') dz'
                x_comoving[i] = simpson(c/H_int, x=z_int)
            else:
                x_comoving[i] = 0.0  # x(z=0) = 0
    
        def sinc_normalized(u):
            return np.sinc(u)  # np.sinc ya maneja bien el caso u=0
        
        # Calcular f(z) usando la transformada de Fourier con x(z) conocido
        f_z = np.zeros(len(z_vec))
        
        for i, (z, x_z) in enumerate(zip(z_vec, x_comoving)):
            # Para cada z, usar su x(z) correspondiente
            # Integrando: k² f(k,z) sinc(k·x(z)) dk
            integrando = (k_array**2) * f_k_array[:, i] * sinc_normalized(k_array * x_z)
            
            # Integrar sobre k
            f_z[i] = simpson(integrando, x=k_array) / (2 * np.pi**2)
        
        return f_z
    

class VectorizedDeltaSolver:
    """
    Versión vectorizada de DeltaSolver que puede calcular múltiples k's simultáneamente
    usando GPU cuando esté disponible.
    """
    def __init__(self, k_array, Om_m_0=DEFAULT_OM_M_0, Om_r_0=DEFAULT_OM_R_0, 
                 z_f=DEFAULT_Z_F, z_ini_HS=DEFAULT_Z_INI_HS, z_0=DEFAULT_Z_0, 
                 c=DEFAULT_C, b=default_b, h=default_h, use_gpu=True):
        """
        Inicializa el solucionador vectorizado
        
        Parameters:
        -----------
        k_array : array-like
            Array de valores k para calcular simultáneamente
        use_gpu : bool
            Usar GPU cuando esté disponible
        """
        self.k_array = np.asarray(k_array)
        self.n_k = len(self.k_array)
        
        # parámetros cosmológicos
        self.Om_m_0 = Om_m_0
        self.Om_r_0 = Om_r_0
        self.Om_l_0 = 1 - Om_m_0 - Om_r_0
        self.z_f = z_f
        self.z_ini_HS = z_ini_HS
        self.z_0 = z_0
        self.c = c
        self.b = b
        self.h = h
        self.a_0 = 1/(1 + z_0)
        
        # Configurar dispositivo
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"VectorizedDeltaSolver usando: {self.device}")
        
        # Convertir arrays a tensores de PyTorch
        self.k_tensor = torch.tensor(self.k_array, dtype=torch.float64, device=self.device)
        
    def H_LCDM_tensor(self, z):
        """Función H_LCDM vectorizada usando tensores"""
        z_tensor = torch.tensor(z, dtype=torch.float64, device=self.device)
        return torch.sqrt(self.Om_r_0 * (1+z_tensor)**4 + self.Om_m_0 * (1+z_tensor)**3 + self.Om_l_0)
    
    def dH_dz_lcdm_tensor(self, z):
        """Derivada de H_LCDM vectorizada"""
        z_tensor = torch.tensor(z, dtype=torch.float64, device=self.device)
        H_z_over_H0 = self.H_LCDM_tensor(z)
        dH = 0.5 * (3*self.Om_m_0*(1+z_tensor)**2 + 4*self.Om_r_0*(1+z_tensor)**3) / H_z_over_H0
        return dH

    def Gamma_tensor(self, r: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Función Gamma vectorizada"""
        return (r + b) * ((r + b)**2 - 2*b) / (4*b*r)

    def condiciones_iniciales_vectorized(self, name='tanh_buena_v2'):
        """
        Condiciones iniciales vectorizadas para todos los k's usando la red neuronal
        """
        device_nn = torch.device('cpu')  # la red neuronal ENTRENADA funciona en CPU
        torch.manual_seed(0)
        model = ImprovedRegressionNN(activation='tanh')
        folder_path = str(name)
        network_name = f'_{folder_path}'
        path_model = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/regression_model{network_name}.pth'
        path_scaler_X = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/scaler_X{network_name}.pkl'
        path_scaler_y = f'/home/pedrorozin/scripts/outputs_pedro/neural_networks/{folder_path}/scaler_y{network_name}.pkl'
        
        model.load_state_dict(torch.load(path_model, map_location=device_nn))
        model.eval()
        scaler_X = joblib.load(path_scaler_X)
        scaler_y = joblib.load(path_scaler_y)
        
        # Crear batch para todos los k's
        a_batch = np.full(self.n_k, self.a_0)
        kh_batch = self.k_array.copy()  # k en unidades 1/Mpc
        h_batch = np.full(self.n_k, self.h)
        Omega_m_batch = np.full(self.n_k, self.Om_m_0)
        
        X_batch = np.column_stack((a_batch, kh_batch, h_batch, Omega_m_batch))
        X_scaled = scaler_X.transform(X_batch)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device_nn)
        
        with torch.no_grad():
            y_scaled = model(X_tensor).numpy()

        #ahora tengo un vector de CI's con un k asociado.
        y_batch = scaler_y.inverse_transform(y_scaled)
        delta_ini = y_batch[:, 0]  # CI's para delta
        delta_p_ini = y_batch[:, 1]  # CI's para delta_prime
        
        return delta_ini, delta_p_ini
    
    def Geff_vectorized(self, z, r_vals, use_tensor_ops=True):
        """
        G_efectivo vectorizado para todos los k's con versión optimizada.
        
        Parameters:
        -----------
        z : float
            Redshift
        r_vals : callable or float
            Valores de r (puede ser función o escalar)
        use_tensor_ops : bool
            Si usar operaciones tensoriales (más rápido para GPU/serial)
            o usar método simple (mejor para multiprocessing)
        """
        if use_tensor_ops and torch.cuda.is_available():
            # versión vectorizada original optimizada para GPU
            # TO DO: ver si andaría sin usar el GPU (creo q sí, pero creo q es más rápido dividir en jobs)
            try:
                if callable(r_vals):
                    r_val = r_vals(z)
                else:
                    r_val = r_vals
                
                # conversión a float de los elementos
                z_val = float(np.array(z).item())
                r_val_float = float(np.array(r_val).item())
                    
                # Convertir a tensores
                z_tensor = torch.tensor(z_val, dtype=torch.float64, device=self.device)
                r_tensor = torch.tensor(r_val_float, dtype=torch.float64, device=self.device)
                b_tensor = torch.tensor(self.b, dtype=torch.float64, device=self.device)
                
                lamb = 3 * (self.h * 100)**2 * (1 - self.Om_m_0) / self.c**2
                lamb_tensor = torch.tensor(lamb, dtype=torch.float64, device=self.device)
                
                # Calcular para todos los k's simultáneamente usando tensores
                FR_num = 1 - 2/(b_tensor * (1 + r_tensor/b_tensor)**2)
                FRR_num = 4/(lamb_tensor * b_tensor**2 * (1 + r_tensor/b_tensor)**3)
                m = FRR_num/FR_num
                eps = m * self.k_tensor**2 * (1+z_tensor)**2
                G_eff = 1/FR_num * (1 + 1/(3 + 1/eps))
                
                return G_eff.cpu().numpy()
                
            except Exception as e:
                print(f"Advertencia: Error en versión tensorizada de Geff, usando fallback: {e}")
                use_tensor_ops = False
        
        if not use_tensor_ops:
            # Versión simple que no usa GPU (mejor para multiprocessing)
            temp_solver = DeltaSolver(k=0.1, h=self.h, Om_m_0=self.Om_m_0, 
                                     b=self.b, z_ini_HS=self.z_ini_HS, z_0=self.z_0)
            
            G_eff = np.zeros_like(self.k_array)
            for i, k in enumerate(self.k_array):
                temp_solver.k = k
                G_eff[i] = temp_solver.Geff(z, r_vals)
            
            return G_eff
        
    def solve_delta_mg_vectorized_parallel(self, num_points=1000, n_jobs=16): #16 son la cantidad de cores que tengo yo
        """
        Versión paralelizada para resolver las ecuaciones delta para todos los k's
        usando multiprocessing para distribuir los cálculos de k's entre múltiples procesos.
        Es una bomba. Y, si hay GPU, más aun.
        Lo que hace es paralelizar sobre los k's que no se pueden vectorizar por el solve_ivp
        1. Crea una grilla común de a's para todos los k's
        2. Usa multiprocessing para resolver cada k en paralelo
        3. Recolecta y organiza los resultados
        Parameters:
        -----------
        num_points : int
            Número de puntos en la grilla de a
        n_jobs : int
            Número de procesos paralelos a usar. Depende de la cantidad de cores disponibles.
        Returns:
        --------
        a_vec : array
            Grilla común de factores de escala
        delta_results : array
            Resultados de delta para cada k (shape: n_k x len(a_vec))
        delta_p_results : array
            Resultados de delta_prime para cada k (shape: n_k x len(a_vec))
        -----------
        Note:
        Esta función requiere que la función `delta_mg` y los métodos relacionados
        sean compatibles con la paralelización y no dependan de estados compartidos.
        
        """
        from multiprocessing import Pool
        
        # Vector de factor de escala (igual para todos los k's)        
        a_ini = self.a_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)
        
        # condiciones iniciales vectorizadas
        delta_ini, delta_p_ini = self.condiciones_iniciales_vectorized()
        
        # Preparar argumentos para la función global
        args_list = []
        for i, k_val in enumerate(self.k_array):
            args = (i, k_val, self.h, self.Om_m_0, self.b, self.z_ini_HS, self.z_0, 
                   delta_ini[i], delta_p_ini[i], a_vec) #son todas las cosas que necesita el DeltaSolver
            args_list.append(args)
        
        print(f"Resolviendo {self.n_k} valores de k en paralelo con {n_jobs} procesos...")
        
        # usar multiprocessing con la función global
        with Pool(n_jobs) as pool:
            results = pool.map(_process_k_parallel, args_list)
        
        # organizar resultados. obtengo una matriz n_k x len(a_vec)
        delta_results = np.zeros((self.n_k, len(a_vec))) 
        delta_p_results = np.zeros((self.n_k, len(a_vec)))
        
        for i, delta_interp, delta_p_interp in results:
            delta_results[i, :] = delta_interp
            delta_p_results[i, :] = delta_p_interp
        
        print("Resolución paralela completada.")
        return a_vec, delta_results, delta_p_results
    
    def solve_delta_lcdm_vectorized_parallel(self, num_points=1000, n_jobs=16):
        """
        Versión paralelizada para resolver las ecuaciones delta de LCDM para todos los k's
        usando multiprocessing para distribuir los cálculos de k's entre múltiples procesos.
        
        Parameters:
        -----------
        num_points : int
            Número de puntos en la grilla de a
        n_jobs : int
            Número de procesos paralelos a usar
        
        Returns:
        --------
        a_vec : array
            Grilla común de factores de escala
        delta_results : array
            Resultados de delta LCDM para cada k (shape: n_k x len(a_vec))
        delta_p_results : array
            Resultados de delta_prime LCDM para cada k (shape: n_k x len(a_vec))
        """
        from multiprocessing import Pool
        
        # Vector de factor de escala (igual para todos los k's)        
        a_ini = self.a_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)
        
        # condiciones iniciales vectorizadas
        delta_ini, delta_p_ini = self.condiciones_iniciales_vectorized()
        
        # Preparar argumentos para la función global LCDM
        args_list = []
        for i, k_val in enumerate(self.k_array):
            args = (i, k_val, self.h, self.Om_m_0, self.z_0, 
                   delta_ini[i], delta_p_ini[i], a_vec)
            args_list.append(args)
        
        print(f"Resolviendo LCDM para {self.n_k} valores de k en paralelo con {n_jobs} procesos...")
        
        # usar multiprocessing con la función global
        with Pool(n_jobs) as pool:
            results = pool.map(_process_k_parallel_lcdm, args_list)
        
        # organizar resultados
        delta_results = np.zeros((self.n_k, len(a_vec))) 
        delta_p_results = np.zeros((self.n_k, len(a_vec)))
        
        for i, delta_interp, delta_p_interp in results:
            delta_results[i, :] = delta_interp
            delta_p_results[i, :] = delta_p_interp
        
        print("Resolución paralela LCDM completada.")
        return a_vec, delta_results, delta_p_results
    
    def solve_delta_mg_vectorized(self, num_points=1000):
        """
        Resuelve las ecuaciones delta para todos los k's simultáneamente vectorizándolos.
        Lo que no puede vectorizar es el solve_ivp, que necesita un k a la vez.
        """
        # Vector de factor de escala (igual para todos los k's)
        a_ini = self.a_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)
        
        # Condiciones iniciales vectorizadas
        delta_ini, delta_p_ini = self.condiciones_iniciales_vectorized()
        
        # Preparar arrays de salida
        delta_results = np.zeros((self.n_k, len(a_vec)))
        delta_p_results = np.zeros((self.n_k, len(a_vec)))
        
        # Para cada k, necesitamos resolver individualmente porque solve_ivp no es vectorizable
        # Pero podemos optimizar con paralelización o GPU para las operaciones internas (tipo Geff)
        print(f"Resolviendo para {self.n_k} valores de k...")
        
        for i, k_val in enumerate(self.k_array):
            if i % 10 == 0:
                print(f"Progreso: {i}/{self.n_k} ({100*i/self.n_k:.1f}%)")
                
            # Crear solver individual para este k
            solver_k = DeltaSolver(k=k_val, h=self.h, Om_m_0=self.Om_m_0, 
                                  b=self.b, z_ini_HS=self.z_ini_HS, z_0=self.z_0)
            
            # Usar condiciones iniciales vectorizadas
            y0 = [delta_ini[i], delta_p_ini[i]]
            
            # Resolver
            try:
                # Resolver usando exactamente la misma grilla a_vec para todos los k's
                H_num, H_prime_num, r_num = solver_k.H_HS()
                
                def delta_wrapper(a, y):
                    return solver_k.delta_mg(a, y, H_num, H_prime_num, r_num)
                
                # Resolver con la grilla común a_vec
                sol_MG = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                                  t_eval=a_vec, method='RK45',
                                  atol=1e-12, rtol=1e-10)
                
                delta_results[i, :] = sol_MG.y[0]
                delta_p_results[i, :] = sol_MG.y[1]
                
            except Exception as e:
                print(f"Error en k={k_val}: {e}")
                delta_results[i, :] = np.nan
                delta_p_results[i, :] = np.nan
        print("Resolución completada.")
        return a_vec, delta_results, delta_p_results

    def solve_delta_lcdm_vectorized(self, num_points=1000):
        """
        Resuelve las ecuaciones delta de LCDM para todos los k's simultáneamente.
        """
        # Vector de factor de escala (igual para todos los k's)
        a_ini = self.a_0
        a_fin = 1
        a_vec = np.logspace(np.log10(a_ini), np.log10(a_fin), num_points)
        
        # Condiciones iniciales vectorizadas
        delta_ini, delta_p_ini = self.condiciones_iniciales_vectorized()
        
        # Preparar arrays de salida
        delta_results = np.zeros((self.n_k, len(a_vec)))
        delta_p_results = np.zeros((self.n_k, len(a_vec)))
        
        print(f"Resolviendo LCDM para {self.n_k} valores de k...")
        
        for i, k_val in enumerate(self.k_array):
            if i % 10 == 0:
                print(f"Progreso LCDM: {i}/{self.n_k} ({100*i/self.n_k:.1f}%)")
                
            # Crear solver individual para este k
            solver_k = DeltaSolver(k=k_val, h=self.h, Om_m_0=self.Om_m_0, z_0=self.z_0)
            
            # Usar condiciones iniciales vectorizadas
            y0 = [delta_ini[i], delta_p_ini[i]]
            
            # Resolver
            try:
                def delta_wrapper(a, y):
                    return solver_k.delta_lcdm(a, y)
                
                # Resolver con la grilla común a_vec
                sol_LCDM = solve_ivp(delta_wrapper, [a_ini, a_fin], y0,
                                    t_eval=a_vec, method='RK45',
                                    atol=1e-12, rtol=1e-10)
                
                delta_results[i, :] = sol_LCDM.y[0]
                delta_p_results[i, :] = sol_LCDM.y[1]
                
            except Exception as e:
                print(f"Error en k={k_val} (LCDM): {e}")
                delta_results[i, :] = np.nan
                delta_p_results[i, :] = np.nan
                
        print("Resolución LCDM completada.")
        return a_vec, delta_results, delta_p_results

    def compute_sigma8_vectorized(self, deltas= None, k=None, R=8.0, h=None, lcdm = False, z_start = 2, n_s=0.9665, A_s=2e-9, use_parallel=True, n_jobs=16):
        """
        Calcula σ8(z) para todos los k's en el redshift dado.
        Esto es, σ8(z) = [ ∫ (k^2 P(k,z) W^2(kR) dk) / (2π^2) ]^0.5,
        con P(k,z) = As (k/k_piv)^(n_s-1) * δ^2(k,z) y W(kR) la window function esférica.
        Como el solver está pensado para [k]=1/Mpc y R=8 Mpc/h, hay que convertir R a Mpc dividiendo por h y
        habría que dividir todo el resultado por h^3 para que quede en unidades de Mpc^3 (regla de la cadena en k).
        Esto NO está hecho: el resultado está a menos de esta normalización
        (hay que hacerlo a mano después para que quedé bien el orden de magnitud de la variable).
        Args:
            deltas : array de (n_k x n_a). If None, los calcula usando LCDM o MG (default MG)
                Array de δ(k,z) para todos los k's en el redshift dado. necesito integrarlo en k (axis = 0)
            k : array de n_k
                Array de valores k correspondientes a los deltas
            R : float
                Radio en Mpc/h para σ8 (default 8 Mpc/h)
            h : float
                Parámetro h (default 0.68)
            lcdm : bool
                Si True, usa LCDM para calcular deltas si no se pasan (default False)
            z_start: float or int
                Redshift inicial para 
            n_s : float
                Índice espectral primordial (default 0.9665)
            A_s : float
                Amplitud del espectro primordial (default 2e-9)
            use_parallel : bool
                Usar paralelización para la integración (default True)
            n_jobs : int
                Número de procesos paralelos (default 16 por cantidad de cores que tengo yo)
        Returns:
            sigma8_z : array de n_a (debería coincidir con num_points)
                Array de σ8(z) para cada redshift en deltas
        -----------      
        """

        if deltas is None:
            if lcdm:
                a, deltas, _dp = self.solve_delta_lcdm_vectorized()
            else:
                a, deltas, _dp = self.solve_delta_mg_vectorized()
            z = 1/a - 1
            # seleccionar solo los z's mayores al z_start
            mask = z <= z_start
            a = a[mask]
            deltas=deltas[:, mask] #recordar que deltas es (n_k x n_a)
            
        if k is None:
            k = self.k_array
        if h is None:
            h = self.h


        # primordial power spec
        R = R / h  # convertir R a Mpc
        P_k_prim = A_s * (k/0.05)**(n_s-1) * (2*np.pi**2) 
        
        #window function:
        # argumento de window function
        kR = self.k_array * R #k tiene que entrar en 1/Mpc
        W_kR = 3 * (np.sin(kR) - kR * np.cos(kR)) / (kR**3) #tener cuidado de no meter k=0. to do: manejarlo

        # integrando
        #el newaxis es para que quede shape (n_k x 1) y pueda multiplicar por deltas (n_k x n_a)
        integrando = (self.k_array[:, np.newaxis]**2 * #jacobiano
                     P_k_prim[:, np.newaxis] * #primordial
                     W_kR[:, np.newaxis]**2 * #window function
                     deltas**2) #deltas numericas
        sigma8_z = np.zeros(deltas.shape[1])
        #integrar integrando en k (axis=0). Resolver en paralelo (simpson no puede vectorizarse)
        for i in range(deltas.shape[1]):
            if use_parallel and self.n_k > 10:
                # usar paralelización si hay suficientes k's
                from multiprocessing import Pool
                
                with Pool(n_jobs) as pool:
                    chunk_size = int(np.ceil(self.n_k / n_jobs))
                    chunks = [integrando[j:j+chunk_size, i] for j in range(0, self.n_k, chunk_size)]
                    k_chunks = [self.k_array[j:j+chunk_size] for j in range(0, self.n_k, chunk_size)]
                    chunk_args = list(zip(chunks, k_chunks))
                    results = pool.map(_simpson_chunk, chunk_args)
                integral = sum(results)
            else:
                integral = simpson(integrando[:, i], x=self.k_array)
            sigma8_z[i] = np.sqrt(integral)
        
        return sigma8_z
   
    def compute_f_k_vectorized(self, a_vec, delta, delta_p):
        """
        Calcula f(k,z) = d ln δ / d ln a = a δ'/δ para todos los k's.
        delta y delta_p son arrays de shape (n_k x n_a)
        Args:
        -----
        delta : array (matriz de n_k x num_points)
            Array de δ(k,z) para todos los k's y z
        delta_p : array (matriz de n_k x num_points)
            Array de δ'(k,z) para todos los k's y z
        a_vec : array (vector de n_a = num_points)
            Array de factores de escala a para cada redshift
        -----------
        Returns:
        k_array : array
            Array de valores k
        z_vec : array
            Array de valores z correspondientes a a_vec
        f_k : array (matriz de n_k x num_points)
            Array de valores f(k,z) para todos los k's y z
        -----------
        Note:
        Esta función asume que delta y delta_p tienen la misma shape y
        que a_vec tiene la misma longitud que el número de columnas en delta.
        Si delta y delta_p son de LCDM, f(z,k) coincide con f(z), pues es scale invariant.
        """      
        z_vec = 1/a_vec - 1

        f_k = (a_vec * delta_p) / delta
        f_k[~np.isfinite(f_k)] = 0.0
        return self.k_array, z_vec, f_k

    
    def compute_f_with_f_k(self, z_vec, f_k_array, x_array=None, k_array=None, use_mg_hubble=True, debug = False):
        """
        Calcula f(z) a partir de la transformada de Fourier de f(k,z).

        
        f(z) = (1/2π²) ∫ k² f(k,z) sinc(k·x(z)) dk

        donde x(z) es la distancia comoving:
        x(z) = ∫_0^z c dz'/H(z')

        Para gravedad modificada, se usa H_HS(z) en lugar de H_LCDM(z).

        Args:
        -----
        z_vec : array
            Array de valores z correspondientes a las columnas de f_k_array
        f_k_array : array (n_k x n_z)
            Array de valores f(k,z) para todos los k's y z
        k_array : array (opcional)
            Array de valores k. Si None, usa un array default
            Array de valores k. Si None, usa un array default
        use_mg_hubble : bool
            Si True, usa H_HS para gravedad modificada. Si False, usa H_LCDM.

        Returns:
        --------
        f_z : array (n_z,)
            Array 1D de valores f(z) para cada redshift en z_vec
        """
        if k_array is None:
            # k_array = np.logspace(-3, 1, f_k_array.shape[0])  # Default k range
            k_array = self.k_array
        if f_k_array.shape[0] != len(k_array):
            raise ValueError("f_k_array debe tener la misma cantidad de filas que k_array.")
        # if x_array is None:
        #     x_array = np.logspace(2*np.pi/np.log(max(k_array), 2*np.pi/min(k_array)), len(k_array))
        if f_k_array.shape[1] != len(z_vec):
            raise ValueError("f_k_array debe tener la misma cantidad de columnas que z_vec.")

        solver = DeltaSolver(k=0.1, h=self.h, Om_m_0=self.Om_m_0, 
                             b=self.b, z_ini_HS=self.z_ini_HS, z_0=self.z_0) #lo necesito para H_LCDM y H_HS (no dependen de k)
        # defino mi H(z) según si uso MG o no
        if use_mg_hubble:
            try:
                H_num, H_prime_num, r_num = solver.H_HS()
                def H_function(z): #esto es por si se le pasa un único escalar
                    if np.isscalar(z):
                        return H_num(z) if z <= self.z_ini_HS else self.H_LCDM_tensor(z)
                    else: 
                        result = np.zeros_like(z)
                        mask_mg = (z <= self.z_ini_HS) #mask de dónde usar H_HS (False si LCDM)
                        result[mask_mg] = H_num(z[mask_mg])
                        # tensor a numpy
                        H_lcdm_result = self.H_LCDM_tensor(z[~mask_mg])
                        result[~mask_mg] = H_lcdm_result.cpu().numpy() if hasattr(H_lcdm_result, 'cpu') else H_lcdm_result
                        return result
            except:
                print("Warning: No se pudo calcular H_HS, usando H_LCDM_tensor")
                def H_function_fallback(z_input):
                    H_result = self.H_LCDM_tensor(z_input)
                    return H_result.cpu().numpy() if hasattr(H_result, 'cpu') else H_result
                H_function = H_function_fallback
        else:
            def H_function_wrapper(z_input):
                H_result = self.H_LCDM_tensor(z_input)
                return H_result.cpu().numpy() if hasattr(H_result, 'cpu') else H_result
            H_function = H_function_wrapper

        # calcular distancias físicas x(z) = ∫_0^z c/H(z') dz' para cada z
        x_fisico = np.zeros_like(z_vec)
        # c = self.c * 3.24078e-20  # velocidad de la luz en Mpc/s
        c = self.c  # velocidad de la luz en km/s
        
        if debug:
            print('DEBUG: Calculando distancias físicas x(z)')
  
        H_values_in_z = np.zeros_like(z_vec)
        H_values_in_z_vec = H_function(z_vec)*(self.h*100)  # recordar que la función devolvía H(z)/H0) esto debería estar en km/s/Mpc
        for i, z in enumerate(z_vec):
            
            z_int = np.logspace(np.log10(0.0001), np.log10(z), 1000)
            H_int = H_function(z_int)*(self.h*100)  # recordar que la función devolvía H(z)/H0) esto debería estar en km/s/Mpc
            # H_int = H_function(z_int)
            # Integrar c/H(z') dz' hasta z
            x_fisico[i] = simpson(c/H_int, x=z_int)
            if debug:
                if i % 10 == 0:
                    print(f'En z={z:.2f}, x(z)={x_fisico[i]:.2f} Mpc. La integral da (sin ctes).: {x_fisico[i]*self.h*100/c:.2f} Mpc')
                    # print(f'En z={z:.2f}, x(z)={x_fisico[i]:.3f} Mpc')
                    #append H values in z of z_vec for debug
                    H_values_in_z[i] = H_function(z)*(self.h*100)

        sinc = lambda u: np.sinc(u)

        # Calcular f(z) usando la transformada de Fourier con x(z) conocido
        f_z = np.zeros(len(z_vec))
        r= 8/self.h #radio en Mpc/h
        for i, (z, x_z) in enumerate(zip(z_vec, x_fisico)):
            # Para cada z, usar su x(z) correspondiente
            # Integrando: k² f(k,z) sinc(k·x(z)) dk
            integrando = (k_array**2) * f_k_array[:, i] * sinc(k_array * x_z)

            # Integrar sobre k
            f_z[i] = simpson(integrando, x=k_array)
        if debug:
            return f_z,  H_values_in_z_vec, z_vec, x_fisico
        else:
            return f_z


def calculate_delta_mg(z_ini_HS=25,Om_m_0=0.305, k=0.1, b=0.1, num_points=1000):
    solver = DeltaSolver(z_ini_HS=z_ini_HS,Om_m_0=Om_m_0,b=b,k=k)
    return solver.solve_delta_mg(num_points)

def calculate_delta_lcdm(Om_m_0=0.305, num_points=1000):
    solver = DeltaSolver(Om_m_0=Om_m_0)
    return solver.solve_delta_lcdm(num_points)

def calculate_fs8_vectorized(k_array, z_select, b_select=0.1, params=None, use_gpu=True):
    """
    Función de conveniencia para calcular f*sigma8 usando el solver vectorizado
    """
    if params is None:
        params = {'h': 0.68, 'Om_m': 0.3}
    
    # Crear solver vectorizado
    solver = VectorizedDeltaSolver(
        k_array=k_array,
        Om_m_0=params['Om_m'],
        h=params['h'],
        b=b_select,
        use_gpu=use_gpu
    )
    
    # Calcular power spectrum y f(k)
    k_calc, P_k, f_k = solver.compute_power_spectrum_vectorized(z_select)
    
    return k_calc, P_k, f_k

def calculate_fs8_vectorized_parallel(k_array, z_select, b_select=0.1, params=None, 
                                    use_gpu=True, n_jobs=4):
    """
    Función de conveniencia para calcular f*sigma8 usando el solver vectorizado con paralelización
    """
    if params is None:
        params = {'h': 0.68, 'Om_m': 0.3}
    
    # Crear solver vectorizado
    solver = VectorizedDeltaSolver(
        k_array=k_array,
        Om_m_0=params['Om_m'],
        h=params['h'],
        b=b_select,
        use_gpu=use_gpu
    )
    
    # Calcular power spectrum y f(k) con paralelización
    k_calc, P_k, f_k = solver.compute_power_spectrum_vectorized(z_select, use_parallel=True, n_jobs=n_jobs)
    
    return k_calc, P_k, f_k




def calculate_sigma8_lcdm_evolution(k_array, params=None, use_gpu=True, n_jobs=4, num_points=1000):
    """
    Función de conveniencia para calcular la evolución σ8(z) de LCDM
    
    Args:
        k_array : array
            Array de valores k para la integración
        params : dict
            Parámetros cosmológicos {'h': float, 'Om_m': float}
        use_gpu : bool
            Usar GPU cuando esté disponible
        n_jobs : int
            Número de procesos paralelos
        num_points : int
            Número de puntos en la grilla de a
            
    Returns:
        a_vec : array
            Factores de escala
        z_vec : array
            Redshifts correspondientes
        sigma8_z : array
            σ8(z) para LCDM
        delta_results : array
            Matriz de δ(k,z) (n_k x num_points)
    """
    if params is None:
        params = {'h': 0.68, 'Om_m': 0.3}
    
    # Crear solver vectorizado
    solver = VectorizedDeltaSolver(
        k_array=k_array,
        Om_m_0=params['Om_m'],
        h=params['h'],
        use_gpu=use_gpu
    )
    
    # Resolver las ecuaciones para todos los k's y todos los a's
    a_vec, delta_results, delta_p_results = solver.solve_delta_lcdm_vectorized_parallel(
        num_points=num_points, n_jobs=n_jobs
    )
    
    # Calcular σ8(z) para cada redshift
    sigma8_z = solver.compute_sigma8_lcdm_vectorized(delta_results, use_parallel=True, n_jobs=n_jobs)
    
    # Calcular redshifts correspondientes
    z_vec = 1/a_vec - 1
    
    return a_vec, z_vec, sigma8_z, delta_results