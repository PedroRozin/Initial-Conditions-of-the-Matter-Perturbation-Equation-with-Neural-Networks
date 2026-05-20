# Paper Project

to do

## Setup

1. Create the local link to the thesis environment if it is not already present:

   ```bash
   ln -s /home/pedrorozin/scripts/venv .venv
   ```

2. Activate it:

   ```bash
   source .venv/bin/activate
   ```

3. Expose the thesis helper modules when you need them:

   ```bash
   export PYTHONPATH="/home/pedrorozin/scripts/python:${PYTHONPATH:-}"
   ```

