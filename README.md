# ProyectoFinal_HPC

Se estudia el uso de la libreria FEniCS de elementos finitos para la solución de la ley de Darcy en una malla rectangular con un agujero en una de sus esquinas.
La versión utilizada corresponde a FEniCSx 0.8, cuya instalación se hace con los siguientes comandos:
# En Ubuntu:
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install fenicsx

# Compilación:

Para el caso serial el comando es:

python3 archivo.py

Para el caso en paralelo, siendo np el número de procesos:

mpirun -n np python3 archivo.py
