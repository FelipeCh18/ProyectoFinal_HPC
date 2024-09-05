from dolfin import *
import numpy as np
from matplotlib import pyplot as plt

comm=MPI.comm_world
size=comm.Get_size()
rank=comm.Get_rank()


#Dimensiones rectangulo
x0,y0= 0, 0
x1,y1= 4, 2
Inicial=Point(x0, y0)
Final=Point(x1, y1)
#Número de elementos:
h=200 #Altura
l=200 #Ancho

#Definición de la malla
mesh= RectangleMesh(Point(x0,y0),Point(x1,y1),l,h,'right')

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], x1)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], y0)

class Top1(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (x0,3*x1/4))and near(x[1], y1))

class Top2(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (3*x1/4, x1))and  near(x[1], y1))

#Se inicializan los subdominios
left = Left()
top = Top1()
right = Right()
bottom = Bottom()
topb = Top2()

#Determinación la frontera
front=MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
subd=MeshFunction("size_t",mesh,mesh.topology().dim(),0)

front.set_all(0)
left.mark(front, 1)
top.mark(front, 2)
topb.mark(front, 3)
right.mark(front, 4)
bottom.mark(front, 5)

#Definición de los vectores normales
n= FacetNormal(mesh)
nx= n[0]
ny= n[1]

#Definición del espacio de elementos finitos
FE = FiniteElement("P",mesh.ufl_cell(),1)
W = FunctionSpace(mesh, FE)

#Definición de las funciones presión y prueba
p=TrialFunction(W)
v=TestFunction(W)

#Condición de Dirichlet para el agujero de salida
cfd=[DirichletBC(W,0.0,front,3)]

#Condiciones del problema y definición de los diferenciales

ds= Measure("ds", domain=mesh, subdomain_data=front)
dx= Measure("dx", domain=mesh, subdomain_data=subd)


kappa=Constant(1e-08)
eta=Constant(0.001)
rho=Constant(1)
gra=Constant(98)
theta=90
ang=Constant(np.sin(theta*(np.pi/180)))
F=Constant(0.0)
zero=Constant(0.0)
j=Constant(gra*rho*ang)

#Condición de entrada de flujo
N0=Constant(0.001)

#Forma variacional
Tf= -(((kappa/eta)*(p-j).dx(0)*v.dx(0) + (kappa/eta)*(p-j).dx(1)*v.dx(1))*dx(0)
     - F*v*dx(0) -N0*v*ds(1)-zero*v*ds(2)-zero*v*ds(4)-zero*v*ds(5))

#Forma bilineal y lineal
a,L = lhs(Tf), rhs(Tf)

#Solución al problema
u= Function(W)
solve(a == L, u, cfd)

#Visualización

vel=plot(u)
#vel.set_cmap("hot")
plot(-grad(u))
plt.colorbar(vel)
plt.show()
