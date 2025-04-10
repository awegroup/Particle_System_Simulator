import numpy as np
import matplotlib.pyplot as plt

class simply_supported_beam:
    def __init__(self,L,EI):
        self.L = L
        self.EI = EI
        self.x = np.arange(0, L+0.01, 0.01)
        self.dy = np.zeros(len(self.x))
    def uniformload(self,w):
        self.dy =+ w*self.x/(24*self.EI)*(self.L**3-2*self.L*self.x**2+self.x**3)
    def pointload(self,F,a):
        b = self.L - a
        left = self.x < a
        right = ~left
        self.dy[left] += F*b*self.x[left]/(6*self.EI*self.L)*(self.L**2-b**2-self.x[left]**2)
        self.dy[right] += F*a*(self.L-self.x[right])/(6*self.EI*self.L)*(self.L**2-a**2-(self.L-self.x[right])**2)


        
#Example beam 
L = 10               #length [m]
EI = 5e6            #effective bending stiffness [N m^2]
w = -200            #uniform load [N/m]

beam = simply_supported_beam(L,EI)
beam.uniformload(w)
plt.plot(beam.x, beam.dy)
plt.title("Simply supported beam with uniform load")
plt.xlabel("x [m]")
plt.ylabel("deflection [m]")
plt.grid()
plt.show()
print(np.min(beam.dy))

beam2 = simply_supported_beam(L,EI)
F = -10000          #point load [N]
a = 3           #point load position [m]
beam2.pointload(F,a)
plt.plot(beam2.x, beam2.dy)
plt.title("Simply supported beam with point load")
plt.xlabel("x [m]")
plt.ylabel("deflection [m]")
plt.grid()
plt.show()
print(np.min(beam2.dy))
