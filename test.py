import numpy as np
import matplotlib.pyplot as plt
import math

#seasonal contact stuff
ampScale = .1001 *10
periodScale = 2*np.pi/12
def cosine(x):
    return max(0,ampScale * (math.cos(periodScale * (x-1)) -.9))

t = np.linspace(0,18,180)
y = list(map(cosine, t))
plt.plot(t, y)
plt.axvline(x=13, color='red', linestyle='--')
plt.axvline(x=1, color='red', linestyle='--')
plt.title('Travel Function')
plt.xlabel('Time (Months)')
plt.ylabel('Proportion of Travelers')
plt.grid(axis='both')
plt.xticks(np.linspace(1,18,18))
#plt.show()
plt.close()

#initialize concact and matrix stuff
M = (np.diag(np.array([1,2,3,4])) @ np.ones([4,4])) / 3
max_eig = np.max(np.abs(np.linalg.eig(M)[0]))
target_R0 = 1.5
cbar_xi = target_R0/max_eig


def SIR(susceptibility,
        recovery,
        population_prop,
        contact_maxtrix,
        initial_s, initial_i, initial_r,
        t_max, dt, mu, contactFrequency):

    n_groups = len(initial_s)
    n_steps = int(np.round(t_max / dt))

    # Create Diagonal Matrices
    Dsusc = np.diag(susceptibility)
    Dgamma = np.diag(recovery)
    Dpop = np.diag(population_prop)
    invDpop = np.linalg.inv(Dpop)
    M = (Dpop @ Dpop @ contact_maxtrix @ invDpop) @ np.linalg.inv(Dgamma)
    eigen = np.linalg.eigvals(M)

    #Print R0 values 
    #print(M)
    #print(eigen)

    #initialize arrays
    S = np.zeros([n_groups, n_steps])
    I = np.zeros([n_groups, n_steps])
    R = np.zeros([n_groups, n_steps])
    T = np.zeros(n_steps)

    S[:,0] = initial_s
    I[:,0] = initial_i
    R[:,0] = initial_r
    
    travBucket  = []
    for idx in range(1, n_steps):
        # dynamic operator
        travelors = contactFrequency * cosine(idx*dt)
        travBucket.append(travelors)
        CmatWithT = contact_maxtrix

        CmatWithT[3][3] = contactFrequency - travelors 
        CmatWithT[1][3] = travelors / 2
        CmatWithT[3][1] = travelors / 2
        X = np.matmul(np.matmul(Dsusc, CmatWithT), np.linalg.inv(Dpop))
        #calc deaths
        Sd = S[:, idx - 1] * mu * dt
        Id = I[:, idx - 1] * mu * dt
        Rd = R[:, idx - 1] * mu * dt
        births = (Sd + Id + Rd)

        T[idx] = T[idx - 1] + dt
        incidence_operator = np.matmul(np.diag(S[:, idx - 1]), X)

        ds = np.matmul(-incidence_operator,  I[:, idx - 1])
        dr = np.matmul(Dgamma, I[:, idx - 1])
        di = -ds-dr

        S[:, idx] = S[:, idx - 1] + ds*dt + births - Sd
        I[:, idx] = I[:, idx - 1] + di*dt - Id
        R[:, idx] = R[:, idx - 1] + dr*dt - Rd

    return S,I,R,T,travBucket


# Start Parameters
reco = np.ones(4)
reco[3] = .8
prop = 0.25 * np.ones(4)
_contactFrequency = 25
Cmat = _contactFrequency * np.array([[.5,.4,.1,0],
                [.4,.5,.4,0],
                [.1,.4,.5,0],
                [0,0,0,1]])

susc = np.array([.1,.2,.3,1])

s0 = 0.25 * 0.999 * np.ones(4)
i0 = 0.25 * 0.001 * np.ones(4)
r0 = 0 * np.ones(4)
t = 1200
dt = 0.05
mu = .0011
# End Parameters


#Run simulation save result in shape (4, t/dt)
S,I,R,T, travBucket = SIR(susc, reco, prop, Cmat, s0, i0, r0, t, dt, mu, _contactFrequency)
ampScale = 0
Si,Ii,Ri,Ti, travBucket2 = SIR(susc, reco, prop, Cmat, s0, i0, r0, t, dt, mu, _contactFrequency)


#Display outbreak over equilibrium dynamics
randColors = ['#' + ''.join(np.random.choice(list('0123456789ABCDEF'), size=6)) for _ in range(4)]
hexColors = ['#3624EA', '#4F8F4A', '#30F7AF', '#E73336']
fig, ax = plt.subplots(nrows=2,ncols=1)
for i in range(3):
    ax[0].plot(T,I[i,:],
            color=hexColors[i],
            label='$i_{}(t)$'.format(i+1))
    ax[0].plot(T,Ii[i,:],
            color=hexColors[i],
            linestyle='dashed',
            label='$i_{}(t) isolated$'.format(i+1))

ax2 = ax[0].twinx()
ax2.set_ylabel('Travel Load')
ax2.plot(np.linspace(0,18,180),y, color='black', alpha=1/5, label= 'Travel Load')
ax[0].plot(0,0,color='black', alpha=1/5, label= 'Travelers')
ax[0].set_title('Outbreak Dynamics')
ax[0].set_ylabel('Total population proportion')
ax[0].legend()
ax[0].set_ylim(bottom=0)
ax[0].set_xlim([0,8])

sliceStart = int(48 / dt)
xlimVal = T - 24
iSlice = []
for i in range(3):
    iSlice = I[i]
    iSlice = iSlice[-sliceStart:]
    ax[1].plot(T[-sliceStart:],iSlice,
            color=hexColors[i],
            alpha=4/5,
            label='$i_{}(t)$'.format(i+1))

    iSliceI = Ii[i]
    iSliceI = iSliceI[-sliceStart:]
    ax[1].plot(T[-sliceStart:],iSliceI,
            color=hexColors[i],
            linestyle='dashed',
            label='$i_{}(t) isolated$'.format(i+1))

scaled = np.interp(travBucket, (0,max(travBucket)),(0,.1))
ax2 = ax[1].twinx()
ax2.set_ylabel('Travel Load')
ax2.plot(T[-sliceStart:], scaled[-sliceStart:], alpha=1/5, color='black')

ax[1].plot(T[-sliceStart], iSlice[0], color='black', alpha=1/5, label='Travelers')
ax[1].set_title('Equilibrium Dynamics')
ax[1].set_xlabel('Time (Months)')
ax[1].set_ylabel('Total population proportion')
ax[1].legend()
#ax[1].set_ylim(bottom=0.0025)
sliceStart = T  - sliceStart
#ax[1].set_xlim([int(xlimVal),np.max(T)])
ax[1].grid(axis='y')

plt.show()
plt.close()


#Display disease spirals
fig, ax = plt.subplots(ncols=3, nrows=1)
fig.suptitle('Disease Spirals')
for i in range(3):
    ax[i].plot(S[i,:]/.25,I[i,:]/.25, label='Migratory Influence', color=hexColors[i])
    ax[i].plot(Si[i,:]/.25,Ii[i,:]/.25, label='Isolated', color=hexColors[i], linestyle='dashed')
    ax[i].legend()
    #ax[i].set_xlim([0,.38])
    ax[i].set_xlabel('Susceptible Proportion')
    #ax[i].set_ylim([0,.045])
    ax[i].set_ylabel('Infected Proportion')
    ax[i].set_title('Population {}'.format(i+1))
plt.show()


#Sanity check display isolation graph
fig, ax = plt.subplots(ncols=2, nrows=1)
for i in range(3):
    ax[0].plot(T,Ii[i,:],
            color=hexColors[i],
            alpha=(4)/5,
            label='$i_{}(t)$'.format(i+1))
ax[1].plot(T, Ii[3,:],
           color='red',
           alpha=4/5)
for i in range(2):
    ax[i].legend()
    ax[i].set_xlim([0,12])
    ax[i].set_ylim([0,np.max(Ii[3,:]) +.01])
    ax[i].set_ylabel('Proportion of the population')
    ax[i].set_xlabel('Time')
fig.suptitle('Dynamics Under Isolation')
plt.show()


#Show tail end dynamics
fig, ax = plt.subplots(ncols=1, nrows=3)
iSlice = []
sliceStart = int(80*12/dt)
sliceEnd = int(99*12/dt)
for i in range(3):
    #ax.plot(T-40:],I[i,-40:],
    iSlice = I[i]
    iSlice = iSlice[sliceStart: sliceEnd] / .25
    ax[i].plot(T[sliceStart: sliceEnd],iSlice,
            color=hexColors[i],
            alpha=(4)/5,
            label='Migratory Influence')
    iSliceI = Ii[i]
    iSliceI = iSliceI[sliceStart: sliceEnd] /.25
    ax[i].plot(T[sliceStart: sliceEnd],iSliceI,
            color=hexColors[i],
            #alpha=(i+1)/5,
            label='Isolated',
            linestyle='--')
    #ax[i].axhline(y=np.mean(I[sliceStart:sliceEnd]), color='r', alpha=1/5, label='$i_{eq}$ Migratory')
    #ax[i].axhline(y=np.mean(Ii[sliceStart:sliceEnd]), color='r', linestyle='dashed', label='$i_{eq}$ Isolated')
    ax[i].plot(T[sliceStart], iSlice[0], color='black', alpha=1/5)
    ax2 = ax[i].twinx()
    ax2.plot(T[sliceStart:sliceEnd], scaled[sliceStart:sliceEnd], color='black', alpha=1/5)
    ax2.set_ylabel('Travel Load')
    #formating
    ax[i].set_xlabel('Time')
    ax[i].set_ylabel('Proportion of Population {}'.format(i+1))
    ax[i].legend()
    ax[i].grid(axis='y')

fig.suptitle('Tail End Dynamics')
plt.show()
plt.close()
