from cProfile import label
from cmath import atan, exp
from hashlib import new
import random
from re import U
import pygame
import numpy as np
from pygame.locals import *
import matplotlib.pyplot as plt 
import math
pygame.init()

display_width = 1800
display_height = 1600

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('A bit Racey')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False
points=[]
points.append([0,0])
points_measure=[]
points_gt=[]
points_gt.append([0,0])
points_measure.append([0,0])
measured_positions_x=[]
measured_positions_y=[]
predicted_positions_x=[]
predicted_positions_y=[]
cov_width=[]
cov_hight=[]
position=np.array([[0],[0]])
position_measure=position
p_0=np.array([[0,0],[0,0]])
teta=-1
point_prev=[[0],[0]]

def Gaussian(self, mu, sigma, x):
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        print(x)
        ppppppp
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    
def measurement_prob(x,y):
    # calculates how likely a measurement should be
    prob_x = 1.0

    dist = math.sqrt((x - position_measure[0][0]) ** 2 + (y - position_measure[0][0]) ** 2)

    return dist

	

def particle_generation():
    global position
    global  position_measure
    N_particles=100
    particles=[]
    particles_x=np.random.normal(position[0][0],0.005,N_particles)
    particles_y=np.random.normal(position[1][0],0.075,N_particles)
    for i in range(0,N_particles):

        particles.append(np.array([[particles_x[i]],[particles_y[i]]]))  
    return particles
    # for i in range(0,100):
    #     dist[0,i]=measurement_prob(particles_x[i],particles_y[i])
    # print(dist)
    # print(np.sum(dist))
    # dist=dist/np.sum(dist)
    # new_x=0
    # new_y=0
    # for i in range(0,100):
    #     new_x=dist[0,i]*particles_x[i]
    #     new_y=dist[0,i]*particles_y[i]
    # # position=[[new_x],[new_y]]   
    # position[0][0]=new_x
    # position[1][0]=new_y 
    # print(dist)

# def resample(particles,dist):
#     cumulative_sum = np.cumsum(dist[0][:])
#     print(cumulative_sum)
#     # pppppppppppppppppppppppppppp
#     cumulative_sum[-1] = 1. # avoid round-off error
#     print(len(particles[0]))
#     rand=random.Random()
#     print(rand)

#     indexes = np.searchsorted(cumulative_sum,number )
#     print(indexes)
#     ppppppppppppppppppppppppp
#     # resample according to indexes
#     self.particles = self.particles[indexes]
#     self.weights = self.weights[indexes]
#     self.weights /= np.sum(self.weights) # normalize


def resample(particles, dist):
    N=len(particles[0])
    new_particles = []
    index = int(random.random() * N)
    beta = 0.0
    for i in range(0,N):

        dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])
    dist=dist/np.sum(dist)
    mw = dist.max()
    for i in range(N):
        beta += random.random() * 2.0 * mw
        print ("beta =", beta)
        while beta > dist[0][index]:
            beta -= dist[0][index]
            index = (index + 1) % N
            print ("\tbeta= %f, index = %d, weight = %f" % (beta, index, dist[0][index]))
        new_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]]]))
    new_sample = np.random.choice(a=particles, size=10000, replace=True, p=dist[0]) 
    # print("new particle is "+str(new_particles))
    # print(len(new_particles))
    # return new_particles
    #     N = len(old_particles)
    # new_particles = []
    # index = int(random.random() * N)
    # beta = 0.0
    # mw = max(weights)
    # for i in range(N):
    #     beta += random.random() * 2.0 * mw
    #     print "beta =", beta
    #     while beta > weights[index]:
    #         beta -= weights[index]
    #         index = (index + 1) % N
    #         print "\tbeta= %f, index = %d, weight = %f" % (beta, index, weights[index])
    #     new_particles.append(old_particles[index])
    return new_particles
# def resample(particles,dist):


#     p3=[]
#     index = int(random.random()*len(particles[0]))
#     beta = 0
#     mw = dist.max()

#     for i in range(len(particles[0])):
#         beta += random.random() * 2 * mw
#         while beta > dist[0][index]:
#             beta -= dist[0][index]
#             index = (index + 1)%len(particles[0])
#         print(index)
        
#         p3.append((particles[0][index],particles[1][index]))
#         # print(particles[2][index])

#         print("p3"+ str(p3))
        
#     particles = p3
#     print(len(p3))
#     ppppppppppp

def compute_weight(position,particles):

        global position_measure
        dist=np.zeros((1,len(particles)))
        for i in range(0,len(particles)):
            dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])
        dist=dist/np.sum(dist)
        print(dist)
        print(particles[0][0][0])
        print(particles[1][0][0])
        
        new_x=0
        new_y=0
        for i in range(0,len(particles[0])):
            new_x=dist[0,i]*particles[i][0][0]
            new_y=dist[0,i]*particles[i][1][0]
            print(new_x)
        print(new_y)
        pose=np.array([[new_x],[new_y]])
        # particles=resample(particles,dist)
        print(pose)
        return pose
            # print(particles[i])
    #     dist[0,i]=measurement_prob(particles_x[i],particles_y[i])
def car(x,y):
    gameDisplay.blit(carImg, (x,y))


#In this function the position estimation is being conducted 
def estimate_pose(position,particles):

    global position_new_true
    global position_new_true
    
    F=np.array([[1,0],[0,1]])
    r=0.1
    delta_t=1/8
    G=np.array([[r/2*delta_t,r/2*delta_t],[r/2*delta_t,r/2*delta_t]])
    u=np.array([[1],[1]])
    
    position_new = np.matmul(F,position) + np.matmul(G,u) + np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
    position_new_true = np.matmul(F,position)+np.matmul(G,u)
    for i in range(0,len(particles)):
            temp = np.matmul(F,particles[i]) + np.matmul(G,u)+ np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
            particles[i]=temp
    
    return position_new,particles

x_change = 0


#The state covariance Matrix is being updated in this function 
def update():
    global p_0 
    delta_t=1/8
    F=np.array([[1,0],[0,1]])
    temp=np.matmul(F,p_0)
    Q=np.array([[0.1,0],[0,0.15]])*1/8
    p_new=np.matmul(temp,F.transpose())+Q
    p_0=p_new
    
#The measurement data is being computed 
def measurement():
    global position_measure
    global position
    H=np.array([[1,0],[0,2]])
    R=np.array([[0.05],[0.075]])
    
    Z=np.matmul(H,position_new_true) + np.asarray([[np.random.normal(0,0.05)],[np.random.normal(0,0.075)]])
    
    position_measure=Z

#Computing Kalman gain    
def correction():
    global p_0

    H=np.array([[1,0],[0,2]])
    R=np.array([[0.05,0],[0,0.075]])
    temp1=np.matmul(p_0,H.transpose())
    temp2=np.matmul(np.matmul(H,p_0),H.transpose()) + R

    k=temp1/temp2
    k[np.isnan(k)] = 0

    return H,k
#final update 
def update_final(H,K):
    global p_0
    global position    
    p_0=np.matmul((np.identity(2)-np.matmul(K,H)),p_0)
    H = np.array([[1,0],[0,2]])
    # position = position + np.matmul( K , (position_measure - np.matmul(H,position)) )

t=1   
first_flag=0
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x_change = -5
            elif event.key == pygame.K_RIGHT:
                x_change = 5
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                x_change = 0

    if first_flag==0:
        particles=particle_generation()
        first_flag=1
    position,particles=estimate_pose(position,particles)
    position=compute_weight(position_new_true,particles)
    print("pose is",position)
    # print(particles)
    # print(len(particles))
    update()

    if(t%8==0):
        measurement()
        H_new,K_new=correction()
        update_final(H_new,K_new)

        # position,particles=compute_weight(position,particles)
        #print("len particle is"+ str(len(particles)))

    gameDisplay.fill(white)
    WHITE=(255,255,255)
    BLUE=(0,0,255)
    RED=(255,0,0)
    GREEN=(0,255,0)
    for p in particles:
        pygame.draw.rect(gameDisplay,RED,(p[0][0]*1000+50,(p[1][0]/2)*1000+50,2,2))
    
    red = (180, 50, 50)
    size = (position[0,0]*1000+50-(p_0[0,0]*2000)/2, position[1,0]*1000+50-(2000*p_0[1,1])/2, p_0[0,0]*2000, 2000*p_0[1,1])
    pygame.draw.ellipse(gameDisplay, red, size,1)  
    # pygame.draw.rect(gameDisplay,BLUE,(position[0,0]*1000+50,position[1,0]*1000+50,10,10))
    # points1=[(position[0,0]*1000+50,position[1,0]*1000+50), (position[0,0]*1000+50,position[1,0]*1000+500), (position[0,0]*1000+250,position[1,0]*1000+500)]
    # pygame.draw.polygone(gameDisplay, color=(255,0,0),points=points1)
    # teta=atan((point_prev[1,0]-(position[1,0]*1000+50))/(point_prev[0,0]-(position[1,0]*1000+50)))
    pygame.draw.polygon(gameDisplay, BLUE,
                        [[position[0,0]*1000+50,position[1,0]*1000+50],[position[0,0]*1000+40,position[1,0]*1000+35] ,
                        [position[0,0]*1000+40,position[1,0]*1000+65]])
  
    points.append([position[0,0]*1000+50,position[1,0]*1000+50])
    points_gt.append([position_new_true[0,0]*1000+50,position_new_true[1,0]*1000+50])
    points_measure.append([position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50])
    pygame.draw.lines(gameDisplay,BLUE,False,points,5)
    pygame.draw.lines(gameDisplay,GREEN,False,points_gt,5)
    pygame.draw.lines(gameDisplay,RED,False,points_measure,5)

# pygame.display.update()
    # pygame.draw.polygon(surface=gameDisplay, color=(255, 0, 0),points=[(position[0,0]*1000+50,position[1,0]*1000+50), (position[0,0]*1000+40,position[1,0]*1000+40), (position[0,0]*1000+30,position[1,0]*1000+30)])

    if(t%8==0):
        pygame.draw.rect(gameDisplay,RED,(position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50,10,10))
        pygame.draw.rect(gameDisplay,GREEN,(position_new_true[0,0]*1000+50,(position_new_true[1,0]/2)*1000+50,10,10))

    pygame.draw.rect(gameDisplay,RED,(position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50,10,10))
    pygame.draw.rect(gameDisplay,GREEN,(position_new_true[0,0]*1000+50,(position_new_true[1,0])*1000+50,10,10))   
    measured_positions_x.append(position_measure[0,0])
    measured_positions_y.append(position_measure[1,0])
    predicted_positions_x.append(position[0,0])
    predicted_positions_y.append(position[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    # plt.plot(measured_positions_x)
    # plt.show()
    pygame.display.update()
    clock.tick(8) 
        
    t+=1
plt.plot(measured_positions_x,label='x values')
plt.plot(measured_positions_y,label='y values')
plt.xlabel("iteation")
plt.ylabel("value")
plt.legend()
plt.show()
plt.plot(predicted_positions_x,label='x values')
plt.plot(predicted_positions_y,label='y values')
plt.xlabel("iteation")
plt.ylabel("value")
plt.legend()
plt.show()
plt.plot(cov_hight,label='covariance of x values')
plt.plot(cov_width,label='covariance y values')
plt.xlabel("iteation")
plt.ylabel("value")
plt.legend()
plt.show()
pygame.quit()
quit()

