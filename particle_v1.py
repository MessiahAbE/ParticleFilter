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
position_new_true=position#ME
p_0=np.array([[0,0],[0,0]])
teta=-1
point_prev=[[0],[0]]
position_particle=np.array([[0],[0]])

def Gaussian(self, mu, sigma, x):
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
#ME    
def measurement_prob(x,y):
    # calculates how likely a measurement should be
    prob_x = 1.0
    distO = math.sqrt((x - position_measure[0][0]) ** 2 + (y - position_measure[1][0]) ** 2)
    #ME - Euclidant distance is not probablity and dist is low for good particles. Idea is that the weight be high for good particle. so, best way is to  consider a probablity function like gaussian and pass the distance to it to get probablity. 
    dist = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - position_measure[0][0])**2/(2*0.005*0.005))+((y - position_measure[1][0])**2/(2*0.075*0.075))))
    #ME IMPORTANT USE position measure
    
    ##another example:
    #dist = np.exp(-(distO**2)/(2*0.01*0.01)) 

    dist = dist + 1e-9 #get rid of 0
    return dist

def compute_cov(x,y):
    # print(x)
    # print(y)
    x=np.array(x)
    y=np.array(y)
    print(x)
    print(np.mean(x))
    print(np.std(x))
    
    return np.mean(x),np.std(x),np.mean(y),np.std(y)

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

def resample(particles, dist):
    N=len(particles)
    new_particles = []
    index = int(random.random() * N)
    beta = 0.0
    
    for i in range(0,N):
        dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])
        
    dist=dist/np.sum(dist)
    mw = dist.max()

    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > dist[0][index]:
            beta -= dist[0][index]
            index = (index + 1) % N
                  
        new_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]]]))
        
    return new_particles

def compute_weight(particles):

        global position_measure

        dist=np.zeros((1,len(particles)))

        new_x = 0
        new_y = 0

        for i in range(0,len(particles)):
            dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])


        dist=dist/np.sum(dist)

###ME top - python array not np so particles[:][0][0]] do not work::
        
        for i in range(0, len(particles)):	
            new_x = new_x + dist[0,i] * particles[i][0][0]
            new_y = new_y + dist[0,i] * particles[i][1][0]

        pose=np.array([[new_x],[new_y]])#ME
 
        return pose,dist

def car(x,y):
    gameDisplay.blit(carImg, (x,y))


#In this function the position estimation is being conducted 
def estimate_pose(position,particles):
    x=[]
    y=[]
    global position_new_true
    
    F=np.array([[1,0],[0,1]])
    r=0.1
    delta_t=1/8
    G=np.array([[r/2*delta_t,r/2*delta_t],[r/2*delta_t,r/2*delta_t]])
    u=np.array([[1],[1]])
    
    position_new = np.matmul(F,position) + np.matmul(G,u) + np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
    position_new_true = np.matmul(F,position_new_true)+np.matmul(G,u)#ME
    
 
    for i in range(0,len(particles)):

            temp = np.matmul(F,particles[i]) + np.matmul(G,u)+ np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
            particles[i]=temp
            x.append(temp[0][0])
            y.append(temp[1][0])
    return particles,position_new,x,y

x_change = 0


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
#ME FOLLOWING UPDaTeD
    particles,position,x,y=estimate_pose(position,particles)##ME - what is position -  its no use was for kalman can remove
    mean_x,std_x,mean_y,std_y=compute_cov(x,y)
    
    if(t%8==0):
        measurement()
        position_particle,dist=compute_weight(particles)
        particles=resample(particles, dist)
        
    gameDisplay.fill(white)
    WHITE=(255,255,255)
    BLUE=(0,0,255)
    RED=(255,0,0)
    GREEN=(0,255,0)
    yellow=(234, 221, 202)
    
    for p in particles:        
        pygame.draw.rect(gameDisplay,RED,(p[0][0]*1000+50,(p[1][0]/2)*1000+50,2,2))##particles are red

    red = (180, 50, 50)
    size = (position_particle[0,0]*1000+50-(std_x*2000)/2, position_particle[1,0]/2*1000+50-(2000*std_y)/2, std_x*2000, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, red, size,1)  

    pygame.draw.polygon(gameDisplay, BLUE,
                        [[position_particle[0,0]*1000+50,position_particle[1,0]/2*1000+50],[position_particle[0,0]*1000+40,position_particle[1,0]/2*1000+35] ,
                        [position_particle[0,0]*1000+40,position_particle[1,0]/2*1000+65]])##blue is the position
  
    points.append([position_particle[0,0]*1000+50,position_particle[1,0]/2*1000+50])####ME
    points_gt.append([position_new_true[0,0]*1000+50,position_new_true[1,0]*1000+50])
    points_measure.append([position_measure[0,0]*1000+50,(position_measure[1,0]/2)*1000+50])
    pygame.draw.lines(gameDisplay,BLUE,False,points,5) #BLUE: mean position
    pygame.draw.lines(gameDisplay,GREEN,False,points_gt,5) #GREEN: ground truth
    pygame.draw.lines(gameDisplay,RED,False,points_measure,5) #RED: measurement



   
    pygame.draw.rect(gameDisplay,yellow,(position_particle[0,0]*1000+50,(position_particle[1,0]/2)*1000+50,10,10))

    measured_positions_x.append(position_measure[0,0])
    measured_positions_y.append(position_measure[1,0])#ME
    predicted_positions_x.append(position_particle[0,0])###ME
    predicted_positions_y.append(position_particle[1,0])##ME
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])

    pygame.display.update()
    clock.tick(8) 
        
    t+=1
    
plt.plot(measured_positions_x,measured_positions_y, label='measurement')
plt.plot(predicted_positions_x,predicted_positions_y, label='mean particle point')
plt.show()

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

