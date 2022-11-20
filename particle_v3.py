from array import array
from cmath import polar
from dis import dis
from hashlib import new
from re import U
import pygame
import numpy as np
from pygame.locals import *
from math import *
pygame.init()

###### elipse
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
import multiprocessing
import math
import random

######

display_width = 1800
display_height = 1600

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('A bit Racey')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()
crashed = False
measured_positions_x=[]
measured_positions_y=[]
predicted_positions_x=[]
predicted_positions_y=[]
cov_width=[]
cov_hight=[]
points=[]
points.append([400,400])
#carImg = pygame.image.load('car_128.png')
points=[]
points.append([400,400])
center=np.array([[10],[10]])
points_measure=[]
points_gt=[]
points_gt.append([400,400])
points_measure.append([400,400])
position_particle=np.array([[0],[0],[0]])


position=np.array([[0],[0],[0]])
position_new_true = np.array([[0],[0],[0]])
position_polar=np.array([[0],[0]])
polar_measure=position_polar
position_measure=position
p_0=np.array([[0,0,0],[0,0,0],[0,0,0]])

def compute_cov(x,y):

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
    particles_x=np.random.normal(position[0][0],0.0005,N_particles)
    particles_y=np.random.normal(position[1][0],0.0075,N_particles)
    particles_theta=np.random.normal(position[1][0],0,N_particles)
    for i in range(0,N_particles):

        particles.append(np.array([[particles_x[i]],[particles_y[i]],[particles_theta[i]]]))  
    
    return particles
def measurement_prob(x,y):

    prob_x = 1.0
    a=center[0,0] + [polar_measure[0,0]*cos(polar_measure[1,0])]
    b=center[0,0] + [polar_measure[0,0]*sin(polar_measure[1,0])]
 
    dist = (1.0/(2*np.pi*0.005*0.075))*np.exp(-(((x - a)**2/(2*0.005*0.005))+((y - b)**2/(2*0.075*0.075))))


    dist = dist + 1e-9 
    return dist

def compute_weight(particles):

        global position_new_true
        dist=np.zeros((1,len(particles)))
        for i in range(0,len(particles)):
            dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])
        dist=dist/np.sum(dist)
        new_x = 0
        new_y = 0

        for i in range(0, len(particles)):	
            new_x = new_x + dist[0,i] * particles[i][0][0]
            new_y = new_y + dist[0,i] * particles[i][1][0]


        pose=np.array([[new_x],[new_y],[0]])
        return pose,dist    
def resample(particles, dist):
    N=len(particles)
    new_particles = []
    index = int(random.random() * N)
    beta = 0.0
    print(dist[0,1])
    print(particles)
    
    for i in range(0,N):

        dist[0,i]=measurement_prob(particles[i][0][0],particles[i][1][0])
        print(dist[0,i])
        
    dist=dist/np.sum(dist)
    mw = dist.max()
    for i in range(N):
        beta += random.random() * 2.0 * mw
        print ("beta =", beta)
        while beta > dist[0][index]:
            beta -= dist[0][index]
            index = (index + 1) % N
            print ("\tbeta= %f, index = %d, weight = %f" % (beta, index, dist[0][index]))
        new_particles.append(np.array([[particles[index][0][0]],[particles[index][1][0]],[particles[index][2][0]]]))
        

    return new_particles   
def car(x,y):
    gameDisplay.blit(carImg, (x,y))

def convert_polar(p):

    rho = np.sqrt((p[0,0] - center[0,0])**2 + (p[1,0] - center[1,0])**2)
    phi = np.arctan2(p[1,0] - center[1,0], p[0,0] - center[0,0])# - p[2,0]
    new_pose=np.array([[rho],[phi]])

    return new_pose

def estimate_pose(position,particles):
    print(particles)
    
    global position_new_true 
    F=np.array([[1,0,0],[0,1,0],[0,0,1]])

    r=0.1
    l=0.3
    x=[]
    y=[]
    delta_t=1/8
    u_r=u_l=1

    if (dist([position[0,0],position[1,0]],center)<10):
        
        u_r=1
        u_l=0

    if (dist([position[0,0],position[1,0]],center)>11):
        
        u_r=0
        u_l=1

    G=np.array([[r*delta_t*cos(position[2,0]),0],[r*delta_t*sin(position[2,0]),0],[0,delta_t*r/l]])

    u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
    
    position_new = np.matmul(F,position)+np.matmul(G,u) + np.array([[np.random.normal(0,0.001)],[np.random.normal(0,0.001)],[0]])*1/8
    position_new_true = np.matmul(F,position_new_true) + np.matmul(G,u)
    
    polar_pose = convert_polar(position_new)
    print(position)
    for i in range(0,len(particles)):
            print(particles[i][0][0])
            print(particles[i][1][0])
            print(particles[i][2][0])
            if (math.dist([particles[i][0][0],particles[i][1][0]],center)<10):
                u_r=1
                u_l=0
            if (math.dist([particles[i][0][0],particles[i][1][0]],center)>11):
                u_r=0
                u_l=1

            u=np.array([[(u_r+u_l)/2],[u_r-u_l]])
            
            temp=np.matmul(F,particles[i])+np.matmul(G,u) + delta_t*np.array([[np.random.normal(0,0.01)],[np.random.normal(0,0.1)],[0]])
            # temp = np.matmul(F,particles[i]) + np.matmul(G,u)+ np.array([[np.random.normal(0,0.1)],[np.random.normal(0,0.15)]])*delta_t
            particles[i]=temp    
        
            x.append(temp[0][0])
            y.append(temp[1][0])
    return position_new,polar_pose,particles,x,y
    
x_change = 0


def update():
    global p_0 
    delta_t=1/8
    F=np.array([[1,0,0],[0,1,0],[0,0,1]])
    temp=np.matmul(F,p_0)
    Q=np.array([[0.01,0,0],[0,0.1,0],[0,0,0]])*1/8

    p_new=np.matmul(temp,F.transpose())+Q

    p_0=p_new
    

def measurement():
    global position_measure
    global position_new_true
    global position
    global polar_measure
        
    R = np.asarray([[np.random.normal(0,0.001)],[np.random.normal(0,0.001)]])
    
    Z = convert_polar(position_new_true) + R

    polar_measure = Z
    
    
def correction():
    global p_0
    global position_new
    
    rho = np.sqrt((position[0,0] - center[0,0])**2 + (position[1,0] - center[1,0])**2)
    phi = np.arctan2(position[1,0] - center[1,0], position[0,0] - center[0,0])

    x1 = (position[0,0] - center[0,0])/rho
    y1 = (position[1,0] - center[1,0])/rho
    
    x2 = -1*(position[1,0] - center[1,0])/(rho**2)
    y2 = 1*(position[0,0] - center[0,0])/(rho**2)
    
    H=np.array([[x1,y1,0],[x2,y2,-1]])

    R=np.array([[0.1,0],[0,0.01]])

    temp1=np.matmul(p_0,H.transpose())
    temp2=np.matmul(np.matmul(H,p_0),H.transpose()) + R
    
    k = np.matmul(temp1,np.linalg.inv(temp2)) 
    print(temp1)
    print(temp2)
    print(k)
    k[np.isnan(k)] = 0

    return H,k
    
def update_final(H,K):
    global p_0
    global position
    global position_polar


    p_0=np.matmul((np.identity(3)-np.matmul(K,H)),p_0)
    K = [[K[0,0],K[0,1]],[K[1,0],K[1,1]]]
    
    position_polar = position_polar + np.matmul( K , (polar_measure - position_polar) )

    
    Pose = np.array([center[0,0] + [position_polar[0,0]*cos(position_polar[1,0])], center[1,0] + [position_polar[0,0]*sin(position_polar[1,0])]])

    position[0,0] = Pose[0,0]
    position[1,0] = Pose[1,0]


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
    print(particles)
    position,position_polar,particles,x,y=estimate_pose(position,particles)
    mean_x,std_x,mean_y,std_y=compute_cov(x,y)
    update()
    
    if(t%8==0):
        measurement()
        H_new,K_new=correction()
        update_final(H_new,K_new)
        position_particle,dist1=compute_weight(particles)
        particles=resample(particles, dist1) 

    WHITE=(255,255,255)
    BLUE=(0,0,255)
    RED=(255,0,0)
    GREEN=(0,255,0)
    yellow=(234, 221, 202)

    gameDisplay.fill(WHITE)
    for p in particles:
        pygame.draw.rect(gameDisplay,RED,(p[0][0]*1000+400,(p[1][0])*1000+400,2,2)) 
    surface = pygame.Surface((320, 240))
    red = (180, 50, 50)
    size = (position_particle[0,0]*1000+400-(std_x*2000)/2, position_particle[1,0]*1000+400-(2000*std_y)/2, std_x*2000, 2000*std_y)
    pygame.draw.ellipse(gameDisplay, red, size,1)    
    
    Pose = np.array([center[0,0] + [position_polar[0,0]*cos(position_polar[1,0])], center[1,0] + [position_polar[0,0]*sin(position_polar[1,0])]])
    Pose_measure = np.array([center[0,0] + [polar_measure[0,0]*cos(polar_measure[1,0])], center[1,0] + [polar_measure[0,0]*sin(polar_measure[1,0])]])
    pygame.draw.polygon(gameDisplay, BLUE,
                        [[position_particle[0,0]*1000+400,position_particle[1,0]*1000+400],[position_particle[0,0]*1000+390,position_particle[1,0]*1000+390] ,
                        [position_particle[0,0]*1000+400,position_particle[1,0]*1000+410]])
    if(Pose_measure[0,0]!=10):
        print(Pose_measure)
        measured_positions_x.append(Pose_measure[0,0])
        measured_positions_y.append(Pose_measure[1,0])
        points_measure.append([Pose_measure[0,0]*1000+400,(Pose_measure[1,0])*1000+400])
        pygame.draw.lines(gameDisplay,RED,False,points_measure,5)
    pygame.draw.rect(gameDisplay,GREEN,(400+1000*(Pose_measure[0,0]),400+1000*(Pose_measure[1,0]),10,10))
    points.append([position_particle[0,0]*1000+400,position_particle[1,0]*1000+400])
    points_gt.append([position_new_true[0,0]*1000+400,position_new_true[1,0]*1000+400])
    pygame.draw.rect(gameDisplay,yellow,(position_particle[0,0]*1000+400,(position_particle[1,0])*1000+400,10,10))
    pygame.draw.lines(gameDisplay,BLUE,False,points,5)
    pygame.draw.lines(gameDisplay,GREEN,False,points_gt,5)

    print("********")
    print(points_measure)
    pygame.display.update()
    clock.tick(8)

    predicted_positions_x.append(position_particle[0,0])
    predicted_positions_y.append(position_particle[1,0])
    cov_hight.append(p_0[0,0])
    cov_width.append(p_0[1,1])
    t+=1

plt.plot(measured_positions_x,measured_positions_y, label='measurement')
plt.plot(predicted_positions_x,predicted_positions_y, label='mean particle point')
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
