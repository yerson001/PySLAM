import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

# 1) KITTI Dataset
# 2) College Library Indoors
# 3) Indoor Hallway




print("1. KITTI Dataset")
print("2. College Library Indoors")
print("3. Indoor Hallway")

selection = 1
#selection = input("Enter your choice number: ")

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
#cam = PinholeCamera(1280.0, 720.0, 921.170702, 919.018377, 459.904354, 351.238301, -0.033458, 0.105152 , 0.001256, -0.006647, 0.000000)

#camera_matrix = numpy.array([
# [921.170702, 0.000000, 459.904354],
# [0.000000, 919.018377, 351.238301],
# [0.000000, 0.000000, 1.000000]])

# distorsión = numpy.array(
# [-0.033458,0.105152 , 0.001256, -0.006647, 0.000000])

"""
if selection == "1":
	cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157) 

elif selection == "2" or selection == "3":
	cam = PinholeCamera(1080.0, 1920.0, 718.8560, 718.8560, 607.1928, 185.2157) #phone camera
"""
vo = VisualOdometry(cam, './mono_vo/ground_truth.txt')

#traj = np.zeros((600,600,3), dtype=np.uint8)

traj = np.zeros((700, 700, 3), dtype=np.uint8)
traj[:, :, :] = 128  # Establece todos los valores de los canales a 128 (gris medio)


grid_size = 10  # Tamaño de la celda de la cuadrícula
color = (120, 120, 120)  # Color de las líneas de la cuadrícula

# Dibujar las líneas horizontales de la cuadrícula
for y in range(0, traj.shape[0], grid_size):
    cv2.line(traj, (0, y), (traj.shape[1], y), color, 1)

# Dibujar las líneas verticales de la cuadrícula
for x in range(0, traj.shape[1], grid_size):
    cv2.line(traj, (x, 0), (x, traj.shape[0]), color, 1)


range1 = 4541
range2 = 1363
range3 = 1115


"""
		if selection == "1":
			

		elif selection == "2":
			img = cv2.imread('data/college-library/'+str(img_id).zfill(6)+'.jpg', 0)

		else:
			img = cv2.imread('/home/yerson/Videos/video_.mp4')
"""

def init():
	for img_id in range(range1):
		img = cv2.imread('/home/yerson/data/kitty/sequences/15/image_0/'+str(img_id).zfill(6)+'.png',0)

		print(img.shape)

		vo.update(img, img_id)

		cur_t = vo.cur_t
		if(img_id > 2):
			x, y, z = cur_t[0], cur_t[1], cur_t[2]
		else:
			x, y, z = 0., 0., 0.
		draw_x, draw_y = int(x)+290, int(z)+90
		true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

		#cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
		cv2.circle(traj, (draw_x, 400 - draw_y), 1, (img_id*255/4540, 255-img_id*255/4540, 0), 1)

		cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
		text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
		cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

		cv2.imshow('Camera View', img)
		cv2.imshow('Trajectory', traj)
		cv2.waitKey(1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.imwrite("kitty1.png",traj)

def init2():
	vid = cv2.VideoCapture('/home/yerson/Videos/video_2.mp4')
	img_id = 0
	while(vid.isOpened()):
		ret, img = vid.read()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#print(img.shape,"-->")

		#"""
		vo.update(img, img_id)

		cur_t = vo.cur_t
		if(img_id > 2):
			x, y, z = cur_t[0], cur_t[1], cur_t[2]
		else:
			x, y, z = 0., 0., 0.
		draw_x, draw_y = int(x)+290, int(z)+90
		true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

		#cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
		cv2.circle(traj, (draw_x, 700 - draw_y), 1, (img_id*255/4540, 255-img_id*255/4540, 0), 1)

		cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
		text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
		cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
		#"""

		cv2.imshow('Camera View', img)
		cv2.imshow('Trajectory', traj)
		cv2.waitKey(1)

		# 720 1280
		# 376 1241
		img_id += 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# After the loop release the cap object
	vid.release()
	# Destroy all the windows
	cv2.destroyAllWindows()


x, y = 0, 0

def draw_trajectory(event, x_pos, y_pos, flags, param):
    global x, y

    # Actualizar las coordenadas cuando se presione una tecla
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = x_pos, y_pos

# Crear una ventana llamada "Trajectory"
cv2.namedWindow('Trajectory')

# Asociar la función de dibujo a la ventana
cv2.setMouseCallback('Trajectory', draw_trajectory)

while True:
    # Crear una imagen en blanco para dibujar la trayectoria
    traj = np.zeros((700, 700, 3), np.uint8)

    # Dibujar un círculo en las coordenadas actuales
    cv2.circle(traj, (x, y), 10, (0, 255, 0), -1)

    # Mostrar la imagen de la trayectoria
    cv2.imshow('Trajectory', traj)

    # Esperar por la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todas las ventanas
cv2.destroyAllWindows()


init()
#draw_trajectory()
#cv2.imwrite('map.png', traj)
