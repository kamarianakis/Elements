from pyassimp import load
from gate_module_euclidean import *
import Elements.features.GA.quaternion as quat
import imgui

alpha = 0
flag = False
tempo = 0.05
anim = True
animationLevel = 2

def lerp(a, b, t):
    return (1 - t) * a + t * b

#need to add 2 M arrays, one for Keyframe1 and and one for Keyframe2 
def animationInitialize(file, keyframe1, keyframe2, keyframe3 = None ):
    figure = load(file)
    mesh_id = 3

    #Vertices, Incdices/Faces, Bones from the scene we loaded with pyassimp
    mesh = figure.meshes[mesh_id]
    v = mesh.vertices
    f = mesh.faces
    b = mesh.bones

    #Populating vw with the bone weight and id
    vw = vertex_weight(len(v))
    vw.populate(b)

    #Homogenous coordinates for the vertices
    v2 = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1)

    #Creating random colors for our vertices, will be removed later
    c = []
    min_y = min(v, key=lambda v: v[1])[1]
    max_y = max(v, key=lambda v: v[1])[1]
    for i in range(len(v)):
        color_y = (v[i][1] - min_y) / (max_y - min_y)
        c.append([0, color_y, 1-color_y , 1])

    #Flattening the faces array
    f2 = f.flatten()

    transform = True

    #Initialising M array
    M = initialize_M(b)

    #Initialising first keyframe
    M[1] = np.dot(np.diag([1,1,1,1]),M[1])
    keyframe1.array_MM.append(read_tree(figure,mesh_id,M,transform))

    #Initialising second keyframe
    M[1][0:3,0:3] = eulerAnglesToRotationMatrix([0.3,0.3,0.4])
    M[1][0:3,3] = [0.5,0.5,0.5]
    keyframe2.array_MM.append(read_tree(figure,mesh_id,M,transform))


    M[1][0:3,0:3] = eulerAnglesToRotationMatrix([-0.5,0.3,0.4])
    M[1][0:3,3] = [0.5,0.5,0.5]
    keyframe3.array_MM.append(read_tree(figure,mesh_id,M,transform))
    #Initialising BB array
    BB = [b[i].offsetmatrix for i in range(len(b))]

    # Flattening BB array to pass as uniform variable
    BBData = np.array(BB, dtype=np.float32).reshape((len(BB), 16))
    
    return v2, c, vw.weight, vw.id, f2, BBData


def animationLoop(keyframe1, keyframe2, keyframe3 = None):
    global alpha, tempo, anim, flag, animationLevel
    if alpha > animationLevel:
        alpha = animationLevel
    if alpha < 0:
        alpha = 0

    MM1 = []

    #So we can have repeating animation
    if alpha == animationLevel:
        flag = True
    elif alpha == 0:
        flag = False

    #Filling MM1 with 4x4 identity matrices
    for i in range(len(keyframe1.rotate)):
        MM1.append(np.eye(4))


    if alpha <= 1 or keyframe3== None:
        for i in range(len(keyframe1.rotate)):
            #SLERP
            MM1[i][:3, :3] = quat.Quaternion.to_rotation_matrix(quat.quaternion_slerp(keyframe1.rotate[i], keyframe2.rotate[i], alpha))
            #LERP
            MM1[i][:3, 3] = lerp(keyframe1.translate[i], keyframe2.translate[i], alpha)
    elif keyframe3 != None:
        for i in range(len(keyframe1.rotate)):
            #SLERP
            MM1[i][:3, :3] = quat.Quaternion.to_rotation_matrix(quat.quaternion_slerp(keyframe2.rotate[i], keyframe3.rotate[i], alpha-1))
            #LERP
            MM1[i][:3, 3] = lerp(keyframe2.translate[i], keyframe3.translate[i], alpha-1)

    # Flattening MM1 array to pass as uniform variable
    MM1Data =  np.array(MM1, dtype=np.float32).reshape((len(MM1), 16))

    #So we can have repeating animation
    if alpha >= 0 and alpha < animationLevel and flag == False:
        if anim == True:
            alpha += tempo
        elif anim == False:
            alpha += 0
    elif alpha > 0 and alpha <= animationLevel and flag == True:
        if anim == True:
            alpha -= tempo
        elif anim == False:
            alpha += 0

    return MM1Data


def animationGUI():
    global tempo, anim

    imgui.begin("Animation controls", True)

    _, tempo = imgui.drag_float("Alpha Tempo", tempo, 0.0025, 0, 1)
    _, anim = imgui.checkbox("Animation", anim)

    imgui.end()