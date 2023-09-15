import warnings
warnings.filterwarnings('ignore')

import numpy as np
from numpy import e,pi
from pyassimp import *
import meshplot as mp
import math
# from numba import jit,njit


class vertex_weight:
    # =================================================
    # Creating a class that contains the vertices and 
    # and the respective weights that are affected by the
    # bones. 
    # =================================================
  def __init__(self,v_length):
    self.id = np.full([v_length,4],-1)
    self.weight = np.zeros([v_length,4])

  def add(self,v,b,w):
    position = 0
    for i in range(4):
        if self.weight[v][position]>0 :
            position +=1
    if position <= 4:
        self.weight[v][position]=w
        self.id[v][position]=b

  def populate(self,b):
    for boneID in range(len(b)):
      for i in range(len(b[boneID].weights)):
        self.add(b[boneID].weights[i].vertexid, boneID ,b[boneID].weights[i].weight)


def get_bone_names(b):
    return [str(b[i]) for i in range(len(b))]


def vertex_apply_M(v,M):
    return np.dot(M,np.append(v,[1.0]))[0:3]


def eulerAnglesToRotationMatrix(theta) :  
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])             
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                  
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def B(x):
    return b[x].offsetmatrix
#     if x == 0:
#         return b[x].offsetmatrix
#     else :
#         return np.dot(B(x-1),b[x].offsetmatrix)
  

def draw_original_wireframe():
    p.add_lines(v[f[:,0]], v[f[:,1]], shading={"line_color": "magenta"});
    p.add_lines(v[f[:,2]], v[f[:,1]], shading={"line_color": "magenta"});
    p.add_lines(v[f[:,2]], v[f[:,0]], shading={"line_color": "magenta"});


def initialize_M(b):
  M = np.zeros([len(b),4,4])
  for i in range(len(b)):
    M[i] = np.identity(4)
  return M


def get_bone_names(b):
    return [str(b[i]) for i in range(len(b))]


def read_tree(scene,mesh_id,M,transform):
  b = scene.meshes[mesh_id].bones
  MM = np.zeros([len(b),4,4])
  G = np.linalg.inv(scene.rootnode.transformation)
  bone_names = get_bone_names(b)
  read_tree_names(MM,M,scene.rootnode,G,bone_names,transform)
  return MM


def read_tree_names(MM,M,node,parentmatrix,bone_names, transform = False):
    p = np.dot(parentmatrix,node.transformation)
    if transform == True :
        if node.name in bone_names:
            index = bone_names.index(node.name)
            p = np.dot(p,M[index])
            MM[index] = p
        for child in node.children:
            read_tree_names(MM,M,child,p,bone_names,True)            
    else:
        if node.name in bone_names:
            index = bone_names.index(node.name)
            MM[index] = p
        for child in node.children:
            read_tree_names(MM,M,child,p,bone_names,False)        
    

def draw_vertices_affected_by_boneID_on_new(boneID):
    vertices_ids_affected_by_boneID = [b[boneID].weights[i].vertexid for i in range(len(b[boneID].weights))]
    weights_affected_by_boneID = [b[boneID].weights[i].weight for i in range(len(b[boneID].weights))]
    IDs = vertices_ids_affected_by_boneID
    print("red points are the ones affected by :", b[boneID])
#     p.add_points(v[IDs], shading={"point_size": 0.7,"point_color": "red"}); 
    p.add_points(newv[IDs], shading={"point_size": 1,"point_color": "red"}); 


def draw_vertices_affected_by_boneID(boneID):
    vertices_ids_affected_by_boneID = [b[boneID].weights[i].vertexid for i in range(len(b[boneID].weights))]
    IDs = vertices_ids_affected_by_boneID
    print("red points are the ones affected by :", b[boneID])
    p.add_points(v[IDs], shading={"point_size": 0.7,"point_color": "red"}); 


##### ALL BELOW ARE NEEDED FOR GA WAY
def print_original():
    newv = np.zeros([(len(v)),3])
    for i in range(len(v)):
        for j in range(4):
            if vw.id[i][j] >=0:
                rotor =  (MMM[vw.id[i][j]])*BBB[vw.id[i][j]] 
    #             rotor =  (rotors[vw.id[i][j]])*BBB[vw.id[i][j]] 
                temp = up_cga_point_to_euc_point(rotor*cgav[i]*~rotor)
                newv[i] = newv[i] + vw.weight[i][j]*temp
    # p = mp.plot(newv, f,newv[:, 1],shading={"scale": 2.5,"wireframe":True},return_plot=True)

    p.add_lines(newv[f[:,0]], newv[f[:,1]], shading={"line_color": "magenta"});
    p.add_lines(newv[f[:,2]], newv[f[:,1]], shading={"line_color": "magenta"});
    p.add_lines(newv[f[:,2]], newv[f[:,0]], shading={"line_color": "magenta"});


def matrix_to_angle_axis_translation(matrix):
    # taken from:   https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = math.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    if abs(theta) < 1e-6 :
        axis = [1,0,0]
    
    return theta, axis, matrix[0:3,3]


def up_vertex(v):
  return up(v[0]*e1+v[1]*e2+v[2]*e3)


# AUXILIARY 

def rotation_matrix( direction, angle, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[0.0, -direction[2], direction[1]],
                      [direction[2], 0.0, -direction[0]],
                      [-direction[1], direction[0], 0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data*data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None