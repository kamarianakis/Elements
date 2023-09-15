import numpy as np
from Elements.pyECSS.Component import Component, CompNullIterator
import Elements.pyECSS.System
import Elements.features.GA.quaternion as quat

class Keyframe(Component):

    def __init__(self, name=None, type=None, id=None, array_MM=None):
        super().__init__(name, type, id)

        self._parent = self
        if not array_MM:
            self._array_MM = [] 
        else:
            self._array_MM = array_MM
    
    @property
    def array_MM(self):
        return self._array_MM
    
    @array_MM.setter
    def array_MM(self, value):
        self._array_MM = value 

    @property #translation vector
    def translate(self):
        tempMatrix = self.array_MM.copy();
        translateMatrix = [];
        for i in range(len(tempMatrix)):
            for j in range(len(tempMatrix[i])):
                translateMatrix.append(tempMatrix[i][j][:3,3])
        return translateMatrix

    @property #rotation vector
    def rotate(self):
        # First get rotation matrix from trs. Divide by scale
        tempMatrix = self.array_MM.copy();
        rotateMatrix = [];
        for i in range(len(tempMatrix)):
            for j in range(len(tempMatrix[i])):
                rotateMatrix.append(quat.Quaternion.from_rotation_matrix(tempMatrix[i][j]))
        return rotateMatrix


    def update(self):
        pass
   
    def accept(self, system: Elements.pyECSS.System, event = None):
        #system.apply2Keyframe(self)
        pass
    
    def init(self):
        pass
    
    def print(self):
        """
        prints out name, type, id, parent of this Component
        """
        print(f"\n {self.getClassName()} name: {self._name}, type: {self._type}, id: {self._id}, parent: {self._parent._name}, array_MM: \n{self._array_MM}")
        print(f" ______________________________________________________________")
    
    def __str__(self):
        return f"\n {self.getClassName()} name: {self._name}, type: {self._type}, id: {self._id}, parent: {self._parent._name}, array_MM: \n{self._array_MM}"

    def __iter__(self) ->CompNullIterator:
        """ A component does not have children to iterate, thus a NULL iterator
        """
        return CompNullIterator(self) 
    

class AnimationComponents(Component):

    def __init__(self, name=None, type=None, id=None, bones=None, MM=None, alpha=0, tempo=2, time_add=0, animation_start = True, anim_keys = 2, time = [0, 100, 200], flag = True, inter = 'SLERP'):
        super().__init__(name, type, id)
        self._parent = self

        self.alpha = alpha
        self.tempo = tempo
        self.time_add = time_add
        self.anition_start = animation_start
        self.animKeys = anim_keys
        self.inter = inter
        self.time = time
        self.flag = flag

        self.MM = []

        if not bones:
            self._bones = [] 
        else:
            self._bones = bones

    @property
    def bones(self):
        return self._bones
    
    @bones.setter
    def bones(self, value):
        self._bones = value 

    #Animation loop
    def animation_loop(self, key1, key2, key3=None):
        #Filling MM1 with 4x4 identity matrices
        self.MM = [np.eye(4) for _ in key1]

        if (self.time_add >= self.time[1] and key3 is None) or (self.time_add >= self.time[2]):
            self.flag = False
        elif self.time_add <= self.time[0]:
            self.flag = True


        if self.time_add >= self.time[0] and self.time_add <= self.time[1]:
            self.animation_for_loop(key1,key2, self.time[0], self.time[1])
        elif self.time_add > self.time[1] and self.time_add <= self.time[2] and key3 is not None:
            self.animation_for_loop(key2,key3, self.time[1], self.time[2])
        
        
        #So we can have repeating animation
        if self.flag == True:
            if self.anition_start == True:
                self.time_add += self.tempo
            else:
                self.time_add = self.time_add
        else:
            if self.anition_start == True:
                self.time_add -= self.tempo
            else:
                self.time_add = self.time_add

        # Flattening MM1 array to pass as uniform variable
        MM1Data =  np.array(self.MM, dtype=np.float32).reshape((len(self.MM), 16))

        return MM1Data
    
    def animation_for_loop(self, k_1, k_2, t0, t1):
        self.alpha = (self.time_add - t0) / abs(t1 - t0)

        keyframe1 = Keyframe(array_MM=[k_1])
        keyframe2 = Keyframe(array_MM=[k_2])

        for i in range(len(k_1)):
            if(self.inter == "LERP"):
                self.MM[i][:3, :3] = quat.Quaternion.to_rotation_matrix(quat.quaternion_lerp(keyframe1.rotate[i], keyframe2.rotate[i], self.alpha))
                self.MM[i][:3, 3] = self.lerp(keyframe1.translate[i], keyframe2.translate[i], self.alpha)
            else:
                #SLERP
                self.MM[i][:3, :3] = quat.Quaternion.to_rotation_matrix(quat.quaternion_slerp(keyframe1.rotate[i], keyframe2.rotate[i], self.alpha))
                #LERP
                self.MM[i][:3, 3] = self.lerp(keyframe1.translate[i], keyframe2.translate[i], self.alpha)

    def lerp(self,a, b, t):
        return (1 - t) * a + t * b
     
    def update(self):
        pass
   
    def accept(self, system: Elements.pyECSS.System, event = None):
        pass
    
    def init(self):
        pass