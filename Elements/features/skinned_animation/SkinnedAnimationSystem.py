import Elements.pyECSS
from Elements.pyECSS.System import System
import Elements.pyECSS.Component
from Elements.features.skinned_animation.AnimationComponent import AnimationComponents

class SkinnedAnimationSystem(System):
    def __init__(self, name=None, type=None, id=None):
        super().__init__(name, type, id)
        self.keyframes = []  # Store your keyframes here

    def apply2AnimationComponents(self, animationComponents: AnimationComponents):

        if (isinstance(animationComponents, AnimationComponents)) == False:
            return #in Python due to duck typing we need to check this!
        #print(self.getClassName(), ": apply2AnimationComponents called")

        if len(self.keyframes) == 2:
            animation_data = animationComponents.animation_loop(self.keyframes[0], self.keyframes[1])
        elif len(self.keyframes) == 3:
            animation_data = animationComponents.animation_loop(self.keyframes[0], self.keyframes[1], self.keyframes[2])
        else:
            return
        
        return animation_data
        