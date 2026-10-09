"""A visual test bench: separate left/right patterns, no random pixels."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
textures = ROOT/'worlds/textures'
textures.mkdir(parents=True,exist_ok=True)
pattern = np.array([[35,220,35,220],[220,35,220,35],[35,220,35,220],[220,35,220,35]],dtype=np.uint8)
cv2.imwrite(str(textures/'probe.png'),cv2.resize(pattern,(256,256),interpolation=cv2.INTER_NEAREST))
base = (ROOT/'worlds/corridor.wbt').read_text().split('# Broad wall panels')[0]
base = base.replace('Flybrain: doorway and offset obstacle','Flybrain: controlled visual trials')
for side,y,color in [('LEFT',0.57,'0.08 0.38 0.62'),('RIGHT',-0.57,'0.85 0.34 0.12')]:
    base += f'''
Solid {{
  translation 1.3 {y} 1
  name "{side.lower()} stimulus screen"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }}
      geometry Box {{ size 0.06 1.13 1.85 }}
    }}
    Pose {{
      translation -0.036 0 0
      rotation 0 1 0 -1.5707963267948966
      children [ Shape {{
        appearance PBRAppearance {{
          baseColorMap ImageTexture {{ url [ "textures/probe.png" ] }}
          textureTransform DEF {side}_PATTERN TextureTransform {{ scale 2 2 }}
          roughness 1
          metalness 0
        }}
        geometry Plane {{ size 1.62 1.01 }}
      }} ]
    }}
  ]
  boundingObject Box {{ size 0.06 1.13 1.85 }}
}}
'''
# For a plane rotated about Y, its local X becomes vertical; local Y stays
# horizontal. The experiment controller maps texture coordinates accordingly.
base += '''
DEF DRONE Crazyflie {
  translation -0.8 0 0.02
  controller "flybrain_webots"
  supervisor TRUE
  extensionSlot [
    DEF PROBE_CAMERA_POSE Pose {
      children [ Camera { name "probe_camera" fieldOfView 0.87 width 324 height 324 } ]
    }
  ]
}
'''
(ROOT/'worlds/explain.wbt').write_text(base)
