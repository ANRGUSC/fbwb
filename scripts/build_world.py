"""Create a minimal, offline Webots world with visual texture for optic flow."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    textures = ROOT / 'worlds/textures'
    textures.mkdir(parents=True, exist_ok=True)
    # Deterministic irregular tiles prevent a textureless floor/walls.
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(41)
    gray = rng.integers(70, 210, size=(16, 16, 1), dtype=np.uint8)
    tiles = np.repeat(gray, 3, axis=2)
    cv2.imwrite(str(textures / 'tiles.png'), cv2.resize(tiles, (512, 512), interpolation=cv2.INTER_NEAREST))
    checker = np.array([[170,190],[190,170]], dtype=np.uint8)
    cv2.imwrite(str(textures / 'floor.png'), cv2.resize(checker, (128,128), interpolation=cv2.INTER_NEAREST))
    world = '''#VRML_SIM R2025a utf8
EXTERNPROTO "../protos/Crazyflie.proto"
WorldInfo {
  basicTimeStep 20
  coordinateSystem "ENU"
  title "Flybrain / Crazyflie"
}
Viewpoint {
  orientation ORIENTATION
  position -1.2 -1.2 1.8
  follow "Crazyflie"
  followType "Pan and Tilt Shot"
}
Background { skyColor [ 0.65 0.75 0.88 ] }
DirectionalLight { direction -1 -1 -2 intensity 1 }
Solid {
  translation 0 0 -0.05
  children [
    Shape {
      appearance DEF TEXTURE PBRAppearance {
        baseColorMap ImageTexture { url [ "textures/tiles.png" ] }
        textureTransform TextureTransform { scale 3 3 }
        roughness 1
        metalness 0
      }
      geometry Box { size 8 8 0.1 }
    }
  ]
  name "floor"
  boundingObject Box { size 8 8 0.1 }
}
'''
    for name, position, size in [('front', '4 0 1.5', '0.1 8 3'), ('back', '-4 0 1.5', '0.1 8 3'), ('left', '0 4 1.5', '8 0.1 3'), ('right', '0 -4 1.5', '8 0.1 3')]:
        world += f'''Solid {{
  translation {position}
  name "{name} wall"
  children [ Shape {{ appearance USE TEXTURE geometry Box {{ size {size} }} }} ]
  boundingObject Box {{ size {size} }}
}}
'''
    world += '''DEF DRONE Crazyflie {
  translation 0 0 0.02
  controller "flybrain_webots"
  supervisor TRUE
}
'''
    forward = np.array([1.2, 1.2, -0.8])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Webots ENU view frame looks along local +X, with local +Z up.
    rot = Rotation.from_matrix(np.column_stack([forward, -right, up])).as_rotvec()
    angle = np.linalg.norm(rot)
    world = world.replace('ORIENTATION', ' '.join(map(str, [*(rot/angle), angle])))
    (ROOT / 'worlds/flybrain.wbt').write_text(world)
    runtime = ROOT / 'controllers/flybrain_webots/runtime.ini'
    runtime.write_text('[python]\nCOMMAND = "' + sys.executable.replace('\\', '/') + '"\n')


if __name__ == '__main__':
    main()
