"""
@author: lks-ai
@title: StableAudioSampler
@nickname: stableaudio
@description: A Simple integration of Stable Audio Diffusion with knobs and stuff!
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 主程序入口 ---------------------------------------------------------
# if __name__ == "__main__":
#     print(f'测试文本到声音')
#     stable_node = StableAudioFG()
#     stable_node.generate("a bird singing", "noise", 30, 7.0, 5)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
# WEB_DIRECTORY = "./web"
