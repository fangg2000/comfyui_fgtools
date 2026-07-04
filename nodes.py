import os, sys, json, uuid
import torch
import numpy as np
import requests

from PIL import Image
from collections import deque
from typing import List, Tuple, Optional


def get_comfyui_root():
    main_module = sys.modules.get('__main__')
    if main_module and hasattr(main_module, '__file__'):
        main_path = os.path.abspath(main_module.__file__)
        root_dir = os.path.dirname(main_path)
        return root_dir
    return None

"""
查找图片中透明度大于指定阈值的连续区域边界
"""

def find_transparent_regions(image: Image.Image, alpha_threshold: float) -> List[Tuple[int, int, int, int]]:
    """
    查找图片中透明度大于指定阈值的连续区域边界

    Args:
        image: 要分析的图片 (PIL Image对象，需要是RGBA模式)
        alpha_threshold: 透明度阈值(0-1)，大于此值的像素被视为透明点

    Returns:
        包含所有连续区域的边界矩形列表 [(x, y, width, height), ...]
    """
    regions = []

    # 确保图像是RGBA模式
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    width, height = image.size
    pixels = image.load()

    # 创建标记数组，标记像素是否已访问
    visited = [[False for _ in range(width)] for _ in range(height)]

    # 遍历所有像素
    for y in range(height):
        for x in range(width):
            # 如果像素未访问且透明度大于阈值
            if not visited[y][x] and _is_transparent_enough(pixels[x, y], alpha_threshold):
                # 使用BFS查找连续区域
                region = _find_connected_region(pixels, x, y, visited, alpha_threshold, width,
                                                                     height)
                if region:
                    regions.append(region)

    return regions

def _find_connected_region(pixels, start_x: int, start_y: int, visited: List[List[bool]],
                           alpha_threshold: float, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    查找单个连续透明区域的最小最大XY位置
    """
    # 初始化边界值
    min_x = max_x = start_x
    min_y = max_y = start_y

    # 使用队列进行BFS遍历
    queue = deque()
    queue.append((start_x, start_y))
    visited[start_y][start_x] = True

    # 定义4个方向的偏移量：右、下、左、上
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    while queue:
        x, y = queue.popleft()

        # 更新边界值
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

        # 检查四个方向的相邻像素
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy

            # 检查边界和访问状态
            if 0 <= new_x < width and 0 <= new_y < height:
                if not visited[new_y][new_x] and _is_transparent_enough(pixels[new_x, new_y],
                                                                                             alpha_threshold):
                    visited[new_y][new_x] = True
                    queue.append((new_x, new_y))

    # 返回区域的边界矩形 (x, y, width, height)
    return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

def _is_transparent_enough(pixel: Tuple[int, int, int, int], alpha_threshold: float) -> bool:
    """
    检查像素的透明度是否大于阈值

    Args:
        pixel: RGBA像素值 (r, g, b, a)
        alpha_threshold: 透明度阈值

    Returns:
        True表示透明度足够高
    """
    # 获取alpha通道的值（0-255）
    alpha = pixel[3]

    # 将alpha值转换为0-1范围
    alpha_normalized = alpha / 255.0

    # 判断透明度是否大于阈值
    # alpha值越小表示越透明，所以检查 alpha_normalized < (1 - alpha_threshold)
    return alpha_normalized < (1 - alpha_threshold)

def get_largest_transparent_region(image: Image.Image, alpha_threshold: float) -> Optional[
    Tuple[int, int, int, int]]:
    """
    获取图片中最大连续透明区域的边界

    Returns:
        最大的透明区域边界 (x, y, width, height)，如果没有则返回None
    """
    regions = find_transparent_regions(image, alpha_threshold)

    if not regions:
        return None

    # 按面积排序，返回最大的区域
    regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    return regions[0]

def print_region_bounds(regions: List[Tuple[int, int, int, int]]) -> None:
    """
    打印透明区域的边界信息
    """
    if not regions:
        print("未找到透明度大于阈值的连续区域")
        return

    print(f"找到 {len(regions)} 个连续透明区域:")
    for i, rect in enumerate(regions):
        x, y, width, height = rect
        print(f"区域 {i + 1}:")
        print(f"  X轴范围: [{x}, {x + width - 1}]")
        print(f"  Y轴范围: [{y}, {y + height - 1}]")
        print(f"  宽度: {width}, 高度: {height}")

def calculate_combined_bounds(regions: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    计算所有区域的合并边界

    Args:
        regions: 区域列表 [(x, y, width, height), ...]

    Returns:
        合并后的边界 (x, y, width, height)
    """
    if not regions:
        return (0, 0, 0, 0)

    # 初始化边界值
    min_x = min(rect[0] for rect in regions)
    min_y = min(rect[1] for rect in regions)
    max_x = max(rect[0] + rect[2] for rect in regions)
    max_y = max(rect[1] + rect[3] for rect in regions)

    return (min_x, min_y, max_x - min_x, max_y - min_y)


class IsEmptyString:
    def __init__(self):
        self.result_txt = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "txt": ("STRING",),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "generate"
    DESCRIPT = "check string is empty or not"
    CATEGORY = "fg/tools"
    OUTPUT_NODE = True

    def generate(self, txt: str):
        if txt is None or len(txt.strip()) == 0:
            return (True,)
        return (False,)

class SwitchString:
    def __init__(self):
        self.result_txt = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "txt1": ("STRING",),
                "txt2": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "generate"
    DESCRIPT = "output not null or empty one"
    CATEGORY = "fg/tools"
    OUTPUT_NODE = True

    def generate(self, txt1: str, txt2: str):
        if txt1 is None or len(txt1) == 0:
            if txt2 is None or len(txt2) == 0:
                return ("",)
            return (txt2,)
        return (txt1,)


class InpaintCut:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "padding": ("INT", {"default": 8, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("INPAINT_AREA", "IMAGE", "MASK")
    RETURN_NAMES = ("inpaint_area", "cropped_image", "cropped_mask")
    FUNCTION = "cut"
    CATEGORY = "fg/inpaint"

    def cut(self, image, mask, padding=8):
        # 获取图像尺寸 (B, H, W, C)
        B, H, W, C = image.shape

        # 处理mask：取第一个batch并确保是2D
        if len(mask.shape) == 3:
            mask_2d = mask[0].clone()  # 取第一个batch
        elif len(mask.shape) == 2:
            mask_2d = mask.clone()
        else:
            mask_2d = mask.squeeze()

        # 如果mask是2D但值在0-255之间，归一化到0-1
        mask_np = mask_2d.cpu().numpy()
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0

        # 调整mask大小以匹配图像（如果需要）
        mask_H, mask_W = mask_np.shape
        if mask_H != H or mask_W != W:
            # 使用PIL进行插值
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert('L')
            mask_pil = mask_pil.resize((W, H), Image.BILINEAR)
            mask_np = np.array(mask_pil) / 255.0

        # 关键：正确创建RGBA图像
        # 创建一个黑色背景的RGBA图像，alpha通道使用mask值
        rgba_array = np.zeros((H, W, 4), dtype=np.uint8)

        # 设置alpha通道：mask值越高，alpha值越低（越透明）
        # mask中，1表示需要修复的区域（透明），0表示保留的区域（不透明）
        alpha_channel = ((1.0 - mask_np) * 255).astype(np.uint8)
        rgba_array[:, :, 3] = alpha_channel  # 设置alpha通道

        # 将numpy数组转换为PIL图像
        rgba_image = Image.fromarray(rgba_array, 'RGBA')

        # 使用TransparencyAnalyzer的方法查找透明区域
        alpha_threshold = 0.8  # 透明度阈值

        # 调用TransparencyAnalyzer的方法
        regions = find_transparent_regions(rgba_image, alpha_threshold)

        if not regions:
            # 如果没有找到透明区域，返回整个图像
            x, y, width, height = 0, 0, W, H
        else:
            # 使用calculate_combined_bounds计算合并边界
            x, y, width, height = calculate_combined_bounds(regions)

        # 添加padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(W - x, width + 2 * padding)
        height = min(H - y, height + 2 * padding)

        # 确保宽高至少为1
        width = max(1, width)
        height = max(1, height)

        # 截取图像区域
        cropped_images = []
        for b in range(B):
            # 确保索引在范围内
            x_end = min(x + width, W)
            y_end = min(y + height, H)
            actual_width = x_end - x
            actual_height = y_end - y

            crop = image[b, y:y_end, x:x_end, :]
            cropped_images.append(crop)

        # 堆叠回tensor
        cropped_image = torch.stack(cropped_images, dim=0)

        # 截取mask区域
        mask_tensor = torch.from_numpy(mask_np).float()
        cropped_mask = mask_tensor[y:y_end, x:x_end]

        # 构建inpaint_area字典
        inpaint_area = {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "original_image": image,
            "original_mask": mask_tensor.unsqueeze(0),
        }

        return (inpaint_area, cropped_image, cropped_mask.unsqueeze(0))


class InpaintConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpaint_area": ("INPAINT_AREA",),
                "image": ("IMAGE",),
            },
            "optional": {
                "blend_width": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "concat"
    CATEGORY = "fg/inpaint"

    def concat(self, inpaint_area, image, blend_width=30):
        # 从inpaint_area中获取信息
        x = inpaint_area["x"]
        y = inpaint_area["y"]
        width = inpaint_area["width"]
        height = inpaint_area["height"]
        original_image = inpaint_area["original_image"]
        original_mask = inpaint_area["original_mask"]

        # 获取原始图像尺寸
        B, H, W, C = original_image.shape

        # 调整输入图像大小以匹配截取区域
        cropped_B, cropped_H, cropped_W, cropped_C = image.shape

        # 如果尺寸不匹配，调整图像大小
        if cropped_H != height or cropped_W != width:
            # 使用双线性插值调整图像大小
            image_resized = torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),  # 转换为 (B, C, H, W)
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # 转换回 (B, H, W, C)
        else:
            image_resized = image

        # 创建一个原始图像的副本
        result_image = original_image.clone()

        # 确保索引在有效范围内
        y_end = min(y + height, H)
        x_end = min(x + width, W)
        actual_height = y_end - y
        actual_width = x_end - x

        # 如果实际区域与期望区域大小不同，调整image_resized
        if actual_height != height or actual_width != width:
            image_resized = torch.nn.functional.interpolate(
                image_resized.permute(0, 3, 1, 2),
                size=(actual_height, actual_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)

        # 获取对应区域的mask
        if len(original_mask.shape) == 3:
            mask_region = original_mask[0, y:y_end, x:x_end].cpu().numpy()
        else:
            mask_region = original_mask[y:y_end, x:x_end].cpu().numpy()

        # 创建边界融合mask
        # 1. 创建一个全0的mask（大小与修复区域相同）
        blend_mask = np.zeros((actual_height, actual_width), dtype=np.float32)

        # 2. 确定内部区域（去掉边界blend_width像素）
        inner_top = blend_width
        inner_bottom = actual_height - blend_width
        inner_left = blend_width
        inner_right = actual_width - blend_width

        # 确保内部区域有效
        inner_top = max(0, inner_top)
        inner_bottom = min(actual_height, inner_bottom)
        inner_left = max(0, inner_left)
        inner_right = min(actual_width, inner_right)

        # 3. 内部区域设为1（完全使用修复后的图像）
        if inner_top < inner_bottom and inner_left < inner_right:
            blend_mask[inner_top:inner_bottom, inner_left:inner_right] = 1.0

        # 4. 处理四个边界区域，创建渐变
        # 上边界渐变
        for i in range(min(blend_width, actual_height)):
            if i < inner_top:
                weight = i / blend_width if blend_width > 0 else 1.0
                blend_mask[i, :] = weight

        # 下边界渐变
        for i in range(max(0, actual_height - blend_width), actual_height):
            if i >= inner_bottom:
                dist_from_edge = actual_height - i - 1
                weight = dist_from_edge / blend_width if blend_width > 0 else 1.0
                blend_mask[i, :] = weight

        # 左边界渐变（覆盖上下的渐变，取最大值）
        for j in range(min(blend_width, actual_width)):
            if j < inner_left:
                weight = j / blend_width if blend_width > 0 else 1.0
                # 取列中每个像素的最大值（避免重复渐变过度）
                for i in range(actual_height):
                    current_weight = blend_mask[i, j]
                    blend_mask[i, j] = max(current_weight, weight)

        # 右边界渐变
        for j in range(max(0, actual_width - blend_width), actual_width):
            if j >= inner_right:
                dist_from_edge = actual_width - j - 1
                weight = dist_from_edge / blend_width if blend_width > 0 else 1.0
                # 取列中每个像素的最大值
                for i in range(actual_height):
                    current_weight = blend_mask[i, j]
                    blend_mask[i, j] = max(current_weight, weight)

        # 将blend_mask转换为tensor
        blend_mask_tensor = torch.from_numpy(blend_mask).float().to(original_image.device)

        # 扩展mask维度以匹配图像通道数
        if len(blend_mask_tensor.shape) == 2:
            blend_mask_tensor = blend_mask_tensor.unsqueeze(-1)

        # 将mask扩展为4D (B, H, W, C)
        blend_mask_tensor = blend_mask_tensor.unsqueeze(0).expand(B, -1, -1, C)

        # 混合图像
        for b in range(B):
            # 确保batch维度匹配
            current_image_resized = image_resized[b % image_resized.shape[0]]

            # 混合公式: result = image * blend_mask + original * (1 - blend_mask)
            result_image[b, y:y_end, x:x_end, :] = (
                    current_image_resized * blend_mask_tensor[b] +
                    original_image[b, y:y_end, x:x_end, :] * (1 - blend_mask_tensor[b])
            )

        return (result_image,)

import requests
import json
import os

class DoubaoChat:
    # 配置项
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
    MODEL_NAME = "doubao-seed-1-6-250615"
    # 密钥持久化文件路径 (保存在当前节点目录下)
    _KEY_FILE = os.path.join(os.path.dirname(__file__), ".ark_api_key")
    
    # 类变量：缓存当前密钥
    _cached_api_key = None

    @classmethod
    def INPUT_TYPES(cls):
        # 1. 尝试加载已保存的密钥
        cls._load_saved_key()
        
        # 2. 生成掩码显示的密钥 (前6位 + *** + 后4位)
        default_key = ""
        if cls._cached_api_key:
            key_len = len(cls._cached_api_key)
            if key_len > 10:
                default_key = f"{cls._cached_api_key[:6]}***{cls._cached_api_key[-4:]}"
            else:
                default_key = cls._cached_api_key # 太短则不掩码

        return {
            "required": {
                "input": ("STRING", {
                    "multiline": True, 
                    "default": "你好呀。",
                    "placeholder": "请输入你的问题..."
                }),
                "api_key": ("STRING", {
                    "default": default_key,
                    "multiline": False,
                    "placeholder": "请输入 ARK_API_KEY"
                }),
                # 【新增】超时时长参数
                "timeout_seconds": ("INT", {
                    "default": 60,
                    "min": 10,      # 最小1秒
                    "max": 300,    # 最大5分钟（防止过长等待）
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "🚀 AI/LLM"

    def execute(self, input, api_key, timeout_seconds):
        # --- 1. 确定使用的密钥 ---
        use_key = None
        
        # 如果输入的密钥不含 '*'，认为是新密钥
        if "*" not in api_key and api_key.strip() != "":
            use_key = api_key.strip()
        else:
            # 使用缓存的密钥
            if not self._cached_api_key:
                raise ValueError("未保存有效密钥，请输入完整的 ARK_API_KEY (不含星号)")
            use_key = self._cached_api_key

        # --- 2. 构建请求 ---
        headers = {
            "Authorization": f"Bearer {use_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.MODEL_NAME,
            "input": input
        }

        # --- 3. 发送请求 (使用配置的超时时长) ---
        try:
            # 【修改】timeout参数使用传入的 timeout_seconds
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=timeout_seconds)
            response.raise_for_status() # 抛出HTTP错误 (如 401, 500)
            result = response.json()
            
            # --- 4. 解析响应 (适配 Responses API 格式) ---
            output_content = ""
            if "output" in result:
                for item in result["output"]:
                    if item.get("type") == "message":
                        content_list = item.get("content", [])
                        for cnt in content_list:
                            if cnt.get("type") == "output_text":
                                output_content += cnt.get("text", "")
            
            # --- 5. 如果成功，保存新密钥 (如果是新输入的) ---
            if "*" not in api_key and api_key.strip() != "":
                self._save_key(api_key.strip())

            return (output_content, )

        except requests.exceptions.Timeout:
            # 超时无返回，直接返回原始input
            return (input, )
        except requests.exceptions.RequestException as e:
            # 其他请求错误仍抛出异常
            raise Exception(f"API 请求失败: {str(e)}")

    # --- 辅助方法：密钥持久化 ---
    @classmethod
    def _load_saved_key(cls):
        if cls._cached_api_key is None:
            if os.path.exists(cls._KEY_FILE):
                try:
                    with open(cls._KEY_FILE, 'r', encoding='utf-8') as f:
                        cls._cached_api_key = f.read().strip()
                except Exception:
                    pass

    @classmethod
    def _save_key(cls, key):
        cls._cached_api_key = key
        try:
            with open(cls._KEY_FILE, 'w', encoding='utf-8') as f:
                f.write(key)
        except Exception as e:
            print(f"[Warn] 无法保存密钥: {e}")


class DeepSeekChat:
    # 配置项
    API_URL = "https://api.deepseek.com/chat/completions"
    MODEL_NAME = "deepseek-v4-flash"
    # 密钥持久化文件
    _KEY_FILE = os.path.join(os.path.dirname(__file__), ".deepseek_api_key")
    
    # 缓存密钥
    _cached_api_key = None

    @classmethod
    def INPUT_TYPES(cls):
        # 加载已保存密钥
        cls._load_saved_key()
        
        # 掩码显示
        default_key = ""
        if cls._cached_api_key:
            key_len = len(cls._cached_api_key)
            if key_len > 10:
                default_key = f"{cls._cached_api_key[:6]}***{cls._cached_api_key[-4:]}"
            else:
                default_key = cls._cached_api_key

        return {
            "required": {
                "input": ("STRING", {
                    "multiline": True,
                    "default": "Hello!",
                    "placeholder": "输入你的问题..."
                }),
                "api_key": ("STRING", {
                    "default": default_key,
                    "multiline": False,
                    "placeholder": "输入 DeepSeek API Key"
                }),
                "timeout_seconds": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "🚀 AI/LLM"

    def execute(self, input, api_key, timeout_seconds):
        # 1. 确定使用的密钥
        use_key = None
        if "*" not in api_key and api_key.strip() != "":
            use_key = api_key.strip()
        else:
            if not self._cached_api_key:
                raise ValueError("未保存有效密钥，请输入完整的 DeepSeek API Key")
            use_key = self._cached_api_key

        # 2. 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {use_key}"
        }

        # 3. DeepSeek 标准请求体
        payload = {
            "model": self.MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input}
            ],
            "thinking": {"type": "enabled"},
            "stream": False
        }

        # 4. 发送请求
        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )
            response.raise_for_status()
            result = response.json()

            # 5. 解析 DeepSeek 返回格式
            output_content = ""
            try:
                choices = result.get("choices", [])
                if choices:
                    output_content = choices[0]["message"]["content"]
            except:
                output_content = ""

            # 6. 成功则保存新密钥
            if "*" not in api_key and api_key.strip() != "":
                self._save_key(api_key.strip())

            return (output_content,)

        except requests.exceptions.Timeout:
            # 超时返回原始 input
            return (input,)
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API 请求失败: {str(e)}")

    # 密钥保存/加载
    @classmethod
    def _load_saved_key(cls):
        if cls._cached_api_key is None and os.path.exists(cls._KEY_FILE):
            try:
                with open(cls._KEY_FILE, 'r', encoding='utf-8') as f:
                    cls._cached_api_key = f.read().strip()
            except:
                pass

    @classmethod
    def _save_key(cls, key):
        cls._cached_api_key = key
        try:
            with open(cls._KEY_FILE, 'w', encoding='utf-8') as f:
                f.write(key)
        except:
            print("[Warn] 无法保存 DeepSeek API Key")


class PriorityLLMNode:
    # DeepSeek 配置
    DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
    DEEPSEEK_MODEL = "deepseek-v4-flash"
    DEEPSEEK_KEY_FILE = os.path.join(os.path.dirname(__file__), ".deepseek_api_key")
    DEEPSEEK_TIMEOUT = 10  # 固定10秒超时
    
    # 豆包 Ark 配置
    ARK_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
    ARK_MODEL = "doubao-seed-1-6-250615"
    ARK_KEY_FILE = os.path.join(os.path.dirname(__file__), ".ark_api_key")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {
                    "multiline": True,
                    "default": "你好呀。",
                    "placeholder": "输入你的问题..."
                }),
                "ds_timeout": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
                "doubao_timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "🚀 AI/LLM"

    def execute(self, input, ds_timeout, doubao_timeout):
        # --- 1. 优先尝试 DeepSeek (10秒超时) ---
        deepseek_key = self._load_key(self.DEEPSEEK_KEY_FILE)
        if deepseek_key:
            try:
                result = self._call_deepseek(deepseek_key, input, ds_timeout)
                if result:
                    return (result,)
            except requests.exceptions.Timeout:
                print(f"[Info] DeepSeek {ds_timeout}秒超时，切换至豆包")
            except Exception as e:
                print(f"[Info] DeepSeek 调用失败 ({str(e)})，切换至豆包")
        
        # --- 2. DeepSeek 失败，尝试豆包 ---
        ark_key = self._load_key(self.ARK_KEY_FILE)
        if not ark_key:
            raise ValueError("未找到 DeepSeek 或 豆包 的有效密钥，请先在对应节点中配置并调用成功一次")
        
        try:
            result = self._call_ark(ark_key, input, doubao_timeout)
            return (result,)
        except requests.exceptions.Timeout:
            # 豆包也超时，返回原始 input
            return (input,)
        except Exception as e:
            raise Exception(f"豆包 API 调用失败: {str(e)}")

    # --- DeepSeek 调用 ---
    def _call_deepseek(self, api_key, input_text, ds_timeout):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": self.DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            "thinking": {"type": "disabled"},
            "stream": False
        }
        response = requests.post(
            self.DEEPSEEK_URL,
            headers=headers,
            json=payload,
            timeout=ds_timeout
        )
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
        return ""

    # --- 豆包 Ark 调用 ---
    def _call_ark(self, api_key, input_text, timeout):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.ARK_MODEL,
            "input": input_text
        }
        response = requests.post(
            self.ARK_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        output_content = ""
        if "output" in result:
            for item in result["output"]:
                if item.get("type") == "message":
                    content_list = item.get("content", [])
                    for cnt in content_list:
                        if cnt.get("type") == "output_text":
                            output_content += cnt.get("text", "")
        return output_content

    # --- 通用密钥加载 ---
    def _load_key(self, key_file):
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
                    if key:
                        return key
            except:
                pass
        return None

# 定位图片对象
import torch
import numpy as np
from PIL import Image

class ImageTrimAndCenter:
    """
    自动移除图片 / Mask 四周的空白区域，并将内容居中放置到指定尺寸的画布上
    - 支持 IMAGE 和 MASK 独立输入或同时输入
    - 开启对齐模式时，以 Mask 边界为准裁剪图片，保证两者完全对齐
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "fill_color": ("STRING", {"default": "#FFFFFF"}),
                "align_with_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "crop_x", "crop_y", "crop_width", "crop_height")
    FUNCTION = "trim_and_center"
    CATEGORY = "image/transform"

    def hex_to_rgb(self, hex_color):
        """十六进制颜色转 0-1 范围 RGB 值"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def find_content_bounds(self, arr, threshold):
        """
        查找非空白区域的边界
        arr: 图片 [H, W, C] 或 Mask [H, W]，数值范围 0-1
        threshold: 空白判定阈值，值越高判定越严格
        返回: (left, top, right, bottom)
        """
        if arr.ndim == 3:
            # 图片处理逻辑
            h, w, c = arr.shape
            if c >= 4:
                # 带 Alpha 通道：透明视为空白
                mask = arr[:, :, 3] > 0.01
            else:
                # RGB 图：接近白色视为空白
                mask = np.any(arr[:, :, :3] < threshold, axis=2)
        else:
            # Mask 处理逻辑：接近黑色（值为0）视为空白
            h, w = arr.shape
            mask = arr > (1 - threshold)
        
        # 全空白时返回原图边界
        if not np.any(mask):
            return 0, 0, w, h
        
        # 计算有效内容的上下左右边界
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        top = np.argmax(rows)
        bottom = h - np.argmax(rows[::-1])
        left = np.argmax(cols)
        right = w - np.argmax(cols[::-1])
        
        return left, top, right, bottom

    def trim_and_center(self, width, height, threshold, fill_color, align_with_mask, image=None, mask=None):
        if image is None and mask is None:
            raise ValueError("至少需要输入 image 或 mask 中的一个")
        
        fill_rgb = self.hex_to_rgb(fill_color)
        crop_x = crop_y = crop_w = crop_h = 0
        mask_bounds = []

        # ========== 预计算 Mask 边界（用于对齐模式） ==========
        if mask is not None and align_with_mask:
            for b in range(mask.shape[0]):
                m_np = mask[b].cpu().numpy()
                bounds = self.find_content_bounds(m_np, threshold)
                mask_bounds.append(bounds)
            # 输出裁剪参数以 Mask 为准
            crop_x, crop_y, crop_w, crop_h = mask_bounds[0][0], mask_bounds[0][1], \
                                             mask_bounds[0][2] - mask_bounds[0][0], \
                                             mask_bounds[0][3] - mask_bounds[0][1]

        # ========== 处理图片 ==========
        result_img_tensor = None
        if image is not None:
            batch_size = image.shape[0]
            result_img_list = []
            
            for b in range(batch_size):
                img_np = image[b].cpu().numpy()
                img_h, img_w, img_c = img_np.shape
                
                # 对齐模式下使用 Mask 边界，否则自行计算
                if mask_bounds and b < len(mask_bounds):
                    left, top, right, bottom = mask_bounds[b]
                else:
                    left, top, right, bottom = self.find_content_bounds(img_np, threshold)
                    # 非对齐模式下用图片边界作为输出参数
                    if not mask_bounds and b == 0:
                        crop_x, crop_y = left, top
                        crop_w = right - left
                        crop_h = bottom - top
                
                # 裁剪有效内容
                content = img_np[top:bottom, left:right, :]
                content_h, content_w = content.shape[:2]
                
                # 等比例缩放：仅缩小不放大，保持原始比例
                scale = min(width / content_w, height / content_h)
                if scale > 1.0:
                    scale = 1.0
                
                new_content_w = int(content_w * scale)
                new_content_h = int(content_h * scale)
                
                # 高质量缩放
                content_pil = Image.fromarray((content * 255).astype(np.uint8))
                content_pil = content_pil.resize((new_content_w, new_content_h), Image.LANCZOS)
                content_scaled = np.array(content_pil).astype(np.float32) / 255.0
                
                # 创建目标画布
                if img_c >= 4:
                    canvas = np.zeros((height, width, img_c), dtype=np.float32)
                    canvas[:, :, :3] = fill_rgb
                    canvas[:, :, 3] = 1.0
                else:
                    canvas = np.zeros((height, width, 3), dtype=np.float32)
                    canvas[:, :, 0] = fill_rgb[0]
                    canvas[:, :, 1] = fill_rgb[1]
                    canvas[:, :, 2] = fill_rgb[2]
                
                # 计算居中偏移
                offset_x = (width - new_content_w) // 2
                offset_y = (height - new_content_h) // 2
                
                # 粘贴内容（带 Alpha 混合）
                if img_c >= 4:
                    alpha = content_scaled[:, :, 3:4]
                    for c in range(3):
                        canvas[offset_y:offset_y+new_content_h, offset_x:offset_x+new_content_w, c] = \
                            content_scaled[:, :, c] * alpha + \
                            canvas[offset_y:offset_y+new_content_h, offset_x:offset_x+new_content_w, c] * (1 - alpha)
                    canvas[offset_y:offset_y+new_content_h, offset_x:offset_x+new_content_w, 3] = \
                        np.maximum(canvas[offset_y:offset_y+new_content_h, offset_x:offset_x+new_content_w, 3], 
                                   content_scaled[:, :, 3])
                else:
                    canvas[offset_y:offset_y+new_content_h, offset_x:offset_x+new_content_w, :] = content_scaled
                
                result_img_list.append(canvas)
            
            result_img = np.stack(result_img_list, axis=0)
            result_img_tensor = torch.from_numpy(result_img)
        else:
            # 无图片输入时返回填充色占位图
            batch_size = mask.shape[0] if mask is not None else 1
            canvas = np.zeros((height, width, 3), dtype=np.float32)
            canvas[:, :, 0] = fill_rgb[0]
            canvas[:, :, 1] = fill_rgb[1]
            canvas[:, :, 2] = fill_rgb[2]
            result_img = np.stack([canvas] * batch_size, axis=0)
            result_img_tensor = torch.from_numpy(result_img)

        # ========== 处理 Mask ==========
        result_mask_tensor = None
        if mask is not None:
            mask_batch = mask.shape[0]
            result_mask_list = []
            
            for b in range(mask_batch):
                m_np = mask[b].cpu().numpy()
                
                # 使用自身边界（对齐模式下与预计算一致）
                left, top, right, bottom = self.find_content_bounds(m_np, threshold)
                
                # 裁剪有效内容
                content = m_np[top:bottom, left:right]
                content_h, content_w = content.shape
                
                # 等比例缩放，与图片使用完全相同的缩放逻辑
                scale = min(width / content_w, height / content_h)
                if scale > 1.0:
                    scale = 1.0
                
                new_content_w = int(content_w * scale)
                new_content_h = int(content_h * scale)
                
                # 缩放 Mask（灰度模式，保持边缘平滑）
                content_pil = Image.fromarray((content * 255).astype(np.uint8), mode='L')
                content_pil = content_pil.resize((new_content_w, new_content_h), Image.LANCZOS)
                content_scaled = np.array(content_pil).astype(np.float32) / 255.0
                
                # 创建目标 Mask 画布（背景为 0，即空白/无掩码）
                canvas = np.zeros((height, width), dtype=np.float32)
                
                # 居中粘贴，偏移量与图片完全一致
                offset_x = (width - new_content_w) // 2
                offset_y = (height - new_content_h) // 2
                canvas[offset_y:offset_y+new_content_h, offset_x:offset_x+new_content_w] = content_scaled
                
                result_mask_list.append(canvas)
            
            result_mask = np.stack(result_mask_list, axis=0)
            result_mask_tensor = torch.from_numpy(result_mask)
        else:
            # 无 Mask 输入时返回全 0 占位 Mask
            batch_size = image.shape[0] if image is not None else 1
            result_mask_tensor = torch.zeros((batch_size, height, width), dtype=torch.float32)
        
        return (result_img_tensor, result_mask_tensor, crop_x, crop_y, crop_w, crop_h)



# --- 注册节点 ---
# 节点映射
NODE_CLASS_MAPPINGS = {
    "IsEmptyString": IsEmptyString,
    "SwitchString": SwitchString,
    "InpaintCut": InpaintCut,
    "InpaintConcat": InpaintConcat,
    "DoubaoChat": DoubaoChat,
    "DeepSeekChat": DeepSeekChat,
    "PriorityLLMNode": PriorityLLMNode,
    "ImageTrimAndCenter": ImageTrimAndCenter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IsEmptyString": "IsEmptyString",
    "SwitchString": "NotEmptyString",
    "InpaintCut": "Inpaint Cut",
    "InpaintConcat": "Inpaint Concat",
    "DoubaoChat": "豆包 (chat API)",
    "DeepSeekChat": "DeepSeek (chat API)",
    "PriorityLLMNode": "LLM优先级 (DeepSeek→豆包)",
    "ImageTrimAndCenter": "Trim Blank & Center",
}
