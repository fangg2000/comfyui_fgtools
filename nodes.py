import os, sys, json, uuid
import torch
import numpy as np
import numpy as np

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

# 节点映射
NODE_CLASS_MAPPINGS = {
    "IsEmptyString": IsEmptyString,
    "SwitchString": SwitchString,
    "InpaintCut": InpaintCut,
    "InpaintConcat": InpaintConcat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IsEmptyString": "IsEmptyString",
    "SwitchString": "NotEmptyString",
    "InpaintCut": "Inpaint Cut",
    "InpaintConcat": "Inpaint Concat",
}