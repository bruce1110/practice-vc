"""
中文字体设置模块
解决 matplotlib 中文显示乱码问题
"""

import matplotlib.pyplot as plt
import platform


def setup_chinese_font():
    """设置 matplotlib 中文字体"""
    system = platform.system()
    
    # 根据操作系统选择合适的中文字体
    if system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Heiti SC', 'STSong']
    elif system == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'SimSun']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    # 获取系统可用字体列表
    try:
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]
    except:
        available_fonts = []
    
    # 尝试设置字体
    for font_name in chinese_fonts:
        if font_name in available_fonts or len(available_fonts) == 0:
            try:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 已设置中文字体: {font_name}")
                return True
            except:
                continue
    
    # 如果所有字体都不可用
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠ 未能自动找到中文字体，中文可能显示为方框")
    return False


def list_chinese_fonts():
    """列出系统中可用的中文字体"""
    try:
        from matplotlib.font_manager import FontManager
        
        fm = FontManager()
        all_fonts = [f.name for f in fm.ttflist]
        
        # 常见中文字体关键词
        chinese_keywords = ['SC', 'CN', 'Chinese', 'Hei', 'Song', 'Kai', 'Fang', 
                           'YaHei', 'SimHei', 'SimSun', 'PingFang', 'STHeiti', 
                           'WenQuanYi', 'Noto', 'Droid']
        
        chinese_fonts = []
        for font in sorted(set(all_fonts)):
            for keyword in chinese_keywords:
                if keyword in font:
                    chinese_fonts.append(font)
                    break
        
        if chinese_fonts:
            print("可用的中文字体：")
            for i, font in enumerate(chinese_fonts[:20], 1):
                print(f"  {i}. {font}")
            if len(chinese_fonts) > 20:
                print(f"  ... 还有 {len(chinese_fonts) - 20} 个字体")
        else:
            print("未找到中文字体")
        
        return chinese_fonts
    except Exception as e:
        print(f"无法列出字体: {e}")
        return []

