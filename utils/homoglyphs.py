# utils/homoglyphs.py

def get_homoglyph_map():
    """
    构建视觉相似字符映射数据库 (Homoglyph Database)。
    主要利用西里尔字母(Cyrillic)替换拉丁字母(Latin)。
    这些字符在某些字体下肉眼几乎无法分辨，但编码不同。
    """
    return {
        # Latin : Cyrillic (Visual Lookalikes)
        'a': 'а',  # U+0061 : U+0430
        'c': 'с',  # U+0063 : U+0441
        'e': 'е',  # U+0065 : U+0435
        'o': 'о',  # U+006F : U+043E
        'p': 'р',  # U+0070 : U+0440
        'x': 'х',  # U+0078 : U+0445
        'y': 'у',  # U+0079 : U+0443
        'A': 'А',
        'B': 'В',
        'C': 'С',
        'E': 'Е',
        'H': 'Н',
        'K': 'К',
        'M': 'М',
        'O': 'О',
        'P': 'Р',
        'T': 'Т',
        'X': 'Х'
    }