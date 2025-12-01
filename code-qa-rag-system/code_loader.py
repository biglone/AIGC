"""
代码加载器

从本地目录加载代码文件，支持多种编程语言
"""

import os
from pathlib import Path
from typing import List, Dict
import fnmatch
from config import SUPPORTED_EXTENSIONS, IGNORE_DIRS, IGNORE_PATTERNS, logger


class CodeLoader:
    """
    代码文件加载器

    功能：
    1. 递归扫描目录
    2. 过滤无关文件
    3. 读取代码内容
    4. 提取元数据
    """

    def __init__(self):
        """初始化"""
        self.supported_extensions = SUPPORTED_EXTENSIONS
        self.ignore_dirs = IGNORE_DIRS
        self.ignore_patterns = IGNORE_PATTERNS

    def load_from_directory(self, directory: str) -> List[Dict]:
        """
        从目录加载所有代码文件

        Args:
            directory: 代码库根目录

        Returns:
            代码文件列表，每个元素包含：
            {
                'content': str,      # 文件内容
                'metadata': {
                    'source': str,   # 文件路径
                    'filename': str, # 文件名
                    'extension': str,# 扩展名
                    'size': int,     # 文件大小（字节）
                    'lines': int     # 行数
                }
            }
        """
        if not os.path.exists(directory):
            raise ValueError(f"目录不存在: {directory}")

        logger.info(f"开始扫描目录: {directory}")

        documents = []
        directory_path = Path(directory)

        # 递归遍历所有文件
        for file_path in self._scan_files(directory_path):
            try:
                # 读取文件内容
                content = self._read_file(file_path)

                if not content.strip():
                    logger.debug(f"跳过空文件: {file_path}")
                    continue

                # 提取元数据
                metadata = self._extract_metadata(file_path, directory_path)

                documents.append({
                    'content': content,
                    'metadata': metadata
                })

                logger.debug(f"已加载: {metadata['source']} ({metadata['lines']} 行)")

            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")
                continue

        logger.info(f"扫描完成，共加载 {len(documents)} 个文件")
        return documents

    def _scan_files(self, directory: Path):
        """
        递归扫描目录，生成符合条件的文件路径
        """
        for root, dirs, files in os.walk(directory):
            # 过滤忽略的目录
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            for file in files:
                file_path = Path(root) / file

                # 检查扩展名
                if file_path.suffix not in self.supported_extensions:
                    continue

                # 检查忽略模式
                if self._should_ignore(file):
                    continue

                yield file_path

    def _should_ignore(self, filename: str) -> bool:
        """
        检查文件是否应该被忽略
        """
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _read_file(self, file_path: Path) -> str:
        """
        读取文件内容（自动检测编码）
        """
        # 尝试常见编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue

        # 如果都失败，使用二进制模式读取并忽略错误
        logger.warning(f"无法解码文件 {file_path}，使用替换模式")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def _extract_metadata(self, file_path: Path, base_path: Path) -> Dict:
        """
        提取文件元数据
        """
        # 获取相对路径
        try:
            relative_path = file_path.relative_to(base_path)
        except ValueError:
            relative_path = file_path

        # 统计行数
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = sum(1 for _ in f)
        except:
            lines = 0

        return {
            'source': str(relative_path),
            'filename': file_path.name,
            'extension': file_path.suffix,
            'size': file_path.stat().st_size,
            'lines': lines,
            'language': self._detect_language(file_path.suffix)
        }

    def _detect_language(self, extension: str) -> str:
        """
        根据扩展名检测编程语言
        """
        language_map = {
            '.py': 'Python',
            '.cpp': 'C++',
            '.cc': 'C++',
            '.cxx': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.java': 'Java',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript (React)',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript (React)',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.rb': 'Ruby',
            '.php': 'PHP'
        }
        return language_map.get(extension, 'Unknown')

    def load_single_file(self, file_path: str) -> Dict:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            文件文档
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if path.suffix not in self.supported_extensions:
            raise ValueError(f"不支持的文件类型: {path.suffix}")

        content = self._read_file(path)
        metadata = self._extract_metadata(path, path.parent)

        return {
            'content': content,
            'metadata': metadata
        }

    def get_file_stats(self, directory: str) -> Dict:
        """
        获取目录的统计信息

        Returns:
            统计信息字典
        """
        directory_path = Path(directory)

        if not directory_path.exists():
            raise ValueError(f"目录不存在: {directory}")

        stats = {
            'total_files': 0,
            'total_lines': 0,
            'total_size': 0,
            'languages': {},
            'extensions': {}
        }

        for file_path in self._scan_files(directory_path):
            stats['total_files'] += 1
            stats['total_size'] += file_path.stat().st_size

            # 统计语言
            language = self._detect_language(file_path.suffix)
            stats['languages'][language] = stats['languages'].get(language, 0) + 1

            # 统计扩展名
            ext = file_path.suffix
            stats['extensions'][ext] = stats['extensions'].get(ext, 0) + 1

            # 统计行数
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    stats['total_lines'] += sum(1 for _ in f)
            except:
                pass

        return stats


# ============================================================
# 测试代码
# ============================================================

def test_code_loader():
    """
    测试代码加载器
    """
    print("=" * 70)
    print("测试代码加载器")
    print("=" * 70)

    loader = CodeLoader()

    # 测试1：加载示例项目
    print("\n[测试1] 加载示例项目")
    try:
        # 使用当前目录作为测试
        test_dir = "."
        documents = loader.load_from_directory(test_dir)

        print(f"✅ 成功加载 {len(documents)} 个文件")

        if documents:
            # 显示第一个文件
            print(f"\n示例文件:")
            doc = documents[0]
            print(f"  文件: {doc['metadata']['source']}")
            print(f"  语言: {doc['metadata']['language']}")
            print(f"  行数: {doc['metadata']['lines']}")
            print(f"  大小: {doc['metadata']['size']} 字节")
            print(f"  内容预览: {doc['content'][:200]}...")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # 测试2：获取统计信息
    print(f"\n[测试2] 获取统计信息")
    try:
        stats = loader.get_file_stats(test_dir)
        print(f"✅ 统计信息:")
        print(f"  总文件数: {stats['total_files']}")
        print(f"  总行数: {stats['total_lines']:,}")
        print(f"  总大小: {stats['total_size']:,} 字节")
        print(f"  语言分布: {stats['languages']}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print()


if __name__ == "__main__":
    test_code_loader()
