import os
import random

# 定义凯撒密码加密和解密函数
def caesar_cipher(data, shift):
    result = bytearray()
    for b in data:
        # 处理所有字节的加密和解密
        result.append((b + shift) % 256)  # 允许所有字节值
    return result

def encrypt_file(file_path):
    with open(file_path, 'rb') as file:
        # 读取文件内容
        file_data = file.read()
        # 随机生成位移值
        shift = random.randint(1, 255)  # 随机生成位移值
        # 加密文件内容
        encrypted_data = caesar_cipher(file_data, shift)  # 加密
        # 保存为 .bin 文件
        with open(file_path.replace('.raw', '.bin'), 'wb') as encrypted_file:
            encrypted_file.write(encrypted_data)
    # 删除原始 .raw 文件
    os.remove(file_path)
    # 保存位移值
    with open(file_path.replace('.raw', '.shift'), 'w') as shift_file:
        shift_file.write(str(shift))

def decrypt_file(file_path):
    with open(file_path, 'rb') as file:
        # 读取加密文件内容
        encrypted_data = file.read()
    # 读取位移值
    with open(file_path.replace('.bin', '.shift'), 'r') as shift_file:
        shift = int(shift_file.read())
    # 解密文件内容
    decrypted_data = caesar_cipher(encrypted_data, -shift)  # 解密
    # 保存为 .raw 文件
    with open(file_path.replace('.bin', '.raw'), 'wb') as decrypted_file:
        decrypted_file.write(decrypted_data)
    # 删除 .bin 和 .shift 文件
    os.remove(file_path)
    os.remove(file_path.replace('.bin', '.shift'))

def process_files():
    for filename in os.listdir():
        if filename.endswith('.raw'):
            encrypt_file(filename)
        elif filename.endswith('.bin'):
            decrypt_file(filename)

if __name__ == '__main__':
    process_files()
