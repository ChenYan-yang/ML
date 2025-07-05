### 目录结构

```plaintext
code/
├── main.py               # 主程序入口
├── model.py              # 主要模型构建
├── preprocess.py         # 数据预处理
├── utils.py              # 工具函数

Data/
├── original_data.zip/    # 原始数据
├── daily_test/       # 处理好的 test 数据
└── daily_train/      # 处理好的 train 数据

Result/               # 运行结果
```

### 运行脚本

```bash
cd code
python main.py
```
