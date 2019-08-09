## Realization of ASTPN Network

文件结构：

1. Dataset：数据集

2. DataReader：数据处理

3. Layers：ASTPN网络的4层结构，CNN，SPP，RNN，ATP，以及组合成的ASTPN网络

4. models：保存下来的模型

5. RelatedPaper：相关论文

运行：

1. 从网上下载数据集
   
   ```
   ./Dataset/setup.sh
   ```

2. 训练并保存模型
   
   ```
   python train.py
   ```
