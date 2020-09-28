# gearbox fault detection

## 20200927

1. 数据，只要采样频率低，采样时间长的
   1. 犁牛坪，128K，256K，CMS没有风速，SCADA发电机转速与齿轮箱转速之间的关系，待确认，采样频率可能有问题（答，发电机转速与齿轮箱转速成比例，采样频率正确）
      1. 256K: 5120Hz
      2. 128K: 51200Hz
   2. 牛家岭，采样频率，RPM，提取转速信号，**应该去掉不转的部分**，**需要重新提取特征**
      1. 12800，低速轴
      2. 6400，齿圈
      3. 25600，高速轴
   3. ~~王四营子，就10天的数据~~
   4. ~~三塘湖，有大量18年5月之前的数据，5月之后只有三个月的数据~~
   5. SCADA
      1. 发电机转速，正值
      2. 60s平均风速，3以上
      3. 工作模式，32发电运行
2. 阶次跟踪，需考虑时间复杂度，计算效率
   1. Vold-Kalman，放弃
   2. 考虑风速分区，分档
3. 特征集构建，特征提取，特征选择
   1. 看看海装的CMS指标和自己提取的有没有差异，不一致!
   2. 额外提一些特征
4. 模型融合
5. 阈值调整

### 牛家岭

1000-1250

1250-1500

1500以上

### 犁牛坪

需要处理SCADA发电机转速数据

## 20200928

1. 牛家岭、犁牛坪哪几个风机存在误报警？
2. 转速分档建模 -> 软聚类工况辨识
3. 转速分档，效果不行？为什么？
4. 