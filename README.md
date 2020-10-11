# gearbox fault detection

## 20200930

1. 异常点剔除
   1. 工作模式
   2. 60s平均风速，应大于切入风速，小于切出风速
   3. 60s平均有功功率，应大于0
   4. 发电机转速，发电机转速应大与0
2. 工况辨识
3. 特征
   1. 降维
   2. 不受工况影响的部分特征权值上调
4. 模型
   1. 模型选择与融合（pyod）
   2. 模型超参数优化（遗传算法、粒子群、网格搜索）
5. 阈值
   1. 指数加权移动平均
   2. 滑动窗口，2020.2
   3. 极值理论自适应阈值，2017.4，F分布，可以尝试吧

### TODO

3. 再确定一下，不划分的现有效果

## 20200929

1. 犁牛坪SCADA处理
2. 尝试pyod里其他模型
3. 建立评价指标，以故障前多少天以内的数据为异常数据。2周？单机单模型下，好像不合适。

## 20200928

1. 牛家岭、犁牛坪哪几个风机存在误报警？
   1. 犁牛坪：2，8，24
   2. 牛家岭：2，7，8
2. 故障机组：
   1. 犁牛坪：3
   2. 牛家岭：11，14
3. 转速分档建模 -> 软聚类工况辨识
4. 转速分档，效果不行？为什么？

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
      4. 60s平均有功功率
2. ~~阶次跟踪，需考虑时间复杂度，计算效率~~
   1. ~~Vold-Kalman，放弃~~
   2. 考虑风速分区，分档
3. 特征集构建，特征提取，特征选择，特征降维
   1. 看看海装的CMS指标和自己提取的有没有差异，不一致!
   2. ~~额外提一些特征~~
4. 模型融合
5. 阈值调整

### 牛家岭

1000-1250

1250-1500

1500以上

### 犁牛坪

需要处理SCADA发电机转速数据