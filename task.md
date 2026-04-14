任务: 基于 Qwen2.5-VL:3B 微调一个判别式的眼底病识别模型
- 输入为眼底彩照(CFP) + 光学相干断层扫描(OCT)
- 输出：分类结果(closed-set)

技术方案
- 输入：1张CFP + 1~3张OCT图像
- 输出：闭集分类结果 y∈{1,…,K}

模型结构
1. 视觉编码器
  - Qwen2.5-VL自带的Vision Encoder
2. 语言模型
  - 用 Qwen2.5 LM Decoder 作为文本侧与跨模态建模模块
  - 利用其特征加工能力
3. 分类头
  - 最终接一个线性分类头 / MLP 分类头
  - 输出 closed-set 类别概率

image_root: /home/sqw/VisualSearch/mm_linyan/imgdata
