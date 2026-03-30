"""
Feature Hook模块
用于提取中间层特征（构建轨迹的关键）
"""

class FeatureHook:
    def __init__(self, model):
        """
        自动注册hook到模型的中间层

        设计思想：
        - 捕获多层feature
        - 构建trajectory
        """
        self.features = []   # 存储每一层输出
        self.handles = []

        # 遍历所有模块
        for name, module in model.named_modules():
            # 选择ResNet的layer层（避免conv级别过细）
            if "layer" in name and "conv" not in name:
                handle = module.register_forward_hook(self._hook_fn)
                self.handles.append(handle)

    def _hook_fn(self, module, input, output):
        """
        hook函数：自动在forward时调用

        输入:
            output: [B, C, H, W] 或 [B, D]
        输出:
            转为 [B, D] 并保存
        """
        # 如果是卷积特征，做全局平均池化
        if len(output.shape) == 4:
            output = output.mean(dim=[2, 3])  # GAP

        # 存储特征
        self.features.append(output)

    def clear(self):
        """每次forward前清空"""
        self.features = []

    def close(self):
        """移除hook"""
        for h in self.handles:
            h.remove()