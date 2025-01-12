import torch
import numpy as np

def normalize(x): 
    x = x / torch.norm(x, dim=1, p=2, keepdim=True)
    return x

class NormalDistribution:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance
        self.distribution = torch.distributions.MultivariateNormal(mean, covariance)

    def sample(self, num_samples):
        return self.distribution.sample((num_samples,))


class FeatureDistribution:
    def __init__(self):
        self.class_distributions = {}

                
    def update_distribution(self, features, labels, session, args):
        # with torch.no_grad():
        class_list = np.arange(args.base_class + session * args.way) 
        
        for cls in class_list:
            if cls not in self.class_distributions:
                idx = (labels == cls)
                if idx.sum() == 0:
                    continue
                feature = features[idx]
                mean = torch.tensor(np.mean(feature, axis=0), dtype=torch.float32).cuda()
                variance = torch.tensor(np.var(feature, axis=0), dtype=torch.float32).cuda()
                epsilon = 1e-4
                variance = (torch.diag(variance) + epsilon * torch.eye(variance.size(0)).cuda()).float()
                self.class_distributions[cls] = NormalDistribution(mean, variance)
                            

    def sample_from_class(self, class_label, num_samples=1):
        if class_label in self.class_distributions.keys():
            return class_label,self.class_distributions[class_label].sample(num_samples)
        else:
            raise ValueError(f"No distribution found for class {class_label}")

    def sample_groups(self, num_classes, num_samples_per_class=5, num_groups=4):
        all_samples = []
        all_labels = []

        # 采样四组，每组的标签顺序相同
        for _ in range(num_groups):
            group_samples = []
            group_labels = []
            for class_label in range(num_classes):
                try:
                    label, samples = self.sample_from_class(class_label, num_samples_per_class)
                    group_samples.append(samples)
                    group_labels.extend([label] * num_samples_per_class)  # 每个标签重复num_samples_per_class次
                except ValueError as e:
                    print(e)

            all_samples.append(torch.cat(group_samples, dim=0))  # 每组的样本拼接起来
            all_labels.append(torch.tensor(group_labels))

        return all_samples, all_labels


        # if not isinstance(class_labels, list):
        #     raise ValueError("class_labels must be a list of labels")

        # samples = []
        # for label in class_labels:
        #     if label in self.class_distributions:
        #         samples.append(self.class_distributions[label].sample(num_samples))
        #     else:
        #         raise ValueError(f"No distribution found for class {label}")
        # split_size=num_samples//2
        # # 返回一个包含每个类标签样本的列表
        # f=[]
        # for i in range(2):
        #     f.append(torch.cat([feature[split_size*i:split_size*(i+1)] for feature in samples]))
        # class_labels = torch.tensor([element for element in class_labels for _ in range(split_size//2)])
        # #打乱
        # group_size = 2
        # indices = torch.randperm(class_labels.shape[0])
        # class_labels=class_labels[indices]
        # for i in range(2):
        #     split_features = f[i].view(-1, group_size, f[i].size(1))  # 变成形状为 [num_groups, 2, D]

        #     # Step 2: 打乱组的顺序
        #     shuffled_features = split_features[indices]      # 按组打乱

        #     # Step 3: 恢复打乱后的特征形状
        #     f[i] = shuffled_features.view(-1, f[i].size(1))  # 恢复为 [N, D]
        # return class_labels, f[0],f[1]


# # 使用示例
# calculator = DistributionCalculator()

# # 假设你在循环中接收每个类的数据
# for images, labels in dataloader:  # 这里你可以将 dataloader 移到外面
#     calculator.update_distribution(images, labels)

# # 从某个类中采样
# class_label = 0  # 替换为你想采样的类标签
# num_samples = 5
# samples = calculator.sample_from_class(class_label, num_samples)

if __name__=="__main__":
    A=FeatureDistribution()
    features=torch.rand(10,4)
    labels=torch.zeros(10)
    A.update_distribution(features,labels)
    features_1=torch.rand(10,4)
    labels_1=torch.ones(10)
    A.update_distribution(features_1,labels_1)

    class_labels,f,samples=A.sample_from_class(list(range(0,2)),num_samples=20)
    print(class_labels,f,samples)
    print(class_labels.shape,f.shape,samples.shape)