from utils import *
from tqdm import tqdm
import torch.nn.functional as F

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    cw_acc = Averager()
    base_cw = Averager()
    novel_cw = Averager()
    
    all_targets=[]
    all_probs=[]

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            logits, _ = model(data)
            logits = logits[:, :test_class]

            all_targets.append(test_label)
            all_probs.append(logits)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

    # Concatenate all_targets and probs
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs, axis=0)

    # Compute class wise accuracy
    for l in all_targets.unique():
        # Get class l mask
        class_mask = all_targets == l
        pred = torch.argmax(all_probs, dim=1)[class_mask]
        label_ = all_targets[class_mask]
        class_acc = (pred == label_).type(torch.cuda.FloatTensor).mean().item()
        cw_acc.add(class_acc)
        
        if l < args.base_class:
            base_cw.add(class_acc)
        else:
            novel_cw.add(class_acc)
    
    # Compute va using class-wise accuracy
    pred = torch.argmax(all_probs, dim=1)
    va = class_acc = (pred == all_targets).type(torch.cuda.FloatTensor).mean().item()
    
    return va, novel_cw.item(), base_cw.item()

def get_base_prototypes(trainset, transform, model, args, mode="encoder"):
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    og_mode = model.module.mode
    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        tqdm_gen.set_description("Generating Features: ")
        model.module.mode = mode
        for i, batch in enumerate(tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            
            embedding = model(data)
            
            embedding_list.append(embedding)
            label_list.append(label)

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]

        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    prototypes = torch.stack(proto_list, dim=0)

    model.module.mode = og_mode
    
    return prototypes

def get_base_prototypes_v2(trainloader, model, args):
    model = model.eval()

    with torch.no_grad():
        embedding_list = []
        label_list = []
        tqdm_gen = tqdm(trainloader)
        tqdm_gen.set_description("Generating Features: ")
        for i, batch in enumerate(tqdm_gen, 1):
            # data, label = [_.cuda() for _ in batch]
            data, label = batch
            
            data = torch.cat((data[0], data[1]), dim=0).cuda()
            label = label.cuda()
            embedding = model.module.encode(data, aug=args.base_aug, base=True)
            
            label = label.repeat(2 * args.pretrain_num_aug)
                        
            embedding_list.append(embedding)
            label_list.append(label)

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]

        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    prototypes = torch.stack(proto_list, dim=0)

    
    return prototypes

def get_base_prototypes_ardataset(trainset, transform, model, args, mode="encoder"):
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    og_mode = model.module.mode
    with torch.no_grad():
        tqdm_gen = tqdm(trainloader)
        tqdm_gen.set_description("Generating Features: ")
        model.module.mode = mode
        for i, batch in enumerate(tqdm_gen, 1):
            # data, label = [_.cuda() for _ in batch]
            inputs, label = batch
            for key in inputs[0]:
                inputs[0][key] = inputs[0][key].cuda()
            label = label.cuda()
            
            embedding = model(inputs[0])

            embedding_list.append(embedding)
            label_list.append(label)

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]

        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    prototypes = torch.stack(proto_list, dim=0)

    model.module.mode = og_mode
    
    return prototypes


def store_prototype_v2(dataloader, model, session=0, args=None, mode="prototype"):
     
    model = model.eval()
    
    feature_list = []
    label_list = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Calculate Prototype")):
            images, label = batch
            #这里为两次次性加载，batchsize=250*2
            images = torch.cat([images[0], images[1]], dim=0).cuda()
            label=label.cuda()
            #如果batchsize过大，可将数据分成多块，依次经过encode计算
            features = model.module.encode(images, aug=True, base=True, store=True)
            label = label.repeat(2 * args.pretrain_num_aug)
            # features = self.encode(images, store=True)
            feature_list.append(features.data.cpu().numpy())
            label_list.append(label.data.cpu().numpy())

    feature_list = np.concatenate(feature_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    model.module.calculator.update_distribution_v2(feature_list, label_list, session=session, args=args)

def test_v2(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    cw_acc = Averager()
    base_cw = Averager()
    novel_cw = Averager()
    session_cw = Averager()
    
    all_targets=[]
    all_probs=[]
    session_acc = []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            # inputs, test_label = batch
            # for key in inputs[0]:
            #     inputs[0][key] = inputs[0][key].cuda()
            
            test_label = test_label.cuda()
            
            logits, _ = model(data)
            logits = logits[:, :test_class]

            all_targets.append(test_label)
            all_probs.append(logits)
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()

    # Concatenate all_targets and probs
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs, axis=0)

    # Compute class wise accuracy
    s = 0
    for l in all_targets.unique():
        # Get class l mask
        class_mask = all_targets == l
        pred = torch.argmax(all_probs, dim=1)[class_mask]
        label_ = all_targets[class_mask]
        class_acc = (pred == label_).type(torch.cuda.FloatTensor).mean().item()
        cw_acc.add(class_acc)
               
        if l < args.base_class:
            base_cw.add(class_acc)
        else:
            s += 1
            session_cw.add(class_acc)
            novel_cw.add(class_acc)
            if s == args.way:
                session_acc.append(session_cw.item())
                session_cw = Averager()
                s = 0

    
    # Compute va using class-wise accuracy
    pred = torch.argmax(all_probs, dim=1)
    va = class_acc = (pred == all_targets).type(torch.cuda.FloatTensor).mean().item()
    
    return va, novel_cw.item(), base_cw.item(), session_acc

