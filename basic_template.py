# @author : Vijaya Kumar Thanelanka
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from models.resnet import ResNet18
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def main():
	parser = argparse.ArgumentParser(description='This is for introducing argparse')
	parser.add_argument('--lr', default=0.03, type=float, help='this takes learning as input')
	parser.add_argument('--epochs', default=100, type=int, help='this takes no.of epochs')

	parser.add_argument('--dataset', default='CIFAR10',type=str, choices=['MNIST', 'CIFAR10', 'CIFAR100'])
	parser.add_argument('--batch_size', default=128, type=int, help = 'batch_size')
	args = parser.parse_args()
	

	# --------------------------------------------------------------------------------------------------------
	transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
									transforms.RandomVerticalFlip(),
									transforms.ToTensor()])
	test_transform = transforms.Compose([transforms.ToTensor()])

	if args.dataset == 'MNIST' :
		num_classes = 10
		train_dataset = torchvision.datasets.MNIST('data', train=True, transform = transform, download = True)
		test_dataset = torchvision.datasets.MNIST('data', train=False, transform = test_transform, download = True)
 
	if args.dataset == 'CIFAR10':
		num_classes = 10
		train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform = transform, download = True)
		test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform = test_transform, download = True)
	if args.dataset == 'CIFAR100':
		num_classes = 100
		train_dataset = torchvision.datasets.CIFAR100('data', train=True, transform = transform, download = True)
		test_dataset = torchvision.datasets.CIFAR100('data', train=False, transform = test_transform, download = True)
	# ----------------------------------------------------------------------------------------------------
	
	train_data_loader = DataLoader(dataset = train_dataset,
								batch_size = args.batch_size,
								shuffle = True,
								num_workers = 2,
								)

	test_data_loader = DataLoader(dataset = test_dataset,
								batch_size = args.batch_size,
								)


	# ----------------------------------------------------------------------------------------------------
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	
	model = ResNet18(num_classes)
	model = model.to(device)
	model.train()
	# parm = [{'parms': model.parameter(), 'lr': 1e-5}, {'parms':model2.parameters(), 'lr':1e-4}]
	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	schedular = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
	
	for epoch in range(args.epochs):
		total_correct = 0
		toal_samples = 0
		for idx, (inp_data, gt) in tqdm(enumerate(train_data_loader)):
			inp_data, gt  = inp_data.to(device) , gt.to(device)
			pred = model(inp_data)	
			
			loss = F.cross_entropy(pred, gt)
			
			loss.backward()  
			optimizer.zero_grad()
			optimizer.step()

			sub_correct = sum(torch.argmax(pred, axis=1) == gt)

			total_correct += sub_correct
			toal_samples += pred.shape[0]
			schedular.step()
		print('Final Accuracy:', total_correct/toal_samples)

if __name__ == '__main__':
	main()
 
