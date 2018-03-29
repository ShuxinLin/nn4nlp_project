To run GPU on AWS:

1. Launch EC2 instance with Deep Learning AMI (Ubuntu). The latest version at this writing is Version 6.0 - ami-bc09d9c1.

2. Choose instance type p2.xlarge. (May need to request limit increase.) Note that there is the spot instance option. Configure security group and key pairs, then launch.

3. After launching, in shell, first export the AWS key information:
```
export AWS_ACCESS_KEY_ID=<your key id>
export AWS_SECRET_ACCESS_KEY=<your key>
```
Then ssh into the instance by
```
ssh -i <path to key pair .pem file> ubuntu@<Public DNS (IPv4) of the instance>
```

4. Git clone this repo into the instance.

5. To use PyTorch, activate
```
source activate pytorch_p36
```

6. Also pip install matplotlib
```
python -mpip install -U pip
python -mpip install -U matplotlib
```

7. Now it's ready to run experiments using GPU.
```
python run.py
```

8. To see GPU utilization, use
```
nvidia-smi
```
