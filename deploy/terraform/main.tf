terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Key Pair for SSH access
resource "tls_private_key" "pk" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "kp" {
  key_name   = "swarmnet-deploy-key"
  public_key = tls_private_key.pk.public_key_openssh
}

# Save private key locally
resource "local_file" "ssh_key" {
  filename = "${path.module}/swarmnet-key.pem"
  content  = tls_private_key.pk.private_key_pem
  file_permission = "0400"
}

# Security Group allowing HTTP and UDP DHT traffic
resource "aws_security_group" "swarmnet_sg" {
  name        = "swarmnet_security_group"
  description = "Allow Swarmnet P2P traffic"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Miner HTTP API"
    from_port   = 8000
    to_port     = 8005
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Miner Kademlia DHT (UDP)"
    from_port   = 8460
    to_port     = 8480
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Find latest Ubuntu 22.04 AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# EC2 Instance
resource "aws_instance" "swarmnet_miner" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro" # Free tier eligible, cheap for testing
  key_name      = aws_key_pair.kp.key_name
  
  vpc_security_group_ids = [aws_security_group.swarmnet_sg.id]

  # Automatically install docker via cloud-init
  user_data = file("${path.module}/../cloud_init.yaml")

  tags = {
    Name = "Swarmnet-Cloud-Miner"
  }
}

output "miner_public_ip" {
  value       = aws_instance.swarmnet_miner.public_ip
  description = "The public IP address of the Swarmnet Cloud Miner"
}

output "ssh_command" {
  value = "ssh -i swarmnet-key.pem ubuntu@${aws_instance.swarmnet_miner.public_ip}"
}
