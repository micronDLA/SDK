#!/bin/bash
if [ `which lsb_release 2>/dev/null` ] && [ `lsb_release -i | cut -f2` == 'Ubuntu' ]
then
	echo "In order to install this package, you have to accept the EULA"
	echo "Please use PgDown and PgUp to read the EULA, then press q"
	read -p "Now please press ENTER " -r
	less EULA
	read -p "Do you accept this EULA (y/n)? " -r
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
		apt-get update
		apt-get -y install python3-pip
		apt-get -y install git
		if [ `lsb_release --release | cut -f2 | cut -d. -f1` -lt '16' ]
		then
			cp libsnowflake-gcc4.8.so /usr/local/lib
			ln -s libsnowflake-gcc4.8.so libsnowflake.so
			# Protobuf
			echo "Installing protobuf from source"
			cd /tmp
			wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz
			tar xf protobuf-all-3.5.1.tar.gz
			cd protobuf-3.5.1
			./configure
			make
			make install

			# Pytorch
			apt-get -y install zlib1g-dev
			apt-get -y install libjpeg-dev
			apt-get -y install libyaml-dev
			pip3 install pyyaml
			pip3 install --upgrade numpy
			pip3 install Pillow
			cd /tmp
			wget https://cmake.org/files/v3.10/cmake-3.10.2-Linux-x86_64.tar.gz
			tar xf cmake-3.10.2-Linux-x86_64.tar.gz
			cp -r cmake-3.10.2-Linux-x86_64/* /usr/local/
			git clone --recursive https://github.com/pytorch/pytorch
			cd pytorch
			python3 setup.py install
		else
			cp libsnowflake-gcc5.4.so /usr/local/lib
			ln -s libsnowflake-gcc5.4.so libsnowflake.so
			PYVERSION=`python3 --version|cut -d. -f2`
			if [ $PYVERSION == 6 ]
			then
				pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
				pip3 install --upgrade torchvision
			elif [ $PYVERSION == 5 ]
			then
				pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
				pip3 install --upgrade torchvision
			else
				echo 'Unknown python version, torch will not be installed'
			fi
		fi
		echo "Installing thnets from source"
		cd /tmp
		git clone https://github.com/mvitez/thnets
		cd thnets
		make ONNX=1
		make install
		ldconfig
		echo 'Installation finished'
	fi
elif [ -f /etc/redhat-release ] && [ `cut -d ' ' -f1 /etc/redhat-release` == 'CentOS' ] && \
	[ `cut -d ' ' -f4 /etc/redhat-release | cut -d . -f1 -` == '7' ]
then
	echo "In order to install this package, you have to accept the EULA"
	echo "Please use PgDown and PgUp to read the EULA, then press q"
	read -p "Now please press ENTER " -r
	less EULA
	read -p "Do you accept this EULA (y/n)? " -r
	if [[ $REPLY =~ ^[Yy]$ ]]
	then
		cp libsnowflake-gcc4.8.so /usr/local/lib
		ln -s libsnowflake-gcc4.8.so libsnowflake.so
		yum install unzip
		yum install yum-utils
		yum-builddep python
		echo "Installing Python3 from source"
		cd /tmp
		curl -LO https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tar.xz
		tar xf Python-3.6.5.tar.xz
		cd Python-3.6.5
		./configure
		make
		make install
		pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
		pip3 install --upgrade torchvision
		echo "Installing protobuf from source"
		cd /tmp
		curl -LO https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz
		tar xf protobuf-all-3.5.1.tar.gz
		cd protobuf-3.5.1
		./configure
		make
		make install
		echo "Installing thnets from source"
		cd /tmp
		curl -LO https://github.com/mvitez/thnets/archive/master.zip
		unzip master.zip
		cd thnets-master
		make ONNX=1
		make install
		echo /usr/local/lib >/etc/ld.so.conf.d/local.conf
		ldconfig
		echo 'Installation finished'
	fi
else
	echo 'This installer works only for Ubuntu and CentOS 7.x, sorry'
	exit
fi
