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
		cp libsnowflake.so /usr/local/lib
		apt-get update
		apt-get -y install python3-pip
		pip3 install --upgrade numpy
		# Protobuf
		if hash protoc 2>/dev/null
		then
			echo "protobuf compiler is installed, skipping protobuf installation"
		else
		        echo "Installing protobuf from source"
			cd /tmp
			wget https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
			tar xf protobuf-all-3.6.1.tar.gz
			cd protobuf-3.6.1
			./configure
			make -j4
			make install
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
	 	cp libsnowflake.so /usr/local/lib
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
		echo "Installing protobuf from source"
		cd /tmp
		curl -LO https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
		tar xf protobuf-all-3.6.1.tar.gz
		cd protobuf-3.6.1
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
