#!/bin/bash
accept_eula=''
skip_lib_copy=''

print_usage() {
  printf "Usage: $0 [ -a Accept EULA ] [ -s skip library copy ]"
}

while getopts 'as' flag; do
  case "${flag}" in
    a) accept_eula='y' ;;
    s) skip_lib_copy='y' ;;
    *) print_usage
       exit 1 ;;
  esac
done

read_eula() {
	if [[ -z $accept_eula ]];
	then
		echo "In order to install this package, you have to accept the EULA"
		echo "Please use PgDown and PgUp to read the EULA, then press q"
		read -p "Now please press ENTER " -r
		less EULA
		read -p "Do you accept this EULA (y/n)? " -r
		accept_eula=$REPLY		
	fi
}

if [ `which lsb_release 2>/dev/null` ] && [ `lsb_release -i | cut -f2` == 'Ubuntu' ]
then
	read_eula
	if [[ $accept_eula =~ ^[Yy]$ ]]
	then
		if [[ -z "$skip_lib_copy" ]]; then cp libmicrondla.so /usr/local/lib; fi
		apt-get update
		apt-get -y install python3-pip
		pip3 install --upgrade numpy
		# Protobuf
		ldconfig -p | grep libprotobuf.so.17 >/dev/null
		if [ $? -eq 0 ]; then
			echo "protobuf is installed, skipping protobuf installation"
		else
		        echo "Installing protobuf 3.6.1 from source"
			cd /tmp
			wget https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
			tar xf protobuf-all-3.6.1.tar.gz
			cd protobuf-3.6.1
			./configure
			make -j4
			make install
		fi
		ldconfig
		echo 'Installation finished'
	fi
elif [ -f /etc/redhat-release ] && [ `cut -d ' ' -f1 /etc/redhat-release` == 'CentOS' ] && \
	[ `cut -d ' ' -f4 /etc/redhat-release | cut -d . -f1 -` == '7' ]
then
	read_eula
	if [[ $accept_eula =~ ^[Yy]$ ]]
	then
		if [ -f /etc/ld.so.conf.d/local.conf ]; then
			if ! grep -q /usr/local/bin /etc/ld.so.conf.d/local.conf; then
				echo /usr/local/lib >>/etc/ld.so.conf.d/local.conf
			fi
		else
			echo /usr/local/lib >/etc/ld.so.conf.d/local.conf
		fi
		if [[ -z "$skip_lib_copy" ]]; then cp libmicrondla.so /usr/local/lib; fi
		yum install unzip
		if ! [ -x "$(command -v python3)" ]; then
			echo "Installing Python3 from source"
			yum install yum-utils
			yum-builddep python
			cd /tmp
			curl -LO https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tar.xz
			tar xf Python-3.6.5.tar.xz
			cd Python-3.6.5
			./configure
			make
			make install
		fi
		ldconfig -p | grep libprotobuf.so.17 >/dev/null
		if [ $? -eq 0 ]; then
			echo "protobuf is installed, skipping protobuf installation"
		else
			echo "Installing protobuf 3.6.1 from source"
			cd /tmp
			curl -LO https://github.com/google/protobuf/releases/download/v3.6.1/protobuf-all-3.6.1.tar.gz
			tar xf protobuf-all-3.6.1.tar.gz
			cd protobuf-3.6.1
			./configure
			make
			make install
			ldconfig
		fi
		ldconfig
		echo 'Installation finished'
	fi
else
	echo 'This installer works only for Ubuntu and CentOS 7.x, sorry'
	exit
fi
