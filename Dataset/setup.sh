./clean.sh
mkdir i-LIDS-VID
cd i-LIDS-VID
sudo apt-get install wget
wget -c http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar
tar -xvf iLIDS-VID.tar
rm iLIDS-VID.tar
cd ..

