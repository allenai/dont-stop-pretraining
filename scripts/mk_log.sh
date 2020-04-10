log=$1
dest=$2
mkdir -p $dest
rm $log/model.tar.gz
rm $log/best.th
cp -r $log $dest