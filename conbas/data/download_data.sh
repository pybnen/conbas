ARCHIVE_NAME=data.zip
DATA_DIR=../../data/

# download data.zip
DATASET_URL="https://docs.google.com/uc?export=download&id=1nke45xyN8DERa__No-J_MBolfXLMekY5"
wget --no-check-certificate $DATASET_URL -O $ARCHIVE_NAME

# unzip archive 
mkdir $DATA_DIR
unzip $ARCHIVE_NAME -d $DATA_DIR

# clean up
rm $ARCHIVE_NAME