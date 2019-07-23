# AIlias
## Doodle Recogniser and Classifier

This is a project designed to recognise and classify doodles from camera input. You can read more about it [here]().

### To start this classifier you need to:
* Create new python enviroment. Python's version must be 3.53
* Install all required dependecies: `pip install -r requirements.txt`
* If you are going to train the model yourself, you need to run script that downloads the data to train on. Simply go to `data/` and run `sh get_data.sh`
* If you just want to use my trained model, you have to download it to main directory: `wget https://www.dropbox.com/s/713cunyz7vufvej/doodle_model.h5`
* After the set up just run: `python predict.py`