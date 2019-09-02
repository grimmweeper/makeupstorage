================================================================
Steps to train Tensorflow model and run Computer Vision
================================================================

================To prepare images for training==================
1)Rename photos and put them into folders, train_set and test_set respectively
2)Label them in pascal/voc format and save it as xml files using labelimg
3)Run xml_to_csv.py in spyder3 to create test_set.csv and train_set.csv (Change the code accordingly)
4)Open Anaconda Prompt and change directory to "cd C:\Users\drant\.spyder-py3\DIP\train_set"

===========Generate tensorflow records from xml files===========
5)"python generate_tfrecord.py --csv_input=train_labels.csv  --output_path=train.record"
6)"python generate_tfrecord.py --csv_input=test_labels.csv  --output_path=test.record"

=====Prepare label map and training pipelines for training======
7)Create label map: Go to "research/object_detection/training" & change object-detection.pbtxt
8)Create training configuration: Go to "research/object_detection/training" & change faster rcnn..inception..v2..pets.config accordingly (num classes, training/test directories, activation, steps,num_examples etc)
9)Clear the files in the folder, leave only model.ckpt stuff, faster_rcnn.config, object_detection.pbtxt, pipeline.config

=================To train your model============================
10)Change directory: cd C:\Users\drant\models\research\object_detection
11)python model_main.py --alsologtostderr --model_dir=training --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config --num_train_steps=20000 --sample_1_of_n_eval_examples=1

=================Logging data onto tensorboard==================
12)tensorboard --logdir=training

====================Export inference graph======================
13) Choose the one with the largest steps
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-495 --output_directory training

====================Run OpenCV using jupyter notebook===========
14)jupyter notebook object_detection_tutorial.ipynb
15)Change the code accordingly

