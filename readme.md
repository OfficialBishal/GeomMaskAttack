## Read.Me
- 'line.py' and 'spot.py' contains code to generate adversarial samples with HLine and Spot respectively.

### Dataset
- Download the dataset from this [link](https://www.kaggle.com/datasets/officialbishal/nepals-embossed-license-plate-character-set) or [link](www.kaggle.com/dataset/38dcaa3c94bc09e5e785599d8b83bf4e30154a02bde618457b2f470b2a80b346).
- Put the dataset inside the 'images/Segments_Sorted/' directory.

### Train LPR model
- run 'ALPR.ipynb'
- Configure 'data_path', and 'save_path' as necessary.

### To run the project
- run 'main.py' 
- Configure 'mode', 'model_path', 'op_path, and 'output_path' as necessary.
- The 'model_path' must be directed to the model intended to use. 

### To test the dataset on model
- run 'testing.py'

### Adversarial Training
- run 'AdversarialTraining.ipynb'
- Configure 'save_path' as necessary.

### To generate heatmap
- run 'heatmap.py'
- Configure 'model_path' and 'output_path' as necessary.

### To create gif
- run 'gif_creation.py'
- Consifure 'output_path'
- Copy and paste code below at appropriate location in spot.py or line.py of which gif you want to create
```python
pert_img = create_spot(img.copy(), center_i, radius, rgb)
pert_image = numpy_PIL_tensor(pert_img)
#Saving the perturbed image
output_path = 'outputs/gif3/'
output_file_name = f"{radius}-{center_i}.png"
os.makedirs(output_path, exist_ok=True)
pert_image = pert_image.view(3, 160, 105)
save_image(pert_image, output_path + output_file_name)
```
- Use the generated image to create gif from [here](https://ezgif.com/maker).

### Illustration of Adversarial Attack
![](https://github.com/OfficialBishal/Adversarial-Image-Generation/blob/master/outputs/gif/hline.gif)
![](https://github.com/OfficialBishal/Adversarial-Image-Generation/blob/master/outputs/gif/vline.gif)
![](https://github.com/OfficialBishal/Adversarial-Image-Generation/blob/master/outputs/gif/spot.gif)