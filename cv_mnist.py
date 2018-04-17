from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
import pandas as pd
import os
import time

# Replace with a valid key
from msrest.exceptions import HttpOperationError

TRAINING_KEY = "Replace with your key"
PREDICTION_KEY = "Replace with your key"
PROJECT_NAME = "MNIST"


def main():
    trainer = training_api.TrainingApi(TRAINING_KEY)

    # Create a new project
    print("Creating project...")
    project = None
    projects = trainer.get_projects()
    existing_project = list(filter(lambda t: t.name == PROJECT_NAME, projects))
    if existing_project:
        project = existing_project[0]
    if project is None:
        project = trainer.create_project(PROJECT_NAME)
    print(project.id)

    # Make tags in the new project
    print("Creating tags...")
    tag_list = trainer.get_tags(project.id)
    if len(tag_list.tags) == 0:
        for i in range(10):
            trainer.create_tag(project.id, i)
            tag_list = trainer.get_tags(project.id)

    # Import labels
    labels = pd.read_csv('mnist/train-labels.csv', header=None)
    labels.columns = ['filename', 'label']

    # Upload tagged images
    print("Uploading tagged images...")
    tagged_images = trainer.get_tagged_images(project.id)
    if len(tagged_images) == 0:
        for index, row in labels.iterrows():
            # comment the next two rows if you want to upload the whole MNIST dataset
            if index > 1000:
                break
            with open("mnist/" + os.fsdecode(row.filename), mode="rb") as img_data:
                # Lookup tag from label
                tag = list(filter(lambda t: t.name == str(row.label), tag_list.tags))[0]
                buffer = img_data.read()
                response = trainer.create_images_from_data(project.id, buffer, [tag.id])
                print(response)

    # Train
    print("Training...")
    iteration_id = project.current_iteration_id
    try:
        iteration = trainer.train_project(project.id)
        while iteration.status == "Training":
            iteration = trainer.get_iteration(project.id, iteration.id)
            print("Training status: " + iteration.status)
            time.sleep(1)
        iteration_id = iteration.id
    except HttpOperationError as error:
        print(error.response.text)

    # The iteration is now trained. Make it the default project endpoint
    trainer.update_iteration(project.id, iteration_id, is_default=True)
    print("Done!")

    # Make a predictions
    predictor = prediction_endpoint.PredictionEndpoint(PREDICTION_KEY)
    with open("mnist/test-images/128.jpg", mode="rb") as test_data:
        results = predictor.predict_image(project.id, test_data.read(), iteration.id)

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag + ": {0:.2f}%".format(prediction.probability * 100))


if __name__ == "__main__":
    main()
