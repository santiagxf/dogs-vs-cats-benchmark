from azureml.core import Workspace
workspace_name = "aa-ml-aml-workspace"
subscription_id = "18522758-626e-4d88-92ac-dc9c7a5c26d4"
resource_group = "AdvanceAnalytics.Aml.Experiments"
ws = Workspace(workspace_name = workspace_name, subscription_id = subscription_id, resource_group=resource_group)

from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("fastai")
myenv.add_conda_package("pytorch")
myenv.add_conda_package("torchvision")
myenv.add_channel("pytorch")
myenv.add_channel("fastai")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())


from azureml.core.image import ContainerImage
# Image configuration
image_config = ContainerImage.image_configuration(execution_script = "score.py", runtime = "python",
                                                 conda_file = "myenv.yml",
                                                 enable_gpu = True,
                                                 description = "Image classficiation service cats vs dogs",
                                                 tags = {"data": "cats-vs-dogs", "type": "classification"})

from azureml.core.model import Model

model = Model.register(model_path = "export.pkl",
                       model_name = "cats_vs_dogs",
                       tags = {"key": "0.1"},
                       description = "cats_vs_dogs",
                       workspace = ws)


# Register the image from the image configuration
image = ContainerImage.create(name = "azureml-gpubenchmark-fastai", 
                              models = [model], #this is the model object
                              image_config = image_config,
                              workspace = ws)

image.wait_for_creation(show_output=True)

from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 2, 
                                               memory_gb = 4, 
                                               tags = {"data": "cats-vs-dogs", "type": "classification"}, 
                                               description = 'Image classficiation service cats vs dogs')


from azureml.core.webservice import Webservice

service_name = 'azureml-gpubenchmark'
service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                            image = image,
                                            name = service_name,
                                            workspace = ws)
service.wait_for_deployment(show_output = True)
print(service.state)