import azureml.core
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.train.dnn import PyTorch

subscription_id = "0000-00000000-00000000-0000" # The ID of the Azure Subscription
resource_group = "AdvanceAnalytics.Aml.Experiments" # Name of a logical resource group
workspace_name = "aa-ml-aml-workspace" # The name of the workspace to look for or to create
workspace_region = 'eastus' # Location of the workspace
computetarget_vm= 'Standard_NC6' # Size of the VM to use
experiment_name = 'azureml-gpubenchmark'
train_script = 'train.py'

ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

src = PyTorch(source_directory =  r'.\fastai', compute_target='amlcompute', vm_size=computetarget_vm, entry_script = train_script, use_gpu = True, pip_packages = ['fastai'])
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(src)

run.wait_for_completion(show_output = True)




