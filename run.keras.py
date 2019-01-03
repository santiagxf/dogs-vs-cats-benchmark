import azureml.core
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies

subscription_id = "MASKED"
resource_group = "AdvanceAnalytics.ML"
workspace_name = "aa-ml-aml-workspace"
workspace_region = 'eastus'
computetarget_vm= 'Standard_NC6'

ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

AmlCompute.supported_vmsizes(ws)

# Create a new runconfig object
run_config = RunConfiguration()
run_config.target = "amlcompute"
run_config.amlcompute.vm_size = computetarget_vm
run_config.framework = 'python'
run_config.environment.docker.base_image = 'tensorflow/tensorflow:1.6.0-gpu'
run_config.environment.docker.gpu_support = True
run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['opencv==3.4.1','urllib3', 'tqdm', 'scikit-learn', 'pandas', 'tensorflow-gpu', 'keras-gpu'])

working_dir = r'.\keras'
src = ScriptRunConfig(source_directory = working_dir, script = 'train.py', run_config = run_config)
experiment = Experiment(workspace=ws, name="azureml-benchmark")
run = experiment.submit(src)

run.wait_for_completion(show_output = True)


from azureml.widgets import RunDetails
RunDetails(run).show()

