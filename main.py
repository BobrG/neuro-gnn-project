from nni.experiment import Experiment, RemoteMachineConfig

if __name__=='__main__':
    experiment = Experiment('remote')
    
    search_space = {
                    'hidden_dim': {'_type': 'choice', '_value': [4, 16, 32, 64]},
                    'n_GNN_layers': {'_type': 'choice', '_value': [2, 4, 8]},
                    'n_MLP_layers': {'_type': 'choice', '_value': [1, 2, 4, 8]},
                    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
                    'momentum': {'_type': 'uniform', '_value': [0, 1]},
                    }
    
    experiment.config.trial_command = 'python3 main_explainer.py --train --explain --dataset_path=/home/src/datasets/ --dataset_name=Schiza'
    experiment.config.trial_code_directory = './IBGNN_modified/'
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.search_space = search_space
    experiment.config.max_experiment_duration = '1h'
    experiment.config.trial_concurrency = 2
    print(experiment.config)
    experiment.config.training_service.machine_list.append(RemoteMachineConfig(ip='10.5.1.1', 
                                                          user_name='g.bobrovskih', 
                                                          preCommand='SINGULARITY_SHELL=/bin/bash singularity shell --nv --bind /trinity/home/g.bobrovskih/ongoing/neuro-gnn-project/:/home/src/ \
                                                                /trinity/home/g.bobrovskih/ongoing/neuro-gnn-project/gbobrovskikh.gnn_neuro-2022-11-23-869d9e3a0d90.sif'))
    
    experiment.run(50123)
    
    experiment.stop()
    input()
    