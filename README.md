OPEL: Optimal Transport Guided ProcedurE Learning

Codebase:

1. Dataset Organisation
Please download the following datasets and annotations from their respective websites.

Follow the following directory structure:

--Datasets
	--annotations
		--dataset_name
			--subtasks(if present)
				-- *.csv
	--dataset_name_1
		--subtasks(if present)
			-- *.avi
	--dataset_name_2
		--subtasks(if present)
			-- *.mp4
	.
	.
	.
	.
	--dataset_name_n
		--subtasks(if present)
			-- *.mp4
			
	--Frames (generated when running main_otuprel_exp.py for the first time)
		-- individual dataset
			--subtasks(if present)
				-- *.h5
				
2. Training the model:

Execute the following python code as follows:

a) All base configurations and hyperparameters are defined in './configs/config.py'
It contains a detailed documentation of all the variables used in this repository.

b) Modify demo_config.yaml for individual datasets and update the file paths. This will overwrite existing paths in ./configs/config.py.

c) Once the paths are updated and correct, run the following command to train:

python -m main_otuprel_exp --cfg configs/demo_config.yaml

The trained models will be saved in LOG.DIR (./logs).


3. Testing the Trained Network:
 
Generating the embeddings and clustering the frames

Once the network is trained, it can be tested using the following command:

Execute the command:
python -m procedure_learning_eval --cfg configs/demo_config.yaml TCC.MODEL_PATH path/to/the/model.pth

The embeddings will be saved in ./embeds
The results will be will be saved in LOG.DIR (./logs).


The code will be published post acceptance.



