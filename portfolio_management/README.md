## Folder Structure  

.  
├ agents ~~~~~~~~~~~~~~~~~~~~~ # Definition of agent classes, loss, and ensemble loss functions  
│ ├── agent_loss ~~~~~~~~~~~~~~~~ # Loss and ensemble loss functions  
│ ├── capm_agent ~~~~~~~~~~~~~~~ # Capital Asset Pricing Model  
│ ├── dpm_agent ~~~~~~~~~~~~~~~~ # Deep Portfolio Model  
│ ├── mvp_agent ~~~~~~~~~~~~~~~~ # Minimum Variance Portfolio  
│ └── uniform_agent ~~~~~~~~~~~~ # Uniform Weights Portfolio  
├ data ~~~~~~~~~~~~~~~~~~~~~~~ # Data collection and storage  
│ ├── archive_data ~~~~~~~~~~~~~~ # Stored data  
│ └ get_raw_data ~~~~~~~~~~~~~~~~ # Classes for data collection  
│  └── configs ~~~~~~~~~~~~~~~~~~~~ # configuration files  
├ trading_env ~~~~~~~~~~~~~~~~ # Reinforcement Learning Environment  
├ agent_comparison ~~~~~~~~~~~ # Post training agent evaluation  
├ dpm_agent_training ~~~~~~~~~~ # Training for Deep Portfolio Model  
├ test_training_synth_data ~~~~~~ # Test training validity on simple synthetic data   
└ README.md  
