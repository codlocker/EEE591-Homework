Steps for the setting up project 4 via scp

- Create a folder named **Proj 4** in the machine in which you have logged in
- In the **Proj 4** folder scp the following files 'cmoslibrary.lib', 'project4.py', 'utils.py', 'InvChainTemplate.sp'. All the files should be in the same Proj 4 folder and in the same subtree.
    - scp project4.py utils.py cmoslibrary.lib InvChainTemplate.sp <target_remote_path>/Proj_4
- Run project4.py as follows
    
    ```python3 project4.py```
- After execution is complete, you can delete the Proj_4 folder using 
    ```rm -r Proj_4/```