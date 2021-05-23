from git import Repo
import os
import json


def intro_repository(config, mode="Train"):

 # Get git info to put in the config_*.txt file
    repo = Repo(os.getcwd())
    repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]
    branch_name = repo.head.reference
    commit = branch_name.commit
    commit_small = str(commit)[0:8]

    config_str = f"""
      -------------- Repository information --------------
      Repo name : {repo_name}
      Branch name : {branch_name}
      Commit : {commit} ({commit_small})
      ----------------------------------------------------
      Environment Related Config
      {json.dumps(config, indent=8, default=str)}
      {mode} Related Config
      """
    return config_str
