import subprocess

subprocess.run([
    'streamlit', 'run', '/home/deepread2.0/web/run_web.py', 
    '--server.address', '0.0.0.0', 
    '--server.port', '8080',
    # '--theme.base', 'dark',
    '--server.fileWatcherType', 'none',
    '--browser.gatherUsageStats', 'false'
    # '--server.headless', 'true'
])
