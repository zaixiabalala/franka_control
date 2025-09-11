from setuptools import setup, find_packages

basics = [
    'numpy<2.0.0', 
    'opencv-python>=4.10.0', 
    'open3d>=0.18.0', 
    'matplotlib', 
    'scipy', 
    'psutil', 
    'yourdfpy', 
    'pynput', 
    'tqdm', 
]
extras = {
    'comm': [
        'pysoem', # needed by ftsensor flexiv pyft
        'pyserial', # needed by encoder pdcd angle and ftsensor robotiq ft300
        'pymodbus', # needed by ftsensor robotiq ft300
    ], 
    'rs': [
        'pyrealsense2==2.53.1.4623', # needed by camera realsense
    ], 
    'xr': [
        'pyopenxr', # needed by camera vive
    ], 
    'franka': [
        'frankx', # needed by robot franka
    ], 
}
extras['all'] = list(set({pkg for pkgs in extras.values() for pkg in pkgs}))

setup(
    name = 'r3kit', 
    version = '0.0.1', 
    license = 'MIT', 
    description = 'Research kits for real robots', 
    author = "Junbo Wang", 
    author_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    maintainer = "Junbo Wang", 
    maintainer_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    url = "https://github.com/dadadadawjb/r3kit", 
    packages = find_packages(), 
    include_package_data = True, 
    install_requires = basics, 
    extras_require = extras, 
    zip_safe = False
)
