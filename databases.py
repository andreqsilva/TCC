import os

def bug(dir):
    subdirs_path = os.path.join(dir, "bug2017_stainnorm_validation_1000px")
    subdirs = os.listdir(subdirs_path)
    targets = []
    sources = []
    for subdir in subdirs:
        image_path = os.path.join(subdirs_path, subdir)
        files = os.listdir(image_path)
        targets.extend(os.path.join(image_path, files[2]) for i in range(len(files) - 1))
        del files[2]
        sources.extend([os.path.join(image_path, file) for file in files.copy()])
    return targets, sources        

def displasia(dir):
    subdirs_path = os.path.join(dir, "ROIs_no_pre_processing")
    subdirs = os.listdir(subdirs_path)
    sources = []
    for subdir in subdirs:
        image_path = os.path.join(subdirs_path, subdir)
        files = os.listdir(image_path)
        sources.extend([os.path.join(image_path, file) for file in files])
    return sources

def MITOS(dir):
    target_path = os.path.join(dir, "cortes 256\\Scanner Aperio")
    target_files = os.listdir(target_path)
    targets = [os.path.join(target_path, file) for file in target_files if file.startswith("A")]

    source_path = os.path.join(dir, "cortes 256\\Scanner Hamamatsu")
    source_files = os.listdir(source_path)
    sources = [os.path.join(source_path, file) for file in source_files if file.startswith("H")]
    return targets, sources