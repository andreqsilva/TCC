import os

def BUG(dir):
    targets = []
    sources = []
    for currentdir, subdirs, files in os.walk(dir):
        nfiles = len(files)
        if nfiles != 0:
            for file in files:
                if file.endswith("03.tif"):
                    repeated_files = [os.path.join(currentdir, file)] * 8
                    targets.extend(repeated_files)
                else:
                    sources.append(os.path.join(currentdir, file))
    return targets, sources 

def DISPLASIA(dir, target):
    sources = []
    for currentdir, subdirs, files in os.walk(dir):
        sources.extend([os.path.join(currentdir, file) for file in files])
    targets = [target] * len(sources)
    return targets, sources

def MITOS(dir):
    for currentdir, subdirs, files in os.walk(dir):
        if currentdir.endswith("Aperio"):
            targets = [os.path.join(currentdir, file) for file in files if file.startswith("A")]
        elif currentdir.endswith("Hamamatsu"):
            sources = [os.path.join(currentdir, file) for file in files if file.startswith("H")]
    return targets, sources       

def BREAKHIST(dir, magnification, target):
    sources = []
    magnification = str(magnification) + "X"
    for currentdir, subdirs, files in os.walk(dir):
        if currentdir.endswith(magnification):
            sources.extend([os.path.join(currentdir, file) for file in files])
    targets = [target] * len(sources)
    return targets, sources

def UNITOPATHO(dir, target):
    sources = []
    for currentdir, subdir, files in os.walk(dir):
        sources.extend([os.path.join(currentdir, file) for file in files])
    targets = [target] * len(sources)
    return targets, sources