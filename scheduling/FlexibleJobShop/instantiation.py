import os
from os import pardir, walk, sep
from os.path import join, abspath, dirname, basename, normpath, relpath

from FlexibleJobShop.core import FjsSchedulingData

DIRECTORY = 'instances'
STARTPATH = abspath(join(dirname(__file__), DIRECTORY))

__all__ = ['get_instance']


def clean(inp):
    if isinstance(inp, str):
        return inp.lower().replace('.fjs', '')
    elif isinstance(inp, list):
        return [clean(el) for el in inp]
    elif isinstance(inp, tuple):
        return tuple(clean(list(inp)))
    raise TypeError("inp must be in {str, list, tuple}, not {}".format(type(inp)))
    

def list_files():
    # https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    for root, dirs, files in walk(STARTPATH):
        level = root.replace(STARTPATH, '').count(sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def get_instance(name: str, group: str = None):
    """Get a problem instance (from the ones stored in the directory 'instances/')

    Arguments:
    - group[str]: directory within 'instances/' (case-insensitive)
    - name[str]: file name within 'instances/[group]/'
        (case-insensitive, no need to add extension .fjs)

    Returns:
    - FjsSchedulingData instance
    """
    group = clean(group or "")
    name = clean(name)
    
    for root, _, files_list in walk(STARTPATH):
        dirs_list = normpath(relpath(root)).split(os.sep)
        dirs_clean = clean(dirs_list)
        files_clean = clean(files_list)
        if group.lower() in dirs_clean or group == "":
            if name.lower() in files_clean:
                file = files_list[files_clean.index(name)]
                if group != "":
                    dir_ = dirs_list[dirs_clean.index(group)]
                    path = join(STARTPATH, dir_, file)
                else:
                    path = join(STARTPATH, file)
                data = FjsSchedulingData()
                data.from_file(path)
                return data
    raise ValueError("Problem instance not found")

if __name__ == '__main__':
    mini = get_instance('mini')
    mk01 = get_instance('mk01', 'brandimarte')