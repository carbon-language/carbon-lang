# clangify2010.py - Created by Clark Gaebel [ Free as in speech. ]
#
# Python 3.x, so don't you dare 2.6 this!
#
# This script is used to generate the proper clang call from a VC/VC++ 10
# file. As an example of this, a project with files [foo.c, bar.c, baz.c] will
# generate the command "clang --analyze foo.c bar.c baz.c" This includes C++
# support so if we had [foo.cpp, bar.cpp, baz.cpp], it will generate the command
# "clang++ --analyze foo.cpp bar.cpp baz.cpp.
import sys
import os
from xml.dom import minidom


###### CUSTOMIZATION #######

def pre_analysis():
    '''
        Put any pre-analysis tasks in here. They will be performed before
        enumeration of the project file and the actual clangification begins.
    '''
    return

def post_anlalysis():
    '''
        Put any post-analysis tasks in here, such as cleaning up from your
        pre-analysis. This will be called after the actual clangification.
    '''
    return

##### END CUSTOMIZATION #####

def die(message):
    print("ERROR: " + message)
    exit()

# returns a list of files to clang (Oh em gee I just verbed clang).
# To support other project file types, just implement a function with
# the same signature and the rest is trivial.
def parse_vs2010_project_file(filename):
    output_list = list()

    file_contents = minidom.parse(filename)
    elements = file_contents.getElementsByTagName('ClCompile')

    for current_element in elements:
        if current_element.hasAttribute('Include'):
            output_list.append(current_element.attributes['Include'].value)

    return output_list

# returns "c" for "foo.c"
def get_file_extension(filename):
    extension = str()

    for char in reversed(filename):
        if char == '.':
            break;
        else:
            extension = char + extension
            
    return extension

# returns the homogenous file extension if successful, "" otherwise.
def file_extensions_are_homogenous(list_of_files):
    if len(list_of_files) < 1:
        return ""
    
    extension = get_file_extension(list_of_files[0])
    
    for current_file in list_of_files:
        if get_file_extension(current_file) != extension:
            return ""

    return extension

def is_in_list(lst, elem):
    try:
        lst.index(elem)
    except ValueError:
        return False
    return True

# fixes a list of files such as [foo.c, bar.c, baz.c]
# so that they are relative to a path.
# if this function is called as
# "fix_paths("./a/b/c.q", ['foo.c', 'bar.c', 'baz.c'])",
# it will return ['./a/b/foo.c', './a/b/bar.c', './a/b/baz.c']
def fix_paths(base_filename, pathless_files):
    fixed_paths = list()
    for i in pathless_files:
        fixed_paths.append(os.path.dirname(base_filename) + '/' + i)
    return fixed_paths

###### MAIN ######

pre_analysis()

# Handle the "I don't know how to use this thing" case.
if len(sys.argv) != 2:
    print(
       """clangify.py

        Usage: python clangify.py [location of .vcxproj file]

        This will call clang's analysis engine on all of your C/C++ files in
        the project. Please ensure that clang and clang++ are in your system
        PATH. For now, this only works for VS10 project files.""")
    project_file = ""
else:
    project_file = sys.argv[1]
    
files_to_clang = list()
if get_file_extension(project_file) == "vcxproj":
    files_to_clang = parse_vs2010_project_file(project_file)
else:
    die("Project file type not supported. clangify only works for VS2010.")
    
    
file_extension = file_extensions_are_homogenous(files_to_clang)

clang_command = str()

# feel free to add more extension/language support here.
c_extensions = ['c']
cpp_extensions = ['cpp', 'cxx', 'cc']

if is_in_list(c_extensions, file_extension):
    clang_command = 'clang --analyze'
elif is_in_list(cpp_extensions, file_extension):
    clang_command = 'clang++ --analyze -ccc-clang-cxx'
elif file_extension == '':
    die(
        "The project's file extensions are not homogenous. Are you mixing"
        ".c and .cpp files in the same project?")
else:
    die(
        "Unrecognized file extension. clangify/clang only support C and"
        "C++ projects.")

files_to_clang = fix_paths(project_file, files_to_clang)
    
for current_file in files_to_clang:
    clang_command += ' ' + current_file
    
os.system(clang_command)

post_analysis()
