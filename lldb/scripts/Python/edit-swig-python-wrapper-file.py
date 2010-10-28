#
# edit-swig-python-wrapper-file.py
#
# This script performs some post-processing editing on the C++ file that
# SWIG generates for python, after running on 'lldb.swig'.   In
# particular, on Apple systems we want to include the Python.h file that
# is used in the /System/Library/Frameworks/Python.framework, but on other
# systems we want to include plain <Python.h>.  So we need to replace:
#
# #include <Python.h>
#
# with:
#
# #if defined (__APPLE__)
# #include <Python/Python.h>
# #else
# #include <Python.h>
# #endif
#
# That's what this python script does.
#


import os

input_dir_name = os.environ["SRCROOT"]
full_input_name = input_dir_name + "/source/LLDBWrapPython.cpp"
full_output_name = full_input_name + ".edited"

try:
    f_in = open (full_input_name, 'r')
except IOError:
    print "Error:  Unable to open file for reading: " + full_input_name
else:
    try:
        f_out = open (full_output_name, 'w')
    except IOError:
        print "Error:  Unable to open file for writing: " + full_output_name
    else:
        include_line_found = False

        try:
            line = f_in.readline()
        except IOError:
            print "Error occurred while reading file."
        else:
            while line:
                #
                #
                if not include_line_found:
                    if (line.find ("#include <Python.h>") == 0):
                        f_out.write ("#if defined (__APPLE__)\n");
                        f_out.write ("#include <Python/Python.h>\n");
                        f_out.write ("#else\n");
                        f_out.write (line);
                        f_out.write ("#endif\n");
                        include_line_found = True
                    else:
                        f_out.write (line)
                else:
                    f_out.write (line)
                try:
                    line = f_in.readline()
                except IOError:
                    print "Error occurred while reading file."
                    
            try:
                f_in.close()
                f_out.close()
            except:
                print "Error occurred while closing files"
