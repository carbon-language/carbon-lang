#
# edit-swig-python-wrapper-file.py
#
# This script performs some post-processing editing on the C++ file that
# SWIG generates for python, after running on 'lldb.swig'.   In
# particular, the types SWIGTYPE_p_SBThread and SWIGTYPE_p_SBTarget are
# being used, when the types that *should* be used are 
# SWIGTYPE_p_lldb__SBThread and SWIGTYPE_p_lldb__SBTarget.
# This script goes through the C++ file SWIG generated, reading it in line
# by line and doing a search-and-replace for these strings.
#


import os

full_input_name = os.environ["SCRIPT_INPUT_FILE_1"];
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
        target_typedef_found = False
        thread_typedef_found = False

        try:
            line = f_in.readline()
        except IOError:
            print "Error occurred while reading file."
        else:
            while line:
                #
                #
                if (line.find ("SWIGTYPE_p_SBTarget")):
                    if (line.find ("define") < 0):
                        line = line.replace ("SWIGTYPE_p_SBTarget", 
                                             "SWIGTYPE_p_lldb__SBTarget")
                if (line.find ("SWIGTYPE_p_SBThread")):
                    if (line.find ("define") < 0):
                        line = line.replace ("SWIGTYPE_p_SBThread", 
                                             "SWIGTYPE_p_lldb__SBThread")
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
