#!/bin/sh

# build-swig-Python.sh


debug_flag=$1

if [ -n "$debug_flag" -a "$debug_flag" == "-debug" ]
then
    Debug=1
else
    Debug=0
fi


HEADER_FILES="${SRCROOT}/include/lldb/lldb-types.h"\
" ${SRCROOT}/include/lldb/API/SBAddress.h"\
" ${SRCROOT}/include/lldb/API/SBBlock.h"\
" ${SRCROOT}/include/lldb/API/SBBreakpoint.h"\
" ${SRCROOT}/include/lldb/API/SBBreakpointLocation.h"\
" ${SRCROOT}/include/lldb/API/SBBroadcaster.h"\
" ${SRCROOT}/include/lldb/API/SBCommandContext.h"\
" ${SRCROOT}/include/lldb/API/SBCommandInterpreter.h"\
" ${SRCROOT}/include/lldb/API/SBCommandReturnObject.h"\
" ${SRCROOT}/include/lldb/API/SBCompileUnit.h"\
" ${SRCROOT}/include/lldb/API/SBDebugger.h"\
" ${SRCROOT}/include/lldb/API/SBError.h"\
" ${SRCROOT}/include/lldb/API/SBEvent.h"\
" ${SRCROOT}/include/lldb/API/SBFrame.h"\
" ${SRCROOT}/include/lldb/API/SBFunction.h"\
" ${SRCROOT}/include/lldb/API/SBLineEntry.h"\
" ${SRCROOT}/include/lldb/API/SBListener.h"\
" ${SRCROOT}/include/lldb/API/SBModule.h"\
" ${SRCROOT}/include/lldb/API/SBProcess.h"\
" ${SRCROOT}/include/lldb/API/SBSourceManager.h"\
" ${SRCROOT}/include/lldb/API/SBStringList.h"\
" ${SRCROOT}/include/lldb/API/SBSymbol.h"\
" ${SRCROOT}/include/lldb/API/SBSymbolContext.h"\
" ${SRCROOT}/include/lldb/API/SBTarget.h"\
" ${SRCROOT}/include/lldb/API/SBThread.h"\
" ${SRCROOT}/include/lldb/API/SBType.h"\
" ${SRCROOT}/include/lldb/API/SBValue.h"


if [ $Debug == 1 ]
then
    echo "Header files are:"
    echo ${HEADER_FILES}
fi

NeedToUpdate=0

swig_output_file=${SCRIPT_INPUT_FILE_1}

if [ ! -f $swig_output_file ]
then
    NeedToUpdate=1
    if [ $Debug == 1 ]
    then
        echo "Failed to find LLDBWrapPython.cpp"
    fi
fi

if [ $NeedToUpdate == 0 ]
then
    for hdrfile in ${HEADER_FILES}
    do
        if [ $hdrfile -nt $swig_output_file ]
        then
            NeedToUpdate=1
            if [ $Debug == 1 ]
            then
                echo "${hdrfile} is newer than ${swig_output_file}"
                echo "swig file will need to be re-built."
            fi
        fi
    done
fi

framework_python_dir="${CONFIGURATION_BUILD_DIR}/LLDB.framework/Versions/A/Resources/Python"

if [ ! -L "${framework_python_dir}/_lldb.so" ]
then
    NeedToUpdate=1
fi

if [ ! -f "${framework_python_dir}/lldb.py" ]
then
    NeedToUpdate=1
fi


if [ $NeedToUpdate == 0 ]
then
    echo "Everything is up-to-date."
    exit 0
else
    echo "SWIG needs to be re-run."
fi


# Build the SWIG C++ wrapper file for Python.

swig -c++ -shadow -python -I"${SRCROOT}/include" -I./. -outdir "${CONFIGURATION_BUILD_DIR}" -o "${SCRIPT_INPUT_FILE_1}" "${SCRIPT_INPUT_FILE_0}"

# Edit the C++ wrapper file that SWIG generated for Python.  There are two
# global string replacements needed, which the following script file takes
# care of.  It reads in 'LLDBWrapPython.cpp' and generates 
# 'LLDBWrapPython.cpp.edited'.

# The need for this has been eliminated by fixing the namespace qualifiers on return types.
# Leaving this here for now, just in case...
#
#if [ -f "${SRCROOT}/scripts/Python/edit-swig-python-wrapper-file.py" ]
#then
#    python "${SRCROOT}/scripts/Python/edit-swig-python-wrapper-file.py"
#fi

#
# Now that we've got a C++ file we're happy with (hopefully), rename the
# edited file and move it to the appropriate places.
#

if [ -f "${SCRIPT_INPUT_FILE_1}.edited" ]
then
    mv "${SCRIPT_INPUT_FILE_1}.edited" "${SCRIPT_INPUT_FILE_1}"
fi
