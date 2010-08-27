#!/bin/sh

# build-swig-Python.sh

# SRC_ROOT is the root of the lldb source tree.
# TARGET_DIR is where the lldb framework/shared library gets put.
# CONFIG_BUILD_DIR is where the build-swig-Python-LLDB.sh  shell script 
#           put the lldb.py file it was generated from running SWIG.
# PREFIX is the root directory used to determine where third-party modules
#         for scripting languages should be installed.
# debug_flag (optional) determines whether or not this script outputs 
#           additional information when running.

SRC_ROOT=$1
TARGET_DIR=$2
CONFIG_BUILD_DIR=$3
PYTHON_INSTALL_DIR=$4
debug_flag=$5 

swig_output_file=${SRC_ROOT}/source/LLDBWrapPython.cpp
swig_input_file=${SRC_ROOT}/scripts/lldb.swig


if [ -n "$debug_flag" -a "$debug_flag" == "-debug" ]
then
    Debug=1
else
    Debug=0
fi


HEADER_FILES="${SRC_ROOT}/include/lldb/lldb-types.h"\
" ${SRC_ROOT}/include/lldb/API/SBAddress.h"\
" ${SRC_ROOT}/include/lldb/API/SBBlock.h"\
" ${SRC_ROOT}/include/lldb/API/SBBreakpoint.h"\
" ${SRC_ROOT}/include/lldb/API/SBBreakpointLocation.h"\
" ${SRC_ROOT}/include/lldb/API/SBBroadcaster.h"\
" ${SRC_ROOT}/include/lldb/API/SBCommandContext.h"\
" ${SRC_ROOT}/include/lldb/API/SBCommandInterpreter.h"\
" ${SRC_ROOT}/include/lldb/API/SBCommandReturnObject.h"\
" ${SRC_ROOT}/include/lldb/API/SBCompileUnit.h"\
" ${SRC_ROOT}/include/lldb/API/SBDebugger.h"\
" ${SRC_ROOT}/include/lldb/API/SBError.h"\
" ${SRC_ROOT}/include/lldb/API/SBEvent.h"\
" ${SRC_ROOT}/include/lldb/API/SBFileSpec.h"\
" ${SRC_ROOT}/include/lldb/API/SBFrame.h"\
" ${SRC_ROOT}/include/lldb/API/SBFunction.h"\
" ${SRC_ROOT}/include/lldb/API/SBLineEntry.h"\
" ${SRC_ROOT}/include/lldb/API/SBListener.h"\
" ${SRC_ROOT}/include/lldb/API/SBModule.h"\
" ${SRC_ROOT}/include/lldb/API/SBProcess.h"\
" ${SRC_ROOT}/include/lldb/API/SBSourceManager.h"\
" ${SRC_ROOT}/include/lldb/API/SBStringList.h"\
" ${SRC_ROOT}/include/lldb/API/SBSymbol.h"\
" ${SRC_ROOT}/include/lldb/API/SBSymbolContext.h"\
" ${SRC_ROOT}/include/lldb/API/SBTarget.h"\
" ${SRC_ROOT}/include/lldb/API/SBThread.h"\
" ${SRC_ROOT}/include/lldb/API/SBType.h"\
" ${SRC_ROOT}/include/lldb/API/SBValue.h"


if [ $Debug == 1 ]
then
    echo "Header files are:"
    echo ${HEADER_FILES}
fi

NeedToUpdate=0


if [ ! -f ${swig_output_file} ]
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
        if [ $hdrfile -nt ${swig_output_file} ]
        then
            NeedToUpdate=1
            if [ $Debug == 1 ]
            then
                echo "${hdrfile} is newer than ${swig_output_file}"
                echo "swig file will need to be re-built."
            fi
            break
        fi
    done
fi

if [ $NeedToUpdate == 0 ]
then
    if [ ${swig_input_file} -nt ${swig_output_file} ]
    then
        NeedToUpdate=1
        if [ $Debug == 1 ]
        then
            echo "${swig_input_file} is newer than ${swig_output_file}"
            echo "swig file will need to be re-built."
        fi
    fi
fi

os_name=`uname -s`
python_version=`/usr/bin/python --version 2>&1 | sed -e 's,Python ,,' -e 's,[.][0-9],,2' -e 's,[a-z][a-z][0-9],,'`

if [ "$os_name" == "Darwin" ]
then
    framework_python_dir="${TARGET_DIR}/LLDB.framework/Resources/Python"
else
    framework_python_dir="${PYTHON_INSTALL_DIR}/python${python_version}"
fi


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

swig -c++ -shadow -python -I"${SRC_ROOT}/include" -I./. -outdir "${CONFIG_BUILD_DIR}" -o "${swig_output_file}" "${swig_input_file}"

# Append global variable to lldb Python module.

current_dir=`pwd`
if [ -f "${current_dir}/append-debugger-id.py" ]
then
    python ${current_dir}/append-debugger-id.py ${CONFIG_BUILD_DIR}
fi
