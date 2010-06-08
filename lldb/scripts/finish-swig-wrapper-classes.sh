#! /bin/sh

# finish-swig-wrapper-classes.sh
#
# For each scripting language liblldb supports, we need to create the
# appropriate Script Bridge wrapper classes for that language so that 
# users can call Script Bridge functions from within the script interpreter.
# 
# We use SWIG to create a C++ file containing the appropriate wrapper classes
# and funcitons for each scripting language, before liblldb is built (thus
# the C++ file can be compiled into liblldb.  In some cases, additional work
# may need to be done after liblldb has been compiled, to make the scripting
# language stuff fully functional.  Any such post-processing is handled through
# the shell scripts called here.

debug_flag=$1

if [ -n "$debug_flag" -a "$debug_flag" == "-debug" ]
then
    Debug=1
else
    Debug=0
fi


#
# For each scripting language, see if a post-processing script for that 
# language exists, and if so, call it.
#
# For now the only language we support is Python, but we expect this to
# change.

languages="Python"
cwd=${SRCROOT}/scripts

for curlang in $languages
do
    if [ $Debug == 1 ]
    then
        echo "Current language is $curlang"
    fi

    if [ ! -d "$cwd/$curlang" ]
    then
        echo "error:  unable to find $curlang script sub-dirctory" >&2
        continue
    else

        if [ $Debug == 1 ]
        then
            echo "Found $curlang sub-directory"
        fi

        cd $cwd/$curlang

        filename="./finish-swig-${curlang}-${TARGET_NAME}.sh"

        if [ -f $filename ]
        then
            if [ $Debug == 1 ]
            then
                echo "Found $curlang post-processing script for ${TARGET_NAME}"
                echo "Executing $curlang post-processing script..."
            fi

            ./finish-swig-${curlang}-${TARGET_NAME}.sh
        fi
    fi
done

exit 0
