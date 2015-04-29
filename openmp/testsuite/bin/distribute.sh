#!/bin/bash

# add header for .ll files

# get tmp header
cp header /tmp/tmp.header
echo >> /tmp/tmp.header

# create temporary test package
mkdir c-$MACHTYPE$OSTYPE
`cp c/*.ll c-$MACHTYPE$OSTYPE/`

# add new header into .ll files
for file in c-$MACHTYPE$OSTYPE/*
do
    cp $file /tmp/tmp.ll.bf
    cat /tmp/tmp.header /tmp/tmp.ll.bf > /tmp/tmp.ll
    mv /tmp/tmp.ll $file
done


# in bin/, target is ../LLVM-IR/ARCH/OS
LEVEL=../LLVM-IR/
ARCH_PATH=../LLVM-IR/
OS_PATH=../LLVM-IR/

# for linux system, add your arch and os here
declare -a ARCHes=(x86 x86_64 powerpc arm mips darwin)
declare -a OSes=(linux macosx windows darwin)

declare lowerARCH
declare lowerOS

# target directory name
declare upperARCH
declare upperOS

lowerARCH=$(echo "$MACHTYPE" | tr '[:upper:]' '[:lower:]')
lowerOS=$(echo "$OSTYPE" | tr '[:upper:]' '[:lower:]')

# ARCH
for i in ${ARCHes[@]}
do
    result=$(echo "${lowerARCH}" | grep $i)
    if [[ "$result" != "" ]]
    then
        # upperARCH=$i
		upperARCH=$(echo "$i" | tr '[:lower:]' '[:upper:]')
    fi
done

if [[ "$upperARCH" == "" ]]
then
    echo "Not found ${lowerARCH} in the [${ARCHes[@]}]!"
    exit
fi

# OS
for i in ${OSes[@]}
do
    result=$(echo "${lowerOS}" | grep $i)
    if [[ "$result" != "" ]]
    then
        # upperOS=$i
		upperOS=$(echo "$i" | tr '[:lower:]' '[:upper:]')
    fi
done

if [[ "$upperOS" == "" ]]
then
    echo "Not found ${lowerOS} in the [${OSes[@]}]!"
    exit
fi

# survived, assemble the path
# ARCH_PATH+=$upperARCH/
# OS_PATH+=$upperARCH/$upperOS/
ARCH_newFormat=.
if [ $upperARCH = "X86" ]; then
    ARCH_newFormat=32
else
    ARCH_newFormat=32e
fi
OS_newFormat=.
if [ $upperOS = "LINUX" ]; then
    OS_newFormat=lin
elif [ $upperOS = "MACOSX" ]; then
    OS_newFormat=mac
elif [ $upperOS = "WINDOWS" ]; then
    OS_newFormat=win
elif [ $upperOS = "DARWIN" ]; then
    OS_newFormat=dar
else
    OS_newFormat=unknown
fi
OS_PATH+=$OS_newFormat"_"$ARCH_newFormat

# test and create directory
if [ ! -d "$LEVEL" ]; then
    mkdir $LEVEL
    mkdir $OS_PATH
else
    if [ ! -d "$OS_PATH" ]; then
        mkdir $OS_PATH
    fi
fi

# reserve the tmp path to LLVM-IR/ARCH/OS
echo $OS_PATH"/" > lit.tmp

# OS_ARCH=$OS_newFormat"_"$ARCH_newFormat
# echo -e "if not '$OS_ARCH' in config.root.targets:" > $OS_PATH'/'lit.local.cfg
# echo -e "\tconfig.unsupported = True" >> $OS_PATH'/'lit.local.cfg

# copy *.ll files to ARCH/OS
`cp lit.* $LEVEL`

# omit orph test
`rm c-$MACHTYPE$OSTYPE/ctest_*.ll`
`rm c-$MACHTYPE$OSTYPE/orph_ctest_*.ll`
`cp c-$MACHTYPE$OSTYPE/*.ll $OS_PATH`

# clean
`rm /tmp/tmp.*`
rm -rf c-$MACHTYPE$OSTYPE/
