// RUN: mkdir -p %T/Inputs/basic
// RUN: grep -Ev "// *[A-Z-]+:" %S/Inputs/basic/basic.h > %T/Inputs/basic/basic.h
// RUN: sed "s#\$(path)#%/T/Inputs/basic#" %S/Inputs/basic/file1.yaml > %T/Inputs/basic/file1.yaml
// RUN: sed "s#\$(path)#%/T/Inputs/basic#" %S/Inputs/basic/file2.yaml > %T/Inputs/basic/file2.yaml
// RUN: clang-replace %T/Inputs/basic
// RUN: FileCheck -input-file=%T/Inputs/basic/basic.h %S/Inputs/basic/basic.h
