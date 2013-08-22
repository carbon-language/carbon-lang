// RUN: mkdir -p %T/basic
// RUN: grep -Ev "// *[A-Z-]+:" %S/basic/basic.h > %T/basic/basic.h
// RUN: sed "s#\$(path)#%/T/basic#" %S/basic/file1.yaml > %T/basic/file1.yaml
// RUN: sed "s#\$(path)#%/T/basic#" %S/basic/file2.yaml > %T/basic/file2.yaml
// RUN: clang-replace %T/basic
// RUN: FileCheck -input-file=%T/basic/basic.h %S/basic/basic.h
