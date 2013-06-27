// RUN: not %clang_cc1 -emit-llvm -o %S/doesnotexist/somename %s 2> %t
// RUN: FileCheck -check-prefix=OUTPUTFAIL -input-file=%t %s

// OUTPUTFAIL:  unable to open output file '{{.*}}doesnotexist{{.*}}': 'Parent directory doesn't exist'
