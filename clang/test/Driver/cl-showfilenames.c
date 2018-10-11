// RUN: %clang_cl /c /showFilenames -- %s 2>&1 | FileCheck -check-prefix=show %s
// RUN: %clang_cl /c /showFilenames -- %s %S/Inputs/wildcard*.c 2>&1 | FileCheck -check-prefix=multiple %s

// RUN: %clang_cl /c -- %s 2>&1 | FileCheck -check-prefix=noshow %s
// RUN: %clang_cl /c /showFilenames /showFilenames- -- %s 2>&1 | FileCheck -check-prefix=noshow %s


#pragma message "Hello"

// show: cl-showfilenames.c
// show-NEXT: warning: Hello

// multiple: cl-showfilenames.c
// multiple-NEXT: warning: Hello
// multiple: wildcard1.c
// multiple-NEXT: wildcard2.c

// noshow: warning: Hello
// noshow-NOT: cl-showfilenames.c
