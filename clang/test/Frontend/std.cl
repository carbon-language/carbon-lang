// RUN: %clang_cc1 %s -fsyntax-only -cl-std=CL
// RUN: %clang_cc1 %s -fsyntax-only -cl-std=CL1.1
// RUN: %clang_cc1 %s -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -fsyntax-only -cl-std=CL2.0
// RUN: not %clang_cc1 %s -fsyntax-only -cl-std=invalid -DINVALID 2>&1 | FileCheck %s

#ifdef INVALID 
// CHECK: invalid value 'invalid' in '-cl-std=invalid'
#endif
