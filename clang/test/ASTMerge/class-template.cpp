// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/class-template1.cpp
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/class-template2.cpp
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: class-template1.cpp:7:14: error: non-type template parameter declared with incompatible types in different translation units ('int' vs. 'long')
// CHECK: class-template2.cpp:7:15: note: declared here with type 'long'

// CHECK: class-template1.cpp:10:14: error: template parameter has different kinds in different translation units
// CHECK: class-template2.cpp:10:10: note: template parameter declared here

// CHECK: class-template1.cpp:16:23: error: non-type template parameter declared with incompatible types in different translation units ('long' vs. 'int')
// CHECK: class-template2.cpp:16:23: note: declared here with type 'int'

// CHECK: class-template1.cpp:19:10: error: template parameter has different kinds in different translation units
// CHECK: class-template2.cpp:19:10: note: template parameter declared here
