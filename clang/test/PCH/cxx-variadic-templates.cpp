// Test this without pch.
// RUN: %clang_cc1 -std=c++0x -include %S/cxx-variadic-templates.h -verify %s -ast-dump -o -
// RUN: %clang_cc1 -std=c++0x -include %S/cxx-variadic-templates.h %s -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -std=c++0x -x c++-header -emit-pch -o %t %S/cxx-variadic-templates.h
// RUN: %clang_cc1 -std=c++0x -include-pch %t -verify %s -ast-dump  -o -
// RUN: %clang_cc1 -std=c++0x -include-pch %t %s -emit-llvm -o - | FileCheck %s

// CHECK: allocate_shared
shared_ptr<int> spi = shared_ptr<int>::allocate_shared(1, 2);
