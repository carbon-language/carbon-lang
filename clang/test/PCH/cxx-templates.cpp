// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-templates.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s 

S<float> v;
