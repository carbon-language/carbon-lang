// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -include %S/cxx-reference.h -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -emit-pch -o %t %S/cxx-reference.h
// RUN: %clang_cc1 -x c++ -std=c++0x -include-pch %t -fsyntax-only -emit-llvm -o - %s 
