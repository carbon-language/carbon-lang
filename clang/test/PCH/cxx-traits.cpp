// Test this without pch.
// RUN: %clang_cc1 -include %S/cxx-traits.h -fsyntax-only -verify %s

// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/cxx-traits.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s

bool _Is_pod_comparator = __is_pod<int>::__value;
bool _Is_empty_check = __is_empty<int>::__value;
