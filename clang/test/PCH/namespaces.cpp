// Test this without pch.
// RUN: %clang_cc1 -x c++ -include %S/Inputs/namespaces.h -fsyntax-only %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -emit-pch -o %t %S/Inputs/namespaces.h
// RUN: %clang_cc1 -x c++ -include-pch %t -fsyntax-only %s 

int int_val;
N1::t1 *ip1 = &int_val;
N1::t2 *ip2 = &int_val;

float float_val;
namespace N2 { }
N2::t1 *fp1 = &float_val;
