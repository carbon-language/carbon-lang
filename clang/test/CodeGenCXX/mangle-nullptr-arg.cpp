// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s | FileCheck %s

template<int *ip> struct IP {};

// CHECK: define void @_Z5test12IPILPi0EE
void test1(IP<nullptr>) {}

struct X{ };
template<int X::*pm> struct PM {};

// CHECK: define void @_Z5test22PMILM1Xi0EE
void test2(PM<nullptr>) { }

