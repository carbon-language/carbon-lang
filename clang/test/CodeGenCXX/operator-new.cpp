// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -o %t-1.ll %s
// RUN: FileCheck -check-prefix SANE --input-file=%t-1.ll %s
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -emit-llvm -fno-assume-sane-operator-new -o %t-2.ll %s
// RUN: FileCheck -check-prefix SANENOT --input-file=%t-2.ll %s


class teste {
  int A;
  teste() : A(2) {}
  void* operator new(unsigned) {return ::new teste();}
};

void f1() {
  new teste();
}

// CHECK-SANE: define linkonce_odr noalias i8* @_ZN5testenwEj(
// CHECK-SANE: declare noalias i8* @_Znwj(

// CHECK-SANENOT: define linkonce_odr i8* @_ZN5testenwEj(
// CHECK-SANENOT: declare i8* @_Znwj(
