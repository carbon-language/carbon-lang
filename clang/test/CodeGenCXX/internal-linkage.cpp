// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

struct Global { Global(); };
template<typename T> struct X { X(); };


namespace {
  struct Anon { Anon(); };

  // CHECK: @_ZN12_GLOBAL__N_15anon0E = internal global
  Global anon0;
}

// CHECK: @anon1 = internal global
Anon anon1;

// CHECK: @anon2 = internal global
X<Anon> anon2;

