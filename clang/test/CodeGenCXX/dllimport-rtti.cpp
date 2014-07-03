// RUN: %clang_cc1 -triple i686-windows-msvc -emit-llvm -std=c++1y -O1 -disable-llvm-optzns -o - %s | FileCheck %s

struct __declspec(dllimport) S {
  virtual void f();
} s;
// CHECK-DAG: @"\01??_7S@@6B@" = available_externally dllimport
// CHECK-DAG: @"\01??_R0?AUS@@@8" = linkonce_odr
// CHECK-DAG: @"\01??_R1A@?0A@EA@S@@8" = linkonce_odr
// CHECK-DAG: @"\01??_R2S@@8" = linkonce_odr
// CHECK-DAG: @"\01??_R3S@@8" = linkonce_odr

struct U : S {
} u;
