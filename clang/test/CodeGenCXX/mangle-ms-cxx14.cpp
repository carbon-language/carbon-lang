// RUN: %clang_cc1 -std=c++1y -fms-extensions -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

template <typename> int x = 0;

// CHECK: "\01??$x@X@@3HA"
template <> int x<void>;
// CHECK: "\01??$x@H@@3HA"
template <> int x<int>;
