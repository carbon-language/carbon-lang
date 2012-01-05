// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin -emit-llvm -o - %s -std=c++11 | FileCheck %s

namespace CrossFuncLabelDiff {
  // Make sure we refuse to constant-fold the variable b.
  constexpr long a() { return (long)&&lbl + (0 && ({lbl: 0;})); }
  void test() { static long b = (long)&&lbl - a(); lbl: return; }
  // CHECK: sub nsw i64 ptrtoint (i8* blockaddress(@_ZN18CrossFuncLabelDiff4testEv, {{.*}}) to i64),
  // CHECK: store i64 {{.*}}, i64* @_ZZN18CrossFuncLabelDiff4testEvE1b, align 8
}
