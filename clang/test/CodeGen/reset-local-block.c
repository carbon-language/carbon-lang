// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm  -fblocks -fcatch-undefined-behavior -o - %s | FileCheck %s
// rdar://9227352

typedef int (^BLOCK)();

BLOCK FUNC() {
  int i;
  double d;
  BLOCK block = ^{ return i + (int)d; };
  if (!block)
    block = ^{ return i; };
  return block;
}

//CHECK: call void @llvm.memset{{.*}}, i8 -51, i64 36, i32 8, i1 false)
//CHECK: call void @llvm.memset{{.*}}, i8 -51, i64 44, i32 8, i1 false)
