// RUN: %clang_cc1 %s -std=c++11 -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// rdar://9971124

int foo(int && i)
{ 
     return ^{ return i; }();
}

int main() {
  return foo(100);
}

// CHECK: [[B:%.*]] = bitcast i8* [[BD:%.*]] to <{ {{.*}} i32 }>*
// CHECK: [[C:%.*]] = getelementptr inbounds <{ {{.*}} i32 }>* [[B]]
// CHECK: [[R:%.*]] = load i32* [[C]], align 4
// CHECK: ret i32 [[R]]
