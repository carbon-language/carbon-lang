// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s
// rdar://9309454

typedef struct { int v; } RetType;

RetType func();

int main () {
 __attribute__((__blocks__(byref))) RetType a = {100};

 a = func();
}
// CHECK: [[C1:%.*]] = call i32 (...)* @func()
// CHECK-NEXT: [[CO:%.*]] = getelementptr
// CHECK-NEXT: store i32 [[C1]], i32* [[CO]]
// CHECK-NEXT: [[FORWARDING:%.*]] = getelementptr inbounds [[BR:%.*]]* [[A:%.*]], i32 0, i32 1
// CHECK-NEXT: [[O:%.*]] = load [[BR]]** [[FORWARDING]]
