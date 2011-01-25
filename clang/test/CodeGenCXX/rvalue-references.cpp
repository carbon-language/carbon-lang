// RUN: %clang_cc1 -std=c++0x -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s


struct Spacer { int x; };
struct A { double array[2]; };
struct B : Spacer, A { };

B &getB();

// CHECK: define %struct.A* @_Z4getAv()
// CHECK: call %struct.B* @_Z4getBv()
// CHECK-NEXT: bitcast %struct.B*
// CHECK-NEXT: getelementptr i8*
// CHECK-NEXT: bitcast i8* {{.*}} to %struct.A*
// CHECK-NEXT: ret %struct.A*
A &&getA() { return static_cast<A&&>(getB()); }

int &getIntLValue();
int &&getIntXValue();
int getIntPRValue();

// CHECK: define i32* @_Z2f0v()
// CHECK: call i32* @_Z12getIntLValuev()
// CHECK-NEXT: ret i32*
int &&f0() { return static_cast<int&&>(getIntLValue()); }

// CHECK: define i32* @_Z2f1v()
// CHECK: call i32* @_Z12getIntXValuev()
// CHECK-NEXT: ret i32*
int &&f1() { return static_cast<int&&>(getIntXValue()); }

// CHECK: define i32* @_Z2f2v
// CHECK: call i32 @_Z13getIntPRValuev()
// CHECK-NEXT: store i32 {{.*}}, i32*
// CHECK-NEXT: ret i32*
int &&f2() { return static_cast<int&&>(getIntPRValue()); }
