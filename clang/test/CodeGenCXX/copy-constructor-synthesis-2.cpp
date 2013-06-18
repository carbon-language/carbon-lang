// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i686-unknown-linux | FileCheck --check-prefix=CHECKX86 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=arm-linux-gnueabihf | FileCheck --check-prefix=CHECKARM %s

struct A { virtual void a(); };
A x(A& y) { return y; }

// CHECKX86: define linkonce_odr {{.*}} @_ZN1AC1ERKS_(%struct.A* %this, %struct.A*) unnamed_addr
// CHECKX86: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1A, i64 0, i64 2)

// CHECKARM: define linkonce_odr {{.*}} @_ZN1AC1ERKS_(%struct.A* returned %this, %struct.A*) unnamed_addr
// CHECKARM: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1A, i64 0, i64 2)
