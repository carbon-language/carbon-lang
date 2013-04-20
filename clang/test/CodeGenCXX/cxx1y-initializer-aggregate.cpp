// RUN: %clang_cc1 -std=c++1y %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s

struct A {
  int n = 0;
  const char *p;
  char k = p[n];
  int f();
  int x = f();
  union {
    char c;
    double d = 1.0;
  };
};

int f();

union B {
  int a;
  int f();
  int b = f();
};

A a { .p = "foobar" };
A b { 4, "bazquux", .x = 42, .c = 9 };
A c { 1, 0, 'A', f(), { 3 } };

// CHECK: @[[STR_A:.*]] = {{.*}} [7 x i8] c"foobar\00"
// CHECK: @[[STR_B:.*]] = {{.*}} [8 x i8] c"bazquux\00"

B x;
B y {};
B z { 1 };
// CHECK: @z = global {{.*}} { i32 1 }

// Initialization of 'a':

// CHECK: store i32 0, i32* getelementptr inbounds ({{.*}} @a, i32 0, i32 0)
// CHECK: store i8* {{.*}} @[[STR_A]]{{.*}}, i8** getelementptr inbounds ({{.*}} @a, i32 0, i32 1)
// CHECK: load i32* getelementptr inbounds ({{.*}} @a, i32 0, i32 0)
// CHECK: load i8** getelementptr inbounds ({{.*}} @a, i32 0, i32 1)
// CHECK: getelementptr inbounds i8* %{{.*}}, {{.*}} %{{.*}}
// CHECK: store i8 %{{.*}}, i8* getelementptr inbounds ({{.*}} @a, i32 0, i32 2)
// CHECK: call i32 @_ZN1A1fEv({{.*}} @a)
// CHECK: store i32 %{{.*}}, i32* getelementptr inbounds ({{.*}}* @a, i32 0, i32 3)
// CHECK: call void @{{.*}}C1Ev({{.*}} getelementptr inbounds (%struct.A* @a, i32 0, i32 4))

// Initialization of 'b':

// CHECK: store i32 4, i32* getelementptr inbounds ({{.*}} @b, i32 0, i32 0)
// CHECK: store i8* {{.*}} @[[STR_B]]{{.*}}, i8** getelementptr inbounds ({{.*}} @b, i32 0, i32 1)
// CHECK: load i32* getelementptr inbounds ({{.*}} @b, i32 0, i32 0)
// CHECK: load i8** getelementptr inbounds ({{.*}} @b, i32 0, i32 1)
// CHECK: getelementptr inbounds i8* %{{.*}}, {{.*}} %{{.*}}
// CHECK: store i8 %{{.*}}, i8* getelementptr inbounds ({{.*}} @b, i32 0, i32 2)
// CHECK-NOT: @_ZN1A1fEv
// CHECK: store i32 42, i32* getelementptr inbounds ({{.*}}* @b, i32 0, i32 3)
// CHECK-NOT: C1Ev
// CHECK: store i8 9, i8* {{.*}} @b, i32 0, i32 4)

// Initialization of 'c':

// CHECK: store i32 1, i32* getelementptr inbounds ({{.*}} @c, i32 0, i32 0)
// CHECK: store i8* null, i8** getelementptr inbounds ({{.*}} @c, i32 0, i32 1)
// CHECK-NOT: load
// CHECK: store i8 65, i8* getelementptr inbounds ({{.*}} @c, i32 0, i32 2)
// CHECK: call i32 @_Z1fv()
// CHECK: store i32 %{{.*}}, i32* getelementptr inbounds ({{.*}}* @c, i32 0, i32 3)
// CHECK-NOT: C1Ev
// CHECK: store i8 3, i8* {{.*}} @c, i32 0, i32 4)

// CHECK: call void @_ZN1BC1Ev({{.*}} @x)

// CHECK: call i32 @_ZN1B1fEv({{.*}} @y)
// CHECK: store i32 %{{.*}}, i32* getelementptr inbounds ({{.*}} @y, i32 0, i32 0)
