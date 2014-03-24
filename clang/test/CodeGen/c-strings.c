// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// Should be 3 hello strings, two global (of different sizes), the rest are
// shared.

// CHECK: @align = global i8 [[ALIGN:[0-9]+]]
// CHECK: @.str = private unnamed_addr constant [6 x i8] c"hello\00"
// CHECK: @f1.x = internal global i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0)
// CHECK: @f2.x = internal global [6 x i8] c"hello\00", align [[ALIGN]]
// CHECK: @f3.x = internal global [8 x i8] c"hello\00\00\00", align [[ALIGN]]
// CHECK: @f4.x = internal global %struct.s { i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0) }
// CHECK: @x = global [3 x i8] c"ola", align [[ALIGN]]

#if defined(__s390x__)
unsigned char align = 2;
#else
unsigned char align = 1;
#endif

void bar(const char *);

// CHECK-LABEL: define void @f0()
void f0() {
  bar("hello");
  // CHECK: call void @bar({{.*}} @.str
}

// CHECK-LABEL: define void @f1()
void f1() {
  static char *x = "hello";
  bar(x);
  // CHECK: [[T1:%.*]] = load i8** @f1.x
  // CHECK: call void @bar(i8* [[T1:%.*]])
}

// CHECK-LABEL: define void @f2()
void f2() {
  static char x[] = "hello";
  bar(x);
  // CHECK: call void @bar({{.*}} @f2.x
}

// CHECK-LABEL: define void @f3()
void f3() {
  static char x[8] = "hello";
  bar(x);
  // CHECK: call void @bar({{.*}} @f3.x
}

void gaz(void *);

// CHECK-LABEL: define void @f4()
void f4() {
  static struct s {
    char *name;
  } x = { "hello" };
  gaz(&x);
  // CHECK: call void @gaz({{.*}} @f4.x
}

char x[3] = "ola";

