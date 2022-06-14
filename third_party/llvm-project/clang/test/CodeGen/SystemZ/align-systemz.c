// RUN: %clang_cc1 -no-opaque-pointers -triple s390x-linux-gnu -emit-llvm %s -o - | FileCheck %s

// SystemZ prefers to align all global variables to two bytes.

struct test {
   signed char a;
};

char c;
// CHECK-DAG: @c ={{.*}} global i8 0, align 2

struct test s;
// CHECK-DAG: @s ={{.*}} global %struct.test zeroinitializer, align 2

extern char ec;
// CHECK-DAG: @ec = external global i8, align 2

extern struct test es;
// CHECK-DAG: @es = external global %struct.test, align 2

// Dummy function to make sure external symbols are used.
void func (void)
{
  c = ec;
  s = es;
}


// Alignment should be respected for coerced argument loads

struct arg { long y __attribute__((packed, aligned(4))); };

extern struct arg x;
void f(struct arg);

void test (void)
{
  f(x);
}

// CHECK-LABEL: @test
// CHECK: load i64, i64* getelementptr inbounds (%struct.arg, %struct.arg* @x, i32 0, i32 0), align 4

