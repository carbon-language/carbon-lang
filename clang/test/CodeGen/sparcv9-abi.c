// RUN: %clang_cc1 -triple sparcv9-unknown-linux -emit-llvm %s -o - | FileCheck %s

// CHECK: define void @f_void()
void f_void(void) {}

// Arguments and return values smaller than the word size are extended.

// CHECK: define signext i32 @f_int_1(i32 signext %x)
int f_int_1(int x) { return x; }

// CHECK: define zeroext i32 @f_int_2(i32 zeroext %x)
unsigned f_int_2(unsigned x) { return x; }

// CHECK: define i64 @f_int_3(i64 %x)
long long f_int_3(long long x) { return x; }

// CHECK: define signext i8 @f_int_4(i8 signext %x)
char f_int_4(char x) { return x; }

// Small structs are passed in registers.
struct small {
  int *a, *b;
};

// CHECK: define %struct.small @f_small(i32* %x.coerce0, i32* %x.coerce1)
struct small f_small(struct small x) {
  x.a += *x.b;
  x.b = 0;
  return x;
}

// Medium-sized structs are passed indirectly, but can be returned in registers.
struct medium {
  int *a, *b;
  int *c, *d;
};

// CHECK: define %struct.medium @f_medium(%struct.medium* %x)
struct medium f_medium(struct medium x) {
  x.a += *x.b;
  x.b = 0;
  return x;
}

// Large structs are also returned indirectly.
struct large {
  int *a, *b;
  int *c, *d;
  int x;
};

// CHECK: define void @f_large(%struct.large* noalias sret %agg.result, %struct.large* %x)
struct large f_large(struct large x) {
  x.a += *x.b;
  x.b = 0;
  return x;
}

