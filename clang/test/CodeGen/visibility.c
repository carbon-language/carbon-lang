// RUN: %clang_cc1 %s -triple i386-unknown-unknown -fvisibility default -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -fvisibility protected -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-PROTECTED
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -fvisibility hidden -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-HIDDEN

// CHECK-DEFAULT: @g_def = global i32 0
// CHECK-DEFAULT: @g_com = common global i32 0
// CHECK-DEFAULT: @g_ext = external global i32
// CHECK-DEFAULT: @g_deferred = internal global
// CHECK-PROTECTED: @g_def = protected global i32 0
// CHECK-PROTECTED: @g_com = common protected global i32 0
// CHECK-PROTECTED: @g_ext = external global i32
// CHECK-PROTECTED: @g_deferred = internal global
// CHECK-HIDDEN: @g_def = hidden global i32 0
// CHECK-HIDDEN: @g_com = common hidden global i32 0
// CHECK-HIDDEN: @g_ext = external global i32
// CHECK-HIDDEN: @g_deferred = internal global
int g_com;
int g_def = 0;
extern int g_ext;
static char g_deferred[] = "hello";

// CHECK-DEFAULT: define i32 @f_def()
// CHECK-DEFAULT: declare void @f_ext()
// CHECK-DEFAULT: define internal void @f_deferred()
// CHECK-PROTECTED: define protected i32 @f_def()
// CHECK-PROTECTED: declare void @f_ext()
// CHECK-PROTECTED: define internal void @f_deferred()
// CHECK-HIDDEN: define hidden i32 @f_def()
// CHECK-HIDDEN: declare void @f_ext()
// CHECK-HIDDEN: define internal void @f_deferred()

extern void f_ext(void);

static void f_deferred(void) {
}

int f_def(void) {
  f_ext();
  f_deferred();
  return g_com + g_def + g_ext + g_deferred[0];
}

// PR8457
// CHECK-DEFAULT: define void @test1(
// CHECK-PROTECTED: define void @test1(
// CHECK-HIDDEN: define void @test1(
struct Test1 { int field; };
void  __attribute__((visibility("default"))) test1(struct Test1 *v) { }
