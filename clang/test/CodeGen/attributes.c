// RUN: %clang_cc1 -emit-llvm -triple i386-linux-gnu -o %t %s
// RUN: FileCheck --input-file=%t %s

// CHECK: @t5 = weak global i32 2
int t5 __attribute__((weak)) = 2;

// CHECK: @t13 = global %0 zeroinitializer, section "SECT"
struct s0 { int x; };
struct s0 t13 __attribute__((section("SECT"))) = { 0 };

// CHECK: @t14.x = internal global i32 0, section "SECT"
void t14(void) {
  static int x __attribute__((section("SECT"))) = 0;
}

// CHECK: @t18 = global i32 1, align 4
extern int t18 __attribute__((weak_import));
int t18 = 1;

// CHECK: @t16 = extern_weak global i32
extern int t16 __attribute__((weak_import));

// CHECK: @t6 = common protected global i32 0
int t6 __attribute__((visibility("protected")));

// CHECK: @t12 = global i32 0, section "SECT"
int t12 __attribute__((section("SECT")));

// CHECK: @t9 = alias weak bitcast (void ()* @__t8 to void (...)*)
void __t8() {}
void t9() __attribute__((weak, alias("__t8")));

// CHECK: declare extern_weak i32 @t15()
int __attribute__((weak_import)) t15(void);
int t17() {
  return t15() + t16;
}

// CHECK: define void @t1() noreturn nounwind {
void t1() __attribute__((noreturn));
void t1() { while (1) {} }

// CHECK: define void @t2() nounwind {
void t2() __attribute__((nothrow));
void t2() {}

// CHECK: define weak void @t3() nounwind {
void t3() __attribute__((weak));
void t3() {}

// CHECK: define hidden void @t4() nounwind {
void t4() __attribute__((visibility("hidden")));
void t4() {}

// CHECK: define void @t7() noreturn nounwind {
void t7() __attribute__((noreturn, nothrow));
void t7() { while (1) {} }

// CHECK: define void @t10() nounwind section "SECT" {
void t10(void) __attribute__((section("SECT")));
void t10(void) {}
// CHECK: define void @t11() nounwind section "SECT" {
void __attribute__((section("SECT"))) t11(void) {}

// CHECK: define i32 @t19() nounwind {
extern int t19(void) __attribute__((weak_import));
int t19(void) {
  return 10;
}

// CHECK:define void @t20() nounwind {
// CHECK: call void @abort()
// CHECK-NEXT: unreachable
void t20(void) {
  __builtin_abort();
}

void (__attribute__((fastcall)) *fptr)(int);
void t21(void) {
  fptr(10);
}
// CHECK: [[FPTRVAR:%[a-z0-9]+]] = load void (i32)** @fptr
// CHECK-NEXT: call x86_fastcallcc void [[FPTRVAR]](i32 10)
