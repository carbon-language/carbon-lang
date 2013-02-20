// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -fms-compatibility -o - | FileCheck %s

struct __declspec(align(16)) S {
  char x;
};
union { struct S s; } u;

// CHECK: @u = {{.*}}zeroinitializer, align 16


// CHECK: define void @t3() #0 {
__declspec(naked) void t3() {}

// CHECK: define void @t22() #1
void __declspec(nothrow) t22();
void t22() {}

// CHECK: define void @t2() #2 {
__declspec(noinline) void t2() {}

// CHECK: call void @f20_t() noreturn
__declspec(noreturn) void f20_t(void);
void f20(void) { f20_t(); }

// CHECK: attributes #0 = { naked noinline nounwind "target-features"={{.*}} }
// CHECK: attributes #1 = { nounwind "target-features"={{.*}} }
// CHECK: attributes #2 = { noinline nounwind "target-features"={{.*}} }
// CHECK: attributes #3 = { noreturn "target-features"={{.*}} }
