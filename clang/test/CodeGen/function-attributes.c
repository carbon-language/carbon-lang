// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -Os -o - %s | FileCheck %s
// CHECK: define signext i8 @f0(i32 %x) #0
// CHECK: define zeroext i8 @f1(i32 %x) #0
// CHECK: define void @f2(i8 signext %x) #0
// CHECK: define void @f3(i8 zeroext %x) #0
// CHECK: define signext i16 @f4(i32 %x) #0
// CHECK: define zeroext i16 @f5(i32 %x) #0
// CHECK: define void @f6(i16 signext %x) #0
// CHECK: define void @f7(i16 zeroext %x) #0

signed char f0(int x) { return x; }

unsigned char f1(int x) { return x; }

void f2(signed char x) { }

void f3(unsigned char x) { }

signed short f4(int x) { return x; }

unsigned short f5(int x) { return x; }

void f6(signed short x) { }

void f7(unsigned short x) { }

// CHECK: define void @f8()
// CHECK: #1
// CHECK: {
void __attribute__((always_inline)) f8(void) { }

// CHECK: call void @f9_t()
// CHECK: [[F9:#[0-9]+]]
// CHECK: }
void __attribute__((noreturn)) f9_t(void);
void f9(void) { f9_t(); }

// CHECK: call void @f9a()
// CHECK: [[F9]]
// CHECK: }
_Noreturn void f9a(void);
void f9b(void) { f9a(); }

// FIXME: We should be setting nounwind on calls.
// CHECK: call i32 @f10_t()
// CHECK: [[F10_T:#[0-9]+]]
// CHECK: {
int __attribute__((const)) f10_t(void);
int f10(void) { return f10_t(); }
int f11(void) {
 exit:
  return f10_t();
}
int f12(int arg) {
  return arg ? 0 : f10_t();
}

// CHECK: define void @f13() #0
void f13(void) __attribute__((pure)) __attribute__((const));
void f13(void){}


// Ensure that these get inlined: rdar://6853279
// CHECK: define void @f14
// CHECK-NOT: @ai_
// CHECK: call void @f14_end
static __inline__ __attribute__((always_inline))
int ai_1() {  return 4; }

static __inline__ __attribute__((always_inline))
struct {
  int a, b, c, d, e;
} ai_2() { while (1) {} }

void f14(int a) {
  extern void f14_end(void);
  if (a)
    ai_2();
  ai_1();
  f14_end();
}

// <rdar://problem/7102668> [irgen] clang isn't setting the optsize bit on functions
// CHECK: define void @f15
// CHECK: #0
// CHECK: {
void f15(void) {
}

// PR5254
// CHECK: define void @f16
// CHECK: #6
// CHECK: {
void __attribute__((force_align_arg_pointer)) f16(void) {
}

// PR11038
// CHECK: define void @f18()
// CHECK: #7
// CHECK: {
// CHECK: call void @f17()
// CHECK: [[F17:#[0-9]+]]
// CHECK: ret void
__attribute__ ((returns_twice)) void f17(void);
__attribute__ ((returns_twice)) void f18(void) {
        f17();
}

// CHECK: define void @f19()
// CHECK: {
// CHECK: call i32 @setjmp(i32* null)
// CHECK: [[F17]]
// CHECK: ret void
typedef int jmp_buf[((9 * 2) + 3 + 16)];
int setjmp(jmp_buf);
void f19(void) {
  setjmp(0);
}

// CHECK: attributes #0 = { nounwind optsize readnone "target-features"={{.*}} }
// CHECK: attributes #1 = { alwaysinline nounwind optsize readnone "target-features"={{.*}} }
// CHECK: attributes #2 = { noreturn nounwind optsize "target-features"={{.*}} }
// CHECK: attributes #3 = { noreturn optsize "target-features"={{.*}} }
// CHECK: attributes #4 = { nounwind optsize "target-features"={{.*}} }
// CHECK: attributes #5 = { optsize "target-features"={{.*}} }
// CHECK: attributes #6 = { nounwind optsize readnone alignstack=16 "target-features"={{.*}} }
// CHECK: attributes #7 = { nounwind optsize returns_twice "target-features"={{.*}} }
// CHECK: attributes #8 = { optsize returns_twice "target-features"={{.*}}
// CHECK: attributes [[F9]] = { noreturn nounwind optsize }
// CHECK: attributes [[F10_T]] = { nounwind optsize readnone }
// CHECK: attributes [[F17]] = { nounwind optsize returns_twice }
