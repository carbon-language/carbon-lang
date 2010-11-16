// RUN: %clang_cc1 -Wno-unused-value -emit-llvm %s -o - | FileCheck %s

// CHECK: @i = common global [[INT:i[0-9]+]] 0
volatile int i, j, k;
volatile int ar[5];
volatile char c;
// CHECK: @ci = common global [[CINT:%.*]] zeroinitializer
volatile _Complex int ci;
volatile struct S {
#ifdef __cplusplus
  void operator =(volatile struct S&o) volatile;
#endif
  int i;
} a, b;

//void operator =(volatile struct S&o1, volatile struct S&o2) volatile;
int printf(const char *, ...);


// Note that these test results are very much specific to C!
// Assignments in C++ yield l-values, not r-values, and the situations
// that do implicit lvalue-to-rvalue conversion are substantially
// reduced.

// CHECK: define void @test()
void test() {
  // CHECK: volatile load [[INT]]* @i
  i;
  // CHECK-NEXT: volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: sitofp [[INT]]
  (float)(ci);
  // CHECK-NEXT: volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  (void)ci;
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: memcpy
  (void)a;
  // CHECK-NEXT: [[R:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: volatile store [[INT]] [[R]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: volatile store [[INT]] [[I]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  (void)(ci=ci);
  // CHECK-NEXT: [[T:%.*]] = volatile load [[INT]]* @j
  // CHECK-NEXT: volatile store [[INT]] [[T]], [[INT]]* @i
  (void)(i=j);
  // CHECK-NEXT: [[R1:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I1:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I2:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // Not sure why they're ordered this way.
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: volatile store [[INT]] [[R]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: volatile store [[INT]] [[I]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  ci+=ci;

  // CHECK-NEXT: [[R1:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I1:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I2:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R:%.*]] = add [[INT]] [[R2]], [[R1]]
  // CHECK-NEXT: [[I:%.*]] = add [[INT]] [[I2]], [[I1]]
  // CHECK-NEXT: volatile store [[INT]] [[R]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: volatile store [[INT]] [[I]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: [[R2:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 0)
  // CHECK-NEXT: [[I2:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // These additions can be elided
  // CHECK-NEXT: add [[INT]] [[R]], [[R2]]
  // CHECK-NEXT: add [[INT]] [[I]], [[I2]]
  (ci += ci) + ci;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add nsw [[INT]]
  (i += j) + k;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: add nsw [[INT]]
  (i += j) + 1;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add [[INT]]
  // CHECK-NEXT: add [[INT]]
  ci+ci;

  // CHECK-NEXT: volatile load
  __real i;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  +ci;
  // CHECK-NEXT: call void asm
  asm("nop");
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  (void)(i=i);
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: sitofp
  (float)(i=i);
  // CHECK-NEXT: volatile load
  (void)i;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  i=i;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  i=i=i;
#ifndef __cplusplus
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  (void)__builtin_choose_expr(0, i=i, j=j);
#endif
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: icmp
  // CHECK-NEXT: br i1
  // CHECK: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: br label
  // CHECK: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: br label
  k ? (i=i) : (j=j);
  // CHECK: phi
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  (void)(i,(i=i));
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile load
  i=i,i;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  (i=j,k=j);
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile load
  (i=j,k);
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  (i,j);
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: trunc
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: sext
  // CHECK-NEXT: volatile store
  i=c=k;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add nsw [[INT]]
  // CHECK-NEXT: volatile store
  i+=k;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  ci;
#ifndef __cplusplus
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  (int)ci;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: icmp ne
  // CHECK-NEXT: or i1
  (_Bool)ci;
#endif
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  ci=ci;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  ci=ci=ci;
  // CHECK-NEXT: [[T:%.*]] = volatile load [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: volatile store [[INT]] [[T]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  // CHECK-NEXT: volatile store [[INT]] [[T]], [[INT]]* getelementptr inbounds ([[CINT]]* @ci, i32 0, i32 1)
  __imag ci = __imag ci = __imag ci;
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  __real (i = j);
  // CHECK-NEXT: volatile load
  __imag i;
  
  // ============================================================
  // FIXME: Test cases we get wrong.

  // A use.  We load all of a into a copy of a, then load i.  gcc forgets to do
  // the assignment.
  // (a = a).i;

  // ============================================================
  // Test cases where we intentionally differ from gcc, due to suspected bugs in
  // gcc.

  // Not a use.  gcc forgets to do the assignment.
  // CHECK-NEXT: call void @llvm.memcpy{{.*}}, i1 true
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.memcpy{{.*}}, i1 true
  ((a=a),a);

  // Not a use.  gcc gets this wrong, it doesn't emit the copy!  
  // (void)(a=a);

  // Not a use.  gcc got this wrong in 4.2 and omitted the side effects
  // entirely, but it is fixed in 4.4.0.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  __imag (i = j);

#ifndef __cplusplus
  // A use of the real part
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: sitofp
  (float)(ci=ci);
  // Not a use, bug?  gcc treats this as not a use, that's probably a bug due to
  // tree folding ignoring volatile.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  (int)(ci=ci);
#endif

  // A use.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: sitofp
  (float)(i=i);
  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  (int)(i=i);

  // A use.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: sub
  -(i=j);
  // A use.  gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  +(i=k);

  // A use. gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile store
  __real (ci=ci);

  // A use.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add
  i + 0;
  // A use.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: add
  (i=j) + i;
  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  // CHECK-NEXT: volatile load
  // CHECK-NEXT: volatile store
  // CHECK-NEXT: add
  (i=j) + 0;

#ifdef __cplusplus
  (i,j)=k;
  (j=k,i)=i;
  struct { int x; } s, s1;
  printf("s is at %p\n", &s);
  printf("s is at %p\n", &(s = s1));
  printf("s.x is at %p\n", &((s = s1).x));
#endif
}
