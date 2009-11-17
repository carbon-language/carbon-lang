// RUN: clang-cc -Wno-unused-value -emit-llvm < %s -o %t
// RUN: grep volatile %t | count 145
// RUN: grep memcpy %t | count 4

volatile int i, j, k;
volatile int ar[5];
volatile char c;
volatile _Complex int ci;
volatile struct S {
#ifdef __cplusplus
  void operator =(volatile struct S&o) volatile;
#endif
  int i;
} a, b;

//void operator =(volatile struct S&o1, volatile struct S&o2) volatile;
int printf(const char *, ...);

int main() {
  // A use.
  i;
  // A use of the real part
  (float)(ci);
  // A use.
  (void)ci;
  // A use.
  (void)a;
  // Not a use.
  (void)(ci=ci);
  // Not a use.
  (void)(i=j);
  ci+=ci;
  (ci += ci) + ci;
  asm("nop");
  (i += j) + k;
  asm("nop");
  // A use
  (i += j) + 1;
  asm("nop");
  ci+ci;
  // A use.
  __real i;
  // A use.
  +ci;
  asm("nop");
  // Not a use.
  (void)(i=i);
  (float)(i=i);
  // A use.
  (void)i;
  i=i;
  i=i=i;
#ifndef __cplusplus
  // Not a use.
  (void)__builtin_choose_expr(0, i=i, j=j);
#endif
  // A use.
  k ? (i=i) : (j=j);
  (void)(i,(i=i));
  i=i,i;
  (i=j,k=j);
  (i=j,k);
  (i,j);
  i=c=k;
  i+=k;
  // A use of both.
  ci;
#ifndef __cplusplus
  // A use of _real.
  (int)ci;
  // A use of both.
  (_Bool)ci;
#endif
  ci=ci;
  ci=ci=ci;
  __imag ci = __imag ci = __imag ci;
  // Not a use.
  __real (i = j);
  // Not a use.
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
  ((a=a),a);

  // Not a use.  gcc gets this wrong, it doesn't emit the copy!  
  // (void)(a=a);

  // Not a use.  gcc got this wrong in 4.2 and omitted the side effects
  // entirely, but it is fixed in 4.4.0.
  __imag (i = j);

#ifndef __cplusplus
  // A use of the real part
  (float)(ci=ci);
  // Not a use, bug?  gcc treats this as not a use, that's probably a bug due to
  // tree folding ignoring volatile.
  (int)(ci=ci);
#endif

  // A use.
  (float)(i=i);
  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  (int)(i=i);

  // A use.
  -(i=j);
  // A use.  gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  +(i=k);

  // A use. gcc treats this a not a use, that's probably a bug due to tree
  // folding ignoring volatile.
  __real (ci=ci);

  // A use.
  i + 0;
  // A use.
  (i=j) + i;
  // A use.  gcc treats this as not a use, that's probably a bug due to tree
  // folding ignoring volatile.
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
