// RUN: clang-cc %s -emit-llvm -o -

// PR1895
// sizeof function
int zxcv(void);
int x=sizeof(zxcv);
int y=__alignof__(zxcv);


void *test(int *i) {
 short a = 1;
 i += a;
 i + a;
 a + i;
}

_Bool test2b; 
int test2() { if (test2b); return 0; }

// PR1921
int test3() {
  const unsigned char *bp;
  bp -= (short)1;
}

// PR2080 - sizeof void
int t1 = sizeof(void);
int t2 = __alignof__(void);
void test4() {
  t1 = sizeof(void);
  t2 = __alignof__(void);
  
  t1 = sizeof(test4());
  t2 = __alignof__(test4());
}

// 'const float' promotes to double in varargs.
int test5(const float x, float float_number) {
  return __builtin_isless(x, float_number);
}

// this one shouldn't fold
int ola() {
  int a=2;
  if ((0, (int)a) & 2) { return 1; }
  return 2;
}

// this one shouldn't fold as well
void eMaisUma() {
  double t[1];
  if (*t)
    return;
}

// rdar://6520707
void f0(void (*fp)(void), void (*fp2)(void)) {
  int x = fp - fp2;
}

// noop casts as lvalues.
struct X {
  int Y;
};
struct X foo();
int bar() {
  return ((struct X)foo()).Y + 1;
}

// PR3809: INC/DEC of function pointers.
void f2(void);
unsigned f1(void) {
  void (*fp)(void) = f2;
  
  ++fp;
  fp++;
  --fp;
  fp--;
  return (unsigned) fp;
}  

union f3_x {int x; float y;};
int f3() {return ((union f3_x)2).x;}

union f4_y {int x; _Complex float y;};
_Complex float f4() {return ((union f4_y)(_Complex float)2.0).y;}

struct f5_a { int a; } f5_a;
union f5_z {int x; struct f5_a y;};
struct f5_a f5() {return ((union f5_z)f5_a).y;}

// ?: in "lvalue"
struct s6 { int f0; };
int f6(int a0, struct s6 a1, struct s6 a2) {
  return (a0 ? a1 : a2).f0;
}

// PR4026
void f7() {
  __func__;
}

// PR4067
int f8() {
  return ({ foo(); }).Y;
}

// rdar://6880558
struct S;
struct C {
  int i;
  struct S *tab[];
};
struct S { struct C c; };
void f9(struct S *x) {
  foo(((void)1, x->c).tab[0]);
}

