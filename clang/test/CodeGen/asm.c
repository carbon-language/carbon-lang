// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
void t1(int len) {
  __asm__ volatile("" : "=&r"(len), "+&r"(len));
}

void t2(unsigned long long t)  {
  __asm__ volatile("" : "+m"(t));
}

void t3(unsigned char *src, unsigned long long temp) {
  __asm__ volatile("" : "+m"(temp), "+r"(src));
}

void t4() {
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

  __asm__ volatile ("":: "m"(a), "m"(b));
}

// PR3417
void t5(int i) {
  asm("nop" : "=r"(i) : "0"(t5));
}

// PR3641
void t6(void) {
  __asm__ volatile("" : : "i" (t6));
}

void t7(int a) {
  __asm__ volatile("T7 NAMED: %[input]" : "+r"(a): [input] "i" (4));
  // CHECK: @t7(i32
  // CHECK: T7 NAMED: $1
}

void t8() {
  __asm__ volatile("T8 NAMED MODIFIER: %c[input]" :: [input] "i" (4));
  // CHECK: @t8()
  // CHECK: T8 NAMED MODIFIER: ${0:c}
}

// PR3682
unsigned t9(unsigned int a) {
  asm("bswap %0 %1" : "+r" (a));
  return a;
}

// PR3908
void t10(int r) {
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]" : [r] "+r" (r) : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
  
// CHECK: @t10(
// CHECK:PR3908 $1 $3 $2 $0
}         


// PR3373
unsigned t11(signed char input) {
  unsigned  output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

// PR3373
unsigned char t12(unsigned input) {
  unsigned char output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

unsigned char t13(unsigned input) {
  unsigned char output;
  __asm__("xyz %1"
          : "=a" (output)
          : "0" (input));
  return output;
}

struct large {
  int x[1000];
};

unsigned long t15(int x, struct large *P) {
  __asm__("xyz "
          : "=r" (x)
          : "m" (*P), "0" (x));
  return x;
}




// bitfield destination of an asm.
struct S {
  int a : 4;
};

void t14(struct S *P) {
  __asm__("abc %0" : "=r"(P->a) );
}


// PR4938
int t16() {
  int a,b;
  asm ( "nop;"
       :"=%c" (a)
       : "r" (b)
       );
  return 0;
}

// PR6475
void t17() {
  int i;
  __asm__ ( "nop": "=m"(i));
  
// CHECK: @t17()
// CHECK: call void asm "nop", "=*m,
}

// <rdar://problem/6841383>
int t18(unsigned data) {
  int a, b;
  
  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
// CHECK: t18(i32
// CHECK: = call {{.*}}asm "xyz"
// CHECK-NEXT: extractvalue
// CHECK-NEXT: extractvalue
}


// PR6780
int t19(unsigned data) {
  int a, b;
  
  asm("x{abc|def|ghi}z" :"=r"(a): "r"(data));
  return a + b;
  // CHECK: t19(i32
  // CHECK: = call {{.*}}asm "x$(abc$|def$|ghi$)z"
}

