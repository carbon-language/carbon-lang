// RUN: %clang_cc1 %s -fsyntax-only -verify

typedef unsigned __uint32_t;

#define __byte_swap_int_var(x) \
__extension__ ({ register __uint32_t __X = (x); \
   __asm ("bswap %0" : "+r" (__X)); \
   __X; })

int test(int _x) {
 return (__byte_swap_int_var(_x));
}

// PR2374
int test2() { return ({L:5;}); }
int test3() { return ({ {5;} }); }         // expected-error {{returning 'void' from a function with incompatible result type 'int'}}\
                                           // expected-warning {{expression result unused}}
int test4() { return ({ ({5;}); }); }
int test5() { return ({L1: L2: L3: 5;}); }
int test6() { return ({5;}); }
void test7() { ({5;}); }                   // expected-warning {{expression result unused}}

// PR3062
int test8[({10;})]; // expected-error {{statement expression not allowed at file scope}}

// PR3912
void test9(const void *P) {
  __builtin_prefetch(P);
}


void *test10() { 
bar:
  return &&bar;  // expected-warning {{returning address of label, which is local}}
}

// PR6034
void test11(int bit) {
  switch (bit)
  switch (env->fpscr)  // expected-error {{use of undeclared identifier 'env'}}
  {
  }
}

// rdar://3271964
enum Numbers { kOne,  kTwo,  kThree,  kFour};
int test12(enum Numbers num) {
  switch (num == kOne) {// expected-warning {{switch condition has boolean value}}
  default: 
  case kThree:
    break;
  }
}