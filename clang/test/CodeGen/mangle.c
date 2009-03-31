// RUN: clang-cc -arch i386 -emit-llvm -o %t %s &&
// RUN: grep '@_Z2f0i' %t &&
// RUN: grep '@_Z2f0l' %t &&

// Make sure we mangle overloadable, even in C system headers.

# 1 "somesystemheader.h" 1 3 4
void __attribute__((__overloadable__)) f0(int a) {}
void __attribute__((__overloadable__)) f0(long b) {}



// These should get merged.
void foo() __asm__("bar");
void foo2() __asm__("bar");

// RUN: grep '@"\\01foo"' %t &&
// RUN: grep '@"\\01bar"' %t

int nux __asm__("foo");
extern float nux2 __asm__("foo");

int test() { 
  foo();
  foo2();
  
  return nux + nux2;
}


// Function becomes a variable.
void foo3() __asm__("var");

void test2() {
  foo3();
}
int foo4 __asm__("var") = 4;


// Variable becomes a function
extern int foo5 __asm__("var2");

void test3() {
  foo5 = 1;
}

void foo6() __asm__("var2");
void foo6() {
}



int foo7 __asm__("foo7") __attribute__((used));
float foo8 __asm__("foo7") = 42;
