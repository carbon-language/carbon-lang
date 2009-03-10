// RUN: clang %s -arch=i386 -verify -fsyntax-only

void
f()
{
  int i;

  asm ("foo\n" : : "a" (i + 2));
  asm ("foo\n" : : "a" (f())); // expected-error {{invalid type 'void' in asm input}}
  
  asm ("foo\n" : "=a" (f())); // expected-error {{invalid lvalue in asm output}}
  asm ("foo\n" : "=a" (i + 2)); // expected-error {{invalid lvalue in asm output}}

  asm ("foo\n" : [symbolic_name] "=a" (i) : "[symbolic_name]" (i));
  asm ("foo\n" : "=a" (i) : "[" (i)); // expected-error {{invalid input constraint '[' in asm}}
  asm ("foo\n" : "=a" (i) : "[foo" (i)); // expected-error {{invalid input constraint '[foo' in asm}}
  asm ("foo\n" : "=a" (i) : "[symbolic_name]" (i)); // expected-error {{invalid input constraint '[symbolic_name]' in asm}}
}

void
clobbers()
{
  asm ("nop" : : : "ax", "#ax", "%ax");
  asm ("nop" : : : "eax", "rax", "ah", "al");
  asm ("nop" : : : "0", "%0", "#0");
  asm ("nop" : : : "foo"); // expected-error {{unknown register name 'foo' in asm}}
  asm ("nop" : : : "52");
  asm ("nop" : : : "53"); // expected-error {{unknown register name '53' in asm}}
  asm ("nop" : : : "-1"); // expected-error {{unknown register name '-1' in asm}}
  asm ("nop" : : : "+1"); // expected-error {{unknown register name '+1' in asm}}
}

// rdar://6094010
void test3() {
  int x;
  asm(L"foo" : "=r"(x)); // expected-error {{wide string}}
  asm("foo" : L"=r"(x)); // expected-error {{wide string}}
}

// <rdar://problem/6156893>
void test4(const volatile void *addr)
{
    asm ("nop" : : "r"(*addr)); // expected-error {{invalid type 'void const volatile' in asm input for constraint 'r'}}
    asm ("nop" : : "m"(*addr));

    asm ("nop" : : "r"(test4(addr))); // expected-error {{invalid type 'void' in asm input for constraint 'r'}}
    asm ("nop" : : "m"(test4(addr))); // expected-error {{invalid lvalue in asm input for constraint 'm'}}

    asm ("nop" : : "m"(f())); // expected-error {{invalid lvalue in asm input for constraint 'm'}}
}

// <rdar://problem/6512595>
void test5() {
  asm("nop" : : "X" (8)); 
}

// PR3385
void test6(long i) {
  asm("nop" : : "er"(i));
}

void test7() {
  asm("%!");   // simple asm string, %! is not an error.   
  asm("%!" : );   // expected-error {{invalid % escape in inline assembly string}}
}
