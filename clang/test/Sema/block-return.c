// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

typedef void (^CL)(void);

CL foo() {
  short y;
  short (^add1)(void) = ^{ return y+1; }; // expected-error {{incompatible block pointer types initializing 'int (^)(void)', expected 'short (^)(void)'}}

  CL X = ^{
    if (2)
      return;
    return 1;  // expected-error {{void block should not return a value}}
  };

  int (^Y) (void)  = ^{
    if (3)
      return 1;
    else
      return; // expected-error {{non-void block should return a value}}
  };

  char *(^Z)(void) = ^{
    if (3)
      return "";
    else
      return (char*)0;
  };

  double (^A)(void) = ^ { // expected-error {{incompatible block pointer types initializing 'float (^)(void)', expected 'double (^)(void)'}}
    if (1)
      return (float)1.0;
    else
      if (2)
        return (double)2.0;
    return 1;
  };
  char *(^B)(void) = ^{
    if (3)
      return "";
    else
      return 2; // expected-warning {{incompatible integer to pointer conversion returning 'int', expected 'char *'}}
  };

  return ^{ return 1; }; // expected-error {{incompatible block pointer types returning 'int (^)(void)', expected 'CL'}}
}

typedef int (^CL2)(void);

CL2 foo2() {
  return ^{ return 1; };
}

typedef unsigned int * uintptr_t;
typedef char Boolean;
typedef int CFBasicHash;

#define INVOKE_CALLBACK2(P, A, B) (P)(A, B)

typedef struct {
    Boolean (^isEqual)(const CFBasicHash *, uintptr_t stack_value_or_key1, uintptr_t stack_value_or_key2, Boolean is_key);
} CFBasicHashCallbacks;

int foo3() {
    CFBasicHashCallbacks cb;
    
    Boolean (*value_equal)(uintptr_t, uintptr_t) = 0;
            
    cb.isEqual = ^(const CFBasicHash *table, uintptr_t stack_value_or_key1, uintptr_t stack_value_or_key2, Boolean is_key) {
      return (Boolean)(uintptr_t)INVOKE_CALLBACK2(value_equal, (uintptr_t)stack_value_or_key1, (uintptr_t)stack_value_or_key2);
    };
}

static int funk(char *s) {
  if (^{} == ((void*)0))
    return 1;
  else 
    return 0;
}
void next();
void foo4() {
  int (^xx)(const char *s) = ^(char *s) { return 1; }; // expected-error {{incompatible block pointer types initializing 'int (^)(char *)', expected 'int (^)(char const *)'}}
  int (*yy)(const char *s) = funk; // expected-warning {{incompatible pointer types initializing 'int (char *)', expected 'int (*)(char const *)'}}
  
  int (^nested)(char *s) = ^(char *str) { void (^nest)(void) = ^(void) { printf("%s\n", str); }; next(); return 1; }; // expected-warning{{implicitly declaring C library function 'printf' with type 'int (char const *, ...)'}} \
  // expected-note{{please include the header <stdio.h> or explicitly provide a declaration for 'printf'}}
}

typedef void (^bptr)(void);

bptr foo5(int j) {
  __block int i;
  if (j)
    return ^{ ^{ i=0; }(); };  // expected-error {{returning block that lives on the local stack}}
  return ^{ i=0; };  // expected-error {{returning block that lives on the local stack}}
  return (^{ i=0; });  // expected-error {{returning block that lives on the local stack}}
  return (void*)(^{ i=0; });  // expected-error {{returning block that lives on the local stack}}
}

int (*funcptr3[5])(long);
int sz8 = sizeof(^int (*[5])(long) {return funcptr3;}); // expected-error {{block declared as returning an array}}

void foo6() {
  int (^b)(int) __attribute__((noreturn));
  b = ^ (int i) __attribute__((noreturn)) { return 1; };  // expected-error {{block declared 'noreturn' should not return}}
  b(1);
  int (^c)(void) __attribute__((noreturn)) = ^ __attribute__((noreturn)) { return 100; }; // expected-error {{block declared 'noreturn' should not return}}
}


void foo7()
{
 const int (^BB) (void) = ^{ const int i = 1; return i; }; // OK 
 const int (^CC) (void)  = ^const int{ const int i = 1; return i; }; // OK

  int i;
  int (^FF) (void)  = ^{ return i; }; // OK
  int (^EE) (void)  = ^{ return i+1; }; // OK

  __block int j;
  int (^JJ) (void)  = ^{ return j; }; // OK
  int (^KK) (void)  = ^{ return j+1; }; // OK

  __block const int k;
  const int cint = 100;

  int (^MM) (void)  = ^{ return k; }; // expected-error {{incompatible block pointer types initializing 'int const (^)(void)', expected 'int (^)(void)'}}
  int (^NN) (void)  = ^{ return cint; }; // expected-error {{incompatible block pointer types initializing 'int const (^)(void)', expected 'int (^)(void)'}}

}


