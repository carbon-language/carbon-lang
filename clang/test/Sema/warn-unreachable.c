// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks -Wunreachable-code -Wno-unused-value -Wno-covered-switch-default -I %S/Inputs

#include "warn-unreachable.h"

int halt() __attribute__((noreturn));
int live();
int dead();

void test1() {
  goto c;
  d:
  goto e;       // expected-warning {{will never be executed}}
  c: ;
  int i;
  return;
  goto b;        // expected-warning {{will never be executed}}
  goto a;        // expected-warning {{will never be executed}}
  b:
  i = 1;
  a:
  i = 2;
  goto f;
  e:
  goto d;
  f: ;
}

void test2() {
  int i;
  switch (live()) {
  case 1:
    halt(),
      dead();   // expected-warning {{will never be executed}}

  case 2:
    live(), halt(),
      dead();   // expected-warning {{will never be executed}}

  case 3:
  live()
    +           // expected-warning {{will never be executed}}
    halt();
  dead();

  case 4:
  a4:
    live(),
      halt();
    goto a4;    // expected-warning {{will never be executed}}

  case 5:
    goto a5;
  c5:
    dead();     // expected-warning {{will never be executed}}
    goto b5;
  a5:
    live(),
      halt();
  b5:
    goto c5;

  case 6:
    if (live())
      goto e6;
    live(),
      halt();
  d6:
    dead();     // expected-warning {{will never be executed}}
    goto b6;
  c6:
    dead();
    goto b6;
  e6:
    live(),
      halt();
  b6:
    goto c6;
  case 7:
    halt()
      +
      dead();   // expected-warning {{will never be executed}}
    -           // expected-warning {{will never be executed}}
      halt();
  case 8:
    i
      +=        // expected-warning {{will never be executed}}
      halt();
  case 9:
    halt()
      ?         // expected-warning {{will never be executed}}
      dead() : dead();
  case 10:
    (           // expected-warning {{will never be executed}}
      float)halt();
  case 11: {
    int a[5];
    live(),
      a[halt()
        ];      // expected-warning {{will never be executed}}
  }
  }
}

enum Cases { C1, C2, C3 };
int test_enum_cases(enum Cases C) {
  switch (C) {
    case C1:
    case C2:
    case C3:
      return 1;
    default: {
      int i = 0; // no-warning
      ++i;
      return i;
    }
  }  
}

// Handle unreachable code triggered by macro expansions.
void __myassert_rtn(const char *, const char *, int, const char *) __attribute__((__noreturn__));

#define myassert(e) \
    (__builtin_expect(!(e), 0) ? __myassert_rtn(__func__, __FILE__, __LINE__, #e) : (void)0)

void test_assert() {
  myassert(0 && "unreachable");
  return; // no-warning
}

// Test case for PR 9774.  Tests that dead code in macros aren't warned about.
#define MY_MAX(a,b)     ((a) >= (b) ? (a) : (b))
void PR9774(int *s) {
    for (int i = 0; i < MY_MAX(2, 3); i++) // no-warning
        s[i] = 0;
}

// Test case for <rdar://problem/11005770>.  We should treat code guarded
// by 'x & 0' and 'x * 0' as unreachable.
int calledFun();
void test_mul_and_zero(int x) {
  if (x & 0) calledFun(); // expected-warning {{will never be executed}}
  if (0 & x) calledFun(); // expected-warning {{will never be executed}}
  if (x * 0) calledFun(); // expected-warning {{will never be executed}}
  if (0 * x) calledFun(); // expected-warning {{will never be executed}}
}

void raze() __attribute__((noreturn));
void warn_here();

int test_break_preceded_by_noreturn(int i) {
  switch (i) {
    case 1:
      raze();
      break; // no-warning
    case 2:
      raze();
      break; // no-warning
      warn_here(); // expected-warning {{will never be executed}}
    case 3:
      return 1;
      break; // expected-warning {{will never be executed}}
    default:
      break;
  }
  return i;
}

// Don't warn about unreachable 'default' cases, as that is covered
// by -Wcovered-switch-default.
typedef enum { Value1 = 1 } MyEnum;
void unreachable_default(MyEnum e) {
  switch (e) {
  case Value1:
    calledFun();
    break;
  case 2: // expected-warning {{case value not in enumerated type 'MyEnum'}}
    calledFun();
    break;
  default:
    calledFun(); // no-warning
    break;
  }
}
void unreachable_in_default(MyEnum e) {
  switch (e) {
  default:
    raze();
    calledFun(); // expected-warning {{will never be executed}}
    break;
  }
}

// Don't warn about trivial dead returns.
int trivial_dead_return() {
  raze();
  // Use the '()' to test that we unwrap such stuff
  // when looking for dead code.
  return ((0)); // no-warning
}

MyEnum trival_dead_return_enum() {
  raze();
  return Value1; // no-warning
}

// Test unreachable code depending on configuration values
#define CONFIG_CONSTANT 1
int test_config_constant(int x) {
  if (!CONFIG_CONSTANT) {
    calledFun(); // no-warning
    return 1;
  }
  if (!1) {
    calledFun(); // expected-warning {{will never be executed}}
    return 1;
  }
  if (sizeof(int) > sizeof(char)) {
    calledFun(); // no-warning
    return 1;
  }
  if (x > 10)
    return CONFIG_CONSTANT ? calledFun() : calledFun(); // no-warning
  else
    return 1 ?
      calledFun() :
      calledFun(); // expected-warning {{will never be executed}}
}

int sizeof_int() {
  if (sizeof(long) == sizeof(int))
    return 1; // no-warning
  if (sizeof(long) != sizeof(int))
    return 0; // no-warning
  return 2; // no-warning
}

