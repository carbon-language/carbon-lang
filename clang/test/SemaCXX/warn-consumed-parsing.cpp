// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -std=c++11 %s

#define CONSUMES                  __attribute__ ((consumes))
#define TESTS_UNCONSUMED          __attribute__ ((tests_unconsumed))
#define CALLABLE_WHEN_UNCONSUMED  __attribute__ ((callable_when_unconsumed))

class AttrTester0 {
  void Consumes(void)        __attribute__ ((consumes(42))); // expected-error {{attribute takes no arguments}}
  bool TestsUnconsumed(void) __attribute__ ((tests_unconsumed(42))); // expected-error {{attribute takes no arguments}}
  void CallableWhenUnconsumed(void) 
    __attribute__ ((callable_when_unconsumed(42))); // expected-error {{attribute takes no arguments}}
};

int var0 CONSUMES; // expected-warning {{'consumes' attribute only applies to methods}}
int var1 TESTS_UNCONSUMED; // expected-warning {{'tests_unconsumed' attribute only applies to methods}}
int var2 CALLABLE_WHEN_UNCONSUMED; // expected-warning {{'callable_when_unconsumed' attribute only applies to methods}}

void function0(void) CONSUMES; // expected-warning {{'consumes' attribute only applies to methods}}
void function1(void) TESTS_UNCONSUMED; // expected-warning {{'tests_unconsumed' attribute only applies to methods}}
void function2(void) CALLABLE_WHEN_UNCONSUMED; // expected-warning {{'callable_when_unconsumed' attribute only applies to methods}}

class AttrTester1 {
  void consumes(void)        CONSUMES;
  bool testsUnconsumed(void) TESTS_UNCONSUMED;
};

class AttrTester2 {
  void callableWhenUnconsumed(void) CALLABLE_WHEN_UNCONSUMED;
};
