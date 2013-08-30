// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -std=c++11 %s

#define CONSUMABLE                __attribute__ ((consumable))
#define CONSUMES                  __attribute__ ((consumes))
#define TESTS_UNCONSUMED          __attribute__ ((tests_unconsumed))
#define CALLABLE_WHEN_UNCONSUMED  __attribute__ ((callable_when_unconsumed))

class AttrTester0 {
  void Consumes()        __attribute__ ((consumes(42))); // expected-error {{attribute takes no arguments}}
  bool TestsUnconsumed() __attribute__ ((tests_unconsumed(42))); // expected-error {{attribute takes no arguments}}
  void CallableWhenUnconsumed() 
    __attribute__ ((callable_when_unconsumed(42))); // expected-error {{attribute takes no arguments}}
};

int var0 CONSUMES; // expected-warning {{'consumes' attribute only applies to methods}}
int var1 TESTS_UNCONSUMED; // expected-warning {{'tests_unconsumed' attribute only applies to methods}}
int var2 CALLABLE_WHEN_UNCONSUMED; // expected-warning {{'callable_when_unconsumed' attribute only applies to methods}}
int var3 CONSUMABLE; // expected-warning {{'consumable' attribute only applies to classes}}

void function0() CONSUMES; // expected-warning {{'consumes' attribute only applies to methods}}
void function1() TESTS_UNCONSUMED; // expected-warning {{'tests_unconsumed' attribute only applies to methods}}
void function2() CALLABLE_WHEN_UNCONSUMED; // expected-warning {{'callable_when_unconsumed' attribute only applies to methods}}
void function3() CONSUMABLE; // expected-warning {{'consumable' attribute only applies to classes}}

class CONSUMABLE AttrTester1 {
  void callableWhenUnconsumed() CALLABLE_WHEN_UNCONSUMED;
  void consumes()               CONSUMES;
  bool testsUnconsumed()        TESTS_UNCONSUMED;
};

class AttrTester2 {
  void callableWhenUnconsumed() CALLABLE_WHEN_UNCONSUMED; // expected-warning {{consumed analysis attribute is attached to class 'AttrTester2' which isn't marked as consumable}}
  void consumes()               CONSUMES; // expected-warning {{consumed analysis attribute is attached to class 'AttrTester2' which isn't marked as consumable}}
  bool testsUnconsumed()        TESTS_UNCONSUMED; // expected-warning {{consumed analysis attribute is attached to class 'AttrTester2' which isn't marked as consumable}}
};
