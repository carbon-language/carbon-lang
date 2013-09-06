// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -std=c++11 %s

#define CALLABLE_WHEN_UNCONSUMED __attribute__ ((callable_when_unconsumed))
#define CONSUMABLE(state)        __attribute__ ((consumable(state)))
#define CONSUMES                 __attribute__ ((consumes))
#define RETURN_TYPESTATE(state)  __attribute__ ((return_typestate(state)))
#define TESTS_UNCONSUMED         __attribute__ ((tests_unconsumed))

// FIXME: This test is here because the warning is issued by the Consumed
//        analysis, not SemaDeclAttr.  The analysis won't run after an error
//        has been issued.  Once the attribute propagation for template
//        instantiation has been fixed, this can be moved somewhere else and the
//        definition can be removed.
int returnTypestateForUnconsumable() RETURN_TYPESTATE(consumed); // expected-warning {{return state set for an unconsumable type 'int'}}
int returnTypestateForUnconsumable() {
  return 0;
}

class AttrTester0 {
  void consumes()        __attribute__ ((consumes(42))); // expected-error {{attribute takes no arguments}}
  bool testsUnconsumed() __attribute__ ((tests_unconsumed(42))); // expected-error {{attribute takes no arguments}}
  void callableWhenUnconsumed() __attribute__ ((callable_when_unconsumed(42))); // expected-error {{attribute takes no arguments}}
};

int var0 CONSUMES; // expected-warning {{'consumes' attribute only applies to methods}}
int var1 TESTS_UNCONSUMED; // expected-warning {{'tests_unconsumed' attribute only applies to methods}}
int var2 CALLABLE_WHEN_UNCONSUMED; // expected-warning {{'callable_when_unconsumed' attribute only applies to methods}}
int var3 CONSUMABLE(consumed); // expected-warning {{'consumable' attribute only applies to classes}}
int var4 RETURN_TYPESTATE(consumed); // expected-warning {{'return_typestate' attribute only applies to functions}}

void function0() CONSUMES; // expected-warning {{'consumes' attribute only applies to methods}}
void function1() TESTS_UNCONSUMED; // expected-warning {{'tests_unconsumed' attribute only applies to methods}}
void function2() CALLABLE_WHEN_UNCONSUMED; // expected-warning {{'callable_when_unconsumed' attribute only applies to methods}}
void function3() CONSUMABLE(consumed); // expected-warning {{'consumable' attribute only applies to classes}}

class CONSUMABLE(unknown) AttrTester1 {
  void callableWhenUnconsumed() CALLABLE_WHEN_UNCONSUMED;
  void consumes()               CONSUMES;
  bool testsUnconsumed()        TESTS_UNCONSUMED;
};

AttrTester1 returnTypestateTester0() RETURN_TYPESTATE(not_a_state); // expected-warning {{unknown consumed analysis state 'not_a_state'}}
AttrTester1 returnTypestateTester1() RETURN_TYPESTATE(42); // expected-error {{'return_typestate' attribute requires an identifier}}

class AttrTester2 {
  void callableWhenUnconsumed() CALLABLE_WHEN_UNCONSUMED; // expected-warning {{consumed analysis attribute is attached to member of class 'AttrTester2' which isn't marked as consumable}}
  void consumes()               CONSUMES; // expected-warning {{consumed analysis attribute is attached to member of class 'AttrTester2' which isn't marked as consumable}}
  bool testsUnconsumed()        TESTS_UNCONSUMED; // expected-warning {{consumed analysis attribute is attached to member of class 'AttrTester2' which isn't marked as consumable}}
};

class CONSUMABLE(42) AttrTester3; // expected-error {{'consumable' attribute requires an identifier}}
