// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -verify %s

struct __attribute__((capability("thread role"))) ThreadRole {};
struct __attribute__((shared_capability("mutex"))) Mutex {};
struct NotACapability {};

int Test1 __attribute__((capability("test1")));  // expected-error {{'capability' attribute only applies to structs}}
int Test2 __attribute__((shared_capability("test2"))); // expected-error {{'shared_capability' attribute only applies to structs}}

struct __attribute__((capability(12))) Test3 {}; // expected-error {{'capability' attribute requires a string}}
struct __attribute__((shared_capability(Test2))) Test4 {}; // expected-error {{'shared_capability' attribute requires a string}}

struct __attribute__((capability)) Test5 {}; // expected-error {{'capability' attribute takes one argument}}
struct __attribute__((shared_capability("test1", 12))) Test6 {}; // expected-error {{'shared_capability' attribute takes one argument}}

struct NotACapability BadCapability;
struct ThreadRole GUI, Worker;
void Func1(void) __attribute__((requires_capability(GUI))) {}
void Func2(void) __attribute__((requires_shared_capability(Worker))) {}

void Func3(void) __attribute__((requires_capability)) {}  // expected-error {{'requires_capability' attribute takes at least 1 argument}}
void Func4(void) __attribute__((requires_shared_capability)) {}  // expected-error {{'requires_shared_capability' attribute takes at least 1 argument}}

void Func5(void) __attribute__((requires_capability(1))) {}  // expected-warning {{'requires_capability' attribute requires arguments that are class type or point to class type}}
void Func6(void) __attribute__((requires_shared_capability(BadCapability))) {}  // expected-warning {{'requires_shared_capability' attribute requires arguments whose type is annotated with 'capability' attribute; type here is 'struct NotACapability'}}
