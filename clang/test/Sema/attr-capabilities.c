// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -verify %s

struct __attribute__((capability("role"))) ThreadRole {};
struct __attribute__((shared_capability("mutex"))) Mutex {};
struct NotACapability {};

// Test an invalid capability name
struct __attribute__((capability("wrong"))) IncorrectName {}; // expected-warning {{invalid capability name 'wrong'; capability name must be 'mutex' or 'role'}}

int Test1 __attribute__((capability("test1")));  // expected-error {{'capability' attribute only applies to structs}}
int Test2 __attribute__((shared_capability("test2"))); // expected-error {{'shared_capability' attribute only applies to structs}}
int Test3 __attribute__((acquire_capability("test3")));  // expected-error {{'acquire_capability' attribute only applies to functions}}
int Test4 __attribute__((try_acquire_capability("test4"))); // expected-error {{'try_acquire_capability' attribute only applies to functions}}
int Test5 __attribute__((release_capability("test5"))); // expected-error {{'release_capability' attribute only applies to functions}}

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

void Func7(void) __attribute__((assert_capability(GUI))) {}
void Func8(void) __attribute__((assert_shared_capability(GUI))) {}

void Func9(void) __attribute__((assert_capability())) {} // expected-error {{'assert_capability' attribute takes one argument}}
void Func10(void) __attribute__((assert_shared_capability())) {} // expected-error {{'assert_shared_capability' attribute takes one argument}}

void Func11(void) __attribute__((acquire_capability(GUI))) {}
void Func12(void) __attribute__((acquire_shared_capability(GUI))) {}

void Func13(void) __attribute__((acquire_capability())) {} // expected-error {{'acquire_capability' attribute takes at least 1 argument}}
void Func14(void) __attribute__((acquire_shared_capability())) {} // expected-error {{'acquire_shared_capability' attribute takes at least 1 argument}}

void Func15(void) __attribute__((release_capability(GUI))) {}
void Func16(void) __attribute__((release_shared_capability(GUI))) {}
void Func17(void) __attribute__((release_generic_capability(GUI))) {}

void Func18(void) __attribute__((release_capability())) {} // expected-error {{'release_capability' attribute takes at least 1 argument}}
void Func19(void) __attribute__((release_shared_capability())) {} // expected-error {{'release_shared_capability' attribute takes at least 1 argument}}
void Func20(void) __attribute__((release_generic_capability())) {} // expected-error {{'release_generic_capability' attribute takes at least 1 argument}}

void Func21(void) __attribute__((try_acquire_capability(1))) {}
void Func22(void) __attribute__((try_acquire_shared_capability(1))) {}

void Func23(void) __attribute__((try_acquire_capability(1, GUI))) {}
void Func24(void) __attribute__((try_acquire_shared_capability(1, GUI))) {}

void Func25(void) __attribute__((try_acquire_capability())) {} // expected-error {{'try_acquire_capability' attribute takes at least 1 argument}}
void Func26(void) __attribute__((try_acquire_shared_capability())) {} // expected-error {{'try_acquire_shared_capability' attribute takes at least 1 argument}}