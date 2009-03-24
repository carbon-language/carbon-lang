// RUN: clang-cc -fsyntax-only -verify %s

struct SimpleValueInit {
  int i;
};

struct InitViaConstructor {
  InitViaConstructor(int i = 7);
};

// FIXME: error messages for implicitly-declared special member
// function candidates are very poor
struct NoValueInit { // expected-note 2 {{candidate function}} 
  NoValueInit(int i, int j); // expected-note 2 {{candidate function}}
};

void test_cxx_functional_value_init() {
  (void)SimpleValueInit();
  (void)InitViaConstructor();
  (void)NoValueInit(); // expected-error{{no matching constructor for initialization}}
}

void test_cxx_function_cast_multi() { 
  (void)NoValueInit(0, 0);
  (void)NoValueInit(0, 0, 0); // expected-error{{no matching constructor for initialization}}
  (void)int(1, 2); // expected-error{{function-style cast to a builtin type can only take one argument}}
}
