// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://9584012

typedef struct {
	char *str;
} Class;

typedef union {
	Class *object;
} Instance __attribute__((transparent_union));

__attribute__((nonnull(1))) void Class_init(Instance this, char *str) {
	this.object->str = str;
}

int main(void) {
	Class *obj;
	Class_init(0, "Hello World"); // expected-warning {{null passed to a callee which requires a non-null argument}}
	Class_init(obj, "Hello World");
}

void foo(const char *str) __attribute__((nonnull("foo"))); // expected-error{{'nonnull' attribute requires parameter 1 to be an integer constant}}
void bar(int i) __attribute__((nonnull(1))); // expected-warning {{'nonnull' attribute only applies to pointer arguments}} expected-warning {{'nonnull' attribute applied to function with no pointer arguments}}

void baz(__attribute__((nonnull)) const char *str);
void baz2(__attribute__((nonnull(1))) const char *str); // expected-warning {{'nonnull' attribute when used on parameters takes no arguments}}
void baz3(__attribute__((nonnull)) int x); // expected-warning {{'nonnull' attribute only applies to pointer arguments}}

void test_baz() {
  baz(0); // expected-warning {{null passed to a callee which requires a non-null argument}}
  baz2(0); // no-warning
  baz3(0); // no-warning
}

void test_void_returns_nonnull(void) __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to return values that are pointers}}
int test_int_returns_nonnull(void) __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to return values that are pointers}}
void *test_ptr_returns_nonnull(void) __attribute__((returns_nonnull)); // no-warning

int i __attribute__((nonnull)); // expected-warning {{'nonnull' attribute only applies to functions, methods, and parameters}}
int j __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to functions and methods}}
void *test_no_fn_proto() __attribute__((returns_nonnull)); // no-warning
void *test_with_fn_proto(void) __attribute__((returns_nonnull)); // no-warning

__attribute__((returns_nonnull))
void *test_bad_returns_null(void) {
  return 0; // expected-warning {{null returned from function that requires a non-null return value}}
}

void PR18795(int (*g)(const char *h, ...) __attribute__((nonnull(1))) __attribute__((nonnull))) {
  g(0); // expected-warning{{null passed to a callee which requires a non-null argument}}
}
void PR18795_helper() {
  PR18795(0); // expected-warning{{null passed to a callee which requires a non-null argument}}
}


