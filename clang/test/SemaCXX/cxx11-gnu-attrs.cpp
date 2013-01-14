// RUN: %clang -cc1 -std=c++11 -verify %s

// Error cases.

[[gnu::this_attribute_does_not_exist]] int unknown_attr;
// expected-warning@-1 {{unknown attribute 'this_attribute_does_not_exist' ignored}}
int [[gnu::unused]] attr_on_type;
// expected-warning@-1 {{attribute 'unused' ignored, because it is not attached to a declaration}}
int *[[gnu::unused]] attr_on_ptr;
// expected-warning@-1 {{attribute 'unused' ignored, because it cannot be applied to a type}}

// Valid cases.

void alias1() {}
void alias2 [[gnu::alias("_Z6alias1v")]] ();

[[gnu::aligned(8)]] int aligned;
void aligned_fn [[gnu::aligned(32)]] ();
struct [[gnu::aligned(8)]] aligned_struct {};

[[gnu::malloc, gnu::alloc_size(1,2)]] void *alloc_size(int a, int b);

void always_inline [[gnu::always_inline]] ();

__thread int tls_model [[gnu::tls_model("local-exec")]];

void cleanup(int *p) {
  int n [[gnu::cleanup(cleanup)]];
}

void deprecated1 [[gnu::deprecated]] (); // expected-note {{here}}
[[gnu::deprecated("custom message")]] void deprecated2(); // expected-note {{here}}
void deprecated3() {
  deprecated1(); // expected-warning {{deprecated}}
  deprecated2(); // expected-warning {{custom message}}
}

[[gnu::naked(1,2,3)]] void naked(); // expected-error {{takes no arguments}}

void nonnull [[gnu::nonnull]] (); // expected-warning {{applied to function with no pointer arguments}}

int noreturn [[gnu::noreturn]]; // expected-warning {{'noreturn' only applies to function types}}

struct [[gnu::packed]] packed { char c; int n; };
static_assert(sizeof(packed) == sizeof(char) + sizeof(int), "not packed");
