// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

int align_illegal [[align(3)]]; //expected-error {{requested alignment is not a power of 2}}
char align_big [[align(int)]];
int align_small [[align(1)]]; // FIXME: this should be rejected
int align_multiple [[align(1), align(8), align(1)]];

struct align_member {
  int member [[align(8)]];
};

static_assert(alignof(align_big) == alignof(int), "k's alignment is wrong");
static_assert(alignof(align_small) == 1, "j's alignment is wrong");
static_assert(alignof(align_multiple) == 8, "l's alignment is wrong");
static_assert(alignof(align_member) == 8, "quuux's alignment is wrong");
static_assert(sizeof(align_member) == 8, "quuux's size is wrong");

int bc_fail [[base_check]]; // expected-error {{'base_check' attribute only applies to class types}}
int hiding_fail [[hiding]]; // expected-error {{'hiding' attribute only applies to member types}}
int override_fail [[override]]; // expected-error {{'override' attribute only applies to virtual method types}}

struct base {
  virtual void function();
  virtual void other_function();
};

struct [[base_check, base_check]] bc : base { // expected-error {{'base_check' attribute cannot be repeated}}
};
