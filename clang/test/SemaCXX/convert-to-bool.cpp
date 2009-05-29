// RUN: clang-cc -fsyntax-only -verify %s 
struct ConvToBool {
  operator bool() const;
};

struct ConvToInt {
  operator int();
};

struct ExplicitConvToBool {
  explicit operator bool(); // expected-warning{{explicit conversion functions are a C++0x extension}}
};

void test_conv_to_bool(ConvToBool ctb, ConvToInt cti, ExplicitConvToBool ecb) {
  if (ctb) { }
  if (cti) { }
  if (ecb) { }
  for (; ctb; ) { }
  for (; cti; ) { }
  for (; ecb; ) { }
  while (ctb) { };
  while (cti) { }
  while (ecb) { }
  do { } while (ctb);
  do { } while (cti);
  do { } while (ecb);

  if (!ctb) { }
  if (!cti) { }
  if (!ecb) { }

  bool b1 = !ecb;
  if (ctb && ecb) { }
  bool b2 = ctb && ecb;
  if (ctb || ecb) { }
  bool b3 = ctb || ecb;
}

void accepts_bool(bool) { } // expected-note{{candidate function}}

struct ExplicitConvToRef {
  explicit operator int&(); // expected-warning{{explicit conversion functions are a C++0x extension}}
};

void test_explicit_bool(ExplicitConvToBool ecb) {
  bool b1(ecb); // okay
  bool b2 = ecb; // expected-error{{incompatible type initializing 'struct ExplicitConvToBool', expected 'bool'}}
  accepts_bool(ecb); // expected-error{{no matching function for call to}}
}

void test_explicit_conv_to_ref(ExplicitConvToRef ecr) {
  int& i1 = ecr; // expected-error{{non-const lvalue reference to type 'int' cannot be initialized with a value of type 'struct ExplicitConvToRef'}}
  int& i2(ecr); // okay
}

struct A { };
struct B { };
struct C {
  explicit operator A&(); // expected-warning{{explicit conversion functions are a C++0x extension}}
  operator B&();
};

void test_copy_init_conversions(C c) {
  A &a = c; // expected-error{{non-const lvalue reference to type 'struct A' cannot be initialized with a value of type 'struct C'}}
  B &b = b; // okay
}

