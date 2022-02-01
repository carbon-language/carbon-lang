// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only  -verify -std=c++11 %s

typedef int&& irr;
typedef irr& ilr_c1; // Collapses to int&
typedef int& ilr;
typedef ilr&& ilr_c2; // Collapses to int&

irr ret_irr() {
  return 0; // expected-warning {{returning reference to local temporary}}
}

struct not_int {};

int over(int&);
not_int over(int&&);

int over2(const int&);
not_int over2(int&&);

struct conv_to_not_int_rvalue {
  operator not_int &&();
};

typedef void (fun_type)();
void fun();
fun_type &&make_fun();

void f() {
  int &&virr1; // expected-error {{declaration of reference variable 'virr1' requires an initializer}}
  int &&virr2 = 0;
  int &&virr3 = virr2; // expected-error {{rvalue reference to type 'int' cannot bind to lvalue of type 'int'}}
  int i1 = 0;
  const double d1 = 0;
  const int ci1 = 1;
  int &&virr4 = i1; // expected-error {{rvalue reference to type 'int' cannot bind to lvalue of type 'int'}}
  int &&virr5 = ret_irr();
  int &&virr6 = static_cast<int&&>(i1);
  (void)static_cast<not_int &&>(i1); // expected-error {{reference to type 'not_int' could not bind to an lvalue of type 'int'}}
  (void)static_cast<int &&>(static_cast<int const&&>(i1)); // expected-error {{cannot cast from rvalue of type 'const int' to rvalue reference type 'int &&'}}
  (void)static_cast<int &&>(ci1);    // expected-error {{types are not compatible}}
  (void)static_cast<int &&>(d1);
  int i2 = over(i1);
  not_int ni1 = over(0);
  int i3 = over(virr2);
  not_int ni2 = over(ret_irr());

  int i4 = over2(i1);
  not_int ni3 = over2(0);

  ilr_c1 vilr1 = i1;
  ilr_c2 vilr2 = i1;

  conv_to_not_int_rvalue cnir;
  not_int &&ni4 = cnir;
  not_int &ni5 = cnir; // expected-error{{non-const lvalue reference to type 'not_int' cannot bind to a value of unrelated type 'conv_to_not_int_rvalue'}}
  not_int &&ni6 = conv_to_not_int_rvalue();

  fun_type &&fun_ref = fun; // works because functions are special
  fun_type &&fun_ref2 = make_fun(); // same
  fun_type &fun_lref = make_fun(); // also special

  try {
  } catch(int&&) { // expected-error {{cannot catch exceptions by rvalue reference}}
  }
}

int&& should_warn(int i) {
  return static_cast<int&&>(i); // expected-warning {{reference to stack memory associated with parameter 'i' returned}}
}
int&& should_not_warn(int&& i) {
  return static_cast<int&&>(i);
}


// Test the return dance. This also tests IsReturnCopyElidable.
struct MoveOnly {
  MoveOnly();
  MoveOnly(const MoveOnly&) = delete;	// expected-note 3{{explicitly marked deleted here}}
};

MoveOnly gmo;
MoveOnly returningNonEligible() {
  static MoveOnly mo;
  MoveOnly &r = mo;
  if (0) // Copy from global can't be elided
    return gmo; // expected-error {{call to deleted constructor}}
  else if (0) // Copy from local static can't be elided
    return mo; // expected-error {{call to deleted constructor}}
  else // Copy from reference can't be elided
    return r; // expected-error {{call to deleted constructor}}
}
