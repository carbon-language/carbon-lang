// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
typedef char char16 __attribute__ ((__vector_size__ (16)));
typedef long long longlong16 __attribute__ ((__vector_size__ (16)));
typedef char char16_e __attribute__ ((__ext_vector_type__ (16)));
typedef long long longlong16_e __attribute__ ((__ext_vector_type__ (2)));

#if 1
// Test overloading and function calls with vector types.
void f0(char16);

void f0_test(char16 c16, longlong16 ll16, char16_e c16e, longlong16_e ll16e) {
  f0(c16);
  f0(ll16);
  f0(c16e);
  f0(ll16e);
}

int &f1(char16); // expected-note 2{{candidate function}}
float &f1(longlong16); // expected-note 2{{candidate function}}

void f1_test(char16 c16, longlong16 ll16, char16_e c16e, longlong16_e ll16e) {
  int &ir1 = f1(c16);
  float &fr1 = f1(ll16);
  f1(c16e); // expected-error{{call to 'f1' is ambiguous}}
  f1(ll16e); // expected-error{{call to 'f1' is ambiguous}}
}

void f2(char16_e); // expected-note{{no known conversion from 'longlong16_e' to 'char16_e' for 1st argument}}

void f2_test(char16 c16, longlong16 ll16, char16_e c16e, longlong16_e ll16e) {
  f2(c16);
  f2(ll16);
  f2(c16e);
  f2(ll16e); // expected-error{{no matching function}}
  f2('a');
  f2(17);
}
#endif

// Test the conditional operator with vector types.
void conditional(bool Cond, char16 c16, longlong16 ll16, char16_e c16e, 
                 longlong16_e ll16e) {
  // Conditional operators with the same type.
  __typeof__(Cond? c16 : c16) *c16p1 = &c16;
  __typeof__(Cond? ll16 : ll16) *ll16p1 = &ll16;
  __typeof__(Cond? c16e : c16e) *c16ep1 = &c16e;
  __typeof__(Cond? ll16e : ll16e) *ll16ep1 = &ll16e;

  // Conditional operators with similar types.
  __typeof__(Cond? c16 : c16e) *c16ep2 = &c16e;
  __typeof__(Cond? c16e : c16) *c16ep3 = &c16e;
  __typeof__(Cond? ll16 : ll16e) *ll16ep2 = &ll16e;
  __typeof__(Cond? ll16e : ll16) *ll16ep3 = &ll16e;

  // Conditional operators with incompatible types.
  (void)(Cond? c16 : ll16); // expected-error{{can't convert between vector values}}
  (void)(Cond? ll16e : c16e); // expected-error{{can't convert between vector values}}
  (void)(Cond? ll16e : c16); // expected-error{{can't convert between vector values}}
}

// Test C++ cast'ing of vector types.
void casts(longlong16 ll16, longlong16_e ll16e) {
  // C-style casts.
  (void)(char16)ll16;
  (void)(char16_e)ll16;
  (void)(longlong16)ll16;
  (void)(longlong16_e)ll16;
  (void)(char16)ll16e;
  (void)(char16_e)ll16e;
  (void)(longlong16)ll16e;
  (void)(longlong16_e)ll16e;

  // Function-style casts.
  (void)char16(ll16);
  (void)char16_e(ll16);
  (void)longlong16(ll16);
  (void)longlong16_e(ll16);
  (void)char16(ll16e);
  (void)char16_e(ll16e);
  (void)longlong16(ll16e);
  (void)longlong16_e(ll16e);

  // static_cast
  (void)static_cast<char16>(ll16);
  (void)static_cast<char16_e>(ll16);
  (void)static_cast<longlong16>(ll16);
  (void)static_cast<longlong16_e>(ll16);
  (void)static_cast<char16>(ll16e);
  (void)static_cast<char16_e>(ll16e); // expected-error{{static_cast from 'longlong16_e' to 'char16_e' is not allowed}}
  (void)static_cast<longlong16>(ll16e);
  (void)static_cast<longlong16_e>(ll16e);

  // reinterpret_cast
  (void)reinterpret_cast<char16>(ll16);
  (void)reinterpret_cast<char16_e>(ll16);
  (void)reinterpret_cast<longlong16>(ll16);
  (void)reinterpret_cast<longlong16_e>(ll16);
  (void)reinterpret_cast<char16>(ll16e);
  (void)reinterpret_cast<char16_e>(ll16e);
  (void)reinterpret_cast<longlong16>(ll16e);
  (void)reinterpret_cast<longlong16_e>(ll16e);
}
