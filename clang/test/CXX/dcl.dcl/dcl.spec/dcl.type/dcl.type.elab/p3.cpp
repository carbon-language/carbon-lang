// RUN: clang-cc -fsyntax-only -verify %s

class A {}; // expected-note 3 {{previous use is here}}

void a1(struct A);
void a2(class A);
void a3(union A); // expected-error {{use of 'A' with tag type that does not match previous declaration}}
void a4(enum A); // expected-error {{use of 'A' with tag type that does not match previous declaration}}

class A1 {
  friend struct A;
  friend class A;
  friend union A; // expected-error {{use of 'A' with tag type that does not match previous declaration}}

  // FIXME: a better error would be something like 'enum types cannot be friends'
  friend enum A; // expected-error {{ISO C++ forbids forward references to 'enum' types}}
};

template <class T> struct B { // expected-note {{previous use is here}}
  class Member {}; // expected-note 2 {{previous use is here}}
};

template <> class B<int> {
  // no type Member
};

template <> struct B<A> {
  // FIXME: the error here should be associated with the use at "void foo..."
  union Member { // expected-note 4 {{previous use is here}} expected-error {{tag type that does not match previous declaration}}
    void* a;
  };
};

void b1(struct B<float>);
void b2(class B<float>);
void b3(union B<float>); // expected-error {{use of 'B<float>' with tag type that does not match previous declaration}}
//void b4(enum B<float>); // this just doesn't parse; you can't template an enum directly

void c1(struct B<float>::Member);
void c2(class B<float>::Member);
void c3(union B<float>::Member); // expected-error {{use of 'Member' with tag type that does not match previous declaration}}
void c4(enum B<float>::Member); // expected-error {{use of 'Member' with tag type that does not match previous declaration}}

void d1(struct B<int>::Member); // expected-error {{'Member' does not name a tag member in the specified scope}}
void d2(class B<int>::Member); // expected-error {{'Member' does not name a tag member in the specified scope}}
void d3(union B<int>::Member); // expected-error {{'Member' does not name a tag member in the specified scope}}
void d4(enum B<int>::Member); // expected-error {{'Member' does not name a tag member in the specified scope}}

void e1(struct B<A>::Member); // expected-error {{use of 'Member' with tag type that does not match previous declaration}}
void e2(class B<A>::Member); // expected-error {{use of 'Member' with tag type that does not match previous declaration}}
void e3(union B<A>::Member);
void e4(enum B<A>::Member); // expected-error {{use of 'Member' with tag type that does not match previous declaration}}

template <class T> struct C {
  void foo(class B<T>::Member); // expected-error{{no type named 'Member' in 'B<int>'}}
};

C<float> f1;
C<int> f2; // expected-note {{in instantiation of template class}}
C<A> f3; // expected-note {{in instantiation of template class}}
