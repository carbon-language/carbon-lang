// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

struct A {
  virtual void a(); // expected-note{{overridden virtual function is here}}
  virtual void b() = delete; // expected-note{{overridden virtual function is here}}
};

struct B: A {
  virtual void a() = delete; // expected-error{{deleted function 'a' cannot override a non-deleted function}}
  virtual void b(); // expected-error{{non-deleted function 'b' cannot override a deleted function}}
};

struct C: A {
  virtual void a();
  virtual void b() = delete;
};

struct E;
struct F;
struct G;
struct H;
struct D {
  virtual E &operator=(const E &); // expected-note {{here}}
  virtual F &operator=(const F &);
  virtual G &operator=(G&&);
  virtual H &operator=(H&&); // expected-note {{here}}
  friend struct F;

private:
  D &operator=(const D&) = default;
  D &operator=(D&&) = default;
  virtual ~D(); // expected-note 2{{here}}
};
struct E : D {}; // expected-error {{deleted function '~E' cannot override a non-deleted function}} \
                 // expected-error {{deleted function 'operator=' cannot override a non-deleted function}}
struct F : D {};
// No move ctor here, because it would be deleted.
struct G : D {}; // expected-error {{deleted function '~G' cannot override a non-deleted function}}
struct H : D {
  H &operator=(H&&) = default; // expected-error {{deleted function 'operator=' cannot override a non-deleted function}}
  ~H();
};
