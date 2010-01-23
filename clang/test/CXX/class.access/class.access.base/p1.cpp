// RUN: %clang_cc1 -fsyntax-only -faccess-control -verify %s

// C++0x [class.access.base]p1(a):
//   If a class is declared to be a base class for another class using
//   the public access specifier, the public members of the base class
//   are accessible as public members of the derived class and protected
//   members of the base class are accessible as protected members of
//   the derived class.
namespace test0 {
  class Base {
  public: int pub; static int spub;
  protected: int prot; static int sprot; // expected-note 4 {{declared protected here}}
  private: int priv; static int spriv; // expected-note 8 {{declared private here}}
  };

  class Test : public Base {
    void test() {
      pub++;
      spub++;
      prot++;
      sprot++;
      priv++; // expected-error {{private member}}
      spriv++; // expected-error {{private member}}

      Base::pub++;
      Base::spub++;
      Base::prot++;
      Base::sprot++;
      Base::priv++; // expected-error {{private member}}
      Base::spriv++; // expected-error {{private member}}
    }
  };

  void test(Test *t) {
    t->pub++;
    t->spub++;
    t->prot++; // expected-error {{protected member}}
    t->sprot++; // expected-error {{protected member}}
    t->priv++; // expected-error {{private member}}
    t->spriv++; // expected-error {{private member}}

    t->Base::pub++;
    t->Base::spub++;
    t->Base::prot++; // expected-error {{protected member}}
    t->Base::sprot++; // expected-error {{protected member}}
    t->Base::priv++; // expected-error {{private member}}
    t->Base::spriv++; // expected-error {{private member}}
  }
}

// C++0x [class.access.base]p1(b):
//   If a class is declared to be a base class for another class using
//   the protected access specifier, the public and protected members
//   of the base class are accessible as protected members of the
//   derived class.
namespace test1 {
  class Base { // expected-note 6 {{constrained by protected inheritance}}
  public: int pub; static int spub; // expected-note 2 {{constrained by protected inheritance}}
  protected: int prot; static int sprot; // expected-note 4 {{declared protected here}}
  private: int priv; static int spriv; // expected-note 8 {{declared private here}}
  };

  class Test : protected Base {
    void test() {
      pub++;
      spub++;
      prot++;
      sprot++;
      priv++; // expected-error {{private member}}
      spriv++; // expected-error {{private member}}

      Base::pub++;
      Base::spub++;
      Base::prot++;
      Base::sprot++;
      Base::priv++; // expected-error {{private member}}
      Base::spriv++; // expected-error {{private member}}
    }
  };

  void test(Test *t) {
    t->pub++; // expected-error {{protected member}}
    t->spub++; // expected-error {{protected member}}
    t->prot++; // expected-error {{protected member}}
    t->sprot++; // expected-error {{protected member}}
    t->priv++; // expected-error {{private member}}
    t->spriv++; // expected-error {{private member}}

    // Two possible errors here: one for Base, one for the member
    t->Base::pub++; // expected-error {{protected member}}
    t->Base::spub++; // expected-error {{protected member}}
    t->Base::prot++; // expected-error 2 {{protected member}}
    t->Base::sprot++; // expected-error 2 {{protected member}}
    t->Base::priv++; // expected-error {{protected member}} expected-error {{private member}}
    t->Base::spriv++; // expected-error {{protected member}} expected-error {{private member}}
  }
}

// C++0x [class.access.base]p1(b):
//   If a class is declared to be a base class for another class using
//   the private access specifier, the public and protected members of
//   the base class are accessible as private members of the derived
//   class.
namespace test2 {
  class Base { //expected-note 6 {{constrained by private inheritance}}
  public:
    int pub; // expected-note {{constrained by private inheritance}}
    static int spub; // expected-note {{constrained by private inheritance}}
  protected:
    int prot; // expected-note {{constrained by private inheritance}} \
              // expected-note {{declared protected here}}
    static int sprot; // expected-note {{constrained by private inheritance}} \
                      // expected-note {{declared protected here}}
  private:
    int priv; // expected-note 4 {{declared private here}}
    static int spriv; // expected-note 4 {{declared private here}}
  };

  class Test : private Base { // expected-note 6 {{'private' inheritance specifier here}}
    void test() {
      pub++;
      spub++;
      prot++;
      sprot++;
      priv++; // expected-error {{private member}}
      spriv++; // expected-error {{private member}}

      Base::pub++;
      Base::spub++;
      Base::prot++;
      Base::sprot++;
      Base::priv++; // expected-error {{private member}}
      Base::spriv++; // expected-error {{private member}}
    }
  };

  void test(Test *t) {
    t->pub++; // expected-error {{private member}} expected-error {{inaccessible base class}}
    t->spub++; // expected-error {{private member}}
    t->prot++; // expected-error {{private member}} expected-error {{inaccessible base class}}
    t->sprot++; // expected-error {{private member}}
    t->priv++; // expected-error {{private member}} expected-error {{inaccessible base class}}
    t->spriv++; // expected-error {{private member}}

    t->Base::pub++; // expected-error {{private member}} expected-error {{inaccessible base class}}
    t->Base::spub++; // expected-error {{private member}}
    t->Base::prot++; // expected-error {{protected member}} expected-error {{private member}} expected-error {{inaccessible base class}}
    t->Base::sprot++; // expected-error {{protected member}} expected-error {{private member}}
    t->Base::priv++; // expected-error 2 {{private member}} expected-error {{inaccessible base class}}
    t->Base::spriv++; // expected-error 2 {{private member}}
  }
}
