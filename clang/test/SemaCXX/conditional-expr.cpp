// RUN: clang-cc -fsyntax-only -verify -faccess-control -std=c++0x %s

// C++ rules for ?: are a lot stricter than C rules, and have to take into
// account more conversion options.
// This test runs in C++0x mode for the contextual conversion of the condition.

struct ToBool { explicit operator bool(); };

struct B;
struct A { A(); A(const B&); };
struct B { operator A() const; };
struct I { operator int(); };
struct J { operator I(); };
struct K { operator double(); };
typedef void (*vfn)();
struct F { operator vfn(); };
struct G { operator vfn(); };

struct Base {
  int trick();
  A trick() const;
  void fn1();
};
struct Derived : Base {
  void fn2();
};
struct Convertible { operator Base&(); };
struct Priv : private Base {}; // expected-note 2 {{'private' inheritance specifier here}}
struct Mid : Base {};
struct Fin : Mid, Derived {};
typedef void (Derived::*DFnPtr)();
struct ToMemPtr { operator DFnPtr(); };

struct BadDerived;
struct BadBase { operator BadDerived&(); };
struct BadDerived : BadBase {};

struct Fields {
  int i1, i2, b1 : 3, b2 : 3;
};
struct MixedFields {
  int i;
  volatile int vi;
  const int ci;
  const volatile int cvi;
};
struct MixedFieldsDerived : MixedFields {
};

enum Enum { EVal };

struct Ambig {
  operator short();
  operator signed char();
};

void test()
{
  // This function tests C++0x 5.16

  // p1 (contextually convert to bool)
  int i1 = ToBool() ? 0 : 1;

  // p2 (one or both void, and throwing)
  i1 ? throw 0 : throw 1;
  i1 ? test() : throw 1;
  i1 ? throw 0 : test();
  i1 ? test() : test();
  i1 = i1 ? throw 0 : 0;
  i1 = i1 ? 0 : throw 0;
  i1 ? 0 : test(); // expected-error {{right operand to ? is void, but left operand is of type 'int'}}
  i1 ? test() : 0; // expected-error {{left operand to ? is void, but right operand is of type 'int'}}
  (i1 ? throw 0 : i1) = 0; // expected-error {{expression is not assignable}}
  (i1 ? i1 : throw 0) = 0; // expected-error {{expression is not assignable}}

  // p3 (one or both class type, convert to each other)
  // b1 (lvalues)
  Base base;
  Derived derived;
  Convertible conv;
  Base &bar1 = i1 ? base : derived;
  Base &bar2 = i1 ? derived : base;
  Base &bar3 = i1 ? base : conv;
  Base &bar4 = i1 ? conv : base;
  // these are ambiguous
  BadBase bb;
  BadDerived bd;
  (void)(i1 ? bb : bd); // expected-error {{conditional expression is ambiguous; 'struct BadBase' can be converted to 'struct BadDerived' and vice versa}}
  (void)(i1 ? bd : bb); // expected-error {{conditional expression is ambiguous}}
  // curiously enough (and a defect?), these are not
  // for rvalues, hierarchy takes precedence over other conversions
  (void)(i1 ? BadBase() : BadDerived());
  (void)(i1 ? BadDerived() : BadBase());

  // b2.1 (hierarchy stuff)
  const Base constret();
  const Derived constder();
  // should use const overload
  A a1((i1 ? constret() : Base()).trick());
  A a2((i1 ? Base() : constret()).trick());
  A a3((i1 ? constret() : Derived()).trick());
  A a4((i1 ? Derived() : constret()).trick());
  // should use non-const overload
  i1 = (i1 ? Base() : Base()).trick();
  i1 = (i1 ? Base() : Base()).trick();
  i1 = (i1 ? Base() : Derived()).trick();
  i1 = (i1 ? Derived() : Base()).trick();
  // should fail: const lost
  (void)(i1 ? Base() : constder()); // expected-error {{incompatible operand types ('struct Base' and 'struct Derived const')}}
  (void)(i1 ? constder() : Base()); // expected-error {{incompatible operand types ('struct Derived const' and 'struct Base')}}

  // FIXME: these are invalid hierarchy conversions
  Priv priv;
  Fin fin;
  (void)(i1 ? Base() : Priv()); // xpected-error private base
  (void)(i1 ? Priv() : Base()); // xpected-error private base
  (void)(i1 ? Base() : Fin()); // xpected-error ambiguous base
  (void)(i1 ? Fin() : Base()); // xpected-error ambiguous base
  (void)(i1 ? base : priv); // expected-error {{conversion from 'struct Priv' to inaccessible base class 'struct Base'}}
  (void)(i1 ? priv : base); // expected-error {{conversion from 'struct Priv' to inaccessible base class 'struct Base'}}
  (void)(i1 ? base : fin); // expected-error {{ambiguous conversion from derived class 'struct Fin' to base class 'struct Base'}}
  (void)(i1 ? fin : base); // expected-error {{ambiguous conversion from derived class 'struct Fin' to base class 'struct Base'}}

  // b2.2 (non-hierarchy)
  i1 = i1 ? I() : i1;
  i1 = i1 ? i1 : I();
  I i2(i1 ? I() : J());
  I i3(i1 ? J() : I());
  // "the type [it] woud have if E2 were converted to an rvalue"
  vfn pfn = i1 ? F() : test;
  pfn = i1 ? test : F();
  // these are ambiguous - better messages would be nice
  (void)(i1 ? A() : B()); // expected-error {{incompatible operand types}}
  (void)(i1 ? B() : A()); // expected-error {{incompatible operand types}}
  (void)(i1 ? 1 : Ambig()); // expected-error {{incompatible operand types}}
  (void)(i1 ? Ambig() : 1); // expected-error {{incompatible operand types}}
  // By the way, this isn't an lvalue:
  &(i1 ? i1 : i2); // expected-error {{address expression must be an lvalue or a function designator}}

  // p4 (lvalue, same type)
  Fields flds;
  int &ir1 = i1 ? flds.i1 : flds.i2;
  (i1 ? flds.b1 : flds.i2) = 0;
  (i1 ? flds.i1 : flds.b2) = 0;
  (i1 ? flds.b1 : flds.b2) = 0;

  // p5 (conversion to built-in types)
  // GCC 4.3 fails these
  double d1 = i1 ? I() : K();
  pfn = i1 ? F() : G();
  DFnPtr pfm;
  pfm = i1 ? DFnPtr() : &Base::fn1;
  pfm = i1 ? &Base::fn1 : DFnPtr();

  // p6 (final conversions)
  i1 = i1 ? i1 : ir1;
  int *pi1 = i1 ? &i1 : 0;
  pi1 = i1 ? 0 : &i1;
  i1 = i1 ? i1 : EVal;
  i1 = i1 ? EVal : i1;
  d1 = i1 ? 'c' : 4.0;
  d1 = i1 ? 4.0 : 'c';
  Base *pb = i1 ? (Base*)0 : (Derived*)0;
  pb = i1 ? (Derived*)0 : (Base*)0;
  pfm = i1 ? &Base::fn1 : &Derived::fn2;
  pfm = i1 ? &Derived::fn2 : &Base::fn1;
  pfm = i1 ? &Derived::fn2 : 0;
  pfm = i1 ? 0 : &Derived::fn2;
  const int (MixedFieldsDerived::*mp1) =
    i1 ? &MixedFields::ci : &MixedFieldsDerived::i;
  const volatile int (MixedFields::*mp2) =
    i1 ? &MixedFields::ci : &MixedFields::cvi;
  (void)(i1 ? &MixedFields::ci : &MixedFields::vi);
  // Conversion of primitives does not result in an lvalue.
  &(i1 ? i1 : d1); // expected-error {{address expression must be an lvalue or a function designator}}

  (void)&(i1 ? flds.b1 : flds.i1); // expected-error {{address of bit-field requested}}
  (void)&(i1 ? flds.i1 : flds.b1); // expected-error {{address of bit-field requested}}
  
  // Note the thing that this does not test: since DR446, various situations
  // *must* create a separate temporary copy of class objects. This can only
  // be properly tested at runtime, though.
}
