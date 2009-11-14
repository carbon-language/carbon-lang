// RUN: clang-cc -fsyntax-only -verify -faccess-control %s

struct A {};
struct B : public A {};             // Single public base.
struct C1 : public virtual B {};    // Single virtual base.
struct C2 : public virtual B {};
struct D : public C1, public C2 {}; // Diamond
struct E : private A {};            // Single private base. expected-note 2 {{'private' inheritance specifier here}}
struct F : public C1 {};            // Single path to B with virtual.
struct G1 : public B {};
struct G2 : public B {};
struct H : public G1, public G2 {}; // Ambiguous path to B.

enum Enum { En1, En2 };
enum Onom { On1, On2 };

struct Co1 { operator int(); };
struct Co2 { Co2(int); };
struct Co3 { };
struct Co4 { Co4(Co3); operator Co3(); };

// Explicit implicits
void t_529_2()
{
  int i = 1;
  (void)static_cast<float>(i);
  double d = 1.0;
  (void)static_cast<float>(d);
  (void)static_cast<int>(d);
  (void)static_cast<char>(i);
  (void)static_cast<unsigned long>(i);
  (void)static_cast<int>(En1);
  (void)static_cast<double>(En1);
  (void)static_cast<int&>(i);
  (void)static_cast<const int&>(i);

  int ar[1];
  (void)static_cast<const int*>(ar);
  (void)static_cast<void (*)()>(t_529_2);

  (void)static_cast<void*>(0);
  (void)static_cast<void*>((int*)0);
  (void)static_cast<volatile const void*>((const int*)0);
  (void)static_cast<A*>((B*)0);
  (void)static_cast<A&>(*((B*)0));
  (void)static_cast<const B*>((C1*)0);
  (void)static_cast<B&>(*((C1*)0));
  (void)static_cast<A*>((D*)0);
  (void)static_cast<const A&>(*((D*)0));
  (void)static_cast<int B::*>((int A::*)0);
  (void)static_cast<void (B::*)()>((void (A::*)())0);

  (void)static_cast<int>(Co1());
  (void)static_cast<Co2>(1);
  (void)static_cast<Co3>(static_cast<Co4>(Co3()));

  // Bad code below

  (void)static_cast<void*>((const int*)0); // expected-error {{static_cast from 'int const *' to 'void *' is not allowed}}
  //(void)static_cast<A*>((E*)0); // {{static_cast from 'struct E *' to 'struct A *' is not allowed}}
  //(void)static_cast<A*>((H*)0); // {{static_cast from 'struct H *' to 'struct A *' is not allowed}}
  (void)static_cast<int>((int*)0); // expected-error {{static_cast from 'int *' to 'int' is not allowed}}
  (void)static_cast<A**>((B**)0); // expected-error {{static_cast from 'struct B **' to 'struct A **' is not allowed}}
  (void)static_cast<char&>(i); // expected-error {{non-const lvalue reference to type 'char' cannot be initialized with a value of type 'int'}}
}

// Anything to void
void t_529_4()
{
  static_cast<void>(1);
  static_cast<void>(t_529_4);
}

// Static downcasts
void t_529_5_8()
{
  (void)static_cast<B*>((A*)0);
  (void)static_cast<B&>(*((A*)0));
  (void)static_cast<const G1*>((A*)0);
  (void)static_cast<const G1&>(*((A*)0));

  // Bad code below

  (void)static_cast<C1*>((A*)0); // expected-error {{cannot cast 'struct A *' to 'struct C1 *' via virtual base 'struct B'}}
  (void)static_cast<C1&>(*((A*)0)); // expected-error {{cannot cast 'struct A' to 'struct C1 &' via virtual base 'struct B'}}
  (void)static_cast<D*>((A*)0); // expected-error {{cannot cast 'struct A *' to 'struct D *' via virtual base 'struct B'}}
  (void)static_cast<D&>(*((A*)0)); // expected-error {{cannot cast 'struct A' to 'struct D &' via virtual base 'struct B'}}
  (void)static_cast<B*>((const A*)0); // expected-error {{static_cast from 'struct A const *' to 'struct B *' casts away constness}}
  (void)static_cast<B&>(*((const A*)0)); // expected-error {{static_cast from 'struct A const' to 'struct B &' casts away constness}}
  (void)static_cast<E*>((A*)0); // expected-error {{cannot cast 'struct A' to 'struct E' due to inaccessible}}
  (void)static_cast<E&>(*((A*)0)); // expected-error {{cannot cast 'struct A' to 'struct E' due to inaccessible}}
  (void)static_cast<H*>((A*)0); // expected-error {{ambiguous cast from base 'struct A' to derived 'struct H':\n    struct A -> struct B -> struct G1 -> struct H\n    struct A -> struct B -> struct G2 -> struct H}}
  (void)static_cast<H&>(*((A*)0)); // expected-error {{ambiguous cast from base 'struct A' to derived 'struct H':\n    struct A -> struct B -> struct G1 -> struct H\n    struct A -> struct B -> struct G2 -> struct H}}
  (void)static_cast<E*>((B*)0); // expected-error {{static_cast from 'struct B *' to 'struct E *' is not allowed}}
  (void)static_cast<E&>(*((B*)0)); // expected-error {{non-const lvalue reference to type 'struct E' cannot be initialized with a value of type 'struct B'}}

  // TODO: Test inaccessible base in context where it's accessible, i.e.
  // member function and friend.

  // TODO: Test DR427. This requires user-defined conversions, though.
}

// Enum conversions
void t_529_7()
{
  (void)static_cast<Enum>(1);
  (void)static_cast<Enum>(1.0);
  (void)static_cast<Onom>(En1);

  // Bad code below

  (void)static_cast<Enum>((int*)0); // expected-error {{static_cast from 'int *' to 'enum Enum' is not allowed}}
}

// Void pointer to object pointer
void t_529_10()
{
  (void)static_cast<int*>((void*)0);
  (void)static_cast<const A*>((void*)0);

  // Bad code below

  (void)static_cast<int*>((const void*)0); // expected-error {{static_cast from 'void const *' to 'int *' casts away constness}}
  (void)static_cast<void (*)()>((void*)0); // expected-error {{static_cast from 'void *' to 'void (*)()' is not allowed}}
}

// Member pointer upcast.
void t_529_9()
{
  (void)static_cast<int A::*>((int B::*)0);

  // Bad code below
  (void)static_cast<int A::*>((int H::*)0); // expected-error {{ambiguous conversion from pointer to member of derived class 'struct H'}}
  (void)static_cast<int A::*>((int F::*)0); // expected-error {{conversion from pointer to member of class 'struct F'}}
}

// PR 5261 - static_cast should instantiate template if possible
namespace pr5261 {
  struct base {};
  template<typename E> struct derived : public base {};
  template<typename E> struct outer {
    base *pb;
    ~outer() { (void)static_cast<derived<E>*>(pb); }
  };
  outer<int> EntryList;
}


// Initialization by constructor
struct X0;

struct X1 {
  X1();
  X1(X1&);
  X1(const X0&);
  
  operator X0() const;
};

struct X0 { };

void test_ctor_init() {
  (void)static_cast<X1>(X1());
}
