// RUN: clang -fsyntax-only -verify %s

struct A {};
struct B : public A {};             // Single public base.
struct C1 : public virtual B {};    // Single virtual base.
struct C2 : public virtual B {};
struct D : public C1, public C2 {}; // Diamond
struct E : private A {};            // Single private base.
struct F : public C1 {};            // Single path to B with virtual.
struct G1 : public B {};
struct G2 : public B {};
struct H : public G1, public G2 {}; // Ambiguous path to B.

enum Enum { En1, En2 };
enum Onom { On1, On2 };

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
  // TryCopyInitialization doesn't handle references yet.
  (void)static_cast<A&>(*((B*)0));
  (void)static_cast<const B*>((C1*)0);
  (void)static_cast<B&>(*((C1*)0));
  (void)static_cast<A*>((D*)0);
  (void)static_cast<const A&>(*((D*)0));

  // TODO: User-defined conversions

  // Bad code below

  (void)static_cast<void*>((const int*)0); // expected-error {{static_cast from 'int const *' to 'void *' is not allowed}}
  //(void)static_cast<A*>((E*)0); // {{static_cast from 'struct E *' to 'struct A *' is not allowed}}
  //(void)static_cast<A*>((H*)0); // {{static_cast from 'struct H *' to 'struct A *' is not allowed}}
  (void)static_cast<int>((int*)0); // expected-error {{static_cast from 'int *' to 'int' is not allowed}}
  (void)static_cast<A**>((B**)0); // expected-error {{static_cast from 'struct B **' to 'struct A **' is not allowed}}
  (void)static_cast<char&>(i); // expected-error {{static_cast from 'int' to 'char &' is not allowed}}
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

  (void)static_cast<C1*>((A*)0); // expected-error {{static_cast from 'struct A *' to 'struct C1 *' is not allowed}}
  (void)static_cast<C1&>(*((A*)0)); // expected-error {{static_cast from 'struct A' to 'struct C1 &' is not allowed}}
  (void)static_cast<D*>((A*)0); // expected-error {{static_cast from 'struct A *' to 'struct D *' is not allowed}}
  (void)static_cast<D&>(*((A*)0)); // expected-error {{static_cast from 'struct A' to 'struct D &' is not allowed}}
  (void)static_cast<B*>((const A*)0); // expected-error {{static_cast from 'struct A const *' to 'struct B *' is not allowed}}
  (void)static_cast<B&>(*((const A*)0)); // expected-error {{static_cast from 'struct A const' to 'struct B &' is not allowed}}
  // Accessibility is not yet tested
  //(void)static_cast<E*>((A*)0); // {{static_cast from 'struct A *' to 'struct E *' is not allowed}}
  //(void)static_cast<E&>(*((A*)0)); // {{static_cast from 'struct A' to 'struct E &' is not allowed}}
  (void)static_cast<H*>((A*)0); // expected-error {{static_cast from 'struct A *' to 'struct H *' is not allowed}}
  (void)static_cast<H&>(*((A*)0)); // expected-error {{static_cast from 'struct A' to 'struct H &' is not allowed}}
  (void)static_cast<E*>((B*)0); // expected-error {{static_cast from 'struct B *' to 'struct E *' is not allowed}}
  (void)static_cast<E&>(*((B*)0)); // expected-error {{static_cast from 'struct B' to 'struct E &' is not allowed}}

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

  (void)static_cast<int*>((const void*)0); // expected-error {{static_cast from 'void const *' to 'int *' is not allowed}}
  (void)static_cast<void (*)()>((void*)0); // expected-error {{static_cast from 'void *' to 'void (*)(void)' is not allowed}}
}

// TODO: Test member pointers.
