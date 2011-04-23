// RUN: %clang_cc1 -fsyntax-only -verify %s 
#define T(b) (b) ? 1 : -1
#define F(b) (b) ? -1 : 1

struct NonPOD { NonPOD(int); };

// PODs
enum Enum { EV };
struct POD { Enum e; int i; float f; NonPOD* p; };
struct Empty {};
typedef Empty EmptyAr[10];
typedef int Int;
typedef Int IntAr[10];
typedef Int IntArNB[];
class Statics { static int priv; static NonPOD np; };
union EmptyUnion {};
union Union { int i; float f; };
struct HasFunc { void f (); };
struct HasOp { void operator *(); };
struct HasConv { operator int(); };
struct HasAssign { void operator =(int); };

struct HasAnonymousUnion {
  union {
    int i;
    float f;
  };
};

// Not PODs
typedef const void cvoid;
struct Derives : POD {};
struct DerivesEmpty : Empty {};
struct HasCons { HasCons(int); };
struct HasCopyAssign { HasCopyAssign operator =(const HasCopyAssign&); };
struct HasDest { ~HasDest(); };
class  HasPriv { int priv; };
class  HasProt { protected: int prot; };
struct HasRef { int i; int& ref; HasRef() : i(0), ref(i) {} };
struct HasNonPOD { NonPOD np; };
struct HasVirt { virtual void Virt() {}; };
typedef Derives NonPODAr[10];
typedef HasVirt VirtAr[10];
typedef HasCons NonPODArNB[];
union NonPODUnion { int i; Derives n; };

struct HasNoThrowCopyAssign {
  void operator =(const HasNoThrowCopyAssign&) throw();
};
struct HasMultipleCopyAssign {
  void operator =(const HasMultipleCopyAssign&) throw();
  void operator =(volatile HasMultipleCopyAssign&);
};
struct HasMultipleNoThrowCopyAssign {
  void operator =(const HasMultipleNoThrowCopyAssign&) throw();
  void operator =(volatile HasMultipleNoThrowCopyAssign&) throw();
};

struct HasNoThrowConstructor { HasNoThrowConstructor() throw(); };
struct HasNoThrowConstructorWithArgs {
  HasNoThrowConstructorWithArgs(HasCons i = HasCons(0)) throw();
};

struct HasNoThrowCopy { HasNoThrowCopy(const HasNoThrowCopy&) throw(); };
struct HasMultipleCopy {
  HasMultipleCopy(const HasMultipleCopy&) throw();
  HasMultipleCopy(volatile HasMultipleCopy&);
};
struct HasMultipleNoThrowCopy {
  HasMultipleNoThrowCopy(const HasMultipleNoThrowCopy&) throw();
  HasMultipleNoThrowCopy(volatile HasMultipleNoThrowCopy&) throw();
};

struct HasVirtDest { virtual ~HasVirtDest(); };
struct DerivedVirtDest : HasVirtDest {};
typedef HasVirtDest VirtDestAr[1];

void is_pod()
{
  { int arr[T(__is_pod(int))]; }
  { int arr[T(__is_pod(Enum))]; }
  { int arr[T(__is_pod(POD))]; }
  { int arr[T(__is_pod(Int))]; }
  { int arr[T(__is_pod(IntAr))]; }
  { int arr[T(__is_pod(Statics))]; }
  { int arr[T(__is_pod(Empty))]; }
  { int arr[T(__is_pod(EmptyUnion))]; }
  { int arr[T(__is_pod(Union))]; }
  { int arr[T(__is_pod(HasFunc))]; }
  { int arr[T(__is_pod(HasOp))]; }
  { int arr[T(__is_pod(HasConv))]; }
  { int arr[T(__is_pod(HasAssign))]; }
  { int arr[T(__is_pod(IntArNB))]; }
  { int arr[T(__is_pod(HasAnonymousUnion))]; }

  { int arr[F(__is_pod(Derives))]; }
  { int arr[F(__is_pod(HasCons))]; }
  { int arr[F(__is_pod(HasCopyAssign))]; }
  { int arr[F(__is_pod(HasDest))]; }
  { int arr[F(__is_pod(HasPriv))]; }
  { int arr[F(__is_pod(HasProt))]; }
  { int arr[F(__is_pod(HasRef))]; }
  { int arr[F(__is_pod(HasNonPOD))]; }
  { int arr[F(__is_pod(HasVirt))]; }
  { int arr[F(__is_pod(NonPODAr))]; }
  { int arr[F(__is_pod(DerivesEmpty))]; }
  { int arr[F(__is_pod(void))]; }
  { int arr[F(__is_pod(cvoid))]; }
  { int arr[F(__is_pod(NonPODArNB))]; }
// { int arr[F(__is_pod(NonPODUnion))]; }
}

typedef Empty EmptyAr[10];
struct Bit0 { int : 0; };
struct Bit0Cons { int : 0; Bit0Cons(); };
struct BitOnly { int x : 3; };
//struct DerivesVirt : virtual POD {};

void is_empty()
{
  { int arr[T(__is_empty(Empty))]; }
  { int arr[T(__is_empty(DerivesEmpty))]; }
  { int arr[T(__is_empty(HasCons))]; }
  { int arr[T(__is_empty(HasCopyAssign))]; }
  { int arr[T(__is_empty(HasDest))]; }
  { int arr[T(__is_empty(HasFunc))]; }
  { int arr[T(__is_empty(HasOp))]; }
  { int arr[T(__is_empty(HasConv))]; }
  { int arr[T(__is_empty(HasAssign))]; }
  { int arr[T(__is_empty(Bit0))]; }
  { int arr[T(__is_empty(Bit0Cons))]; }

  { int arr[F(__is_empty(Int))]; }
  { int arr[F(__is_empty(POD))]; }
  { int arr[F(__is_empty(EmptyUnion))]; }
  { int arr[F(__is_empty(EmptyAr))]; }
  { int arr[F(__is_empty(HasRef))]; }
  { int arr[F(__is_empty(HasVirt))]; }
  { int arr[F(__is_empty(BitOnly))]; }
  { int arr[F(__is_empty(void))]; }
  { int arr[F(__is_empty(IntArNB))]; }
  { int arr[F(__is_empty(HasAnonymousUnion))]; }
//  { int arr[F(__is_empty(DerivesVirt))]; }
}

typedef Derives ClassType;

void is_class()
{
  { int arr[T(__is_class(Derives))]; }
  { int arr[T(__is_class(HasPriv))]; }
  { int arr[T(__is_class(ClassType))]; }
  { int arr[T(__is_class(HasAnonymousUnion))]; }

  { int arr[F(__is_class(int))]; }
  { int arr[F(__is_class(Enum))]; }
  { int arr[F(__is_class(Int))]; }
  { int arr[F(__is_class(IntAr))]; }
  { int arr[F(__is_class(NonPODAr))]; }
  { int arr[F(__is_class(Union))]; }
  { int arr[F(__is_class(cvoid))]; }
  { int arr[F(__is_class(IntArNB))]; }
}

typedef Union UnionAr[10];
typedef Union UnionType;

void is_union()
{
  { int arr[T(__is_union(Union))]; }
  { int arr[T(__is_union(UnionType))]; }

  { int arr[F(__is_union(int))]; }
  { int arr[F(__is_union(Enum))]; }
  { int arr[F(__is_union(Int))]; }
  { int arr[F(__is_union(IntAr))]; }
  { int arr[F(__is_union(UnionAr))]; }
  { int arr[F(__is_union(cvoid))]; }
  { int arr[F(__is_union(IntArNB))]; }
  { int arr[F(__is_union(HasAnonymousUnion))]; }
}

typedef Enum EnumType;

void is_enum()
{
  { int arr[T(__is_enum(Enum))]; }
  { int arr[T(__is_enum(EnumType))]; }

  { int arr[F(__is_enum(int))]; }
  { int arr[F(__is_enum(Union))]; }
  { int arr[F(__is_enum(Int))]; }
  { int arr[F(__is_enum(IntAr))]; }
  { int arr[F(__is_enum(UnionAr))]; }
  { int arr[F(__is_enum(Derives))]; }
  { int arr[F(__is_enum(ClassType))]; }
  { int arr[F(__is_enum(cvoid))]; }
  { int arr[F(__is_enum(IntArNB))]; }
  { int arr[F(__is_enum(HasAnonymousUnion))]; }
}

typedef HasVirt Polymorph;
struct InheritPolymorph : Polymorph {};

void is_polymorphic()
{
  { int arr[T(__is_polymorphic(Polymorph))]; }
  { int arr[T(__is_polymorphic(InheritPolymorph))]; }

  { int arr[F(__is_polymorphic(int))]; }
  { int arr[F(__is_polymorphic(Union))]; }
  { int arr[F(__is_polymorphic(Int))]; }
  { int arr[F(__is_polymorphic(IntAr))]; }
  { int arr[F(__is_polymorphic(UnionAr))]; }
  { int arr[F(__is_polymorphic(Derives))]; }
  { int arr[F(__is_polymorphic(ClassType))]; }
  { int arr[F(__is_polymorphic(Enum))]; }
  { int arr[F(__is_polymorphic(cvoid))]; }
  { int arr[F(__is_polymorphic(IntArNB))]; }
}

typedef Int& IntRef;
typedef const IntAr ConstIntAr;
typedef ConstIntAr ConstIntArAr[4];

struct HasCopy {
  HasCopy(HasCopy& cp);
};

struct HasTemplateCons {
  HasVirt Annoying;

  template <typename T>
  HasTemplateCons(const T&);
};

void has_trivial_default_constructor() {
  { int arr[T(__has_trivial_constructor(Int))]; }
  { int arr[T(__has_trivial_constructor(IntAr))]; }
  { int arr[T(__has_trivial_constructor(Union))]; }
  { int arr[T(__has_trivial_constructor(UnionAr))]; }
  { int arr[T(__has_trivial_constructor(POD))]; }
  { int arr[T(__has_trivial_constructor(Derives))]; }
  { int arr[T(__has_trivial_constructor(ConstIntAr))]; }
  { int arr[T(__has_trivial_constructor(ConstIntArAr))]; }
  { int arr[T(__has_trivial_constructor(HasDest))]; }
  { int arr[T(__has_trivial_constructor(HasPriv))]; }
  { int arr[F(__has_trivial_constructor(HasCons))]; }
  { int arr[F(__has_trivial_constructor(HasRef))]; }
  { int arr[F(__has_trivial_constructor(HasCopy))]; }
  { int arr[F(__has_trivial_constructor(IntRef))]; }
  { int arr[T(__has_trivial_constructor(HasCopyAssign))]; }
  { int arr[T(__has_trivial_constructor(const Int))]; }
  { int arr[T(__has_trivial_constructor(NonPODAr))]; }
  { int arr[F(__has_trivial_constructor(VirtAr))]; }
  { int arr[F(__has_trivial_constructor(void))]; }
  { int arr[F(__has_trivial_constructor(cvoid))]; }
  { int arr[F(__has_trivial_constructor(HasTemplateCons))]; }
}

void has_trivial_copy_constructor() {
  { int arr[T(__has_trivial_copy(Int))]; }
  { int arr[T(__has_trivial_copy(IntAr))]; }
  { int arr[T(__has_trivial_copy(Union))]; }
  { int arr[T(__has_trivial_copy(UnionAr))]; }
  { int arr[T(__has_trivial_copy(POD))]; }
  { int arr[T(__has_trivial_copy(Derives))]; }
  { int arr[T(__has_trivial_copy(ConstIntAr))]; }
  { int arr[T(__has_trivial_copy(ConstIntArAr))]; }
  { int arr[T(__has_trivial_copy(HasDest))]; }
  { int arr[T(__has_trivial_copy(HasPriv))]; }
  { int arr[T(__has_trivial_copy(HasCons))]; }
  { int arr[T(__has_trivial_copy(HasRef))]; }
  { int arr[F(__has_trivial_copy(HasCopy))]; }
  { int arr[T(__has_trivial_copy(IntRef))]; }
  { int arr[T(__has_trivial_copy(HasCopyAssign))]; }
  { int arr[T(__has_trivial_copy(const Int))]; }
  { int arr[F(__has_trivial_copy(NonPODAr))]; }
  { int arr[F(__has_trivial_copy(VirtAr))]; }
  { int arr[F(__has_trivial_copy(void))]; }
  { int arr[F(__has_trivial_copy(cvoid))]; }
  { int arr[F(__has_trivial_copy(HasTemplateCons))]; }
}

void has_trivial_copy_assignment() {
  { int arr[T(__has_trivial_assign(Int))]; }
  { int arr[T(__has_trivial_assign(IntAr))]; }
  { int arr[T(__has_trivial_assign(Union))]; }
  { int arr[T(__has_trivial_assign(UnionAr))]; }
  { int arr[T(__has_trivial_assign(POD))]; }
  { int arr[T(__has_trivial_assign(Derives))]; }
  { int arr[F(__has_trivial_assign(ConstIntAr))]; }
  { int arr[F(__has_trivial_assign(ConstIntArAr))]; }
  { int arr[T(__has_trivial_assign(HasDest))]; }
  { int arr[T(__has_trivial_assign(HasPriv))]; }
  { int arr[T(__has_trivial_assign(HasCons))]; }
  { int arr[T(__has_trivial_assign(HasRef))]; }
  { int arr[T(__has_trivial_assign(HasCopy))]; }
  { int arr[F(__has_trivial_assign(IntRef))]; }
  { int arr[F(__has_trivial_assign(HasCopyAssign))]; }
  { int arr[F(__has_trivial_assign(const Int))]; }
  { int arr[F(__has_trivial_assign(NonPODAr))]; }
  { int arr[F(__has_trivial_assign(VirtAr))]; }
  { int arr[F(__has_trivial_assign(void))]; }
  { int arr[F(__has_trivial_assign(cvoid))]; }
}

void has_trivial_destructor() {
  { int arr[T(__has_trivial_destructor(Int))]; }
  { int arr[T(__has_trivial_destructor(IntAr))]; }
  { int arr[T(__has_trivial_destructor(Union))]; }
  { int arr[T(__has_trivial_destructor(UnionAr))]; }
  { int arr[T(__has_trivial_destructor(POD))]; }
  { int arr[T(__has_trivial_destructor(Derives))]; }
  { int arr[T(__has_trivial_destructor(ConstIntAr))]; }
  { int arr[T(__has_trivial_destructor(ConstIntArAr))]; }
  { int arr[F(__has_trivial_destructor(HasDest))]; }
  { int arr[T(__has_trivial_destructor(HasPriv))]; }
  { int arr[T(__has_trivial_destructor(HasCons))]; }
  { int arr[T(__has_trivial_destructor(HasRef))]; }
  { int arr[T(__has_trivial_destructor(HasCopy))]; }
  { int arr[T(__has_trivial_destructor(IntRef))]; }
  { int arr[T(__has_trivial_destructor(HasCopyAssign))]; }
  { int arr[T(__has_trivial_destructor(const Int))]; }
  { int arr[T(__has_trivial_destructor(NonPODAr))]; }
  { int arr[T(__has_trivial_destructor(VirtAr))]; }
  { int arr[F(__has_trivial_destructor(void))]; }
  { int arr[F(__has_trivial_destructor(cvoid))]; }
}

struct A { ~A() {} };
template<typename> struct B : A { };

void f() {
  { int arr[F(__has_trivial_destructor(A))]; }
  { int arr[F(__has_trivial_destructor(B<int>))]; }
}

void has_nothrow_assign() {
  { int arr[T(__has_nothrow_assign(Int))]; }
  { int arr[T(__has_nothrow_assign(IntAr))]; }
  { int arr[T(__has_nothrow_assign(Union))]; }
  { int arr[T(__has_nothrow_assign(UnionAr))]; }
  { int arr[T(__has_nothrow_assign(POD))]; }
  { int arr[T(__has_nothrow_assign(Derives))]; }
  { int arr[F(__has_nothrow_assign(ConstIntAr))]; }
  { int arr[F(__has_nothrow_assign(ConstIntArAr))]; }
  { int arr[T(__has_nothrow_assign(HasDest))]; }
  { int arr[T(__has_nothrow_assign(HasPriv))]; }
  { int arr[T(__has_nothrow_assign(HasCons))]; }
  { int arr[T(__has_nothrow_assign(HasRef))]; }
  { int arr[T(__has_nothrow_assign(HasCopy))]; }
  { int arr[F(__has_nothrow_assign(IntRef))]; }
  { int arr[F(__has_nothrow_assign(HasCopyAssign))]; }
  { int arr[F(__has_nothrow_assign(const Int))]; }
  { int arr[F(__has_nothrow_assign(NonPODAr))]; }
  { int arr[F(__has_nothrow_assign(VirtAr))]; }
  { int arr[T(__has_nothrow_assign(HasNoThrowCopyAssign))]; }
  { int arr[F(__has_nothrow_assign(HasMultipleCopyAssign))]; }
  { int arr[T(__has_nothrow_assign(HasMultipleNoThrowCopyAssign))]; }
  { int arr[F(__has_nothrow_assign(void))]; }
  { int arr[F(__has_nothrow_assign(cvoid))]; }
  { int arr[T(__has_nothrow_assign(HasVirtDest))]; }
}

void has_nothrow_copy() {
  { int arr[T(__has_nothrow_copy(Int))]; }
  { int arr[T(__has_nothrow_copy(IntAr))]; }
  { int arr[T(__has_nothrow_copy(Union))]; }
  { int arr[T(__has_nothrow_copy(UnionAr))]; }
  { int arr[T(__has_nothrow_copy(POD))]; }
  { int arr[T(__has_nothrow_copy(Derives))]; }
  { int arr[T(__has_nothrow_copy(ConstIntAr))]; }
  { int arr[T(__has_nothrow_copy(ConstIntArAr))]; }
  { int arr[T(__has_nothrow_copy(HasDest))]; }
  { int arr[T(__has_nothrow_copy(HasPriv))]; }
  { int arr[T(__has_nothrow_copy(HasCons))]; }
  { int arr[T(__has_nothrow_copy(HasRef))]; }
  { int arr[F(__has_nothrow_copy(HasCopy))]; }
  { int arr[T(__has_nothrow_copy(IntRef))]; }
  { int arr[T(__has_nothrow_copy(HasCopyAssign))]; }
  { int arr[T(__has_nothrow_copy(const Int))]; }
  { int arr[F(__has_nothrow_copy(NonPODAr))]; }
  { int arr[F(__has_nothrow_copy(VirtAr))]; }

  { int arr[T(__has_nothrow_copy(HasNoThrowCopy))]; }
  { int arr[F(__has_nothrow_copy(HasMultipleCopy))]; }
  { int arr[T(__has_nothrow_copy(HasMultipleNoThrowCopy))]; }
  { int arr[F(__has_nothrow_copy(void))]; }
  { int arr[F(__has_nothrow_copy(cvoid))]; }
  { int arr[T(__has_nothrow_copy(HasVirtDest))]; }
  { int arr[T(__has_nothrow_copy(HasTemplateCons))]; }
}

void has_nothrow_constructor() {
  { int arr[T(__has_nothrow_constructor(Int))]; }
  { int arr[T(__has_nothrow_constructor(IntAr))]; }
  { int arr[T(__has_nothrow_constructor(Union))]; }
  { int arr[T(__has_nothrow_constructor(UnionAr))]; }
  { int arr[T(__has_nothrow_constructor(POD))]; }
  { int arr[T(__has_nothrow_constructor(Derives))]; }
  { int arr[T(__has_nothrow_constructor(ConstIntAr))]; }
  { int arr[T(__has_nothrow_constructor(ConstIntArAr))]; }
  { int arr[T(__has_nothrow_constructor(HasDest))]; }
  { int arr[T(__has_nothrow_constructor(HasPriv))]; }
  { int arr[F(__has_nothrow_constructor(HasCons))]; }
  { int arr[F(__has_nothrow_constructor(HasRef))]; }
  { int arr[F(__has_nothrow_constructor(HasCopy))]; }
  { int arr[F(__has_nothrow_constructor(IntRef))]; }
  { int arr[T(__has_nothrow_constructor(HasCopyAssign))]; }
  { int arr[T(__has_nothrow_constructor(const Int))]; }
  { int arr[T(__has_nothrow_constructor(NonPODAr))]; }
  // { int arr[T(__has_nothrow_constructor(VirtAr))]; } // not implemented

  { int arr[T(__has_nothrow_constructor(HasNoThrowConstructor))]; }
  { int arr[F(__has_nothrow_constructor(HasNoThrowConstructorWithArgs))]; }
  { int arr[F(__has_nothrow_constructor(void))]; }
  { int arr[F(__has_nothrow_constructor(cvoid))]; }
  { int arr[T(__has_nothrow_constructor(HasVirtDest))]; }
  { int arr[F(__has_nothrow_constructor(HasTemplateCons))]; }
}

void has_virtual_destructor() {
  { int arr[F(__has_virtual_destructor(Int))]; }
  { int arr[F(__has_virtual_destructor(IntAr))]; }
  { int arr[F(__has_virtual_destructor(Union))]; }
  { int arr[F(__has_virtual_destructor(UnionAr))]; }
  { int arr[F(__has_virtual_destructor(POD))]; }
  { int arr[F(__has_virtual_destructor(Derives))]; }
  { int arr[F(__has_virtual_destructor(ConstIntAr))]; }
  { int arr[F(__has_virtual_destructor(ConstIntArAr))]; }
  { int arr[F(__has_virtual_destructor(HasDest))]; }
  { int arr[F(__has_virtual_destructor(HasPriv))]; }
  { int arr[F(__has_virtual_destructor(HasCons))]; }
  { int arr[F(__has_virtual_destructor(HasRef))]; }
  { int arr[F(__has_virtual_destructor(HasCopy))]; }
  { int arr[F(__has_virtual_destructor(IntRef))]; }
  { int arr[F(__has_virtual_destructor(HasCopyAssign))]; }
  { int arr[F(__has_virtual_destructor(const Int))]; }
  { int arr[F(__has_virtual_destructor(NonPODAr))]; }
  { int arr[F(__has_virtual_destructor(VirtAr))]; }

  { int arr[T(__has_virtual_destructor(HasVirtDest))]; }
  { int arr[T(__has_virtual_destructor(DerivedVirtDest))]; }
  { int arr[F(__has_virtual_destructor(VirtDestAr))]; }
  { int arr[F(__has_virtual_destructor(void))]; }
  { int arr[F(__has_virtual_destructor(cvoid))]; }
}


class Base {};
class Derived : Base {};
class Derived2a : Derived {};
class Derived2b : Derived {};
class Derived3 : virtual Derived2a, virtual Derived2b {};
template<typename T> struct BaseA { T a;  };
template<typename T> struct DerivedB : BaseA<T> { };
template<typename T> struct CrazyDerived : T { };


class class_forward; // expected-note {{forward declaration of 'class_forward'}}

template <typename Base, typename Derived>
void isBaseOfT() {
  int t[T(__is_base_of(Base, Derived))];
};
template <typename Base, typename Derived>
void isBaseOfF() {
  int t[F(__is_base_of(Base, Derived))];
};

template <class T> class DerivedTemp : Base {};
template <class T> class NonderivedTemp {};
template <class T> class UndefinedTemp; // expected-note {{declared here}}

void is_base_of() {
  { int arr[T(__is_base_of(Base, Derived))]; }
  { int arr[T(__is_base_of(const Base, Derived))]; }
  { int arr[F(__is_base_of(Derived, Base))]; }
  { int arr[F(__is_base_of(Derived, int))]; }
  { int arr[T(__is_base_of(Base, Base))]; }
  { int arr[T(__is_base_of(Base, Derived3))]; }
  { int arr[T(__is_base_of(Derived, Derived3))]; }
  { int arr[T(__is_base_of(Derived2b, Derived3))]; }
  { int arr[T(__is_base_of(Derived2a, Derived3))]; }
  { int arr[T(__is_base_of(BaseA<int>, DerivedB<int>))]; }
  { int arr[F(__is_base_of(DerivedB<int>, BaseA<int>))]; }
  { int arr[T(__is_base_of(Base, CrazyDerived<Base>))]; }
  { int arr[F(__is_base_of(Union, Union))]; }
  { int arr[T(__is_base_of(Empty, Empty))]; }
  { int arr[T(__is_base_of(class_forward, class_forward))]; }
  { int arr[F(__is_base_of(Empty, class_forward))]; } // expected-error {{incomplete type 'class_forward' used in type trait expression}}
  { int arr[F(__is_base_of(Base&, Derived&))]; }
  int t18[F(__is_base_of(Base[10], Derived[10]))];
  { int arr[F(__is_base_of(int, int))]; }
  { int arr[F(__is_base_of(long, int))]; }
  { int arr[T(__is_base_of(Base, DerivedTemp<int>))]; }
  { int arr[F(__is_base_of(Base, NonderivedTemp<int>))]; }
  { int arr[F(__is_base_of(Base, UndefinedTemp<int>))]; } // expected-error {{implicit instantiation of undefined template 'UndefinedTemp<int>'}}

  isBaseOfT<Base, Derived>();
  isBaseOfF<Derived, Base>();

  isBaseOfT<Base, CrazyDerived<Base> >();
  isBaseOfF<CrazyDerived<Base>, Base>();

  isBaseOfT<BaseA<int>, DerivedB<int> >();
  isBaseOfF<DerivedB<int>, BaseA<int> >();
}

struct FromInt { FromInt(int); };
struct ToInt { operator int(); };
typedef void Function();

void is_convertible_to();
class PrivateCopy {
  PrivateCopy(const PrivateCopy&);
  friend void is_convertible_to();
};

template<typename T>
struct X0 { 
  template<typename U> X0(const X0<U>&);
};

void is_convertible_to() {
  { int arr[T(__is_convertible_to(Int, Int))]; }
  { int arr[F(__is_convertible_to(Int, IntAr))]; }
  { int arr[F(__is_convertible_to(IntAr, IntAr))]; }
  { int arr[T(__is_convertible_to(void, void))]; }
  { int arr[T(__is_convertible_to(cvoid, void))]; }
  { int arr[T(__is_convertible_to(void, cvoid))]; }
  { int arr[T(__is_convertible_to(cvoid, cvoid))]; }
  { int arr[T(__is_convertible_to(int, FromInt))]; }
  { int arr[T(__is_convertible_to(long, FromInt))]; }
  { int arr[T(__is_convertible_to(double, FromInt))]; }
  { int arr[T(__is_convertible_to(const int, FromInt))]; }
  { int arr[T(__is_convertible_to(const int&, FromInt))]; }
  { int arr[T(__is_convertible_to(ToInt, int))]; }
  { int arr[T(__is_convertible_to(ToInt, const int&))]; }
  { int arr[T(__is_convertible_to(ToInt, long))]; }
  { int arr[F(__is_convertible_to(ToInt, int&))]; }
  { int arr[F(__is_convertible_to(ToInt, FromInt))]; }
  { int arr[T(__is_convertible_to(IntAr&, IntAr&))]; }
  { int arr[T(__is_convertible_to(IntAr&, const IntAr&))]; }
  { int arr[F(__is_convertible_to(const IntAr&, IntAr&))]; }
  { int arr[F(__is_convertible_to(Function, Function))]; }
  { int arr[F(__is_convertible_to(PrivateCopy, PrivateCopy))]; }
  { int arr[T(__is_convertible_to(X0<int>, X0<float>))]; }
}

void is_trivial()
{
  { int arr[T(__is_trivial(int))]; }
  { int arr[T(__is_trivial(Enum))]; }
  { int arr[T(__is_trivial(POD))]; }
  { int arr[T(__is_trivial(Int))]; }
  { int arr[T(__is_trivial(IntAr))]; }
  { int arr[T(__is_trivial(Statics))]; }
  { int arr[T(__is_trivial(Empty))]; }
  { int arr[T(__is_trivial(EmptyUnion))]; }
  { int arr[T(__is_trivial(Union))]; }
  { int arr[T(__is_trivial(HasFunc))]; }
  { int arr[T(__is_trivial(HasOp))]; }
  { int arr[T(__is_trivial(HasConv))]; }
  { int arr[T(__is_trivial(HasAssign))]; }
  { int arr[T(__is_trivial(HasAnonymousUnion))]; }
  { int arr[T(__is_trivial(Derives))]; }
  { int arr[T(__is_trivial(DerivesEmpty))]; }
  { int arr[T(__is_trivial(NonPODAr))]; }
  { int arr[T(__is_trivial(HasPriv))]; }
  { int arr[T(__is_trivial(HasProt))]; }

  { int arr[F(__is_trivial(IntArNB))]; }
  { int arr[F(__is_trivial(HasCons))]; }
  { int arr[F(__is_trivial(HasCopyAssign))]; }
  { int arr[F(__is_trivial(HasDest))]; }
  { int arr[F(__is_trivial(HasRef))]; }
  { int arr[F(__is_trivial(HasNonPOD))]; }
  { int arr[F(__is_trivial(HasVirt))]; }
  { int arr[F(__is_trivial(void))]; }
  { int arr[F(__is_trivial(cvoid))]; }
  { int arr[F(__is_trivial(NonPODArNB))]; }
}
