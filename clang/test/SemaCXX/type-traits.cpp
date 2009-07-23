// RUN: clang-cc -fsyntax-only -verify %s 
#define T(b) (b) ? 1 : -1
#define F(b) (b) ? -1 : 1

struct NonPOD { NonPOD(int); };

// PODs
enum Enum { EV };
struct POD { Enum e; int i; float f; NonPOD* p; };
typedef int Int;
typedef Int IntAr[10];
class Statics { static int priv; static NonPOD np; };

// Not PODs
struct Derives : POD {};
struct HasCons { HasCons(int); };
struct HasAssign { HasAssign operator =(const HasAssign&); };
struct HasDest { ~HasDest(); };
class  HasPriv { int priv; };
class  HasProt { protected: int prot; };
struct HasRef { int i; int& ref; HasRef() : i(0), ref(i) {} };
struct HasNonPOD { NonPOD np; };
struct HasVirt { virtual void Virt() {}; };
typedef Derives NonPODAr[10];
typedef HasVirt VirtAr[10];

void is_pod()
{
  int t01[T(__is_pod(int))];
  int t02[T(__is_pod(Enum))];
  int t03[T(__is_pod(POD))];
  int t04[T(__is_pod(Int))];
  int t05[T(__is_pod(IntAr))];
  int t06[T(__is_pod(Statics))];

  int t21[F(__is_pod(Derives))];
  int t22[F(__is_pod(HasCons))];
  int t23[F(__is_pod(HasAssign))];
  int t24[F(__is_pod(HasDest))];
  int t25[F(__is_pod(HasPriv))];
  int t26[F(__is_pod(HasProt))];
  int t27[F(__is_pod(HasRef))];
  int t28[F(__is_pod(HasNonPOD))];
  int t29[F(__is_pod(HasVirt))];
  int t30[F(__is_pod(NonPODAr))];
}

union Union { int i; float f; };
typedef Derives ClassType;

void is_class()
{
  int t01[T(__is_class(Derives))];
  int t02[T(__is_class(HasPriv))];
  int t03[T(__is_class(ClassType))];

  int t11[F(__is_class(int))];
  int t12[F(__is_class(Enum))];
  int t13[F(__is_class(Int))];
  int t14[F(__is_class(IntAr))];
  int t15[F(__is_class(NonPODAr))];
  int t16[F(__is_class(Union))];
}

typedef Union UnionAr[10];
typedef Union UnionType;

void is_union()
{
  int t01[T(__is_union(Union))];
  int t02[T(__is_union(UnionType))];

  int t11[F(__is_union(int))];
  int t12[F(__is_union(Enum))];
  int t13[F(__is_union(Int))];
  int t14[F(__is_union(IntAr))];
  int t15[F(__is_union(UnionAr))];
}

typedef Enum EnumType;

void is_enum()
{
  int t01[T(__is_enum(Enum))];
  int t02[T(__is_enum(EnumType))];

  int t11[F(__is_enum(int))];
  int t12[F(__is_enum(Union))];
  int t13[F(__is_enum(Int))];
  int t14[F(__is_enum(IntAr))];
  int t15[F(__is_enum(UnionAr))];
  int t16[F(__is_enum(Derives))];
  int t17[F(__is_enum(ClassType))];
}

struct Polymorph { virtual void f(); };
struct InheritPolymorph : Polymorph {};

void is_polymorphic()
{
  int t01[T(__is_polymorphic(Polymorph))];
  int t02[T(__is_polymorphic(InheritPolymorph))];

  int t11[F(__is_polymorphic(int))];
  int t12[F(__is_polymorphic(Union))];
  int t13[F(__is_polymorphic(Int))];
  int t14[F(__is_polymorphic(IntAr))];
  int t15[F(__is_polymorphic(UnionAr))];
  int t16[F(__is_polymorphic(Derives))];
  int t17[F(__is_polymorphic(ClassType))];
  int t18[F(__is_polymorphic(Enum))];
}

typedef Int& IntRef;
typedef const IntAr ConstIntAr;
typedef ConstIntAr ConstIntArAr[4];

struct HasCopy {
  HasCopy(HasCopy& cp);
};

void has_trivial_default_constructor() {
  int t01[T(__has_trivial_constructor(Int))];
  int t02[T(__has_trivial_constructor(IntAr))];
  int t03[T(__has_trivial_constructor(Union))];
  int t04[T(__has_trivial_constructor(UnionAr))];
  int t05[T(__has_trivial_constructor(POD))];
  int t06[T(__has_trivial_constructor(Derives))];
  int t07[T(__has_trivial_constructor(ConstIntAr))];
  int t08[T(__has_trivial_constructor(ConstIntArAr))];
  int t09[T(__has_trivial_constructor(HasDest))];
  int t10[T(__has_trivial_constructor(HasPriv))];
  int t11[F(__has_trivial_constructor(HasCons))];
  int t12[F(__has_trivial_constructor(HasRef))];
  int t13[F(__has_trivial_constructor(HasCopy))];
  int t14[F(__has_trivial_constructor(IntRef))];
  int t15[T(__has_trivial_constructor(HasAssign))];
  int t16[T(__has_trivial_constructor(const Int))];
  int t17[T(__has_trivial_constructor(NonPODAr))];
  int t18[F(__has_trivial_constructor(VirtAr))];
}

void has_trivial_copy_constructor() {
  int t01[T(__has_trivial_copy(Int))];
  int t02[T(__has_trivial_copy(IntAr))];
  int t03[T(__has_trivial_copy(Union))];
  int t04[T(__has_trivial_copy(UnionAr))];
  int t05[T(__has_trivial_copy(POD))];
  int t06[T(__has_trivial_copy(Derives))];
  int t07[T(__has_trivial_copy(ConstIntAr))];
  int t08[T(__has_trivial_copy(ConstIntArAr))];
  int t09[T(__has_trivial_copy(HasDest))];
  int t10[T(__has_trivial_copy(HasPriv))];
  int t11[T(__has_trivial_copy(HasCons))];
  int t12[T(__has_trivial_copy(HasRef))];
  int t13[F(__has_trivial_copy(HasCopy))];
  int t14[T(__has_trivial_copy(IntRef))];
  int t15[T(__has_trivial_copy(HasAssign))];
  int t16[T(__has_trivial_copy(const Int))];
  int t17[F(__has_trivial_copy(NonPODAr))];
  int t18[F(__has_trivial_copy(VirtAr))];
}

void has_trivial_copy_assignment() {
  int t01[T(__has_trivial_assign(Int))];
  int t02[T(__has_trivial_assign(IntAr))];
  int t03[T(__has_trivial_assign(Union))];
  int t04[T(__has_trivial_assign(UnionAr))];
  int t05[T(__has_trivial_assign(POD))];
  int t06[T(__has_trivial_assign(Derives))];
  int t07[F(__has_trivial_assign(ConstIntAr))];
  int t08[F(__has_trivial_assign(ConstIntArAr))];
  int t09[T(__has_trivial_assign(HasDest))];
  int t10[T(__has_trivial_assign(HasPriv))];
  int t11[T(__has_trivial_assign(HasCons))];
  int t12[T(__has_trivial_assign(HasRef))];
  int t13[T(__has_trivial_assign(HasCopy))];
  int t14[F(__has_trivial_assign(IntRef))];
  int t15[F(__has_trivial_assign(HasAssign))];
  int t16[F(__has_trivial_assign(const Int))];
  int t17[F(__has_trivial_assign(NonPODAr))];
  int t18[F(__has_trivial_assign(VirtAr))];
}

void has_trivial_destructor() {
  int t01[T(__has_trivial_destructor(Int))];
  int t02[T(__has_trivial_destructor(IntAr))];
  int t03[T(__has_trivial_destructor(Union))];
  int t04[T(__has_trivial_destructor(UnionAr))];
  int t05[T(__has_trivial_destructor(POD))];
  int t06[T(__has_trivial_destructor(Derives))];
  int t07[T(__has_trivial_destructor(ConstIntAr))];
  int t08[T(__has_trivial_destructor(ConstIntArAr))];
  int t09[F(__has_trivial_destructor(HasDest))];
  int t10[T(__has_trivial_destructor(HasPriv))];
  int t11[T(__has_trivial_destructor(HasCons))];
  int t12[T(__has_trivial_destructor(HasRef))];
  int t13[T(__has_trivial_destructor(HasCopy))];
  int t14[T(__has_trivial_destructor(IntRef))];
  int t15[T(__has_trivial_destructor(HasAssign))];
  int t16[T(__has_trivial_destructor(const Int))];
  int t17[T(__has_trivial_destructor(NonPODAr))];
  int t18[T(__has_trivial_destructor(VirtAr))];
}
