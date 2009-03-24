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
