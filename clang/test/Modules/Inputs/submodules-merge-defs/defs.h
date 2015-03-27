struct A { int a_member; };
inline int use_a(A a) { return a.a_member; }

class B {
  struct Inner1 {};
public:
  struct Inner2;
  template<typename T> void f();
};
// Check that lookup and access checks are performed in the right context.
struct B::Inner2 : Inner1 {};
template<typename T> void B::f() {}

// Check that base-specifiers are correctly disambiguated.
template<int N> struct C_Base { struct D { constexpr operator int() const { return 0; } }; };
const int C_Const = 0;
struct C1 : C_Base<C_Base<0>::D{}> {} extern c1;
struct C2 : C_Base<C_Const<0>::D{} extern c2;

typedef struct { int a; void f(); struct X; } D;
struct D::X { int dx; } extern dx;
inline int use_dx(D::X dx) { return dx.dx; }

template<typename T> int E(T t) { return t; }

template<typename T> struct F {
  int f();
  template<typename U> int g();
};
template<typename T> int F<T>::f() { return 0; }
template<typename T> template<typename U> int F<T>::g() { return 0; }
