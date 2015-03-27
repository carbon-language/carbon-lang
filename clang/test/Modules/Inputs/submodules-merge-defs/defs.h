struct A { int a_member; };
namespace { inline int use_a(A a) { return a.a_member; } }

class B {
  struct Inner1 {};
public:
  struct Inner2;
};
// Check that lookup and access checks are performed in the right context.
struct B::Inner2 : Inner1 {};

// Check that base-specifiers are correctly disambiguated.
template<int N> struct C_Base { struct D { constexpr operator int() const { return 0; } }; };
const int C_Const = 0;
struct C1 : C_Base<C_Base<0>::D{}> {} extern c1;
struct C2 : C_Base<C_Const<0>::D{} extern c2;

typedef struct { int a; void f(); struct X; } D;
struct D::X { int dx; } extern dx;
namespace { inline int use_dx(D::X dx) { return dx.dx; } }
