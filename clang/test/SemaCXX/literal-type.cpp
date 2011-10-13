// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

static_assert(__is_literal(int), "fail");
static_assert(__is_literal_type(int), "fail"); // alternate spelling for GCC
static_assert(__is_literal(void*), "fail");
enum E { E1 };
static_assert(__is_literal(E), "fail");
static_assert(__is_literal(decltype(E1)), "fail");
typedef int IAR[10];
static_assert(__is_literal(IAR), "fail");
typedef int Vector __attribute__((vector_size(16)));
typedef int VectorExt __attribute__((ext_vector_type(4)));
static_assert(__is_literal(Vector), "fail");
static_assert(__is_literal(VectorExt), "fail");

// C++0x [basic.types]p10:
//   A type is a literal type if it is:
//    [...]
//    -- a class type that has all of the following properties:
//        -- it has a trivial destructor
//        -- every constructor call and full-expression in the
//           brace-or-equal-initializers for non-static data members (if an) is
//           a constant expression,
//        -- it is an aggregate type or has at least one constexpr constructor
//           or constructor template that is not a copy or move constructor, and
//        -- it has all non-static data members and base classes of literal
//           types
struct Empty {};
struct LiteralType {
  int x;
  E e;
  IAR arr;
  Empty empty;
  int method();
};
struct HasDtor { ~HasDtor(); };

class NonAggregate { int x; };
struct HasNonLiteralBase : NonAggregate {};
struct HasNonLiteralMember { HasDtor x; };

static_assert(__is_literal(Empty), "fail");
static_assert(__is_literal(LiteralType), "fail");
static_assert(!__is_literal(HasDtor), "fail");
static_assert(!__is_literal(NonAggregate), "fail");
static_assert(!__is_literal(HasNonLiteralBase), "fail");
static_assert(!__is_literal(HasNonLiteralMember), "fail");

// FIXME: Test constexpr constructors and non-static members with initializers
// when Clang supports them:
#if 0
extern int f();
struct HasNonConstExprMemInit {
  int x = f();
  constexpr HasNonConstExprMemInit(int y) {}
};
static_assert(!__is_literal(HasNonConstExprMemInit), "fail");

class HasConstExprCtor {
  int x;
public:
  constexpr HasConstExprCtor(int x) : x(x) {}
};
template <typename T> class HasConstExprCtorTemplate {
  T x;
public:
  template <typename U> constexpr HasConstExprCtorTemplate(U y) : x(y) {}
};
static_assert(__is_literal(HasConstExprCtor), "fail");
static_assert(__is_literal(HasConstExprCtorTemplate), "fail");
#endif
