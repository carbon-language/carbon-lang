// RUN: %clang_cc1 -fsyntax-only -verify %s

extern int funcInt(int);
extern float funcFloat(float);
extern double funcDouble(double);
// figure out why "char *" doesn't work (with gcc, nothing to do with clang)
//extern void funcCharPtr(char *);

#define func(expr) \
  do { \
    typeof(expr) tmp; \
    if (__builtin_types_compatible_p(typeof(expr), int)) funcInt(tmp); \
    else if (__builtin_types_compatible_p(typeof(expr), float)) funcFloat(tmp); \
    else if (__builtin_types_compatible_p(typeof(expr), double)) funcDouble(tmp); \
  } while (0)
#define func_choose(expr) \
  __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), int), funcInt(expr), \
    __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), float), funcFloat(expr), \
      __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), double), funcDouble(expr), (void)0)))

static void test(void)
{
  int a;
  float b;
  double d;

  func(a);
  func(b);
  func(d);
  a = func_choose(a);
  b = func_choose(b);
  d = func_choose(d);

  int c; 
  struct xx { int a; } x, y;
  
  c = __builtin_choose_expr(a+3-7, b, x); // expected-error{{'__builtin_choose_expr' requires a constant expression}}
  c = __builtin_choose_expr(0, b, x); // expected-error{{assigning to 'int' from incompatible type 'struct xx'}}
  c = __builtin_choose_expr(5+3-7, b, x);
  y = __builtin_choose_expr(4+3-7, b, x);

}

enum E1 { E1Foo };
enum E2 { E2Foo };

static void testGccCompatibility(void) {
  _Static_assert(__builtin_types_compatible_p(const volatile int, int), "");
  _Static_assert(__builtin_types_compatible_p(int[5], int[]), "");
  _Static_assert(!__builtin_types_compatible_p(int[5], int[4]), "");
  _Static_assert(!__builtin_types_compatible_p(int *, int **), "");
  _Static_assert(!__builtin_types_compatible_p(const int *, int *), "");
  _Static_assert(!__builtin_types_compatible_p(enum E1, enum E2), "");

  // GCC's __builtin_types_compatible_p ignores qualifiers on arrays.
  _Static_assert(__builtin_types_compatible_p(const int[4], int[4]), "");
  _Static_assert(__builtin_types_compatible_p(int[4], const int[4]), "");
  _Static_assert(__builtin_types_compatible_p(const int[5][4], int[][4]), "");
  _Static_assert(!__builtin_types_compatible_p(const int(*)[], int(*)[]), "");
}
