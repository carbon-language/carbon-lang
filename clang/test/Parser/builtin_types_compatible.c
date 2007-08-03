// RUN: clang -parse-ast-check

extern void funcInt(int);
extern void funcFloat(float);
extern void funcDouble(double);
// figure out why "char *" doesn't work (with gcc, nothing to do with clang)
//extern void funcCharPtr(char *);

#define func(expr) \
  ({ \
    typeof(expr) tmp; \
    if (__builtin_types_compatible_p(typeof(expr), int)) funcInt(tmp); \
    else if (__builtin_types_compatible_p(typeof(expr), float)) funcFloat(tmp); \
    else if (__builtin_types_compatible_p(typeof(expr), double)) funcDouble(tmp); \
  })
#define func_choose(expr) \
  __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), int), funcInt(expr), \
    __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), float), funcFloat(expr), \
      __builtin_choose_expr(__builtin_types_compatible_p(typeof(expr), double), funcDouble(expr), \
  (void)0)))

static void test()
{
  int a;
  float b;
  double d;

  func(a);
  func(b);
  func(d);
  func_choose(a);
  func_choose(b);
  func_choose(d);
}

