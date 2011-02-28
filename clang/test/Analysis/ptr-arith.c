// RUN: %clang_cc1 -analyze -analyzer-checker=core.experimental.FixedAddr,core.experimental.PointerArithm,core.experimental.PointerSub -analyzer-store=region -verify -triple x86_64-apple-darwin9 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core.experimental.FixedAddr,core.experimental.PointerArithm,core.experimental.PointerSub -analyzer-store=region -verify -triple i686-apple-darwin9 %s

// Used to trigger warnings for unreachable paths.
#define WARN do { int a, b; int c = &b-&a; } while (0)

void f1() {
  int a[10];
  int *p = a;
  ++p;
}

char* foo();

void f2() {
  char *p = foo();
  ++p;
}

// This test case checks if we get the right rvalue type of a TypedViewRegion.
// The ElementRegion's type depends on the array region's rvalue type. If it was
// a pointer type, we would get a loc::SymbolVal for '*p'.
void* memchr();
static int
domain_port (const char *domain_b, const char *domain_e,
             const char **domain_e_ptr)
{
  int port = 0;
  
  const char *p;
  const char *colon = memchr (domain_b, ':', domain_e - domain_b);
  
  for (p = colon + 1; p < domain_e ; p++)
    port = 10 * port + (*p - '0');
  return port;
}

void f3() {
  int x, y;
  int d = &y - &x; // expected-warning{{Subtraction of two pointers that do not point to the same memory chunk may cause incorrect result.}}

  int a[10];
  int *p = &a[2];
  int *q = &a[8];
  d = q-p; // no-warning
}

void f4() {
  int *p;
  p = (int*) 0x10000; // expected-warning{{Using a fixed address is not portable because that address will probably not be valid in all environments or platforms.}}
}

void f5() {
  int x, y;
  int *p;
  p = &x + 1;  // expected-warning{{Pointer arithmetic done on non-array variables means reliance on memory layout, which is dangerous.}}

  int a[10];
  p = a + 1; // no-warning
}

// Allow arithmetic on different symbolic regions.
void f6(int *p, int *q) {
  int d = q - p; // no-warning
}

void null_operand(int *a) {
start:
  // LHS is a label, RHS is NULL
  if (&&start == 0)
    WARN; // no-warning
  if (&&start <  0)
    WARN; // no-warning
  if (&&start <= 0)
    WARN; // no-warning
  if (!(&&start != 0))
    WARN; // no-warning
  if (!(&&start >  0))
    WARN; // no-warning
  if (!(&&start >= 0))
    WARN; // no-warning
  if (!(&&start - 0))
    WARN; // no-warning

  // LHS is a non-symbolic value, RHS is NULL
  if (&a == 0)
    WARN; // no-warning
  if (&a <  0)
    WARN; // no-warning
  if (&a <= 0)
    WARN; // no-warning
  if (!(&a != 0))
    WARN; // no-warning
  if (!(&a >  0))
    WARN; // no-warning
  if (!(&a >= 0))
    WARN; // no-warning

  if (!(&a - 0)) // expected-warning{{Pointer arithmetic done on non-array variables}}
    WARN; // no-warning

  // LHS is NULL, RHS is non-symbolic
  // The same code is used for labels and non-symbolic values.
  if (0 == &a)
    WARN; // no-warning
  if (0 >  &a)
    WARN; // no-warning
  if (0 >= &a)
    WARN; // no-warning
  if (!(0 != &a))
    WARN; // no-warning
  if (!(0 <  &a))
    WARN; // no-warning
  if (!(0 <= &a))
    WARN; // no-warning

  // LHS is a symbolic value, RHS is NULL
  if (a == 0)
    WARN; // expected-warning{{}}
  if (a <  0)
    WARN; // no-warning
  if (a <= 0)
    WARN; // expected-warning{{}}
  if (!(a != 0))
    WARN; // expected-warning{{}}
  if (!(a >  0))
    WARN; // expected-warning{{}}
  if (!(a >= 0))
    WARN; // no-warning
  if (!(a - 0))
    WARN; // expected-warning{{}}

  // LHS is NULL, RHS is a symbolic value
  if (0 == a)
    WARN; // expected-warning{{}}
  if (0 >  a)
    WARN; // no-warning
  if (0 >= a)
    WARN; // expected-warning{{}}
  if (!(0 != a))
    WARN; // expected-warning{{}}
  if (!(0 <  a))
    WARN; // expected-warning{{}}
  if (!(0 <= a))
    WARN; // no-warning
}

void const_locs() {
  char *a = (char*)0x1000;
  char *b = (char*)0x1100;
start:
  if (a==b)
    WARN; // no-warning
  if (!(a!=b))
    WARN; // no-warning
  if (a>b)
    WARN; // no-warning
  if (b<a)
    WARN; // no-warning
  if (a>=b)
    WARN; // no-warning
  if (b<=a)
    WARN; // no-warning
  if (b-a != 0x100)
    WARN; // no-warning

  if (&&start == a)
    WARN; // expected-warning{{}}
  if (a == &&start)
    WARN; // expected-warning{{}}
  if (&a == (char**)a)
    WARN; // expected-warning{{}}
  if ((char**)a == &a)
    WARN; // expected-warning{{}}
}

void array_matching_types() {
  int array[10];
  int *a = &array[2];
  int *b = &array[5];

  if (a==b)
    WARN; // no-warning
  if (!(a!=b))
    WARN; // no-warning
  if (a>b)
    WARN; // no-warning
  if (b<a)
    WARN; // no-warning
  if (a>=b)
    WARN; // no-warning
  if (b<=a)
    WARN; // no-warning
  if ((b-a) == 0)
    WARN; // no-warning
}

// This takes a different code path than array_matching_types()
void array_different_types() {
  int array[10];
  int *a = &array[2];
  char *b = (char*)&array[5];

  if (a==b) // expected-warning{{comparison of distinct pointer types}}
    WARN; // no-warning
  if (!(a!=b)) // expected-warning{{comparison of distinct pointer types}}
    WARN; // no-warning
  if (a>b) // expected-warning{{comparison of distinct pointer types}}
    WARN; // no-warning
  if (b<a) // expected-warning{{comparison of distinct pointer types}}
    WARN; // no-warning
  if (a>=b) // expected-warning{{comparison of distinct pointer types}}
    WARN; // no-warning
  if (b<=a) // expected-warning{{comparison of distinct pointer types}}
    WARN; // no-warning
}

struct test { int x; int y; };
void struct_fields() {
  struct test a, b;

  if (&a.x == &a.y)
    WARN; // no-warning
  if (!(&a.x != &a.y))
    WARN; // no-warning
  if (&a.x > &a.y)
    WARN; // no-warning
  if (&a.y < &a.x)
    WARN; // no-warning
  if (&a.x >= &a.y)
    WARN; // no-warning
  if (&a.y <= &a.x)
    WARN; // no-warning

  if (&a.x == &b.x)
    WARN; // no-warning
  if (!(&a.x != &b.x))
    WARN; // no-warning
  if (&a.x > &b.x)
    WARN; // expected-warning{{}}
  if (&b.x < &a.x)
    WARN; // expected-warning{{}}
  if (&a.x >= &b.x)
    WARN; // expected-warning{{}}
  if (&b.x <= &a.x)
    WARN; // expected-warning{{}}
}

void mixed_region_types() {
  struct test s;
  int array[2];
  void *a = &array, *b = &s;

  if (&a == &b)
    WARN; // no-warning
  if (!(&a != &b))
    WARN; // no-warning
  if (&a > &b)
    WARN; // expected-warning{{}}
  if (&b < &a)
    WARN; // expected-warning{{}}
  if (&a >= &b)
    WARN; // expected-warning{{}}
  if (&b <= &a)
    WARN; // expected-warning{{}}
}

void symbolic_region(int *p) {
  int a;

  if (&a == p)
    WARN; // expected-warning{{}}
  if (&a != p)
    WARN; // expected-warning{{}}
  if (&a > p)
    WARN; // expected-warning{{}}
  if (&a < p)
    WARN; // expected-warning{{}}
  if (&a >= p)
    WARN; // expected-warning{{}}
  if (&a <= p)
    WARN; // expected-warning{{}}
}

void PR7527 (int *p) {
  if (((int) p) & 1) // not crash
    return;
}
