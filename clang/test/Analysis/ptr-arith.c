// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify -triple x86_64-apple-darwin9 %s
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify -triple i686-apple-darwin9 %s

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
}
