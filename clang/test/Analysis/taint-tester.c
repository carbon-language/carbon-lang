// RUN: %clang_cc1  -analyze -analyzer-checker=experimental.security.taint,debug.TaintTest -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

#define BUFSIZE 10
int Buffer[BUFSIZE];

struct XYStruct {
  int x;
  int y;
  char z;
};

void taintTracking(int x) {
  int n;
  int *addr = &Buffer[0];
  scanf("%d", &n);
  addr += n;// expected-warning 2 {{tainted}}
  *addr = n; // expected-warning 3 {{tainted}}

  double tdiv = n / 30; // expected-warning 3 {{tainted}}
  char *loc_cast = (char *) n; // expected-warning {{tainted}}
  char tinc = tdiv++; // expected-warning {{tainted}}
  int tincdec = (char)tinc--; // expected-warning 2 {{tainted}}

  // Tainted ptr arithmetic/array element address.
  int tprtarithmetic1 = *(addr+1); // expected-warning 2 {{tainted}}

  // Dereference.
  int *ptr;
  scanf("%p", &ptr);
  int ptrDeref = *ptr; // expected-warning 2 {{tainted}}
  int _ptrDeref = ptrDeref + 13; // expected-warning 2 {{tainted}}

  // Pointer arithmetic + dereferencing.
  // FIXME: We fail to propagate the taint here because RegionStore does not
  // handle ElementRegions with symbolic indexes.
  int addrDeref = *addr; // expected-warning {{tainted}}
  int _addrDeref = addrDeref;

  // Tainted struct address, casts.
  struct XYStruct *xyPtr = 0;
  scanf("%p", &xyPtr);
  void *tXYStructPtr = xyPtr; // expected-warning 2 {{tainted}}
  struct XYStruct *xyPtrCopy = tXYStructPtr; // expected-warning 2 {{tainted}}
  int ptrtx = xyPtr->x;// expected-warning 2 {{tainted}}
  int ptrty = xyPtr->y;// expected-warning 2 {{tainted}}

  // Taint on fields of a struct.
  struct XYStruct xy = {2, 3, 11};
  scanf("%d", &xy.y);
  scanf("%d", &xy.x);
  int tx = xy.x; // expected-warning {{tainted}}
  int ty = xy.y; // FIXME: This should be tainted as well.
  char ntz = xy.z;// no warning
}

void BitwiseOp(int in, char inn) {
  // Taint on bitwise operations, integer to integer cast.
  int m;
  int x = 0;
  scanf("%d", &x);
  int y = (in << (x << in)) * 5;// expected-warning 4 {{tainted}}
  // The next line tests integer to integer cast.
  int z = y & inn; // expected-warning 2 {{tainted}}
  if (y == 5) // expected-warning 2 {{tainted}}
    m = z | z;// expected-warning 4 {{tainted}}
  else
    m = inn;
  int mm = m; // expected-warning   {{tainted}}
}
