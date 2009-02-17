// RUN: clang -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=basic --verify -fblocks %s &&
// RUN: clang -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=range --verify -fblocks %s &&
// RUN: clang -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=basic --verify -fblocks %s &&
// RUN: clang -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=range --verify -fblocks %s


// Reduced test case from crash in <rdar://problem/6253157>
@class NSObject;
@interface A @end
@implementation A
- (void)foo:(void (^)(NSObject *x))block {
  if (!((block != ((void *)0)))) {}
}
@end

// Reduced test case from crash in PR 2796;
//  http://llvm.org/bugs/show_bug.cgi?id=2796

unsigned foo(unsigned x) { return __alignof__((x)) + sizeof(x); }

// Improvement to path-sensitivity involving compound assignments.
//  Addresses false positive in <rdar://problem/6268365>
//

unsigned r6268365Aux();

void r6268365() {
  unsigned x = 0;
  x &= r6268365Aux();
  unsigned j = 0;
    
  if (x == 0) ++j;
  if (x == 0) x = x / j; // no-warning
}

void divzeroassume(unsigned x, unsigned j) {  
  x /= j;  
  if (j == 0) x /= 0;     // no-warning
  if (j == 0) x /= j;     // no-warning
  if (j == 0) x = x / 0;  // no-warning
}

void divzeroassumeB(unsigned x, unsigned j) {  
  x = x / j;  
  if (j == 0) x /= 0;     // no-warning
  if (j == 0) x /= j;     // no-warning
  if (j == 0) x = x / 0;  // no-warning
}

// InitListExpr processing

typedef float __m128 __attribute__((__vector_size__(16), __may_alias__));
__m128 return128() {
  // This compound literal has a Vector type.  We currently just
  // return UnknownVal.
  return __extension__(__m128) { 0.0f, 0.0f, 0.0f, 0.0f };
}

typedef long long __v2di __attribute__ ((__vector_size__ (16)));
typedef long long __m128i __attribute__ ((__vector_size__ (16), __may_alias__));
__m128i vec128i(long long __q1, long long __q0) {
  // This compound literal returns true for both isVectorType() and 
  // isIntegerType().
  return __extension__ (__m128i)(__v2di){ __q0, __q1 };
}

// Zero-sized VLAs.
void check_zero_sized_VLA(int x) {
  if (x)
    return;

  int vla[x]; // expected-warning{{VLAs with no elements have undefined behavior}}
}

void check_uninit_sized_VLA() {
  int x;
  int vla[x]; // expected-warning{{The expression used to specify the number of elements in the VLA 'vla' evaluates to an undefined or garbage value.}}
}

// sizeof(void)
// - Tests a regression reported in PR 3211: http://llvm.org/bugs/show_bug.cgi?id=3211
void handle_sizeof_void(unsigned flag) {
  int* p = 0;

  if (flag) {
    if (sizeof(void) == 1)
      return;
    // Infeasible.
    *p = 1; // no-warning
  }
  
  void* q;
  
  if (!flag) {
    if (sizeof(*q) == 1)
      return;
    // Infeasibe.
    *p = 1; // no-warning
  }
    
  // Infeasible.
  *p = 1; // no-warning
}

// PR 3422
void pr3422_helper(char *p);
void pr3422() {
  char buf[100];
  char *q = &buf[10];
  pr3422_helper(&q[1]);
}

// PR 3543 (handle empty statement expressions)
int pr_3543(void) {
  ({});
}


