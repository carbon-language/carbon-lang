// RUN: clang-cc -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=basic --verify -fblocks %s &&
// RUN: clang-cc -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=range --verify -fblocks %s

// NOWORK: clang-cc -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=basic --verify -fblocks %s &&
// NOWORK: clang-cc -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=range --verify -fblocks %s

typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} - (id)init; @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;
typedef float CGFloat;
typedef struct _NSPoint {
    CGFloat x;
    CGFloat y;
} NSPoint;
typedef struct _NSSize {
    CGFloat width;
    CGFloat height;
} NSSize;
typedef struct _NSRect {
    NSPoint origin;
    NSSize size;
} NSRect;

// Reduced test case from crash in <rdar://problem/6253157>
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

  int vla[x]; // expected-warning{{Variable-length array 'vla' has zero elements (undefined behavior)}}
}

void check_uninit_sized_VLA() {
  int x;
  int vla[x]; // expected-warning{{Variable-length array 'vla' garbage value for array size}}
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

// <rdar://problem/6611677>
// This test case test the use of a vector type within an array subscript
// expression.
typedef long long __a64vector __attribute__((__vector_size__(8)));
typedef long long __a128vector __attribute__((__vector_size__(16)));
static inline __a64vector __attribute__((__always_inline__, __nodebug__))  
my_test_mm_movepi64_pi64(__a128vector a) {
  return (__a64vector)a[0];
}

// Test basic tracking of ivars associated with 'self'.
@interface SelfIvarTest : NSObject {
  int flag;
}
- (void)test_self_tracking;
@end

@implementation SelfIvarTest
- (void)test_self_tracking {
  char *p = 0;
  char c;

  if (flag)
    p = "hello";

  if (flag)
    c = *p; // no-warning
}
@end

// PR 3770
char pr3770(int x) {
  int y = x & 0x2;
  char *p = 0;
  if (y == 1)
    p = "hello";

  if (y == 1)
    return p[0]; // no-warning
    
  return 'a';
}

// PR 3772
// - We just want to test that this doesn't crash the analyzer.
typedef struct st ST;
struct st { char *name; };
extern ST *Cur_Pu;

void pr3772(void)
{
  static ST *last_Cur_Pu;
  if (last_Cur_Pu == Cur_Pu) {
    return;
  } 
}

// PR 3780 - This tests that StmtIterator isn't broken for VLAs in DeclGroups.
void pr3780(int sz) { typedef double MAT[sz][sz]; }

// <rdar://problem/6695527> - Test that we don't symbolicate doubles before
// we are ready to do something with them.
int rdar6695527(double x) {
  if (!x) { return 0; }
  return 1;
}

// <rdar://problem/6708148> - Test that we properly invalidate structs
//  passed-by-reference to a function.
void pr6708148_invalidate(NSRect *x);
void pr6708148_use(NSRect x);
void pr6708148_test(void) {
  NSRect x;
  pr6708148_invalidate(&x);
  pr6708148_use(x); // no-warning
}

// Handle both kinds of noreturn attributes for pruning paths.
void rdar_6777003_noret() __attribute__((noreturn));
void rdar_6777003_analyzer_noret() __attribute__((analyzer_noreturn));

void rdar_6777003(int x) {
  int *p = 0;
  
  if (x == 1) {
    rdar_6777003_noret();
    *p = 1; // no-warning;    
  }
  
  if (x == 2) {
    rdar_6777003_analyzer_noret();
    *p = 1; // no-warning;
  }
  
  *p = 1; // expected-warning{{Dereference of null pointer}}  
}

// For pointer arithmetic, --/++ should be treated as preserving non-nullness,
// regardless of how well the underlying StoreManager reasons about pointer
// arithmetic.
// <rdar://problem/6777209>
void rdar_6777209(char *p) {
  if (p == 0)
    return;
  
  ++p;
  
  // This branch should always be infeasible.
  if (p == 0)
    *p = 'c'; // no-warning
}

// PR 4033.  A symbolic 'void *' pointer can be used as the address for a
// computed goto.
typedef void *Opcode;
Opcode pr_4033_getOpcode();
void pr_4033(void) {
next_opcode:
  {
    Opcode op = pr_4033_getOpcode();
    if (op) goto *op;
  }
}

// Test invalidating pointers-to-pointers with slightly different types.  This
// example came from a recent false positive due to a regression where the
// branch condition was falsely reported as being uninitialized.
void invalidate_by_ref(char **x);
int test_invalidate_by_ref() {
  unsigned short y;
  invalidate_by_ref((char**) &y);
  if (y) // no-warning
    return 1;
  return 0;  
}

