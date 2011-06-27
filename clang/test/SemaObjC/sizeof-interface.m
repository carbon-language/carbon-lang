// RUN: %clang_cc1 -fobjc-nonfragile-abi -verify -fsyntax-only %s

@class I0;

// rdar://6811884
int g0 = sizeof(I0); // expected-error{{invalid application of 'sizeof' to an incomplete type 'I0'}}

// rdar://6821047
void *g3(I0 *P) {
  P = P+5;        // expected-error {{arithmetic on a pointer to an incomplete type 'I0'}}

  return &P[4];   // expected-error{{subscript of pointer to incomplete type 'I0'}}
}



@interface I0 {
@public
  char x[4];
}

@property int p0;
@end

// size == 4
int g1[ sizeof(I0)     // expected-error {{invalid application of 'sizeof' to interface 'I0' in non-fragile ABI}}
       == 4 ? 1 : -1];

@implementation I0
@synthesize p0 = _p0;
@end

// size == 4 (we do not include extended properties in the
// sizeof).
int g2[ sizeof(I0)   // expected-error {{invalid application of 'sizeof' to interface 'I0' in non-fragile ABI}}
       == 4 ? 1 : -1];

@interface I1
@property int p0;
@end

@implementation I1
@synthesize p0 = _p0;
@end

typedef struct { @defs(I1); } I1_defs; // expected-error {{invalid application of @defs in non-fragile ABI}}

// FIXME: This is currently broken due to the way the record layout we
// create is tied to whether we have seen synthesized properties. Ugh.
// int g3[ sizeof(I1) == 0 ? 1 : -1];

// rdar://6821047
int bar(I0 *P) {
  P = P+5;  // expected-error {{arithmetic on pointer to interface 'I0', which is not a constant size in non-fragile ABI}}
  P = 5+P;  // expected-error {{arithmetic on pointer to interface 'I0', which is not a constant size in non-fragile ABI}}
  P = P-5;  // expected-error {{arithmetic on pointer to interface 'I0', which is not a constant size in non-fragile ABI}}
  
  return P[4].x[2];  // expected-error {{subscript requires size of interface 'I0', which is not constant in non-fragile ABI}}
}


@interface I @end

@interface XCAttributeRunDirectNode
{
    @public
    unsigned long attributeRuns[1024 + sizeof(I)]; // expected-error {{invalid application of 'sizeof' to interface 'I' in non-fragile ABI}}
    int i;
}
@end

@implementation XCAttributeRunDirectNode

- (unsigned long)gatherStats:(id )stats
{
        return attributeRuns[i];
}
@end


@interface Foo @end

int foo()
{
  Foo *f;
  
  // Both of these crash clang nicely
  ++f; 	// expected-error {{arithmetic on pointer to interface 'Foo', which is not a constant size in non-fragile ABI}}
  --f; 	// expected-error {{arithmetic on pointer to interface 'Foo', which is not a constant size in non-fragile ABI}}
}
