// RUN: clang-cc -triple x86_64-apple-darwin9 -verify -fsyntax-only %s

@class I0;

// rdar://6811884
int g0 = sizeof(I0); // expected-error{{invalid application of 'sizeof' to a forward declared interface 'I0'}}

@interface I0 {
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

typedef struct { @defs(I1) } I1_defs; // expected-error {{invalid application of @defs in non-fragile ABI}}

// FIXME: This is currently broken due to the way the record layout we
// create is tied to whether we have seen synthesized properties. Ugh.
// int g3[ sizeof(I1) == 0 ? 1 : -1];
