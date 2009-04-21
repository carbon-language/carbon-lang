// RUN: clang-cc -triple x86_64-apple-darwin9 -fsyntax-only %s

@class I0;
// FIXME: Reject sizeof on incomplete interface; this breaks the test!
//int g0 = sizeof(I0); // exxpected-error{{invalid application of 'sizeof' to an incomplete type ...}}

@interface I0 {
  char x[4];
}

@property int p0;
@end

// size == 4
int g1[ sizeof(I0) == 4 ? 1 : -1];

@implementation I0
@synthesize p0 = _p0;
@end

// size == 4 (we do not include extended properties in the
// sizeof).
int g2[ sizeof(I0) == 4 ? 1 : -1];

@interface I1
@property int p0;
@end

@implementation I1
@synthesize p0 = _p0;
@end

// FIXME: This is currently broken due to the way the record layout we
// create is tied to whether we have seen synthesized properties. Ugh.
// int g3[ sizeof(I1) == 0 ? 1 : -1];
