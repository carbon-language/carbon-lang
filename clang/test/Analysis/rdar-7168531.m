// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -triple i386-apple-darwin10 -analyzer-store=region
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -triple i386-apple-darwin10 -analyzer-store=basic

// Note that the target triple is important for this test case.  It specifies that we use the
// fragile Objective-C ABI.

@interface Foo {
  int x;
}
@end

@implementation Foo
static Foo* bar(Foo *p) {
  if (p->x)
   return ++p;  // This is only valid for the fragile ABI.

  return p;
}
@end
