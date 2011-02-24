// RUN: %clang_cc1 -triple i386-apple-darwin8 -analyze -analyzer-checker=core.experimental -analyzer-check-objc-mem -analyzer-constraints=range -analyzer-store=region -verify %s

// <rdar://problem/6888289> - This test case shows that a nil instance
// variable can possibly be initialized by a method.
typedef struct RDar6888289_data {
  long data[100];
} RDar6888289_data;

@interface RDar6888289
{
  RDar6888289 *x;
}
- (RDar6888289_data) test;
- (RDar6888289_data) test2;
- (void) invalidate;
- (RDar6888289_data) getData;
@end

@implementation RDar6888289
- (RDar6888289_data) test {
  if (!x)
    [self invalidate];
  return [x getData];
}
- (RDar6888289_data) test2 {
  if (!x) {}
  return [x getData]; // expected-warning{{The receiver of message 'getData' is nil and returns a value of type 'RDar6888289_data' that will be garbage}}
}

- (void) invalidate {
  x = self;
}

- (RDar6888289_data) getData {
  return (RDar6888289_data) { 0 };
}
@end

