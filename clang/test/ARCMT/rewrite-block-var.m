// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-nonfragile-abi -fblocks -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.7 -fobjc-nonfragile-abi -fblocks -fsyntax-only %s > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface Foo : NSObject
-(Foo *)something;
@end

void bar(void (^block)());

void test1(Foo *p) {
  __block Foo *x = p; // __block used just to break cycle.
  bar(^{
    [x something];
  });
}

void test2(Foo *p) {
  __block Foo *x; // __block used as output variable.
  bar(^{
    x = [p something];
  });
}
