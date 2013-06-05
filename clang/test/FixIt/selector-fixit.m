// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -Wundeclared-selector -fixit %t
// RUN: %clang_cc1 -x objective-c -Wundeclared-selector -Werror %t
// rdar://14039037

@interface NSObject @end

@interface LogoutController : NSObject
- (void)close;
- (void)closed;
- (void) open : (id) file_id;
@end

@implementation LogoutController

- (void)close  { }
- (void)closed  { }

- (SEL)Meth
{
  return @selector(cloze);
}
- (void) open : (id) file_id {}

- (SEL)Meth1
{
  return @selector(ope:);
}

@end
