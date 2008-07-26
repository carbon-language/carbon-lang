// RUN: clang -fsyntax-only %s

@interface foo
- (int)meth;
@end

@implementation foo
- (int) meth { return [self meth]; }
@end

