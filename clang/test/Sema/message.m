// RUN: clang -fsyntax-only -verify %s

@interface foo
- (void)meth;
@end

@implementation foo
- (void) contents {}			// No declaration in @interface!
- (void) meth { [self contents]; } // expected-warning {{method '-contents' not found (return type defaults to 'id')}}
@end

