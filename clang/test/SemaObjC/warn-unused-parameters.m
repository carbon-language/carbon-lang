// RUN: clang -fsyntax-only -Wunused -Xclang -verify %s

@interface foo
- (int)meth: (int)x: (int)y: (int)z ;
@end

@implementation foo
- (int) meth: (int)x: 
              (int)y: // expected-warning{{unused}} 
              (int) __attribute__((unused))z { return x; }
@end
