// RUN: clang -fsyntax-only -verify %s

@interface NSSound
@end
@interface NSFont
@end

@interface NSSound (Adds)
@end

@implementation NSSound (Adds)
- foo {
  return self;
}
- (void)setFoo:obj {
}
@end

@implementation NSFont (Adds)

- xx {
  NSSound *x;
  id o;

  o = [x foo]; 
  o = x.foo;
  [x setFoo:o];
  x.foo = o;
}

@end

