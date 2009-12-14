// RUN: clang -cc1 -analyze -checker-cfref -analyzer-store=basic -verify %s
// RUN: clang -cc1 -analyze -checker-cfref -analyzer-store=region -verify %s

typedef unsigned int NSUInteger;

@interface A
- (NSUInteger)foo;
@end

NSUInteger f8(A* x){
  const NSUInteger n = [x foo];
  int* bogus;  

  if (n > 0) {    // tests const cast transfer function logic
    NSUInteger i;
    
    for (i = 0; i < n; ++i)
      bogus = 0;

    if (bogus)  // no-warning
      return n+1;
  }
  
  return n;
}
