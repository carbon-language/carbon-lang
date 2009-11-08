// RUN: clang-cc -triple i386-apple-darwin9 -analyze -checker-cfref -analyzer-constraints=basic -analyzer-store=basic %s -verify
// RUN: clang-cc -triple i386-apple-darwin9 -analyze -checker-cfref -analyzer-constraints=basic -analyzer-store=region %s -verify

@interface MyClass {}
- (void *)voidPtrM;
- (int)intM;
- (long long)longlongM;
- (double)doubleM;
- (long double)longDoubleM;
- (void)voidM;
@end
@implementation MyClass
- (void *)voidPtrM { return (void *)0; }
- (int)intM { return 0; }
- (long long)longlongM { return 0; }
- (double)doubleM { return 0.0; }
- (long double)longDoubleM { return 0.0; }
- (void)voidM {}
@end

void createFoo() {
  MyClass *obj = 0;  
  
  void *v = [obj voidPtrM]; // no-warning
  int i = [obj intM]; // no-warning
}

void createFoo2() {
  MyClass *obj = 0;  
  
  long double ld = [obj longDoubleM]; // expected-warning{{The receiver in the message expression is 'nil' and results in the returned value}}
}

void createFoo3() {
  MyClass *obj;
  obj = 0;  
  
  long long ll = [obj longlongM]; // expected-warning{{The receiver in the message expression is 'nil' and results in the returned value}}
}

void createFoo4() {
  MyClass *obj = 0;  
  
  double d = [obj doubleM]; // expected-warning{{The receiver in the message expression is 'nil' and results in the returned value}}
}

void createFoo5() {
  MyClass *obj = @"";  
  
  double d = [obj doubleM]; // no-warning
}

void handleNilPruneLoop(MyClass *obj) {
  if (!!obj)
    return;
  
  // Test if [obj intM] evaluates to 0, thus pruning the entire loop.
  for (int i = 0; i < [obj intM]; i++) {
    long long j = [obj longlongM]; // no-warning
  }
  
  long long j = [obj longlongM]; // expected-warning{{The receiver in the message expression is 'nil' and results in the returned value}}
}

int handleVoidInComma() {
  MyClass *obj = 0;
  return [obj voidM], 0;
}
