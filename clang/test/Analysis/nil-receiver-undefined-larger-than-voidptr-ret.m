// RUN: clang -cc1 -triple i386-apple-darwin8 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-constraints=basic -analyzer-store=basic %s  2>&1 | FileCheck -check-prefix=darwin8 %s
// RUN: clang -cc1 -triple i386-apple-darwin8 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-constraints=basic -analyzer-store=region %s 2>&1 | FileCheck -check-prefix=darwin8 %s
// RUN: clang -cc1 -triple i386-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-constraints=basic -analyzer-store=basic %s 2>&1 | FileCheck -check-prefix=darwin9 %s
// RUN: clang -cc1 -triple i386-apple-darwin9 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-constraints=basic -analyzer-store=region %s 2>&1 | FileCheck -check-prefix=darwin9 %s

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
  
  long double ld = [obj longDoubleM]; // expected-warning{{The receiver of message 'longDoubleM' is nil and returns a value of type 'long double' that will be garbage}}
}

void createFoo3() {
  MyClass *obj;
  obj = 0;  
  
  long long ll = [obj longlongM]; // expected-warning{{The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage}}
}

void createFoo4() {
  MyClass *obj = 0;  
  
  double d = [obj doubleM]; // expected-warning{{The receiver of message 'doubleM' is nil and returns a value of type 'double' that will be garbage}}
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
  
  long long j = [obj longlongM]; // expected-warning{{The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage}}
}

int handleVoidInComma() {
  MyClass *obj = 0;
  return [obj voidM], 0;
}

int marker(void) { // control reaches end of non-void function
}

// CHECK-darwin8: warning: The receiver of message 'longDoubleM' is nil and returns a value of type 'long double' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'doubleM' is nil and returns a value of type 'double' that will be garbage
// CHECK-darwin8: warning: The receiver of message 'longlongM' is nil and returns a value of type 'long long' that will be garbage
// CHECK-darwin8: control reaches end of non-void function
// CHECK-darwin8: 5 diagnostics generated
// CHECK-darwin9: control reaches end of non-void function
// CHECK-darwin9: 1 diagnostic generated
