// RUN: clang -cc1 -fsyntax-only -verify -fblocks %s
@protocol NSObject;

void bar(id(^)(void));
void foo(id <NSObject>(^objectCreationBlock)(void)) {
    return bar(objectCreationBlock);
}

void bar2(id(*)(void));
void foo2(id <NSObject>(*objectCreationBlock)(void)) {
    return bar2(objectCreationBlock);
}

void bar3(id(*)());
void foo3(id (*objectCreationBlock)(int)) {
    return bar3(objectCreationBlock);
}

void bar4(id(^)());
void foo4(id (^objectCreationBlock)(int)) {
    return bar4(objectCreationBlock);
}

void bar5(id(^)(void));
void foo5(id (^objectCreationBlock)(int)) {
    return bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(int)', expected 'id (^)(void)'}}
}

void bar6(id(^)(int));
void foo6(id (^objectCreationBlock)()) {
    return bar6(objectCreationBlock);
}

void foo7(id (^x)(int)) {
  if (x) { }
}

@interface itf
@end

void foo8() {
  void *P = ^(itf x) {};  // expected-error {{Objective-C interface type 'itf' cannot be passed by value}}
  P = ^itf(int x) {};     // expected-error {{Objective-C interface type 'itf' cannot be returned by value}}
  P = ^itf() {};          // expected-error {{Objective-C interface type 'itf' cannot be returned by value}}
  P = ^itf{};             // expected-error {{Objective-C interface type 'itf' cannot be returned by value}}
}


int foo9() {
  typedef void (^DVTOperationGroupScheduler)();
  id _suboperationSchedulers;

  for (DVTOperationGroupScheduler scheduler in _suboperationSchedulers) {
            ;
        }

}
