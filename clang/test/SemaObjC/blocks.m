// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s

#define bool _Bool
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

void bar5(id(^)(void)); // expected-note 3{{passing argument to parameter here}}
void foo5(id (^objectCreationBlock)(bool)) {
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(bool)' to parameter of type 'id (^)(void)'}}
#undef bool
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(_Bool)' to parameter of type 'id (^)(void)'}}
#define bool int
    bar5(objectCreationBlock); // expected-error {{incompatible block pointer types passing 'id (^)(_Bool)' to parameter of type 'id (^)(void)'}}
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
  void *P = ^(itf x) {};  // expected-error {{Objective-C interface type 'itf' cannot be passed by value; did you forget * in 'itf'}}
  P = ^itf(int x) {};     // expected-error {{Objective-C interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
  P = ^itf() {};          // expected-error {{Objective-C interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
  P = ^itf{};             // expected-error {{Objective-C interface type 'itf' cannot be returned by value; did you forget * in 'itf'}}
}


int foo9() {
  typedef void (^DVTOperationGroupScheduler)();
  id _suboperationSchedulers;

  for (DVTOperationGroupScheduler scheduler in _suboperationSchedulers) {
            ;
        }

}

// rdar 7725203
@class NSString;

extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

void foo10() {
    void(^myBlock)(void) = ^{
    };
    NSLog(@"%@", myBlock);
}

