// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://9092208

__attribute__((unavailable("not available")))
@interface MyClass { // expected-note 5 {{function has been explicitly marked unavailable here}}
@public
    void *_test;
}

- (id)self;
- new;
+ (void)addObject:(id)anObject;

@end

int main() {
 [MyClass new]; // expected-error {{'MyClass' is unavailable: not available}}
 [MyClass self]; // expected-error {{'MyClass' is unavailable: not available}}
 [MyClass addObject:((void *)0)]; // expected-error {{'MyClass' is unavailable: not available}}

 MyClass *foo = [MyClass new]; // expected-error 2 {{'MyClass' is unavailable: not available}}

 return 0;
}
