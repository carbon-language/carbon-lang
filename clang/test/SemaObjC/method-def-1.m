// RUN: clang -fsyntax-only -verify %s

@interface foo
- (int)meth;
@end

@implementation foo
- (int) meth { return [self meth]; }
@end

// PR2708
@interface MyClass
+- (void)myMethod;   // expected-error {{expected selector for Objective-C method}}
- (vid)myMethod2;    // expected-error {{expected a type}}
@end

@implementation MyClass
- (void)myMethod { }
- (void)myMethod2 { }
@end

