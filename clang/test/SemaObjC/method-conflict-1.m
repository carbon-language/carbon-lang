// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://7933061

@interface NSObject @end

@interface NSArray : NSObject @end

@interface MyClass : NSObject {
}
- (void)myMethod:(NSArray *)object; // expected-note {{previous definition is here}}
- (void)myMethod1:(NSObject *)object; // expected-note {{previous definition is here}}
@end

@implementation MyClass
// Warn about this contravariant use for now:
- (void)myMethod:(NSObject *)object { // expected-warning {{conflicting parameter types in implementation of 'myMethod:': 'NSArray *' vs 'NSObject *'}}
}
- (void)myMethod1:(NSArray *)object { // expected-warning {{conflicting parameter types in implementation of 'myMethod1:': 'NSObject *' vs 'NSArray *'}}
}
@end


@protocol MyProtocol @end

@interface MyOtherClass : NSObject <MyProtocol> {
}
- (void)myMethod:(id <MyProtocol>)object; // expected-note {{previous definition is here}}
- (void)myMethod1:(id <MyProtocol>)object; // expected-note {{previous definition is here}}
@end

@implementation MyOtherClass
- (void)myMethod:(MyClass *)object { // expected-warning {{conflicting parameter types in implementation of 'myMethod:': 'id<MyProtocol>' vs 'MyClass *'}}
}
- (void)myMethod1:(MyClass<MyProtocol> *)object { // expected-warning {{conflicting parameter types in implementation of 'myMethod1:': 'id<MyProtocol>' vs 'MyClass<MyProtocol> *'}}
}
@end
