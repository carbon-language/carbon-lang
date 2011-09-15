// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://10109725

@interface NSObject {
    Class isa;
}
- (id)addObserver:(NSObject *)observer; // expected-note 2 {{passing argument to parameter 'observer' here}}
@end

@interface MyClass : NSObject {
}
@end

@implementation NSObject
+ (void)initialize
{
        NSObject *obj = 0;
        [obj addObserver:self];
        [obj addObserver:(Class)0];
}

- init
{
        NSObject *obj = 0;
        [obj addObserver:self];
        return [obj addObserver:(Class)0]; // expected-warning {{incompatible pointer types sending 'Class' to parameter of type 'NSObject *'}}
}
- (id)addObserver:(NSObject *)observer { return 0; }
@end

@implementation MyClass

+ (void)initialize
{
        NSObject *obj = 0;
        [obj addObserver:self];
        [obj addObserver:(Class)0];
}

- init
{
        NSObject *obj = 0;
        [obj addObserver:self];
        return [obj addObserver:(Class)0]; // expected-warning {{incompatible pointer types sending 'Class' to parameter of type 'NSObject *'}}
}
@end
