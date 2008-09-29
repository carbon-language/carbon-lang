// RUN: clang %s -fsyntax-only -verify

@interface Whatever
- copy;
@end

typedef long (^MyBlock)(id obj1, id obj2);

void foo(MyBlock b) {
    id bar = [b copy];
}

