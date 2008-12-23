// RUN: clang -fsyntax-only -verify -fblocks %s
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
    return bar4(objectCreationBlock); // expected-warning{{incompatible block pointer types passing 'id (^)(int)', expected 'id (^)()'}}
}

void foo5(id (^x)(int)) {
  if (x) { }
}
