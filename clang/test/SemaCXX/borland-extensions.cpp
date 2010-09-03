// RUN: %clang_cc1 %s -fsyntax-only -verify -fborland-extensions

// Borland extensions

// 1. test  -fborland-extensions
int dummy_function() { return 0; }

// 2. test __pascal
int _pascal f2();

float __pascal gi2(int, int); 
template<typename T> T g2(T (__pascal * const )(int, int)) { return 0; }

struct M {
    int __pascal addP();
    float __pascal subtractP(); 
};
template<typename T> int h2(T (__pascal M::* const )()) { return 0; }
void m2() {
    int i; float f;
    i = f2();
    f = gi2(2, i);
    f = g2(gi2);
    i = h2<int>(&M::addP);
    f = h2(&M::subtractP);
} 
