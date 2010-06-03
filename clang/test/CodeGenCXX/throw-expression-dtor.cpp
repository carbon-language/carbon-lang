// RUN: %clang_cc1 %s -emit-llvm-only -verify
// PR7281

class A {
public:
    ~A();
};
class B : public A {
    void ice_throw();
};
void B::ice_throw() {
    throw *this;
}

