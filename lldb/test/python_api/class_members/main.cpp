//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Base {
public:
    int foo(int x, int y) { return 1; }
    char bar(int x, char y) { return 2; }
    void dat() {}
    static int sfunc(char, int, float) { return 3; }
};

class Derived: public Base {
protected:
    int dImpl() { return 1; }
public:
    float baz(float b) { return b + 1.0; }
};

int main() {
    Derived d;
    return 0; // set breakpoint here
}
