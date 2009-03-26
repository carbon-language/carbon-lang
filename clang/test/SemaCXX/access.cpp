// RUN: clang-cc -fsyntax-only -verify %s

class C {
    struct S; // expected-note {{previously declared 'private' here}}
public:
    
    struct S {}; // expected-error {{'S' redeclared with 'public' access}}
};

struct S {
    class C; // expected-note {{previously declared 'public' here}}
    
private:
    class C { }; // expected-error {{'C' redeclared with 'private' access}}
};

class T {
protected:
    template<typename T> struct A; // expected-note {{previously declared 'protected' here}}
    
private:
    template<typename T> struct A {}; // expected-error {{'A' redeclared with 'private' access}}
};
