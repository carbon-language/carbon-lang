// RUN: clang-cc -fsyntax-only %s

class A;

class S {
public:
   template<typename T> struct A { 
     struct Nested {
       typedef T type;
     };
   };
};

int i;
S::A<int>::Nested::type *ip = &i;

