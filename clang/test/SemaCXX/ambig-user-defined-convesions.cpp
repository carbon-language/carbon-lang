// RUN: clang-cc -fsyntax-only -verify %s

struct BASE { 
  operator int &(); // expected-note {{candidate function}}
}; 
struct BASE1 { 
  operator int &(); // expected-note {{candidate function}}
}; 

struct B : public BASE, BASE1 { 

}; 

extern B f(); 

const int main() {
  return f(); // expected-error {{conversion from 'struct B' to 'int const' is ambiguous}}
}

