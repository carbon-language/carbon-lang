// RUN: clang-cc -fsyntax-only -verify %s

struct BASE { 
  operator int &(); // expected-note 4 {{candidate function}}
}; 
struct BASE1 { 
  operator int &(); // expected-note 4 {{candidate function}}
}; 

struct B : public BASE, BASE1 { 

}; 

extern B f(); 

B b1;
void func(const int ci, const char cc); // expected-note {{function not viable because of ambiguity in conversion of argument 1}}
void func(const char ci, const B b); // expected-note {{function not viable because of ambiguity in conversion of argument 1}}
void func(const B b, const int ci); // expected-note {{function not viable because of ambiguity in conversion of argument 2}}


const int main() {
  func(b1, f()); // expected-error {{no matching function for call to 'func'}}
  return f(); // expected-error {{conversion from 'struct B' to 'int const' is ambiguous}}
}

