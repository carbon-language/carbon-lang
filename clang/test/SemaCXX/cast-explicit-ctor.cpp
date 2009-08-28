// RUN: clang-cc -fsyntax-only -verify %s
struct B { B(bool); };
void f() { 
  (void)(B)true;
  (void)B(true); 
}
