// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -o - %s

// PR9923
enum class Color { red, blue, green };

void f(Color);
void g() {
  f(Color::red);
}

// See that struct is handled equally.
// CHECK: [ DW_TAG_enumeration_type ] [Colour]
enum struct Colour { grey };

void h(Colour);
void i() {
  h(Colour::grey);
}
