// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -o - %s | FileCheck %s
// Test that we are emitting debug info and base types for scoped enums.

// CHECK: [ DW_TAG_enumeration_type ] [Color] {{.*}} [from int]
enum class Color { gray };

void f(Color);
void g() {
  f(Color::gray);
}

// CHECK: [ DW_TAG_enumeration_type ] [Colour] {{.*}} [from int]
enum struct Colour { grey };

void h(Colour);
void i() {
  h(Colour::grey);
}

// CHECK: [ DW_TAG_enumeration_type ] [Couleur] {{.*}} [from unsigned char]
enum class Couleur : unsigned char { gris };

void j(Couleur);
void k() {
  j(Couleur::gris);
}
