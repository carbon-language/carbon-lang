// RUN: %clang_cc1 -std=c++0x -emit-llvm -o - %s

// PR9923
enum class Color { red, blue, green };

void f(Color);
void g() {
  f(Color::red);
}
