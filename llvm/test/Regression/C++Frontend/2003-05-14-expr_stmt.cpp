
struct Foo {  int X; };

void bar() {}

int main() {
  Foo X;
  X = ({ bar(); Foo(); });
}
