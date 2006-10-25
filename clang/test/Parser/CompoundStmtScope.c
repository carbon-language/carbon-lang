// RUN: not clang %s -fsyntax-only

int foo() {
  {
    typedef float X;
  }
  X Y;  
}
