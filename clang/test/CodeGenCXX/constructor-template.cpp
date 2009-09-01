// RUN: clang-cc %s -emit-llvm -o -

// PR4826
struct A {
  A() {
  }
};

template<typename T>
struct B {
  B(T) {}
  
  A nodes;
};

int main() {
  B<int> *n = new B<int>(4);
}
