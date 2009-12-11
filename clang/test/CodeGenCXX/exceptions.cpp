// RUN: clang-cc %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fexceptions

struct allocator {
  allocator();
  allocator(const allocator&);
  ~allocator();
};

void f();
void g(bool b, bool c) {
  if (b) {
    if (!c)
    throw allocator();

    return;
  }
  f();
}
