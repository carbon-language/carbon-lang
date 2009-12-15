// RUN: %clang -emit-llvm -S -o - %s

// PR5775
class Twine {
  Twine(const char *Str) { }
};

static void error(const Twine &Message);

template<typename>
struct opt_storage {
  void f() {
    error("cl::location(x) specified more than once!");
  }
};

void f(opt_storage<int> o) {
  o.f();
}
