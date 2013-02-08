class DummyClass {
  int a_;
 public:
  DummyClass(int a) : a_(a) {}
  int add(int b) {
    return a_ + b;
  }
};

int f(int a, int b) {
  DummyClass c(a);
  return c.add(b);
}

int main() {
  return f(2, 3);
}

// Built with Clang 3.2:
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-test.cc /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ clang++ -g dwarfdump-test.cc -o <output>
