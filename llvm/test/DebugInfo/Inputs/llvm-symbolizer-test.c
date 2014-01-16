int f(int a, int b) {
  return a + b;
}

int g(int a) {
  return a + 1;
}


int main() {
  return f(2, g(2));
}

// Built with Clang 3.3:
// $ mkdir -p /tmp/dbginfo
// $ cp llvm-symbolizer-test.c /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ clang -g llvm-symbolizer-test.c -o <output>
