// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,llvm-twine-local' -fix -- > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MESSAGES %s

namespace llvm {
class Twine {
public:
  Twine(const char *);
  Twine(int);
  Twine &operator+(const Twine &);
};
}

using namespace llvm;

void foo(const Twine &x);

static Twine Moo = Twine("bark") + "bah";
// CHECK-MASSAGES: twine variables are prone to use after free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK: static std::string Moo = (Twine("bark") + "bah").str();

int main() {
  const Twine t = Twine("a") + "b" + Twine(42);
// CHECK-MASSAGES: twine variables are prone to use after free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK: std::string t = (Twine("a") + "b" + Twine(42)).str();
  foo(Twine("a") + "b");

  Twine Prefix = false ? "__INT_FAST" : "__UINT_FAST";
// CHECK-MASSAGES: twine variables are prone to use after free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK: const char * Prefix = false ? "__INT_FAST" : "__UINT_FAST";
}
