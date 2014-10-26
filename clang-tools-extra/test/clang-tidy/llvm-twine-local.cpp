// RUN: $(dirname %s)/check_clang_tidy.sh %s llvm-twine-local %t
// REQUIRES: shell

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
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: static std::string Moo = (Twine("bark") + "bah").str();

int main() {
  const Twine t = Twine("a") + "b" + Twine(42);
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: std::string t = (Twine("a") + "b" + Twine(42)).str();
  foo(Twine("a") + "b");

  Twine Prefix = false ? "__INT_FAST" : "__UINT_FAST";
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: twine variables are prone to use-after-free bugs
// CHECK-MESSAGES: note: FIX-IT applied suggested code changes
// CHECK-FIXES: const char * Prefix = false ? "__INT_FAST" : "__UINT_FAST";
}
