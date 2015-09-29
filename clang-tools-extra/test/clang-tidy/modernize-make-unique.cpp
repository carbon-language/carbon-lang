// RUN: %python %S/check_clang_tidy.py %s modernize-make-unique %t

namespace std {

template <typename type>
class unique_ptr {
public:
  unique_ptr(type *ptr);
  unique_ptr(const unique_ptr<type> &t) = delete;
  unique_ptr(unique_ptr<type> &&t);
  ~unique_ptr();
  type &operator*() { return *ptr; }
  type *operator->() { return ptr; }
  type *release();
  void reset();
  void reset(type *pt);

private:
  type *ptr;
};

}

struct Base {
  Base();
  Base(int, int);
};

struct Derived : public Base {
  Derived();
  Derived(int, int);
};

struct Pair {
  int a, b;
};

template<class T> using unique_ptr_ = std::unique_ptr<T>;

int g(std::unique_ptr<int> P);

std::unique_ptr<Base> getPointer() {
  return std::unique_ptr<Base>(new Base);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_unique instead
  // CHECK-FIXES: return std::make_unique<Base>();
}

void f() {
  std::unique_ptr<int> P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P1 = std::make_unique<int>();

  // Without parenthesis.
  std::unique_ptr<int> P2 = std::unique_ptr<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P2 = std::make_unique<int>();

  // With auto.
  auto P3 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: auto P3 = std::make_unique<int>();

  {
    // No std.
    using namespace std;
    unique_ptr<int> Q = unique_ptr<int>(new int());
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use std::make_unique instead
    // CHECK-FIXES: unique_ptr<int> Q = std::make_unique<int>();
  }

  std::unique_ptr<int> R(new int());

  // Create the unique_ptr as a parameter to a function.
  int T = g(std::unique_ptr<int>(new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: int T = g(std::make_unique<int>());

  // Arguments are correctly handled.
  std::unique_ptr<Base> Pbase = std::unique_ptr<Base>(new Base(5, T));
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<Base> Pbase = std::make_unique<Base>(5, T);

  // Works with init lists.
  std::unique_ptr<Pair> Ppair = std::unique_ptr<Pair>(new Pair{T, 1});
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<Pair> Ppair = std::make_unique<Pair>({T, 1});

  // Only replace if the type in the template is the same than the type returned
  // by the new operator.
  auto Pderived = std::unique_ptr<Base>(new Derived());

  // The pointer is returned by the function, nothing to do.
  std::unique_ptr<Base> RetPtr = getPointer();

  // Aliases.
  typedef std::unique_ptr<int> IntPtr;
  IntPtr Typedef = IntPtr(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use std::make_unique instead
  // CHECK-FIXES: IntPtr Typedef = std::make_unique<int>();

#define PTR unique_ptr<int>
  std::unique_ptr<int> Macro = std::PTR(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<int> Macro = std::make_unique<int>();
#undef PTR

  std::unique_ptr<int> Using = unique_ptr_<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use std::make_unique instead
  // CHECK-FIXES: std::unique_ptr<int> Using = std::make_unique<int>();

  // This emulates std::move.
  std::unique_ptr<int> Move = static_cast<std::unique_ptr<int>&&>(P1);

  // Adding whitespaces.
  auto Space = std::unique_ptr <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use std::make_unique instead
  // CHECK-FIXES: auto Space = std::make_unique<int>();

  auto Spaces = std  ::    unique_ptr  <int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::make_unique instead
  // CHECK-FIXES: auto Spaces = std::make_unique<int>();
}
