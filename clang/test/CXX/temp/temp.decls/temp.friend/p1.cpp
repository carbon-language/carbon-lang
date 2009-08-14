// RUN: clang-cc -fsyntax-only -verify %s

template <typename T> class Num {
  T value_;

public:
  Num(T value) : value_(value) {}
  T get() const { return value_; }
  
  friend Num operator+(const Num &a, const Num &b) {
    return a.value_ + b.value_;
  }
};

int main() {
  Num<int> left = -1;
  Num<int> right = 1;
  Num<int> result = left + right;
  return result.get();
}
