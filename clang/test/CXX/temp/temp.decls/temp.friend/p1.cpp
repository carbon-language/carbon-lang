// RUN: clang-cc %s

template <typename T> struct Num {
  T value_;

public:
  Num(T value) : value_(value) {}
  T get() const { return value_; }

  template <typename U> struct Rep {
    U count_;
    Rep(U count) : count_(count) {}

    friend Num operator*(const Num &a, const Rep &n) {
      Num x = 0;
      for (U count = n.count_; count; --count)
        x += a;
      return x;
    } 
  };

  friend Num operator+(const Num &a, const Num &b) {
    return a.value_ + b.value_;
  }

  Num& operator+=(const Num& b) {
    value_ += b.value_;
    return *this;
  }
};

int calc1() {
  Num<int> left = -1;
  Num<int> right = 1;
  Num<int> result = left + right;
  return result.get();
}

int calc2() {
  Num<int> x = 3;
  Num<int>::Rep<char> n = (cast) 10;
  Num<int> result = x * n;
  return result.get();
}
