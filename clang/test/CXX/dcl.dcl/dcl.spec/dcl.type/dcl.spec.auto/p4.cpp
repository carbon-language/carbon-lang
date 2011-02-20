// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete;
};

void f() {
  if (auto a = true) {
  }

  switch (auto a = 0) {
  }

  while (auto a = false) {
  }

  for (; auto a = false; ) {
  }

  new const auto (0);
  new (auto) (0.0);

#if 0
  // When clang supports for-range:
  for (auto i : {1,2,3}) {
  }

  // When clang supports inline initialization of members.
  class X {
    static const auto &n = 'x';
  };
#endif
}
