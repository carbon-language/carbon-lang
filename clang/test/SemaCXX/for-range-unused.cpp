// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++0x -Wunused

// PR9968: We used to warn that __range is unused in a dependent for-range.

template <typename T>
  struct Vector {
    void doIt() {
      // FIXME: PR10168: Only warn once for this!
      int a; // expected-warning 2{{unused variable 'a'}}

      for (auto& e : elements)
        ;
    }

    T elements[10];
  };


int main(int, char**) {
  Vector<int>    vector;
  vector.doIt(); // expected-note {{requested here}}
}
