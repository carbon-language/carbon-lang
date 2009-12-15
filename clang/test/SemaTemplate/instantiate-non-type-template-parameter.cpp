// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5311
template<typename T>
class StringSwitch {
public:
  template<unsigned N>
  void Case(const char (&S)[N], const int & Value) {
  }
};

int main(int argc, char *argv[]) {
  (void)StringSwitch<int>();
}
