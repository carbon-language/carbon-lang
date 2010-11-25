// RUN: %llvmgxx -xc++ %s -S -o - | grep unwind

struct S { ~S(); };

int mightthrow();

int test() {
  S s;
  mightthrow();
}
