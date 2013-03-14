// A collection of various initializers which shouldn't trip up initialization
// order checking.  If successful, this will just return 0.

// RUN: %clangxx_asan -m64 -O0 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m64 -O1 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m64 -O2 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m64 -O3 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m32 -O0 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m32 -O0 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m32 -O1 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m32 -O2 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1
// RUN: %clangxx_asan -m32 -O3 %s %p/Helpers/initialization-nobug-extra.cc\
// RUN:   --std=c++11 -fsanitize=init-order -o %t
// RUN: ASAN_OPTIONS=check_initialization_order=true %t 2>&1

// Simple access:
// Make sure that accessing a global in the same TU is safe

bool condition = true;
int initializeSameTU() {
  return condition ? 0x2a : 052;
}
int sameTU = initializeSameTU();

// Linker initialized:
// Check that access to linker initialized globals originating from a different
// TU's initializer is safe.

int A = (1 << 1) + (1 << 3) + (1 << 5), B;
int getAB() {
  return A * B;
}

// Function local statics:
// Check that access to function local statics originating from a different
// TU's initializer is safe.

int countCalls() {
  static int calls;
  return ++calls;
}

// Constexpr:
// We need to check that a global variable initialized with a constexpr
// constructor can be accessed during dynamic initialization (as a constexpr
// constructor implies that it was initialized during constant initialization,
// not dynamic initialization).

class Integer {
  private:
  int value;

  public:
  constexpr Integer(int x = 0) : value(x) {}
  int getValue() {return value;}
};
Integer coolestInteger(42);
int getCoolestInteger() { return coolestInteger.getValue(); }

int main() { return 0; }
