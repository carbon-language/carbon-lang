// Header for PCH test cxx0x-delegating-ctors.cpp

struct foo {
  foo(int) : foo() { }
  foo();
};
