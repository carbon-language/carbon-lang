//===---------------- exception_object_alignment.sh.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcxxabi-no-exceptions

// RUN: %cxx %flags %compile_flags -O1 %s -o %t.exe
// RUN: %exec %t.exe

// This test used to segfault on Darwin because field unwindHeader of struct
// __cxa_exception was not 16B aligned.

namespace {

struct S {
  int a;
  int __attribute__((aligned(16))) b;
};

class base1 {
protected:
  virtual ~base1() throw() {}
};

class derived: public base1 {
public:
  derived() : member() {}
private:
  S member;
};

}

int main() {
  try {
    throw derived();
  }
  catch(...) {
  }
  return 0;
}
