// RUN: %clang_cc1 -emit-llvm %s -o -

struct Pass {} ;
template<typename PassName>
Pass *callDefaultCtor() { return new PassName(); }

void foo(Pass *(*C)());

#include <string>

bool foo(std::string &X) {
  return X.empty();
}
