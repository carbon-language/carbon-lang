// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null



struct Pass {} ;
template<typename PassName>
Pass *callDefaultCtor() { return new PassName(); }

void foo(Pass *(*C)());

#include <bits/c++config.h>
#include <bits/stringfwd.h>
#include <bits/char_traits.h>
#include <memory>       // For allocator.
#include <bits/basic_string.h>

bool foo(std::string &X) {
  return X.empty();
}
