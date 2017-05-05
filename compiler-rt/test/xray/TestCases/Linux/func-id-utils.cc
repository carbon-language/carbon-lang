// Check that we can turn a function id to a function address, and also get the
// maximum function id for the current binary.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false" %run %t | FileCheck %s

#include "xray/xray_interface.h"
#include <algorithm>
#include <cstdio>
#include <set>
#include <iterator>

[[clang::xray_always_instrument]] void bar(){
    // do nothing!
}

    [[clang::xray_always_instrument]] void foo() {
  bar();
}

[[clang::xray_always_instrument]] int main(int argc, char *argv[]) {
  printf("max function id: %zu\n", __xray_max_function_id());
  // CHECK: max function id: [[MAX:.*]]

  std::set<void *> must_be_instrumented;
  must_be_instrumented.insert(reinterpret_cast<void*>(&foo));
  must_be_instrumented.insert(reinterpret_cast<void*>(&bar));
  printf("addresses:\n");
  std::set<void *> all_instrumented;
  for (auto i = __xray_max_function_id(); i != 0; --i) {
    auto addr = __xray_function_address(i);
    printf("#%lu -> @%04lx\n", i, addr);
    all_instrumented.insert(reinterpret_cast<void *>(addr));
  }

  // CHECK-LABEL: addresses:
  // CHECK: #[[MAX]] -> @[[ADDR:.*]]
  // CHECK-NOT: #0 -> @{{.*}}
  std::set<void *> common;

  std::set_intersection(all_instrumented.begin(), all_instrumented.end(),
                        must_be_instrumented.begin(),
                        must_be_instrumented.end(),
                        std::inserter(common, common.begin()));
  return common == must_be_instrumented ? 0 : 1;
}
