//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <experimental/memory_resource>

// memory_resource * new_delete_resource()

// The lifetime of the value returned by 'new_delete_resource()' should
// never end, even very late into program termination. This test constructs
// attempts to use 'new_delete_resource()' very late in program termination
// to detect lifetime issues.

#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace ex = std::experimental::pmr;

struct POSType {
  ex::memory_resource* res = nullptr;
  void* ptr = nullptr;
  int n = 0;
  POSType() {}
  POSType(ex::memory_resource* r, void* p, int s) : res(r), ptr(p), n(s) {}
  ~POSType() {
      if (ptr) {
          if (!res) res = ex::get_default_resource();
          res->deallocate(ptr, n);
      }
  }
};

void swap(POSType & L, POSType & R) {
    std::swap(L.res, R.res);
    std::swap(L.ptr, R.ptr);
    std::swap(L.n, R.n);
}

POSType constructed_before_resources;
POSType constructed_before_resources2;

// Constructs resources
ex::memory_resource* resource = ex::get_default_resource();

POSType constructed_after_resources(resource, resource->allocate(1024), 1024);
POSType constructed_after_resources2(nullptr, resource->allocate(1024), 1024);

int main(int, char**)
{
    swap(constructed_after_resources, constructed_before_resources);
    swap(constructed_before_resources2, constructed_after_resources2);

  return 0;
}
