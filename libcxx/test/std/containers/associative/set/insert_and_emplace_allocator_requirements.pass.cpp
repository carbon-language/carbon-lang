//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// insert(...)
// emplace(...)
// emplace_hint(...)

// UNSUPPORTED: c++03

#include <set>
#include "test_macros.h"
#include "container_test_types.h"
#include "../../set_allocator_requirement_test_templates.h"

int main(int, char**)
{
  testSetInsert<TCT::set<> >();
  testSetEmplace<TCT::set<> >();
  testSetEmplaceHint<TCT::set<> >();

  return 0;
}
