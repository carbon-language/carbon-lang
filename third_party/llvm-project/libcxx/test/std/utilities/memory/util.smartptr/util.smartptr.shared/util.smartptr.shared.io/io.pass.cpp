//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-localization

// <memory>

// shared_ptr

// template<class CharT, class Traits, class Y>
//   basic_ostream<CharT, Traits>&
//   operator<<(basic_ostream<CharT, Traits>& os, shared_ptr<Y> const& p);

#include <memory>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::shared_ptr<int> p(new int(3));
    std::ostringstream os;
    assert(os.str().empty());
    os << p;
    assert(!os.str().empty());

  return 0;
}
