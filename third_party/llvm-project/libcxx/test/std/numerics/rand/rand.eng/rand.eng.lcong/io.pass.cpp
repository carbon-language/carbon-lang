//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-localization

// <random>

// template <class UIntType, UIntType a, UIntType c, UIntType m>
//   class linear_congruential_engine;

// template <class charT, class traits,
//           class UIntType, UIntType a, UIntType c, UIntType m>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os,
//            const linear_congruential_engine<UIntType, a, c, m>& x);
//
// template <class charT, class traits,
//           class UIntType, UIntType a, UIntType c, UIntType m>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is,
//            linear_congruential_engine<UIntType, a, c, m>& x);

#include <random>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::linear_congruential_engine<unsigned, 48271, 0, 2147483647> E;
        E e1;
        e1.discard(100);
        std::ostringstream os;
        os << e1;
        std::istringstream is(os.str());
        E e2;
        is >> e2;
        assert(e1 == e2);
    }

  return 0;
}
