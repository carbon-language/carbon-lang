//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// explicit tuple(const T&...);

#include <tuple>
#include <string>
#include <cassert>

int main()
{
    {
        std::tuple<int, char*, std::string, double&> t(2, nullptr, "text");
    }
}
