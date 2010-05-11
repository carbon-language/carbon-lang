//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// constexpr tuple();

#include <tuple>
#include <string>
#include <cassert>

#include "../DefaultOnly.h"

int main()
{
    {
        std::tuple<> t;
    }
    {
        std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
    {
        std::tuple<int, char*, std::string> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
    }
    {
        std::tuple<int, char*, std::string, DefaultOnly> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
        assert(std::get<3>(t) == DefaultOnly());
    }
}
