//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// basic_string<char> name() const;

#include <locale>
#include <cassert>

int main()
{
    {
        std::locale loc;
        assert(loc.name() == "C");
    }
    {
        std::locale loc("en_US.UTF-8");
        assert(loc.name() == "en_US.UTF-8");
    }
}
