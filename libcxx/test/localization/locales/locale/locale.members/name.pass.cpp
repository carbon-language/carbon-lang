//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
        std::locale loc("en_US");
        assert(loc.name() == "en_US");
    }
}
