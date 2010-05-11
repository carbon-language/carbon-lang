//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class charT, class traits> class basic_ios

// iostate exceptions() const;

#include <ios>
#include <streambuf>
#include <cassert>

struct testbuf : public std::streambuf {};

int main()
{
    {
        std::ios ios(0);
        assert(ios.exceptions() == std::ios::goodbit);
        ios.exceptions(std::ios::eofbit);
        assert(ios.exceptions() == std::ios::eofbit);
        try
        {
            ios.exceptions(std::ios::badbit);
            assert(false);
        }
        catch (std::ios::failure&)
        {
        }
        assert(ios.exceptions() == std::ios::badbit);
    }
    {
        testbuf sb;
        std::ios ios(&sb);
        assert(ios.exceptions() == std::ios::goodbit);
        ios.exceptions(std::ios::eofbit);
        assert(ios.exceptions() == std::ios::eofbit);
        ios.exceptions(std::ios::badbit);
        assert(ios.exceptions() == std::ios::badbit);
    }
}
