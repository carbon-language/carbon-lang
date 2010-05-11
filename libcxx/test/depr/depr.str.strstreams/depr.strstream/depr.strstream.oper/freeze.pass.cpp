//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstream

// void freeze(bool freezefl = true);

#include <strstream>
#include <cassert>

int main()
{
    {
        std::strstream out;
        out.freeze();
        assert(!out.fail());
        out << 'a';
        assert(out.fail());
        out.clear();
        out.freeze(false);
        out << 'a';
        out << char(0);
        assert(out.str() == std::string("a"));
    }
}
