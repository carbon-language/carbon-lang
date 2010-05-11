//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <strstream>

// class strstreambuf

// char* str();

#include <strstream>
#include <cassert>

int main()
{
    {
        std::strstreambuf sb;
        assert(sb.sputc('a') == 'a');
        assert(sb.sputc(0) == 0);
        assert(sb.str() == std::string("a"));
    }
}
