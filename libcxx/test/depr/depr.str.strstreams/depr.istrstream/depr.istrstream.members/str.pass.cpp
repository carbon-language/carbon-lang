//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <strstream>

// class istrstream

// char* str();

#include <strstream>
#include <cassert>

int main()
{
    {
        const char buf[] = "123 4.5 dog";
        std::istrstream in(buf);
        assert(in.str() == std::string("123 4.5 dog"));
    }
}
