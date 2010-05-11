//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// void close();

#include <fstream>
#include <cassert>

int main()
{
    {
        std::ofstream fs;
        assert(!fs.is_open());
        fs.open("test.dat");
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    remove("test.dat");
    {
        std::wofstream fs;
        assert(!fs.is_open());
        fs.open("test.dat");
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    remove("test.dat");
}
