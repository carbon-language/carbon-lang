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
// class basic_fstream

// void close();

#include <fstream>
#include <cassert>

int main()
{
    {
        std::fstream fs;
        assert(!fs.is_open());
        fs.open("test.dat", std::ios_base::out);
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    remove("test.dat");
    {
        std::wfstream fs;
        assert(!fs.is_open());
        fs.open("test.dat", std::ios_base::out);
        assert(fs.is_open());
        fs.close();
        assert(!fs.is_open());
    }
    remove("test.dat");
}
