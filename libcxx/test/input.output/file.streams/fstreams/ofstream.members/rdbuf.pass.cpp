//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_ofstream

// basic_filebuf<charT,traits>* rdbuf() const;

#include <fstream>
#include <cassert>

int main()
{
    {
        std::ofstream fs("test.dat");
        std::filebuf* fb = fs.rdbuf();
        assert(fb->sputc('r') == 'r');
    }
    remove("test.dat");
    {
        std::wofstream fs("test.dat");
        std::wfilebuf* fb = fs.rdbuf();
        assert(fb->sputc(L'r') == L'r');
    }
    remove("test.dat");
}
