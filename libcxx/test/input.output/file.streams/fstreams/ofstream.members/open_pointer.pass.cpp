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

// void open(const char* s, ios_base::openmode mode = ios_base::out);

#include <fstream>
#include <cassert>

int main()
{
    {
        std::ofstream fs;
        assert(!fs.is_open());
        char c = 'a';
        fs << c;
        assert(fs.fail());
        fs.open("test.dat");
        assert(fs.is_open());
        fs << c;
    }
    {
        std::ifstream fs("test.dat");
        char c = 0;
        fs >> c;
        assert(c == 'a');
    }
    remove("test.dat");
    {
        std::wofstream fs;
        assert(!fs.is_open());
        wchar_t c = L'a';
        fs << c;
        assert(fs.fail());
        fs.open("test.dat");
        assert(fs.is_open());
        fs << c;
    }
    {
        std::wifstream fs("test.dat");
        wchar_t c = 0;
        fs >> c;
        assert(c == L'a');
    }
    remove("test.dat");
}
