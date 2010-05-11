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

// void open(const string& s, ios_base::openmode mode = ios_base::in|ios_base::out);

#include <fstream>
#include <cassert>

int main()
{
    {
        std::fstream fs;
        assert(!fs.is_open());
        fs.open(std::string("test.dat"), std::ios_base::in | std::ios_base::out
                                                           | std::ios_base::trunc);
        assert(fs.is_open());
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove("test.dat");
    {
        std::wfstream fs;
        assert(!fs.is_open());
        fs.open(std::string("test.dat"), std::ios_base::in | std::ios_base::out
                                                           | std::ios_base::trunc);
        assert(fs.is_open());
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove("test.dat");
}
