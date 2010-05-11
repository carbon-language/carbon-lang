//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <fstream>

// template <class charT, class traits = char_traits<charT> >
// class basic_fstream

// explicit basic_fstream(const char* s, ios_base::openmode mode = ios_base::in | ios_base::out);

#include <fstream>
#include <cassert>

int main()
{
    {
        std::fstream fs("test.dat", std::ios_base::in | std::ios_base::out
                                                      | std::ios_base::trunc);
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove("test.dat");
    {
        std::wfstream fs("test.dat", std::ios_base::in | std::ios_base::out
                                                       | std::ios_base::trunc);
        double x = 0;
        fs << 3.25;
        fs.seekg(0);
        fs >> x;
        assert(x == 3.25);
    }
    std::remove("test.dat");
}
