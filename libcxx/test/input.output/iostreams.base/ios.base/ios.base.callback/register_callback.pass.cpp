//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// class ios_base

// void register_callback(event_callback fn, int index);

#include <ios>
#include <string>
#include <locale>
#include <cassert>

class test
    : public std::ios
{
public:
    test()
    {
        init(0);
    }
};

int f1_called = 0;

void f1(std::ios_base::event ev, std::ios_base& stream, int index)
{
    if (ev == std::ios_base::imbue_event)
    {
        assert(stream.getloc().name() == "en_US");
        assert(index == 4);
        ++f1_called;
    }
}

int main()
{
    test t;
    std::ios_base& b = t;
    b.register_callback(f1, 4);
    b.register_callback(f1, 4);
    b.register_callback(f1, 4);
    std::locale l = b.imbue(std::locale("en_US"));
    assert(f1_called == 3);
}
