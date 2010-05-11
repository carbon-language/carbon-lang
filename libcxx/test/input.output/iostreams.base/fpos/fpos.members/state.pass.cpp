//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// template <class StateT> class fpos

// void state(stateT s);

#include <ios>
#include <cassert>

int main()
{
    std::fpos<int> f;
    f.state(3);
    assert(f.state() == 3);
}
