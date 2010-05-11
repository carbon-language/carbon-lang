//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// tuple& operator=(const tuple& u);

#include <tuple>
#include <cassert>

#include "../MoveOnly.h"

int main()
{
    {
        typedef std::tuple<MoveOnly> T;
        T t0(MoveOnly(2));
        T t;
        t = t0;
    }
}
