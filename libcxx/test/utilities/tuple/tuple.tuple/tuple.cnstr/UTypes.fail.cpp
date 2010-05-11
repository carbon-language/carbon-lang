//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

#include <tuple>
#include <cassert>

#include "../MoveOnly.h"

int main()
{
    {
        std::tuple<MoveOnly> t = MoveOnly(0);
    }
}
