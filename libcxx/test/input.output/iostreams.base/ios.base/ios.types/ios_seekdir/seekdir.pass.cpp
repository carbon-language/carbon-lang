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

// static const seekdir beg;
// static const seekdir cur;
// static const seekdir end;

#include <ios>
#include <cassert>

int main()
{
    assert(std::ios_base::beg != std::ios_base::cur);
    assert(std::ios_base::beg != std::ios_base::end);
    assert(std::ios_base::cur != std::ios_base::end);
}
