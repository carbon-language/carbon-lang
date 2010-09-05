//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test unexpected

#include <exception>
#include <cstdlib>
#include <cassert>

void f1()
{
    std::exit(0);
}

int main()
{
    std::set_unexpected(f1);
    std::unexpected();
    assert(false);
}
