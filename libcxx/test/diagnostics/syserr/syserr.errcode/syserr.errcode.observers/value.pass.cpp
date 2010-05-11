//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_code

// int value() const;

#include <system_error>
#include <cassert>

int main()
{
    const std::error_code ec(6, std::system_category());
    assert(ec.value() == 6);
}
