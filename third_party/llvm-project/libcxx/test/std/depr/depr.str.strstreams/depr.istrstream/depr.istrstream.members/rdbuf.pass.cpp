//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <strstream>

// class istrstream

// strstreambuf* rdbuf() const;

#include <strstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        const char buf[] = "123 4.5 dog";
        const std::istrstream in(buf);
        std::strstreambuf* sb = in.rdbuf();
        assert(sb->sgetc() == '1');
    }

  return 0;
}
