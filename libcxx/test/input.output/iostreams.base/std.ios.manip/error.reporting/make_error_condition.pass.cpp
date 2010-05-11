//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>

// error_condition make_error_condition(io_errc e);

#include <ios>
#include <cassert>

int main()
{
    {
        const std::error_condition ec1 = std::make_error_condition(std::io_errc::stream);
        assert(ec1.value() == static_cast<int>(std::io_errc::stream));
        assert(ec1.category() == std::iostream_category());
    }
}
