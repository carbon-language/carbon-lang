//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_condition

// template <ErrorConditionEnum E> error_condition(E e); 

#include <system_error>
#include <cassert>

int main()
{
    {
        std::error_condition ec(std::errc::not_a_directory);
        assert(ec.value() == static_cast<int>(std::errc::not_a_directory));
        assert(ec.category() == std::generic_category());
    }
}
