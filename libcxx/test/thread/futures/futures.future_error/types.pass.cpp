//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class future_error : public logic_error {...};

#include <future>
#include <type_traits>

int main()
{
    static_assert((std::is_convertible<std::future_error*,
                                       std::logic_error*>::value), "");
}
