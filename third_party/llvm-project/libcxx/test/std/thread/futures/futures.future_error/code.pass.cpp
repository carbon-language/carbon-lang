//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// class future_error
//     future_error(error_code __ec);  // exposition only
//     explicit future_error(future_errc _Ev) : __ec_(make_error_code(_Ev)) {} // C++17

// const error_code& code() const throw();

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::error_code ec = std::make_error_code(std::future_errc::broken_promise);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
    {
        std::error_code ec = std::make_error_code(std::future_errc::future_already_retrieved);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
    {
        std::error_code ec = std::make_error_code(std::future_errc::promise_already_satisfied);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
    {
        std::error_code ec = std::make_error_code(std::future_errc::no_state);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
#if TEST_STD_VER > 14
    {
        std::future_error f(std::future_errc::broken_promise);
        assert(f.code() == std::make_error_code(std::future_errc::broken_promise));
    }
    {
        std::future_error f(std::future_errc::no_state);
        assert(f.code() == std::make_error_code(std::future_errc::no_state));
    }
#endif

  return 0;
}
