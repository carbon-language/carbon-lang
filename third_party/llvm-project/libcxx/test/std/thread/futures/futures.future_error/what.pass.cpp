//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// LWG 2056 changed the values of future_errc, so if we're using new headers
// with an old library we'll get incorrect messages.
//
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// <future>

// class future_error

// const char* what() const throw();

#include <future>
#include <cstring>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::future_error f(std::make_error_code(std::future_errc::broken_promise));
        LIBCPP_ASSERT(std::strcmp(f.what(), "The associated promise has been destructed prior "
                      "to the associated state becoming ready.") == 0);
    }
    {
        std::future_error f(std::make_error_code(std::future_errc::future_already_retrieved));
        LIBCPP_ASSERT(std::strcmp(f.what(), "The future has already been retrieved from "
                      "the promise or packaged_task.") == 0);
    }
    {
        std::future_error f(std::make_error_code(std::future_errc::promise_already_satisfied));
        LIBCPP_ASSERT(std::strcmp(f.what(), "The state of the promise has already been set.") == 0);
    }
    {
        std::future_error f(std::make_error_code(std::future_errc::no_state));
        LIBCPP_ASSERT(std::strcmp(f.what(), "Operation not permitted on an object without "
                      "an associated state.") == 0);
    }

  return 0;
}
