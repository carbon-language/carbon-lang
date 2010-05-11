//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// class condition_variable;

// typedef pthread_cond_t* native_handle_type;
// native_handle_type native_handle();

#include <condition_variable>
#include <cassert>

int main()
{
    static_assert((std::is_same<std::condition_variable::native_handle_type,
                                pthread_cond_t*>::value), "");
    std::condition_variable cv;
    std::condition_variable::native_handle_type h = cv.native_handle();
    assert(h != nullptr);
}
