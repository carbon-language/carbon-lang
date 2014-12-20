//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// reference operator[](size_type n);

#include <vector>
#include <cassert>
#include <cstdlib>

#include "min_allocator.h"
#include "asan_testing.h"

#ifndef _LIBCPP_HAS_NO_ASAN
extern "C" void __asan_set_error_exit_code(int);

int main()
{
#if __cplusplus >= 201103L
    {
        typedef int T;
        typedef std::vector<T, min_allocator<T>> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t), std::end(t));
        c.reserve(2*c.size());
        T foo = c[c.size()];    // bad, but not caught by ASAN
    }
#endif
    
    __asan_set_error_exit_code(0);
    {
        typedef int T;
        typedef std::vector<T> C;
        const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t), std::end(t));
        c.reserve(2*c.size());
        assert(is_contiguous_container_asan_correct(c)); 
        assert(!__sanitizer_verify_contiguous_container ( c.data(), c.data() + 1, c.data() + c.capacity()));
        T foo = c[c.size()];    // should trigger ASAN
        assert(false);          // if we got here, ASAN didn't trigger
    }
}
#else
int main () { return 0; }
#endif
