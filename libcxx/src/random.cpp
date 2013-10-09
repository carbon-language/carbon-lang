//===-------------------------- random.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32)
// Must be defined before including stdlib.h to enable rand_s().
#define _CRT_RAND_S
#include <stdio.h>
#endif

#include "random"
#include "system_error"

#ifdef __sun__
#define rename solaris_headers_are_broken
#endif
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(_WIN32)
random_device::random_device(const string&)
{
}

random_device::~random_device()
{
}

unsigned
random_device::operator()()
{
    unsigned r;
    errno_t err = rand_s(&r);
    if (err)
        __throw_system_error(err, "random_device rand_s failed.");
    return r;
}
#else
random_device::random_device(const string& __token)
    : __f_(open(__token.c_str(), O_RDONLY))
{
    if (__f_ <= 0)
        __throw_system_error(errno, ("random_device failed to open " + __token).c_str());
}

random_device::~random_device()
{
    close(__f_);
}

unsigned
random_device::operator()()
{
    unsigned r;
    read(__f_, &r, sizeof(r));
    return r;
}
#endif // defined(_WIN32)

double
random_device::entropy() const _NOEXCEPT
{
    return 0;
}

_LIBCPP_END_NAMESPACE_STD
