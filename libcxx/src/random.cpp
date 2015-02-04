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
#endif // defined(_WIN32)

#include "random"
#include "system_error"

#if defined(__sun__)
#define rename solaris_headers_are_broken
#endif // defined(__sun__)
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif // !defined(_WIN32)
#include <errno.h>
#if defined(_LIBCPP_USING_NACL_RANDOM)
#include <nacl/nacl_random.h>
#endif // defined(_LIBCPP_USING_NACL_RANDOM)

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

#elif defined(_LIBCPP_USING_NACL_RANDOM)

random_device::random_device(const string& __token)
{
    if (__token != "/dev/urandom")
        __throw_system_error(ENOENT, ("random device not supported " + __token).c_str());
    int error = nacl_secure_random_init();
    if (error)
        __throw_system_error(error, ("random device failed to open " + __token).c_str());
}

random_device::~random_device()
{
}

unsigned
random_device::operator()()
{
    unsigned r;
    size_t n = sizeof(r);
    size_t bytes_written;
    int error = nacl_secure_random(&r, n, &bytes_written);
    if (error != 0)
        __throw_system_error(error, "random_device failed getting bytes");
    else if (bytes_written != n)
        __throw_runtime_error("random_device failed to obtain enough bytes");
    return r;
}

#else // !defined(_WIN32) && !defined(_LIBCPP_USING_NACL_RANDOM)

random_device::random_device(const string& __token)
    : __f_(open(__token.c_str(), O_RDONLY))
{
    if (__f_ < 0)
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
    size_t n = sizeof(r);
    char* p = reinterpret_cast<char*>(&r);
    while (n > 0)
    {
        ssize_t s = read(__f_, p, n);
        if (s == 0)
            __throw_system_error(ENODATA, "random_device got EOF");
        if (s == -1)
        {
            if (errno != EINTR)
                __throw_system_error(errno, "random_device got an unexpected error");
            continue;
        }
        n -= static_cast<size_t>(s);
        p += static_cast<size_t>(s);
    }
    return r;
}

#endif // defined(_WIN32) || defined(_LIBCPP_USING_NACL_RANDOM)

double
random_device::entropy() const _NOEXCEPT
{
    return 0;
}

_LIBCPP_END_NAMESPACE_STD
