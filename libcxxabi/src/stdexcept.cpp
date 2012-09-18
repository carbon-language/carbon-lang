//===------------------------ stdexcept.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "stdexcept"
#include "new"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>

#if __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#endif

// Note:  optimize for size

#pragma GCC visibility push(hidden)

namespace
{

class __libcpp_nmstr
{
private:
    const char* str_;

    typedef std::size_t unused_t;
    typedef std::ptrdiff_t count_t;

    static const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(2*sizeof(unused_t) +
                                                                       sizeof(count_t));

    count_t& count() const _NOEXCEPT {return (count_t&)(*(str_ - sizeof(count_t)));}

#if __APPLE__
    static
    const void*
    compute_gcc_empty_string_storage() _LIBCPP_CANTTHROW
    {
        void* handle = dlopen("libstdc++.dylib", RTLD_LAZY);
        if (handle == 0)
            return 0;
        return (const char*)dlsym(handle, "_ZNSs4_Rep20_S_empty_rep_storageE") + offset;
    }
    
    static
    const void*
    get_gcc_empty_string_storage() _LIBCPP_CANTTHROW
    {
        static const void* p = compute_gcc_empty_string_storage();
        return p;
    }
#endif

public:
    explicit __libcpp_nmstr(const char* msg);
    __libcpp_nmstr(const __libcpp_nmstr& s) _LIBCPP_CANTTHROW;
    __libcpp_nmstr& operator=(const __libcpp_nmstr& s) _LIBCPP_CANTTHROW;
    ~__libcpp_nmstr() _LIBCPP_CANTTHROW;
    const char* c_str() const _NOEXCEPT {return str_;}
};

__libcpp_nmstr::__libcpp_nmstr(const char* msg)
{
    std::size_t len = strlen(msg);
    str_ = new char[len + 1 + offset];
    unused_t* c = (unused_t*)str_;
    c[0] = c[1] = len;
    str_ += offset;
    count() = 0;
    std::strcpy(const_cast<char*>(c_str()), msg);
}

inline
__libcpp_nmstr::__libcpp_nmstr(const __libcpp_nmstr& s)
    : str_(s.str_)
{
#if __APPLE__
    if (str_ != get_gcc_empty_string_storage())
#endif
        __sync_add_and_fetch(&count(), 1);
}

__libcpp_nmstr&
__libcpp_nmstr::operator=(const __libcpp_nmstr& s)
{
    const char* p = str_;
    str_ = s.str_;
#if __APPLE__
    if (str_ != get_gcc_empty_string_storage())
#endif
        __sync_add_and_fetch(&count(), 1);
#if __APPLE__
    if (p != get_gcc_empty_string_storage())
#endif
        if (__sync_add_and_fetch((count_t*)(p-sizeof(count_t)), count_t(-1)) < 0)
            delete [] (p-offset);
    return *this;
}

inline
__libcpp_nmstr::~__libcpp_nmstr()
{
#if __APPLE__
    if (str_ != get_gcc_empty_string_storage())
#endif
        if (__sync_add_and_fetch(&count(), count_t(-1)) < 0)
            delete [] (str_ - offset);
}

}

#pragma GCC visibility pop

namespace std  // purposefully not using versioning namespace
{

logic_error::~logic_error() _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    s.~__libcpp_nmstr();
}

const char*
logic_error::what() const _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    return s.c_str();
}

runtime_error::~runtime_error() _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    s.~__libcpp_nmstr();
}

const char*
runtime_error::what() const _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    return s.c_str();
}

domain_error::~domain_error() _NOEXCEPT {}
invalid_argument::~invalid_argument() _NOEXCEPT {}
length_error::~length_error() _NOEXCEPT {}
out_of_range::~out_of_range() _NOEXCEPT {}

range_error::~range_error() _NOEXCEPT {}
overflow_error::~overflow_error() _NOEXCEPT {}
underflow_error::~underflow_error() _NOEXCEPT {}

}  // std
