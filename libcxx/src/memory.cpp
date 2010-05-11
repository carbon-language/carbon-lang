//===------------------------ memory.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "memory"
#include <libkern/OSAtomic.h>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace
{

template <class T>
inline
typename enable_if
<
    sizeof(T) * __CHAR_BIT__ == 32,
    T
>::type
increment(T& t)
{
    return OSAtomicIncrement32Barrier((volatile int32_t*)&t);
}

template <class T>
inline
typename enable_if
<
    sizeof(T) * __CHAR_BIT__ == 32,
    T
>::type
decrement(T& t)
{
    return OSAtomicDecrement32Barrier((volatile int32_t*)&t);
}

#ifndef __ppc__

template <class T>
inline
typename enable_if
<
    sizeof(T) * __CHAR_BIT__ == 64,
    T
>::type
increment(T& t)
{
    return OSAtomicIncrement64Barrier((volatile int64_t*)&t);
}

template <class T>
inline
typename enable_if
<
    sizeof(T) * __CHAR_BIT__ == 64,
    T
>::type
decrement(T& t)
{
    return OSAtomicDecrement64Barrier((volatile int64_t*)&t);
}

#endif

}  // namespace


const allocator_arg_t allocator_arg = allocator_arg_t();

bad_weak_ptr::~bad_weak_ptr() throw() {}

const char*
bad_weak_ptr::what() const throw()
{
    return "bad_weak_ptr";
}

__shared_count::~__shared_count()
{
}

void
__shared_count::__add_shared()
{
    increment(__shared_owners_);
}

void
__shared_count::__release_shared()
{
    if (decrement(__shared_owners_) == -1)
        __on_zero_shared();
}

__shared_weak_count::~__shared_weak_count()
{
}

void
__shared_weak_count::__add_shared()
{
    __shared_count::__add_shared();
    __add_weak();
}

void
__shared_weak_count::__add_weak()
{
    increment(__shared_weak_owners_);
}

void
__shared_weak_count::__release_shared()
{
    __shared_count::__release_shared();
    __release_weak();
}

void
__shared_weak_count::__release_weak()
{
    if (decrement(__shared_weak_owners_) == -1)
        __on_zero_shared_weak();
}

__shared_weak_count*
__shared_weak_count::lock()
{
    long object_owners = __shared_owners_;
    while (object_owners != -1)
    {
        if (OSAtomicCompareAndSwapLongBarrier(object_owners,
                                              object_owners+1,
                                              &__shared_owners_))
        {
            __add_weak();
            return this;
        }
        object_owners = __shared_owners_;
    }
    return 0;
}

const void*
__shared_weak_count::__get_deleter(const type_info&) const
{
    return 0;
}

void
declare_reachable(void*)
{
}

void
declare_no_pointers(char*, size_t)
{
}

void
undeclare_no_pointers(char*, size_t)
{
}

pointer_safety
get_pointer_safety()
{
    return pointer_safety::relaxed;
}

void*
__undeclare_reachable(void* p)
{
    return p;
}

void*
align(size_t alignment, size_t size, void*& ptr, size_t& space)
{
    void* r = nullptr;
    if (size <= space)
    {
        char* p1 = static_cast<char*>(ptr);
        char* p2 = (char*)((size_t)(p1 + (alignment - 1)) & -alignment);
        ptrdiff_t d = p2 - p1;
        if (d <= space - size)
        {
            r = p2;
            ptr = r;
            space -= d;
        }
    }
    return r;
}

_LIBCPP_END_NAMESPACE_STD
