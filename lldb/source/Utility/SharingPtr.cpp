//===---------------------SharingPtr.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/SharingPtr.h"

namespace lldb_private {

namespace imp
{

template <class T>
inline T
increment(T& t)
{
    return __sync_add_and_fetch(&t, 1);
}

template <class T>
inline T
decrement(T& t)
{
    return __sync_add_and_fetch(&t, -1);
}

shared_count::~shared_count()
{
}

void
shared_count::add_shared()
{
    increment(shared_owners_);
}

void
shared_count::release_shared()
{
    if (decrement(shared_owners_) == -1)
    {
        on_zero_shared();
        delete this;
    }
}

} // imp

} // namespace lldb
