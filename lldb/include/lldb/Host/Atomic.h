//===-- Atomic.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Atomic_h_
#define liblldb_Atomic_h_

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace lldb_private {

#ifdef _MSC_VER
typedef long cas_flag;
#else
typedef uint32_t cas_flag;
#endif

inline cas_flag
AtomicIncrement(volatile cas_flag* ptr)
{
#ifdef _MSC_VER
    return _InterlockedIncrement(ptr);
#else
    return __sync_add_and_fetch(ptr, 1);
#endif
}

inline cas_flag
AtomicDecrement(volatile cas_flag* ptr)
{
#ifdef _MSC_VER
    return _InterlockedDecrement(ptr);
#else
    return __sync_add_and_fetch(ptr, -1);
#endif
}

}

#endif
