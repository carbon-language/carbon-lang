//===-- gtest_common.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(LLDB_GTEST_COMMON_H)
#error "gtest_common.h should not be included manually."
#else
#define LLDB_GTEST_COMMON_H
#endif

// This header file is force included by all of LLDB's unittest compilation
// units.  Be very leary about putting anything in this file.

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// MSVC's STL implementation tries to work well with /EHs-c- and
// _HAS_EXCEPTIONS=0.  But <thread> in particular doesn't work with it, because
// it relies on <concrt.h> which tries to throw an exception without checking
// for _HAS_EXCEPTIONS=0.  This causes the linker to require a definition of
// __uncaught_exception(), but the STL doesn't define this function when
// _HAS_EXCEPTIONS=0.  The workaround here is to just provide a stub
// implementation to get it to link.
inline bool
__uncaught_exception()
{
    return true;
}
#endif
