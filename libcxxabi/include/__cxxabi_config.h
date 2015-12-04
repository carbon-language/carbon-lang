//===-------------------------- __cxxabi_config.h -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ____CXXABI_CONFIG_H
#define ____CXXABI_CONFIG_H

#if defined(__arm__) && !defined(__USING_SJLJ_EXCEPTIONS__) &&                 \
    !defined(__ARM_DWARF_EH__)
#define LIBCXXABI_ARM_EHABI 1
#else
#define LIBCXXABI_ARM_EHABI 0
#endif

#endif // ____CXXABI_CONFIG_H
