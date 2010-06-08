/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- libunwind_priv.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __LIBUNWIND_PRIV__
#define __LIBUNWIND_PRIV__

namespace lldb_private {
#include "libunwind.h"

#ifdef __cplusplus
extern "C" {
#endif
	// SPI
	extern void unw_iterate_dwarf_unwind_cache(void (*func)(unw_word_t ip_start, unw_word_t ip_end, unw_word_t fde, unw_word_t mh));

	// IPI
	extern void _unw_add_dynamic_fde(unw_word_t fde);
	extern void _unw_remove_dynamic_fde(unw_word_t fde);

#ifdef __cplusplus
}
#endif

}; // namespace lldb_private


#endif

