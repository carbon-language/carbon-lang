//===------- bswapsi2 - Implement bswapsi2 ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// extern uint32_t __bswapsi2(uint32_t);
//
// Reverse all the bytes in a 32-bit integer.
//
DEFINE_COMPILERRT_FUNCTION(__bswapsi2)
	rev	r0, r0		// reverse bytes in parameter and put into result register
	bx	lr
