//===-- lldb_ARMUtils.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_ARMUtils_h_
#define lldb_ARMUtils_h_

// Utility functions for the ARM/Thumb Instruction Set Architecture.

namespace lldb_private {

// This function performs the check for the register numbers 13 and 15 that are
// not permitted for many Thumb register specifiers.
static inline bool BadReg(uint32_t n) { return n == 13 || n == 15; }

// Returns an integer result equal to the number of bits of x that are ones.
static inline uint32_t BitCount(uint32_t x)
{
    // c accumulates the total bits set in x
    uint32_t c;
    for (c = 0; x; ++c)
    {
        x &= x - 1; // clear the least significant bit set
    }
    return c;
}

}   // namespace lldb_private

#endif  // lldb_ARMUtils_h_
