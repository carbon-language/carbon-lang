//===- BitCodes.h - Enum values for the bitcode format ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header Bitcode enum values.
//
// The enum values defined in this file should be considered permanent.  If
// new features are added, they should have values added at the end of the
// respective lists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_BITCODES_H
#define LLVM_BITCODE_BITCODES_H

namespace llvm {
namespace bitc {
  enum StandardWidths {
    BlockIDWidth = 8,  // We use VBR-8 for block IDs.
    CodeLenWidth = 4,  // Codelen are VBR-4.
    BlockSizeWidth = 32  // BlockSize up to 2^32 32-bit words = 32GB per block.
  };
  
  // The standard code namespace always has a way to exit a block, enter a
  // nested block, define abbrevs, and define an unabbreviated record.
  enum FixedCodes {
    END_BLOCK = 0,  // Must be zero to guarantee termination for broken bitcode.
    ENTER_SUBBLOCK = 1,
    
    // Two codes are reserved for defining abbrevs and for emitting an
    // unabbreviated record.
    DEFINE_ABBREVS = 2,
    UNABBREV_RECORD = 3,
    
    // This is not a code, this is a marker for the first abbrev assignment.
    FIRST_ABBREV = 4
  };
} // End bitc namespace
} // End llvm namespace

#endif
