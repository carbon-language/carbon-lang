//===-- RegAllocCommon.h --------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
//  Shared declarations for register allocation.
// 
//===----------------------------------------------------------------------===//

#ifndef REGALLOCCOMMON_H
#define REGALLOCCOMMON_H

namespace llvm {

enum RegAllocDebugLevel_t {
  RA_DEBUG_None         = 0,
  RA_DEBUG_Results      = 1,
  RA_DEBUG_Coloring     = 2,
  RA_DEBUG_Interference = 3,
  RA_DEBUG_LiveRanges   = 4,
  RA_DEBUG_Verbose      = 5
};

extern RegAllocDebugLevel_t DEBUG_RA;

} // End llvm namespace

#endif
