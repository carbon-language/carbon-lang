//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

TargetLowering::TargetLowering(TargetMachine &tm)
  : TM(tm), TD(TM.getTargetData()) {
  assert(ISD::BUILTIN_OP_END <= 128 &&
         "Fixed size array in TargetLowering is not large enough!");

  IsLittleEndian = TD.isLittleEndian();
  PointerTy = getValueType(TD.getIntPtrType());
  memset(UnsupportedOps, 0, ISD::BUILTIN_OP_END*sizeof(short));
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
}

/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  // Everything defaults to one.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i)
    NumElementsForVT[i] = 1;
  
  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::i128;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  unsigned ExpandedReg = LargestIntReg; ++LargestIntReg;
  for (++ExpandedReg; MVT::isInteger((MVT::ValueType)ExpandedReg);++ExpandedReg)
    NumElementsForVT[ExpandedReg] = 2*NumElementsForVT[ExpandedReg-1];
}

