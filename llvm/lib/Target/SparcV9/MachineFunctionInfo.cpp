//===-- SparcV9FunctionInfo.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SparcV9 specific MachineFunctionInfo class.
//
//===----------------------------------------------------------------------===//

#include "MachineFunctionInfo.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
using namespace llvm;

static unsigned
ComputeMaxOptionalArgsSize(const TargetMachine& target, const Function *F,
                           unsigned &maxOptionalNumArgs)
{
  unsigned maxSize = 0;

  for (Function::const_iterator BB = F->begin(), BBE = F->end(); BB !=BBE; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (const CallInst *callInst = dyn_cast<CallInst>(I))
        {
          unsigned numOperands = callInst->getNumOperands() - 1;
          int numExtra = numOperands-6;
          if (numExtra <= 0)
            continue;

          unsigned sizeForThisCall = numExtra * 8;

          if (maxSize < sizeForThisCall)
            maxSize = sizeForThisCall;

          if ((int)maxOptionalNumArgs < numExtra)
            maxOptionalNumArgs = (unsigned) numExtra;
        }

  return maxSize;
}

// Align data larger than one L1 cache line on L1 cache line boundaries.
// Align all smaller data on the next higher 2^x boundary (4, 8, ...),
// but not higher than the alignment of the largest type we support
// (currently a double word). -- see class TargetData).
//
// This function is similar to the corresponding function in EmitAssembly.cpp
// but they are unrelated.  This one does not align at more than a
// double-word boundary whereas that one might.
//
inline unsigned
SizeToAlignment(unsigned size, const TargetMachine& target)
{
  const unsigned short cacheLineSize = 16;
  if (size > (unsigned) cacheLineSize / 2)
    return cacheLineSize;
  else
    for (unsigned sz=1; /*no condition*/; sz *= 2)
      if (sz >= size || sz >= target.getTargetData().getDoubleAlignment())
        return sz;
}


void SparcV9FunctionInfo::CalculateArgSize() {
  maxOptionalArgsSize = ComputeMaxOptionalArgsSize(MF.getTarget(),
						   MF.getFunction(),
                                                   maxOptionalNumArgs);
  staticStackSize = maxOptionalArgsSize + 176;
}

int
SparcV9FunctionInfo::computeOffsetforLocalVar(const Value* val,
					      unsigned &getPaddedSize,
					      unsigned  sizeToUse)
{
  if (sizeToUse == 0) {
    // All integer types smaller than ints promote to 4 byte integers.
    if (val->getType()->isIntegral() && val->getType()->getPrimitiveSize() < 4)
      sizeToUse = 4;
    else
      sizeToUse = MF.getTarget().getTargetData().getTypeSize(val->getType());
  }
  unsigned align = SizeToAlignment(sizeToUse, MF.getTarget());

  bool growUp;
  int firstOffset = MF.getTarget().getFrameInfo()->getFirstAutomaticVarOffset(MF,
						 			     growUp);
  int offset = growUp? firstOffset + getAutomaticVarsSize()
                     : firstOffset - (getAutomaticVarsSize() + sizeToUse);

  int aligned = MF.getTarget().getFrameInfo()->adjustAlignment(offset, growUp, align);
  getPaddedSize = sizeToUse + abs(aligned - offset);

  return aligned;
}


int SparcV9FunctionInfo::allocateLocalVar(const Value* val,
                                          unsigned sizeToUse) {
  assert(! automaticVarsAreaFrozen &&
         "Size of auto vars area has been used to compute an offset so "
         "no more automatic vars should be allocated!");

  // Check if we've allocated a stack slot for this value already
  //
  hash_map<const Value*, int>::const_iterator pair = offsets.find(val);
  if (pair != offsets.end())
    return pair->second;

  unsigned getPaddedSize;
  unsigned offset = computeOffsetforLocalVar(val, getPaddedSize, sizeToUse);
  offsets[val] = offset;
  incrementAutomaticVarsSize(getPaddedSize);
  return offset;
}

int
SparcV9FunctionInfo::allocateSpilledValue(const Type* type)
{
  assert(! spillsAreaFrozen &&
         "Size of reg spills area has been used to compute an offset so "
         "no more register spill slots should be allocated!");

  unsigned size  = MF.getTarget().getTargetData().getTypeSize(type);
  unsigned char align = MF.getTarget().getTargetData().getTypeAlignment(type);

  bool growUp;
  int firstOffset = MF.getTarget().getFrameInfo()->getRegSpillAreaOffset(MF, growUp);

  int offset = growUp? firstOffset + getRegSpillsSize()
                     : firstOffset - (getRegSpillsSize() + size);

  int aligned = MF.getTarget().getFrameInfo()->adjustAlignment(offset, growUp, align);
  size += abs(aligned - offset); // include alignment padding in size

  incrementRegSpillsSize(size);  // update size of reg. spills area

  return aligned;
}

int
SparcV9FunctionInfo::pushTempValue(unsigned size)
{
  unsigned align = SizeToAlignment(size, MF.getTarget());

  bool growUp;
  int firstOffset = MF.getTarget().getFrameInfo()->getTmpAreaOffset(MF, growUp);

  int offset = growUp? firstOffset + currentTmpValuesSize
                     : firstOffset - (currentTmpValuesSize + size);

  int aligned = MF.getTarget().getFrameInfo()->adjustAlignment(offset, growUp,
							      align);
  size += abs(aligned - offset); // include alignment padding in size

  incrementTmpAreaSize(size);    // update "current" size of tmp area

  return aligned;
}

void SparcV9FunctionInfo::popAllTempValues() {
  resetTmpAreaSize();            // clear tmp area to reuse
}
