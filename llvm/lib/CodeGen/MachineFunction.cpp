//===-- MachineCodeForMethod.cpp -------------------------------------------=//
// 
// Purpose:
//   Collect native machine code information for a function.
//   This allows target-specific information about the generated code
//   to be stored with each function.
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/MachineInstr.h"  // For debug output
#include "llvm/CodeGen/MachineCodeForBasicBlock.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/Target/MachineCacheInfo.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"
#include <limits.h>
#include <iostream>

const int INVALID_FRAME_OFFSET = INT_MAX; // std::numeric_limits<int>::max();

static AnnotationID MCFM_AID(
                 AnnotationManager::getID("CodeGen::MachineCodeForFunction"));

// The next two methods are used to construct and to retrieve
// the MachineCodeForFunction object for the given function.
// construct() -- Allocates and initializes for a given function and target
// get()       -- Returns a handle to the object.
//                This should not be called before "construct()"
//                for a given Function.
// 
MachineCodeForMethod&
MachineCodeForMethod::construct(const Function *M, const TargetMachine &Tar)
{
  assert(M->getAnnotation(MCFM_AID) == 0 &&
         "Object already exists for this function!");
  MachineCodeForMethod* mcInfo = new MachineCodeForMethod(M, Tar);
  M->addAnnotation(mcInfo);
  return *mcInfo;
}

void
MachineCodeForMethod::destruct(const Function *M)
{
  bool Deleted = M->deleteAnnotation(MCFM_AID);
  assert(Deleted && "Machine code did not exist for function!");
}

MachineCodeForMethod&
MachineCodeForMethod::get(const Function *F)
{
  MachineCodeForMethod *mc = (MachineCodeForMethod*)F->getAnnotation(MCFM_AID);
  assert(mc && "Call construct() method first to allocate the object");
  return *mc;
}

static unsigned
ComputeMaxOptionalArgsSize(const TargetMachine& target, const Function *F,
                           unsigned &maxOptionalNumArgs)
{
  const MachineFrameInfo& frameInfo = target.getFrameInfo();
  
  unsigned maxSize = 0;
  
  for (Function::const_iterator BB = F->begin(), BBE = F->end(); BB !=BBE; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (const CallInst *callInst = dyn_cast<CallInst>(&*I))
        {
          unsigned numOperands = callInst->getNumOperands() - 1;
          int numExtra = (int)numOperands-frameInfo.getNumFixedOutgoingArgs();
          if (numExtra <= 0)
            continue;
          
          unsigned int sizeForThisCall;
          if (frameInfo.argsOnStackHaveFixedSize())
            {
              int argSize = frameInfo.getSizeOfEachArgOnStack(); 
              sizeForThisCall = numExtra * (unsigned) argSize;
            }
          else
            {
              assert(0 && "UNTESTED CODE: Size per stack argument is not "
                     "fixed on this architecture: use actual arg sizes to "
                     "compute MaxOptionalArgsSize");
              sizeForThisCall = 0;
              for (unsigned i = 0; i < numOperands; ++i)
                sizeForThisCall += target.findOptimalStorageSize(callInst->
                                              getOperand(i)->getType());
            }
          
          if (maxSize < sizeForThisCall)
            maxSize = sizeForThisCall;
          
          if ((int)maxOptionalNumArgs < numExtra)
            maxOptionalNumArgs = (unsigned) numExtra;
        }
  
  return maxSize;
}

// Align data larger than one L1 cache line on L1 cache line boundaries.
// Align all smaller data on the next higher 2^x boundary (4, 8, ...).
//
// THIS FUNCTION HAS BEEN COPIED FROM EMITASSEMBLY.CPP AND
// SHOULD BE USED DIRECTLY THERE
// 
inline unsigned int
SizeToAlignment(unsigned int size, const TargetMachine& target)
{
  unsigned short cacheLineSize = target.getCacheInfo().getCacheLineSize(1); 
  if (size > (unsigned) cacheLineSize / 2)
    return cacheLineSize;
  else
    for (unsigned sz=1; /*no condition*/; sz *= 2)
      if (sz >= size)
        return sz;
}



/*ctor*/
MachineCodeForMethod::MachineCodeForMethod(const Function *F,
                                           const TargetMachine& target)
  : Annotation(MCFM_AID),
    method(F), staticStackSize(0),
    automaticVarsSize(0), regSpillsSize(0),
    maxOptionalArgsSize(0), maxOptionalNumArgs(0),
    currentTmpValuesSize(0), maxTmpValuesSize(0), compiledAsLeaf(false),
    spillsAreaFrozen(false), automaticVarsAreaFrozen(false)
{
  maxOptionalArgsSize = ComputeMaxOptionalArgsSize(target, method,
                                                   maxOptionalNumArgs);
  staticStackSize = maxOptionalArgsSize
                    + target.getFrameInfo().getMinStackFrameSize();
}

int
MachineCodeForMethod::computeOffsetforLocalVar(const TargetMachine& target,
                                               const Value* val,
                                               unsigned int& getPaddedSize,
                                               unsigned int  sizeToUse = 0)
{
  bool growUp;
  int firstOffset =target.getFrameInfo().getFirstAutomaticVarOffset(*this,
                                                                    growUp);
  unsigned char align;
  if (sizeToUse == 0)
    {
      sizeToUse = target.findOptimalStorageSize(val->getType());
      // align = target.DataLayout.getTypeAlignment(val->getType());
    }
  
  align = SizeToAlignment(sizeToUse, target);
          
  int offset = getAutomaticVarsSize();
  if (! growUp)
    offset += sizeToUse; 
      
  if (unsigned int mod = offset % align)
    {
      offset        += align - mod;
      getPaddedSize  = sizeToUse + align - mod;
    }
  else
    getPaddedSize  = sizeToUse;
  
  offset = growUp? firstOffset + offset
    : firstOffset - offset;
  
  return offset;
}

int
MachineCodeForMethod::allocateLocalVar(const TargetMachine& target,
                                       const Value* val,
                                       unsigned int sizeToUse = 0)
{
  assert(! automaticVarsAreaFrozen &&
         "Size of auto vars area has been used to compute an offset so "
         "no more automatic vars should be allocated!");
  
  // Check if we've allocated a stack slot for this value already
  // 
  int offset = getOffset(val);
  if (offset == INVALID_FRAME_OFFSET)
    {
      unsigned int getPaddedSize;
      offset = this->computeOffsetforLocalVar(target, val, getPaddedSize,
                                              sizeToUse);
      offsets[val] = offset;
      incrementAutomaticVarsSize(getPaddedSize);
    }
  return offset;
}
  
int
MachineCodeForMethod::allocateSpilledValue(const TargetMachine& target,
                                           const Type* type)
{
  assert(! spillsAreaFrozen &&
         "Size of reg spills area has been used to compute an offset so "
         "no more register spill slots should be allocated!");
  
  unsigned int size  = target.findOptimalStorageSize(type);
  unsigned char align = target.DataLayout.getTypeAlignment(type);
  
  bool growUp;
  int firstOffset = target.getFrameInfo().getRegSpillAreaOffset(*this, growUp);
  
  int offset = getRegSpillsSize();
  if (! growUp)
    offset += size; 
  
  if (unsigned int mod = offset % align)
    {
      offset    += align - mod;
      size += align - mod;
    }
  
  offset = growUp? firstOffset + offset
                 : firstOffset - offset;
  
  incrementRegSpillsSize(size);
  
  return offset;
}

int
MachineCodeForMethod::pushTempValue(const TargetMachine& target,
                                    unsigned int size)
{
  // Compute a power-of-2 alignment according to the possible sizes,
  // but not greater than the alignment of the largest type we support
  // (currently a double word -- see class TargetData).
  unsigned char align = 1;
  for (; align < size && align < target.DataLayout.getDoubleAlignment();
         align = 2*align)
    ;
  
  bool growUp;
  int firstTmpOffset = target.getFrameInfo().getTmpAreaOffset(*this, growUp);
  
  int offset = currentTmpValuesSize;
  if (! growUp)
    offset += size; 
  
  if (unsigned int mod = offset % align)
    {
      offset += align - mod;
      size   += align - mod;
    }
  
  offset = growUp ? firstTmpOffset + offset : firstTmpOffset - offset;
  
  incrementTmpAreaSize(size);
  return offset;
}

void
MachineCodeForMethod::popAllTempValues(const TargetMachine& target)
{
  resetTmpAreaSize();
}

int
MachineCodeForMethod::getOffset(const Value* val) const
{
  std::hash_map<const Value*, int>::const_iterator pair = offsets.find(val);
  return (pair == offsets.end())? INVALID_FRAME_OFFSET : pair->second;
}

void
MachineCodeForMethod::dump() const
{
  std::cerr << "\n" << method->getReturnType()
            << " \"" << method->getName() << "\"\n";
  
  for (Function::const_iterator BB = method->begin(); BB != method->end(); ++BB)
    {
      std::cerr << std::endl << (*BB).getName() << " (" << (const void*) BB << ")" << ":" << std::endl;
      MachineCodeForBasicBlock& mvec = MachineCodeForBasicBlock::get(BB);
      for (unsigned i=0; i < mvec.size(); i++)
	std::cerr << "\t" << *mvec[i];
    } 
  std::cerr << "\nEnd function \"" << method->getName() << "\"\n\n";
}
