//===-- MachineCodeForMethod.cpp --------------------------------------------=//
// 
// Purpose:
//   Collect native machine code information for a method.
//   This allows target-specific information about the generated code
//   to be stored with each method.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/MachineInstr.h"  // For debug output
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/Target/MachineCacheInfo.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/iOther.h"
#include <limits.h>
#include <iostream>

const int INVALID_FRAME_OFFSET = INT_MAX; // std::numeric_limits<int>::max();

static AnnotationID MCFM_AID(
                 AnnotationManager::getID("CodeGen::MachineCodeForMethod"));

// The next two methods are used to construct and to retrieve
// the MachineCodeForMethod object for the given method.
// construct() -- Allocates and initializes for a given method and target
// get()       -- Returns a handle to the object.
//                This should not be called before "construct()"
//                for a given Method.
// 
MachineCodeForMethod &MachineCodeForMethod::construct(const Method *M,
                                                      const TargetMachine &Tar){
  assert(M->getAnnotation(MCFM_AID) == 0 &&
         "Object already exists for this method!");
  MachineCodeForMethod* mcInfo = new MachineCodeForMethod(M, Tar);
  M->addAnnotation(mcInfo);
  return *mcInfo;
}

void MachineCodeForMethod::destruct(const Method *M) {
  bool Deleted = M->deleteAnnotation(MCFM_AID);
  assert(Deleted && "Machine code did not exist for method!");
}


MachineCodeForMethod &MachineCodeForMethod::get(const Method* method) {
  MachineCodeForMethod* mc = (MachineCodeForMethod*)
    method->getAnnotation(MCFM_AID);
  assert(mc && "Call construct() method first to allocate the object");
  return *mc;
}

static unsigned
ComputeMaxOptionalArgsSize(const TargetMachine& target, const Method* method)
{
  const MachineFrameInfo& frameInfo = target.getFrameInfo();
  
  unsigned int maxSize = 0;
  
  for (Method::const_iterator MI=method->begin(), ME=method->end();
       MI != ME; ++MI) {
    const BasicBlock *BB = *MI;
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (CallInst *callInst = dyn_cast<CallInst>(*I)) {
        unsigned int numOperands = callInst->getNumOperands() - 1;
        int numExtra = (int) numOperands - frameInfo.getNumFixedOutgoingArgs();
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
            assert(0 && "UNTESTED CODE: Size per stack argument is not fixed on this architecture: use actual arg sizes to compute MaxOptionalArgsSize");
            sizeForThisCall = 0;
            for (unsigned i=0; i < numOperands; ++i)
              sizeForThisCall += target.findOptimalStorageSize(callInst->
                                                    getOperand(i)->getType());
          }
        
        if (maxSize < sizeForThisCall)
          maxSize = sizeForThisCall;
      }
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
MachineCodeForMethod::MachineCodeForMethod(const Method* _M,
                                           const TargetMachine& target)
  : Annotation(MCFM_AID),
    method(_M), compiledAsLeaf(false), staticStackSize(0),
    automaticVarsSize(0), regSpillsSize(0),
    currentOptionalArgsSize(0), maxOptionalArgsSize(0),
    currentTmpValuesSize(0)
{
  maxOptionalArgsSize = ComputeMaxOptionalArgsSize(target, method);
  staticStackSize = maxOptionalArgsSize +
                    target.getFrameInfo().getMinStackFrameSize();
}

int
MachineCodeForMethod::allocateLocalVar(const TargetMachine& target,
                                       const Value* val,
                                       unsigned int size)
{
  // Check if we've allocated a stack slot for this value already
  // 
  int offset = getOffset(val);
  if (offset == INVALID_FRAME_OFFSET)
    {
      bool growUp;
      int firstOffset =target.getFrameInfo().getFirstAutomaticVarOffset(*this,
                                                                       growUp);
      unsigned char align;
      if (size == 0)
        {
          size  = target.findOptimalStorageSize(val->getType());
          // align = target.DataLayout.getTypeAlignment(val->getType());
        }
      
      align = SizeToAlignment(size, target);
          
      offset = getAutomaticVarsSize();
      if (! growUp)
        offset += size; 
      
      if (unsigned int mod = offset % align)
        {
          offset += align - mod;
          size   += align - mod;
        }
      
      offset = growUp? firstOffset + offset
                     : firstOffset - offset;
      
      offsets[val] = offset;
      
      incrementAutomaticVarsSize(size);
    }
  return offset;
}

int
MachineCodeForMethod::allocateSpilledValue(const TargetMachine& target,
                                           const Type* type)
{
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
MachineCodeForMethod::allocateOptionalArg(const TargetMachine& target,
                                          const Type* type)
{
  const MachineFrameInfo& frameInfo = target.getFrameInfo();
  
  int size = INT_MAX;
  if (frameInfo.argsOnStackHaveFixedSize())
    size = frameInfo.getSizeOfEachArgOnStack(); 
  else
    {
      size = target.findOptimalStorageSize(type);
      assert(0 && "UNTESTED CODE: Size per stack argument is not fixed on this architecture: use actual argument sizes for computing optional arg offsets");
    }
  unsigned char align = target.DataLayout.getTypeAlignment(type);
  
  bool growUp;
  int firstOffset = frameInfo.getFirstOptionalOutgoingArgOffset(*this, growUp);
  
  int offset = getCurrentOptionalArgsSize();
  if (! growUp)
    offset += size; 
  
  if (unsigned int mod = offset % align)
    {
      offset += align - mod;
      size   += align - mod;
    }
  
  offset = growUp? firstOffset + offset
                 : firstOffset - offset;
  
  incrementCurrentOptionalArgsSize(size);
  
  return offset;
}

void
MachineCodeForMethod::resetOptionalArgs(const TargetMachine& target)
{
  currentOptionalArgsSize = 0;
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
  
  currentTmpValuesSize += size;
  return offset;
}

void
MachineCodeForMethod::popAllTempValues(const TargetMachine& target)
{
  currentTmpValuesSize = 0;
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
  
  for (Method::const_iterator BI = method->begin(); BI != method->end(); ++BI)
    {
      BasicBlock* bb = *BI;
      std::cerr << "\n" << bb->getName() << " (" << bb << ")" << ":\n";

      MachineCodeForBasicBlock& mvec = bb->getMachineInstrVec();
      for (unsigned i=0; i < mvec.size(); i++)
	std::cerr << "\t" << *mvec[i];
    } 
  std::cerr << "\nEnd method \"" << method->getName() << "\"\n\n";
}
