//===-- MachineFunction.cpp -----------------------------------------------===//
// 
// Collect native machine code information for a function.  This allows
// target-specific information about the generated code to be stored with each
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/Target/MachineCacheInfo.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"
#include <limits.h>

const int INVALID_FRAME_OFFSET = INT_MAX; // std::numeric_limits<int>::max();

static AnnotationID MF_AID(
                 AnnotationManager::getID("CodeGen::MachineCodeForFunction"));


//===---------------------------------------------------------------------===//
// Code generation/destruction passes
//===---------------------------------------------------------------------===//

namespace {
  class ConstructMachineFunction : public FunctionPass {
    TargetMachine &Target;
  public:
    ConstructMachineFunction(TargetMachine &T) : Target(T) {}
    
    const char *getPassName() const {
      return "ConstructMachineFunction";
    }
    
    bool runOnFunction(Function &F) {
      MachineFunction::construct(&F, Target).CalculateArgSize();
      return false;
    }
  };

  struct DestroyMachineFunction : public FunctionPass {
    const char *getPassName() const { return "FreeMachineFunction"; }
    
    static void freeMachineCode(Instruction &I) {
      MachineCodeForInstruction::destroy(&I);
    }
    
    bool runOnFunction(Function &F) {
      for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
        for (BasicBlock::iterator I = FI->begin(), E = FI->end(); I != E; ++I)
          MachineCodeForInstruction::get(I).dropAllReferences();
      
      for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
        for_each(FI->begin(), FI->end(), freeMachineCode);
      
      return false;
    }
  };

  struct Printer : public FunctionPass {
    const char *getPassName() const { return "MachineFunction Printer"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    bool runOnFunction(Function &F) {
      MachineFunction::get(&F).dump();
      return false;
    }
  };
}

Pass *createMachineCodeConstructionPass(TargetMachine &Target) {
  return new ConstructMachineFunction(Target);
}

Pass *createMachineCodeDestructionPass() {
  return new DestroyMachineFunction();
}

Pass *createMachineFunctionPrinterPass() {
  return new Printer();
}


//===---------------------------------------------------------------------===//
// MachineFunction implementation
//===---------------------------------------------------------------------===//

MachineFunction::MachineFunction(const Function *F,
                                 const TargetMachine& target)
  : Annotation(MF_AID), Fn(F), Target(target) {
  SSARegMapping = new SSARegMap();

  // FIXME: move state into another class
  staticStackSize = automaticVarsSize = regSpillsSize = 0;
  maxOptionalArgsSize = maxOptionalNumArgs = currentTmpValuesSize = 0;
  maxTmpValuesSize = 0;
  compiledAsLeaf = spillsAreaFrozen = automaticVarsAreaFrozen = false;
}

MachineFunction::~MachineFunction() { 
  delete SSARegMapping;
}

void MachineFunction::dump() const { print(std::cerr); }

void MachineFunction::print(std::ostream &OS) const {
  OS << "\n" << *(Value*)Fn->getReturnType() << " \"" << Fn->getName()<< "\"\n";
  
  for (const_iterator BB = begin(); BB != end(); ++BB) {
    BasicBlock *LBB = BB->getBasicBlock();
    OS << "\n" << LBB->getName() << " (" << (const void*)LBB << "):\n";
    for (MachineBasicBlock::const_iterator I = BB->begin(); I != BB->end();++I){
      OS << "\t";
      (*I)->print(OS, Target);
    }
  }
  OS << "\nEnd function \"" << Fn->getName() << "\"\n\n";
}


// The next two methods are used to construct and to retrieve
// the MachineCodeForFunction object for the given function.
// construct() -- Allocates and initializes for a given function and target
// get()       -- Returns a handle to the object.
//                This should not be called before "construct()"
//                for a given Function.
// 
MachineFunction&
MachineFunction::construct(const Function *Fn, const TargetMachine &Tar)
{
  assert(Fn->getAnnotation(MF_AID) == 0 &&
         "Object already exists for this function!");
  MachineFunction* mcInfo = new MachineFunction(Fn, Tar);
  Fn->addAnnotation(mcInfo);
  return *mcInfo;
}

void
MachineFunction::destruct(const Function *Fn)
{
  bool Deleted = Fn->deleteAnnotation(MF_AID);
  assert(Deleted && "Machine code did not exist for function!");
}

MachineFunction& MachineFunction::get(const Function *F)
{
  MachineFunction *mc = (MachineFunction*)F->getAnnotation(MF_AID);
  assert(mc && "Call construct() method first to allocate the object");
  return *mc;
}

void MachineFunction::clearSSARegMap() {
  delete SSARegMapping;
  SSARegMapping = 0;
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
                sizeForThisCall += target.DataLayout.getTypeSize(callInst->
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
// Align all smaller data on the next higher 2^x boundary (4, 8, ...),
// but not higher than the alignment of the largest type we support
// (currently a double word). -- see class TargetData).
//
// This function is similar to the corresponding function in EmitAssembly.cpp
// but they are unrelated.  This one does not align at more than a
// double-word boundary whereas that one might.
// 
inline unsigned int
SizeToAlignment(unsigned int size, const TargetMachine& target)
{
  unsigned short cacheLineSize = target.getCacheInfo().getCacheLineSize(1); 
  if (size > (unsigned) cacheLineSize / 2)
    return cacheLineSize;
  else
    for (unsigned sz=1; /*no condition*/; sz *= 2)
      if (sz >= size || sz >= target.DataLayout.getDoubleAlignment())
        return sz;
}


void MachineFunction::CalculateArgSize() {
  maxOptionalArgsSize = ComputeMaxOptionalArgsSize(Target, Fn,
                                                   maxOptionalNumArgs);
  staticStackSize = maxOptionalArgsSize
    + Target.getFrameInfo().getMinStackFrameSize();
}

int
MachineFunction::computeOffsetforLocalVar(const TargetMachine& target,
                                               const Value* val,
                                               unsigned int& getPaddedSize,
                                               unsigned int  sizeToUse)
{
  if (sizeToUse == 0)
    sizeToUse = target.findOptimalStorageSize(val->getType());
  unsigned int align = SizeToAlignment(sizeToUse, target);

  bool growUp;
  int firstOffset = target.getFrameInfo().getFirstAutomaticVarOffset(*this,
                                                                     growUp);
  int offset = growUp? firstOffset + getAutomaticVarsSize()
                     : firstOffset - (getAutomaticVarsSize() + sizeToUse);

  int aligned = target.getFrameInfo().adjustAlignment(offset, growUp, align);
  getPaddedSize = sizeToUse + abs(aligned - offset);

  return aligned;
}

int
MachineFunction::allocateLocalVar(const TargetMachine& target,
                                       const Value* val,
                                       unsigned int sizeToUse)
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
      offset = computeOffsetforLocalVar(target, val, getPaddedSize, sizeToUse);
      offsets[val] = offset;
      incrementAutomaticVarsSize(getPaddedSize);
    }
  return offset;
}

int
MachineFunction::allocateSpilledValue(const TargetMachine& target,
                                           const Type* type)
{
  assert(! spillsAreaFrozen &&
         "Size of reg spills area has been used to compute an offset so "
         "no more register spill slots should be allocated!");
  
  unsigned int size  = target.DataLayout.getTypeSize(type);
  unsigned char align = target.DataLayout.getTypeAlignment(type);
  
  bool growUp;
  int firstOffset = target.getFrameInfo().getRegSpillAreaOffset(*this, growUp);
  
  int offset = growUp? firstOffset + getRegSpillsSize()
                     : firstOffset - (getRegSpillsSize() + size);

  int aligned = target.getFrameInfo().adjustAlignment(offset, growUp, align);
  size += abs(aligned - offset); // include alignment padding in size
  
  incrementRegSpillsSize(size);  // update size of reg. spills area

  return aligned;
}

int
MachineFunction::pushTempValue(const TargetMachine& target,
                                    unsigned int size)
{
  unsigned int align = SizeToAlignment(size, target);

  bool growUp;
  int firstOffset = target.getFrameInfo().getTmpAreaOffset(*this, growUp);

  int offset = growUp? firstOffset + currentTmpValuesSize
                     : firstOffset - (currentTmpValuesSize + size);

  int aligned = target.getFrameInfo().adjustAlignment(offset, growUp, align);
  size += abs(aligned - offset); // include alignment padding in size

  incrementTmpAreaSize(size);    // update "current" size of tmp area

  return aligned;
}

void
MachineFunction::popAllTempValues(const TargetMachine& target)
{
  resetTmpAreaSize();            // clear tmp area to reuse
}

int
MachineFunction::getOffset(const Value* val) const
{
  hash_map<const Value*, int>::const_iterator pair = offsets.find(val);
  return (pair == offsets.end()) ? INVALID_FRAME_OFFSET : pair->second;
}
