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
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetCacheInfo.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"
#include "Config/limits.h"

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
      MachineFunction::construct(&F, Target).getInfo()->CalculateArgSize();
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
                                 const TargetMachine &TM)
  : Annotation(MF_AID), Fn(F), Target(TM) {
  SSARegMapping = new SSARegMap();
  MFInfo = new MachineFunctionInfo(*this);
  FrameInfo = new MachineFrameInfo();
  ConstantPool = new MachineConstantPool();
}

MachineFunction::~MachineFunction() { 
  delete SSARegMapping;
  delete MFInfo;
  delete FrameInfo;
  delete ConstantPool;
}

void MachineFunction::dump() const { print(std::cerr); }

void MachineFunction::print(std::ostream &OS) const {
  OS << "\n" << *(Value*)Fn->getFunctionType() << " \"" << Fn->getName()
     << "\"\n";

  // Print Frame Information
  getFrameInfo()->print(*this, OS);

  // Print Constant Pool
  getConstantPool()->print(OS);
  
  for (const_iterator BB = begin(); BB != end(); ++BB) {
    const BasicBlock *LBB = BB->getBasicBlock();
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

//===----------------------------------------------------------------------===//
//  MachineFrameInfo implementation
//===----------------------------------------------------------------------===//

/// CreateStackObject - Create a stack object for a value of the specified type.
///
int MachineFrameInfo::CreateStackObject(const Type *Ty, const TargetData &TD) {
  return CreateStackObject(TD.getTypeSize(Ty), TD.getTypeAlignment(Ty));
}

int MachineFrameInfo::CreateStackObject(const TargetRegisterClass *RC) {
  return CreateStackObject(RC->getSize(), RC->getAlignment());
}


void MachineFrameInfo::print(const MachineFunction &MF, std::ostream &OS) const{
  int ValOffset = MF.getTarget().getFrameInfo().getOffsetOfLocalArea();

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  <fi #" << (int)(i-NumFixedObjects) << "> is ";
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << SO.Size << " byte" << (SO.Size != 1 ? "s" : " ");
    
    if (i < NumFixedObjects)
      OS << " fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int Off = SO.SPOffset + ValOffset;
      OS << " at location [SP";
      if (Off > 0)
	OS << "+" << Off;
      else if (Off < 0)
	OS << Off;
      OS << "]";
    }
    OS << "\n";
  }

  if (HasVarSizedObjects)
    OS << "  Stack frame contains variable sized objects\n";
}

void MachineFrameInfo::dump(const MachineFunction &MF) const {
  print(MF, std::cerr);
}


//===----------------------------------------------------------------------===//
//  MachineConstantPool implementation
//===----------------------------------------------------------------------===//

void MachineConstantPool::print(std::ostream &OS) const {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    OS << "  <cp #" << i << "> is" << *(Value*)Constants[i] << "\n";
}

void MachineConstantPool::dump() const { print(std::cerr); }

//===----------------------------------------------------------------------===//
//  MachineFunctionInfo implementation
//===----------------------------------------------------------------------===//

static unsigned
ComputeMaxOptionalArgsSize(const TargetMachine& target, const Function *F,
                           unsigned &maxOptionalNumArgs)
{
  const TargetFrameInfo &frameInfo = target.getFrameInfo();
  
  unsigned maxSize = 0;
  
  for (Function::const_iterator BB = F->begin(), BBE = F->end(); BB !=BBE; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (const CallInst *callInst = dyn_cast<CallInst>(I))
        {
          unsigned numOperands = callInst->getNumOperands() - 1;
          int numExtra = (int)numOperands-frameInfo.getNumFixedOutgoingArgs();
          if (numExtra <= 0)
            continue;
          
          unsigned sizeForThisCall;
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
                sizeForThisCall += target.getTargetData().getTypeSize(callInst->
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
inline unsigned
SizeToAlignment(unsigned size, const TargetMachine& target)
{
  unsigned short cacheLineSize = target.getCacheInfo().getCacheLineSize(1); 
  if (size > (unsigned) cacheLineSize / 2)
    return cacheLineSize;
  else
    for (unsigned sz=1; /*no condition*/; sz *= 2)
      if (sz >= size || sz >= target.getTargetData().getDoubleAlignment())
        return sz;
}


void MachineFunctionInfo::CalculateArgSize() {
  maxOptionalArgsSize = ComputeMaxOptionalArgsSize(MF.getTarget(),
						   MF.getFunction(),
                                                   maxOptionalNumArgs);
  staticStackSize = maxOptionalArgsSize
    + MF.getTarget().getFrameInfo().getMinStackFrameSize();
}

int
MachineFunctionInfo::computeOffsetforLocalVar(const Value* val,
					      unsigned &getPaddedSize,
					      unsigned  sizeToUse)
{
  if (sizeToUse == 0)
    sizeToUse = MF.getTarget().findOptimalStorageSize(val->getType());
  unsigned align = SizeToAlignment(sizeToUse, MF.getTarget());

  bool growUp;
  int firstOffset = MF.getTarget().getFrameInfo().getFirstAutomaticVarOffset(MF,
									     growUp);
  int offset = growUp? firstOffset + getAutomaticVarsSize()
                     : firstOffset - (getAutomaticVarsSize() + sizeToUse);

  int aligned = MF.getTarget().getFrameInfo().adjustAlignment(offset, growUp, align);
  getPaddedSize = sizeToUse + abs(aligned - offset);

  return aligned;
}

int
MachineFunctionInfo::allocateLocalVar(const Value* val,
				      unsigned sizeToUse)
{
  assert(! automaticVarsAreaFrozen &&
         "Size of auto vars area has been used to compute an offset so "
         "no more automatic vars should be allocated!");
  
  // Check if we've allocated a stack slot for this value already
  // 
  int offset = getOffset(val);
  if (offset == INVALID_FRAME_OFFSET)
    {
      unsigned getPaddedSize;
      offset = computeOffsetforLocalVar(val, getPaddedSize, sizeToUse);
      offsets[val] = offset;
      incrementAutomaticVarsSize(getPaddedSize);
    }
  return offset;
}

int
MachineFunctionInfo::allocateSpilledValue(const Type* type)
{
  assert(! spillsAreaFrozen &&
         "Size of reg spills area has been used to compute an offset so "
         "no more register spill slots should be allocated!");
  
  unsigned size  = MF.getTarget().getTargetData().getTypeSize(type);
  unsigned char align = MF.getTarget().getTargetData().getTypeAlignment(type);
  
  bool growUp;
  int firstOffset = MF.getTarget().getFrameInfo().getRegSpillAreaOffset(MF, growUp);
  
  int offset = growUp? firstOffset + getRegSpillsSize()
                     : firstOffset - (getRegSpillsSize() + size);

  int aligned = MF.getTarget().getFrameInfo().adjustAlignment(offset, growUp, align);
  size += abs(aligned - offset); // include alignment padding in size
  
  incrementRegSpillsSize(size);  // update size of reg. spills area

  return aligned;
}

int
MachineFunctionInfo::pushTempValue(unsigned size)
{
  unsigned align = SizeToAlignment(size, MF.getTarget());

  bool growUp;
  int firstOffset = MF.getTarget().getFrameInfo().getTmpAreaOffset(MF, growUp);

  int offset = growUp? firstOffset + currentTmpValuesSize
                     : firstOffset - (currentTmpValuesSize + size);

  int aligned = MF.getTarget().getFrameInfo().adjustAlignment(offset, growUp,
							      align);
  size += abs(aligned - offset); // include alignment padding in size

  incrementTmpAreaSize(size);    // update "current" size of tmp area

  return aligned;
}

void MachineFunctionInfo::popAllTempValues() {
  resetTmpAreaSize();            // clear tmp area to reuse
}

int
MachineFunctionInfo::getOffset(const Value* val) const
{
  hash_map<const Value*, int>::const_iterator pair = offsets.find(val);
  return (pair == offsets.end()) ? INVALID_FRAME_OFFSET : pair->second;
}
