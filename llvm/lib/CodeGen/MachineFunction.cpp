//===-- MachineFunction.cpp -----------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Collect native machine code information for a function.  This allows
// target-specific information about the generated code to be stored with each
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/Type.h"
#include "Support/LeakDetector.h"

using namespace llvm;

static AnnotationID MF_AID(
                 AnnotationManager::getID("CodeGen::MachineCodeForFunction"));


namespace {
  struct Printer : public MachineFunctionPass {
    std::ostream *OS;
    const std::string Banner;

    Printer (std::ostream *_OS, const std::string &_Banner) :
      OS (_OS), Banner (_Banner) { }

    const char *getPassName() const { return "MachineFunction Printer"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    bool runOnMachineFunction(MachineFunction &MF) {
      (*OS) << Banner;
      MF.print (*OS);
      return false;
    }
  };
}

/// Returns a newly-created MachineFunction Printer pass. The default output
/// stream is std::cerr; the default banner is empty.
///
FunctionPass *llvm::createMachineFunctionPrinterPass(std::ostream *OS,
                                                     const std::string &Banner) {
  return new Printer(OS, Banner);
}

namespace {
  struct Deleter : public MachineFunctionPass {
    const char *getPassName() const { return "Machine Code Deleter"; }

    bool runOnMachineFunction(MachineFunction &MF) {
      // Delete the annotation from the function now.
      MachineFunction::destruct(MF.getFunction());
      return true;
    }
  };
}

/// MachineCodeDeletion Pass - This pass deletes all of the machine code for
/// the current function, which should happen after the function has been
/// emitted to a .s file or to memory.
FunctionPass *llvm::createMachineCodeDeleter() {
  return new Deleter();
}



//===---------------------------------------------------------------------===//
// MachineFunction implementation
//===---------------------------------------------------------------------===//
MachineBasicBlock* ilist_traits<MachineBasicBlock>::createNode()
{
    MachineBasicBlock* dummy = new MachineBasicBlock();
    LeakDetector::removeGarbageObject(dummy);
    return dummy;
}

void ilist_traits<MachineBasicBlock>::transferNodesFromList(
    iplist<MachineBasicBlock, ilist_traits<MachineBasicBlock> >& toList,
    ilist_iterator<MachineBasicBlock> first,
    ilist_iterator<MachineBasicBlock> last)
{
    if (Parent != toList.Parent)
        for (; first != last; ++first)
            first->Parent = toList.Parent;
}

MachineFunction::MachineFunction(const Function *F,
                                 const TargetMachine &TM)
  : Annotation(MF_AID), Fn(F), Target(TM), NextMBBNumber(0) {
  SSARegMapping = new SSARegMap();
  MFInfo = new MachineFunctionInfo(*this);
  FrameInfo = new MachineFrameInfo();
  ConstantPool = new MachineConstantPool();
  BasicBlocks.Parent = this;
}

MachineFunction::~MachineFunction() { 
  delete SSARegMapping;
  delete MFInfo;
  delete FrameInfo;
  delete ConstantPool;
}

void MachineFunction::dump() const { print(std::cerr); }

void MachineFunction::print(std::ostream &OS) const {
  OS << "# Machine code for " << Fn->getName () << "():\n";

  // Print Frame Information
  getFrameInfo()->print(*this, OS);

  // Print Constant Pool
  getConstantPool()->print(OS);
  
  for (const_iterator BB = begin(); BB != end(); ++BB)
    BB->print(OS);

  OS << "\n# End machine code for " << Fn->getName () << "().\n\n";
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

void MachineFunction::destruct(const Function *Fn) {
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
  int ValOffset = MF.getTarget().getFrameInfo()->getOffsetOfLocalArea();

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
  const TargetFrameInfo &frameInfo = *target.getFrameInfo();
  
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
  const unsigned short cacheLineSize = 16;
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
    + MF.getTarget().getFrameInfo()->getMinStackFrameSize();
}

int
MachineFunctionInfo::computeOffsetforLocalVar(const Value* val,
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


int MachineFunctionInfo::allocateLocalVar(const Value* val,
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
MachineFunctionInfo::allocateSpilledValue(const Type* type)
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
MachineFunctionInfo::pushTempValue(unsigned size)
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

void MachineFunctionInfo::popAllTempValues() {
  resetTmpAreaSize();            // clear tmp area to reuse
}

