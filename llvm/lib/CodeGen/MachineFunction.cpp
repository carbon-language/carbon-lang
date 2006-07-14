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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Visibility.h"
#include "llvm/Config/config.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace llvm;

static AnnotationID MF_AID(
  AnnotationManager::getID("CodeGen::MachineCodeForFunction"));

// Out of line virtual function to home classes.
void MachineFunctionPass::virtfn() {}

namespace {
  struct VISIBILITY_HIDDEN Printer : public MachineFunctionPass {
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
                                                     const std::string &Banner){
  return new Printer(OS, Banner);
}

namespace {
  struct VISIBILITY_HIDDEN Deleter : public MachineFunctionPass {
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

MachineBasicBlock* ilist_traits<MachineBasicBlock>::createSentinel() {
  MachineBasicBlock* dummy = new MachineBasicBlock();
  LeakDetector::removeGarbageObject(dummy);
  return dummy;
}

void ilist_traits<MachineBasicBlock>::transferNodesFromList(
  iplist<MachineBasicBlock, ilist_traits<MachineBasicBlock> >& toList,
  ilist_iterator<MachineBasicBlock> first,
  ilist_iterator<MachineBasicBlock> last) {
  if (Parent != toList.Parent)
    for (; first != last; ++first)
      first->Parent = toList.Parent;
}

MachineFunction::MachineFunction(const Function *F,
                                 const TargetMachine &TM)
  : Annotation(MF_AID), Fn(F), Target(TM), UsedPhysRegs(0) {
  SSARegMapping = new SSARegMap();
  MFInfo = 0;
  FrameInfo = new MachineFrameInfo();
  ConstantPool = new MachineConstantPool(TM.getTargetData());
  JumpTableInfo = new MachineJumpTableInfo(TM.getTargetData());
  BasicBlocks.Parent = this;
}

MachineFunction::~MachineFunction() {
  BasicBlocks.clear();
  delete SSARegMapping;
  delete MFInfo;
  delete FrameInfo;
  delete ConstantPool;
  delete JumpTableInfo;
  delete[] UsedPhysRegs;
}

void MachineFunction::dump() const { print(std::cerr); }

void MachineFunction::print(std::ostream &OS) const {
  OS << "# Machine code for " << Fn->getName () << "():\n";

  // Print Frame Information
  getFrameInfo()->print(*this, OS);
  
  // Print JumpTable Information
  getJumpTableInfo()->print(OS);

  // Print Constant Pool
  getConstantPool()->print(OS);
  
  const MRegisterInfo *MRI = getTarget().getRegisterInfo();
  
  if (livein_begin() != livein_end()) {
    OS << "Live Ins:";
    for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I) {
      if (MRI)
        OS << " " << MRI->getName(I->first);
      else
        OS << " Reg #" << I->first;
      
      if (I->second)
        OS << " in VR#" << I->second << " ";
    }
    OS << "\n";
  }
  if (liveout_begin() != liveout_end()) {
    OS << "Live Outs:";
    for (liveout_iterator I = liveout_begin(), E = liveout_end(); I != E; ++I)
      if (MRI)
        OS << " " << MRI->getName(*I);
      else
        OS << " Reg #" << *I;
    OS << "\n";
  }
  
  for (const_iterator BB = begin(); BB != end(); ++BB)
    BB->print(OS);

  OS << "\n# End machine code for " << Fn->getName () << "().\n\n";
}

/// CFGOnly flag - This is used to control whether or not the CFG graph printer
/// prints out the contents of basic blocks or not.  This is acceptable because
/// this code is only really used for debugging purposes.
///
static bool CFGOnly = false;

namespace llvm {
  template<>
  struct DOTGraphTraits<const MachineFunction*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(const MachineFunction *F) {
      return "CFG for '" + F->getFunction()->getName() + "' function";
    }

    static std::string getNodeLabel(const MachineBasicBlock *Node,
                                    const MachineFunction *Graph) {
      if (CFGOnly && Node->getBasicBlock() &&
          !Node->getBasicBlock()->getName().empty())
        return Node->getBasicBlock()->getName() + ":";

      std::ostringstream Out;
      if (CFGOnly) {
        Out << Node->getNumber() << ':';
        return Out.str();
      }

      Node->print(Out);

      std::string OutStr = Out.str();
      if (OutStr[0] == '\n') OutStr.erase(OutStr.begin());

      // Process string output to make it nicer...
      for (unsigned i = 0; i != OutStr.length(); ++i)
        if (OutStr[i] == '\n') {                            // Left justify
          OutStr[i] = '\\';
          OutStr.insert(OutStr.begin()+i+1, 'l');
        }
      return OutStr;
    }
  };
}

void MachineFunction::viewCFG() const
{
#ifndef NDEBUG
  ViewGraph(this, "mf" + getFunction()->getName());
#else
  std::cerr << "SelectionDAG::viewGraph is only available in debug builds on "
            << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

void MachineFunction::viewCFGOnly() const
{
  CFGOnly = true;
  viewCFG();
  CFGOnly = false;
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

void MachineFrameInfo::print(const MachineFunction &MF, std::ostream &OS) const{
  int ValOffset = MF.getTarget().getFrameInfo()->getOffsetOfLocalArea();

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  <fi #" << (int)(i-NumFixedObjects) << ">: ";
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << "size is " << SO.Size << " byte" << (SO.Size != 1 ? "s," : ",");
    OS << " alignment is " << SO.Alignment << " byte"
       << (SO.Alignment != 1 ? "s," : ",");

    if (i < NumFixedObjects)
      OS << " fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int Off = SO.SPOffset - ValOffset;
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
//  MachineJumpTableInfo implementation
//===----------------------------------------------------------------------===//

/// getJumpTableIndex - Create a new jump table entry in the jump table info
/// or return an existing one.
///
unsigned MachineJumpTableInfo::getJumpTableIndex(
                                     std::vector<MachineBasicBlock*> &DestBBs) {
  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i)
    if (JumpTables[i].MBBs == DestBBs)
      return i;
  
  JumpTables.push_back(MachineJumpTableEntry(DestBBs));
  return JumpTables.size()-1;
}


void MachineJumpTableInfo::print(std::ostream &OS) const {
  // FIXME: this is lame, maybe we could print out the MBB numbers or something
  // like {1, 2, 4, 5, 3, 0}
  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i) {
    OS << "  <jt #" << i << "> has " << JumpTables[i].MBBs.size() 
       << " entries\n";
  }
}

unsigned MachineJumpTableInfo::getEntrySize() const { 
  return TD->getPointerSize(); 
}

unsigned MachineJumpTableInfo::getAlignment() const { 
  return TD->getPointerAlignment(); 
}

void MachineJumpTableInfo::dump() const { print(std::cerr); }


//===----------------------------------------------------------------------===//
//  MachineConstantPool implementation
//===----------------------------------------------------------------------===//

/// getConstantPoolIndex - Create a new entry in the constant pool or return
/// an existing one.  User must specify an alignment in bytes for the object.
///
unsigned MachineConstantPool::getConstantPoolIndex(Constant *C, 
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;
  
  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  unsigned AlignMask = (1 << Alignment)-1;
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (Constants[i].Val == C && (Constants[i].Offset & AlignMask) == 0)
      return i;
  
  unsigned Offset = 0;
  if (!Constants.empty()) {
    Offset = Constants.back().Offset;
    Offset += TD->getTypeSize(Constants.back().Val->getType());
    Offset = (Offset+AlignMask)&~AlignMask;
  }
  
  Constants.push_back(MachineConstantPoolEntry(C, Offset));
  return Constants.size()-1;
}


void MachineConstantPool::print(std::ostream &OS) const {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    OS << "  <cp #" << i << "> is" << *(Value*)Constants[i].Val;
    OS << " , offset=" << Constants[i].Offset;
    OS << "\n";
  }
}

void MachineConstantPool::dump() const { print(std::cerr); }
