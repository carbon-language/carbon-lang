//===-- MachineFunction.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect native machine code information for a function.  This allows
// target-specific information about the generated code to be stored with each
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  struct Printer : public MachineFunctionPass {
    static char ID;

    raw_ostream &OS;
    const std::string Banner;

    Printer(raw_ostream &os, const std::string &banner) 
      : MachineFunctionPass(&ID), OS(os), Banner(banner) {}

    const char *getPassName() const { return "MachineFunction Printer"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) {
      OS << Banner;
      MF.print(OS);
      return false;
    }
  };
  char Printer::ID = 0;
}

/// Returns a newly-created MachineFunction Printer pass. The default banner is
/// empty.
///
FunctionPass *llvm::createMachineFunctionPrinterPass(raw_ostream &OS,
                                                     const std::string &Banner){
  return new Printer(OS, Banner);
}

//===---------------------------------------------------------------------===//
// MachineFunction implementation
//===---------------------------------------------------------------------===//

// Out of line virtual method.
MachineFunctionInfo::~MachineFunctionInfo() {}

void ilist_traits<MachineBasicBlock>::deleteNode(MachineBasicBlock *MBB) {
  MBB->getParent()->DeleteMachineBasicBlock(MBB);
}

MachineFunction::MachineFunction(Function *F,
                                 const TargetMachine &TM)
  : Fn(F), Target(TM) {
  if (TM.getRegisterInfo())
    RegInfo = new (Allocator.Allocate<MachineRegisterInfo>())
                  MachineRegisterInfo(*TM.getRegisterInfo());
  else
    RegInfo = 0;
  MFInfo = 0;
  FrameInfo = new (Allocator.Allocate<MachineFrameInfo>())
                  MachineFrameInfo(*TM.getFrameInfo());
  ConstantPool = new (Allocator.Allocate<MachineConstantPool>())
                     MachineConstantPool(TM.getTargetData());
  Alignment = TM.getTargetLowering()->getFunctionAlignment(F);

  // Set up jump table.
  const TargetData &TD = *TM.getTargetData();
  bool IsPic = TM.getRelocationModel() == Reloc::PIC_;
  unsigned EntrySize = IsPic ? 4 : TD.getPointerSize();
  unsigned TyAlignment = IsPic ?
                       TD.getABITypeAlignment(Type::getInt32Ty(F->getContext()))
                               : TD.getPointerABIAlignment();
  JumpTableInfo = new (Allocator.Allocate<MachineJumpTableInfo>())
                      MachineJumpTableInfo(EntrySize, TyAlignment);
}

MachineFunction::~MachineFunction() {
  BasicBlocks.clear();
  InstructionRecycler.clear(Allocator);
  BasicBlockRecycler.clear(Allocator);
  if (RegInfo) {
    RegInfo->~MachineRegisterInfo();
    Allocator.Deallocate(RegInfo);
  }
  if (MFInfo) {
    MFInfo->~MachineFunctionInfo();
    Allocator.Deallocate(MFInfo);
  }
  FrameInfo->~MachineFrameInfo();         Allocator.Deallocate(FrameInfo);
  ConstantPool->~MachineConstantPool();   Allocator.Deallocate(ConstantPool);
  JumpTableInfo->~MachineJumpTableInfo(); Allocator.Deallocate(JumpTableInfo);
}


/// RenumberBlocks - This discards all of the MachineBasicBlock numbers and
/// recomputes them.  This guarantees that the MBB numbers are sequential,
/// dense, and match the ordering of the blocks within the function.  If a
/// specific MachineBasicBlock is specified, only that block and those after
/// it are renumbered.
void MachineFunction::RenumberBlocks(MachineBasicBlock *MBB) {
  if (empty()) { MBBNumbering.clear(); return; }
  MachineFunction::iterator MBBI, E = end();
  if (MBB == 0)
    MBBI = begin();
  else
    MBBI = MBB;
  
  // Figure out the block number this should have.
  unsigned BlockNo = 0;
  if (MBBI != begin())
    BlockNo = prior(MBBI)->getNumber()+1;
  
  for (; MBBI != E; ++MBBI, ++BlockNo) {
    if (MBBI->getNumber() != (int)BlockNo) {
      // Remove use of the old number.
      if (MBBI->getNumber() != -1) {
        assert(MBBNumbering[MBBI->getNumber()] == &*MBBI &&
               "MBB number mismatch!");
        MBBNumbering[MBBI->getNumber()] = 0;
      }
      
      // If BlockNo is already taken, set that block's number to -1.
      if (MBBNumbering[BlockNo])
        MBBNumbering[BlockNo]->setNumber(-1);

      MBBNumbering[BlockNo] = MBBI;
      MBBI->setNumber(BlockNo);
    }
  }    

  // Okay, all the blocks are renumbered.  If we have compactified the block
  // numbering, shrink MBBNumbering now.
  assert(BlockNo <= MBBNumbering.size() && "Mismatch!");
  MBBNumbering.resize(BlockNo);
}

/// CreateMachineInstr - Allocate a new MachineInstr. Use this instead
/// of `new MachineInstr'.
///
MachineInstr *
MachineFunction::CreateMachineInstr(const TargetInstrDesc &TID,
                                    DebugLoc DL, bool NoImp) {
  return new (InstructionRecycler.Allocate<MachineInstr>(Allocator))
    MachineInstr(TID, DL, NoImp);
}

/// CloneMachineInstr - Create a new MachineInstr which is a copy of the
/// 'Orig' instruction, identical in all ways except the the instruction
/// has no parent, prev, or next.
///
MachineInstr *
MachineFunction::CloneMachineInstr(const MachineInstr *Orig) {
  return new (InstructionRecycler.Allocate<MachineInstr>(Allocator))
             MachineInstr(*this, *Orig);
}

/// DeleteMachineInstr - Delete the given MachineInstr.
///
void
MachineFunction::DeleteMachineInstr(MachineInstr *MI) {
  MI->~MachineInstr();
  InstructionRecycler.Deallocate(Allocator, MI);
}

/// CreateMachineBasicBlock - Allocate a new MachineBasicBlock. Use this
/// instead of `new MachineBasicBlock'.
///
MachineBasicBlock *
MachineFunction::CreateMachineBasicBlock(const BasicBlock *bb) {
  return new (BasicBlockRecycler.Allocate<MachineBasicBlock>(Allocator))
             MachineBasicBlock(*this, bb);
}

/// DeleteMachineBasicBlock - Delete the given MachineBasicBlock.
///
void
MachineFunction::DeleteMachineBasicBlock(MachineBasicBlock *MBB) {
  assert(MBB->getParent() == this && "MBB parent mismatch!");
  MBB->~MachineBasicBlock();
  BasicBlockRecycler.Deallocate(Allocator, MBB);
}

MachineMemOperand *
MachineFunction::getMachineMemOperand(const Value *v, unsigned f,
                                      int64_t o, uint64_t s,
                                      unsigned base_alignment) {
  return new (Allocator.Allocate<MachineMemOperand>())
             MachineMemOperand(v, f, o, s, base_alignment);
}

MachineMemOperand *
MachineFunction::getMachineMemOperand(const MachineMemOperand *MMO,
                                      int64_t Offset, uint64_t Size) {
  return new (Allocator.Allocate<MachineMemOperand>())
             MachineMemOperand(MMO->getValue(), MMO->getFlags(),
                               int64_t(uint64_t(MMO->getOffset()) +
                                       uint64_t(Offset)),
                               Size, MMO->getBaseAlignment());
}

MachineInstr::mmo_iterator
MachineFunction::allocateMemRefsArray(unsigned long Num) {
  return Allocator.Allocate<MachineMemOperand *>(Num);
}

std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator>
MachineFunction::extractLoadMemRefs(MachineInstr::mmo_iterator Begin,
                                    MachineInstr::mmo_iterator End) {
  // Count the number of load mem refs.
  unsigned Num = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I)
    if ((*I)->isLoad())
      ++Num;

  // Allocate a new array and populate it with the load information.
  MachineInstr::mmo_iterator Result = allocateMemRefsArray(Num);
  unsigned Index = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I) {
    if ((*I)->isLoad()) {
      if (!(*I)->isStore())
        // Reuse the MMO.
        Result[Index] = *I;
      else {
        // Clone the MMO and unset the store flag.
        MachineMemOperand *JustLoad =
          getMachineMemOperand((*I)->getValue(),
                               (*I)->getFlags() & ~MachineMemOperand::MOStore,
                               (*I)->getOffset(), (*I)->getSize(),
                               (*I)->getBaseAlignment());
        Result[Index] = JustLoad;
      }
      ++Index;
    }
  }
  return std::make_pair(Result, Result + Num);
}

std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator>
MachineFunction::extractStoreMemRefs(MachineInstr::mmo_iterator Begin,
                                     MachineInstr::mmo_iterator End) {
  // Count the number of load mem refs.
  unsigned Num = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I)
    if ((*I)->isStore())
      ++Num;

  // Allocate a new array and populate it with the store information.
  MachineInstr::mmo_iterator Result = allocateMemRefsArray(Num);
  unsigned Index = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I) {
    if ((*I)->isStore()) {
      if (!(*I)->isLoad())
        // Reuse the MMO.
        Result[Index] = *I;
      else {
        // Clone the MMO and unset the load flag.
        MachineMemOperand *JustStore =
          getMachineMemOperand((*I)->getValue(),
                               (*I)->getFlags() & ~MachineMemOperand::MOLoad,
                               (*I)->getOffset(), (*I)->getSize(),
                               (*I)->getBaseAlignment());
        Result[Index] = JustStore;
      }
      ++Index;
    }
  }
  return std::make_pair(Result, Result + Num);
}

void MachineFunction::dump() const {
  print(errs());
}

void MachineFunction::print(raw_ostream &OS) const {
  OS << "# Machine code for " << Fn->getName() << "():\n";

  // Print Frame Information
  FrameInfo->print(*this, OS);
  
  // Print JumpTable Information
  JumpTableInfo->print(OS);

  // Print Constant Pool
  ConstantPool->print(OS);
  
  const TargetRegisterInfo *TRI = getTarget().getRegisterInfo();
  
  if (RegInfo && !RegInfo->livein_empty()) {
    OS << "Live Ins:";
    for (MachineRegisterInfo::livein_iterator
         I = RegInfo->livein_begin(), E = RegInfo->livein_end(); I != E; ++I) {
      if (TRI)
        OS << " " << TRI->getName(I->first);
      else
        OS << " Reg #" << I->first;
      
      if (I->second)
        OS << " in VR#" << I->second << ' ';
    }
    OS << '\n';
  }
  if (RegInfo && !RegInfo->liveout_empty()) {
    OS << "Live Outs:";
    for (MachineRegisterInfo::liveout_iterator
         I = RegInfo->liveout_begin(), E = RegInfo->liveout_end(); I != E; ++I)
      if (TRI)
        OS << ' ' << TRI->getName(*I);
      else
        OS << " Reg #" << *I;
    OS << '\n';
  }
  
  for (const_iterator BB = begin(), E = end(); BB != E; ++BB)
    BB->print(OS);

  OS << "\n# End machine code for " << Fn->getName() << "().\n\n";
}

namespace llvm {
  template<>
  struct DOTGraphTraits<const MachineFunction*> : public DefaultDOTGraphTraits {
    static std::string getGraphName(const MachineFunction *F) {
      return "CFG for '" + F->getFunction()->getNameStr() + "' function";
    }

    static std::string getNodeLabel(const MachineBasicBlock *Node,
                                    const MachineFunction *Graph,
                                    bool ShortNames) {
      if (ShortNames && Node->getBasicBlock() &&
          !Node->getBasicBlock()->getName().empty())
        return Node->getBasicBlock()->getNameStr() + ":";

      std::string OutStr;
      {
        raw_string_ostream OSS(OutStr);
        
        if (ShortNames)
          OSS << Node->getNumber() << ':';
        else
          Node->print(OSS);
      }

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
  ViewGraph(this, "mf" + getFunction()->getNameStr());
#else
  errs() << "SelectionDAG::viewGraph is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

void MachineFunction::viewCFGOnly() const
{
#ifndef NDEBUG
  ViewGraph(this, "mf" + getFunction()->getNameStr(), true);
#else
  errs() << "SelectionDAG::viewGraph is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

/// addLiveIn - Add the specified physical register as a live-in value and
/// create a corresponding virtual register for it.
unsigned MachineFunction::addLiveIn(unsigned PReg,
                                    const TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = getRegInfo().createVirtualRegister(RC);
  getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

/// getDebugLocTuple - Get the DebugLocTuple for a given DebugLoc object.
DebugLocTuple MachineFunction::getDebugLocTuple(DebugLoc DL) const {
  unsigned Idx = DL.getIndex();
  assert(Idx < DebugLocInfo.DebugLocations.size() &&
         "Invalid index into debug locations!");
  return DebugLocInfo.DebugLocations[Idx];
}

//===----------------------------------------------------------------------===//
//  MachineFrameInfo implementation
//===----------------------------------------------------------------------===//

/// CreateFixedObject - Create a new object at a fixed location on the stack.
/// All fixed objects should be created before other objects are created for
/// efficiency. By default, fixed objects are immutable. This returns an
/// index with a negative value.
///
int MachineFrameInfo::CreateFixedObject(uint64_t Size, int64_t SPOffset,
                                        bool Immutable) {
  assert(Size != 0 && "Cannot allocate zero size fixed stack objects!");
  Objects.insert(Objects.begin(), StackObject(Size, 1, SPOffset, Immutable));
  return -++NumFixedObjects;
}


BitVector
MachineFrameInfo::getPristineRegs(const MachineBasicBlock *MBB) const {
  assert(MBB && "MBB must be valid");
  const MachineFunction *MF = MBB->getParent();
  assert(MF && "MBB must be part of a MachineFunction");
  const TargetMachine &TM = MF->getTarget();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  BitVector BV(TRI->getNumRegs());

  // Before CSI is calculated, no registers are considered pristine. They can be
  // freely used and PEI will make sure they are saved.
  if (!isCalleeSavedInfoValid())
    return BV;

  for (const unsigned *CSR = TRI->getCalleeSavedRegs(MF); CSR && *CSR; ++CSR)
    BV.set(*CSR);

  // The entry MBB always has all CSRs pristine.
  if (MBB == &MF->front())
    return BV;

  // On other MBBs the saved CSRs are not pristine.
  const std::vector<CalleeSavedInfo> &CSI = getCalleeSavedInfo();
  for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
         E = CSI.end(); I != E; ++I)
    BV.reset(I->getReg());

  return BV;
}


void MachineFrameInfo::print(const MachineFunction &MF, raw_ostream &OS) const{
  const TargetFrameInfo *FI = MF.getTarget().getFrameInfo();
  int ValOffset = (FI ? FI->getOffsetOfLocalArea() : 0);

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  <fi#" << (int)(i-NumFixedObjects) << ">: ";
    if (SO.Size == ~0ULL) {
      OS << "dead\n";
      continue;
    }
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << "size is " << SO.Size << " byte" << (SO.Size != 1 ? "s," : ",");
    OS << " alignment is " << SO.Alignment << " byte"
       << (SO.Alignment != 1 ? "s," : ",");

    if (i < NumFixedObjects)
      OS << " fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int64_t Off = SO.SPOffset - ValOffset;
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
  print(MF, errs());
}

//===----------------------------------------------------------------------===//
//  MachineJumpTableInfo implementation
//===----------------------------------------------------------------------===//

/// getJumpTableIndex - Create a new jump table entry in the jump table info
/// or return an existing one.
///
unsigned MachineJumpTableInfo::getJumpTableIndex(
                               const std::vector<MachineBasicBlock*> &DestBBs) {
  assert(!DestBBs.empty() && "Cannot create an empty jump table!");
  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i)
    if (JumpTables[i].MBBs == DestBBs)
      return i;
  
  JumpTables.push_back(MachineJumpTableEntry(DestBBs));
  return JumpTables.size()-1;
}

/// ReplaceMBBInJumpTables - If Old is the target of any jump tables, update
/// the jump tables to branch to New instead.
bool
MachineJumpTableInfo::ReplaceMBBInJumpTables(MachineBasicBlock *Old,
                                             MachineBasicBlock *New) {
  assert(Old != New && "Not making a change?");
  bool MadeChange = false;
  for (size_t i = 0, e = JumpTables.size(); i != e; ++i) {
    MachineJumpTableEntry &JTE = JumpTables[i];
    for (size_t j = 0, e = JTE.MBBs.size(); j != e; ++j)
      if (JTE.MBBs[j] == Old) {
        JTE.MBBs[j] = New;
        MadeChange = true;
      }
  }
  return MadeChange;
}

void MachineJumpTableInfo::print(raw_ostream &OS) const {
  // FIXME: this is lame, maybe we could print out the MBB numbers or something
  // like {1, 2, 4, 5, 3, 0}
  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i) {
    OS << "  <jt#" << i << "> has " << JumpTables[i].MBBs.size() 
       << " entries\n";
  }
}

void MachineJumpTableInfo::dump() const { print(errs()); }


//===----------------------------------------------------------------------===//
//  MachineConstantPool implementation
//===----------------------------------------------------------------------===//

const Type *MachineConstantPoolEntry::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}


unsigned MachineConstantPoolEntry::getRelocationInfo() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getRelocationInfo();
  return Val.ConstVal->getRelocationInfo();
}

MachineConstantPool::~MachineConstantPool() {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (Constants[i].isMachineConstantPoolEntry())
      delete Constants[i].Val.MachineCPVal;
}

/// CanShareConstantPoolEntry - Test whether the given two constants
/// can be allocated the same constant pool entry.
static bool CanShareConstantPoolEntry(Constant *A, Constant *B,
                                      const TargetData *TD) {
  // Handle the trivial case quickly.
  if (A == B) return true;

  // If they have the same type but weren't the same constant, quickly
  // reject them.
  if (A->getType() == B->getType()) return false;

  // For now, only support constants with the same size.
  if (TD->getTypeStoreSize(A->getType()) != TD->getTypeStoreSize(B->getType()))
    return false;

  // If a floating-point value and an integer value have the same encoding,
  // they can share a constant-pool entry.
  if (ConstantFP *AFP = dyn_cast<ConstantFP>(A))
    if (ConstantInt *BI = dyn_cast<ConstantInt>(B))
      return AFP->getValueAPF().bitcastToAPInt() == BI->getValue();
  if (ConstantFP *BFP = dyn_cast<ConstantFP>(B))
    if (ConstantInt *AI = dyn_cast<ConstantInt>(A))
      return BFP->getValueAPF().bitcastToAPInt() == AI->getValue();

  // Two vectors can share an entry if each pair of corresponding
  // elements could.
  if (ConstantVector *AV = dyn_cast<ConstantVector>(A))
    if (ConstantVector *BV = dyn_cast<ConstantVector>(B)) {
      if (AV->getType()->getNumElements() != BV->getType()->getNumElements())
        return false;
      for (unsigned i = 0, e = AV->getType()->getNumElements(); i != e; ++i)
        if (!CanShareConstantPoolEntry(AV->getOperand(i), BV->getOperand(i), TD))
          return false;
      return true;
    }

  // TODO: Handle other cases.

  return false;
}

/// getConstantPoolIndex - Create a new entry in the constant pool or return
/// an existing one.  User must specify the log2 of the minimum required
/// alignment for the object.
///
unsigned MachineConstantPool::getConstantPoolIndex(Constant *C, 
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;

  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (!Constants[i].isMachineConstantPoolEntry() &&
        CanShareConstantPoolEntry(Constants[i].Val.ConstVal, C, TD)) {
      if ((unsigned)Constants[i].getAlignment() < Alignment)
        Constants[i].Alignment = Alignment;
      return i;
    }
  
  Constants.push_back(MachineConstantPoolEntry(C, Alignment));
  return Constants.size()-1;
}

unsigned MachineConstantPool::getConstantPoolIndex(MachineConstantPoolValue *V,
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;
  
  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  int Idx = V->getExistingMachineCPValue(this, Alignment);
  if (Idx != -1)
    return (unsigned)Idx;

  Constants.push_back(MachineConstantPoolEntry(V, Alignment));
  return Constants.size()-1;
}

void MachineConstantPool::print(raw_ostream &OS) const {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    OS << "  <cp#" << i << "> is";
    if (Constants[i].isMachineConstantPoolEntry())
      Constants[i].Val.MachineCPVal->print(OS);
    else
      OS << *(Value*)Constants[i].Val.ConstVal;
    OS << " , alignment=" << Constants[i].getAlignment();
    OS << "\n";
  }
}

void MachineConstantPool::dump() const { print(errs()); }
