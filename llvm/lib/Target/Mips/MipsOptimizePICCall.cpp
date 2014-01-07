//===--------- MipsOptimizePICCall.cpp - Optimize PIC Calls ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates unnecessary instructions that set up $gp and replace
// instructions that load target function addresses with copy instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "optimize-mips-pic-call"

#include "Mips.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MipsMachineFunction.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool> LoadTargetFromGOT("mips-load-target-from-got",
                                       cl::init(true),
                                       cl::desc("Load target address from GOT"),
                                       cl::Hidden);

static cl::opt<bool> EraseGPOpnd("mips-erase-gp-opnd",
                                 cl::init(true), cl::desc("Erase GP Operand"),
                                 cl::Hidden);

namespace {
typedef std::pair<unsigned, unsigned> CntRegP;
typedef RecyclingAllocator<BumpPtrAllocator,
                           ScopedHashTableVal<const Value *, CntRegP> >
AllocatorTy;
typedef ScopedHashTable<const Value *, CntRegP, DenseMapInfo<const Value *>,
                        AllocatorTy> ScopedHTType;

class MBBInfo {
public:
  MBBInfo(MachineDomTreeNode *N);
  const MachineDomTreeNode *getNode() const;
  bool isVisited() const;
  void preVisit(ScopedHTType &ScopedHT);
  void postVisit();

private:
  MachineDomTreeNode *Node;
  ScopedHTType::ScopeTy *HTScope;
};

class OptimizePICCall : public MachineFunctionPass {
public:
  OptimizePICCall(TargetMachine &tm) : MachineFunctionPass(ID) {}

  virtual const char *getPassName() const { return "Mips OptimizePICCall"; }

  bool runOnMachineFunction(MachineFunction &F);

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  /// \brief Visit MBB.
  bool visitNode(MBBInfo &MBBI);

  /// \brief Test if MI jumps to a function via a register.
  ///
  /// Also, return the virtual register containing the target function's address
  /// and the underlying object in Reg and Val respectively, if the function's
  /// address can be resolved lazily.
  bool isCallViaRegister(MachineInstr &MI, unsigned &Reg,
                         const Value *&Val) const;

  /// \brief Return the number of instructions that dominate the current
  /// instruction and load the function address from object Entry.
  unsigned getCount(const Value *Entry);

  /// \brief Return the destination virtual register of the last instruction
  /// that loads from object Entry.
  unsigned getReg(const Value *Entry);

  /// \brief Update ScopedHT.
  void incCntAndSetReg(const Value *Entry, unsigned Reg);

  ScopedHTType ScopedHT;
  static char ID;
};

char OptimizePICCall::ID = 0;
} // end of anonymous namespace

/// Return the first MachineOperand of MI if it is a used virtual register.
static MachineOperand *getCallTargetRegOpnd(MachineInstr &MI) {
  if (MI.getNumOperands() == 0)
    return 0;

  MachineOperand &MO = MI.getOperand(0);

  if (!MO.isReg() || !MO.isUse() ||
      !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    return 0;

  return &MO;
}

/// Return type of register Reg.
static MVT::SimpleValueType getRegTy(unsigned Reg, MachineFunction &MF) {
  const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(Reg);
  assert(RC->vt_end() - RC->vt_begin() == 1);
  return *RC->vt_begin();
}

/// Do the following transformation:
///
/// jalr $vreg
/// =>
/// copy $t9, $vreg
/// jalr $t9
static void setCallTargetReg(MachineBasicBlock *MBB,
                             MachineBasicBlock::iterator I) {
  MachineFunction &MF = *MBB->getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  unsigned SrcReg = I->getOperand(0).getReg();
  unsigned DstReg = getRegTy(SrcReg, MF) == MVT::i32 ? Mips::T9 : Mips::T9_64;
  BuildMI(*MBB, I, I->getDebugLoc(), TII.get(TargetOpcode::COPY), DstReg)
      .addReg(SrcReg);
  I->getOperand(0).setReg(DstReg);
}

/// Search MI's operands for register GP and erase it.
static void eraseGPOpnd(MachineInstr &MI) {
  if (!EraseGPOpnd)
    return;

  MachineFunction &MF = *MI.getParent()->getParent();
  MVT::SimpleValueType Ty = getRegTy(MI.getOperand(0).getReg(), MF);
  unsigned Reg = Ty == MVT::i32 ? Mips::GP : Mips::GP_64;

  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.getReg() == Reg) {
      MI.RemoveOperand(I);
      return;
    }
  }

  llvm_unreachable(0);
}

MBBInfo::MBBInfo(MachineDomTreeNode *N) : Node(N), HTScope(0) {}

const MachineDomTreeNode *MBBInfo::getNode() const { return Node; }

bool MBBInfo::isVisited() const { return HTScope; }

void MBBInfo::preVisit(ScopedHTType &ScopedHT) {
  HTScope = new ScopedHTType::ScopeTy(ScopedHT);
}

void MBBInfo::postVisit() {
  delete HTScope;
}

// OptimizePICCall methods.
bool OptimizePICCall::runOnMachineFunction(MachineFunction &F) {
  if (F.getTarget().getSubtarget<MipsSubtarget>().inMips16Mode())
    return false;

  // Do a pre-order traversal of the dominator tree.
  MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();
  bool Changed = false;

  SmallVector<MBBInfo, 8> WorkList(1, MBBInfo(MDT->getRootNode()));

  while (!WorkList.empty()) {
    MBBInfo &MBBI = WorkList.back();

    // If this MBB has already been visited, destroy the scope for the MBB and
    // pop it from the work list.
    if (MBBI.isVisited()) {
      MBBI.postVisit();
      WorkList.pop_back();
      continue;
    }

    // Visit the MBB and add its children to the work list.
    MBBI.preVisit(ScopedHT);
    Changed |= visitNode(MBBI);
    const MachineDomTreeNode *Node = MBBI.getNode();
    const std::vector<MachineDomTreeNode *> &Children = Node->getChildren();
    WorkList.append(Children.begin(), Children.end());
  }

  return Changed;
}

bool OptimizePICCall::visitNode(MBBInfo &MBBI) {
  bool Changed = false;
  MachineBasicBlock *MBB = MBBI.getNode()->getBlock();

  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I) {
    unsigned Reg;
    const Value *Entry;

    // Skip instructions that are not call instructions via registers.
    if (!isCallViaRegister(*I, Reg, Entry))
      continue;

    Changed = true;
    unsigned N = getCount(Entry);

    if (N != 0) {
      // If a function has been called more than twice, we do not have to emit a
      // load instruction to get the function address from the GOT, but can
      // instead reuse the address that has been loaded before.
      if (N >= 2 && !LoadTargetFromGOT)
        getCallTargetRegOpnd(*I)->setReg(getReg(Entry));

      // Erase the $gp operand if this isn't the first time a function has
      // been called. $gp needs to be set up only if the function call can go
      // through a lazy binding stub.
      eraseGPOpnd(*I);
    }

    if (Entry)
      incCntAndSetReg(Entry, Reg);

    setCallTargetReg(MBB, I);
  }

  return Changed;
}

bool OptimizePICCall::isCallViaRegister(MachineInstr &MI, unsigned &Reg,
                                        const Value *&Val) const {
  if (!MI.isCall())
    return false;

  MachineOperand *MO = getCallTargetRegOpnd(MI);

  // Return if MI is not a function call via a register.
  if (!MO)
    return false;

  // Get the instruction that loads the function address from the GOT.
  Reg = MO->getReg();
  Val = 0;
  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  MachineInstr *DefMI = MRI.getVRegDef(Reg);

  assert(DefMI);

  // See if DefMI is an instruction that loads from a GOT entry that holds the
  // address of a lazy binding stub.
  if (!DefMI->mayLoad() || DefMI->getNumOperands() < 3)
    return true;

  unsigned Flags = DefMI->getOperand(2).getTargetFlags();

  if (Flags != MipsII::MO_GOT_CALL && Flags != MipsII::MO_CALL_LO16)
    return true;

  // Return the underlying object for the GOT entry in Val.
  assert(DefMI->hasOneMemOperand());
  Val = (*DefMI->memoperands_begin())->getValue();
  return true;
}

unsigned OptimizePICCall::getCount(const Value *Entry) {
  return ScopedHT.lookup(Entry).first;
}

unsigned OptimizePICCall::getReg(const Value *Entry) {
  unsigned Reg = ScopedHT.lookup(Entry).second;
  assert(Reg);
  return Reg;
}

void OptimizePICCall::incCntAndSetReg(const Value *Entry, unsigned Reg) {
  CntRegP P = ScopedHT.lookup(Entry);
  ScopedHT.insert(Entry, std::make_pair(P.first + 1, Reg));
}

/// Return an OptimizeCall object.
FunctionPass *llvm::createMipsOptimizePICCallPass(MipsTargetMachine &TM) {
  return new OptimizePICCall(TM);
}
