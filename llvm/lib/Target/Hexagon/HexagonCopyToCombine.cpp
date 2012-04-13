//===------- HexagonCopyToCombine.cpp - Hexagon Copy-To-Combine Pass ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass replaces transfer instructions by combine instructions.
// We walk along a basic block and look for two combinable instructions and try
// to move them together. If we can move them next to each other we do so and
// replace them with a combine instruction.
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "hexagon-copy-combine"

#include "llvm/PassSupport.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "HexagonMachineFunctionInfo.h"

using namespace llvm;

static
cl::opt<bool> IsCombinesDisabled("disable-merge-into-combines",
                                 cl::Hidden, cl::ZeroOrMore,
                                 cl::init(false),
                                 cl::desc("Disable merging into combines"));

namespace {

class HexagonCopyToCombine : public MachineFunctionPass  {
  const HexagonInstrInfo *TII;
  const TargetRegisterInfo *TRI;
public:
  static char ID;

  HexagonCopyToCombine() : MachineFunctionPass(ID) { }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const char *getPassName() const {
    return "Hexagon Copy-To-Combine Pass";
  }

  virtual bool runOnMachineFunction(MachineFunction &Fn);

private:
  MachineInstr *findPairable(MachineInstr *I1, bool &DoInsertAtI1);

  void combine(MachineInstr *I1, MachineInstr *I2,
               MachineBasicBlock::iterator &MI, bool DoInsertAtI1);

  bool isSafeToMoveTogether(MachineInstr *I1, MachineInstr *I2,
                            unsigned I1DestReg, unsigned I2DestReg,
                            bool &DoInsertAtI1);

  void emitCombineRR(MachineBasicBlock::iterator &Before, unsigned DestReg,
                     MachineOperand &HiOperand, MachineOperand &LoOperand);

  void emitCombineRI(MachineBasicBlock::iterator &Before, unsigned DestReg,
                     MachineOperand &HiOperand, MachineOperand &LoOperand);

  void emitCombineIR(MachineBasicBlock::iterator &Before, unsigned DestReg,
                     MachineOperand &HiOperand, MachineOperand &LoOperand);

  void emitCombineII(MachineBasicBlock::iterator &Before, unsigned DestReg,
                     MachineOperand &HiOperand, MachineOperand &LoOperand);
};

} // End anonymous namespace.

char HexagonCopyToCombine::ID = 0;

static bool isCombinableInstType(MachineInstr *MI,
                                 const HexagonInstrInfo *TII) {
  switch(MI->getOpcode()) {
  case Hexagon::TFR: {
    // A COPY instruction can be combined if its arguments are IntRegs (32bit).
    assert(MI->getOperand(0).isReg() && MI->getOperand(1).isReg());

    unsigned DestReg = MI->getOperand(0).getReg();
    unsigned SrcReg = MI->getOperand(1).getReg();
    return Hexagon::IntRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg);
  }

  case Hexagon::TFRI: {
    // A transfer-immediate can be combined if its argument is a signed 8bit
    // value.
    assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
    unsigned DestReg = MI->getOperand(0).getReg();
    return Hexagon::IntRegsRegClass.contains(DestReg) &&
      isInt<8>(MI->getOperand(1).getImm());
  }
  default:
    break;
  }

  return false;
}


/// areCombinableOperations - Returns true if the two instruction can be merge
/// into a combine (ignoring register constraints).
static bool areCombinableOperations(const TargetRegisterInfo *TRI,
                                    MachineInstr *I1, MachineInstr *I2) {
  assert((I1->getOpcode() == Hexagon::TFR ||
          I1->getOpcode() == Hexagon::TFRI) &&
         (I2->getOpcode() == Hexagon::TFR ||
          I2->getOpcode() == Hexagon::TFRI) &&
         "Assume individual instructions are of a combinable type");

  const HexagonRegisterInfo *QRI =
    static_cast<const HexagonRegisterInfo *>(TRI);

  // V4 added some combine variations (mixed immediate and register source
  // operands), if we are on < V4 we can only combine 2 register-to-register
  // moves and 2 immediate-to-register moves.
  if (QRI->Subtarget.getHexagonArchVersion() < HexagonSubtarget::V4)
    return I1->getOpcode() == I2->getOpcode();

  return true;
}

static bool isEvenReg(unsigned Reg) {
  assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
         Hexagon::IntRegsRegClass.contains(Reg));
  return (Reg - Hexagon::R0) % 2 == 0;
}

static void removeKillInfo(MachineInstr *MI, unsigned RegNotKilled) {
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    MachineOperand &Op = MI->getOperand(I);
    if (!Op.isReg() || Op.getReg() != RegNotKilled || !Op.isKill())
      continue;
    Op.setIsKill(false);
  }
}

/// isUnsafeToMoveAccross - Returns true if it is unsafe to move a copy
/// instruction from \p UseReg to \p DestReg over the instruction \p I.
bool isUnsafeToMoveAccross(MachineInstr *I, unsigned UseReg, unsigned DestReg,
                            const TargetRegisterInfo *TRI) {
  return (UseReg && (I->modifiesRegister(UseReg, TRI))) ||
          I->modifiesRegister(DestReg, TRI) ||
          I->readsRegister(DestReg, TRI) ||
          I->hasUnmodeledSideEffects() ||
          I->isInlineAsm() || I->isDebugValue();
}

/// isSafeToMoveTogether - Returns true if it is safe to move I1 next to I2 such
/// that the two instructions can be paired in a combine.
bool HexagonCopyToCombine::isSafeToMoveTogether(MachineInstr *I1,
                                                MachineInstr *I2,
                                                unsigned I1DestReg,
                                                unsigned I2DestReg,
                                                bool &DoInsertAtI1) {
  bool isSafe = true;

  // First try to move I2 towards I1.
  {
    // A reverse_iterator instantiated like below starts before I2, and I1
    // respectively.
    // Look at instructions I in between I2 and (including) I1.
    MachineBasicBlock::reverse_iterator I(I2),
      End = MachineBasicBlock::reverse_iterator(I1);
    bool IsImmUseReg = I2->getOperand(1).isImm();
    unsigned I2UseReg = IsImmUseReg ? 0 : I2->getOperand(1).getReg();

    // If I2 kills its operand and we move I2 over an instruction that also
    // uses I2's use reg we need to modify that (first) instruction to now kill
    // this reg.
    unsigned KilledOperand = 0;
    if (I2->killsRegister(I2UseReg))
      KilledOperand = I2UseReg;
    MachineInstr *KillingInstr = 0;

    for (; I != End; ++I) {
      // If the intervening instruction I:
      //   * modifies I2's use reg
      //   * modifies I2's def reg
      //   * reads I2's def reg
      //   * or has unmodelled side effects
      // we can't move I2 across it.
      if (isUnsafeToMoveAccross(&*I, I2UseReg, I2DestReg, TRI)) {
        isSafe = false;
        break;
      }

      // Update first use of the killed operand.
      if (!KillingInstr && KilledOperand &&
          I->readsRegister(KilledOperand, TRI))
        KillingInstr = &*I;
    }
    if (isSafe) {
      // Update the intermediate instruction to with the kill flag.
      if (KillingInstr) {
        bool Added = KillingInstr->addRegisterKilled(KilledOperand, TRI, true);
        assert(Added && "Must successfully update kill flag");
        removeKillInfo(I2, KilledOperand);
      }
      DoInsertAtI1 = true;
      return true;
    }
  }

  // Try to move I1 towards I2.
  {
    // Look at instructions I in between I1 and (including) I2.
    MachineBasicBlock::iterator I(I1),
      End(next(MachineBasicBlock::iterator(I2)));
    bool IsImmUseReg = I1->getOperand(1).isImm();
    unsigned I1UseReg = IsImmUseReg ? 0 : I1->getOperand(1).getReg();
    // Track killed operands. If we move accross an instruction that kills our
    // operand, we need to update the kill information on the moved I1. It kills
    // the operand now.
    MachineInstr *KillingInstr = 0;
    unsigned KilledOperand = 0;

    while(++I != End) {
      // If the intervening instruction I:
      //   * modifies I1's use reg
      //   * modifies I1's def reg
      //   * reads I1's def reg
      //   * or has unmodelled side effects
      //   We introduce this special case because llvm has no api to remove a
      //   kill flag for a register (a removeRegisterKilled() analogous to
      //   addRegisterKilled) that handles aliased register correctly.
      //   * or has a killed aliased register use of I1's use reg
      //           %D4<def> = TFRI64 16
      //           %R6<def> = TFR %R9
      //           %R8<def> = KILL %R8, %D4<imp-use,kill>
      //      If we want to move R6 = across the KILL instruction we would have
      //      to remove the %D4<imp-use,kill> operand. For now, we are
      //      conservative and disallow the move.
      // we can't move I1 across it.
      if (isUnsafeToMoveAccross(I, I1UseReg, I1DestReg, TRI) ||
          // Check for an aliased register kill. Bail out if we see one.
          (!I->killsRegister(I1UseReg) && I->killsRegister(I1UseReg, TRI)))
        return false;

      // Check for an exact kill (registers match).
      if (I1UseReg && I->killsRegister(I1UseReg)) {
        assert(KillingInstr == 0 && "Should only see one killing instruction");
        KilledOperand = I1UseReg;
        KillingInstr = &*I;
      }
    }
    if (KillingInstr) {
      removeKillInfo(KillingInstr, KilledOperand);
      // Update I1 to set the kill flag. This flag will later be picked up by
      // the new COMBINE instruction.
      bool Added = I1->addRegisterKilled(KilledOperand, TRI);
      assert(Added && "Must successfully update kill flag");
    }
    DoInsertAtI1 = false;
  }

  return true;
}

bool HexagonCopyToCombine::runOnMachineFunction(MachineFunction &MF) {

  if (IsCombinesDisabled) return false;

  bool HasChanged = false;

  // Get target info.
  TRI = MF.getTarget().getRegisterInfo();
  TII = static_cast<const HexagonInstrInfo *>(MF.getTarget().getInstrInfo());

  // Traverse basic blocks.
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end(); BI != BE;
       ++BI) {
    // Traverse instructions in basic block.
    for(MachineBasicBlock::iterator MI = BI->begin(), End = BI->end();
        MI != End;) {
      MachineInstr *I1 = MI++;

      // Ignore instructions that are not combinable.
      if (!isCombinableInstType(I1, TII))
        continue;

      // Find a second instruction that can be merged into a combine
      // instruction.
      bool DoInsertAtI1 = false;
      MachineInstr *I2 = findPairable(I1, DoInsertAtI1);
      if (I2) {
        HasChanged = true;
        combine(I1, I2, MI, DoInsertAtI1);
      }
    }
  }

  return HasChanged;
}

/// findPairable - Returns an instruction that can be merged with \p I1 into a
/// COMBINE instruction or 0 if no such instruction can be found. Returns true
/// in \p DoInsertAtI1 if the combine must be inserted at instruction \p I1
/// false if the combine must be inserted at the returned instruction.
MachineInstr *HexagonCopyToCombine::findPairable(MachineInstr *I1,
                                                 bool &DoInsertAtI1) {
  MachineBasicBlock::iterator I2 = next(MachineBasicBlock::iterator(I1));
  unsigned I1DestReg = I1->getOperand(0).getReg();

  for (MachineBasicBlock::iterator End = I1->getParent()->end(); I2 != End;
       ++I2) {
    // Bail out early if we see a second definition of I1DestReg.
    if (I2->modifiesRegister(I1DestReg, TRI))
      break;

    // Ignore non-combinable instructions.
    if (!isCombinableInstType(I2, TII))
      continue;

    unsigned I2DestReg = I2->getOperand(0).getReg();

    // Check that registers are adjacent and that the first destination register
    // is even.
    bool IsI1BeforeI2 = (I2DestReg - I1DestReg) == 1;
    bool IsI2BeforeI1 = (I1DestReg - I2DestReg) == 1;
    unsigned FirstRegIndex = IsI1BeforeI2 ? I1DestReg : I2DestReg;
    if ((!IsI1BeforeI2 && !IsI2BeforeI1) || !isEvenReg(FirstRegIndex))
      continue;

    // Check that the two instructions are combinable. V4 allows more
    // instructions to be merged into a combine.
    if (!areCombinableOperations(TRI, I1, I2))
      break;

    if (isSafeToMoveTogether(I1, I2, I1DestReg, I2DestReg,
                             DoInsertAtI1))
      return I2;

    // Not safe. Stop searching.
    break;
  }
  return 0;
}

void HexagonCopyToCombine::combine(MachineInstr *I1, MachineInstr *I2,
                                   MachineBasicBlock::iterator &MI,
                                   bool DoInsertAtI1) {
  // We are going to delete I2. If MI points to I2 advance it to the next
  // instruction.
  if ((MachineInstr *)MI == I2) ++MI;

  // Figure out whether I1 or I2 goes into the lowreg part.
  unsigned I1DestReg = I1->getOperand(0).getReg();
  unsigned I2DestReg = I2->getOperand(0).getReg();
  bool IsI1Loreg = (I2DestReg - I1DestReg) == 1;
  unsigned LoRegDef = IsI1Loreg ? I1DestReg : I2DestReg;

  // Get the double word register.
  unsigned DoubleRegDest =
    TRI->getMatchingSuperReg(LoRegDef, Hexagon::subreg_loreg,
                             Hexagon::DoubleRegsRegisterClass);
  assert(DoubleRegDest != 0 && "Expect a valid register");


  // Setup source operands.
  MachineOperand &LoOperand = IsI1Loreg ? I1->getOperand(1) :
    I2->getOperand(1);
  MachineOperand &HiOperand = IsI1Loreg ? I2->getOperand(1) :
    I1->getOperand(1);

  // Figure out which source is a register and which a constant.
  bool IsHiReg = HiOperand.isReg();
  bool IsLoReg = LoOperand.isReg();

  MachineBasicBlock::iterator InsertPt(DoInsertAtI1 ? I1 : I2);
  // Emit combine.
  if (IsHiReg && IsLoReg)
    emitCombineRR(InsertPt, DoubleRegDest, HiOperand, LoOperand);
  else if (IsHiReg)
    emitCombineRI(InsertPt, DoubleRegDest, HiOperand, LoOperand);
  else if (IsLoReg)
    emitCombineIR(InsertPt, DoubleRegDest, HiOperand, LoOperand);
  else
    emitCombineII(InsertPt, DoubleRegDest, HiOperand, LoOperand);

  I1->eraseFromParent();
  I2->eraseFromParent();
}

void HexagonCopyToCombine::emitCombineII(MachineBasicBlock::iterator &InsertPt,
                                         unsigned DoubleDestReg,
                                         MachineOperand &HiOperand,
                                         MachineOperand &LoOperand) {
  DebugLoc DL = InsertPt->getDebugLoc();
  MachineBasicBlock *BB = InsertPt->getParent();

  // Insert new combine instruction.
  //  DoubleRegDest = combine #HiImm, #LoImm
  BuildMI(*BB, InsertPt, DL, TII->get(Hexagon::COMBINE_ii), DoubleDestReg)
    .addImm(HiOperand.getImm())
    .addImm(LoOperand.getImm());
}

void HexagonCopyToCombine::emitCombineIR(MachineBasicBlock::iterator &InsertPt,
                                         unsigned DoubleDestReg,
                                         MachineOperand &HiOperand,
                                         MachineOperand &LoOperand) {
  unsigned LoReg = LoOperand.getReg();
  unsigned LoRegKillFlag = getKillRegState(LoOperand.isKill());

  DebugLoc DL = InsertPt->getDebugLoc();
  MachineBasicBlock *BB = InsertPt->getParent();

  // Insert new combine instruction.
  //  DoubleRegDest = combine #HiImm, LoReg
  BuildMI(*BB, InsertPt, DL, TII->get(Hexagon::COMBINE_ir_V4), DoubleDestReg)
    .addImm(HiOperand.getImm())
    .addReg(LoReg, LoRegKillFlag);
}

void HexagonCopyToCombine::emitCombineRI(MachineBasicBlock::iterator &InsertPt,
                                         unsigned DoubleDestReg,
                                         MachineOperand &HiOperand,
                                         MachineOperand &LoOperand) {
  unsigned HiRegKillFlag = getKillRegState(HiOperand.isKill());
  unsigned HiReg = HiOperand.getReg();

  DebugLoc DL = InsertPt->getDebugLoc();
  MachineBasicBlock *BB = InsertPt->getParent();

  // Insert new combine instruction.
  //  DoubleRegDest = combine HiReg, #LoImm
  BuildMI(*BB, InsertPt, DL, TII->get(Hexagon::COMBINE_ri_V4), DoubleDestReg)
    .addReg(HiReg, HiRegKillFlag)
    .addImm(LoOperand.getImm());
}

void HexagonCopyToCombine::emitCombineRR(MachineBasicBlock::iterator &InsertPt,
                                         unsigned DoubleDestReg,
                                         MachineOperand &HiOperand,
                                         MachineOperand &LoOperand) {
  unsigned LoRegKillFlag = getKillRegState(LoOperand.isKill());
  unsigned HiRegKillFlag = getKillRegState(HiOperand.isKill());
  unsigned LoReg = LoOperand.getReg();
  unsigned HiReg = HiOperand.getReg();

  DebugLoc DL = InsertPt->getDebugLoc();
  MachineBasicBlock *BB = InsertPt->getParent();

  // Insert new combine instruction.
  //  DoubleRegDest = combine HiReg, LoReg
  BuildMI(*BB, InsertPt, DL, TII->get(Hexagon::COMBINE_rr), DoubleDestReg)
    .addReg(HiReg, HiRegKillFlag)
    .addReg(LoReg, LoRegKillFlag);
}

FunctionPass *llvm::createHexagonCopyToCombine() {
  return new HexagonCopyToCombine();
}

