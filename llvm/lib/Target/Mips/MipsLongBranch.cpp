//===-- MipsLongBranch.cpp - Emit long branches ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass expands a branch or jump instruction into a long branch if its
// offset is too large to fit into its immediate field.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-long-branch"

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

STATISTIC(LongBranches, "Number of long branches.");

static cl::opt<bool> SkipLongBranch(
  "skip-mips-long-branch",
  cl::init(false),
  cl::desc("MIPS: Skip long branch pass."),
  cl::Hidden);

static cl::opt<bool> ForceLongBranch(
  "force-mips-long-branch",
  cl::init(false),
  cl::desc("MIPS: Expand all branches to long format."),
  cl::Hidden);

namespace {
  typedef MachineBasicBlock::iterator Iter;
  typedef MachineBasicBlock::reverse_iterator ReverseIter;

  struct MBBInfo {
    uint64_t Size;
    bool HasLongBranch;
    MachineInstr *Br;

    MBBInfo() : Size(0), HasLongBranch(false), Br(0) {}
  };

  class MipsLongBranch : public MachineFunctionPass {

  public:
    static char ID;
    MipsLongBranch(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm),
        TII(static_cast<const MipsInstrInfo*>(tm.getInstrInfo())) {}

    virtual const char *getPassName() const {
      return "Mips Long Branch";
    }

    bool runOnMachineFunction(MachineFunction &F);

  private:
    void splitMBB(MachineBasicBlock *MBB);
    void initMBBInfo();
    int64_t computeOffset(const MachineInstr *Br);
    void replaceBranch(MachineBasicBlock &MBB, Iter Br, DebugLoc DL,
                       MachineBasicBlock *MBBOpnd);
    void expandToLongBranch(MBBInfo &Info);

    const TargetMachine &TM;
    const MipsInstrInfo *TII;
    MachineFunction *MF;
    SmallVector<MBBInfo, 16> MBBInfos;
  };

  char MipsLongBranch::ID = 0;
} // end of anonymous namespace

/// createMipsLongBranchPass - Returns a pass that converts branches to long
/// branches.
FunctionPass *llvm::createMipsLongBranchPass(MipsTargetMachine &tm) {
  return new MipsLongBranch(tm);
}

/// Iterate over list of Br's operands and search for a MachineBasicBlock
/// operand.
static MachineBasicBlock *getTargetMBB(const MachineInstr &Br) {
  for (unsigned I = 0, E = Br.getDesc().getNumOperands(); I < E; ++I) {
    const MachineOperand &MO = Br.getOperand(I);

    if (MO.isMBB())
      return MO.getMBB();
  }

  assert(false && "This instruction does not have an MBB operand.");
  return 0;
}

// Traverse the list of instructions backwards until a non-debug instruction is
// found or it reaches E.
static ReverseIter getNonDebugInstr(ReverseIter B, ReverseIter E) {
  for (; B != E; ++B)
    if (!B->isDebugValue())
      return B;

  return E;
}

// Split MBB if it has two direct jumps/branches.
void MipsLongBranch::splitMBB(MachineBasicBlock *MBB) {
  ReverseIter End = MBB->rend();
  ReverseIter LastBr = getNonDebugInstr(MBB->rbegin(), End);

  // Return if MBB has no branch instructions.
  if ((LastBr == End) ||
      (!LastBr->isConditionalBranch() && !LastBr->isUnconditionalBranch()))
    return;

  ReverseIter FirstBr = getNonDebugInstr(llvm::next(LastBr), End);

  // MBB has only one branch instruction if FirstBr is not a branch
  // instruction.
  if ((FirstBr == End) ||
      (!FirstBr->isConditionalBranch() && !FirstBr->isUnconditionalBranch()))
    return;

  assert(!FirstBr->isIndirectBranch() && "Unexpected indirect branch found.");

  // Create a new MBB. Move instructions in MBB to the newly created MBB.
  MachineBasicBlock *NewMBB =
    MF->CreateMachineBasicBlock(MBB->getBasicBlock());

  // Insert NewMBB and fix control flow.
  MachineBasicBlock *Tgt = getTargetMBB(*FirstBr);
  NewMBB->transferSuccessors(MBB);
  NewMBB->removeSuccessor(Tgt);
  MBB->addSuccessor(NewMBB);
  MBB->addSuccessor(Tgt);
  MF->insert(llvm::next(MachineFunction::iterator(MBB)), NewMBB);

  NewMBB->splice(NewMBB->end(), MBB, (++LastBr).base(), MBB->end());
}

// Fill MBBInfos.
void MipsLongBranch::initMBBInfo() {
  // Split the MBBs if they have two branches. Each basic block should have at
  // most one branch after this loop is executed.
  for (MachineFunction::iterator I = MF->begin(), E = MF->end(); I != E;)
    splitMBB(I++);

  MF->RenumberBlocks();
  MBBInfos.clear();
  MBBInfos.resize(MF->size());

  for (unsigned I = 0, E = MBBInfos.size(); I < E; ++I) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(I);

    // Compute size of MBB.
    for (MachineBasicBlock::instr_iterator MI = MBB->instr_begin();
         MI != MBB->instr_end(); ++MI)
      MBBInfos[I].Size += TII->GetInstSizeInBytes(&*MI);

    // Search for MBB's branch instruction.
    ReverseIter End = MBB->rend();
    ReverseIter Br = getNonDebugInstr(MBB->rbegin(), End);

    if ((Br != End) && !Br->isIndirectBranch() &&
        (Br->isConditionalBranch() ||
         (Br->isUnconditionalBranch() &&
          TM.getRelocationModel() == Reloc::PIC_)))
      MBBInfos[I].Br = (++Br).base();
  }
}

// Compute offset of branch in number of bytes.
int64_t MipsLongBranch::computeOffset(const MachineInstr *Br) {
  int64_t Offset = 0;
  int ThisMBB = Br->getParent()->getNumber();
  int TargetMBB = getTargetMBB(*Br)->getNumber();

  // Compute offset of a forward branch.
  if (ThisMBB < TargetMBB) {
    for (int N = ThisMBB + 1; N < TargetMBB; ++N)
      Offset += MBBInfos[N].Size;

    return Offset + 4;
  }

  // Compute offset of a backward branch.
  for (int N = ThisMBB; N >= TargetMBB; --N)
    Offset += MBBInfos[N].Size;

  return -Offset + 4;
}

// Replace Br with a branch which has the opposite condition code and a
// MachineBasicBlock operand MBBOpnd.
void MipsLongBranch::replaceBranch(MachineBasicBlock &MBB, Iter Br,
                                   DebugLoc DL, MachineBasicBlock *MBBOpnd) {
  unsigned NewOpc = Mips::GetOppositeBranchOpc(Br->getOpcode());
  const MCInstrDesc &NewDesc = TII->get(NewOpc);

  MachineInstrBuilder MIB = BuildMI(MBB, Br, DL, NewDesc);

  for (unsigned I = 0, E = Br->getDesc().getNumOperands(); I < E; ++I) {
    MachineOperand &MO = Br->getOperand(I);

    if (!MO.isReg()) {
      assert(MO.isMBB() && "MBB operand expected.");
      break;
    }

    MIB.addReg(MO.getReg());
  }

  MIB.addMBB(MBBOpnd);

  Br->eraseFromParent();
}

// Expand branch instructions to long branches.
void MipsLongBranch::expandToLongBranch(MBBInfo &I) {
  I.HasLongBranch = true;

  bool IsPIC = TM.getRelocationModel() == Reloc::PIC_;
  unsigned ABI = TM.getSubtarget<MipsSubtarget>().getTargetABI();
  bool N64 = ABI == MipsSubtarget::N64;

  MachineBasicBlock::iterator Pos;
  MachineBasicBlock *MBB = I.Br->getParent(), *TgtMBB = getTargetMBB(*I.Br);
  DebugLoc DL = I.Br->getDebugLoc();
  const BasicBlock *BB = MBB->getBasicBlock();
  MachineFunction::iterator FallThroughMBB = ++MachineFunction::iterator(MBB);
  MachineBasicBlock *LongBrMBB = MF->CreateMachineBasicBlock(BB);

  MF->insert(FallThroughMBB, LongBrMBB);
  MBB->removeSuccessor(TgtMBB);
  MBB->addSuccessor(LongBrMBB);

  if (IsPIC) {
    // $longbr:
    //  addiu $sp, $sp, -regsize * 2
    //  sw $ra, 0($sp)
    //  bal $baltgt
    //  sw $a3, regsize($sp)
    // $baltgt:
    //  lui $a3, %hi($baltgt)
    //  lui $at, %hi($tgt)
    //  addiu $a3, $a3, %lo($baltgt)
    //  addiu $at, $at, %lo($tgt)
    //  subu $at, $at, $a3
    //  addu $at, $ra, $at
    //
    //  if n64:
    //   lui $a3, %highest($baltgt)
    //   lui $ra, %highest($tgt)
    //   addiu $a3, $a3, %higher($baltgt)
    //   addiu $ra, $ra, %higher($tgt)
    //   dsll $a3, $a3, 32
    //   dsll $ra, $ra, 32
    //   subu $at, $at, $a3
    //   addu $at, $at, $ra
    //
    //  lw $ra, 0($sp)
    //  lw $a3, regsize($sp)
    //  jr $at
    //  addiu $sp, $sp, regsize * 2
    // $fallthrough:
    //
    MF->getInfo<MipsFunctionInfo>()->setEmitNOAT();
    MachineBasicBlock *BalTgtMBB = MF->CreateMachineBasicBlock(BB);
    MF->insert(FallThroughMBB, BalTgtMBB);
    LongBrMBB->addSuccessor(BalTgtMBB);
    BalTgtMBB->addSuccessor(TgtMBB);

    int RegSize = N64 ? 8 : 4;
    unsigned AT = N64 ? Mips::AT_64 : Mips::AT;
    unsigned A3 = N64 ? Mips::A3_64 : Mips::A3;
    unsigned SP = N64 ? Mips::SP_64 : Mips::SP;
    unsigned RA = N64 ? Mips::RA_64 : Mips::RA;
    unsigned Load = N64 ? Mips::LD_P8 : Mips::LW;
    unsigned Store = N64 ? Mips::SD_P8 : Mips::SW;
    unsigned LUi = N64 ? Mips::LUi64 : Mips::LUi;
    unsigned ADDiu = N64 ? Mips::DADDiu : Mips::ADDiu;
    unsigned ADDu = N64 ? Mips::DADDu : Mips::ADDu;
    unsigned SUBu = N64 ? Mips::SUBu : Mips::SUBu;
    unsigned JR = N64 ? Mips::JR64 : Mips::JR;

    Pos = LongBrMBB->begin();

    BuildMI(*LongBrMBB, Pos, DL, TII->get(ADDiu), SP).addReg(SP)
      .addImm(-RegSize * 2);
    BuildMI(*LongBrMBB, Pos, DL, TII->get(Store)).addReg(RA).addReg(SP)
      .addImm(0);
    BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::BAL_BR)).addMBB(BalTgtMBB);
    BuildMI(*LongBrMBB, Pos, DL, TII->get(Store)).addReg(A3).addReg(SP)
      .addImm(RegSize)->setIsInsideBundle();

    Pos = BalTgtMBB->begin();

    BuildMI(*BalTgtMBB, Pos, DL, TII->get(LUi), A3)
      .addMBB(BalTgtMBB, MipsII::MO_ABS_HI);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(LUi), AT)
      .addMBB(TgtMBB, MipsII::MO_ABS_HI);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDiu), A3).addReg(A3)
      .addMBB(BalTgtMBB, MipsII::MO_ABS_LO);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDiu), AT).addReg(AT)
      .addMBB(TgtMBB, MipsII::MO_ABS_LO);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(SUBu), AT).addReg(AT).addReg(A3);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDu), AT).addReg(RA).addReg(AT);

    if (N64) {
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(LUi), A3)
        .addMBB(BalTgtMBB, MipsII::MO_HIGHEST);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(LUi), RA)
        .addMBB(TgtMBB, MipsII::MO_HIGHEST);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDiu), A3).addReg(A3)
        .addMBB(BalTgtMBB, MipsII::MO_HIGHER);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDiu), RA).addReg(RA)
        .addMBB(TgtMBB, MipsII::MO_HIGHER);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::DSLL), A3).addReg(A3)
        .addImm(32);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(Mips::DSLL), RA).addReg(RA)
        .addImm(32);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(SUBu), AT).addReg(AT).addReg(A3);
      BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDu), AT).addReg(AT).addReg(RA);
      I.Size += 4 * 8;
    }

    BuildMI(*BalTgtMBB, Pos, DL, TII->get(Load), RA).addReg(SP).addImm(0);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(Load), A3).addReg(SP).addImm(RegSize);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(JR)).addReg(AT);
    BuildMI(*BalTgtMBB, Pos, DL, TII->get(ADDiu), SP).addReg(SP)
      .addImm(RegSize * 2)->setIsInsideBundle();
    I.Size += 4 * 14;
  } else {
    // $longbr:
    //  j $tgt
    //  nop
    // $fallthrough:
    //
    Pos = LongBrMBB->begin();
    LongBrMBB->addSuccessor(TgtMBB);
    BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::J)).addMBB(TgtMBB);
    BuildMI(*LongBrMBB, Pos, DL, TII->get(Mips::NOP))->setIsInsideBundle();
    I.Size += 4 * 2;
  }

  if (I.Br->isUnconditionalBranch()) {
    // Change branch destination.
    assert(I.Br->getDesc().getNumOperands() == 1);
    I.Br->RemoveOperand(0);
    I.Br->addOperand(MachineOperand::CreateMBB(LongBrMBB));
  } else
    // Change branch destination and reverse condition.
    replaceBranch(*MBB, I.Br, DL, FallThroughMBB);
}

static void emitGPDisp(MachineFunction &F, const MipsInstrInfo *TII) {
  MachineBasicBlock &MBB = F.front();
  MachineBasicBlock::iterator I = MBB.begin();
  DebugLoc DL = MBB.findDebugLoc(MBB.begin());
  BuildMI(MBB, I, DL, TII->get(Mips::LUi), Mips::V0)
    .addExternalSymbol("_gp_disp", MipsII::MO_ABS_HI);
  BuildMI(MBB, I, DL, TII->get(Mips::ADDiu), Mips::V0)
    .addReg(Mips::V0).addExternalSymbol("_gp_disp", MipsII::MO_ABS_LO);
  MBB.removeLiveIn(Mips::V0);
}

bool MipsLongBranch::runOnMachineFunction(MachineFunction &F) {
  if ((TM.getRelocationModel() == Reloc::PIC_) &&
      TM.getSubtarget<MipsSubtarget>().isABI_O32() &&
      F.getInfo<MipsFunctionInfo>()->globalBaseRegSet())
    emitGPDisp(F, TII);

  if (SkipLongBranch)
    return true;

  MF = &F;
  initMBBInfo();

  SmallVector<MBBInfo, 16>::iterator I, E = MBBInfos.end();
  bool EverMadeChange = false, MadeChange = true;

  while (MadeChange) {
    MadeChange = false;

    for (I = MBBInfos.begin(); I != E; ++I) {
      // Skip if this MBB doesn't have a branch or the branch has already been
      // converted to a long branch.
      if (!I->Br || I->HasLongBranch)
        continue;

      if (!ForceLongBranch)
        // Check if offset fits into 16-bit immediate field of branches.
        if (isInt<16>(computeOffset(I->Br) / 4))
          continue;

      expandToLongBranch(*I);
      ++LongBranches;
      EverMadeChange = MadeChange = true;
    }
  }

  if (EverMadeChange)
    MF->RenumberBlocks();

  return true;
}
