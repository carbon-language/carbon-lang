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
    bool offsetFitsIntoField(const MachineInstr *Br);
    unsigned addLongBranch(MachineBasicBlock &MBB, Iter Pos,
                           MachineBasicBlock *Tgt, DebugLoc DL, bool Nop);
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

  ReverseIter FirstBr = getNonDebugInstr(next(LastBr), End);

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
  MF->insert(next(MachineFunction::iterator(MBB)), NewMBB);

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
        (Br->isConditionalBranch() || Br->isUnconditionalBranch()))
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

// Insert the following sequence:
// (pic or N64)
//  lw $at, global_reg_slot
//  lw $at, got($L1)($at)
//  addiu $at, $at, lo($L1)
//  jr $at
//  noop
// (static and !N64)
//  lui $at, hi($L1)
//  addiu $at, $at, lo($L1)
//  jr $at
//  noop
unsigned MipsLongBranch::addLongBranch(MachineBasicBlock &MBB, Iter Pos,
                                       MachineBasicBlock *Tgt, DebugLoc DL,
                                       bool Nop) {
  MF->getInfo<MipsFunctionInfo>()->setEmitNOAT();
  bool IsPIC = (TM.getRelocationModel() == Reloc::PIC_);
  unsigned ABI = TM.getSubtarget<MipsSubtarget>().getTargetABI();
  bool N64 = (ABI == MipsSubtarget::N64);
  unsigned NumInstrs;

  if (IsPIC || N64) {
    bool HasMips64 = TM.getSubtarget<MipsSubtarget>().hasMips64();
    unsigned AT = N64 ? Mips::AT_64 : Mips::AT;
    unsigned Load = N64 ? Mips::LD_P8 : Mips::LW;
    unsigned ADDiu = N64 ? Mips::DADDiu : Mips::ADDiu;
    unsigned JR = N64 ? Mips::JR64 : Mips::JR;
    unsigned GOTFlag = HasMips64 ? MipsII::MO_GOT_PAGE : MipsII::MO_GOT;
    unsigned OFSTFlag = HasMips64 ? MipsII::MO_GOT_OFST : MipsII::MO_ABS_LO;
    const MipsRegisterInfo *MRI =
      static_cast<const MipsRegisterInfo*>(TM.getRegisterInfo());
    unsigned SP = MRI->getFrameRegister(*MF);
    unsigned GlobalRegFI = MF->getInfo<MipsFunctionInfo>()->getGlobalRegFI();
    int64_t Offset = MF->getFrameInfo()->getObjectOffset(GlobalRegFI);

    if (isInt<16>(Offset)) {
      BuildMI(MBB, Pos, DL, TII->get(Load), AT).addReg(SP).addImm(Offset);
      NumInstrs = 1;
    } else {
      unsigned ADDu = N64 ? Mips::DADDu : Mips::ADDu;
      MipsAnalyzeImmediate::Inst LastInst(0, 0);

      MF->getInfo<MipsFunctionInfo>()->setEmitNOAT();
      NumInstrs = Mips::loadImmediate(Offset, N64, *TII, MBB, Pos, DL, true,
                                      &LastInst) + 2;
      BuildMI(MBB, Pos, DL, TII->get(ADDu), AT).addReg(SP).addReg(AT);
      BuildMI(MBB, Pos, DL, TII->get(Load), AT).addReg(AT)
        .addImm(SignExtend64<16>(LastInst.ImmOpnd));
    }

    BuildMI(MBB, Pos, DL, TII->get(Load), AT).addReg(AT).addMBB(Tgt, GOTFlag);
    BuildMI(MBB, Pos, DL, TII->get(ADDiu), AT).addReg(AT).addMBB(Tgt, OFSTFlag);
    BuildMI(MBB, Pos, DL, TII->get(JR)).addReg(Mips::AT, RegState::Kill);
    NumInstrs += 3;
  } else {
    BuildMI(MBB, Pos, DL, TII->get(Mips::LUi), Mips::AT)
      .addMBB(Tgt, MipsII::MO_ABS_HI);
    BuildMI(MBB, Pos, DL, TII->get(Mips::ADDiu), Mips::AT)
      .addReg(Mips::AT).addMBB(Tgt, MipsII::MO_ABS_LO);
    BuildMI(MBB, Pos, DL, TII->get(Mips::JR)).addReg(Mips::AT, RegState::Kill);
    NumInstrs = 3;
  }

  if (Nop) {
    BuildMI(MBB, Pos, DL, TII->get(Mips::NOP))->setIsInsideBundle();
    ++NumInstrs;
  }

  return NumInstrs;
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

  MachineBasicBlock *MBB = I.Br->getParent(), *Tgt = getTargetMBB(*I.Br);
  DebugLoc DL = I.Br->getDebugLoc();

  if (I.Br->isUnconditionalBranch()) {
    // Unconditional branch before transformation:
    //   b $tgt
    //   delay-slot-instr
    //
    // after transformation:
    //   delay-slot-instr
    //   lw $at, global_reg_slot
    //   lw $at, %got($tgt)($at)
    //   addiu $at, $at, %lo($tgt)
    //   jr $at
    //   nop
    I.Size += (addLongBranch(*MBB, next(Iter(I.Br)), Tgt, DL, true) - 1) * 4;

    // Remove branch and clear InsideBundle bit of the next instruction.
    next(MachineBasicBlock::instr_iterator(I.Br))->setIsInsideBundle(false);
    I.Br->eraseFromParent();
    return;
  }

  assert(I.Br->isConditionalBranch() && "Conditional branch expected.");

  // Conditional branch before transformation:
  //   b cc, $tgt
  //   delay-slot-instr
  //  FallThrough:
  //
  // after transformation:
  //   b !cc, FallThrough
  //   delay-slot-instr
  //  NewMBB:
  //   lw $at, global_reg_slot
  //   lw $at, %got($tgt)($at)
  //   addiu $at, $at, %lo($tgt)
  //   jr $at
  //   noop
  //  FallThrough:

  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB->getBasicBlock());
  MF->insert(next(MachineFunction::iterator(MBB)), NewMBB);
  MBB->removeSuccessor(Tgt);
  MBB->addSuccessor(NewMBB);
  NewMBB->addSuccessor(Tgt);

  I.Size += addLongBranch(*NewMBB, NewMBB->begin(), Tgt, DL, true) * 4;
  replaceBranch(*MBB, I.Br, DL, *MBB->succ_begin());
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
    return false;

  MF = &F;
  initMBBInfo();

  bool IsPIC = (TM.getRelocationModel() == Reloc::PIC_);
  SmallVector<MBBInfo, 16>::iterator I, E = MBBInfos.end();
  bool EverMadeChange = false, MadeChange = true;

  while (MadeChange) {
    MadeChange = false;

    for (I = MBBInfos.begin(); I != E; ++I) {
      // Skip if this MBB doesn't have a branch or the branch has already been
      // converted to a long branch.
      if (!I->Br || I->HasLongBranch)
        continue;

      int64_t Offset = computeOffset(I->Br);

      if (!ForceLongBranch) {
        // Check if offset fits into 16-bit immediate field of branches.
        if ((I->Br->isConditionalBranch() || IsPIC) && isInt<16>(Offset / 4))
          continue;

        // Check if offset fits into 26-bit immediate field of jumps (J).
        if (I->Br->isUnconditionalBranch() && !IsPIC && isInt<26>(Offset / 4))
          continue;
      }

      expandToLongBranch(*I);
      ++LongBranches;
      EverMadeChange = MadeChange = true;
    }
  }

  if (EverMadeChange)
    MF->RenumberBlocks();

  return EverMadeChange;
}
