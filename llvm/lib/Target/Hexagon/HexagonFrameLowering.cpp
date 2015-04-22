//===-- HexagonFrameLowering.cpp - Define frame lowering ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "hexagon-pei"

#include "HexagonFrameLowering.h"
#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

// Hexagon stack frame layout as defined by the ABI:
//
//                                                       Incoming arguments
//                                                       passed via stack
//                                                                      |
//                                                                      |
//        SP during function's                 FP during function's     |
//    +-- runtime (top of stack)               runtime (bottom) --+     |
//    |                                                           |     |
// --++---------------------+------------------+-----------------++-+-------
//   |  parameter area for  |  variable-size   |   fixed-size    |LR|  arg
//   |   called functions   |  local objects   |  local objects  |FP|
// --+----------------------+------------------+-----------------+--+-------
//    <-    size known    -> <- size unknown -> <- size known  ->
//
// Low address                                                 High address
//
// <--- stack growth
//
//
// - In any circumstances, the outgoing function arguments are always accessi-
//   ble using the SP, and the incoming arguments are accessible using the FP.
// - If the local objects are not aligned, they can always be accessed using
//   the FP.
// - If there are no variable-sized objects, the local objects can always be
//   accessed using the SP, regardless whether they are aligned or not. (The
//   alignment padding will be at the bottom of the stack (highest address),
//   and so the offset with respect to the SP will be known at the compile-
//   -time.)
//
// The only complication occurs if there are both, local aligned objects, and
// dynamically allocated (variable-sized) objects. The alignment pad will be
// placed between the FP and the local objects, thus preventing the use of the
// FP to access the local objects. At the same time, the variable-sized objects
// will be between the SP and the local objects, thus introducing an unknown
// distance from the SP to the locals.
//
// To avoid this problem, a new register is created that holds the aligned
// address of the bottom of the stack, referred in the sources as AP (aligned
// pointer). The AP will be equal to "FP-p", where "p" is the smallest pad
// that aligns AP to the required boundary (a maximum of the alignments of
// all stack objects, fixed- and variable-sized). All local objects[1] will
// then use AP as the base pointer.
// [1] The exception is with "fixed" stack objects. "Fixed" stack objects get
// their name from being allocated at fixed locations on the stack, relative
// to the FP. In the presence of dynamic allocation and local alignment, such
// objects can only be accessed through the FP.
//
// Illustration of the AP:
//                                                                FP --+
//                                                                     |
// ---------------+---------------------+-----+-----------------------++-+--
//   Rest of the  | Local stack objects | Pad |  Fixed stack objects  |LR|
//   stack frame  | (aligned)           |     |  (CSR, spills, etc.)  |FP|
// ---------------+---------------------+-----+-----------------+-----+--+--
//                                      |<-- Multiple of the -->|
//                                           stack alignment    +-- AP
//
// The AP is set up at the beginning of the function. Since it is not a dedi-
// cated (reserved) register, it needs to be kept live throughout the function
// to be available as the base register for local object accesses.
// Normally, an address of a stack objects is obtained by a pseudo-instruction
// TFR_FI. To access local objects with the AP register present, a different
// pseudo-instruction needs to be used: TFR_FIA. The TFR_FIA takes one extra
// argument compared to TFR_FI: the first input register is the AP register.
// This keeps the register live between its definition and its uses.

// The AP register is originally set up using pseudo-instruction ALIGNA:
//   AP = ALIGNA A
// where
//   A  - required stack alignment
// The alignment value must be the maximum of all alignments required by
// any stack object.

// The dynamic allocation uses a pseudo-instruction ALLOCA:
//   Rd = ALLOCA Rs, A
// where
//   Rd - address of the allocated space
//   Rs - minimum size (the actual allocated can be larger to accommodate
//        alignment)
//   A  - required alignment


using namespace llvm;

static cl::opt<bool> DisableDeallocRet(
                       "disable-hexagon-dealloc-ret",
                       cl::Hidden,
                       cl::desc("Disable Dealloc Return for Hexagon target"));

static cl::opt<bool>
DisableProEpilogCodeMerge("disable-hexagon-prolog-epilog-code-merge",
    cl::Hidden, cl::desc("Disable prolog/epilog code merge"), cl::init(false),
    cl::ZeroOrMore);

static cl::opt<int>
NumberScavengerSlots("number-scavenger-slots",
    cl::Hidden, cl::desc("Set the number of scavenger slots"), cl::init(2),
    cl::ZeroOrMore);

static cl::opt<int>
SpillFuncThreshold("spill-func-threshold",
    cl::Hidden, cl::desc("Specify O2(not Os) spill func threshold"),
    cl::init(6), cl::ZeroOrMore);

static cl::opt<int>
SpillFuncThresholdOs("spill-func-threshold-Os",
    cl::Hidden, cl::desc("Specify Os spill func threshold"),
    cl::init(1), cl::ZeroOrMore);

/// Map a register pair Reg to the subregister that has the greater "number",
/// i.e. D3 (aka R7:6) will be mapped to R7, etc.
static unsigned getMax32BitSubRegister(unsigned Reg,
                                       const TargetRegisterInfo *TRI,
                                       bool hireg = true) {
  if (Reg < Hexagon::D0 || Reg > Hexagon::D15)
    return Reg;

  unsigned RegNo = 0;

  for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
    if (hireg) {
      if (*SubRegs > RegNo)
        RegNo = *SubRegs;
    } else {
      if (!RegNo || *SubRegs < RegNo)
        RegNo = *SubRegs;
    }

  return RegNo;
}

/// Returns the biggest callee saved register in the vector.
static unsigned getMaxCalleeSavedReg(const std::vector<CalleeSavedInfo> &CSI,
                                     const TargetRegisterInfo *TRI) {
  assert(Hexagon::R1 > 0 &&
         "Assume physical registers are encoded as positive integers");
  if (CSI.empty())
    return 0;

  unsigned Max = getMax32BitSubRegister(CSI[0].getReg(), TRI);
  for (unsigned I = 1, E = CSI.size(); I < E; ++I) {
    unsigned Reg = getMax32BitSubRegister(CSI[I].getReg(), TRI);
    if (Reg > Max)
      Max = Reg;
  }

  return Max;
}

void HexagonFrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  auto &HTM = static_cast<const HexagonTargetMachine&>(MF.getTarget());
  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HII = *HST.getInstrInfo();
  auto &HRI = *HST.getRegisterInfo();
  DebugLoc dl;

  unsigned MaxAlign = std::max(MFI->getMaxAlignment(), getStackAlignment());

  // Calculate the total stack frame size.
  // Get the number of bytes to allocate from the FrameInfo.
  unsigned FrameSize = MFI->getStackSize();
  // Round up the max call frame size to the max alignment on the stack.
  unsigned MaxCFA = RoundUpToAlignment(MFI->getMaxCallFrameSize(), MaxAlign);
  MFI->setMaxCallFrameSize(MaxCFA);

  FrameSize = MaxCFA + RoundUpToAlignment(FrameSize, MaxAlign);
  MFI->setStackSize(FrameSize);

  bool AlignStack = (MaxAlign > 8);

  // Check if frame moves are needed for EH.
  bool needsFrameMoves = MMI.hasDebugInfo() ||
    MF.getFunction()->needsUnwindTableEntry();

  // Get the number of bytes to allocate from the FrameInfo.
  unsigned NumBytes = MFI->getStackSize();
  unsigned SP = HRI.getStackRegister();
  unsigned MaxCF = MFI->getMaxCallFrameSize();
  MachineBasicBlock::iterator InsertPt = MBB.begin();

  auto *FuncInfo = MF.getInfo<HexagonMachineFunctionInfo>();
  auto &AdjustRegs = FuncInfo->getAllocaAdjustInsts();

  for (auto MI : AdjustRegs) {
    assert((MI->getOpcode() == Hexagon::ALLOCA) && "Expected alloca");
    expandAlloca(MI, HII, SP, MaxCF);
    MI->eraseFromParent();
  }

  //
  // Only insert ALLOCFRAME if we need to or at -O0 for the debugger.  Think
  // that this shouldn't be required, but doing so now because gcc does and
  // gdb can't break at the start of the function without it.  Will remove if
  // this turns out to be a gdb bug.
  //
  bool NoOpt = (HTM.getOptLevel() == CodeGenOpt::None);
  if (!NoOpt && !FuncInfo->hasClobberLR() && !hasFP(MF))
    return;

  // Check for overflow.
  // Hexagon_TODO: Ugh! hardcoding. Is there an API that can be used?
  const unsigned int ALLOCFRAME_MAX = 16384;

  // Create a dummy memory operand to avoid allocframe from being treated as
  // a volatile memory reference.
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(), MachineMemOperand::MOStore,
                            4, 4);

  if (NumBytes >= ALLOCFRAME_MAX) {
    // Emit allocframe(#0).
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::S2_allocframe))
      .addImm(0)
      .addMemOperand(MMO);

    // Subtract offset from frame pointer.
    // We use a caller-saved non-parameter register for that.
    unsigned CallerSavedReg = HRI.getFirstCallerSavedNonParamReg();
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::CONST32_Int_Real),
            CallerSavedReg).addImm(NumBytes);
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::A2_sub), SP).
      addReg(SP).
      addReg(CallerSavedReg);
  } else {
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::S2_allocframe))
      .addImm(NumBytes)
      .addMemOperand(MMO);
  }

  if (AlignStack) {
    BuildMI(MBB, InsertPt, dl, HII.get(Hexagon::A2_andir), SP)
        .addReg(SP)
        .addImm(-int64_t(MaxAlign));
  }

  if (needsFrameMoves) {
    std::vector<MCCFIInstruction> Instructions = MMI.getFrameInstructions();
    MCSymbol *FrameLabel = MMI.getContext().CreateTempSymbol();

    // Advance CFA. DW_CFA_def_cfa
    unsigned DwFPReg = HRI.getDwarfRegNum(HRI.getFrameRegister(), true);
    unsigned DwRAReg = HRI.getDwarfRegNum(HRI.getRARegister(), true);

    // CFA = FP + 8
    unsigned CFIIndex = MMI.addFrameInst(MCCFIInstruction::createDefCfa(
                                               FrameLabel, DwFPReg, -8));
    BuildMI(MBB, MBBI, dl, HII.get(TargetOpcode::CFI_INSTRUCTION))
           .addCFIIndex(CFIIndex);

    // R31 (return addr) = CFA - #4
    CFIIndex = MMI.addFrameInst(MCCFIInstruction::createOffset(
                                               FrameLabel, DwRAReg, -4));
    BuildMI(MBB, MBBI, dl, HII.get(TargetOpcode::CFI_INSTRUCTION))
           .addCFIIndex(CFIIndex);

    // R30 (frame ptr) = CFA - #8)
    CFIIndex = MMI.addFrameInst(MCCFIInstruction::createOffset(
                                               FrameLabel, DwFPReg, -8));
    BuildMI(MBB, MBBI, dl, HII.get(TargetOpcode::CFI_INSTRUCTION))
           .addCFIIndex(CFIIndex);

    unsigned int regsToMove[] = {
      Hexagon::R1,  Hexagon::R0,  Hexagon::R3,  Hexagon::R2,
      Hexagon::R17, Hexagon::R16, Hexagon::R19, Hexagon::R18,
      Hexagon::R21, Hexagon::R20, Hexagon::R23, Hexagon::R22,
      Hexagon::R25, Hexagon::R24, Hexagon::R27, Hexagon::R26,
      Hexagon::D0,  Hexagon::D1,  Hexagon::D8,  Hexagon::D9,  Hexagon::D10,
      Hexagon::D11, Hexagon::D12, Hexagon::D13, Hexagon::NoRegister
    };

    const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

    for (unsigned i = 0; regsToMove[i] != Hexagon::NoRegister; ++i) {
      for (unsigned I = 0, E = CSI.size(); I < E; ++I) {
        if (CSI[I].getReg() == regsToMove[i]) {
          // Subtract 8 to make room for R30 and R31, which are added above.
          int64_t Offset = getFrameIndexOffset(MF, CSI[I].getFrameIdx()) - 8;

          if (regsToMove[i] < Hexagon::D0 || regsToMove[i] > Hexagon::D15) {
            unsigned DwarfReg = HRI.getDwarfRegNum(regsToMove[i], true);
            unsigned CFIIndex = MMI.addFrameInst(
                                    MCCFIInstruction::createOffset(FrameLabel,
                                                        DwarfReg, Offset));
            BuildMI(MBB, MBBI, dl, HII.get(TargetOpcode::CFI_INSTRUCTION))
                   .addCFIIndex(CFIIndex);
          } else {
            // Split the double regs into subregs, and generate appropriate
            // cfi_offsets.
            // The only reason, we are split double regs is, llvm-mc does not
            // understand paired registers for cfi_offset.
            // Eg .cfi_offset r1:0, -64
            unsigned HiReg = getMax32BitSubRegister(regsToMove[i], &HRI);
            unsigned LoReg = getMax32BitSubRegister(regsToMove[i], &HRI, false);
            unsigned HiDwarfReg = HRI.getDwarfRegNum(HiReg, true);
            unsigned LoDwarfReg = HRI.getDwarfRegNum(LoReg, true);
            unsigned HiCFIIndex = MMI.addFrameInst(
                                    MCCFIInstruction::createOffset(FrameLabel,
                                                        HiDwarfReg, Offset+4));
            BuildMI(MBB, MBBI, dl, HII.get(TargetOpcode::CFI_INSTRUCTION))
                   .addCFIIndex(HiCFIIndex);
            unsigned LoCFIIndex = MMI.addFrameInst(
                                    MCCFIInstruction::createOffset(FrameLabel,
                                                        LoDwarfReg, Offset));
            BuildMI(MBB, MBBI, dl, HII.get(TargetOpcode::CFI_INSTRUCTION))
                   .addCFIIndex(LoCFIIndex);
          }
          break;
        }
      } // for CSI.size()
    } // for regsToMove
  } // needsFrameMoves
}

// Returns true if MBB has a machine instructions that indicates a tail call
// in the block.
bool HexagonFrameLowering::hasTailCall(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  unsigned RetOpcode = MBBI->getOpcode();

  return RetOpcode == Hexagon::TCRETURNi || RetOpcode == Hexagon::TCRETURNr;
}

void HexagonFrameLowering::emitEpilogue(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  //
  // Only insert deallocframe if we need to.  Also at -O0.  See comment
  // in emitPrologue above.
  //
  if (!hasFP(MF) && MF.getTarget().getOptLevel() != CodeGenOpt::None)
    return;

  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HII = *HST.getInstrInfo();

  MachineBasicBlock::iterator MBBI = std::prev(MBB.end());
  MachineBasicBlock::iterator MBBI_end = MBB.end();
  DebugLoc dl = MBBI->getDebugLoc();

  // Handle EH_RETURN.
  if (MBBI->getOpcode() == Hexagon::EH_RETURN_JMPR) {
    MachineOperand &OffsetReg  = MBBI->getOperand(0);
    assert(OffsetReg.isReg() && "Offset should be in register!");
    BuildMI(MBB, MBBI, dl, HII.get(Hexagon::L2_deallocframe));
    BuildMI(MBB, MBBI, dl, HII.get(Hexagon::A2_add),
            Hexagon::R29).addReg(Hexagon::R29).addReg(Hexagon::R28);
    return;
  }
  // Replace 'jumpr r31' instruction with dealloc_return for V4 and higher
  // versions.
  if (MBBI->getOpcode() == Hexagon::JMPret && !DisableDeallocRet) {
    // Check for RESTORE_DEALLOC_RET_JMP_V4 call. Don't emit an extra DEALLOC
    // instruction if we encounter it.
    MachineBasicBlock::iterator BeforeJMPR =
      MBB.begin() == MBBI ? MBBI : std::prev(MBBI);
    if (BeforeJMPR != MBBI &&
        BeforeJMPR->getOpcode() == Hexagon::RESTORE_DEALLOC_RET_JMP_V4) {
      // Remove the JMPR node.
      MBB.erase(MBBI);
      return;
    }

    // Add dealloc_return.
    MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI_end, dl, HII.get(Hexagon::L4_return));
    // Transfer the function live-out registers.
    MIB->copyImplicitOps(*MBB.getParent(), &*MBBI);
    // Remove the JUMPR node.
    MBB.erase(MBBI);
  } else { // Add deallocframe for tail calls.
    // Check for RESTORE_DEALLOC_BEFORE_TAILCALL_V4. We don't need an extra
    // L2_deallocframe instruction after it.
    MachineBasicBlock::iterator Term = MBB.getFirstTerminator();
    MachineBasicBlock::iterator I =
      Term == MBB.begin() ?  MBB.end() : std::prev(Term);
    if (I != MBB.end() &&
        I->getOpcode() == Hexagon::RESTORE_DEALLOC_BEFORE_TAILCALL_V4)
      return;

    BuildMI(MBB, MBBI, dl, HII.get(Hexagon::L2_deallocframe));
  }
}

bool HexagonFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const HexagonMachineFunctionInfo *FuncInfo =
    MF.getInfo<HexagonMachineFunctionInfo>();
  return (MFI->hasCalls() || (MFI->getStackSize() > 0) ||
          FuncInfo->hasClobberLR() );
}

enum SpillKind {
  SK_ToMem,
  SK_FromMem,
  SK_FromMemTailcall
};

static const char *
getSpillFunctionFor(unsigned MaxReg, SpillKind SpillType) {
  const char * V4SpillToMemoryFunctions[] = {
    "__save_r16_through_r17",
    "__save_r16_through_r19",
    "__save_r16_through_r21",
    "__save_r16_through_r23",
    "__save_r16_through_r25",
    "__save_r16_through_r27" };

  const char * V4SpillFromMemoryFunctions[] = {
    "__restore_r16_through_r17_and_deallocframe",
    "__restore_r16_through_r19_and_deallocframe",
    "__restore_r16_through_r21_and_deallocframe",
    "__restore_r16_through_r23_and_deallocframe",
    "__restore_r16_through_r25_and_deallocframe",
    "__restore_r16_through_r27_and_deallocframe" };

  const char * V4SpillFromMemoryTailcallFunctions[] = {
    "__restore_r16_through_r17_and_deallocframe_before_tailcall",
    "__restore_r16_through_r19_and_deallocframe_before_tailcall",
    "__restore_r16_through_r21_and_deallocframe_before_tailcall",
    "__restore_r16_through_r23_and_deallocframe_before_tailcall",
    "__restore_r16_through_r25_and_deallocframe_before_tailcall",
    "__restore_r16_through_r27_and_deallocframe_before_tailcall"
  };

  const char **SpillFunc = nullptr;

  switch(SpillType) {
  case SK_ToMem:
    SpillFunc = V4SpillToMemoryFunctions;
    break;
  case SK_FromMem:
    SpillFunc = V4SpillFromMemoryFunctions;
    break;
  case SK_FromMemTailcall:
    SpillFunc = V4SpillFromMemoryTailcallFunctions;
    break;
  }
  assert(SpillFunc && "Unknown spill kind");

  // Spill all callee-saved registers up to the highest register used.
  switch (MaxReg) {
  case Hexagon::R17:
    return SpillFunc[0];
  case Hexagon::R19:
    return SpillFunc[1];
  case Hexagon::R21:
    return SpillFunc[2];
  case Hexagon::R23:
    return SpillFunc[3];
  case Hexagon::R25:
    return SpillFunc[4];
  case Hexagon::R27:
    return SpillFunc[5];
  default:
    assert(false && "Unhandled maximum callee save register");
    break;
  }
  return 0;
}

/// Adds all callee-saved registers up to MaxReg to the instruction.
static void addCalleeSaveRegistersAsImpOperand(MachineInstr *Inst,
                                           unsigned MaxReg, bool IsDef) {
  // Add the callee-saved registers as implicit uses.
  for (unsigned R = Hexagon::R16; R <= MaxReg; ++R) {
    MachineOperand ImpUse = MachineOperand::CreateReg(R, IsDef, true);
    Inst->addOperand(ImpUse);
  }
}

/// Returns true if we have V4 operations, optimization level is equal to
/// O2 or Os and exception handling is not required.
static bool shouldOptimizeForSize(MachineFunction &MF) {
  if (MF.getInfo<HexagonMachineFunctionInfo>()->hasEHReturn())
    return false;

  if (MF.getTarget().getOptLevel() == CodeGenOpt::Default) /* -O2 or -Os */
    return true;

  const Function *F = MF.getFunction();
  if (F->hasFnAttribute(Attribute::MinSize))
    return true;

  return F->hasFnAttribute(Attribute::OptimizeForSize);
}

bool
HexagonFrameLowering::spillCalleeSavedRegisters(
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
  const HexagonRegisterInfo *HRI =
    static_cast<const HexagonRegisterInfo*>(TRI);

  if (CSI.empty()) {
    return false;
  }

  // Check whether we should emit a function call to spill the registers.
  bool ShouldOptimizeForSize = shouldOptimizeForSize(*MF);

  if (ShouldOptimizeForSize && !DisableProEpilogCodeMerge) {
    // We have different cut offs when we use callee saved stubs at -Os.
    bool HasOptimizeForSizeAttr = MF->getFunction()->getAttributes()
        .hasAttribute(AttributeSet::FunctionIndex, Attribute::OptimizeForSize);
    unsigned CalleeSavedStubCutOff = HasOptimizeForSizeAttr ?
                                     SpillFuncThresholdOs : SpillFuncThreshold;

    // Use a function call if we have to spill registers.
    if (CSI.size() > CalleeSavedStubCutOff) {
      unsigned MaxReg = getMaxCalleeSavedReg(CSI, TRI);
      const char *SpillFun = getSpillFunctionFor(MaxReg, SK_ToMem);
      // Call spill function.
      DebugLoc DL = MI != MBB.end() ? MI->getDebugLoc() : DebugLoc();
      MachineInstr *SaveRegsCall =
        BuildMI(MBB, MI, DL, TII.get(Hexagon::SAVE_REGISTERS_CALL_V4))
          .addExternalSymbol(SpillFun);
      // Add callee-saved registers as use.
      addCalleeSaveRegistersAsImpOperand(SaveRegsCall, MaxReg, false);
      // Add live in registers.
      for (unsigned I = 0; I < CSI.size(); ++I)
        MBB.addLiveIn(CSI[I].getReg());
      return true;
    }
  }

  for (unsigned i = 0; i < CSI.size(); ++i) {
    unsigned Reg = CSI[i].getReg();
    // Add live in registers. We treat eh_return callee saved register r0 - r3
    // specially. They are not really callee saved registers as they are not
    // supposed to be killed.
    bool IsEHReturnCalleeSavedReg = HRI->isEHReturnCalleeSaveReg(Reg);

    if (HRI->isCalleeSaveReg(Reg)) {
      int FrameIdx = CSI[i].getFrameIdx();
      const TargetRegisterClass *RegClass =
          TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(MBB, MI, Reg, !IsEHReturnCalleeSavedReg,
                              FrameIdx, RegClass, TRI);
      if (!IsEHReturnCalleeSavedReg)
        MBB.addLiveIn(Reg);
    } else {
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(MBB, MI, Reg, !IsEHReturnCalleeSavedReg,
                              CSI[i].getFrameIdx(), RC, TRI);
      if (!IsEHReturnCalleeSavedReg)
        MBB.addLiveIn(Reg);
    }
  }
  return true;
}


bool HexagonFrameLowering::restoreCalleeSavedRegisters(
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {

  MachineFunction *MF = MBB.getParent();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();

  if (CSI.empty()) {
    return false;
  }

  // Check whether we should emit a function call to spill the registers.
  bool ShouldOptimizeForSize = shouldOptimizeForSize(*MF);

  if (ShouldOptimizeForSize && !DisableProEpilogCodeMerge) {
    // We have different cut offs when we use callee saved stubs at -Os.
    bool HasOptimizeForSizeAttr = MF->getFunction()->getAttributes()
        .hasAttribute(AttributeSet::FunctionIndex, Attribute::OptimizeForSize);
    unsigned CalleeSavedStubCutOff = HasOptimizeForSizeAttr ?
      (SpillFuncThresholdOs - 1) : SpillFuncThreshold;

    // Use a function call if we have to spill registers.
    if (CSI.size() > CalleeSavedStubCutOff) {
      bool IsTailCallBB = hasTailCall(MBB);
      unsigned MaxReg = getMaxCalleeSavedReg(CSI, TRI);
      const char *SpillFun =
        getSpillFunctionFor(MaxReg, IsTailCallBB ?
                            SK_FromMemTailcall : SK_FromMem);

      // Call spill function.
      DebugLoc DL = MI != MBB.end() ? MI->getDebugLoc() : DebugLoc();
      MachineInstr *DeallocCall = nullptr;

      if (IsTailCallBB) {
        DeallocCall = BuildMI(MBB, MI, DL,
                TII.get(Hexagon::RESTORE_DEALLOC_BEFORE_TAILCALL_V4))
          .addExternalSymbol(SpillFun);
      }
      else {
        MachineBasicBlock::iterator JumprInst = std::prev(MBB.end());
        assert(JumprInst->getOpcode() == Hexagon::JMPret);
        DeallocCall =
          BuildMI(MBB, MI, DL, TII.get(Hexagon::RESTORE_DEALLOC_RET_JMP_V4))
            .addExternalSymbol(SpillFun);
        // Transfer the function live-out registers.
        DeallocCall->copyImplicitOps(*MBB.getParent(), JumprInst);
      }
      addCalleeSaveRegistersAsImpOperand(DeallocCall, MaxReg, true);
      return true;
    }
  }

  const HexagonRegisterInfo *HRI =
    static_cast<const HexagonRegisterInfo*>(TRI);

  for (unsigned i = 0; i < CSI.size(); ++i) {
    unsigned Reg = CSI[i].getReg();
    // Add live in registers. We treat eh_return callee saved register r0 - r3
    // specially. They are not really callee saved registers as they are not
    // supposed to be killed.
    bool IsEHReturnCalleeSavedReg = HRI->isEHReturnCalleeSaveReg(Reg);

    if (HRI->isCalleeSaveReg(Reg)) {
      int FrameIdx = CSI[i].getFrameIdx();
      const TargetRegisterClass *RegClass =
          TRI->getMinimalPhysRegClass(Reg);
      TII.loadRegFromStackSlot(MBB, MI, Reg, FrameIdx,
                               RegClass, TRI);

      if (!IsEHReturnCalleeSavedReg)
        MBB.addLiveIn(Reg);

    } else {
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.loadRegFromStackSlot(MBB, MI, Reg, CSI[i].getFrameIdx(), RC, TRI);
      if (!IsEHReturnCalleeSavedReg)
        MBB.addLiveIn(Reg);
    }
  }
  return true;
}

void HexagonFrameLowering::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  MachineInstr &MI = *I;
  unsigned Opc = MI.getOpcode();
  assert((Opc == Hexagon::ADJCALLSTACKDOWN || Opc == Hexagon::ADJCALLSTACKUP) &&
         "Cannot handle this call frame pseudo instruction");
  MBB.erase(I);
}

int HexagonFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                              int FI) const {
  return MF.getFrameInfo()->getObjectOffset(FI);
}

void HexagonFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  // If this function has uses aligned stack and also has variable sized stack
  // objects, then we need to map all spill slots to fixed positions, so that
  // they can be accessed through FP. Otherwise they would have to be accessed
  // via AP, which may not be available at the particular place in the program.
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool HasAlloca = MFI->hasVarSizedObjects();
  bool HasAligna = (MFI->getMaxAlignment() > getStackAlignment());

  if (!HasAlloca || !HasAligna)
    return;

  unsigned LFS = MFI->getLocalFrameSize();
  int Offset = -LFS;
  for (int i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (!MFI->isSpillSlotObjectIndex(i) || MFI->isDeadObjectIndex(i))
      continue;
    int S = MFI->getObjectSize(i);
    LFS += S;
    Offset -= S;
    MFI->mapLocalFrameObject(i, Offset);
  }

  MFI->setLocalFrameSize(LFS);
  unsigned A = MFI->getLocalFrameMaxAlign();
  assert(A <= 8 && "Unexpected local frame alignment");
  if (A == 0)
    MFI->setLocalFrameMaxAlign(8);
  MFI->setUseLocalStackAllocationBlock(true);
}

/// Returns true if there is no caller saved registers available.
static bool needToReserveScavengingSpillSlots(MachineFunction &MF,
                                              const HexagonRegisterInfo &HRI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const uint16_t *CallerSavedRegs = HRI.getCallerSavedRegs(&MF);
  // Check for an unused caller-saved register.
  for ( ; *CallerSavedRegs; ++CallerSavedRegs) {
    unsigned FreeReg = *CallerSavedRegs;
    if (MRI.isPhysRegUsed(FreeReg))
      continue;

    // Check aliased register usage.
    bool IsCurrentRegUsed = false;
    for (MCRegAliasIterator AI(FreeReg, &HRI, false); AI.isValid(); ++AI)
      if (MRI.isPhysRegUsed(*AI)) {
        IsCurrentRegUsed = true;
        break;
      }
    if (IsCurrentRegUsed)
      continue;

    // Neither directly used nor used through an aliased register.
    return false;
  }
  // All caller-saved registers are used.
  return true;
}

/// Marks physical registers as used so that the callee-saved registers are
/// always a multiple of two.
void HexagonFrameLowering::
adjustForCalleeSavedRegsSpillCall(MachineFunction &MF) const {
  // We need to pretend that the callee-saved registers are a multiple of two
  // because there are only library functions to handle those cases.
  // (save_r16_to_17, save_r16_to_r19, ...)
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const uint16_t *CSRegs = TRI.getCalleeSavedRegs(&MF);

  // Compute the smallest and biggest callee-saved register used.
  unsigned Biggest = 0;
  unsigned Smallest = Hexagon::R31;
  unsigned NumUsed = 0;
  for (; *CSRegs; ++CSRegs){
    unsigned CurrCSR = *CSRegs;
    if (MRI.isPhysRegUsed(CurrCSR)) {
      ++NumUsed;
      if (CurrCSR >= Biggest)
        Biggest = CurrCSR;
      if (CurrCSR <= Smallest)
        Smallest = CurrCSR;
      continue;
    }
    // Also check alias registers.
    for (MCRegAliasIterator AI(CurrCSR, &TRI, false); AI.isValid(); ++AI) {
      if (MRI.isPhysRegUsed(*AI)) {
        ++NumUsed;
        if (CurrCSR >= Biggest)
          Biggest = CurrCSR;
        if (CurrCSR <= Smallest)
          Smallest = CurrCSR;
        break;
      }
    }
  }

  // Don't do anything if the number of register used is zero.
  if (NumUsed == 0)
    return;

  // Ensure that the biggest register is r17, r19, ...
  if ((Biggest - Hexagon::R0) % 2 == 0)
    ++Biggest;
  assert(Smallest != Hexagon::R31 && "Should not happen");
  // Make sure all the physical register are marked as used so that they are
  // assigned stack slots.
  for (unsigned I = Smallest; I <= Biggest; ++I)
    if (!MRI.isPhysRegUsed(I))
      MRI.setPhysRegUsed(I);
}

/// Replaces the predicate spill code pseudo instructions by valid instructions.
bool HexagonFrameLowering::replacePredRegPseudoSpillCode(MachineFunction &MF)
      const {
  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HII = *HST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool HasReplacedPseudoInst = false;
  // Replace predicate spill pseudo instructions by real code.
  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBBb = MF.begin(), MBBe = MF.end();
       MBBb != MBBe; ++MBBb) {
    MachineBasicBlock* MBB = MBBb;
    // Traverse the basic block.
    MachineBasicBlock::iterator NextII;
    for (MachineBasicBlock::iterator MII = MBB->begin(); MII != MBB->end();
         MII = NextII) {
      MachineInstr *MI = MII;
      NextII = std::next(MII);
      int Opc = MI->getOpcode();
      if (Opc == Hexagon::STriw_pred) {
        HasReplacedPseudoInst = true;
        // STriw_pred FI, 0, SrcReg;
        unsigned VirtReg = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
        unsigned SrcReg = MI->getOperand(2).getReg();
        bool IsOrigSrcRegKilled = MI->getOperand(2).isKill();

        assert(MI->getOperand(0).isFI() && "Expect a frame index");
        assert(Hexagon::PredRegsRegClass.contains(SrcReg) &&
               "Not a predicate register");

        // Insert transfer to general purpose register.
        //   VirtReg = C2_tfrpr SrcPredReg
        BuildMI(*MBB, MII, MI->getDebugLoc(), HII.get(Hexagon::C2_tfrpr),
                VirtReg).addReg(SrcReg, getKillRegState(IsOrigSrcRegKilled));

        // Change instruction to S2_storeri_io.
        //   S2_storeri_io FI, 0, VirtReg
        MI->setDesc(HII.get(Hexagon::S2_storeri_io));
        MI->getOperand(2).setReg(VirtReg);
        MI->getOperand(2).setIsKill();

      } else if (Opc == Hexagon::LDriw_pred) {
        // DstReg = LDriw_pred FI, 0
        MachineOperand &M0 = MI->getOperand(0);
        if (M0.isDead()) {
          MBB->erase(MII);
          continue;
        }

        unsigned VirtReg = MRI.createVirtualRegister(&Hexagon::IntRegsRegClass);
        unsigned DestReg = MI->getOperand(0).getReg();

        assert(MI->getOperand(1).isFI() && "Expect a frame index");
        assert(Hexagon::PredRegsRegClass.contains(DestReg) &&
               "Not a predicate register");

        // Change instruction to L2_loadri_io.
        //   VirtReg = L2_loadri_io FI, 0
        MI->setDesc(HII.get(Hexagon::L2_loadri_io));
        MI->getOperand(0).setReg(VirtReg);

        // Insert transfer to general purpose register.
        //   DestReg = C2_tfrrp VirtReg
        const MCInstrDesc &D = HII.get(Hexagon::C2_tfrrp);
        BuildMI(*MBB, std::next(MII), MI->getDebugLoc(), D, DestReg)
          .addReg(VirtReg, getKillRegState(true));
        HasReplacedPseudoInst = true;
      }
    }
  }
  return HasReplacedPseudoInst;
}

void HexagonFrameLowering::processFunctionBeforeCalleeSavedScan(
      MachineFunction &MF, RegScavenger* RS) const {
  auto &HST = static_cast<const HexagonSubtarget&>(MF.getSubtarget());
  auto &HRI = *HST.getRegisterInfo();

  bool HasEHReturn = MF.getInfo<HexagonMachineFunctionInfo>()->hasEHReturn();

  // If we have a function containing __builtin_eh_return we want to spill and
  // restore all callee saved registers. Pretend that they are used.
  if (HasEHReturn) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    for (const uint16_t *CSRegs = HRI.getCalleeSavedRegs(&MF); *CSRegs;
         ++CSRegs)
      if (!MRI.isPhysRegUsed(*CSRegs))
        MRI.setPhysRegUsed(*CSRegs);
  }

  const TargetRegisterClass &RC = Hexagon::IntRegsRegClass;

  // Replace predicate register pseudo spill code.
  bool HasReplacedPseudoInst = replacePredRegPseudoSpillCode(MF);

  // We need to reserve a a spill slot if scavenging could potentially require
  // spilling a scavenged register.
  if (HasReplacedPseudoInst && needToReserveScavengingSpillSlots(MF, HRI)) {
    MachineFrameInfo *MFI = MF.getFrameInfo();
    for (int i=0; i < NumberScavengerSlots; i++)
      RS->addScavengingFrameIndex(
        MFI->CreateSpillStackObject(RC.getSize(), RC.getAlignment()));
  }

  bool ShouldOptimizeForSize = shouldOptimizeForSize(MF);

  // Do nothing if we are not optimizing for size.
  if (!ShouldOptimizeForSize || DisableProEpilogCodeMerge)
    return;

  // Adjust the used callee save registers to be a multiple of two as there are
  // only library functions for them.
  adjustForCalleeSavedRegsSpillCall(MF);
}

bool HexagonFrameLowering::assignCalleeSavedSpillSlots(MachineFunction &MF,
      const TargetRegisterInfo *TRI, std::vector<CalleeSavedInfo> &CSI) const {
  const Function &F = *MF.getFunction();
  DEBUG(dbgs() << LLVM_FUNCTION_NAME << " on " << F.getName() << '\n');
  MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned NR = TRI->getNumRegs();
  BitVector SRegs(NR);

  // Generate a set of unique, callee-saved registers (SRegs), where each
  // register in the set is maximal in terms of sub-/super-register relation,
  // i.e. for each R in SRegs, no proper super-register of R is also in SRegs.

  // 1a. For each callee-saved register, add that register to SRegs.
  DEBUG(dbgs() << "Initial CS registers: {");
  for (unsigned i = 0, n = CSI.size(); i < n; ++i) {
    unsigned R = CSI[i].getReg();
    DEBUG(dbgs() << ' ' << PrintReg(R, TRI));
    for (MCSubRegIterator SR(R, TRI, true); SR.isValid(); ++SR)
      SRegs[*SR] = true;
  }
  DEBUG(dbgs() << " }\n");

  // 1b. Collect all registers that have at least one sub-registers in SRegs.
  BitVector TmpSup(NR);
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    for (MCSuperRegIterator SR(R, TRI); SR.isValid(); ++SR)
      TmpSup[*SR] = true;
  }
  // 1c. Out of all the super-registers found in 1b, add to SRegs those of
  // them, whose all sub-registers are already in SReg, or will be added to
  // SReg (i.e. are in TmpSup).
  // At the end of this step, the following will be true:
  //   R \in SRegs => (if S is sub-register of R, then S \in SRegs)
  //   S1..Sn = { sub-registers of R }, and S1..Sn \in SRegs => R \in SRegs
  for (int x = TmpSup.find_first(); x >= 0; x = TmpSup.find_next(x)) {
    unsigned R = x;
    bool AllSub = true;
    for (MCSubRegIterator SR(R, TRI); SR.isValid(); ++SR) {
      if (SRegs[*SR] || TmpSup[*SR])
        continue;
      AllSub = false;
      break;
    }
    // On Hexagon, a super-register is covered by its sub-registers.
    if (AllSub)
      SRegs[R] = true;
  }

  // 2. For each reserved register, remove that register and all of its
  // sub- and super-registers from SRegs.
  BitVector Reserved = TRI->getReservedRegs(MF);
  for (int x = Reserved.find_first(); x >= 0; x = Reserved.find_next(x)) {
    unsigned R = x;
    for (MCSubRegIterator SR(R, TRI, true); SR.isValid(); ++SR)
      SRegs[*SR] = false;
    for (MCSuperRegIterator SR(R, TRI); SR.isValid(); ++SR)
      SRegs[*SR] = false;
  }

  // 3. For each register R in SRegs, if any super-register of R is in SRegs,
  // remove R from SRegs.
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    for (MCSuperRegIterator SR(R, TRI); SR.isValid(); ++SR) {
      if (!SRegs[*SR])
        continue;
      SRegs[R] = false;
      break;
    }
  }

  // Now, for each register that has a fixed stack slot, create the stack
  // object for it.
  CSI.clear();

  typedef TargetFrameLowering::SpillSlot SpillSlot;
  unsigned NumFixed;
  int MinOffset = 0;  // CS offsets are negative.
  const SpillSlot *FixedSlots = getCalleeSavedSpillSlots(NumFixed);
  for (const SpillSlot *S = FixedSlots; S != FixedSlots+NumFixed; ++S) {
    if (!SRegs[S->Reg])
      continue;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(S->Reg);
    int FI = MFI->CreateFixedSpillStackObject(RC->getSize(), S->Offset);
    MinOffset = std::min(MinOffset, S->Offset);
    CSI.push_back(CalleeSavedInfo(S->Reg, FI));
    SRegs[S->Reg] = false;
  }

  // There can be some registers that don't have fixed slots. For example,
  // we need to store R0-R3 in functions with exception handling. For each
  // such register, create a non-fixed stack object.
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(R);
    int Off = MinOffset - RC->getSize();
    unsigned Align = std::min(RC->getAlignment(), getStackAlignment());
    assert(isPowerOf2_32(Align));
    Off &= -Align;
    int FI = MFI->CreateFixedSpillStackObject(RC->getSize(), Off);
    MinOffset = std::min(MinOffset, Off);
    CSI.push_back(CalleeSavedInfo(R, FI));
    SRegs[R] = false;
  }

  DEBUG({
    dbgs() << "CS information: {";
    for (unsigned i = 0, n = CSI.size(); i < n; ++i) {
      int FI = CSI[i].getFrameIdx();
      int Off = MFI->getObjectOffset(FI);
      dbgs() << ' ' << PrintReg(CSI[i].getReg(), TRI) << ":fi#" << FI << ":sp";
      if (Off >= 0)
        dbgs() << '+';
      dbgs() << Off;
    }
    dbgs() << " }\n";
  });

#ifndef NDEBUG
  // Verify that all registers were handled.
  bool MissedReg = false;
  for (int x = SRegs.find_first(); x >= 0; x = SRegs.find_next(x)) {
    unsigned R = x;
    dbgs() << PrintReg(R, TRI) << ' ';
    MissedReg = true;
  }
  if (MissedReg)
    llvm_unreachable("...there are unhandled callee-saved registers!");
#endif

  return true;
}

void HexagonFrameLowering::expandAlloca(MachineInstr *AI,
      const HexagonInstrInfo &TII, unsigned SP, unsigned CF) const {
  MachineBasicBlock &MB = *AI->getParent();
  DebugLoc DL = AI->getDebugLoc();
  unsigned A = AI->getOperand(2).getImm();

  // Have
  //    Rd  = alloca Rs, #A
  //
  // If Rs and Rd are different registers, use this sequence:
  //    Rd  = sub(r29, Rs)
  //    r29 = sub(r29, Rs)
  //    Rd  = and(Rd, #-A)    ; if necessary
  //    r29 = and(r29, #-A)   ; if necessary
  //    Rd  = add(Rd, #CF)    ; CF size aligned to at most A
  // otherwise, do
  //    Rd  = sub(r29, Rs)
  //    Rd  = and(Rd, #-A)    ; if necessary
  //    r29 = Rd
  //    Rd  = add(Rd, #CF)    ; CF size aligned to at most A

  MachineOperand &RdOp = AI->getOperand(0);
  MachineOperand &RsOp = AI->getOperand(1);
  unsigned Rd = RdOp.getReg(), Rs = RsOp.getReg();

  // Rd = sub(r29, Rs)
  BuildMI(MB, AI, DL, TII.get(Hexagon::A2_sub), Rd)
      .addReg(SP)
      .addReg(Rs);
  if (Rs != Rd) {
    // r29 = sub(r29, Rs)
    BuildMI(MB, AI, DL, TII.get(Hexagon::A2_sub), SP)
        .addReg(SP)
        .addReg(Rs);
  }
  if (A > 8) {
    // Rd  = and(Rd, #-A)
    BuildMI(MB, AI, DL, TII.get(Hexagon::A2_andir), Rd)
        .addReg(Rd)
        .addImm(-int64_t(A));
    if (Rs != Rd)
      BuildMI(MB, AI, DL, TII.get(Hexagon::A2_andir), SP)
          .addReg(SP)
          .addImm(-int64_t(A));
  }
  if (Rs == Rd) {
    // r29 = Rd
    BuildMI(MB, AI, DL, TII.get(TargetOpcode::COPY), SP)
        .addReg(Rd);
  }
  if (CF > 0) {
    // Rd = add(Rd, #CF)
    BuildMI(MB, AI, DL, TII.get(Hexagon::A2_addi), Rd)
        .addReg(Rd)
        .addImm(CF);
  }
}

bool HexagonFrameLowering::needsAligna(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  if (!MFI->hasVarSizedObjects())
    return false;
  unsigned MaxA = MFI->getMaxAlignment();
  if (MaxA <= getStackAlignment())
    return false;
  return true;
}

MachineInstr *HexagonFrameLowering::getAlignaInstr(MachineFunction &MF) const {
  for (auto &B : MF)
    for (auto &I : B)
      if (I.getOpcode() == Hexagon::ALIGNA)
        return &I;
  return nullptr;
}

