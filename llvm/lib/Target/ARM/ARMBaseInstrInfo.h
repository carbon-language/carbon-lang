//===- ARMBaseInstrInfo.h - ARM Base Instruction Information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Base ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMBASEINSTRUCTIONINFO_H
#define ARMBASEINSTRUCTIONINFO_H

#include "ARM.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseRegisterInfo;

/// ARMII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace ARMII {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction Flags.

    //===------------------------------------------------------------------===//
    // This four-bit field describes the addressing mode used.

    AddrModeMask  = 0xf,
    AddrModeNone    = 0,
    AddrMode1       = 1,
    AddrMode2       = 2,
    AddrMode3       = 3,
    AddrMode4       = 4,
    AddrMode5       = 5,
    AddrMode6       = 6,
    AddrModeT1_1    = 7,
    AddrModeT1_2    = 8,
    AddrModeT1_4    = 9,
    AddrModeT1_s    = 10, // i8 * 4 for pc and sp relative data
    AddrModeT2_i12  = 11,
    AddrModeT2_i8   = 12,
    AddrModeT2_so   = 13,
    AddrModeT2_pc   = 14, // +/- i12 for pc relative data
    AddrModeT2_i8s4 = 15, // i8 * 4

    // Size* - Flags to keep track of the size of an instruction.
    SizeShift     = 4,
    SizeMask      = 7 << SizeShift,
    SizeSpecial   = 1,   // 0 byte pseudo or special case.
    Size8Bytes    = 2,
    Size4Bytes    = 3,
    Size2Bytes    = 4,

    // IndexMode - Unindex, pre-indexed, or post-indexed are valid for load
    // and store ops only.  Generic "updating" flag is used for ld/st multiple.
    IndexModeShift = 7,
    IndexModeMask  = 3 << IndexModeShift,
    IndexModePre   = 1,
    IndexModePost  = 2,
    IndexModeUpd   = 3,

    //===------------------------------------------------------------------===//
    // Instruction encoding formats.
    //
    FormShift     = 9,
    FormMask      = 0x3f << FormShift,

    // Pseudo instructions
    Pseudo        = 0  << FormShift,

    // Multiply instructions
    MulFrm        = 1  << FormShift,

    // Branch instructions
    BrFrm         = 2  << FormShift,
    BrMiscFrm     = 3  << FormShift,

    // Data Processing instructions
    DPFrm         = 4  << FormShift,
    DPSoRegFrm    = 5  << FormShift,

    // Load and Store
    LdFrm         = 6  << FormShift,
    StFrm         = 7  << FormShift,
    LdMiscFrm     = 8  << FormShift,
    StMiscFrm     = 9  << FormShift,
    LdStMulFrm    = 10 << FormShift,

    LdStExFrm     = 11 << FormShift,

    // Miscellaneous arithmetic instructions
    ArithMiscFrm  = 12 << FormShift,

    // Extend instructions
    ExtFrm        = 13 << FormShift,

    // VFP formats
    VFPUnaryFrm   = 14 << FormShift,
    VFPBinaryFrm  = 15 << FormShift,
    VFPConv1Frm   = 16 << FormShift,
    VFPConv2Frm   = 17 << FormShift,
    VFPConv3Frm   = 18 << FormShift,
    VFPConv4Frm   = 19 << FormShift,
    VFPConv5Frm   = 20 << FormShift,
    VFPLdStFrm    = 21 << FormShift,
    VFPLdStMulFrm = 22 << FormShift,
    VFPMiscFrm    = 23 << FormShift,

    // Thumb format
    ThumbFrm      = 24 << FormShift,

    // Miscelleaneous format
    MiscFrm       = 25 << FormShift,

    // NEON formats
    NGetLnFrm     = 26 << FormShift,
    NSetLnFrm     = 27 << FormShift,
    NDupFrm       = 28 << FormShift,
    NLdStFrm      = 29 << FormShift,
    N1RegModImmFrm= 30 << FormShift,
    N2RegFrm      = 31 << FormShift,
    NVCVTFrm      = 32 << FormShift,
    NVDupLnFrm    = 33 << FormShift,
    N2RegVShLFrm  = 34 << FormShift,
    N2RegVShRFrm  = 35 << FormShift,
    N3RegFrm      = 36 << FormShift,
    N3RegVShFrm   = 37 << FormShift,
    NVExtFrm      = 38 << FormShift,
    NVMulSLFrm    = 39 << FormShift,
    NVTBLFrm      = 40 << FormShift,

    //===------------------------------------------------------------------===//
    // Misc flags.

    // UnaryDP - Indicates this is a unary data processing instruction, i.e.
    // it doesn't have a Rn operand.
    UnaryDP       = 1 << 15,

    // Xform16Bit - Indicates this Thumb2 instruction may be transformed into
    // a 16-bit Thumb instruction if certain conditions are met.
    Xform16Bit    = 1 << 16,

    //===------------------------------------------------------------------===//
    // Code domain.
    DomainShift   = 17,
    DomainMask    = 3 << DomainShift,
    DomainGeneral = 0 << DomainShift,
    DomainVFP     = 1 << DomainShift,
    DomainNEON    = 2 << DomainShift,

    //===------------------------------------------------------------------===//
    // Field shifts - such shifts are used to set field while generating
    // machine instructions.
    M_BitShift     = 5,
    ShiftImmShift  = 5,
    ShiftShift     = 7,
    N_BitShift     = 7,
    ImmHiShift     = 8,
    SoRotImmShift  = 8,
    RegRsShift     = 8,
    ExtRotImmShift = 10,
    RegRdLoShift   = 12,
    RegRdShift     = 12,
    RegRdHiShift   = 16,
    RegRnShift     = 16,
    S_BitShift     = 20,
    W_BitShift     = 21,
    AM3_I_BitShift = 22,
    D_BitShift     = 22,
    U_BitShift     = 23,
    P_BitShift     = 24,
    I_BitShift     = 25,
    CondShift      = 28
  };

  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // ARM Specific MachineOperand flags.

    MO_NO_FLAG,

    /// MO_LO16 - On a symbol operand, this represents a relocation containing
    /// lower 16 bit of the address. Used only via movw instruction.
    MO_LO16,

    /// MO_HI16 - On a symbol operand, this represents a relocation containing
    /// higher 16 bit of the address. Used only via movt instruction.
    MO_HI16
  };
}

class ARMBaseInstrInfo : public TargetInstrInfoImpl {
  const ARMSubtarget &Subtarget;
protected:
  // Can be only subclassed.
  explicit ARMBaseInstrInfo(const ARMSubtarget &STI);
public:
  // Return the non-pre/post incrementing version of 'Opc'. Return 0
  // if there is not such an opcode.
  virtual unsigned getUnindexedOpcode(unsigned Opc) const =0;

  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  virtual const ARMBaseRegisterInfo &getRegisterInfo() const =0;
  const ARMSubtarget &getSubtarget() const { return Subtarget; }

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;

  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify = false) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;

  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  // Predication support.
  bool isPredicated(const MachineInstr *MI) const {
    int PIdx = MI->findFirstPredOperandIdx();
    return PIdx != -1 && MI->getOperand(PIdx).getImm() != ARMCC::AL;
  }

  ARMCC::CondCodes getPredicate(const MachineInstr *MI) const {
    int PIdx = MI->findFirstPredOperandIdx();
    return PIdx != -1 ? (ARMCC::CondCodes)MI->getOperand(PIdx).getImm()
                      : ARMCC::AL;
  }

  virtual
  bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;

  virtual
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const;

  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const;

  virtual bool isPredicable(MachineInstr *MI) const;

  /// GetInstSize - Returns the size of the specified MachineInstr.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr* MI) const;

  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const;
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const;

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;

  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const;

  virtual MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF,
                                                 int FrameIx,
                                                 uint64_t Offset,
                                                 const MDNode *MDPtr,
                                                 DebugLoc DL) const;

  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg, unsigned SubIdx,
                             const MachineInstr *Orig,
                             const TargetRegisterInfo &TRI) const;

  MachineInstr *duplicate(MachineInstr *Orig, MachineFunction &MF) const;

  virtual bool produceSameValue(const MachineInstr *MI0,
                                const MachineInstr *MI1) const;

  /// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler to
  /// determine if two loads are loading from the same base address. It should
  /// only return true if the base pointers are the same and the only
  /// differences between the two addresses is the offset. It also returns the
  /// offsets by reference.
  virtual bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                       int64_t &Offset1, int64_t &Offset2)const;

  /// shouldScheduleLoadsNear - This is a used by the pre-regalloc scheduler to
  /// determine (in conjuction with areLoadsFromSameBasePtr) if two loads should
  /// be scheduled togther. On some targets if two loads are loading from
  /// addresses in the same cache line, it's better if they are scheduled
  /// together. This function takes two integers that represent the load offsets
  /// from the common base address. It returns true if it decides it's desirable
  /// to schedule the two loads together. "NumLoads" is the number of loads that
  /// have already been scheduled after Load1.
  virtual bool shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                       int64_t Offset1, int64_t Offset2,
                                       unsigned NumLoads) const;

  virtual bool isSchedulingBoundary(const MachineInstr *MI,
                                    const MachineBasicBlock *MBB,
                                    const MachineFunction &MF) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &MBB,
                                   unsigned NumInstrs) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &TMBB,unsigned NumT,
                                   MachineBasicBlock &FMBB,unsigned NumF) const;

  virtual bool isProfitableToDupForIfCvt(MachineBasicBlock &MBB,
                                         unsigned NumInstrs) const {
    return NumInstrs && NumInstrs == 1;
  }

  /// AnalyzeCompare - For a comparison instruction, return the source register
  /// in SrcReg and the value it compares against in CmpValue. Return true if
  /// the comparison instruction can be analyzed.
  virtual bool AnalyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                              int &CmpValue) const;

  /// ConvertToSetZeroFlag - Convert the instruction to set the zero flag so
  /// that we can remove a "comparison with zero".
  virtual bool ConvertToSetZeroFlag(MachineInstr *Instr,
                                    MachineInstr *CmpInstr) const;
};

static inline
const MachineInstrBuilder &AddDefaultPred(const MachineInstrBuilder &MIB) {
  return MIB.addImm((int64_t)ARMCC::AL).addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultCC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultT1CC(const MachineInstrBuilder &MIB,
                                          bool isDead = false) {
  return MIB.addReg(ARM::CPSR, getDefRegState(true) | getDeadRegState(isDead));
}

static inline
const MachineInstrBuilder &AddNoT1CC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

static inline
bool isUncondBranchOpcode(int Opc) {
  return Opc == ARM::B || Opc == ARM::tB || Opc == ARM::t2B;
}

static inline
bool isCondBranchOpcode(int Opc) {
  return Opc == ARM::Bcc || Opc == ARM::tBcc || Opc == ARM::t2Bcc;
}

static inline
bool isJumpTableBranchOpcode(int Opc) {
  return Opc == ARM::BR_JTr || Opc == ARM::BR_JTm || Opc == ARM::BR_JTadd ||
    Opc == ARM::tBR_JTr || Opc == ARM::t2BR_JT;
}

static inline
bool isIndirectBranchOpcode(int Opc) {
  return Opc == ARM::BRIND || Opc == ARM::MOVPCRX || Opc == ARM::tBRIND;
}

/// getInstrPredicate - If instruction is predicated, returns its predicate
/// condition, otherwise returns AL. It also returns the condition code
/// register by reference.
ARMCC::CondCodes getInstrPredicate(const MachineInstr *MI, unsigned &PredReg);

int getMatchingCondBranchOpcode(int Opc);

/// emitARMRegPlusImmediate / emitT2RegPlusImmediate - Emits a series of
/// instructions to materializea destreg = basereg + immediate in ARM / Thumb2
/// code.
void emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                             unsigned DestReg, unsigned BaseReg, int NumBytes,
                             ARMCC::CondCodes Pred, unsigned PredReg,
                             const ARMBaseInstrInfo &TII);

void emitT2RegPlusImmediate(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                            unsigned DestReg, unsigned BaseReg, int NumBytes,
                            ARMCC::CondCodes Pred, unsigned PredReg,
                            const ARMBaseInstrInfo &TII);


/// rewriteARMFrameIndex / rewriteT2FrameIndex -
/// Rewrite MI to access 'Offset' bytes from the FP. Return false if the
/// offset could not be handled directly in MI, and return the left-over
/// portion by reference.
bool rewriteARMFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                          unsigned FrameReg, int &Offset,
                          const ARMBaseInstrInfo &TII);

bool rewriteT2FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                         unsigned FrameReg, int &Offset,
                         const ARMBaseInstrInfo &TII);

} // End llvm namespace

#endif
