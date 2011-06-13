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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

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
    AddrModeMask  = 0x1f, // The AddrMode enums are declared in ARMBaseInfo.h

    // Size* - Flags to keep track of the size of an instruction.
    SizeShift     = 5,
    SizeMask      = 7 << SizeShift,
    SizeSpecial   = 1,   // 0 byte pseudo or special case.
    Size8Bytes    = 2,
    Size4Bytes    = 3,
    Size2Bytes    = 4,

    // IndexMode - Unindex, pre-indexed, or post-indexed are valid for load
    // and store ops only.  Generic "updating" flag is used for ld/st multiple.
    // The index mode enums are declared in ARMBaseInfo.h
    IndexModeShift = 8,
    IndexModeMask  = 3 << IndexModeShift,

    //===------------------------------------------------------------------===//
    // Instruction encoding formats.
    //
    FormShift     = 10,
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
    SatFrm        = 13 << FormShift,

    // Extend instructions
    ExtFrm        = 14 << FormShift,

    // VFP formats
    VFPUnaryFrm   = 15 << FormShift,
    VFPBinaryFrm  = 16 << FormShift,
    VFPConv1Frm   = 17 << FormShift,
    VFPConv2Frm   = 18 << FormShift,
    VFPConv3Frm   = 19 << FormShift,
    VFPConv4Frm   = 20 << FormShift,
    VFPConv5Frm   = 21 << FormShift,
    VFPLdStFrm    = 22 << FormShift,
    VFPLdStMulFrm = 23 << FormShift,
    VFPMiscFrm    = 24 << FormShift,

    // Thumb format
    ThumbFrm      = 25 << FormShift,

    // Miscelleaneous format
    MiscFrm       = 26 << FormShift,

    // NEON formats
    NGetLnFrm     = 27 << FormShift,
    NSetLnFrm     = 28 << FormShift,
    NDupFrm       = 29 << FormShift,
    NLdStFrm      = 30 << FormShift,
    N1RegModImmFrm= 31 << FormShift,
    N2RegFrm      = 32 << FormShift,
    NVCVTFrm      = 33 << FormShift,
    NVDupLnFrm    = 34 << FormShift,
    N2RegVShLFrm  = 35 << FormShift,
    N2RegVShRFrm  = 36 << FormShift,
    N3RegFrm      = 37 << FormShift,
    N3RegVShFrm   = 38 << FormShift,
    NVExtFrm      = 39 << FormShift,
    NVMulSLFrm    = 40 << FormShift,
    NVTBLFrm      = 41 << FormShift,

    //===------------------------------------------------------------------===//
    // Misc flags.

    // UnaryDP - Indicates this is a unary data processing instruction, i.e.
    // it doesn't have a Rn operand.
    UnaryDP       = 1 << 16,

    // Xform16Bit - Indicates this Thumb2 instruction may be transformed into
    // a 16-bit Thumb instruction if certain conditions are met.
    Xform16Bit    = 1 << 17,

    //===------------------------------------------------------------------===//
    // Code domain.
    DomainShift   = 18,
    DomainMask    = 7 << DomainShift,
    DomainGeneral = 0 << DomainShift,
    DomainVFP     = 1 << DomainShift,
    DomainNEON    = 2 << DomainShift,
    DomainNEONA8  = 4 << DomainShift,

    //===------------------------------------------------------------------===//
    // Field shifts - such shifts are used to set field while generating
    // machine instructions.
    //
    // FIXME: This list will need adjusting/fixing as the MC code emitter
    // takes shape and the ARMCodeEmitter.cpp bits go away.
    ShiftTypeShift = 4,

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

  ScheduleHazardRecognizer *
  CreateTargetHazardRecognizer(const TargetMachine *TM,
                               const ScheduleDAG *DAG) const;

  ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                     const ScheduleDAG *DAG) const;

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
                                const MachineInstr *MI1,
                                const MachineRegisterInfo *MRI) const;

  /// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler to
  /// determine if two loads are loading from the same base address. It should
  /// only return true if the base pointers are the same and the only
  /// differences between the two addresses is the offset. It also returns the
  /// offsets by reference.
  virtual bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                       int64_t &Offset1, int64_t &Offset2)const;

  /// shouldScheduleLoadsNear - This is a used by the pre-regalloc scheduler to
  /// determine (in conjunction with areLoadsFromSameBasePtr) if two loads
  /// should be scheduled togther. On some targets if two loads are loading from
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
                                   unsigned NumCycles, unsigned ExtraPredCycles,
                                   float Prob, float Confidence) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &TMBB,
                                   unsigned NumT, unsigned ExtraT,
                                   MachineBasicBlock &FMBB,
                                   unsigned NumF, unsigned ExtraF,
                                   float Probability, float Confidence) const;

  virtual bool isProfitableToDupForIfCvt(MachineBasicBlock &MBB,
                                         unsigned NumCycles,
                                         float Probability,
                                         float Confidence) const {
    return NumCycles == 1;
  }

  /// AnalyzeCompare - For a comparison instruction, return the source register
  /// in SrcReg and the value it compares against in CmpValue. Return true if
  /// the comparison instruction can be analyzed.
  virtual bool AnalyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                              int &CmpMask, int &CmpValue) const;

  /// OptimizeCompareInstr - Convert the instruction to set the zero flag so
  /// that we can remove a "comparison with zero".
  virtual bool OptimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg,
                                    int CmpMask, int CmpValue,
                                    const MachineRegisterInfo *MRI) const;

  /// FoldImmediate - 'Reg' is known to be defined by a move immediate
  /// instruction, try to fold the immediate into the use instruction.
  virtual bool FoldImmediate(MachineInstr *UseMI, MachineInstr *DefMI,
                             unsigned Reg, MachineRegisterInfo *MRI) const;

  virtual unsigned getNumMicroOps(const InstrItineraryData *ItinData,
                                  const MachineInstr *MI) const;

  virtual
  int getOperandLatency(const InstrItineraryData *ItinData,
                        const MachineInstr *DefMI, unsigned DefIdx,
                        const MachineInstr *UseMI, unsigned UseIdx) const;
  virtual
  int getOperandLatency(const InstrItineraryData *ItinData,
                        SDNode *DefNode, unsigned DefIdx,
                        SDNode *UseNode, unsigned UseIdx) const;
private:
  int getVLDMDefCycle(const InstrItineraryData *ItinData,
                      const TargetInstrDesc &DefTID,
                      unsigned DefClass,
                      unsigned DefIdx, unsigned DefAlign) const;
  int getLDMDefCycle(const InstrItineraryData *ItinData,
                     const TargetInstrDesc &DefTID,
                     unsigned DefClass,
                     unsigned DefIdx, unsigned DefAlign) const;
  int getVSTMUseCycle(const InstrItineraryData *ItinData,
                      const TargetInstrDesc &UseTID,
                      unsigned UseClass,
                      unsigned UseIdx, unsigned UseAlign) const;
  int getSTMUseCycle(const InstrItineraryData *ItinData,
                     const TargetInstrDesc &UseTID,
                     unsigned UseClass,
                     unsigned UseIdx, unsigned UseAlign) const;
  int getOperandLatency(const InstrItineraryData *ItinData,
                        const TargetInstrDesc &DefTID,
                        unsigned DefIdx, unsigned DefAlign,
                        const TargetInstrDesc &UseTID,
                        unsigned UseIdx, unsigned UseAlign) const;

  int getInstrLatency(const InstrItineraryData *ItinData,
                      const MachineInstr *MI, unsigned *PredCost = 0) const;

  int getInstrLatency(const InstrItineraryData *ItinData,
                      SDNode *Node) const;

  bool hasHighOperandLatency(const InstrItineraryData *ItinData,
                             const MachineRegisterInfo *MRI,
                             const MachineInstr *DefMI, unsigned DefIdx,
                             const MachineInstr *UseMI, unsigned UseIdx) const;
  bool hasLowDefLatency(const InstrItineraryData *ItinData,
                        const MachineInstr *DefMI, unsigned DefIdx) const;

private:
  /// Modeling special VFP / NEON fp MLA / MLS hazards.

  /// MLxEntryMap - Map fp MLA / MLS to the corresponding entry in the internal
  /// MLx table.
  DenseMap<unsigned, unsigned> MLxEntryMap;

  /// MLxHazardOpcodes - Set of add / sub and multiply opcodes that would cause
  /// stalls when scheduled together with fp MLA / MLS opcodes.
  SmallSet<unsigned, 16> MLxHazardOpcodes;

public:
  /// isFpMLxInstruction - Return true if the specified opcode is a fp MLA / MLS
  /// instruction.
  bool isFpMLxInstruction(unsigned Opcode) const {
    return MLxEntryMap.count(Opcode);
  }

  /// isFpMLxInstruction - This version also returns the multiply opcode and the
  /// addition / subtraction opcode to expand to. Return true for 'HasLane' for
  /// the MLX instructions with an extra lane operand.
  bool isFpMLxInstruction(unsigned Opcode, unsigned &MulOpc,
                          unsigned &AddSubOpc, bool &NegAcc,
                          bool &HasLane) const;

  /// canCauseFpMLxStall - Return true if an instruction of the specified opcode
  /// will cause stalls when scheduled after (within 4-cycle window) a fp
  /// MLA / MLS instruction.
  bool canCauseFpMLxStall(unsigned Opcode) const {
    return MLxHazardOpcodes.count(Opcode);
  }
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
  return Opc == ARM::BX || Opc == ARM::MOVPCRX || Opc == ARM::tBRIND;
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
                             const ARMBaseInstrInfo &TII, unsigned MIFlags = 0);

void emitT2RegPlusImmediate(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                            unsigned DestReg, unsigned BaseReg, int NumBytes,
                            ARMCC::CondCodes Pred, unsigned PredReg,
                            const ARMBaseInstrInfo &TII, unsigned MIFlags = 0);
void emitThumbRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                               unsigned DestReg, unsigned BaseReg,
                               int NumBytes, const TargetInstrInfo &TII,
                               const ARMBaseRegisterInfo& MRI,
                               unsigned MIFlags = 0);


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
