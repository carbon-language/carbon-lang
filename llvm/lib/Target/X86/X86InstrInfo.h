//===- X86InstrInfo.h - X86 Instruction Information ------------*- C++ -*- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class X86RegisterInfo;
  class X86TargetMachine;

namespace X86 {
  // X86 specific condition code. These correspond to X86_*_COND in
  // X86InstrInfo.td. They must be kept in synch.
  enum CondCode {
    COND_A  = 0,
    COND_AE = 1,
    COND_B  = 2,
    COND_BE = 3,
    COND_E  = 4,
    COND_G  = 5,
    COND_GE = 6,
    COND_L  = 7,
    COND_LE = 8,
    COND_NE = 9,
    COND_NO = 10,
    COND_NP = 11,
    COND_NS = 12,
    COND_O  = 13,
    COND_P  = 14,
    COND_S  = 15,

    // Artificial condition codes. These are used by AnalyzeBranch
    // to indicate a block terminated with two conditional branches to
    // the same location. This occurs in code using FCMP_OEQ or FCMP_UNE,
    // which can't be represented on x86 with a single condition. These
    // are never used in MachineInstrs.
    COND_NE_OR_P,
    COND_NP_OR_E,

    COND_INVALID
  };
    
  // Turn condition code into conditional branch opcode.
  unsigned GetCondBranchFromCond(CondCode CC);
  
  /// GetOppositeBranchCondition - Return the inverse of the specified cond,
  /// e.g. turning COND_E to COND_NE.
  CondCode GetOppositeBranchCondition(X86::CondCode CC);

}
  
/// X86II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace X86II {
  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // X86 Specific MachineOperand flags.
    
    MO_NO_FLAG,
    
    /// MO_GOT_ABSOLUTE_ADDRESS - On a symbol operand, this represents a
    /// relocation of:
    ///    SYMBOL_LABEL + [. - PICBASELABEL]
    MO_GOT_ABSOLUTE_ADDRESS,
    
    /// MO_PIC_BASE_OFFSET - On a symbol operand this indicates that the
    /// immediate should get the value of the symbol minus the PIC base label:
    ///    SYMBOL_LABEL - PICBASELABEL
    MO_PIC_BASE_OFFSET,

    /// MO_GOT - On a symbol operand this indicates that the immediate is the
    /// offset to the GOT entry for the symbol name from the base of the GOT.
    ///
    /// See the X86-64 ELF ABI supplement for more details. 
    ///    SYMBOL_LABEL @GOT
    MO_GOT,
    
    /// MO_GOTOFF - On a symbol operand this indicates that the immediate is
    /// the offset to the location of the symbol name from the base of the GOT. 
    ///
    /// See the X86-64 ELF ABI supplement for more details. 
    ///    SYMBOL_LABEL @GOTOFF
    MO_GOTOFF,
    
    /// MO_GOTPCREL - On a symbol operand this indicates that the immediate is
    /// offset to the GOT entry for the symbol name from the current code
    /// location. 
    ///
    /// See the X86-64 ELF ABI supplement for more details. 
    ///    SYMBOL_LABEL @GOTPCREL
    MO_GOTPCREL,
    
    /// MO_PLT - On a symbol operand this indicates that the immediate is
    /// offset to the PLT entry of symbol name from the current code location. 
    ///
    /// See the X86-64 ELF ABI supplement for more details. 
    ///    SYMBOL_LABEL @PLT
    MO_PLT,
    
    /// MO_TLSGD - On a symbol operand this indicates that the immediate is
    /// some TLS offset.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details. 
    ///    SYMBOL_LABEL @TLSGD
    MO_TLSGD,
    
    /// MO_GOTTPOFF - On a symbol operand this indicates that the immediate is
    /// some TLS offset.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details. 
    ///    SYMBOL_LABEL @GOTTPOFF
    MO_GOTTPOFF,
   
    /// MO_INDNTPOFF - On a symbol operand this indicates that the immediate is
    /// some TLS offset.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details. 
    ///    SYMBOL_LABEL @INDNTPOFF
    MO_INDNTPOFF,
    
    /// MO_TPOFF - On a symbol operand this indicates that the immediate is
    /// some TLS offset.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details. 
    ///    SYMBOL_LABEL @TPOFF
    MO_TPOFF,
    
    /// MO_NTPOFF - On a symbol operand this indicates that the immediate is
    /// some TLS offset.
    ///
    /// See 'ELF Handling for Thread-Local Storage' for more details. 
    ///    SYMBOL_LABEL @NTPOFF
    MO_NTPOFF,
    
    /// MO_DLLIMPORT - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "__imp_FOO" symbol.  This is used for
    /// dllimport linkage on windows.
    MO_DLLIMPORT,
    
    /// MO_DARWIN_STUB - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "FOO$stub" symbol.  This is used for calls
    /// and jumps to external functions on Tiger and before.
    MO_DARWIN_STUB,
    
    /// MO_DARWIN_NONLAZY - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "FOO$non_lazy_ptr" symbol, which is a
    /// non-PIC-base-relative reference to a non-hidden dyld lazy pointer stub.
    MO_DARWIN_NONLAZY,

    /// MO_DARWIN_NONLAZY_PIC_BASE - On a symbol operand "FOO", this indicates
    /// that the reference is actually to "FOO$non_lazy_ptr - PICBASE", which is
    /// a PIC-base-relative reference to a non-hidden dyld lazy pointer stub.
    MO_DARWIN_NONLAZY_PIC_BASE,
    
    /// MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE - On a symbol operand "FOO", this
    /// indicates that the reference is actually to "FOO$non_lazy_ptr -PICBASE",
    /// which is a PIC-base-relative reference to a hidden dyld lazy pointer
    /// stub.
    MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE
  };
}

/// isGlobalStubReference - Return true if the specified TargetFlag operand is
/// a reference to a stub for a global, not the global itself.
inline static bool isGlobalStubReference(unsigned char TargetFlag) {
  switch (TargetFlag) {
  case X86II::MO_DLLIMPORT: // dllimport stub.
  case X86II::MO_GOTPCREL:  // rip-relative GOT reference.
  case X86II::MO_GOT:       // normal GOT reference.
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:        // Normal $non_lazy_ptr ref.
  case X86II::MO_DARWIN_NONLAZY:                 // Normal $non_lazy_ptr ref.
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: // Hidden $non_lazy_ptr ref.
    return true;
  default:
    return false;
  }
}

/// isGlobalRelativeToPICBase - Return true if the specified global value
/// reference is relative to a 32-bit PIC base (X86ISD::GlobalBaseReg).  If this
/// is true, the addressing mode has the PIC base register added in (e.g. EBX).
inline static bool isGlobalRelativeToPICBase(unsigned char TargetFlag) {
  switch (TargetFlag) {
  case X86II::MO_GOTOFF:                         // isPICStyleGOT: local global.
  case X86II::MO_GOT:                            // isPICStyleGOT: other global.
  case X86II::MO_PIC_BASE_OFFSET:                // Darwin local global.
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:        // Darwin/32 external global.
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: // Darwin/32 hidden global.
    return true;
  default:
    return false;
  }
}
 
/// X86II - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace X86II {
  enum {
    //===------------------------------------------------------------------===//
    // Instruction encodings.  These are the standard/most common forms for X86
    // instructions.
    //

    // PseudoFrm - This represents an instruction that is a pseudo instruction
    // or one that has not been implemented yet.  It is illegal to code generate
    // it, but tolerated for intermediate implementation stages.
    Pseudo         = 0,

    /// Raw - This form is for instructions that don't have any operands, so
    /// they are just a fixed opcode value, like 'leave'.
    RawFrm         = 1,

    /// AddRegFrm - This form is used for instructions like 'push r32' that have
    /// their one register operand added to their opcode.
    AddRegFrm      = 2,

    /// MRMDestReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is a register.
    ///
    MRMDestReg     = 3,

    /// MRMDestMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a destination, which in this case is memory.
    ///
    MRMDestMem     = 4,

    /// MRMSrcReg - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is a register.
    ///
    MRMSrcReg      = 5,

    /// MRMSrcMem - This form is used for instructions that use the Mod/RM byte
    /// to specify a source, which in this case is memory.
    ///
    MRMSrcMem      = 6,

    /// MRM[0-7][rm] - These forms are used to represent instructions that use
    /// a Mod/RM byte, and use the middle field to hold extended opcode
    /// information.  In the intel manual these are represented as /0, /1, ...
    ///

    // First, instructions that operate on a register r/m operand...
    MRM0r = 16,  MRM1r = 17,  MRM2r = 18,  MRM3r = 19, // Format /0 /1 /2 /3
    MRM4r = 20,  MRM5r = 21,  MRM6r = 22,  MRM7r = 23, // Format /4 /5 /6 /7

    // Next, instructions that operate on a memory r/m operand...
    MRM0m = 24,  MRM1m = 25,  MRM2m = 26,  MRM3m = 27, // Format /0 /1 /2 /3
    MRM4m = 28,  MRM5m = 29,  MRM6m = 30,  MRM7m = 31, // Format /4 /5 /6 /7

    // MRMInitReg - This form is used for instructions whose source and
    // destinations are the same register.
    MRMInitReg = 32,

    FormMask       = 63,

    //===------------------------------------------------------------------===//
    // Actual flags...

    // OpSize - Set if this instruction requires an operand size prefix (0x66),
    // which most often indicates that the instruction operates on 16 bit data
    // instead of 32 bit data.
    OpSize      = 1 << 6,

    // AsSize - Set if this instruction requires an operand size prefix (0x67),
    // which most often indicates that the instruction address 16 bit address
    // instead of 32 bit address (or 32 bit address in 64 bit mode).
    AdSize      = 1 << 7,

    //===------------------------------------------------------------------===//
    // Op0Mask - There are several prefix bytes that are used to form two byte
    // opcodes.  These are currently 0x0F, 0xF3, and 0xD8-0xDF.  This mask is
    // used to obtain the setting of this field.  If no bits in this field is
    // set, there is no prefix byte for obtaining a multibyte opcode.
    //
    Op0Shift    = 8,
    Op0Mask     = 0xF << Op0Shift,

    // TB - TwoByte - Set if this instruction has a two byte opcode, which
    // starts with a 0x0F byte before the real opcode.
    TB          = 1 << Op0Shift,

    // REP - The 0xF3 prefix byte indicating repetition of the following
    // instruction.
    REP         = 2 << Op0Shift,

    // D8-DF - These escape opcodes are used by the floating point unit.  These
    // values must remain sequential.
    D8 = 3 << Op0Shift,   D9 = 4 << Op0Shift,
    DA = 5 << Op0Shift,   DB = 6 << Op0Shift,
    DC = 7 << Op0Shift,   DD = 8 << Op0Shift,
    DE = 9 << Op0Shift,   DF = 10 << Op0Shift,

    // XS, XD - These prefix codes are for single and double precision scalar
    // floating point operations performed in the SSE registers.
    XD = 11 << Op0Shift,  XS = 12 << Op0Shift,

    // T8, TA - Prefix after the 0x0F prefix.
    T8 = 13 << Op0Shift,  TA = 14 << Op0Shift,
    
    // TF - Prefix before and after 0x0F
    TF = 15 << Op0Shift,

    //===------------------------------------------------------------------===//
    // REX_W - REX prefixes are instruction prefixes used in 64-bit mode.
    // They are used to specify GPRs and SSE registers, 64-bit operand size,
    // etc. We only cares about REX.W and REX.R bits and only the former is
    // statically determined.
    //
    REXShift    = 12,
    REX_W       = 1 << REXShift,

    //===------------------------------------------------------------------===//
    // This three-bit field describes the size of an immediate operand.  Zero is
    // unused so that we can tell if we forgot to set a value.
    ImmShift = 13,
    ImmMask  = 7 << ImmShift,
    Imm8     = 1 << ImmShift,
    Imm16    = 2 << ImmShift,
    Imm32    = 3 << ImmShift,
    Imm64    = 4 << ImmShift,

    //===------------------------------------------------------------------===//
    // FP Instruction Classification...  Zero is non-fp instruction.

    // FPTypeMask - Mask for all of the FP types...
    FPTypeShift = 16,
    FPTypeMask  = 7 << FPTypeShift,

    // NotFP - The default, set for instructions that do not use FP registers.
    NotFP      = 0 << FPTypeShift,

    // ZeroArgFP - 0 arg FP instruction which implicitly pushes ST(0), f.e. fld0
    ZeroArgFP  = 1 << FPTypeShift,

    // OneArgFP - 1 arg FP instructions which implicitly read ST(0), such as fst
    OneArgFP   = 2 << FPTypeShift,

    // OneArgFPRW - 1 arg FP instruction which implicitly read ST(0) and write a
    // result back to ST(0).  For example, fcos, fsqrt, etc.
    //
    OneArgFPRW = 3 << FPTypeShift,

    // TwoArgFP - 2 arg FP instructions which implicitly read ST(0), and an
    // explicit argument, storing the result to either ST(0) or the implicit
    // argument.  For example: fadd, fsub, fmul, etc...
    TwoArgFP   = 4 << FPTypeShift,

    // CompareFP - 2 arg FP instructions which implicitly read ST(0) and an
    // explicit argument, but have no destination.  Example: fucom, fucomi, ...
    CompareFP  = 5 << FPTypeShift,

    // CondMovFP - "2 operand" floating point conditional move instructions.
    CondMovFP  = 6 << FPTypeShift,

    // SpecialFP - Special instruction forms.  Dispatch by opcode explicitly.
    SpecialFP  = 7 << FPTypeShift,

    // Lock prefix
    LOCKShift = 19,
    LOCK = 1 << LOCKShift,

    // Segment override prefixes. Currently we just need ability to address
    // stuff in gs and fs segments.
    SegOvrShift = 20,
    SegOvrMask  = 3 << SegOvrShift,
    FS          = 1 << SegOvrShift,
    GS          = 2 << SegOvrShift,

    // Bits 22 -> 23 are unused
    OpcodeShift   = 24,
    OpcodeMask    = 0xFF << OpcodeShift
  };
  
  // getBaseOpcodeFor - This function returns the "base" X86 opcode for the
  // specified machine instruction.
  //
  static inline unsigned char getBaseOpcodeFor(unsigned TSFlags) {
    return TSFlags >> X86II::OpcodeShift;
  }
  
  /// getSizeOfImm - Decode the "size of immediate" field from the TSFlags field
  /// of the specified instruction.
  static inline unsigned getSizeOfImm(unsigned TSFlags) {
    switch (TSFlags & X86II::ImmMask) {
    default: assert(0 && "Unknown immediate size");
    case X86II::Imm8:   return 1;
    case X86II::Imm16:  return 2;
    case X86II::Imm32:  return 4;
    case X86II::Imm64:  return 8;
    }
  }    
}

const int X86AddrNumOperands = 5;

inline static bool isScale(const MachineOperand &MO) {
  return MO.isImm() &&
    (MO.getImm() == 1 || MO.getImm() == 2 ||
     MO.getImm() == 4 || MO.getImm() == 8);
}

inline static bool isLeaMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFI()) return true;
  return Op+4 <= MI->getNumOperands() &&
    MI->getOperand(Op  ).isReg() && isScale(MI->getOperand(Op+1)) &&
    MI->getOperand(Op+2).isReg() &&
    (MI->getOperand(Op+3).isImm() ||
     MI->getOperand(Op+3).isGlobal() ||
     MI->getOperand(Op+3).isCPI() ||
     MI->getOperand(Op+3).isJTI());
}

inline static bool isMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFI()) return true;
  return Op+5 <= MI->getNumOperands() &&
    MI->getOperand(Op+4).isReg() &&
    isLeaMem(MI, Op);
}

class X86InstrInfo : public TargetInstrInfoImpl {
  X86TargetMachine &TM;
  const X86RegisterInfo RI;
  
  /// RegOp2MemOpTable2Addr, RegOp2MemOpTable0, RegOp2MemOpTable1,
  /// RegOp2MemOpTable2 - Load / store folding opcode maps.
  ///
  DenseMap<unsigned*, std::pair<unsigned,unsigned> > RegOp2MemOpTable2Addr;
  DenseMap<unsigned*, std::pair<unsigned,unsigned> > RegOp2MemOpTable0;
  DenseMap<unsigned*, std::pair<unsigned,unsigned> > RegOp2MemOpTable1;
  DenseMap<unsigned*, std::pair<unsigned,unsigned> > RegOp2MemOpTable2;
  
  /// MemOp2RegOpTable - Load / store unfolding opcode map.
  ///
  DenseMap<unsigned*, std::pair<unsigned, unsigned> > MemOp2RegOpTable;
  
public:
  explicit X86InstrInfo(X86TargetMachine &tm);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const X86RegisterInfo &getRegisterInfo() const { return RI; }

  /// Return true if the instruction is a register to register move and return
  /// the source and dest operands and their sub-register indices by reference.
  virtual bool isMoveInstr(const MachineInstr &MI,
                           unsigned &SrcReg, unsigned &DstReg,
                           unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

  /// isCoalescableExtInstr - Return true if the instruction is a "coalescable"
  /// extension instruction. That is, it's like a copy where it's legal for the
  /// source to overlap the destination. e.g. X86::MOVSX64rr32. If this returns
  /// true, then it's expected the pre-extension value is available as a subreg
  /// of the result register. This also returns the sub-register index in
  /// SubIdx.
  virtual bool isCoalescableExtInstr(const MachineInstr &MI,
                                     unsigned &SrcReg, unsigned &DstReg,
                                     unsigned &SubIdx) const;

  unsigned isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  /// isLoadFromStackSlotPostFE - Check for post-frame ptr elimination
  /// stack locations as well.  This uses a heuristic so it isn't
  /// reliable for correctness.
  unsigned isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                     int &FrameIndex) const;

  /// hasLoadFromStackSlot - If the specified machine instruction has
  /// a load from a stack slot, return true along with the FrameIndex
  /// of the loaded stack slot and the machine mem operand containing
  /// the reference.  If not, return false.  Unlike
  /// isLoadFromStackSlot, this returns true for any instructions that
  /// loads from the stack.  This is a hint only and may not catch all
  /// cases.
  bool hasLoadFromStackSlot(const MachineInstr *MI,
                            const MachineMemOperand *&MMO,
                            int &FrameIndex) const;

  unsigned isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  /// isStoreToStackSlotPostFE - Check for post-frame ptr elimination
  /// stack locations as well.  This uses a heuristic so it isn't
  /// reliable for correctness.
  unsigned isStoreToStackSlotPostFE(const MachineInstr *MI,
                                    int &FrameIndex) const;

  /// hasStoreToStackSlot - If the specified machine instruction has a
  /// store to a stack slot, return true along with the FrameIndex of
  /// the loaded stack slot and the machine mem operand containing the
  /// reference.  If not, return false.  Unlike isStoreToStackSlot,
  /// this returns true for any instructions that loads from the
  /// stack.  This is a hint only and may not catch all cases.
  bool hasStoreToStackSlot(const MachineInstr *MI,
                           const MachineMemOperand *&MMO,
                           int &FrameIndex) const;

  bool isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                         AliasAnalysis *AA) const;
  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, unsigned SubIdx,
                     const MachineInstr *Orig,
                     const TargetRegisterInfo *TRI) const;

  /// convertToThreeAddress - This method must be implemented by targets that
  /// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
  /// may be able to convert a two-address instruction into a true
  /// three-address instruction on demand.  This allows the X86 target (for
  /// example) to convert ADD and SHL instructions into LEA instructions if they
  /// would require register copies due to two-addressness.
  ///
  /// This method returns a null pointer if the transformation cannot be
  /// performed, otherwise it returns the new instruction.
  ///
  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  /// commuteInstruction - We have a few instructions that must be hacked on to
  /// commute them.
  ///
  virtual MachineInstr *commuteInstruction(MachineInstr *MI, bool NewMI) const;

  // Branch analysis.
  virtual bool isUnpredicatedTerminator(const MachineInstr* MI) const;
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const;
  virtual bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const;
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const;

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              MachineInstr::mmo_iterator MMOBegin,
                              MachineInstr::mmo_iterator MMOEnd,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const;

  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               MachineInstr::mmo_iterator MMOBegin,
                               MachineInstr::mmo_iterator MMOEnd,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const;
  
  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;

  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;
  
  /// foldMemoryOperand - If this target supports it, fold a load or store of
  /// the specified stack slot into the specified machine instruction for the
  /// specified operand(s).  If this is possible, the target should perform the
  /// folding and return true, otherwise it should return false.  If it folds
  /// the instruction, it is likely that the MachineInstruction the iterator
  /// references has been changed.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const;

  /// canFoldMemoryOperand - Returns true if the specified load / store is
  /// folding is possible.
  virtual bool canFoldMemoryOperand(const MachineInstr*,
                                    const SmallVectorImpl<unsigned> &) const;

  /// unfoldMemoryOperand - Separate a single instruction which folded a load or
  /// a store or a load and a store into two or more instruction. If this is
  /// possible, returns true as well as the new instructions by reference.
  virtual bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                           unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                           SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                           SmallVectorImpl<SDNode*> &NewNodes) const;

  /// getOpcodeAfterMemoryUnfold - Returns the opcode of the would be new
  /// instruction after load / store are unfolded from an instruction of the
  /// specified opcode. It returns zero if the specified unfolding is not
  /// possible. If LoadRegIndex is non-null, it is filled in with the operand
  /// index of the operand which will hold the register holding the loaded
  /// value.
  virtual unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore,
                                      unsigned *LoadRegIndex = 0) const;
  
  /// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler
  /// to determine if two loads are loading from the same base address. It
  /// should only return true if the base pointers are the same and the
  /// only differences between the two addresses are the offset. It also returns
  /// the offsets by reference.
  virtual bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                       int64_t &Offset1, int64_t &Offset2) const;

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

  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  /// isSafeToMoveRegClassDefs - Return true if it's safe to move a machine
  /// instruction that defines the specified register class.
  bool isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const;

  static bool isX86_64NonExtLowByteReg(unsigned reg) {
    return (reg == X86::SPL || reg == X86::BPL ||
          reg == X86::SIL || reg == X86::DIL);
  }
  
  static bool isX86_64ExtendedReg(const MachineOperand &MO);
  static unsigned determineREX(const MachineInstr &MI);

  /// GetInstSize - Returns the size of the specified MachineInstr.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr *MI) const;

  /// getGlobalBaseReg - Return a virtual register initialized with the
  /// the global base register value. Output instructions required to
  /// initialize the register in the function entry block, if necessary.
  ///
  unsigned getGlobalBaseReg(MachineFunction *MF) const;

private:
  MachineInstr * convertToThreeAddressWithLEA(unsigned MIOpc,
                                              MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                     MachineInstr* MI,
                                     unsigned OpNum,
                                     const SmallVectorImpl<MachineOperand> &MOs,
                                     unsigned Size, unsigned Alignment) const;

  /// isFrameOperand - Return true and the FrameIndex if the specified
  /// operand and follow operands form a reference to the stack frame.
  bool isFrameOperand(const MachineInstr *MI, unsigned int Op,
                      int &FrameIndex) const;
};

} // End llvm namespace

#endif
