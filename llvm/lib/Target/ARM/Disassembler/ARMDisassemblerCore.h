//===- ARMDisassemblerCore.h - ARM disassembler helpers ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the ARM Disassembler.
//
// The first part defines the enumeration type of ARM instruction format, which
// specifies the encoding used by the instruction, as well as a helper function
// to convert the enums to printable char strings.
//
// It also contains code to represent the concepts of Builder and DisassembleFP
// to solve the problem of disassembling an ARM instr.
//
//===----------------------------------------------------------------------===//

#ifndef ARMDISASSEMBLERCORE_H
#define ARMDISASSEMBLERCORE_H

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm-c/Disassembler.h"
#include "ARMBaseInstrInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMDisassembler.h"

namespace llvm {
class MCContext;

class ARMUtils {
public:
  static const char *OpcodeName(unsigned Opcode);
};

/////////////////////////////////////////////////////
//                                                 //
//  Enums and Utilities for ARM Instruction Format //
//                                                 //
/////////////////////////////////////////////////////

#define ARM_FORMATS                   \
  ENTRY(ARM_FORMAT_PSEUDO,         0) \
  ENTRY(ARM_FORMAT_MULFRM,         1) \
  ENTRY(ARM_FORMAT_BRFRM,          2) \
  ENTRY(ARM_FORMAT_BRMISCFRM,      3) \
  ENTRY(ARM_FORMAT_DPFRM,          4) \
  ENTRY(ARM_FORMAT_DPSOREGREGFRM,     5) \
  ENTRY(ARM_FORMAT_LDFRM,          6) \
  ENTRY(ARM_FORMAT_STFRM,          7) \
  ENTRY(ARM_FORMAT_LDMISCFRM,      8) \
  ENTRY(ARM_FORMAT_STMISCFRM,      9) \
  ENTRY(ARM_FORMAT_LDSTMULFRM,    10) \
  ENTRY(ARM_FORMAT_LDSTEXFRM,     11) \
  ENTRY(ARM_FORMAT_ARITHMISCFRM,  12) \
  ENTRY(ARM_FORMAT_SATFRM,        13) \
  ENTRY(ARM_FORMAT_EXTFRM,        14) \
  ENTRY(ARM_FORMAT_VFPUNARYFRM,   15) \
  ENTRY(ARM_FORMAT_VFPBINARYFRM,  16) \
  ENTRY(ARM_FORMAT_VFPCONV1FRM,   17) \
  ENTRY(ARM_FORMAT_VFPCONV2FRM,   18) \
  ENTRY(ARM_FORMAT_VFPCONV3FRM,   19) \
  ENTRY(ARM_FORMAT_VFPCONV4FRM,   20) \
  ENTRY(ARM_FORMAT_VFPCONV5FRM,   21) \
  ENTRY(ARM_FORMAT_VFPLDSTFRM,    22) \
  ENTRY(ARM_FORMAT_VFPLDSTMULFRM, 23) \
  ENTRY(ARM_FORMAT_VFPMISCFRM,    24) \
  ENTRY(ARM_FORMAT_THUMBFRM,      25) \
  ENTRY(ARM_FORMAT_MISCFRM,       26) \
  ENTRY(ARM_FORMAT_NEONGETLNFRM,  27) \
  ENTRY(ARM_FORMAT_NEONSETLNFRM,  28) \
  ENTRY(ARM_FORMAT_NEONDUPFRM,    29) \
  ENTRY(ARM_FORMAT_NLdSt,         30) \
  ENTRY(ARM_FORMAT_N1RegModImm,   31) \
  ENTRY(ARM_FORMAT_N2Reg,         32) \
  ENTRY(ARM_FORMAT_NVCVT,         33) \
  ENTRY(ARM_FORMAT_NVecDupLn,     34) \
  ENTRY(ARM_FORMAT_N2RegVecShL,   35) \
  ENTRY(ARM_FORMAT_N2RegVecShR,   36) \
  ENTRY(ARM_FORMAT_N3Reg,         37) \
  ENTRY(ARM_FORMAT_N3RegVecSh,    38) \
  ENTRY(ARM_FORMAT_NVecExtract,   39) \
  ENTRY(ARM_FORMAT_NVecMulScalar, 40) \
  ENTRY(ARM_FORMAT_NVTBL,         41) \
  ENTRY(ARM_FORMAT_DPSOREGIMMFRM, 42)

// ARM instruction format specifies the encoding used by the instruction.
#define ENTRY(n, v) n = v,
typedef enum {
  ARM_FORMATS
  ARM_FORMAT_NA
} ARMFormat;
#undef ENTRY

// Converts enum to const char*.
static const inline char *stringForARMFormat(ARMFormat form) {
#define ENTRY(n, v) case n: return #n;
  switch(form) {
    ARM_FORMATS
  case ARM_FORMAT_NA:
  default:
    return "";
  }
#undef ENTRY
}

/// Expands on the enum definitions from ARMBaseInstrInfo.h.
/// They are being used by the disassembler implementation.
namespace ARMII {
  enum {
    NEONRegMask = 15,
    GPRRegMask = 15,
    NEON_RegRdShift = 12,
    NEON_D_BitShift = 22,
    NEON_RegRnShift = 16,
    NEON_N_BitShift = 7,
    NEON_RegRmShift = 0,
    NEON_M_BitShift = 5
  };
}

/// Utility function for extracting [From, To] bits from a uint32_t.
static inline unsigned slice(uint32_t Bits, unsigned From, unsigned To) {
  assert(From < 32 && To < 32 && From >= To);
  return (Bits >> To) & ((1 << (From - To + 1)) - 1);
}

/// Utility function for setting [From, To] bits to Val for a uint32_t.
static inline void setSlice(unsigned &Bits, unsigned From, unsigned To,
                            unsigned Val) {
  assert(From < 32 && To < 32 && From >= To);
  uint32_t Mask = ((1 << (From - To + 1)) - 1);
  Bits &= ~(Mask << To);
  Bits |= (Val & Mask) << To;
}

// Return an integer result equal to the number of bits of x that are ones.
static inline uint32_t
BitCount (uint64_t x)
{
    // c accumulates the total bits set in x
    uint32_t c;
    for (c = 0; x; ++c)
    {
        x &= x - 1; // clear the least significant bit set
    }
    return c;
}

static inline bool
BitIsSet (const uint64_t value, const uint64_t bit)
{
    return (value & (1ull << bit)) != 0;
}

static inline bool
BitIsClear (const uint64_t value, const uint64_t bit)
{
    return (value & (1ull << bit)) == 0;
}

/// Various utilities for checking the target specific flags.

/// A unary data processing instruction doesn't have an Rn operand.
static inline bool isUnaryDP(uint64_t TSFlags) {
  return (TSFlags & ARMII::UnaryDP);
}

/// A NEON Domain instruction has cond field (Inst{31-28}) as 0b1111.
static inline bool isNEONDomain(uint64_t TSFlags) {
  return (TSFlags & ARMII::DomainNEON) ||
         (TSFlags & ARMII::DomainNEONA8);
}

/// This four-bit field describes the addressing mode used.
/// See also ARMBaseInstrInfo.h.
static inline unsigned getAddrMode(uint64_t TSFlags) {
  return (TSFlags & ARMII::AddrModeMask);
}

/// {IndexModePre, IndexModePost}
/// Only valid for load and store ops.
/// See also ARMBaseInstrInfo.h.
static inline unsigned getIndexMode(uint64_t TSFlags) {
  return (TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;
}

/// Pre-/post-indexed operations define an extra $base_wb in the OutOperandList.
static inline bool isPrePostLdSt(uint64_t TSFlags) {
  return (TSFlags & ARMII::IndexModeMask) != 0;
}

// Forward declaration.
class ARMBasicMCBuilder;

// Builder Object is mostly ignored except in some Thumb disassemble functions.
typedef ARMBasicMCBuilder *BO;

/// DisassembleFP - DisassembleFP points to a function that disassembles an insn
/// and builds the MCOperand list upon disassembly.  It returns false on failure
/// or true on success.  The number of operands added is updated upon success.
typedef bool (*DisassembleFP)(MCInst &MI, unsigned Opcode, uint32_t insn,
    unsigned short NumOps, unsigned &NumOpsAdded, BO Builder);

/// CreateMCBuilder - Return an ARMBasicMCBuilder that can build up the MC
/// infrastructure of an MCInst given the Opcode and Format of the instr.
/// Return NULL if it fails to create/return a proper builder.  API clients
/// are responsible for freeing up of the allocated memory.  Cacheing can be
/// performed by the API clients to improve performance.
extern ARMBasicMCBuilder *CreateMCBuilder(unsigned Opcode, ARMFormat Format);

/// ARMBasicMCBuilder - ARMBasicMCBuilder represents an ARM MCInst builder that
/// knows how to build up the MCOperand list.
class ARMBasicMCBuilder {
  friend ARMBasicMCBuilder *CreateMCBuilder(unsigned Opcode, ARMFormat Format);
  unsigned Opcode;
  ARMFormat Format;
  unsigned short NumOps;
  DisassembleFP Disasm;
  Session *SP;
  int Err; // !=0 if the builder encounters some error condition during build.

private:
  /// Opcode, Format, and NumOperands make up an ARM Basic MCBuilder.
  ARMBasicMCBuilder(unsigned opc, ARMFormat format, unsigned short num);

public:
  ARMBasicMCBuilder(ARMBasicMCBuilder &B)
    : Opcode(B.Opcode), Format(B.Format), NumOps(B.NumOps), Disasm(B.Disasm),
      SP(B.SP), GetOpInfo(0), DisInfo(0), Ctx(0) {
    Err = 0;
  }

  virtual ~ARMBasicMCBuilder() {}

  void SetSession(Session *sp) {
    SP = sp;
  }

  void SetErr(int ErrCode) {
    Err = ErrCode;
  }

  /// DoPredicateOperands - DoPredicateOperands process the predicate operands
  /// of some Thumb instructions which come before the reglist operands.  It
  /// returns true if the two predicate operands have been processed.
  bool DoPredicateOperands(MCInst& MI, unsigned Opcode,
      uint32_t insn, unsigned short NumOpsRemaning);
  
  /// TryPredicateAndSBitModifier - TryPredicateAndSBitModifier tries to process
  /// the possible Predicate and SBitModifier, to build the remaining MCOperand
  /// constituents.
  bool TryPredicateAndSBitModifier(MCInst& MI, unsigned Opcode,
      uint32_t insn, unsigned short NumOpsRemaning);

  /// InITBlock - InITBlock returns true if we are inside an IT block.
  bool InITBlock() {
    if (SP)
      return SP->ITCounter > 0;

    return false;
  }

  /// Build - Build delegates to BuildIt to perform the heavy liftling.  After
  /// that, it invokes RunBuildAfterHook where some housekeepings can be done.
  virtual bool Build(MCInst &MI, uint32_t insn) {
    bool Status = BuildIt(MI, insn);
    return RunBuildAfterHook(Status, MI, insn);
  }

  /// BuildIt - BuildIt performs the build step for this ARM Basic MC Builder.
  /// The general idea is to set the Opcode for the MCInst, followed by adding
  /// the appropriate MCOperands to the MCInst.  ARM Basic MC Builder delegates
  /// to the Format-specific disassemble function for disassembly, followed by
  /// TryPredicateAndSBitModifier() for PredicateOperand and OptionalDefOperand
  /// which follow the Dst/Src Operands.
  virtual bool BuildIt(MCInst &MI, uint32_t insn);

  /// RunBuildAfterHook - RunBuildAfterHook performs operations deemed necessary
  /// after BuildIt is finished.
  virtual bool RunBuildAfterHook(bool Status, MCInst &MI, uint32_t insn);

private:
  /// Get condition of the current IT instruction.
  unsigned GetITCond() {
    assert(SP);
    return slice(SP->ITState, 7, 4);
  }

private:
  //
  // Hooks for symbolic disassembly via the public 'C' interface.
  //
  // The function to get the symbolic information for operands.
  LLVMOpInfoCallback GetOpInfo;
  // The pointer to the block of symbolic information for above call back.
  void *DisInfo;
  // The assembly context for creating symbols and MCExprs in place of
  // immediate operands when there is symbolic information.
  MCContext *Ctx;
  // The address of the instruction being disassembled.
  uint64_t Address;

public:
  void setupBuilderForSymbolicDisassembly(LLVMOpInfoCallback getOpInfo,
                                          void *disInfo, MCContext *ctx,
                                          uint64_t address) {
    GetOpInfo = getOpInfo;
    DisInfo = disInfo;
    Ctx = ctx;
    Address = address;
  }

  uint64_t getBuilderAddress() const { return Address; }

  /// tryAddingSymbolicOperand - tryAddingSymbolicOperand trys to add a symbolic
  /// operand in place of the immediate Value in the MCInst.  The immediate
  /// Value has had any PC adjustment made by the caller.  If the getOpInfo()
  /// function was set as part of the setupBuilderForSymbolicDisassembly() call
  /// then that function is called to get any symbolic information at the
  /// builder's Address for this instrution.  If that returns non-zero then the
  /// symbolic information it returns is used to create an MCExpr and that is
  /// added as an operand to the MCInst.  This function returns true if it adds
  /// an operand to the MCInst and false otherwise.
  bool tryAddingSymbolicOperand(uint64_t Value, uint64_t InstSize, MCInst &MI);

};

} // namespace llvm

#endif
