//===- ARMDisassemblerCore.h - ARM disassembler helpers ----*- C++ -*-===//
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
// It also contains code to represent the concepts of Builder, Builder Factory,
// as well as the Algorithm to solve the problem of disassembling an ARM instr.
//
//===----------------------------------------------------------------------===//

#ifndef ARMDISASSEMBLERCORE_H
#define ARMDISASSEMBLERCORE_H

#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "ARMInstrInfo.h"

namespace llvm {

class ARMUtils {
public:
  static const char *OpcodeName(unsigned Opcode);
};

#define ARM_FORMATS                   \
  ENTRY(ARM_FORMAT_PSEUDO,         0) \
  ENTRY(ARM_FORMAT_MULFRM,         1) \
  ENTRY(ARM_FORMAT_BRFRM,          2) \
  ENTRY(ARM_FORMAT_BRMISCFRM,      3) \
  ENTRY(ARM_FORMAT_DPFRM,          4) \
  ENTRY(ARM_FORMAT_DPSOREGFRM,     5) \
  ENTRY(ARM_FORMAT_LDFRM,          6) \
  ENTRY(ARM_FORMAT_STFRM,          7) \
  ENTRY(ARM_FORMAT_LDMISCFRM,      8) \
  ENTRY(ARM_FORMAT_STMISCFRM,      9) \
  ENTRY(ARM_FORMAT_LDSTMULFRM,    10) \
  ENTRY(ARM_FORMAT_ARITHMISCFRM,  11) \
  ENTRY(ARM_FORMAT_EXTFRM,        12) \
  ENTRY(ARM_FORMAT_VFPUNARYFRM,   13) \
  ENTRY(ARM_FORMAT_VFPBINARYFRM,  14) \
  ENTRY(ARM_FORMAT_VFPCONV1FRM,   15) \
  ENTRY(ARM_FORMAT_VFPCONV2FRM,   16) \
  ENTRY(ARM_FORMAT_VFPCONV3FRM,   17) \
  ENTRY(ARM_FORMAT_VFPCONV4FRM,   18) \
  ENTRY(ARM_FORMAT_VFPCONV5FRM,   19) \
  ENTRY(ARM_FORMAT_VFPLDSTFRM,    20) \
  ENTRY(ARM_FORMAT_VFPLDSTMULFRM, 21) \
  ENTRY(ARM_FORMAT_VFPMISCFRM,    22) \
  ENTRY(ARM_FORMAT_THUMBFRM,      23) \
  ENTRY(ARM_FORMAT_NEONFRM,       24) \
  ENTRY(ARM_FORMAT_NEONGETLNFRM,  25) \
  ENTRY(ARM_FORMAT_NEONSETLNFRM,  26) \
  ENTRY(ARM_FORMAT_NEONDUPFRM,    27) \
  ENTRY(ARM_FORMAT_LDSTEXFRM,     28) \
  ENTRY(ARM_FORMAT_MISCFRM,       29) \
  ENTRY(ARM_FORMAT_THUMBMISCFRM,  30)

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

#define NS_FORMATS                              \
  ENTRY(NS_FORMAT_NONE,                     0)  \
  ENTRY(NS_FORMAT_VLDSTLane,                1)  \
  ENTRY(NS_FORMAT_VLDSTLaneDbl,             2)  \
  ENTRY(NS_FORMAT_VLDSTRQ,                  3)  \
  ENTRY(NS_FORMAT_NVdImm,                   4)  \
  ENTRY(NS_FORMAT_NVdVmImm,                 5)  \
  ENTRY(NS_FORMAT_NVdVmImmVCVT,             6)  \
  ENTRY(NS_FORMAT_NVdVmImmVDupLane,         7)  \
  ENTRY(NS_FORMAT_NVdVmImmVSHLL,            8)  \
  ENTRY(NS_FORMAT_NVectorShuffle,           9)  \
  ENTRY(NS_FORMAT_NVectorShift,             10) \
  ENTRY(NS_FORMAT_NVectorShift2,            11) \
  ENTRY(NS_FORMAT_NVdVnVmImm,               12) \
  ENTRY(NS_FORMAT_NVdVnVmImmVectorShift,    13) \
  ENTRY(NS_FORMAT_NVdVnVmImmVectorExtract,  14) \
  ENTRY(NS_FORMAT_NVdVnVmImmMulScalar,      15) \
  ENTRY(NS_FORMAT_VTBL,                     16)

// NEON instruction sub-format further classify the NEONFrm instruction.
#define ENTRY(n, v) n = v,
typedef enum {
  NS_FORMATS
  NS_FORMAT_NA
} NSFormat;
#undef ENTRY

// Converts enum to const char*.
static const inline char *stringForNSFormat(NSFormat form) {
#define ENTRY(n, v) case n: return #n;
  switch(form) {
    NS_FORMATS
  case NS_FORMAT_NA:
    return "NA";
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
static inline void setSlice(uint32_t &Bits, unsigned From, unsigned To,
                            uint32_t Val) {
  assert(From < 32 && To < 32 && From >= To);
  uint32_t Mask = ((1 << (From - To + 1)) - 1);
  Bits &= ~(Mask << To);
  Bits |= (Val & Mask) << To;
}

/// Various utilities for checking the target specific flags.

/// A unary data processing instruction doesn't have an Rn operand.
static inline bool isUnaryDP(unsigned TSFlags) {
  return (TSFlags & ARMII::UnaryDP);
}

/// This four-bit field describes the addressing mode used.
/// See also ARMBaseInstrInfo.h.
static inline unsigned getAddrMode(unsigned TSFlags) {
  return (TSFlags & ARMII::AddrModeMask);
}

/// {IndexModePre, IndexModePost}
/// Only valid for load and store ops.
/// See also ARMBaseInstrInfo.h.
static inline unsigned getIndexMode(unsigned TSFlags) {
  return (TSFlags & ARMII::IndexModeMask) >> ARMII::IndexModeShift;
}

/// Pre-/post-indexed operations define an extra $base_wb in the OutOperandList.
static inline bool isPrePostLdSt(unsigned TSFlags) {
  return (TSFlags & ARMII::IndexModeMask) != 0;
}

/// AbstractARMMCBuilder - AbstractARMMCBuilder represents an interface of ARM
/// MCInst builder that knows how to build up the MCOperand list.
class AbstractARMMCBuilder {
public:
  /// Build - Build the MCInst fully and return true.  Return false if any
  /// failure occurs.
  virtual bool Build(MCInst &MI, uint32_t insn) { return false; }
};

/// ARMDisassemblyAlgorithm - ARMDisassemblyAlgorithm represents an interface of
/// ARM disassembly algorithm that relies on the entries of target operand info,
/// among other things, to solve the problem of disassembling an ARM machine
/// instruction.
class ARMDisassemblyAlgorithm {
public:
  /// Return true if this algorithm successfully disassembles the instruction.
  /// NumOpsAdded is updated to reflect the number of operands added by the
  /// algorithm.  NumOpsAdded may be less than NumOps, in which case, there are
  /// operands unaccounted for which need to be dealt with by the API client.
  virtual bool Solve(MCInst& MI, unsigned Opcode, uint32_t insn,
      unsigned short NumOps, unsigned &NumOpsAdded) const
    = 0;
};

/// ARMBasicMCBuilder - ARMBasicMCBuilder represents a concrete subclass of
/// ARMAbstractMCBuilder.
class ARMBasicMCBuilder : public AbstractARMMCBuilder {
  unsigned Opcode;
  ARMFormat Format;
  NSFormat NSF;
  unsigned short NumOps;
  const ARMDisassemblyAlgorithm &Algo;
  static unsigned ITCounter; // Possible values: 0, 1, 2, 3, 4.
  static unsigned ITState; // A2.5.2 Consists of IT[7:5] and IT[4:0] initially.

public:
  ARMBasicMCBuilder(ARMBasicMCBuilder &MCB) : AbstractARMMCBuilder(),
    Opcode(MCB.Opcode), Format(MCB.Format), NSF(MCB.NSF), NumOps(MCB.NumOps),
    Algo(MCB.Algo) {}

  /// Opcode, Format, NSF, NumOperands, and Algo make an ARM Basic MCBuilder.
  ARMBasicMCBuilder(unsigned opc, ARMFormat format, NSFormat NSF,
      unsigned short num, const ARMDisassemblyAlgorithm &algo)
    : AbstractARMMCBuilder(), Opcode(opc), Format(format), NumOps(num),
      Algo(algo) {}

  /// TryPredicateAndSBitModifier - TryPredicateAndSBitModifier tries to process
  /// the possible Predicate and SBitModifier, to build the remaining MCOperand
  /// constituents.
  static bool TryPredicateAndSBitModifier(MCInst& MI, unsigned Opcode,
      uint32_t insn, unsigned short NumOpsRemaning);

  /// InITBlock - InITBlock returns true if we are inside an IT block.
  static bool InITBlock() {
    return ITCounter > 0;
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
  /// to the Algo (ARM Disassemble Algorithm) object to perform Format-specific
  /// disassembly, followed by class method TryPredicateAndSBitModifier() to do
  /// PredicateOperand and OptionalDefOperand which follow the Dst/Src Operands.
  virtual bool BuildIt(MCInst &MI, uint32_t insn);

  /// RunBuildAfterHook - RunBuildAfterHook performs operations deemed necessary
  /// after BuildIt is finished.
  virtual bool RunBuildAfterHook(bool Status, MCInst &MI, uint32_t insn);

private:
  /// Get condition of the current IT instruction.
  static unsigned GetITCond() {
    return slice(ITState, 7, 4);
  }

  /// Init ITState.
  static void InitITState(unsigned short bits7_0) {
    ITState = bits7_0;
  }

  /// Update ITState if necessary.
  static void UpdateITState() {
    assert(ITCounter);
    --ITCounter;
    if (ITCounter == 0)
      ITState = 0;
    else {
      unsigned short NewITState4_0 = slice(ITState, 4, 0) << 1;
      setSlice(ITState, 4, 0, NewITState4_0);
    }
  }
};

/// ARMMCBuilderFactory - ARMMCBuilderFactory represents the factory class that
/// vends out ARMAbstractMCBuilder instances through its class method.
class ARMMCBuilderFactory {
private:
  ARMMCBuilderFactory(); // DO NOT IMPLEMENT.

public:
  /// CreateMCBuilder - Return an AbstractARMMCBuilder that can build up the MC
  /// infrastructure of an MCInst given the Opcode and Format of the instr.
  /// Return NULL if it fails to create/return a proper builder.  API clients
  /// are responsible for freeing up of the allocated memory.  Cacheing can be
  /// performed by the API clients to improve performance.
  static AbstractARMMCBuilder *CreateMCBuilder(unsigned Opcode,
      ARMFormat Format, NSFormat NSF);
};

} // namespace llvm

#endif
