//===-- HexagonBaseInfo.h - Top level definitions for Hexagon --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the Hexagon target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBASEINFO_H
#define HEXAGONBASEINFO_H

#include "HexagonMCTargetDesc.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

/// HexagonII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace HexagonII {
  // *** The code below must match HexagonInstrFormat*.td *** //

  // Insn types.
  // *** Must match HexagonInstrFormat*.td ***
  enum Type {
    TypePSEUDO  = 0,
    TypeALU32   = 1,
    TypeCR      = 2,
    TypeJR      = 3,
    TypeJ       = 4,
    TypeLD      = 5,
    TypeST      = 6,
    TypeSYSTEM  = 7,
    TypeXTYPE   = 8,
    TypeMEMOP   = 9,
    TypeNV      = 10,
    TypePREFIX  = 30, // Such as extenders.
    TypeENDLOOP = 31  // Such as end of a HW loop.
  };

  enum SubTarget {
    HasV2SubT     = 0xf,
    HasV2SubTOnly = 0x1,
    NoV2SubT      = 0x0,
    HasV3SubT     = 0xe,
    HasV3SubTOnly = 0x2,
    NoV3SubT      = 0x1,
    HasV4SubT     = 0xc,
    NoV4SubT      = 0x3,
    HasV5SubT     = 0x8,
    NoV5SubT      = 0x7
  };

  enum AddrMode {
    NoAddrMode     = 0,  // No addressing mode
    Absolute       = 1,  // Absolute addressing mode
    AbsoluteSet    = 2,  // Absolute set addressing mode
    BaseImmOffset  = 3,  // Indirect with offset
    BaseLongOffset = 4,  // Indirect with long offset
    BaseRegOffset  = 5,  // Indirect with register offset
    PostInc        = 6   // Post increment addressing mode
  };

  enum MemAccessSize {
    NoMemAccess = 0,            // Not a memory acces instruction.
    ByteAccess = 1,             // Byte access instruction (memb).
    HalfWordAccess = 2,         // Half word access instruction (memh).
    WordAccess = 3,             // Word access instrution (memw).
    DoubleWordAccess = 4        // Double word access instruction (memd)
  };

  // MCInstrDesc TSFlags
  // *** Must match HexagonInstrFormat*.td ***
  enum {
    // This 5-bit field describes the insn type.
    TypePos  = 0,
    TypeMask = 0x1f,

    // Solo instructions.
    SoloPos  = 5,
    SoloMask = 0x1,

    // Predicated instructions.
    PredicatedPos  = 6,
    PredicatedMask = 0x1,
    PredicatedFalsePos  = 7,
    PredicatedFalseMask = 0x1,
    PredicatedNewPos  = 8,
    PredicatedNewMask = 0x1,

    // New-Value consumer instructions.
    NewValuePos  = 9,
    NewValueMask = 0x1,

    // New-Value producer instructions.
    hasNewValuePos  = 10,
    hasNewValueMask = 0x1,

    // Which operand consumes or produces a new value.
    NewValueOpPos  = 11,
    NewValueOpMask = 0x7,

    // Which bits encode the new value.
    NewValueBitsPos  = 14,
    NewValueBitsMask = 0x3,

    // Stores that can become new-value stores.
    mayNVStorePos  = 16,
    mayNVStoreMask = 0x1,

    // New-value store instructions.
    NVStorePos  = 17,
    NVStoreMask = 0x1,

    // Extendable insns.
    ExtendablePos  = 18,
    ExtendableMask = 0x1,

    // Insns must be extended.
    ExtendedPos  = 19,
    ExtendedMask = 0x1,

    // Which operand may be extended.
    ExtendableOpPos  = 20,
    ExtendableOpMask = 0x7,

    // Signed or unsigned range.
    ExtentSignedPos = 23,
    ExtentSignedMask = 0x1,

    // Number of bits of range before extending operand.
    ExtentBitsPos  = 24,
    ExtentBitsMask = 0x1f,

    // Valid subtargets
    validSubTargetPos = 29,
    validSubTargetMask = 0xf,

    // Addressing mode for load/store instructions.
    AddrModePos = 33,
    AddrModeMask = 0x7,

    // Access size of memory access instructions (load/store).
    MemAccessSizePos = 36,
    MemAccesSizeMask = 0x7
  };

  // *** The code above must match HexagonInstrFormat*.td *** //

  // Hexagon specific MO operand flag mask.
  enum HexagonMOTargetFlagVal {
    //===------------------------------------------------------------------===//
    // Hexagon Specific MachineOperand flags.
    MO_NO_FLAG,

    HMOTF_ConstExtended = 1,

    /// MO_PCREL - On a symbol operand, indicates a PC-relative relocation
    /// Used for computing a global address for PIC compilations
    MO_PCREL,

    /// MO_GOT - Indicates a GOT-relative relocation
    MO_GOT,

    // Low or high part of a symbol.
    MO_LO16, MO_HI16,

    // Offset from the base of the SDA.
    MO_GPREL
  };

} // End namespace HexagonII.

} // End namespace llvm.

#endif
