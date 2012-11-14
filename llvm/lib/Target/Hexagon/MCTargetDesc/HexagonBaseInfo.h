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

namespace llvm {

/// HexagonII - This namespace holds all of the target specific flags that
/// instruction info tracks.
///
namespace HexagonII {
  // *** The code below must match HexagonInstrFormat*.td *** //

  // Insn types.
  // *** Must match HexagonInstrFormat*.td ***
  enum Type {
    TypePSEUDO = 0,
    TypeALU32  = 1,
    TypeCR     = 2,
    TypeJR     = 3,
    TypeJ      = 4,
    TypeLD     = 5,
    TypeST     = 6,
    TypeSYSTEM = 7,
    TypeXTYPE  = 8,
    TypeMEMOP  = 9,
    TypeNV     = 10,
    TypePREFIX = 30, // Such as extenders.
    TypeMARKER = 31  // Such as end of a HW loop.
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
    BaseRegOffset  = 5   // Indirect with register offset
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
    PredicatedNewPos  = 7,
    PredicatedNewMask = 0x1,

    // Stores that can be newified.
    mayNVStorePos  = 8,
    mayNVStoreMask = 0x1,

    // Dot new value store instructions.
    NVStorePos  = 9,
    NVStoreMask = 0x1,

    // Extendable insns.
    ExtendablePos  = 10,
    ExtendableMask = 0x1,

    // Insns must be extended.
    ExtendedPos  = 11,
    ExtendedMask = 0x1,

    // Which operand may be extended.
    ExtendableOpPos  = 12,
    ExtendableOpMask = 0x7,

    // Signed or unsigned range.
    ExtentSignedPos = 15,
    ExtentSignedMask = 0x1,

    // Number of bits of range before extending operand.
    ExtentBitsPos  = 16,
    ExtentBitsMask = 0x1f,

    // Valid subtargets
    validSubTargetPos = 21,
    validSubTargetMask = 0xf,

    // Addressing mode for load/store instructions
    AddrModePos = 25,
    AddrModeMask = 0xf

 };

  // *** The code above must match HexagonInstrFormat*.td *** //

} // End namespace HexagonII.

} // End namespace llvm.

#endif
