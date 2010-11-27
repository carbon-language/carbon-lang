//===- MachOFormat.h - Mach-O Format Structures And Constants ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares various structures and constants which are platform
// independent and can be shared by any client which wishes to interact with
// Mach object files.
//
// The definitions here are purposely chosen to match the LLVM style as opposed
// to following the platform specific definition of the format.
//
// On a Mach system, see the <mach-o/...> includes for more information, in
// particular <mach-o/loader.h>.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHOFORMAT_H
#define LLVM_OBJECT_MACHOFORMAT_H

namespace llvm {
namespace object {

/// General Mach platform information.
namespace mach {
  /// @name CPU Type and Subtype Information
  /// {

  /// \brief Capability bits used in CPU type encoding.
  enum CPUTypeFlagsMask {
    CTFM_ArchMask =  0xFF000000,
    CTFM_ArchABI64 = 0x01000000
  };

  /// \brief Machine type IDs used in CPU type encoding.
  enum CPUTypeMachine {
    CTM_i386      = 7,
    CTM_x86_64    = CTM_i386 | CTFM_ArchABI64,
    CTM_ARM       = 12,
    CTM_SPARC     = 14,
    CTM_PowerPC   = 18,
    CTM_PowerPC64 = CTM_PowerPC | CTFM_ArchABI64
  };

  /// \brief Capability bits used in CPU subtype encoding.
  enum CPUSubtypeFlagsMask {
    CSFM_SubtypeMask =  0xFF000000,
    CSFM_SubtypeLib64 = 0x80000000
  };

  /// \brief ARM Machine Subtypes.
  enum CPUSubtypeARM {
    CSARM_ALL    = 0,
    CSARM_V4T    = 5,
    CSARM_V6     = 6,
    CSARM_V5TEJ  = 7,
    CSARM_XSCALE = 8,
    CSARM_V7     = 9
  };

  /// \brief PowerPC Machine Subtypes.
  enum CPUSubtypePowerPC {
    CSPPC_ALL = 0
  };

  /// \brief SPARC Machine Subtypes.
  enum CPUSubtypeSPARC {
    CSSPARC_ALL = 0
  };

  /// \brief x86 Machine Subtypes.
  enum CPUSubtypeX86 {
    CSX86_ALL = 3
  };

  /// @}

} // end namespace mach

/// Format information for Mach object files.
namespace macho {
  /// \brief Constants for header magic field.
  enum HeaderMagic {
    HM_Object32 = 0xFEEDFACE,  ///< 32-bit mach object file
    HM_Object64 = 0xFEEDFACF,  ///< 64-bit mach object file
    HM_Universal = 0xCAFEBABE  ///< Universal object file
  };

  /// \brief Constants for structure sizes.
  enum StructureSizes {
    Header32Size = 28,
    Header64Size = 32,
    SegmentLoadCommand32Size = 56,
    SegmentLoadCommand64Size = 72,
    Section32Size = 68,
    Section64Size = 80,
    SymtabLoadCommandSize = 24,
    DysymtabLoadCommandSize = 80,
    Nlist32Size = 12,
    Nlist64Size = 16,
    RelocationInfoSize = 8
  };

  // See <mach-o/loader.h>.
  enum HeaderFileType {
    HFT_Object = 0x1
  };

  enum HeaderFlags {
    HF_SubsectionsViaSymbols = 0x2000
  };

  enum LoadCommandType {
    LCT_Segment = 0x1,
    LCT_Symtab = 0x2,
    LCT_Dysymtab = 0xb,
    LCT_Segment64 = 0x19
  };

  // See <mach-o/nlist.h>.
  enum SymbolTypeType {
    STT_Undefined = 0x00,
    STT_Absolute  = 0x02,
    STT_Section   = 0x0e
  };

  enum SymbolTypeFlags {
    // If any of these bits are set, then the entry is a stab entry number (see
    // <mach-o/stab.h>. Otherwise the other masks apply.
    STF_StabsEntryMask = 0xe0,

    STF_TypeMask       = 0x0e,
    STF_External       = 0x01,
    STF_PrivateExtern  = 0x10
  };

  /// IndirectSymbolFlags - Flags for encoding special values in the indirect
  /// symbol entry.
  enum IndirectSymbolFlags {
    ISF_Local    = 0x80000000,
    ISF_Absolute = 0x40000000
  };

  /// RelocationFlags - Special flags for addresses.
  enum RelocationFlags {
    RF_Scattered = 0x80000000
  };

  enum RelocationInfoType {
    RIT_Vanilla             = 0,
    RIT_Pair                = 1,
    RIT_Difference          = 2,
    RIT_PreboundLazyPointer = 3,
    RIT_LocalDifference     = 4,
    RIT_TLV                 = 5
  };

  /// X86_64 uses its own relocation types.
  enum RelocationInfoTypeX86_64 {
    RIT_X86_64_Unsigned   = 0,
    RIT_X86_64_Signed     = 1,
    RIT_X86_64_Branch     = 2,
    RIT_X86_64_GOTLoad    = 3,
    RIT_X86_64_GOT        = 4,
    RIT_X86_64_Subtractor = 5,
    RIT_X86_64_Signed1    = 6,
    RIT_X86_64_Signed2    = 7,
    RIT_X86_64_Signed4    = 8,
    RIT_X86_64_TLV        = 9
  };

} // end namespace macho

} // end namespace object
} // end namespace llvm

#endif
