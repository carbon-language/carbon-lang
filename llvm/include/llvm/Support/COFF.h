//===-- llvm/Support/COFF.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an definitions used in Windows COFF Files.
//
// Structures and enums defined within this file where created using
// information from Microsoft's publicly available PE/COFF format document:
//
// Microsoft Portable Executable and Common Object File Format Specification
// Revision 8.1 - February 15, 2008
//
// As of 5/2/2010, hosted by Microsoft at:
// http://www.microsoft.com/whdc/system/platform/firmware/pecoff.mspx
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WIN_COFF_H
#define LLVM_SUPPORT_WIN_COFF_H

#include "llvm/Support/DataTypes.h"
#include <cstring>

namespace llvm {
namespace COFF {

  // Sizes in bytes of various things in the COFF format.
  enum {
    HeaderSize     = 20,
    NameSize       = 8,
    SymbolSize     = 18,
    SectionSize    = 40,
    RelocationSize = 10
  };

  struct header {
    uint16_t Machine;
    uint16_t NumberOfSections;
    uint32_t TimeDateStamp;
    uint32_t PointerToSymbolTable;
    uint32_t NumberOfSymbols;
    uint16_t SizeOfOptionalHeader;
    uint16_t Characteristics;
  };

  enum MachineTypes {
    IMAGE_FILE_MACHINE_I386 = 0x14C,
    IMAGE_FILE_MACHINE_AMD64 = 0x8664
  };

  struct symbol {
    char     Name[NameSize];
    uint32_t Value;
    uint16_t Type;
    uint8_t  StorageClass;
    uint16_t SectionNumber;
    uint8_t  NumberOfAuxSymbols;
  };

  enum SymbolFlags {
    SF_TypeMask = 0x0000FFFF,
    SF_TypeShift = 0,

    SF_ClassMask = 0x00FF0000,
    SF_ClassShift = 16,

    SF_WeakExternal = 0x01000000
  };

  enum SymbolSectionNumber {
    IMAGE_SYM_DEBUG     = -2,
    IMAGE_SYM_ABSOLUTE  = -1,
    IMAGE_SYM_UNDEFINED = 0
  };

  /// Storage class tells where and what the symbol represents
  enum SymbolStorageClass {
    IMAGE_SYM_CLASS_END_OF_FUNCTION  = -1,  ///< Physical end of function
    IMAGE_SYM_CLASS_NULL             = 0,   ///< No symbol
    IMAGE_SYM_CLASS_AUTOMATIC        = 1,   ///< Stack variable
    IMAGE_SYM_CLASS_EXTERNAL         = 2,   ///< External symbol
    IMAGE_SYM_CLASS_STATIC           = 3,   ///< Static
    IMAGE_SYM_CLASS_REGISTER         = 4,   ///< Register variable
    IMAGE_SYM_CLASS_EXTERNAL_DEF     = 5,   ///< External definition
    IMAGE_SYM_CLASS_LABEL            = 6,   ///< Label
    IMAGE_SYM_CLASS_UNDEFINED_LABEL  = 7,   ///< Undefined label
    IMAGE_SYM_CLASS_MEMBER_OF_STRUCT = 8,   ///< Member of structure
    IMAGE_SYM_CLASS_ARGUMENT         = 9,   ///< Function argument
    IMAGE_SYM_CLASS_STRUCT_TAG       = 10,  ///< Structure tag
    IMAGE_SYM_CLASS_MEMBER_OF_UNION  = 11,  ///< Member of union
    IMAGE_SYM_CLASS_UNION_TAG        = 12,  ///< Union tag
    IMAGE_SYM_CLASS_TYPE_DEFINITION  = 13,  ///< Type definition
    IMAGE_SYM_CLASS_UNDEFINED_STATIC = 14,  ///< Undefined static
    IMAGE_SYM_CLASS_ENUM_TAG         = 15,  ///< Enumeration tag
    IMAGE_SYM_CLASS_MEMBER_OF_ENUM   = 16,  ///< Member of enumeration
    IMAGE_SYM_CLASS_REGISTER_PARAM   = 17,  ///< Register parameter
    IMAGE_SYM_CLASS_BIT_FIELD        = 18,  ///< Bit field
    /// ".bb" or ".eb" - beginning or end of block
    IMAGE_SYM_CLASS_BLOCK            = 100,
    /// ".bf" or ".ef" - beginning or end of function
    IMAGE_SYM_CLASS_FUNCTION         = 101,
    IMAGE_SYM_CLASS_END_OF_STRUCT    = 102, ///< End of structure
    IMAGE_SYM_CLASS_FILE             = 103, ///< File name
    /// Line number, reformatted as symbol
    IMAGE_SYM_CLASS_SECTION          = 104,
    IMAGE_SYM_CLASS_WEAK_EXTERNAL    = 105, ///< Duplicate tag
    /// External symbol in dmert public lib
    IMAGE_SYM_CLASS_CLR_TOKEN        = 107
  };

  enum SymbolBaseType {
    IMAGE_SYM_TYPE_NULL   = 0,  ///< No type information or unknown base type.
    IMAGE_SYM_TYPE_VOID   = 1,  ///< Used with void pointers and functions.
    IMAGE_SYM_TYPE_CHAR   = 2,  ///< A character (signed byte).
    IMAGE_SYM_TYPE_SHORT  = 3,  ///< A 2-byte signed integer.
    IMAGE_SYM_TYPE_INT    = 4,  ///< A natural integer type on the target.
    IMAGE_SYM_TYPE_LONG   = 5,  ///< A 4-byte signed integer.
    IMAGE_SYM_TYPE_FLOAT  = 6,  ///< A 4-byte floating-point number.
    IMAGE_SYM_TYPE_DOUBLE = 7,  ///< An 8-byte floating-point number.
    IMAGE_SYM_TYPE_STRUCT = 8,  ///< A structure.
    IMAGE_SYM_TYPE_UNION  = 9,  ///< An union.
    IMAGE_SYM_TYPE_ENUM   = 10, ///< An enumerated type.
    IMAGE_SYM_TYPE_MOE    = 11, ///< A member of enumeration (a specific value).
    IMAGE_SYM_TYPE_BYTE   = 12, ///< A byte; unsigned 1-byte integer.
    IMAGE_SYM_TYPE_WORD   = 13, ///< A word; unsigned 2-byte integer.
    IMAGE_SYM_TYPE_UINT   = 14, ///< An unsigned integer of natural size.
    IMAGE_SYM_TYPE_DWORD  = 15  ///< An unsigned 4-byte integer.
  };

  enum SymbolComplexType {
    IMAGE_SYM_DTYPE_NULL     = 0, ///< No complex type; simple scalar variable.
    IMAGE_SYM_DTYPE_POINTER  = 1, ///< A pointer to base type.
    IMAGE_SYM_DTYPE_FUNCTION = 2, ///< A function that returns a base type.
    IMAGE_SYM_DTYPE_ARRAY    = 3, ///< An array of base type.

    /// Type is formed as (base + (derived << SCT_COMPLEX_TYPE_SHIFT))
    SCT_COMPLEX_TYPE_SHIFT   = 4
  };

  struct section {
    char     Name[NameSize];
    uint32_t VirtualSize;
    uint32_t VirtualAddress;
    uint32_t SizeOfRawData;
    uint32_t PointerToRawData;
    uint32_t PointerToRelocations;
    uint32_t PointerToLineNumbers;
    uint16_t NumberOfRelocations;
    uint16_t NumberOfLineNumbers;
    uint32_t Characteristics;
  };

  enum SectionCharacteristics {
    IMAGE_SCN_TYPE_NO_PAD            = 0x00000008,
    IMAGE_SCN_CNT_CODE               = 0x00000020,
    IMAGE_SCN_CNT_INITIALIZED_DATA   = 0x00000040,
    IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080,
    IMAGE_SCN_LNK_OTHER              = 0x00000100,
    IMAGE_SCN_LNK_INFO               = 0x00000200,
    IMAGE_SCN_LNK_REMOVE             = 0x00000800,
    IMAGE_SCN_LNK_COMDAT             = 0x00001000,
    IMAGE_SCN_GPREL                  = 0x00008000,
    IMAGE_SCN_MEM_PURGEABLE          = 0x00020000,
    IMAGE_SCN_MEM_16BIT              = 0x00020000,
    IMAGE_SCN_MEM_LOCKED             = 0x00040000,
    IMAGE_SCN_MEM_PRELOAD            = 0x00080000,
    IMAGE_SCN_ALIGN_1BYTES           = 0x00100000,
    IMAGE_SCN_ALIGN_2BYTES           = 0x00200000,
    IMAGE_SCN_ALIGN_4BYTES           = 0x00300000,
    IMAGE_SCN_ALIGN_8BYTES           = 0x00400000,
    IMAGE_SCN_ALIGN_16BYTES          = 0x00500000,
    IMAGE_SCN_ALIGN_32BYTES          = 0x00600000,
    IMAGE_SCN_ALIGN_64BYTES          = 0x00700000,
    IMAGE_SCN_ALIGN_128BYTES         = 0x00800000,
    IMAGE_SCN_ALIGN_256BYTES         = 0x00900000,
    IMAGE_SCN_ALIGN_512BYTES         = 0x00A00000,
    IMAGE_SCN_ALIGN_1024BYTES        = 0x00B00000,
    IMAGE_SCN_ALIGN_2048BYTES        = 0x00C00000,
    IMAGE_SCN_ALIGN_4096BYTES        = 0x00D00000,
    IMAGE_SCN_ALIGN_8192BYTES        = 0x00E00000,
    IMAGE_SCN_LNK_NRELOC_OVFL        = 0x01000000,
    IMAGE_SCN_MEM_DISCARDABLE        = 0x02000000,
    IMAGE_SCN_MEM_NOT_CACHED         = 0x04000000,
    IMAGE_SCN_MEM_NOT_PAGED          = 0x08000000,
    IMAGE_SCN_MEM_SHARED             = 0x10000000,
    IMAGE_SCN_MEM_EXECUTE            = 0x20000000,
    IMAGE_SCN_MEM_READ               = 0x40000000,
    IMAGE_SCN_MEM_WRITE              = 0x80000000
  };

  struct relocation {
    uint32_t VirtualAddress;
    uint32_t SymbolTableIndex;
    uint16_t Type;
  };

  enum RelocationTypeX86 {
    IMAGE_REL_I386_ABSOLUTE = 0x0000,
    IMAGE_REL_I386_DIR16    = 0x0001,
    IMAGE_REL_I386_REL16    = 0x0002,
    IMAGE_REL_I386_DIR32    = 0x0006,
    IMAGE_REL_I386_DIR32NB  = 0x0007,
    IMAGE_REL_I386_SEG12    = 0x0009,
    IMAGE_REL_I386_SECTION  = 0x000A,
    IMAGE_REL_I386_SECREL   = 0x000B,
    IMAGE_REL_I386_TOKEN    = 0x000C,
    IMAGE_REL_I386_SECREL7  = 0x000D,
    IMAGE_REL_I386_REL32    = 0x0014,

    IMAGE_REL_AMD64_ABSOLUTE  = 0x0000,
    IMAGE_REL_AMD64_ADDR64    = 0x0001,
    IMAGE_REL_AMD64_ADDR32    = 0x0002,
    IMAGE_REL_AMD64_ADDR32NB  = 0x0003,
    IMAGE_REL_AMD64_REL32     = 0x0004,
    IMAGE_REL_AMD64_REL32_1   = 0x0005,
    IMAGE_REL_AMD64_REL32_2   = 0x0006,
    IMAGE_REL_AMD64_REL32_3   = 0x0007,
    IMAGE_REL_AMD64_REL32_4   = 0x0008,
    IMAGE_REL_AMD64_REL32_5   = 0x0009,
    IMAGE_REL_AMD64_SECTION   = 0x000A,
    IMAGE_REL_AMD64_SECREL    = 0x000B,
    IMAGE_REL_AMD64_SECREL7   = 0x000C,
    IMAGE_REL_AMD64_TOKEN     = 0x000D,
    IMAGE_REL_AMD64_SREL32    = 0x000E,
    IMAGE_REL_AMD64_PAIR      = 0x000F,
    IMAGE_REL_AMD64_SSPAN32   = 0x0010
  };

  enum COMDATType {
    IMAGE_COMDAT_SELECT_NODUPLICATES = 1,
    IMAGE_COMDAT_SELECT_ANY,
    IMAGE_COMDAT_SELECT_SAME_SIZE,
    IMAGE_COMDAT_SELECT_EXACT_MATCH,
    IMAGE_COMDAT_SELECT_ASSOCIATIVE,
    IMAGE_COMDAT_SELECT_LARGEST
  };

  // Auxiliary Symbol Formats
  struct AuxiliaryFunctionDefinition {
    uint32_t TagIndex;
    uint32_t TotalSize;
    uint32_t PointerToLinenumber;
    uint32_t PointerToNextFunction;
    uint8_t  unused[2];
  };

  struct AuxiliarybfAndefSymbol {
    uint8_t  unused1[4];
    uint16_t Linenumber;
    uint8_t  unused2[6];
    uint32_t PointerToNextFunction;
    uint8_t  unused3[2];
  };

  struct AuxiliaryWeakExternal {
    uint32_t TagIndex;
    uint32_t Characteristics;
    uint8_t  unused[10];
  };

  /// These are not documented in the spec, but are located in WinNT.h.
  enum WeakExternalCharacteristics {
    IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY = 1,
    IMAGE_WEAK_EXTERN_SEARCH_LIBRARY   = 2,
    IMAGE_WEAK_EXTERN_SEARCH_ALIAS     = 3
  };

  struct AuxiliaryFile {
    uint8_t FileName[18];
  };

  struct AuxiliarySectionDefinition {
    uint32_t Length;
    uint16_t NumberOfRelocations;
    uint16_t NumberOfLinenumbers;
    uint32_t CheckSum;
    uint16_t Number;
    uint8_t  Selection;
    uint8_t  unused[3];
  };

  union Auxiliary {
    AuxiliaryFunctionDefinition FunctionDefinition;
    AuxiliarybfAndefSymbol      bfAndefSymbol;
    AuxiliaryWeakExternal       WeakExternal;
    AuxiliaryFile               File;
    AuxiliarySectionDefinition  SectionDefinition;
  };

} // End namespace llvm.
} // End namespace COFF.

#endif
