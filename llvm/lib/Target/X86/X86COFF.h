//===--- X86COFF.h - Some definitions from COFF documentations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file just defines some symbols found in COFF documentation. They are
// used to emit function type information for COFF targets (Cygwin/Mingw32).
//
//===----------------------------------------------------------------------===//

#ifndef X86COFF_H
#define X86COFF_H

namespace COFF 
{
enum StorageClass {
  C_EFCN =   -1,  // physical end of function
  C_NULL    = 0,
  C_AUTO    = 1,  // external definition
  C_EXT     = 2,  // external symbol
  C_STAT    = 3,  // static
  C_REG     = 4,  // register variable
  C_EXTDEF  = 5,  // external definition
  C_LABEL   = 6,  // label
  C_ULABEL  = 7,  // undefined label
  C_MOS     = 8,  // member of structure
  C_ARG     = 9,  // function argument
  C_STRTAG  = 10, // structure tag
  C_MOU     = 11, // member of union
  C_UNTAG   = 12, // union tag
  C_TPDEF   = 13, // type definition
  C_USTATIC = 14, // undefined static
  C_ENTAG   = 15, // enumeration tag
  C_MOE     = 16, // member of enumeration
  C_REGPARM = 17, // register parameter
  C_FIELD   = 18, // bit field

  C_BLOCK  = 100, // ".bb" or ".eb"
  C_FCN    = 101, // ".bf" or ".ef"
  C_EOS    = 102, // end of structure
  C_FILE   = 103, // file name
  C_LINE   = 104, // dummy class for line number entry
  C_ALIAS  = 105, // duplicate tag
  C_HIDDEN = 106
};

enum SymbolType {
  T_NULL   = 0,  // no type info
  T_ARG    = 1,  // function argument (only used by compiler)
  T_VOID   = 1,
  T_CHAR   = 2,  // character
  T_SHORT  = 3,  // short integer
  T_INT    = 4,  // integer
  T_LONG   = 5,  // long integer
  T_FLOAT  = 6,  // floating point
  T_DOUBLE = 7,  // double word
  T_STRUCT = 8,  // structure
  T_UNION  = 9,  // union
  T_ENUM   = 10, // enumeration
  T_MOE    = 11, // member of enumeration
  T_UCHAR  = 12, // unsigned character
  T_USHORT = 13, // unsigned short
  T_UINT   = 14, // unsigned integer
  T_ULONG  = 15  // unsigned long
};

enum SymbolDerivedType {
  DT_NON = 0, // no derived type
  DT_PTR = 1, // pointer
  DT_FCN = 2, // function
  DT_ARY = 3  // array
};

enum TypePacking {
  N_BTMASK = 017,
  N_TMASK = 060,
  N_TMASK1 = 0300,
  N_TMASK2 = 0360,
  N_BTSHFT = 4,
  N_TSHIFT = 2
};

}

#endif // X86COFF_H
