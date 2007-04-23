//===- LLVMBitCodes.h - Enum values for the LLVM bitcode format -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines Bitcode enum values for LLVM IR bitcode files.
//
// The enum values defined in this file should be considered permanent.  If
// new features are added, they should have values added at the end of the
// respective lists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_LLVMBITCODES_H
#define LLVM_BITCODE_LLVMBITCODES_H

#include "llvm/Bitcode/BitCodes.h"

namespace llvm {
namespace bitc {
  // The only top-level block type defined is for a module.
  enum BlockIDs {
    // Blocks
    MODULE_BLOCK_ID          = 0,
  
    // Module sub-block id's
    TYPE_BLOCK_ID            = 1,
    MODULEINFO_BLOCK_ID      = 2,
    GLOBALCONSTANTS_BLOCK_ID = 3,
    FUNCTION_BLOCK_ID        = 4,
    TYPE_SYMTAB_BLOCK_ID     = 5,
    GLOBAL_SYMTAB_BLOCK_ID   = 6
  };
  
  
  /// MODULE blocks have a number of optional fields and subblocks.
  enum ModuleCodes {
    MODULE_CODE_VERSION     = 1,    // VERSION:     [version#]
    MODULE_CODE_TRIPLE      = 2,    // TRIPLE:      [strlen, strchr x N]
    MODULE_CODE_DATALAYOUT  = 3,    // DATALAYOUT:  [strlen, strchr x N]
    MODULE_CODE_ASM         = 4,    // ASM:         [strlen, strchr x N]
    MODULE_CODE_SECTIONNAME = 5,    // SECTIONNAME: [strlen, strchr x N]
    MODULE_CODE_DEPLIB      = 6,    // DEPLIB:      [strlen, strchr x N]

    // GLOBALVAR: [type, isconst, initid, 
    //             linkage, alignment, section, visibility, threadlocal]
    MODULE_CODE_GLOBALVAR   = 7,

    // FUNCTION:  [type, callingconv, isproto, linkage, alignment, section,
    //             visibility]
    MODULE_CODE_FUNCTION    = 8
  };
  
  /// TYPE blocks have codes for each type primitive they use.
  enum TypeCodes {
    TYPE_CODE_NUMENTRY =  1,   // TYPE_CODE_NUMENTRY: [numentries]
    TYPE_CODE_META     =  2,   // TYPE_CODE_META: [metacode]... - Future use
    
    // Type Codes
    TYPE_CODE_VOID     =  3,   // VOID
    TYPE_CODE_FLOAT    =  4,   // FLOAT
    TYPE_CODE_DOUBLE   =  5,   // DOUBLE
    TYPE_CODE_LABEL    =  6,   // LABEL
    TYPE_CODE_OPAQUE   =  7,   // OPAQUE
    TYPE_CODE_INTEGER  =  8,   // INTEGER: [width]
    TYPE_CODE_POINTER  =  9,   // POINTER: [pointee type]
    TYPE_CODE_FUNCTION = 10,   // FUNCTION: [vararg, retty, #pararms, paramty N]
    TYPE_CODE_STRUCT   = 11,   // STRUCT: [ispacked, #elts, eltty x N]
    TYPE_CODE_ARRAY    = 12,   // ARRAY: [numelts, eltty]
    TYPE_CODE_VECTOR   = 13    // VECTOR: [numelts, eltty]
    // Any other type code is assumed to be an unknown type.
  };
  
  
  // The type symbol table only has one code (TST_ENTRY_CODE).
  enum TypeSymtabCodes {
    TST_ENTRY_CODE = 1     // TST_ENTRY: [typeid, namelen, namechar x N]
  };
  
} // End bitc namespace
} // End llvm namespace

#endif
