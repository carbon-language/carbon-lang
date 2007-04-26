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
    CONSTANTS_BLOCK_ID       = 3,
    FUNCTION_BLOCK_ID        = 4,
    TYPE_SYMTAB_BLOCK_ID     = 5,
    VALUE_SYMTAB_BLOCK_ID    = 6
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
    MODULE_CODE_FUNCTION    = 8,
    
    // ALIAS: [alias type, aliasee val#, linkage]
    MODULE_CODE_ALIAS       = 9,
    
    /// MODULE_CODE_PURGEVALS: [numvals]
    MODULE_CODE_PURGEVALS   = 10
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
    TST_CODE_ENTRY = 1     // TST_ENTRY: [typeid, namelen, namechar x N]
  };
  
  // The value symbol table only has one code (VST_ENTRY_CODE).
  enum ValueSymtabCodes {
    VST_CODE_ENTRY = 1     // VST_ENTRY: [valid, namelen, namechar x N]
  };
  
  // The constants block (CONSTANTS_BLOCK_ID) describes emission for each
  // constant and maintains an implicit current type value.
  enum ConstantsSymtabCodes {
    CST_CODE_SETTYPE       =  1,  // SETTYPE:       [typeid]
    CST_CODE_NULL          =  2,  // NULL
    CST_CODE_UNDEF         =  3,  // UNDEF
    CST_CODE_INTEGER       =  4,  // INTEGER:       [intval]
    CST_CODE_WIDE_INTEGER  =  5,  // WIDE_INTEGER:  [n, n x intval]
    CST_CODE_FLOAT         =  6,  // FLOAT:         [fpval]
    CST_CODE_AGGREGATE     =  7,  // AGGREGATE:     [n, n x value number]
    CST_CODE_CE_BINOP      =  8,  // CE_BINOP:      [opcode, opval, opval]
    CST_CODE_CE_CAST       =  9,  // CE_CAST:       [opcode, opty, opval]
    CST_CODE_CE_GEP        = 10,  // CE_GEP:        [n, n x operands]
    CST_CODE_CE_SELECT     = 11,  // CE_SELECT:     [opval, opval, opval]
    CST_CODE_CE_EXTRACTELT = 12,  // CE_EXTRACTELT: [opty, opval, opval]
    CST_CODE_CE_INSERTELT  = 13,  // CE_INSERTELT:  [opval, opval, opval]
    CST_CODE_CE_SHUFFLEVEC = 14,  // CE_SHUFFLEVEC: [opval, opval, opval]
    CST_CODE_CE_CMP        = 15   // CE_CMP:        [opty, opval, opval, pred]
  };
  
  /// CastOpcodes - These are values used in the bitcode files to encode which
  /// cast a CST_CODE_CE_CAST or a XXX refers to.  The values of these enums
  /// have no fixed relation to the LLVM IR enum values.  Changing these will
  /// break compatibility with old files.
  enum CastOpcodes {
    CAST_TRUNC    =  0,
    CAST_ZEXT     =  1,
    CAST_SEXT     =  2,
    CAST_FPTOUI   =  3,
    CAST_FPTOSI   =  4,
    CAST_UITOFP   =  5,
    CAST_SITOFP   =  6,
    CAST_FPTRUNC  =  7,
    CAST_FPEXT    =  8,
    CAST_PTRTOINT =  9,
    CAST_INTTOPTR = 10,
    CAST_BITCAST  = 11
  };
  
  /// BinaryOpcodes - These are values used in the bitcode files to encode which
  /// binop a CST_CODE_CE_BINOP or a XXX refers to.  The values of these enums
  /// have no fixed relation to the LLVM IR enum values.  Changing these will
  /// break compatibility with old files.
  enum BinaryOpcodes {
    BINOP_ADD  =  0,
    BINOP_SUB  =  1,
    BINOP_MUL  =  2,
    BINOP_UDIV =  3,
    BINOP_SDIV =  4,    // overloaded for FP
    BINOP_UREM =  5,
    BINOP_SREM =  6,    // overloaded for FP
    BINOP_SHL  =  7,
    BINOP_LSHR =  8,
    BINOP_ASHR =  9,
    BINOP_AND  = 10,
    BINOP_OR   = 11,
    BINOP_XOR  = 12
  };
  
} // End bitc namespace
} // End llvm namespace

#endif
