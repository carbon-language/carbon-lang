// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	Sparc.cpp
// 
// Purpose:
//	
// History:
//	7/15/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_CODEGEN_SPARC_H
#define LLVM_CODEGEN_SPARC_H

#include "llvm/CodeGen/TargetMachine.h"

// OpCodeMask definitions for the Sparc V9
// 
const OpCodeMask	Immed		= 0x00002000; // immed or reg operand?
const OpCodeMask	Annul		= 0x20000000; // annul delay instr?
const OpCodeMask	PredictTaken	= 0x00080000; // predict branch taken?


//---------------------------------------------------------------------------
// enum SparcMachineOpCode. 
// const MachineInstrDescriptor SparcMachineInstrDesc[]
// 
// Purpose:
//   Description of UltraSparc machine instructions.
// 
//---------------------------------------------------------------------------


enum SparcMachineOpCode {

  NOP,
  
  // Synthetic SPARC assembly opcodes for setting a register to a constant
  SETSW,
  SETUW,
  
  // Set high-order bits of register and clear low-order bits
  SETHI,
  
  // Add or add with carry.
  // Immed bit specifies if second operand is immediate(1) or register(0)
  ADD,
  ADDcc,
  ADDC,
  ADDCcc,

  // Subtract or subtract with carry.
  // Immed bit specifies if second operand is immediate(1) or register(0)
  SUB,
  SUBcc,
  SUBC,
  SUBCcc,
  
  // Integer multiply, signed divide, unsigned divide.
  // Note that the deprecated 32-bit multiply and multiply-step are not used.
  MULX,
  SDIVX,
  UDIVX,
  
  // Floating point add, subtract, compare
  FADDS,
  FADDD,
  FADDQ,
  FSUBS,
  FSUBD,
  FSUBQ,
  FCMPS,
  FCMPD,
  FCMPQ,
  // NOTE: FCMPE{S,D,Q}: FP Compare With Exception are currently unused!
  
  // Floating point multiply or divide.
  FMULS,
  FMULD,
  FMULQ,
  FSMULD,
  FDMULQ,
  FDIVS,
  FDIVD,
  FDIVQ,
  
  // Logical operations
  AND,
  ANDcc,
  ANDN,
  ANDNcc,
  OR,
  ORcc,
  ORN,
  ORNcc,
  XOR,
  XORcc,
  XNOR,
  XNORcc,
  
  // Shift operations
  SLL,
  SRL,
  SRA,
  SLLX,
  SRLX,
  SRAX,
  
  // Convert from floating point to floating point formats
  FSTOD,
  FSTOQ,
  FDTOS,
  FDTOQ,
  FQTOS,
  FQTOD,
  
  // Convert from floating point to integer formats
  FSTOX,
  FDTOX,
  FQTOX,
  FSTOI,
  FDTOI,
  FQTOI,
  
  // Convert from integer to floating point formats
  FXTOS,
  FXTOD,
  FXTOQ,
  FITOS,
  FITOD,
  FITOQ,
  
  // Branch on integer comparison with zero.
  // Annul bit specifies if intruction in delay slot is annulled(1) or not(0).
  // PredictTaken bit hints if branch should be predicted taken(1) or not(0).
  BRZ,
  BRLEZ,
  BRLZ,
  BRNZ,
  BRGZ,
  BRGEZ,

  // Branch on integer condition code.
  // Annul bit specifies if intruction in delay slot is annulled(1) or not(0).
  // PredictTaken bit hints if branch should be predicted taken(1) or not(0).
  BA,
  BN,
  BNE,
  BE,
  BG,
  BLE,
  BGE,
  BL,
  BGU,
  BLEU,
  BCC,
  BCS,
  BPOS,
  BNEG,
  BVC,
  BVS,

  // Branch on floating point condition code.
  // Annul bit specifies if intruction in delay slot is annulled(1) or not(0).
  // PredictTaken bit hints if branch should be predicted taken(1) or not(0).
  FBA,
  FBN,
  FBU,
  FBG,
  FBUG,
  FBL,
  FBUL,
  FBLG,
  FBNE,
  FBE,
  FBUE,
  FBGE,
  FBUGE,
  FBLE,
  FBULE,
  FBO,

  // Conditional move on integer comparison with zero.
  MOVRZ,
  MOVRLEZ,
  MOVRLZ,
  MOVRNZ,
  MOVRGZ,
  MOVRGEZ,

  // Conditional move on integer condition code.
  MOVA,
  MOVN,
  MOVNE,
  MOVE,
  MOVG,
  MOVLE,
  MOVGE,
  MOVL,
  MOVGU,
  MOVLEU,
  MOVCC,
  MOVCS,
  MOVPOS,
  MOVNEG,
  MOVVC,
  MOVVS,

  // Conditional move on floating point condition code.
  // Note that the enum name is not the same as the assembly mnemonic below
  // because that would duplicate some entries with those above.
  // Therefore, we use MOVF here instead of MOV.
  MOVFA,
  MOVFN,
  MOVFU,
  MOVFG,
  MOVFUG,
  MOVFL,
  MOVFUL,
  MOVFLG,
  MOVFNE,
  MOVFE,
  MOVFUE,
  MOVFGE,
  MOVFUGE,
  MOVFLE,
  MOVFULE,
  MOVFO,

  // Load integer instructions
  LDSB,
  LDSH,
  LDSW,
  LDUB,
  LDUH,
  LDUW,
  LDX,
  
  // Load floating-point instructions
  LD,
  LDD,			// use of this for integers is deprecated for Sparc V9
  LDQ,
  
  // Store integer instructions
  STB,
  STH,
  STW,
  STX,
  
  // Store floating-point instructions
  ST,
  STD,
  
  // Call, Return, and "Jump and link"
  // Immed bit specifies if second operand is immediate(1) or register(0)
  CALL,
  JMPL,
  RETURN,

  // End-of-array marker
  INVALID_OPCODE
};

const MachineInstrDescriptor SparcMachineInstrDesc[] = {
  
  // Fields of each structure:
  // opCodeString,
  //           numOperands,
  //                resultPosition (0-based; -1 if no result),
  //                     maxImmedConst,
  //                         immedIsSignExtended,
  //                                numDelaySlots (in cycles)
  //					latency (in cycles)
  //					    class flags
  
  { "NOP",	0,  -1,  0,  false, 0,	1,  M_NOP_FLAG },
  
  // Synthetic SPARC assembly opcodes for setting a register to a constant.
  // Max immediate constant should be ignored for both these instructions.
  { "SETSW",	2,   1,  0,  true,  0,  1,  M_INT_FLAG | M_ARITH_FLAG },
  { "SETUW",	2,   1,  0,  false, 0,  1,  M_INT_FLAG | M_LOGICAL_FLAG | M_ARITH_FLAG },

  // Set high-order bits of register and clear low-order bits
  { "SETHI",	2,  1,  (1 << 22) - 1, false, 0,  1,  M_INT_FLAG | M_LOGICAL_FLAG | M_ARITH_FLAG },
  
  // Add or add with carry.
  { "ADD",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "ADDcc",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "ADDC",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "ADDCcc",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },

  // Sub tract or subtract with carry.
  { "SUB",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "SUBcc",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "SUBC",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "SUBCcc",	3,  2,  (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },

  // Integer multiply, signed divide, unsigned divide.
  // Note that the deprecated 32-bit multiply and multiply-step are not used.
  { "MULX",	3,  2,  (1 << 12) - 1, true, 0, 3, M_INT_FLAG | M_ARITH_FLAG },
  { "SDIVX",	3,  2,  (1 << 12) - 1, true, 0, 6, M_INT_FLAG | M_ARITH_FLAG },
  { "UDIVX",	3,  2,  (1 << 12) - 1, true, 0, 6, M_INT_FLAG | M_ARITH_FLAG },
  
  // Floating point add, subtract, compare.
  // Note that destination of FCMP* instructions is operand 0, not operand 2.
  { "FADDS",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FADDD",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FADDQ",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSUBS",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSUBD",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSUBQ",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FCMPS",	3,  0,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FCMPD",	3,  0,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FCMPQ",	3,  0,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  // NOTE: FCMPE{S,D,Q}: FP Compare With Exception are currently unused!
  
  // Floating point multiply or divide.
  { "FMULS",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FMULD",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FMULQ",	3,  2,  0,  false, 0, 0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSMULD",	3,  2,  0,  false, 0, 3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDMULQ",	3,  2,  0,  false, 0, 0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDIVS",	3,  2,  0,  false, 0, 22, M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDIVD",	3,  2,  0,  false, 0, 22, M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDIVQ",	3,  2,  0,  false, 0, 0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  
  // Logical operations
  { "AND",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "ANDcc",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "ANDN",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "ANDNcc",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "OR",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "ORcc",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "ORN",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "ORNcc",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "XOR",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "XORcc",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "XNOR",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "XNORcc",	3,  2, (1 << 12) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  
  // Shift operations
  { "SLL",	3,  2, (1 << 5) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRL",	3,  2, (1 << 5) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRA",	3,  2, (1 << 5) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  { "SLLX",	3,  2, (1 << 6) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRLX",	3,  2, (1 << 6) - 1, true, 0, 1, M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRAX",	3,  2, (1 << 6) - 1, true, 0, 1, M_INT_FLAG | M_ARITH_FLAG },
  
  // Convert from floating point to floating point formats
  { "FSTOD",	2,  1,  0,  false,  0,  3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSTOQ",	2,  1,  0,  false,  0,  0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDTOS",	2,  1,  0,  false,  0,  3,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDTOQ",	2,  1,  0,  false,  0,  0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FQTOS",	2,  1,  0,  false,  0,  0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FQTOD",	2,  1,  0,  false,  0,  0,  M_FLOAT_FLAG | M_ARITH_FLAG },
  
  // Convert from floating point to integer formats.
  // Note that this accesses both integer and floating point registers.
  { "FSTOX",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FDTOX",	2,  1,  0,  false, 0, 0,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FQTOX",	2,  1,  0,  false, 0, 2,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FSTOI",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FDTOI",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FQTOI",	2,  1,  0,  false, 0, 0,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  
  // Convert from integer to floating point formats
  // Note that this accesses both integer and floating point registers.
  { "FXTOS",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FXTOD",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FXTOQ",	2,  1,  0,  false, 0, 0,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FITOS",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FITOD",	2,  1,  0,  false, 0, 3,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FITOQ",	2,  1,  0,  false, 0, 0,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  
  // Branch on integer comparison with zero.
  // Latency includes the delay slot.
  { "BRZ",	2, -1, (1 << 15) - 1, true, 1, 2, M_INT_FLAG | M_BRANCH_FLAG },
  { "BRLEZ",	2, -1, (1 << 15) - 1, true, 1, 2, M_INT_FLAG | M_BRANCH_FLAG },
  { "BRLZ",	2, -1, (1 << 15) - 1, true, 1, 2, M_INT_FLAG | M_BRANCH_FLAG },
  { "BRNZ",	2, -1, (1 << 15) - 1, true, 1, 2, M_INT_FLAG | M_BRANCH_FLAG },
  { "BRGZ",	2, -1, (1 << 15) - 1, true, 1, 2, M_INT_FLAG | M_BRANCH_FLAG },
  { "BRGEZ",	2, -1, (1 << 15) - 1, true, 1, 2, M_INT_FLAG | M_BRANCH_FLAG },

  // Branch on condition code.
  // The first argument specifies the ICC register: %icc or %xcc
  // Latency includes the delay slot.
  { "BA",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BN",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BNE",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BE",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BG",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BLE",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BGE",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BL",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BGU",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BLEU",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BCC",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BCS",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BPOS",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BNEG",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BVC",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "BVS",	2,  -1, (1 << 21) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },

  // Branch on floating point condition code.
  // Annul bit specifies if intruction in delay slot is annulled(1) or not(0).
  // PredictTaken bit hints if branch should be predicted taken(1) or not(0).
  // The first argument is the FCCn register (0 <= n <= 3).
  // Latency includes the delay slot.
  { "FBA",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBN",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBU",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBG",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUG",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBL",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUL",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBLG",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBNE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBGE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUGE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBLE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBULE",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },
  { "FBO",	2,  -1, (1 << 18) - 1, true, 1, 2, M_CC_FLAG | M_BRANCH_FLAG },

  // Conditional move on integer comparison with zero.
  { "MOVRZ",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRLEZ",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRLZ",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRNZ",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRGZ",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRGEZ",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CONDL_FLAG | M_INT_FLAG },

  // Conditional move on integer condition code.
  // The first argument specifies the ICC register: %icc or %xcc
  { "MOVA",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVN",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVNE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVG",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVLE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVGE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVL",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVGU",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVLEU",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVCC",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVCS",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVPOS",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVNEG",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVVC",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVVS",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },

  // Conditional move (of integer register) on floating point condition code.
  // The first argument is the FCCn register (0 <= n <= 3).
  // Note that the enum name above is not the same as the assembly mnemonic
  // because some of the assembly mnemonics are the same as the move on
  // integer CC (e.g., MOVG), and we cannot have the same enum entry twice.
  { "MOVA",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVN",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVU",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVG",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVUG",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVL",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVUL",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVLG",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVNE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVUE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVGE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVUGE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVLE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVULE",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },
  { "MOVO",	3,  2, (1 << 12) - 1,  true, 0, 1, M_CC_FLAG | M_INT_FLAG },

  // Load integer instructions
  // Latency includes 1 cycle for address generation (Sparc IIi)
  // Signed loads of less than 64 bits need an extra cycle for sign-extension.
  //
  // Not reflected here: After a 3-cycle loads, all subsequent consecutive
  // loads also require 3 cycles to avoid contention for the load return
  // stage.  Latency returns to 2 cycles after the first cycle with no load.
  { "LDSB",	3,  2, (1 << 12) - 1,  true, 0, 3, M_INT_FLAG | M_LOAD_FLAG },
  { "LDSH",	3,  2, (1 << 12) - 1,  true, 0, 3, M_INT_FLAG | M_LOAD_FLAG },
  { "LDSW",	3,  2, (1 << 12) - 1,  true, 0, 3, M_INT_FLAG | M_LOAD_FLAG },
  { "LDUB",	3,  2, (1 << 12) - 1,  true, 0, 2, M_INT_FLAG | M_LOAD_FLAG },
  { "LDUH",	3,  2, (1 << 12) - 1,  true, 0, 2, M_INT_FLAG | M_LOAD_FLAG },
  { "LDUW",	3,  2, (1 << 12) - 1,  true, 0, 2, M_INT_FLAG | M_LOAD_FLAG },
  { "LDX",	3,  2, (1 << 12) - 1,  true, 0, 2, M_INT_FLAG | M_LOAD_FLAG },
  
  // Load floating-point instructions
  // Latency includes 1 cycle for address generation (Sparc IIi)
  { "LD",	3, 2, (1 << 12) - 1, true, 0, 2, M_FLOAT_FLAG | M_LOAD_FLAG },
  { "LDD",	3, 2, (1 << 12) - 1, true, 0, 2, M_FLOAT_FLAG | M_LOAD_FLAG },
  { "LDQ",	3, 2, (1 << 12) - 1, true, 0, 2, M_FLOAT_FLAG | M_LOAD_FLAG },
  
  // Store integer instructions
  // Latency includes 1 cycle for address generation (Sparc IIi)
  { "STB",	3, -1, (1 << 12) - 1, true, 0, 2, M_INT_FLAG | M_STORE_FLAG },
  { "STH",	3, -1, (1 << 12) - 1, true, 0, 2, M_INT_FLAG | M_STORE_FLAG },
  { "STW",	3, -1, (1 << 12) - 1, true, 0, 2, M_INT_FLAG | M_STORE_FLAG },
  { "STX",	3, -1, (1 << 12) - 1, true, 0, 3, M_INT_FLAG | M_STORE_FLAG },
  
  // Store floating-point instructions (Sparc IIi)
  { "ST",	3, -1, (1 << 12) - 1, true, 0, 2, M_FLOAT_FLAG | M_STORE_FLAG},
  { "STD",	3, -1, (1 << 12) - 1, true, 0, 2, M_FLOAT_FLAG | M_STORE_FLAG},
  
  // Call, Return and "Jump and link".
  // Latency includes the delay slot.
  { "CALL",	1, -1, (1 << 29) - 1, true, 1, 2, M_BRANCH_FLAG | M_CALL_FLAG},
  { "JMPL",	3, -1, (1 << 12) - 1, true, 1, 2, M_BRANCH_FLAG | M_CALL_FLAG},
  { "RETURN",	2, -1,  0,           false, 1, 2, M_BRANCH_FLAG | M_RET_FLAG },
  
  // End-of-array marker
  { "INVALID_SPARC_OPCODE",	0,  -1,  0,  false, 0, 0,  0x0 }
};



//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class MachineInstrInfo. 
//---------------------------------------------------------------------------

class UltraSparcInstrInfo : public MachineInstrInfo {
public:
  /*ctor*/	UltraSparcInstrInfo()
    : MachineInstrInfo(SparcMachineInstrDesc, INVALID_OPCODE)
  {}
  
  virtual bool		hasResultInterlock	(MachineOpCode opCode)
  {
    // All UltraSPARC instructions have interlocks (note that delay slots
    // are not considered here).
    // However, instructions that use the result of an FCMP produce a
    // 9-cycle stall if they are issued less than 3 cycles after the FCMP.
    // Force the compiler to insert a software interlock (i.e., gap of
    // 2 other groups, including NOPs if necessary).
    return (opCode == FCMPS || opCode == FCMPD || opCode == FCMPQ);
  }
};


//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as MachineInstrInfo. 
//---------------------------------------------------------------------------

class UltraSparc: public TargetMachine {
public:
  /*ctor*/		UltraSparc	();
  /*dtor*/ virtual	~UltraSparc	();
};


/***************************************************************************/

#endif
