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

//************************** System Include Files **************************/

//*************************** User Include Files ***************************/

#include "llvm/CodeGen/TargetMachine.h"
#include "llvm/CodeGen/MachineInstr.h"


//************************* Opaque Declarations ****************************/


//************************ Exported Constants ******************************/


// OpCodeMask definitions for the Sparc V9
// 
const OpCodeMask	Immed		= 0x00002000; // immed or reg operand?
const OpCodeMask	Annul		= 0x20000000; // annul delay instr?
const OpCodeMask	PredictTaken	= 0x00080000; // predict branch taken?


//************************ Exported Data Types *****************************/


//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Machine description.
// 
//---------------------------------------------------------------------------

class UltraSparc: public TargetMachine {
public:
  /*ctor*/		UltraSparc	();
  /*dtor*/ virtual	~UltraSparc	() {}
};


//---------------------------------------------------------------------------
// enum SparcMachineOpCode. 
// const MachineInstrInfo SparcMachineInstrInfo[].
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

const MachineInstrInfo SparcMachineInstrInfo[] = {
  
  { "NOP",	0,  -1,  0,  false },
  
  // Synthetic SPARC assembly opcodes for setting a register to a constant
  { "SETSW",	2,  1,  0, true },	// max immediate constant is ignored
  { "SETUW",	2,  1,  0, false },	// max immediate constant is ignored
  
  // Add or add with carry.
  { "ADD",	3,  2,  (1 << 12) - 1,  true },
  { "ADDcc",	3,  2,  (1 << 12) - 1,  true },
  { "ADDC",	3,  2,  (1 << 12) - 1,  true },
  { "ADDCcc",	3,  2,  (1 << 12) - 1,  true },

  // Sub tract or subtract with carry.
  { "SUB",	3,  2,  (1 << 12) - 1,  true },
  { "SUBcc",	3,  2,  (1 << 12) - 1,  true },
  { "SUBC",	3,  2,  (1 << 12) - 1,  true },
  { "SUBCcc",	3,  2,  (1 << 12) - 1,  true },

  // Integer multiply, signed divide, unsigned divide.
  // Note that the deprecated 32-bit multiply and multiply-step are not used.
  { "MULX",	3,  2,  (1 << 12) - 1,  true },
  { "SDIVX",	3,  2,  (1 << 12) - 1,  true },
  { "UDIVX",	3,  2,  (1 << 12) - 1,  true },
  
  // Floating point add, subtract, compare
  { "FADDS",	3,  2,  0,  false },
  { "FADDD",	3,  2,  0,  false },
  { "FADDQ",	3,  2,  0,  false },
  { "FSUBS",	3,  2,  0,  false },
  { "FSUBD",	3,  2,  0,  false },
  { "FSUBQ",	3,  2,  0,  false },
  { "FCMPS",	3,  2,  0,  false },
  { "FCMPD",	3,  2,  0,  false },
  { "FCMPQ",	3,  2,  0,  false },
  // NOTE: FCMPE{S,D,Q}: FP Compare With Exception are currently unused!
  
  // Floating point multiply or divide.
  { "FMULS",	3,  2,  0,  false },
  { "FMULD",	3,  2,  0,  false },
  { "FMULQ",	3,  2,  0,  false },
  { "FSMULD",	3,  2,  0,  false },
  { "FDMULQ",	3,  2,  0,  false },
  { "FDIVS",	3,  2,  0,  false },
  { "FDIVD",	3,  2,  0,  false },
  { "FDIVQ",	3,  2,  0,  false },
  
  // Logical operations
  { "AND",	3,  2,  (1 << 12) - 1,  true },
  { "ANDcc",	3,  2,  (1 << 12) - 1,  true },
  { "ANDN",	3,  2,  (1 << 12) - 1,  true },
  { "ANDNcc",	3,  2,  (1 << 12) - 1,  true },
  { "OR",	3,  2,  (1 << 12) - 1,  true },
  { "ORcc",	3,  2,  (1 << 12) - 1,  true },
  { "ORN",	3,  2,  (1 << 12) - 1,  true },
  { "ORNcc",	3,  2,  (1 << 12) - 1,  true },
  { "XOR",	3,  2,  (1 << 12) - 1,  true },
  { "XORcc",	3,  2,  (1 << 12) - 1,  true },
  { "XNOR",	3,  2,  (1 << 12) - 1,  true },
  { "XNORcc",	3,  2,  (1 << 12) - 1,  true },
  
  // Shift operations
  { "SLL",	3,  2,  (1 << 5) - 1,  true },
  { "SRL",	3,  2,  (1 << 5) - 1,  true },
  { "SRA",	3,  2,  (1 << 5) - 1,  true },
  { "SLLX",	3,  2,  (1 << 6) - 1,  true },
  { "SRLX",	3,  2,  (1 << 6) - 1,  true },
  { "SRAX",	3,  2,  (1 << 6) - 1,  true },
  
  // Convert from floating point to floating point formats
  { "FSTOD",	2,  1,  0,  false },
  { "FSTOQ",	2,  1,  0,  false },
  { "FDTOS",	2,  1,  0,  false },
  { "FDTOQ",	2,  1,  0,  false },
  { "FQTOS",	2,  1,  0,  false },
  { "FQTOD",	2,  1,  0,  false },
  
  // Convert from floating point to integer formats
  { "FSTOX",	2,  1,  0,  false },
  { "FDTOX",	2,  1,  0,  false },
  { "FQTOX",	2,  1,  0,  false },
  { "FSTOI",	2,  1,  0,  false },
  { "FDTOI",	2,  1,  0,  false },
  { "FQTOI",	2,  1,  0,  false },
  
  // Convert from integer to floating point formats
  { "FXTOS",	2,  1,  0,  false },
  { "FXTOD",	2,  1,  0,  false },
  { "FXTOQ",	2,  1,  0,  false },
  { "FITOS",	2,  1,  0,  false },
  { "FITOD",	2,  1,  0,  false },
  { "FITOQ",	2,  1,  0,  false },
  
  // Branch on integer comparison with zero.
  { "BRZ",	2,  -1,  (1 << 15) - 1,  true },
  { "BRLEZ",	2,  -1,  (1 << 15) - 1,  true },
  { "BRLZ",	2,  -1,  (1 << 15) - 1,  true },
  { "BRNZ",	2,  -1,  (1 << 15) - 1,  true },
  { "BRGZ",	2,  -1,  (1 << 15) - 1,  true },
  { "BRGEZ",	2,  -1,  (1 << 15) - 1,  true },

  // Branch on condition code.
  { "BA",	1,  -1,  (1 << 21) - 1,  true },
  { "BN",	1,  -1,  (1 << 21) - 1,  true },
  { "BNE",	1,  -1,  (1 << 21) - 1,  true },
  { "BE",	1,  -1,  (1 << 21) - 1,  true },
  { "BG",	1,  -1,  (1 << 21) - 1,  true },
  { "BLE",	1,  -1,  (1 << 21) - 1,  true },
  { "BGE",	1,  -1,  (1 << 21) - 1,  true },
  { "BL",	1,  -1,  (1 << 21) - 1,  true },
  { "BGU",	1,  -1,  (1 << 21) - 1,  true },
  { "BLEU",	1,  -1,  (1 << 21) - 1,  true },
  { "BCC",	1,  -1,  (1 << 21) - 1,  true },
  { "BCS",	1,  -1,  (1 << 21) - 1,  true },
  { "BPOS",	1,  -1,  (1 << 21) - 1,  true },
  { "BNEG",	1,  -1,  (1 << 21) - 1,  true },
  { "BVC",	1,  -1,  (1 << 21) - 1,  true },
  { "BVS",	1,  -1,  (1 << 21) - 1,  true },

  // Branch on floating point condition code.
  // Annul bit specifies if intruction in delay slot is annulled(1) or not(0).
  // PredictTaken bit hints if branch should be predicted taken(1) or not(0).
  // The first argument is the FCCn register (0 <= n <= 3).
  { "FBA",	2,  -1,  (1 << 18) - 1,  true },
  { "FBN",	2,  -1,  (1 << 18) - 1,  true },
  { "FBU",	2,  -1,  (1 << 18) - 1,  true },
  { "FBG",	2,  -1,  (1 << 18) - 1,  true },
  { "FBUG",	2,  -1,  (1 << 18) - 1,  true },
  { "FBL",	2,  -1,  (1 << 18) - 1,  true },
  { "FBUL",	2,  -1,  (1 << 18) - 1,  true },
  { "FBLG",	2,  -1,  (1 << 18) - 1,  true },
  { "FBNE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBUE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBGE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBUGE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBLE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBULE",	2,  -1,  (1 << 18) - 1,  true },
  { "FBO",	2,  -1,  (1 << 18) - 1,  true },

  // Load integer instructions
  { "LDSB",	3,  2,  (1 << 12) - 1,  true },
  { "LDSH",	3,  2,  (1 << 12) - 1,  true },
  { "LDSW",	3,  2,  (1 << 12) - 1,  true },
  { "LDUB",	3,  2,  (1 << 12) - 1,  true },
  { "LDUH",	3,  2,  (1 << 12) - 1,  true },
  { "LDUW",	3,  2,  (1 << 12) - 1,  true },
  { "LDX",	3,  2,  (1 << 12) - 1,  true },
  
  // Load floating-point instructions
  { "LD",	3,  2,  (1 << 12) - 1,  true },
  { "LDD",	3,  2,  (1 << 12) - 1,  true },
  { "LDQ",	3,  2,  (1 << 12) - 1,  true },
  
  // Store integer instructions
  { "STB",	3,  -1,  (1 << 12) - 1,  true },
  { "STH",	3,  -1,  (1 << 12) - 1,  true },
  { "STW",	3,  -1,  (1 << 12) - 1,  true },
  { "STX",	3,  -1,  (1 << 12) - 1,  true },
  
  // Store floating-point instructions
  { "ST",	3,  -1,  (1 << 12) - 1,  true },
  { "STD",	3,  -1,  (1 << 12) - 1,  true },
  
  // Call, Return and "Jump and link"
  { "CALL",	1,  -1,  (1 << 29) - 1,  true },
  { "JMPL",	3,  -1,  (1 << 12) - 1,  true },	
  { "RETURN",	2,  -1,  0,  false },
  
  // End-of-array marker
  { "INVALID_SPARC_OPCODE",	0,  -1,  0,  false }
};


/***************************************************************************/

#endif
