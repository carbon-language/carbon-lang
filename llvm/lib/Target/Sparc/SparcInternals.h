//===-- SparcInternals.h - Header file for Sparc backend ---------*- C++ -*--=//
//
// This file defines stuff that is to be private to the Sparc backend, but is 
// shared among different portions of the backend.
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_INTERNALS_H
#define SPARC_INTERNALS_H

#include "SparcRegInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/MachineSchedInfo.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Type.h"

#include <sys/types.h>

class UltraSparc;

// OpCodeMask definitions for the Sparc V9
// 
const OpCodeMask	Immed		= 0x00002000; // immed or reg operand?
const OpCodeMask	Annul		= 0x20000000; // annul delay instr?
const OpCodeMask	PredictTaken	= 0x00080000; // predict branch taken?


enum SparcInstrSchedClass {
  SPARC_NONE,		/* Instructions with no scheduling restrictions */
  SPARC_IEUN,		/* Integer class that can use IEU0 or IEU1 */
  SPARC_IEU0,		/* Integer class IEU0 */
  SPARC_IEU1,		/* Integer class IEU1 */
  SPARC_FPM,		/* FP Multiply or Divide instructions */
  SPARC_FPA,		/* All other FP instructions */	
  SPARC_CTI,		/* Control-transfer instructions */
  SPARC_LD,		/* Load instructions */
  SPARC_ST,		/* Store instructions */
  SPARC_SINGLE,		/* Instructions that must issue by themselves */
  
  SPARC_INV,		/* This should stay at the end for the next value */
  SPARC_NUM_SCHED_CLASSES = SPARC_INV
};


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
  FSQRTS,
  FSQRTD,
  FSQRTQ,
  
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
  
  // Floating point move, negate, and abs instructions
  FMOVS,
  FMOVD,
//FMOVQ,
  FNEGS,
  FNEGD,
//FNEGQ,
  FABSS,
  FABSD,
//FABSQ,
  
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

  // Conditional move of floating point register on each of the above:
  // i.   on integer comparison with zero.
  // ii.  on integer condition code
  // iii. on floating point condition code
  // Note that the same set is repeated for S,D,Q register classes.
  FMOVRSZ,
  FMOVRSLEZ,
  FMOVRSLZ,
  FMOVRSNZ,
  FMOVRSGZ,
  FMOVRSGEZ,

  FMOVSA,
  FMOVSN,
  FMOVSNE,
  FMOVSE,
  FMOVSG,
  FMOVSLE,
  FMOVSGE,
  FMOVSL,
  FMOVSGU,
  FMOVSLEU,
  FMOVSCC,
  FMOVSCS,
  FMOVSPOS,
  FMOVSNEG,
  FMOVSVC,
  FMOVSVS,

  FMOVSFA,
  FMOVSFN,
  FMOVSFU,
  FMOVSFG,
  FMOVSFUG,
  FMOVSFL,
  FMOVSFUL,
  FMOVSFLG,
  FMOVSFNE,
  FMOVSFE,
  FMOVSFUE,
  FMOVSFGE,
  FMOVSFUGE,
  FMOVSFLE,
  FMOVSFULE,
  FMOVSFO,
  
  FMOVRDZ,
  FMOVRDLEZ,
  FMOVRDLZ,
  FMOVRDNZ,
  FMOVRDGZ,
  FMOVRDGEZ,

  FMOVDA,
  FMOVDN,
  FMOVDNE,
  FMOVDE,
  FMOVDG,
  FMOVDLE,
  FMOVDGE,
  FMOVDL,
  FMOVDGU,
  FMOVDLEU,
  FMOVDCC,
  FMOVDCS,
  FMOVDPOS,
  FMOVDNEG,
  FMOVDVC,
  FMOVDVS,

  FMOVDFA,
  FMOVDFN,
  FMOVDFU,
  FMOVDFG,
  FMOVDFUG,
  FMOVDFL,
  FMOVDFUL,
  FMOVDFLG,
  FMOVDFNE,
  FMOVDFE,
  FMOVDFUE,
  FMOVDFGE,
  FMOVDFUGE,
  FMOVDFLE,
  FMOVDFULE,
  FMOVDFO,
  
  FMOVRQZ,
  FMOVRQLEZ,
  FMOVRQLZ,
  FMOVRQNZ,
  FMOVRQGZ,
  FMOVRQGEZ,

  FMOVQA,
  FMOVQN,
  FMOVQNE,
  FMOVQE,
  FMOVQG,
  FMOVQLE,
  FMOVQGE,
  FMOVQL,
  FMOVQGU,
  FMOVQLEU,
  FMOVQCC,
  FMOVQCS,
  FMOVQPOS,
  FMOVQNEG,
  FMOVQVC,
  FMOVQVS,

  FMOVQFA,
  FMOVQFN,
  FMOVQFU,
  FMOVQFG,
  FMOVQFUG,
  FMOVQFL,
  FMOVQFUL,
  FMOVQFLG,
  FMOVQFNE,
  FMOVQFE,
  FMOVQFUE,
  FMOVQFGE,
  FMOVQFUGE,
  FMOVQFLE,
  FMOVQFULE,
  FMOVQFO,
  
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
  RETURN,				// last valid opcode

  // Synthetic phi operation for near-SSA form of machine code
  PHI,
  
  // End-of-array marker
  INVALID_OPCODE,
  NUM_REAL_OPCODES = RETURN+1,		// number of valid opcodes
  NUM_TOTAL_OPCODES = INVALID_OPCODE
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
  //					    instr sched class (defined above)
  //						instr class flags (defined in TargretMachine.h)
  
  { "NOP",	0,  -1,  0,  false, 0,	1,  SPARC_NONE,  M_NOP_FLAG },
  
  // Synthetic SPARC assembly opcodes for setting a register to a constant.
  // Max immediate constant should be ignored for both these instructions.
  { "SETSW",	2,   1,  0,  true,  0,  1,  SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "SETUW",	2,   1,  0,  false, 0,  1,  SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG | M_ARITH_FLAG },

  // Set high-order bits of register and clear low-order bits
  { "SETHI",	2,  1,  (1 << 22) - 1, false, 0,  1,  SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG | M_ARITH_FLAG },
  
  // Add or add with carry.
  { "ADD",	3,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "ADDcc",	4,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_ARITH_FLAG },
  { "ADDC",	3,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "ADDCcc",	4,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_ARITH_FLAG },

  // Sub tract or subtract with carry.
  { "SUB",	3,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "SUBcc",	4,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_ARITH_FLAG },
  { "SUBC",	3,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "SUBCcc",	4,  2,  (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_ARITH_FLAG },

  // Integer multiply, signed divide, unsigned divide.
  // Note that the deprecated 32-bit multiply and multiply-step are not used.
  { "MULX",	3,  2,  (1 << 12) - 1, true, 0, 3, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "SDIVX",	3,  2,  (1 << 12) - 1, true, 0, 6, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  { "UDIVX",	3,  2,  (1 << 12) - 1, true, 0, 6, SPARC_IEUN,  M_INT_FLAG | M_ARITH_FLAG },
  
  // Floating point add, subtract, compare.
  // Note that destination of FCMP* instructions is operand 0, not operand 2.
  { "FADDS",	3,  2,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FADDD",	3,  2,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FADDQ",	3,  2,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSUBS",	3,  2,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSUBD",	3,  2,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSUBQ",	3,  2,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FCMPS",	3,  0,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FCMPD",	3,  0,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FCMPQ",	3,  0,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  // NOTE: FCMPE{S,D,Q}: FP Compare With Exception are currently unused!
  
  // Floating point multiply or divide.
  { "FMULS",	3,  2,  0,  false, 0, 3,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FMULD",	3,  2,  0,  false, 0, 3,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FMULQ",	3,  2,  0,  false, 0, 0,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSMULD",	3,  2,  0,  false, 0, 3,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDMULQ",	3,  2,  0,  false, 0, 0,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDIVS",	3,  2,  0,  false, 0, 12, SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDIVD",	3,  2,  0,  false, 0, 22, SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDIVQ",	3,  2,  0,  false, 0, 0,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSQRTS",	3,  2,  0,  false, 0, 12, SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSQRTD",	3,  2,  0,  false, 0, 22, SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSQRTQ",	3,  2,  0,  false, 0, 0,  SPARC_FPM,  M_FLOAT_FLAG | M_ARITH_FLAG },
  
  // Logical operations
  { "AND",	3,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "ANDcc",	4,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "ANDN",	3,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "ANDNcc",	4,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "OR",	3,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "ORcc",	4,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "ORN",	3,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "ORNcc",	4,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "XOR",	3,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "XORcc",	4,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "XNOR",	3,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEUN,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "XNORcc",	4,  2, (1 << 12) - 1, true, 0, 1, SPARC_IEU1,  M_INT_FLAG | M_LOGICAL_FLAG},
  
  // Shift operations
  { "SLL",	3,  2, (1 << 5) - 1, true, 0, 1, SPARC_IEU0,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRL",	3,  2, (1 << 5) - 1, true, 0, 1, SPARC_IEU0,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRA",	3,  2, (1 << 5) - 1, true, 0, 1, SPARC_IEU0,  M_INT_FLAG | M_ARITH_FLAG },
  { "SLLX",	3,  2, (1 << 6) - 1, true, 0, 1, SPARC_IEU0,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRLX",	3,  2, (1 << 6) - 1, true, 0, 1, SPARC_IEU0,  M_INT_FLAG | M_LOGICAL_FLAG},
  { "SRAX",	3,  2, (1 << 6) - 1, true, 0, 1, SPARC_IEU0,  M_INT_FLAG | M_ARITH_FLAG },
  
  // Floating point move, negate, and abs instructions
  { "FMOVS",	2,  1,  0,  false,  0,  1,  SPARC_FPA,  M_FLOAT_FLAG },
  { "FMOVD",	2,  1,  0,  false,  0,  1,  SPARC_FPA,  M_FLOAT_FLAG },
//{ "FMOVQ",	2,  1,  0,  false,  0,  ?,  SPARC_FPA,  M_FLOAT_FLAG },
  { "FNEGS",	2,  1,  0,  false,  0,  1,  SPARC_FPA,  M_FLOAT_FLAG },
  { "FNEGD",	2,  1,  0,  false,  0,  1,  SPARC_FPA,  M_FLOAT_FLAG },
//{ "FNEGQ",	2,  1,  0,  false,  0,  ?,  SPARC_FPA,  M_FLOAT_FLAG },
  { "FABSS",	2,  1,  0,  false,  0,  1,  SPARC_FPA,  M_FLOAT_FLAG },
  { "FABSD",	2,  1,  0,  false,  0,  1,  SPARC_FPA,  M_FLOAT_FLAG },
//{ "FABSQ",	2,  1,  0,  false,  0,  ?,  SPARC_FPA,  M_FLOAT_FLAG },
  
  // Convert from floating point to floating point formats
  { "FSTOD",	2,  1,  0,  false,  0,  3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FSTOQ",	2,  1,  0,  false,  0,  0,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDTOS",	2,  1,  0,  false,  0,  3,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FDTOQ",	2,  1,  0,  false,  0,  0,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FQTOS",	2,  1,  0,  false,  0,  0,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  { "FQTOD",	2,  1,  0,  false,  0,  0,  SPARC_FPA,  M_FLOAT_FLAG | M_ARITH_FLAG },
  
  // Convert from floating point to integer formats.
  // Note that this accesses both integer and floating point registers.
  { "FSTOX",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FDTOX",	2,  1,  0,  false, 0, 0,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FQTOX",	2,  1,  0,  false, 0, 2,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FSTOI",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FDTOI",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FQTOI",	2,  1,  0,  false, 0, 0,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  
  // Convert from integer to floating point formats
  // Note that this accesses both integer and floating point registers.
  { "FXTOS",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FXTOD",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FXTOQ",	2,  1,  0,  false, 0, 0,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FITOS",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FITOD",	2,  1,  0,  false, 0, 3,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  { "FITOQ",	2,  1,  0,  false, 0, 0,  SPARC_FPA,  M_FLOAT_FLAG | M_INT_FLAG | M_ARITH_FLAG },
  
  // Branch on integer comparison with zero.
  // Latency includes the delay slot.
  { "BRZ",	2, -1, (1 << 15) - 1, true, 1, 2,  SPARC_CTI,  M_INT_FLAG | M_BRANCH_FLAG },
  { "BRLEZ",	2, -1, (1 << 15) - 1, true, 1, 2,  SPARC_CTI,  M_INT_FLAG | M_BRANCH_FLAG },
  { "BRLZ",	2, -1, (1 << 15) - 1, true, 1, 2,  SPARC_CTI,  M_INT_FLAG | M_BRANCH_FLAG },
  { "BRNZ",	2, -1, (1 << 15) - 1, true, 1, 2,  SPARC_CTI,  M_INT_FLAG | M_BRANCH_FLAG },
  { "BRGZ",	2, -1, (1 << 15) - 1, true, 1, 2,  SPARC_CTI,  M_INT_FLAG | M_BRANCH_FLAG },
  { "BRGEZ",	2, -1, (1 << 15) - 1, true, 1, 2,  SPARC_CTI,  M_INT_FLAG | M_BRANCH_FLAG },

  // Branch on condition code.
  // The first argument specifies the ICC register: %icc or %xcc
  // Latency includes the delay slot.
  { "BA",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BN",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BNE",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BE",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BG",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BLE",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BGE",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BL",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BGU",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BLEU",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BCC",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BCS",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BPOS",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BNEG",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BVC",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "BVS",	2,  -1, (1 << 21) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },

  // Branch on floating point condition code.
  // Annul bit specifies if intruction in delay slot is annulled(1) or not(0).
  // PredictTaken bit hints if branch should be predicted taken(1) or not(0).
  // The first argument is the FCCn register (0 <= n <= 3).
  // Latency includes the delay slot.
  { "FBA",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBN",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBU",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBG",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUG",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBL",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUL",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBLG",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBNE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBGE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBUGE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBLE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBULE",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },
  { "FBO",	2,  -1, (1 << 18) - 1, true, 1, 2,  SPARC_CTI,  M_CC_FLAG | M_BRANCH_FLAG },

  // Conditional move on integer comparison with zero.
  { "MOVRZ",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRLEZ",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRLZ",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRNZ",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRGZ",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_INT_FLAG },
  { "MOVRGEZ",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_INT_FLAG },

  // Conditional move on integer condition code.
  // The first argument specifies the ICC register: %icc or %xcc
  { "MOVA",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVN",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVNE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVG",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVLE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVGE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVL",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVGU",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVLEU",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVCC",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVCS",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVPOS",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVNEG",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVVC",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVVS",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },

  // Conditional move (of integer register) on floating point condition code.
  // The first argument is the FCCn register (0 <= n <= 3).
  // Note that the enum name above is not the same as the assembly mnemonic
  // because some of the assembly mnemonics are the same as the move on
  // integer CC (e.g., MOVG), and we cannot have the same enum entry twice.
  { "MOVA",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVN",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVU",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVG",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVUG",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVL",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVUL",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVLG",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVNE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVUE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVGE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVUGE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVLE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVULE",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },
  { "MOVO",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_INT_FLAG },

  // Conditional move of floating point register on each of the above:
  // i.   on integer comparison with zero.
  // ii.  on integer condition code
  // iii. on floating point condition code
  // Note that the same set is repeated for S,D,Q register classes.
  { "FMOVRSZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRSLEZ",3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRSLZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRSNZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRSGZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRSGEZ",3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },

  { "FMOVSA",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSN",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSNE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSLE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSGU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSLEU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSCC",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSCS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSPOS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSNEG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSVC",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSVS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },

  { "FMOVSA",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSN",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSUG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSUL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSLG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSNE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSUE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSUGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSLE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSULE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVSO",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },

  { "FMOVRDZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRDLEZ",3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRDLZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRDNZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRDGZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRDGEZ",3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },

  { "FMOVDA",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDN",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDNE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDLE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDGU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDLEU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDCC",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDCS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDPOS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDNEG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDVC",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDVS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },

  { "FMOVDA",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDN",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDUG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDUL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDLG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDNE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDUE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDUGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDLE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDULE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVDO",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },

  { "FMOVRQZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRQLEZ",3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRQLZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRQNZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRQGZ",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },
  { "FMOVRQGEZ",3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CONDL_FLAG | M_FLOAT_FLAG | M_INT_FLAG },

  { "FMOVQA",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQN",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQNE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQLE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQGU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQLEU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQCC",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQCS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQPOS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQNEG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQVC",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQVS",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },

  { "FMOVQA",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQN",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQU",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQUG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQUL",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQLG",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQNE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQUE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQUGE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQLE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQULE",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  { "FMOVQO",	3,  2, 0,  false, 0, 2,  SPARC_SINGLE,  M_CC_FLAG | M_FLOAT_FLAG },
  
  // Load integer instructions
  // Latency includes 1 cycle for address generation (Sparc IIi)
  // Signed loads of less than 64 bits need an extra cycle for sign-extension.
  //
  // Not reflected here: After a 3-cycle loads, all subsequent consecutive
  // loads also require 3 cycles to avoid contention for the load return
  // stage.  Latency returns to 2 cycles after the first cycle with no load.
  { "LDSB",	3,  2, (1 << 12) - 1,  true, 0, 3,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  { "LDSH",	3,  2, (1 << 12) - 1,  true, 0, 3,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  { "LDSW",	3,  2, (1 << 12) - 1,  true, 0, 3,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  { "LDUB",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  { "LDUH",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  { "LDUW",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  { "LDX",	3,  2, (1 << 12) - 1,  true, 0, 2,  SPARC_LD,  M_INT_FLAG | M_LOAD_FLAG },
  
  // Load floating-point instructions
  // Latency includes 1 cycle for address generation (Sparc IIi)
  { "LD",	3, 2, (1 << 12) - 1, true, 0, 2,  SPARC_LD,  M_FLOAT_FLAG | M_LOAD_FLAG },
  { "LDD",	3, 2, (1 << 12) - 1, true, 0, 2,  SPARC_LD,  M_FLOAT_FLAG | M_LOAD_FLAG },
  { "LDQ",	3, 2, (1 << 12) - 1, true, 0, 2,  SPARC_LD,  M_FLOAT_FLAG | M_LOAD_FLAG },
  
  // Store integer instructions
  // Latency includes 1 cycle for address generation (Sparc IIi)
  { "STB",	3, -1, (1 << 12) - 1, true, 0, 2,  SPARC_ST,  M_INT_FLAG | M_STORE_FLAG },
  { "STH",	3, -1, (1 << 12) - 1, true, 0, 2,  SPARC_ST,  M_INT_FLAG | M_STORE_FLAG },
  { "STW",	3, -1, (1 << 12) - 1, true, 0, 2,  SPARC_ST,  M_INT_FLAG | M_STORE_FLAG },
  { "STX",	3, -1, (1 << 12) - 1, true, 0, 3,  SPARC_ST,  M_INT_FLAG | M_STORE_FLAG },
  
  // Store floating-point instructions (Sparc IIi)
  { "ST",	3, -1, (1 << 12) - 1, true, 0, 2,  SPARC_ST,  M_FLOAT_FLAG | M_STORE_FLAG},
  { "STD",	3, -1, (1 << 12) - 1, true, 0, 2,  SPARC_ST,  M_FLOAT_FLAG | M_STORE_FLAG},
  
  // Call, Return and "Jump and link".
  // Latency includes the delay slot.
  { "CALL",	1, -1, (1 << 29) - 1, true, 1, 2,  SPARC_CTI,  M_BRANCH_FLAG | M_CALL_FLAG},
  { "JMPL",	3, -1, (1 << 12) - 1, true, 1, 2,  SPARC_CTI,  M_BRANCH_FLAG | M_CALL_FLAG},
  { "RETURN",	2, -1,  0,           false, 1, 2,  SPARC_CTI,  M_BRANCH_FLAG | M_RET_FLAG },
  
  // Synthetic phi operation for near-SSA form of machine code
  // Number of operands is variable, indicated by -1.  Result is the first op.

  { "PHI",	-1,  0,  0,  false, 0, 0, SPARC_INV,  M_DUMMY_PHI_FLAG },

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
  /*ctor*/	UltraSparcInstrInfo();
  
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
// class UltraSparcRegInfo 
// 
// Purpose:
//   This class provides info about sparc register classes.
//---------------------------------------------------------------------------

class LiveRange;
class UltraSparc;


class UltraSparcRegInfo : public MachineRegInfo
{

 private:

  enum RegClassIDs { 
    IntRegClassID, 
    FloatRegClassID, 
    IntCCRegClassID,
    FloatCCRegClassID 
  };

  // WARNING: If the above enum order must be changed, also modify 
  // getRegisterClassOfValue method below since it assumes this particular 
  // order for efficiency.


  // reverse pointer to get info about the ultra sparc machine
  const UltraSparc *const UltraSparcInfo;

  // Int arguments can be passed in 6 int regs - %o0 to %o5 (cannot be changed)
  unsigned const NumOfIntArgRegs;

  // Float arguments can be passed in this many regs - can be canged if needed
  // %f0 - %f5 are used (can hold 6 floats or 3 doubles)
  unsigned const NumOfFloatArgRegs;

  void setCallArgColor(LiveRange *const LR, const unsigned RegNo) const;


 public:


  UltraSparcRegInfo(const UltraSparc *const USI )
    : MachineRegInfo(),
      UltraSparcInfo(USI), 
      NumOfIntArgRegs(6), 
      NumOfFloatArgRegs(6) 
  {    
    MachineRegClassArr.push_back( new SparcIntRegClass(IntRegClassID) );
    MachineRegClassArr.push_back( new SparcFloatRegClass(FloatRegClassID) );
    MachineRegClassArr.push_back( new SparcIntCCRegClass(IntCCRegClassID) );
    MachineRegClassArr.push_back( new SparcFloatCCRegClass(FloatCCRegClassID));
    
    assert( SparcFloatRegOrder::StartOfNonVolatileRegs == 6 && 
	    "6 Float regs are used for float arg passing");
  }
  
  // ***** TODO  Delete
  ~UltraSparcRegInfo(void) { }              // empty destructor 


  inline const UltraSparc & getUltraSparcInfo() const { 
    return *UltraSparcInfo;
  }

  // returns the register that is hardwired to zero
  virtual inline int getZeroRegNum() const {
    return (int) SparcIntRegOrder::g0;
  }
  

  inline unsigned getRegClassIDOfValue (const Value *const Val,
					bool isCCReg = false) const {

    Type::PrimitiveID ty = (Val->getType())->getPrimitiveID();

    unsigned res;
    
    if( ty && ty <= Type::LongTyID || (ty == Type::PointerTyID) )  
      res =  IntRegClassID;             // sparc int reg (ty=0: void)
    else if( ty <= Type::DoubleTyID)
      res = FloatRegClassID;           // sparc float reg class
    else { 
      cout << "TypeID: " << ty << endl;
      assert(0 && "Cannot resolve register class for type");

    }

    if(isCCReg)
      return res + 2;      // corresponidng condition code regiser 

    else 
      return res;

  }

  void colorArgs(const Method *const Meth, LiveRangeInfo& LRI) const;

  static void printReg(const LiveRange *const LR)  ;

  void colorCallArgs(vector<const Instruction *> & CallInstrList, 
		     LiveRangeInfo& LRI, 
		     AddedInstrMapType& AddedInstrMap ) const;

  // this method provides a unique number for each register 
  inline int getUnifiedRegNum(int RegClassID, int reg) const {

    if( RegClassID == IntRegClassID && reg < 32 ) 
      return reg;
    else if ( RegClassID == FloatRegClassID && reg < 64)
      return reg + 32;                  // we have 32 int regs
    else if( RegClassID == FloatCCRegClassID && reg < 4)
      return reg + 32 + 64;             // 32 int, 64 float
    else if( RegClassID == IntCCRegClassID ) 
      return 4+ 32 + 64;                // only int cc reg
    else  
      assert(0 && "Invalid register class or reg number");

  }

  // given the unified register number, this gives the name
  inline const string getUnifiedRegName(int reg) const {

    if( reg < 32 ) 
      return SparcIntRegOrder::getRegName(reg);
    else if ( reg < (64 + 32) )
      return SparcFloatRegOrder::getRegName( reg  - 32);                  
    else if( reg < (64+32+4) )
      return SparcFloatCCRegOrder::getRegName( reg -32 - 64);
    else if ( reg == 64+32+4)
      return "xcc";                     // only integer cc reg
    else 
      assert(0 && "Invalid register number");
  }


};




/*---------------------------------------------------------------------------
Scheduling guidelines for SPARC IIi:

I-Cache alignment rules (pg 326)
-- Align a branch target instruction so that it's entire group is within
   the same cache line (may be 1-4 instructions).
** Don't let a branch that is predicted taken be the last instruction
   on an I-cache line: delay slot will need an entire line to be fetched
-- Make a FP instruction or a branch be the 4th instruction in a group.
   For branches, there are tradeoffs in reordering to make this happen
   (see pg. 327).
** Don't put a branch in a group that crosses a 32-byte boundary!
   An artificial branch is inserted after every 32 bytes, and having
   another branch will force the group to be broken into 2 groups. 

iTLB rules:
-- Don't let a loop span two memory pages, if possible

Branch prediction performance:
-- Don't make the branch in a delay slot the target of a branch
-- Try not to have 2 predicted branches within a group of 4 instructions
   (because each such group has a single branch target field).
-- Try to align branches in slots 0, 2, 4 or 6 of a cache line (to avoid
   the wrong prediction bits being used in some cases).

D-Cache timing constraints:
-- Signed int loads of less than 64 bits have 3 cycle latency, not 2
-- All other loads that hit in D-Cache have 2 cycle latency
-- All loads are returned IN ORDER, so a D-Cache miss will delay a later hit
-- Mis-aligned loads or stores cause a trap.  In particular, replace
   mis-aligned FP double precision l/s with 2 single-precision l/s.
-- Simulations of integer codes show increase in avg. group size of
   33% when code (including esp. non-faulting loads) is moved across
   one branch, and 50% across 2 branches.

E-Cache timing constraints:
-- Scheduling for E-cache (D-Cache misses) is effective (due to load buffering)

Store buffer timing constraints:
-- Stores can be executed in same cycle as instruction producing the value
-- Stores are buffered and have lower priority for E-cache until
   highwater mark is reached in the store buffer (5 stores)

Pipeline constraints:
-- Shifts can only use IEU0.
-- CC setting instructions can only use IEU1.
-- Several other instructions must only use IEU1:
   EDGE(?), ARRAY(?), CALL, JMPL, BPr, PST, and FCMP.
-- Two instructions cannot store to the same register file in a single cycle
   (single write port per file).

Issue and grouping constraints:
-- FP and branch instructions must use slot 4.
-- Shift instructions cannot be grouped with other IEU0-specific instructions.
-- CC setting instructions cannot be grouped with other IEU1-specific instrs.
-- Several instructions must be issued in a single-instruction group:
	MOVcc or MOVr, MULs/x and DIVs/x, SAVE/RESTORE, many others
-- A CALL or JMPL breaks a group, ie, is not combined with subsequent instrs.
-- 
-- 

Branch delay slot scheduling rules:
-- A CTI couple (two back-to-back CTI instructions in the dynamic stream)
   has a 9-instruction penalty: the entire pipeline is flushed when the
   second instruction reaches stage 9 (W-Writeback).
-- Avoid putting multicycle instructions, and instructions that may cause
   load misses, in the delay slot of an annulling branch.
-- Avoid putting WR, SAVE..., RESTORE and RETURN instructions in the
   delay slot of an annulling branch.

 *--------------------------------------------------------------------------- */

//---------------------------------------------------------------------------
// List of CPUResources for UltraSPARC IIi.
//---------------------------------------------------------------------------

const CPUResource  AllIssueSlots(   "All Instr Slots", 4);
const CPUResource  IntIssueSlots(   "Int Instr Slots", 3);
const CPUResource  First3IssueSlots("Instr Slots 0-3", 3);
const CPUResource  LSIssueSlots(    "Load-Store Instr Slot", 1);
const CPUResource  CTIIssueSlots(   "Ctrl Transfer Instr Slot", 1);
const CPUResource  FPAIssueSlots(   "Int Instr Slot 1", 1);
const CPUResource  FPMIssueSlots(   "Int Instr Slot 1", 1);

// IEUN instructions can use either Alu and should use IAluN.
// IEU0 instructions must use Alu 1 and should use both IAluN and IAlu0. 
// IEU1 instructions must use Alu 2 and should use both IAluN and IAlu1. 
const CPUResource  IAluN("Int ALU 1or2", 2);
const CPUResource  IAlu0("Int ALU 1",    1);
const CPUResource  IAlu1("Int ALU 2",    1);

const CPUResource  LSAluC1("Load/Store Unit Addr Cycle", 1);
const CPUResource  LSAluC2("Load/Store Unit Issue Cycle", 1);
const CPUResource  LdReturn("Load Return Unit", 1);

const CPUResource  FPMAluC1("FP Mul/Div Alu Cycle 1", 1);
const CPUResource  FPMAluC2("FP Mul/Div Alu Cycle 2", 1);
const CPUResource  FPMAluC3("FP Mul/Div Alu Cycle 3", 1);

const CPUResource  FPAAluC1("FP Other Alu Cycle 1", 1);
const CPUResource  FPAAluC2("FP Other Alu Cycle 2", 1);
const CPUResource  FPAAluC3("FP Other Alu Cycle 3", 1);

const CPUResource  IRegReadPorts("Int Reg ReadPorts", INT_MAX);	 // CHECK
const CPUResource  IRegWritePorts("Int Reg WritePorts", 2);	 // CHECK
const CPUResource  FPRegReadPorts("FP Reg Read Ports", INT_MAX); // CHECK
const CPUResource  FPRegWritePorts("FP Reg Write Ports", 1);	 // CHECK

const CPUResource  CTIDelayCycle( "CTI  delay cycle", 1);
const CPUResource  FCMPDelayCycle("FCMP delay cycle", 1);


//---------------------------------------------------------------------------
// const InstrClassRUsage SparcRUsageDesc[]
// 
// Purpose:
//   Resource usage information for instruction in each scheduling class.
//   The InstrRUsage Objects for individual classes are specified first.
//   Note that fetch and decode are decoupled from the execution pipelines
//   via an instr buffer, so they are not included in the cycles below.
//---------------------------------------------------------------------------

const InstrClassRUsage NoneClassRUsage = {
  SPARC_NONE,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 4,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 0,
  /* V[] */ {
    /*Cycle G */
    /*Cycle E */
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};

const InstrClassRUsage IEUNClassRUsage = {
  SPARC_IEUN,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 3,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 4,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid, 0, 1 },
                 { IntIssueSlots.rid, 0, 1 },
    /*Cycle E */ { IAluN.rid, 1, 1 },
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid, 6, 1  }
  }
};

const InstrClassRUsage IEU0ClassRUsage = {
  SPARC_IEU0,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 5,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid, 0, 1 },
                 { IntIssueSlots.rid, 0, 1 },
    /*Cycle E */ { IAluN.rid, 1, 1 },
                 { IAlu0.rid, 1, 1 },
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage IEU1ClassRUsage = {
  SPARC_IEU1,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 5,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid, 0, 1 },
               { IntIssueSlots.rid, 0, 1 },
    /*Cycle E */ { IAluN.rid, 1, 1 },
               { IAlu1.rid, 1, 1 },
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage FPMClassRUsage = {
  SPARC_FPM,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 7,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,   0, 1 },
                 { FPMIssueSlots.rid,   0, 1 },
    /*Cycle E */ { FPRegReadPorts.rid,  1, 1 },
    /*Cycle C */ { FPMAluC1.rid,        2, 1 },
    /*Cycle N1*/ { FPMAluC2.rid,        3, 1 },
    /*Cycle N1*/ { FPMAluC3.rid,        4, 1 },
    /*Cycle N1*/
    /*Cycle W */ { FPRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage FPAClassRUsage = {
  SPARC_FPA,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 7,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,   0, 1 },
                 { FPAIssueSlots.rid,   0, 1 },
    /*Cycle E */ { FPRegReadPorts.rid,  1, 1 },
    /*Cycle C */ { FPAAluC1.rid,        2, 1 },
    /*Cycle N1*/ { FPAAluC2.rid,        3, 1 },
    /*Cycle N1*/ { FPAAluC3.rid,        4, 1 },
    /*Cycle N1*/
    /*Cycle W */ { FPRegWritePorts.rid, 6, 1 }
  }
};

const InstrClassRUsage LDClassRUsage = {
  SPARC_LD,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2, },
  
  /*numEntries*/ 6,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
                 { First3IssueSlots.rid, 0, 1 },
                 { LSIssueSlots.rid,     0, 1 },
    /*Cycle E */ { LSAluC1.rid,          1, 1 },
    /*Cycle C */ { LSAluC2.rid,          2, 1 },
                 { LdReturn.rid,         2, 1 },
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */ { IRegWritePorts.rid,   6, 1 }
  }
};

const InstrClassRUsage STClassRUsage = {
  SPARC_ST,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 3,
  /* feasibleSlots[] */ { 0, 1, 2 },
  
  /*numEntries*/ 4,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
                 { First3IssueSlots.rid, 0, 1 },
                 { LSIssueSlots.rid,     0, 1 },
    /*Cycle E */ { LSAluC1.rid,          1, 1 },
    /*Cycle C */ { LSAluC2.rid,          2, 1 }
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};

const InstrClassRUsage CTIClassRUsage = {
  SPARC_CTI,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ false,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 4,
  /* feasibleSlots[] */ { 0, 1, 2, 3 },
  
  /*numEntries*/ 4,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
		 { CTIIssueSlots.rid,    0, 1 },
    /*Cycle E */ { IAlu0.rid,            1, 1 },
    /*Cycles E-C */ { CTIDelayCycle.rid, 1, 2 }
    /*Cycle C */             
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};

const InstrClassRUsage SingleClassRUsage = {
  SPARC_SINGLE,
  /*totCycles*/ 7,
  
  /* maxIssueNum */ 1,
  /* isSingleIssue */ true,
  /* breaksGroup */ false,
  /* numBubbles */ 0,
  
  /*numSlots*/ 1,
  /* feasibleSlots[] */ { 0 },
  
  /*numEntries*/ 5,
  /* V[] */ {
    /*Cycle G */ { AllIssueSlots.rid,    0, 1 },
                 { AllIssueSlots.rid,    0, 1 },
                 { AllIssueSlots.rid,    0, 1 },
                 { AllIssueSlots.rid,    0, 1 },
    /*Cycle E */ { IAlu0.rid,            1, 1 }
    /*Cycle C */
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle N1*/
    /*Cycle W */
  }
};


const InstrClassRUsage SparcRUsageDesc[] = {
  NoneClassRUsage,
  IEUNClassRUsage,
  IEU0ClassRUsage,
  IEU1ClassRUsage,
  FPMClassRUsage,
  FPAClassRUsage,
  CTIClassRUsage,
  LDClassRUsage,
  STClassRUsage,
  SingleClassRUsage
};


//---------------------------------------------------------------------------
// const InstrIssueDelta  SparcInstrIssueDeltas[]
// 
// Purpose:
//   Changes to issue restrictions information in InstrClassRUsage for
//   instructions that differ from other instructions in their class.
//---------------------------------------------------------------------------

const InstrIssueDelta  SparcInstrIssueDeltas[] = {

  // opCode,  isSingleIssue,  breaksGroup,  numBubbles

				// Special cases for single-issue only
				// Other single issue cases are below.
//{ LDDA,	true,	true,	0 },
//{ STDA,	true,	true,	0 },
//{ LDDF,	true,	true,	0 },
//{ LDDFA,	true,	true,	0 },
  { ADDC,	true,	true,	0 },
  { ADDCcc,	true,	true,	0 },
  { SUBC,	true,	true,	0 },
  { SUBCcc,	true,	true,	0 },
//{ SAVE,	true,	true,	0 },
//{ RESTORE,	true,	true,	0 },
//{ LDSTUB,	true,	true,	0 },
//{ SWAP,	true,	true,	0 },
//{ SWAPA,	true,	true,	0 },
//{ CAS,	true,	true,	0 },
//{ CASA,	true,	true,	0 },
//{ CASX,	true,	true,	0 },
//{ CASXA,	true,	true,	0 },
//{ LDFSR,	true,	true,	0 },
//{ LDFSRA,	true,	true,	0 },
//{ LDXFSR,	true,	true,	0 },
//{ LDXFSRA,	true,	true,	0 },
//{ STFSR,	true,	true,	0 },
//{ STFSRA,	true,	true,	0 },
//{ STXFSR,	true,	true,	0 },
//{ STXFSRA,	true,	true,	0 },
//{ SAVED,	true,	true,	0 },
//{ RESTORED,	true,	true,	0 },
//{ FLUSH,	true,	true,	9 },
//{ FLUSHW,	true,	true,	9 },
//{ ALIGNADDR,	true,	true,	0 },
  { RETURN,	true,	true,	0 },
//{ DONE,	true,	true,	0 },
//{ RETRY,	true,	true,	0 },
//{ WR,		true,	true,	0 },
//{ WRPR,	true,	true,	4 },
//{ RD,		true,	true,	0 },
//{ RDPR,	true,	true,	0 },
//{ TCC,	true,	true,	0 },
//{ SHUTDOWN,	true,	true,	0 },
  
				// Special cases for breaking group *before*
				// CURRENTLY NOT SUPPORTED!
  { CALL,	false,	false,	0 },
  { JMPL,	false,	false,	0 },
  
				// Special cases for breaking the group *after*
  { MULX,	true,	true,	(4+34)/2 },
  { FDIVS,	false,	true,	0 },
  { FDIVD,	false,	true,	0 },
  { FDIVQ,	false,	true,	0 },
  { FSQRTS,	false,	true,	0 },
  { FSQRTD,	false,	true,	0 },
  { FSQRTQ,	false,	true,	0 },
//{ FCMP{LE,GT,NE,EQ}, false, true, 0 },
  
				// Instructions that introduce bubbles
//{ MULScc,	true,	true,	2 },
//{ SMULcc,	true,	true,	(4+18)/2 },
//{ UMULcc,	true,	true,	(4+19)/2 },
  { SDIVX,	true,	true,	68 },
  { UDIVX,	true,	true,	68 },
//{ SDIVcc,	true,	true,	36 },
//{ UDIVcc,	true,	true,	37 },
//{ WR,		false,	false,	4 },
//{ WRPR,	false,	false,	4 },
};


//---------------------------------------------------------------------------
// const InstrRUsageDelta SparcInstrUsageDeltas[]
// 
// Purpose:
//   Changes to resource usage information in InstrClassRUsage for
//   instructions that differ from other instructions in their class.
//---------------------------------------------------------------------------

const InstrRUsageDelta SparcInstrUsageDeltas[] = {

  // MachineOpCode, Resource, Start cycle, Num cycles

  // 
  // JMPL counts as a load/store instruction for issue!
  //
  { JMPL,     LSIssueSlots.rid,  0,  1 },
  
  // 
  // Many instructions cannot issue for the next 2 cycles after an FCMP
  // We model that with a fake resource FCMPDelayCycle.
  // 
  { FCMPS,    FCMPDelayCycle.rid, 1, 3 },
  { FCMPD,    FCMPDelayCycle.rid, 1, 3 },
  { FCMPQ,    FCMPDelayCycle.rid, 1, 3 },
  
  { MULX,     FCMPDelayCycle.rid, 1, 1 },
  { SDIVX,    FCMPDelayCycle.rid, 1, 1 },
  { UDIVX,    FCMPDelayCycle.rid, 1, 1 },
//{ SMULcc,   FCMPDelayCycle.rid, 1, 1 },
//{ UMULcc,   FCMPDelayCycle.rid, 1, 1 },
//{ SDIVcc,   FCMPDelayCycle.rid, 1, 1 },
//{ UDIVcc,   FCMPDelayCycle.rid, 1, 1 },
  { STD,      FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSZ,  FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSLEZ,FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSLZ, FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSNZ, FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSGZ, FCMPDelayCycle.rid, 1, 1 },
  { FMOVRSGEZ,FCMPDelayCycle.rid, 1, 1 },
  
  // 
  // Some instructions are stalled in the GROUP stage if a CTI is in
  // the E or C stage
  // 
  { LDD,      CTIDelayCycle.rid,  1, 1 },
//{ LDDA,     CTIDelayCycle.rid,  1, 1 },
//{ LDDSTUB,  CTIDelayCycle.rid,  1, 1 },
//{ LDDSTUBA, CTIDelayCycle.rid,  1, 1 },
//{ SWAP,     CTIDelayCycle.rid,  1, 1 },
//{ SWAPA,    CTIDelayCycle.rid,  1, 1 },
//{ CAS,      CTIDelayCycle.rid,  1, 1 },
//{ CASA,     CTIDelayCycle.rid,  1, 1 },
//{ CASX,     CTIDelayCycle.rid,  1, 1 },
//{ CASXA,    CTIDelayCycle.rid,  1, 1 },
  
  //
  // Signed int loads of less than dword size return data in cycle N1 (not C)
  // and put all loads in consecutive cycles into delayed load return mode.
  //
  { LDSB,    LdReturn.rid,  2, -1 },
  { LDSB,    LdReturn.rid,  3,  1 },
  
  { LDSH,    LdReturn.rid,  2, -1 },
  { LDSH,    LdReturn.rid,  3,  1 },
  
  { LDSW,    LdReturn.rid,  2, -1 },
  { LDSW,    LdReturn.rid,  3,  1 },


#undef EXPLICIT_BUBBLES_NEEDED
#ifdef EXPLICIT_BUBBLES_NEEDED
  // 
  // MULScc inserts one bubble.
  // This means it breaks the current group (captured in UltraSparcSchedInfo)
  // *and occupies all issue slots for the next cycle
  // 
//{ MULScc,  AllIssueSlots.rid, 2, 2-1 },
//{ MULScc,  AllIssueSlots.rid, 2, 2-1 },
//{ MULScc,  AllIssueSlots.rid, 2, 2-1 },
//{ MULScc,  AllIssueSlots.rid,  2, 2-1 },
  
  // 
  // SMULcc inserts between 4 and 18 bubbles, depending on #leading 0s in rs1.
  // We just model this with a simple average.
  // 
//{ SMULcc,  AllIssueSlots.rid, 2, ((4+18)/2)-1 },
//{ SMULcc,  AllIssueSlots.rid, 2, ((4+18)/2)-1 },
//{ SMULcc,  AllIssueSlots.rid, 2, ((4+18)/2)-1 },
//{ SMULcc,  AllIssueSlots.rid,  2, ((4+18)/2)-1 },
  
  // SMULcc inserts between 4 and 19 bubbles, depending on #leading 0s in rs1.
//{ UMULcc,  AllIssueSlots.rid, 2, ((4+19)/2)-1 },
//{ UMULcc,  AllIssueSlots.rid, 2, ((4+19)/2)-1 },
//{ UMULcc,  AllIssueSlots.rid, 2, ((4+19)/2)-1 },
//{ UMULcc,  AllIssueSlots.rid,  2, ((4+19)/2)-1 },
  
  // 
  // MULX inserts between 4 and 34 bubbles, depending on #leading 0s in rs1.
  // 
  { MULX,    AllIssueSlots.rid, 2, ((4+34)/2)-1 },
  { MULX,    AllIssueSlots.rid, 2, ((4+34)/2)-1 },
  { MULX,    AllIssueSlots.rid, 2, ((4+34)/2)-1 },
  { MULX,    AllIssueSlots.rid,  2, ((4+34)/2)-1 },
  
  // 
  // SDIVcc inserts 36 bubbles.
  // 
//{ SDIVcc,  AllIssueSlots.rid, 2, 36-1 },
//{ SDIVcc,  AllIssueSlots.rid, 2, 36-1 },
//{ SDIVcc,  AllIssueSlots.rid, 2, 36-1 },
//{ SDIVcc,  AllIssueSlots.rid,  2, 36-1 },
  
  // UDIVcc inserts 37 bubbles.
//{ UDIVcc,  AllIssueSlots.rid, 2, 37-1 },
//{ UDIVcc,  AllIssueSlots.rid, 2, 37-1 },
//{ UDIVcc,  AllIssueSlots.rid, 2, 37-1 },
//{ UDIVcc,  AllIssueSlots.rid,  2, 37-1 },
  
  // 
  // SDIVX inserts 68 bubbles.
  // 
  { SDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { SDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { SDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { SDIVX,   AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // UDIVX inserts 68 bubbles.
  // 
  { UDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { UDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { UDIVX,   AllIssueSlots.rid, 2, 68-1 },
  { UDIVX,   AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // WR inserts 4 bubbles.
  // 
//{ WR,     AllIssueSlots.rid, 2, 68-1 },
//{ WR,     AllIssueSlots.rid, 2, 68-1 },
//{ WR,     AllIssueSlots.rid, 2, 68-1 },
//{ WR,     AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // WRPR inserts 4 bubbles.
  // 
//{ WRPR,   AllIssueSlots.rid, 2, 68-1 },
//{ WRPR,   AllIssueSlots.rid, 2, 68-1 },
//{ WRPR,   AllIssueSlots.rid, 2, 68-1 },
//{ WRPR,   AllIssueSlots.rid,  2, 68-1 },
  
  // 
  // DONE inserts 9 bubbles.
  // 
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
//{ DONE,   AllIssueSlots.rid, 2, 9-1 },
  
  // 
  // RETRY inserts 9 bubbles.
  // 
//{ RETRY,   AllIssueSlots.rid, 2, 9-1 },
//{ RETRY,   AllIssueSlots.rid, 2, 9-1 },
//{ RETRY,   AllIssueSlots.rid, 2, 9-1 },
//{ RETRY,   AllIssueSlots.rid,  2, 9-1 },

#endif EXPLICIT_BUBBLES_NEEDED
};



// Additional delays to be captured in code:
// 1. RDPR from several state registers (page 349)
// 2. RD   from *any* register (page 349)
// 3. Writes to TICK, PSTATE, TL registers and FLUSH{W} instr (page 349)
// 4. Integer store can be in same group as instr producing value to store.
// 5. BICC and BPICC can be in the same group as instr producing CC (pg 350)
// 6. FMOVr cannot be in the same or next group as an IEU instr (pg 351).
// 7. The second instr. of a CTI group inserts 9 bubbles (pg 351)
// 8. WR{PR}, SVAE, SAVED, RESTORE, RESTORED, RETURN, RETRY, and DONE that
//    follow an annulling branch cannot be issued in the same group or in
//    the 3 groups following the branch.
// 9. A predicted annulled load does not stall dependent instructions.
//    Other annulled delay slot instructions *do* stall dependents, so
//    nothing special needs to be done for them during scheduling.
//10. Do not put a load use that may be annulled in the same group as the
//    branch.  The group will stall until the load returns.
//11. Single-prec. FP loads lock 2 registers, for dependency checking.
//
// 
// Additional delays we cannot or will not capture:
// 1. If DCTI is last word of cache line, it is delayed until next line can be
//    fetched.  Also, other DCTI alignment-related delays (pg 352)
// 2. Load-after-store is delayed by 7 extra cycles if load hits in D-Cache.
//    Also, several other store-load and load-store conflicts (pg 358)
// 3. MEMBAR, LD{X}FSR, LDD{A} and a bunch of other load stalls (pg 358)
// 4. There can be at most 8 outstanding buffered store instructions
//     (including some others like MEMBAR, LDSTUB, CAS{AX}, and FLUSH)



//---------------------------------------------------------------------------
// class UltraSparcSchedInfo
// 
// Purpose:
//   Interface to instruction scheduling information for UltraSPARC.
//   The parameter values above are based on UltraSPARC IIi.
//---------------------------------------------------------------------------


class UltraSparcSchedInfo: public MachineSchedInfo {
public:
  /*ctor*/	   UltraSparcSchedInfo	(const MachineInstrInfo* mii);
  /*dtor*/ virtual ~UltraSparcSchedInfo	() {}
protected:
  virtual void	initializeResources	();
};


//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as InstrInfo, SchedInfo and RegInfo. 
//---------------------------------------------------------------------------

class UltraSparc : public TargetMachine {
private:
  UltraSparcInstrInfo instrInfo;
  UltraSparcSchedInfo schedInfo;
  UltraSparcRegInfo   regInfo;
public:
  UltraSparc();
  virtual ~UltraSparc() {}
  
  virtual const MachineInstrInfo&  getInstrInfo() const { return instrInfo; }
  
  virtual const MachineSchedInfo&  getSchedInfo() const { return schedInfo; }
  
  virtual const MachineRegInfo&    getRegInfo()   const { return regInfo; }
  
  // compileMethod - For the sparc, we do instruction selection, followed by
  // delay slot scheduling, then register allocation.
  //
  virtual bool compileMethod(Method *M);
};


#endif
