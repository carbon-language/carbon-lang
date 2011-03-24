//===-- EDInfo.h - LLVM Enhanced Disassembler -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_EDINFO_H
#define LLVM_EDINFO_H

enum {
  EDIS_MAX_OPERANDS = 13,
  EDIS_MAX_SYNTAXES = 2
};

enum OperandTypes {
  kOperandTypeNone,
  kOperandTypeImmediate,
  kOperandTypeRegister,
  kOperandTypeX86Memory,
  kOperandTypeX86EffectiveAddress,
  kOperandTypeX86PCRelative,
  kOperandTypeARMBranchTarget,
  kOperandTypeARMSoReg,
  kOperandTypeARMSoImm,
  kOperandTypeARMSoImm2Part,
  kOperandTypeARMPredicate,
  kOperandTypeARMAddrMode2,
  kOperandTypeARMAddrMode2Offset,
  kOperandTypeARMAddrMode3,
  kOperandTypeARMAddrMode3Offset,
  kOperandTypeARMAddrMode4,
  kOperandTypeARMAddrMode5,
  kOperandTypeARMAddrMode6,
  kOperandTypeARMAddrMode6Offset,
  kOperandTypeARMAddrMode7,
  kOperandTypeARMAddrModePC,
  kOperandTypeARMRegisterList,
  kOperandTypeARMTBAddrMode,
  kOperandTypeThumbITMask,
  kOperandTypeThumbAddrModeS1,
  kOperandTypeThumbAddrModeS2,
  kOperandTypeThumbAddrModeS4,
  kOperandTypeThumbAddrModeRR,
  kOperandTypeThumbAddrModeSP,
  kOperandTypeThumb2SoReg,
  kOperandTypeThumb2SoImm,
  kOperandTypeThumb2AddrModeImm8,
  kOperandTypeThumb2AddrModeImm8Offset,
  kOperandTypeThumb2AddrModeImm12,
  kOperandTypeThumb2AddrModeSoReg,
  kOperandTypeThumb2AddrModeImm8s4,
  kOperandTypeThumb2AddrModeImm8s4Offset,
  kOperandTypeThumb2AddrModeReg
};

enum OperandFlags {
  kOperandFlagSource = 0x1,
  kOperandFlagTarget = 0x2
};

enum InstructionTypes {
  kInstructionTypeNone,
  kInstructionTypeMove,
  kInstructionTypeBranch,
  kInstructionTypePush,
  kInstructionTypePop,
  kInstructionTypeCall,
  kInstructionTypeReturn
};


#endif
