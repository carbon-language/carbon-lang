//===- MicroMipsDSPInstrInfo.td - Micromips DSP instructions -*- tablegen *-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes MicroMips DSP instructions.
//
//===----------------------------------------------------------------------===//

// Instruction encoding.
class ADDU_QB_MM_ENC : POOL32A_3R_FMT<0b00011001101>;

// Instruction defs.
// MIPS DSP Rev 1
def ADDU_QB_MM : DspMMRel, ADDU_QB_MM_ENC, ADDU_QB_DESC, ISA_MICROMIPS;
