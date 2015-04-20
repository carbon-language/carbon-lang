//=- MicroMips32r6InstrFormats.td - Mips32r6 Instruction Formats -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes microMIPS32r6 instruction formats.
//
//===----------------------------------------------------------------------===//

class MMR6Arch<string opstr> {
  string Arch = "micromipsr6";
  string BaseOpcode = opstr;
}

class POOL32A_BITSWAP_FM_MMR6<bits<6> funct> : MipsR6Inst {
  bits<5> rd;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rd;
  let Inst{15-12} = 0b0000;
  let Inst{11-6} = funct;
  let Inst{5-0} = 0b111100;
}
