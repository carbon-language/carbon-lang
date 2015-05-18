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

class CACHE_PREF_FM_MMR6<bits<6> opgroup, bits<4> funct> : MipsR6Inst {
  bits<21> addr;
  bits<5> hint;

  bits<32> Inst;

  let Inst{31-26} = opgroup;
  let Inst{25-21} = hint;
  let Inst{20-16} = addr{20-16};
  let Inst{15-12} = funct;
  let Inst{11-0}  = addr{11-0};
}

class ARITH_FM_MMR6<string instr_asm, bits<10> funct> : MMR6Arch<instr_asm> {
  bits<5> rd;
  bits<5> rt;
  bits<5> rs;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10}    = 0;
  let Inst{9-0}   = funct;
}

class ADDI_FM_MMR6<string instr_asm, bits<6> op> : MMR6Arch<instr_asm> {
  bits<5>  rt;
  bits<5>  rs;
  bits<16> imm16;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0}  = imm16;
}

class PCREL19_FM_MMR6<bits<2> funct> : MipsR6Inst {
  bits<5> rt;
  bits<19> imm;

  bits<32> Inst;

  let Inst{31-26} = 0b011110;
  let Inst{25-21} = rt;
  let Inst{20-19} = funct;
  let Inst{18-0}  = imm;
}

class PCREL16_FM_MMR6<bits<5> funct> : MipsR6Inst {
  bits<5> rt;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = 0b011110;
  let Inst{25-21} = rt;
  let Inst{20-16} = funct;
  let Inst{15-0}  = imm;
}

class POOL32A_FM_MMR6<bits<10> funct> : MipsR6Inst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10}    = 0;
  let Inst{9-0}   = funct;
}

class POOL32A_2R_FM_MMR6<bits<10> funct> : MipsR6Inst {
  bits<5> rs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0b111100;
}

class SPECIAL_2R_FM_MMR6<bits<6> funct> : MipsR6Inst {
  bits<5> rs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rs;
  let Inst{20-16} = 0b00000;
  let Inst{15-11} = rt;
  let Inst{10-6}  = 0b00001;
  let Inst{5-0}   = funct;
}

class POOL32A_ALIGN_FM_MMR6<bits<6> funct> : MipsR6Inst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<2> bp;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-9}  = bp;
  let Inst{8-6}   = 0b000;
  let Inst{5-0}   = funct;
}

class AUI_FM_MMR6 : MipsR6Inst {
  bits<5> rs;
  bits<5> rt;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = 0b000100;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-0} = imm;
}

class POOL32A_LSA_FM<bits<6> funct> : MipsR6Inst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<2> imm2;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-9}  = imm2;
  let Inst{8-6}   = 0b000;
  let Inst{5-0}   = funct;
}
