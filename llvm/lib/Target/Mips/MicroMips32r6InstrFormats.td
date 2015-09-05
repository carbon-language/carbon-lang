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

class SIGN_EXTEND_FM_MMR6<string instr_asm, bits<10> funct>
    : MMR6Arch<instr_asm> {
  bits<5> rd;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rd;
  let Inst{20-16} = rt;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0b111100;
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

class CMP_BRANCH_1R_RT_OFF16_FM_MMR6<bits<6> funct> : MipsR6Inst {
  bits<5> rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = funct;
  let Inst{25-21} = rt;
  let Inst{20-16} = 0b00000;
  let Inst{15-0}  = offset;
}

class CMP_BRANCH_1R_BOTH_OFF16_FM_MMR6<bits<6> funct> : MipsR6Inst {
  bits<5> rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = funct;
  let Inst{25-21} = rt;
  let Inst{20-16} = rt;
  let Inst{15-0}  = offset;
}

class ERET_FM_MMR6<string instr_asm> : MMR6Arch<instr_asm> {
  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-16} = 0x00;
  let Inst{15-6}  = 0x3cd;
  let Inst{5-0}   = 0x3c;
}

class ERETNC_FM_MMR6<string instr_asm> : MMR6Arch<instr_asm> {
  bits<32> Inst;

  let Inst{31-26} = 0x00;
  let Inst{25-17} = 0x00;
  let Inst{16-16} = 0x01;
  let Inst{15-6}  = 0x3cd;
  let Inst{5-0}   = 0x3c;
}

class BREAK_MMR6_ENC<string instr_asm> : MMR6Arch<instr_asm> {
  bits<10> code_1;
  bits<10> code_2;
  bits<32> Inst;
  let Inst{31-26} = 0x0;
  let Inst{25-16} = code_1;
  let Inst{15-6}  = code_2;
  let Inst{5-0}   = 0x07;
}

class BARRIER_MMR6_ENC<string instr_asm, bits<5> op> : MMR6Arch<instr_asm> {
  bits<32> Inst;

  let Inst{31-26} = 0x0;
  let Inst{25-21} = 0x0;
  let Inst{20-16} = 0x0;
  let Inst{15-11} = op;
  let Inst{10-6}  = 0x0;
  let Inst{5-0}   = 0x0;
}

class EIDI_MMR6_ENC<string instr_asm, bits<10> funct> : MMR6Arch<instr_asm> {
  bits<32> Inst;
  bits<5> rt; // Actually rs but we're sharing code with the standard encodings which call it rt

  let Inst{31-26} = 0x00;
  let Inst{25-21} = 0x00;
  let Inst{20-16} = rt;
  let Inst{15-6}  = funct;
  let Inst{5-0}   = 0x3c;
}

class SHIFT_MMR6_ENC<string instr_asm, bits<10> funct, bit rotate> : MMR6Arch<instr_asm> {
  bits<5> rd;
  bits<5> rt;
  bits<5> shamt;

  bits<32> Inst;

  let Inst{31-26} = 0;
  let Inst{25-21} = rd;
  let Inst{20-16} = rt;
  let Inst{15-11} = shamt;
  let Inst{10}    = rotate;
  let Inst{9-0}   = funct;
}

class SW32_FM_MMR6<string instr_asm, bits<6> op> : MMR6Arch<instr_asm> {
  bits<5> rt;
  bits<21> addr;

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = addr{20-16};
  let Inst{15-0}  = addr{15-0};
}

class POOL32C_SWE_FM_MMR6<string instr_asm, bits<6> op, bits<4> fmt,
    bits<3> funct> : MMR6Arch<instr_asm> {
  bits<5> rt;
  bits<21> addr;
  bits<5> base = addr{20-16};
  bits<9> offset = addr{8-0};

  bits<32> Inst;

  let Inst{31-26} = op;
  let Inst{25-21} = rt;
  let Inst{20-16} = base;
  let Inst{15-12} = fmt;
  let Inst{11-9} = funct;
  let Inst{8-0}  = offset;
}

class POOL32F_ARITH_FM_MMR6<string instr_asm, bits<2> fmt, bits<8> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> ft;
  bits<5> fs;
  bits<5> fd;

  bits<32> Inst;

  let Inst{31-26} = 0b010101;
  let Inst{25-21} = ft;
  let Inst{20-16} = fs;
  let Inst{15-11} = fd;
  let Inst{10}    = 0;
  let Inst{9-8}   = fmt;
  let Inst{7-0}   = funct;
}

class POOL32F_ARITHF_FM_MMR6<string instr_asm, bits<2> fmt, bits<9> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> ft;
  bits<5> fs;
  bits<5> fd;

  bits<32> Inst;

  let Inst{31-26} = 0b010101;
  let Inst{25-21} = ft;
  let Inst{20-16} = fs;
  let Inst{15-11} = fd;
  let Inst{10-9}  = fmt;
  let Inst{8-0}   = funct;
}

class POOL32F_MOV_NEG_FM_MMR6<string instr_asm, bits<2> fmt, bits<7> funct>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  bits<5> ft;
  bits<5> fs;

  bits<32> Inst;

  let Inst{31-26} = 0b010101;
  let Inst{25-21} = ft;
  let Inst{20-16} = fs;
  let Inst{15}    = 0;
  let Inst{14-13} = fmt;
  let Inst{12-6}  = funct;
  let Inst{5-0}   = 0b111011;
}
