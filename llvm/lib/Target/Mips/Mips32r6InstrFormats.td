//=- Mips32r6InstrFormats.td - Mips32r6 Instruction Formats -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips32r6 instruction formats.
//
//===----------------------------------------------------------------------===//

class MipsR6Inst : MipsInst<(outs), (ins), "", [], NoItinerary, FrmOther>,
                   PredicateControl {
  let DecoderNamespace = "Mips32r6_64r6";
  let EncodingPredicates = [HasStdEnc];
}

//===----------------------------------------------------------------------===//
//
// Field Values
//
//===----------------------------------------------------------------------===//

class OPGROUP<bits<6> Val> {
  bits<6> Value = Val;
}
def OPGROUP_COP1     : OPGROUP<0b010001>;
def OPGROUP_COP2     : OPGROUP<0b010010>;
def OPGROUP_ADDI     : OPGROUP<0b001000>;
def OPGROUP_AUI      : OPGROUP<0b001111>;
def OPGROUP_BLEZ     : OPGROUP<0b000110>;
def OPGROUP_BGTZ     : OPGROUP<0b000111>;
def OPGROUP_BLEZL    : OPGROUP<0b010110>;
def OPGROUP_BGTZL    : OPGROUP<0b010111>;
def OPGROUP_DADDI    : OPGROUP<0b011000>;
def OPGROUP_DAUI     : OPGROUP<0b011101>;
def OPGROUP_PCREL    : OPGROUP<0b111011>;
def OPGROUP_REGIMM   : OPGROUP<0b000001>;
def OPGROUP_SPECIAL  : OPGROUP<0b000000>;
def OPGROUP_SPECIAL3 : OPGROUP<0b011111>;

class OPCODE2<bits<2> Val> {
  bits<2> Value = Val;
}
def OPCODE2_ADDIUPC : OPCODE2<0b00>;
def OPCODE2_LWPC    : OPCODE2<0b01>;
def OPCODE2_LWUPC   : OPCODE2<0b10>;

class OPCODE5<bits<5> Val> {
  bits<5> Value = Val;
}
def OPCODE5_ALUIPC : OPCODE5<0b11111>;
def OPCODE5_AUIPC  : OPCODE5<0b11110>;
def OPCODE5_DAHI : OPCODE5<0b00110>;
def OPCODE5_DATI : OPCODE5<0b11110>;
def OPCODE5_BC1EQZ : OPCODE5<0b01001>;
def OPCODE5_BC1NEZ : OPCODE5<0b01101>;
def OPCODE5_BC2EQZ : OPCODE5<0b01001>;
def OPCODE5_BC2NEZ : OPCODE5<0b01101>;

class OPCODE6<bits<6> Val> {
  bits<6> Value = Val;
}
def OPCODE6_ALIGN    : OPCODE6<0b100000>;
def OPCODE6_DALIGN   : OPCODE6<0b100100>;
def OPCODE6_BITSWAP  : OPCODE6<0b100000>;
def OPCODE6_DBITSWAP : OPCODE6<0b100100>;

class FIELD_FMT<bits<5> Val> {
  bits<5> Value = Val;
}
def FIELD_FMT_S : FIELD_FMT<0b10000>;
def FIELD_FMT_D : FIELD_FMT<0b10001>;

class FIELD_CMP_COND<bits<5> Val> {
  bits<5> Value = Val;
}
def FIELD_CMP_COND_F    : FIELD_CMP_COND<0b00000>;
def FIELD_CMP_COND_UN   : FIELD_CMP_COND<0b00001>;
def FIELD_CMP_COND_EQ   : FIELD_CMP_COND<0b00010>;
def FIELD_CMP_COND_UEQ  : FIELD_CMP_COND<0b00011>;
def FIELD_CMP_COND_OLT  : FIELD_CMP_COND<0b00100>;
def FIELD_CMP_COND_ULT  : FIELD_CMP_COND<0b00101>;
def FIELD_CMP_COND_OLE  : FIELD_CMP_COND<0b00110>;
def FIELD_CMP_COND_ULE  : FIELD_CMP_COND<0b00111>;
def FIELD_CMP_COND_SF   : FIELD_CMP_COND<0b01000>;
def FIELD_CMP_COND_NGLE : FIELD_CMP_COND<0b01001>;
def FIELD_CMP_COND_SEQ  : FIELD_CMP_COND<0b01010>;
def FIELD_CMP_COND_NGL  : FIELD_CMP_COND<0b01011>;
def FIELD_CMP_COND_LT   : FIELD_CMP_COND<0b01100>;
def FIELD_CMP_COND_NGE  : FIELD_CMP_COND<0b01101>;
def FIELD_CMP_COND_LE   : FIELD_CMP_COND<0b01110>;
def FIELD_CMP_COND_NGT  : FIELD_CMP_COND<0b01111>;

class FIELD_CMP_FORMAT<bits<5> Val> {
  bits<5> Value = Val;
}
def FIELD_CMP_FORMAT_S : FIELD_CMP_FORMAT<0b10100>;
def FIELD_CMP_FORMAT_D : FIELD_CMP_FORMAT<0b10101>;

//===----------------------------------------------------------------------===//
//
// Disambiguators
//
//===----------------------------------------------------------------------===//
//
// Some encodings are ambiguous except by comparing field values.

class DecodeDisambiguates<string Name> {
  string DecoderMethod = !strconcat("Decode", Name);
}

class DecodeDisambiguatedBy<string Name> : DecodeDisambiguates<Name> {
  string DecoderNamespace = "Mips32r6_64r6_Ambiguous";
}

//===----------------------------------------------------------------------===//
//
// Encoding Formats
//
//===----------------------------------------------------------------------===//

class AUI_FM : MipsR6Inst {
  bits<5> rs;
  bits<5> rt;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_AUI.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0} = imm;
}

class DAUI_FM : AUI_FM {
  let Inst{31-26} = OPGROUP_DAUI.Value;
}

class COP1_2R_FM<bits<6> funct, FIELD_FMT Format> : MipsR6Inst {
  bits<5> fs;
  bits<5> fd;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_COP1.Value;
  let Inst{25-21} = Format.Value;
  let Inst{20-16} = 0b00000;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5-0}   = funct;
}

class COP1_3R_FM<bits<6> funct, FIELD_FMT Format> : MipsR6Inst {
  bits<5> ft;
  bits<5> fs;
  bits<5> fd;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_COP1.Value;
  let Inst{25-21} = Format.Value;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6} = fd;
  let Inst{5-0} = funct;
}

class COP1_BCCZ_FM<OPCODE5 Operation> : MipsR6Inst {
  bits<5> ft;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_COP1.Value;
  let Inst{25-21} = Operation.Value;
  let Inst{20-16} = ft;
  let Inst{15-0} = offset;
}

class COP2_BCCZ_FM<OPCODE5 Operation> : MipsR6Inst {
  bits<5> ct;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_COP2.Value;
  let Inst{25-21} = Operation.Value;
  let Inst{20-16} = ct;
  let Inst{15-0} = offset;
}

class PCREL16_FM<OPCODE5 Operation> : MipsR6Inst {
  bits<5> rs;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_PCREL.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = Operation.Value;
  let Inst{15-0} = imm;
}

class PCREL19_FM<OPCODE2 Operation> : MipsR6Inst {
  bits<5> rs;
  bits<19> imm;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_PCREL.Value;
  let Inst{25-21} = rs;
  let Inst{20-19} = Operation.Value;
  let Inst{18-0} = imm;
}

class SPECIAL3_2R_FM<OPCODE6 Operation> : MipsR6Inst {
  bits<5> rd;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_SPECIAL3.Value;
  let Inst{25-21} = 0b00000;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0b00000;
  let Inst{5-0}   = Operation.Value;
}

class SPECIAL_3R_FM<bits<5> mulop, bits<6> funct> : MipsR6Inst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_SPECIAL.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = mulop;
  let Inst{5-0}   = funct;
}

// This class is ambiguous with other branches:
//   BEQC/BNEC require that rs > rt
class CMP_BRANCH_2R_OFF16_FM<OPGROUP funct> : MipsR6Inst {
  bits<5> rs;
  bits<5> rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = funct.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0} = offset;
}

// This class is ambiguous with other branches:
//   BLEZC/BGEZC/BEQZALC/BNEZALC/BGTZALC require that rs == 0 && rt != 0
// The '1R_RT' in the name means 1 register in the rt field.
class CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP funct> : MipsR6Inst {
  bits<5> rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = funct.Value;
  let Inst{25-21} = 0b00000;
  let Inst{20-16} = rt;
  let Inst{15-0} = offset;
}

// This class is ambiguous with other branches:
//   BLTZC/BGTZC/BLTZALC/BGEZALC require that rs == rt && rt != 0
// The '1R_BOTH' in the name means 1 register in both the rs and rt fields.
class CMP_BRANCH_1R_BOTH_OFF16_FM<OPGROUP funct> : MipsR6Inst {
  bits<5> rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = funct.Value;
  let Inst{25-21} = rt;
  let Inst{20-16} = rt;
  let Inst{15-0} = offset;
}

class CMP_BRANCH_OFF21_FM<bits<6> funct> : MipsR6Inst {
  bits<5> rs; // rs != 0
  bits<21> offset;

  bits<32> Inst;

  let Inst{31-26} = funct;
  let Inst{25-21} = rs;
  let Inst{20-0} = offset;
}

class JMP_IDX_COMPACT_FM<bits<6> funct> : MipsR6Inst {
  bits<5> rt;
  bits<16> offset;

  bits<32> Inst;

  let Inst{31-26} = funct;
  let Inst{25-21} = 0b000000;
  let Inst{20-16} = rt;
  let Inst{15-0} = offset;
}

class BRANCH_OFF26_FM<bits<6> funct> : MipsR6Inst {
  bits<32> Inst;
  bits<26> offset;

  let Inst{31-26} = funct;
  let Inst{25-0} = offset;
}

class SPECIAL3_ALIGN_FM<OPCODE6 Operation> : MipsR6Inst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<2> bp;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_SPECIAL3.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-8}  = 0b010;
  let Inst{7-6}   = bp;
  let Inst{5-0}   = Operation.Value;
}

class SPECIAL3_DALIGN_FM<OPCODE6 Operation> : MipsR6Inst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<3> bp;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_SPECIAL3.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-9}  = 0b01;
  let Inst{8-6}   = bp;
  let Inst{5-0}   = Operation.Value;
}

class REGIMM_FM<OPCODE5 Operation> : MipsR6Inst {
  bits<5> rs;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_REGIMM.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = Operation.Value;
  let Inst{15-0} = imm;
}

class COP1_CMP_CONDN_FM<FIELD_CMP_FORMAT Format,
                        FIELD_CMP_COND Cond> : MipsR6Inst {
  bits<5> fd;
  bits<5> fs;
  bits<5> ft;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_COP1.Value;
  let Inst{25-21} = Format.Value;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5}     = 0;
  let Inst{4-0}   = Cond.Value;
}

