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

def OPGROUP_COP1    { bits<6> Value = 0b010001; }
def OPGROUP_AUI     { bits<6> Value = 0b001111; }
def OPGROUP_DAUI    { bits<6> Value = 0b011101; }
def OPGROUP_PCREL   { bits<6> Value = 0b111011; }
def OPGROUP_REGIMM  { bits<6> Value = 0b000001; }
def OPGROUP_SPECIAL { bits<6> Value = 0b000000; }

class OPCODE2<bits<2> Val> {
  bits<2> Value = Val;
}
def OPCODE2_ADDIUPC : OPCODE2<0b00>;

class OPCODE5<bits<5> Val> {
  bits<5> Value = Val;
}
def OPCODE5_ALUIPC : OPCODE5<0b11111>;
def OPCODE5_AUIPC  : OPCODE5<0b11110>;
def OPCODE5_DAHI : OPCODE5<0b00110>;
def OPCODE5_DATI : OPCODE5<0b11110>;

class FIELD_FMT<bits<5> Val> {
  bits<5> Value = Val;
}
def FIELD_FMT_S : FIELD_FMT<0b10000>;
def FIELD_FMT_D : FIELD_FMT<0b10001>;

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

class REGIMM_FM<OPCODE5 Operation> : MipsR6Inst {
  bits<5> rs;
  bits<16> imm;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_REGIMM.Value;
  let Inst{25-21} = rs;
  let Inst{20-16} = Operation.Value;
  let Inst{15-0} = imm;
}
