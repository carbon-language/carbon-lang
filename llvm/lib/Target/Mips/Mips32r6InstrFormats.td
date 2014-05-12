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

def OPGROUP_COP1 { bits<6> Value = 0b010001; }
def OPGROUP_SPECIAL { bits<6> Value = 0b000000; }

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

class COP1_SEL_FM<FIELD_FMT Format> : MipsR6Inst {
  bits<5> ft;
  bits<5> fs;
  bits<5> fd;

  bits<32> Inst;

  let Inst{31-26} = OPGROUP_COP1.Value;
  let Inst{25-21} = Format.Value;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6} = fd;
  let Inst{5-0} = 0b010000;
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
