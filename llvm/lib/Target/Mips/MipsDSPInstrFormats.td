//===- MipsDSPInstrFormats.td - Mips Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def HasDSP : Predicate<"Subtarget.hasDSP()">,
             AssemblerPredicate<"FeatureDSP">;
def HasDSPR2 : Predicate<"Subtarget.hasDSPR2()">,
               AssemblerPredicate<"FeatureDSPR2">;

// Fields.
class Field6<bits<6> val> {
  bits<6> V = val;
}

def SPECIAL3_OPCODE : Field6<0b011111>;
def REGIMM_OPCODE : Field6<0b000001>;

class DSPInst : MipsInst<(outs), (ins), "", [], NoItinerary, FrmOther> {
  let Predicates = [HasDSP];
}

class PseudoDSP<dag outs, dag ins, list<dag> pattern>:
  MipsPseudo<outs, ins, "", pattern> {
  let Predicates = [HasDSP];
}

// DPA.W.PH sub-class format.
class DPA_W_PH_FMT<bits<5> op> : DSPInst {
  bits<2> ac;
  bits<5> rs;
  bits<5> rt;

  let Opcode = SPECIAL3_OPCODE.V;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-13} = 0;
  let Inst{12-11} = ac;
  let Inst{10-6}  = op;
  let Inst{5-0} = 0b110000;
}

// MULT sub-class format.
class MULT_FMT<bits<6> opcode, bits<6> funct> : DSPInst {
  bits<2> ac;
  bits<5> rs;
  bits<5> rt;

  let Opcode = opcode;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-13} = 0;
  let Inst{12-11} = ac;
  let Inst{10-6}  = 0;
  let Inst{5-0} = funct;
}

// EXTR.W sub-class format (type 1).
class EXTR_W_TY1_FMT<bits<5> op> : DSPInst {
  bits<5> rt;
  bits<2> ac;
  bits<5> shift_rs;

  let Opcode = SPECIAL3_OPCODE.V;

  let Inst{25-21} = shift_rs;
  let Inst{20-16} = rt;
  let Inst{15-13} = 0;
  let Inst{12-11} = ac;
  let Inst{10-6} = op;
  let Inst{5-0} = 0b111000;
}

// SHILO sub-class format.
class SHILO_R1_FMT<bits<5> op> : DSPInst {
  bits<2> ac;
  bits<6> shift;

  let Opcode = SPECIAL3_OPCODE.V;

  let Inst{25-20} = shift;
  let Inst{19-13} = 0;
  let Inst{12-11} = ac;
  let Inst{10-6} = op;
  let Inst{5-0} = 0b111000;
}

class SHILO_R2_FMT<bits<5> op> : DSPInst {
  bits<2> ac;
  bits<5> rs;

  let Opcode = SPECIAL3_OPCODE.V;

  let Inst{25-21} = rs;
  let Inst{20-13} = 0;
  let Inst{12-11} = ac;
  let Inst{10-6} = op;
  let Inst{5-0} = 0b111000;
}
