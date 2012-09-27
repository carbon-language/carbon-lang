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
