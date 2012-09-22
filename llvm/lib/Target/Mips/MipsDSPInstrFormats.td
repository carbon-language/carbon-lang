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
