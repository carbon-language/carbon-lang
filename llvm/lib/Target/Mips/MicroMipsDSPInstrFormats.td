//===-- MicroMipsDSPInstrFormats.td - Instruction Formats --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class MMDSPInst<string opstr = "">
    : MipsInst<(outs), (ins), "", [], NoItinerary, FrmOther>, PredicateControl {
  let InsnPredicates = [HasDSP];
  string BaseOpcode = opstr;
  string Arch = "mmdsp";
  let DecoderNamespace = "MicroMips";
}

class POOL32A_3R_FMT<bits<11> op> : MMDSPInst {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-0}  = op;
}
