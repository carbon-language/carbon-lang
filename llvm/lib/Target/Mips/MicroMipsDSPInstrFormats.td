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
  let AdditionalPredicates = [InMicroMips];
  string BaseOpcode = opstr;
  string Arch = "mmdsp";
  let DecoderNamespace = "MicroMips";
}

class POOL32A_3R_FMT<string opstr, bits<11> op> : MMDSPInst<opstr> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-0}  = op;
}

class POOL32A_AC2R_FMT<string opstr, bits<8> funct> : MMDSPInst<opstr> {
  bits<2> ac;
  bits<5> rs;
  bits<5> rt;

  let Inst{31-26} = 0;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-14} = ac;
  let Inst{13-6} = funct;
  let Inst{5-0} = 0b111100;
}
