//===- MipsMSAInstrFormats.td - Mips Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def HasMSA : Predicate<"Subtarget.hasMSA()">,
             AssemblerPredicate<"FeatureMSA">;

class MSAInst : MipsInst<(outs), (ins), "", [], NoItinerary, FrmOther> {
  let Predicates = [HasMSA];
  let Inst{31-26} = 0b011110;
}

class PseudoMSA<dag outs, dag ins, list<dag> pattern,
                InstrItinClass itin = IIPseudo>:
  MipsPseudo<outs, ins, pattern, itin> {
  let Predicates = [HasMSA];
}

class MSA_3R_FMT<bits<3> major, bits<2> df, bits<6> minor>: MSAInst {
  let Inst{25-23} = major;
  let Inst{22-21} = df;
  let Inst{5-0} = minor;
}

class MSA_I5_FMT<bits<3> major, bits<2> df, bits<6> minor>: MSAInst {
  let Inst{25-23} = major;
  let Inst{22-21} = df;
  let Inst{5-0} = minor;
}
