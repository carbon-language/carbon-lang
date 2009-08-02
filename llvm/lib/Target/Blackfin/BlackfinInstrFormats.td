//===--- BlackfinInstrFormats.td ---------------------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

class InstBfin<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  field bits<32> Inst;

  let Namespace = "BF";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
}

// Single-word (16-bit) instructions
class F1<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstBfin<outs, ins, asmstr, pattern> {
}

// Double-word (32-bit) instructions
class F2<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstBfin<outs, ins, asmstr, pattern> {
}
