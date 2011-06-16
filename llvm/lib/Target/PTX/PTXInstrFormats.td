//===- PTXInstrFormats.td - PTX Instruction Formats ----------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// PTX Predicate operand, default to (0, 0) = (zero-reg, always).
// Leave PrintMethod empty; predicate printing is defined elsewhere.
def pred : PredicateOperand<OtherVT, (ops RegPred, i32imm),
                                     (ops (i1 zero_reg), (i32 0))>;

let Namespace = "PTX" in {
  class InstPTX<dag oops, dag iops, string asmstr, list<dag> pattern>
    : Instruction {
      dag OutOperandList = oops;
      dag InOperandList = !con(iops, (ins pred:$_p));
      let AsmString = asmstr; // Predicate printing is defined elsewhere.
      let Pattern = pattern;
      let isPredicable = 1;
  }
}
