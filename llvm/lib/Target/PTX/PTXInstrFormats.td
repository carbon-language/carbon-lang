//===- PTXInstrFormats.td - PTX Instruction Formats ----------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// Rounding Mode Specifier
/*class RoundingMode<bits<3> val> {
  bits<3> Value = val;
}

def RndDefault     : RoundingMode<0>;
def RndNearestEven : RoundingMode<1>;
def RndNearestZero : RoundingMode<2>;
def RndNegInf      : RoundingMode<3>;
def RndPosInf      : RoundingMode<4>;
def RndApprox      : RoundingMode<5>;*/


// Rounding Mode Operand
def RndMode : Operand<i32> {
  let PrintMethod = "printRoundingMode";
}

def RndDefault : PatLeaf<(i32 0)>;

// PTX Predicate operand, default to (0, 0) = (zero-reg, none).
// Leave PrintMethod empty; predicate printing is defined elsewhere.
def pred : PredicateOperand<OtherVT, (ops RegPred, i32imm),
                                     (ops (i1 zero_reg), (i32 2))>;

def RndModeOperand : Operand<OtherVT> {
  let MIOperandInfo = (ops i32imm);
}

// Instruction Types
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
