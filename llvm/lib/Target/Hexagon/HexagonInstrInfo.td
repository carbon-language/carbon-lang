//==- HexagonInstrInfo.td - Target Description for Hexagon -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "HexagonInstrFormats.td"
include "HexagonImmediates.td"

//===----------------------------------------------------------------------===//
// Hexagon Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasV2T                      : Predicate<"Subtarget.hasV2TOps()">;
def HasV2TOnly                  : Predicate<"Subtarget.hasV2TOpsOnly()">;
def NoV2T                       : Predicate<"!Subtarget.hasV2TOps()">;
def HasV3T                      : Predicate<"Subtarget.hasV3TOps()">;
def HasV3TOnly                  : Predicate<"Subtarget.hasV3TOpsOnly()">;
def NoV3T                       : Predicate<"!Subtarget.hasV3TOps()">;
def HasV4T                      : Predicate<"Subtarget.hasV4TOps()">;
def NoV4T                       : Predicate<"!Subtarget.hasV4TOps()">;
def UseMEMOP                    : Predicate<"Subtarget.useMemOps()">;

// Addressing modes.
def ADDRrr : ComplexPattern<i32, 2, "SelectADDRrr", [], []>;
def ADDRri : ComplexPattern<i32, 2, "SelectADDRri", [frameindex], []>;
def ADDRriS11_0 : ComplexPattern<i32, 2, "SelectADDRriS11_0", [frameindex], []>;
def ADDRriS11_1 : ComplexPattern<i32, 2, "SelectADDRriS11_1", [frameindex], []>;
def ADDRriS11_2 : ComplexPattern<i32, 2, "SelectADDRriS11_2", [frameindex], []>;
def ADDRriS11_3 : ComplexPattern<i32, 2, "SelectADDRriS11_3", [frameindex], []>;
def ADDRriU6_0 : ComplexPattern<i32, 2, "SelectADDRriU6_0", [frameindex], []>;
def ADDRriU6_1 : ComplexPattern<i32, 2, "SelectADDRriU6_1", [frameindex], []>;
def ADDRriU6_2 : ComplexPattern<i32, 2, "SelectADDRriU6_2", [frameindex], []>;

// Address operands.
def MEMrr : Operand<i32> {
  let PrintMethod = "printMEMrrOperand";
  let MIOperandInfo = (ops IntRegs, IntRegs);
}

// Address operands
def MEMri : Operand<i32> {
  let PrintMethod = "printMEMriOperand";
  let MIOperandInfo = (ops IntRegs, IntRegs);
}

def MEMri_s11_2 : Operand<i32>,
  ComplexPattern<i32, 2, "SelectMEMriS11_2", []> {
  let PrintMethod = "printMEMriOperand";
  let MIOperandInfo = (ops IntRegs, s11Imm);
}

def FrameIndex : Operand<i32> {
  let PrintMethod = "printFrameIndexOperand";
  let MIOperandInfo = (ops IntRegs, s11Imm);
}

let PrintMethod = "printGlobalOperand" in
  def globaladdress : Operand<i32>;

let PrintMethod = "printJumpTable" in
 def jumptablebase : Operand<i32>;

def brtarget : Operand<OtherVT>;
def calltarget : Operand<i32>;

def bblabel : Operand<i32>;
def bbl   : SDNode<"ISD::BasicBlock", SDTPtrLeaf   , [], "BasicBlockSDNode">;

def symbolHi32 : Operand<i32> {
  let PrintMethod = "printSymbolHi";
}
def symbolLo32 : Operand<i32> {
  let PrintMethod = "printSymbolLo";
}

// Multi-class for logical operators.
multiclass ALU32_rr_ri<string OpcStr, SDNode OpNode> {
  def rr : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set IntRegs:$dst, (OpNode IntRegs:$b, IntRegs:$c))]>;
  def ri : ALU32_ri<(outs IntRegs:$dst), (ins s10Imm:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "(#$b, $c)")),
                 [(set IntRegs:$dst, (OpNode s10Imm:$b, IntRegs:$c))]>;
}

// Multi-class for compare ops.
let isCompare = 1 in {
multiclass CMP64_rr<string OpcStr, PatFrag OpNode> {
  def rr : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$b, DoubleRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst, (OpNode DoubleRegs:$b, DoubleRegs:$c))]>;
}
multiclass CMP32_rr<string OpcStr, PatFrag OpNode> {
  def rr : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, IntRegs:$c))]>;
}

multiclass CMP32_rr_ri_s10<string OpcStr, PatFrag OpNode> {
  def rr : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, IntRegs:$c))]>;
  def ri : ALU32_ri<(outs PredRegs:$dst), (ins IntRegs:$b, s10Imm:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, #$c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, s10ImmPred:$c))]>;
}

multiclass CMP32_rr_ri_u9<string OpcStr, PatFrag OpNode> {
  def rr : ALU32_rr<(outs PredRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, IntRegs:$c))]>;
  def ri : ALU32_ri<(outs PredRegs:$dst), (ins IntRegs:$b, u9Imm:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, #$c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, u9ImmPred:$c))]>;
}

multiclass CMP32_ri_u9<string OpcStr, PatFrag OpNode> {
  def ri : ALU32_ri<(outs PredRegs:$dst), (ins IntRegs:$b, u9Imm:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, #$c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, u9ImmPred:$c))]>;
}

multiclass CMP32_ri_s8<string OpcStr, PatFrag OpNode> {
  def ri : ALU32_ri<(outs PredRegs:$dst), (ins IntRegs:$b, s8Imm:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, #$c)")),
                 [(set PredRegs:$dst, (OpNode IntRegs:$b, s8ImmPred:$c))]>;
}
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// http://qualnet.qualcomm.com/~erich/v1/htmldocs/index.html
// http://qualnet.qualcomm.com/~erich/v2/htmldocs/index.html
// http://qualnet.qualcomm.com/~erich/v3/htmldocs/index.html
// http://qualnet.qualcomm.com/~erich/v4/htmldocs/index.html
// http://qualnet.qualcomm.com/~erich/v5/htmldocs/index.html
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU32/ALU +
//===----------------------------------------------------------------------===//
// Add.
let isPredicable = 1 in
def ADD_rr : ALU32_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = add($src1, $src2)",
            [(set IntRegs:$dst, (add IntRegs:$src1, IntRegs:$src2))]>;

let isPredicable = 1 in
def ADD_ri : ALU32_ri<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s16Imm:$src2),
            "$dst = add($src1, #$src2)",
            [(set IntRegs:$dst, (add IntRegs:$src1, s16ImmPred:$src2))]>;

// Logical operations.
let isPredicable = 1 in
def XOR_rr : ALU32_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = xor($src1, $src2)",
            [(set IntRegs:$dst, (xor IntRegs:$src1, IntRegs:$src2))]>;

let isPredicable = 1 in
def AND_rr : ALU32_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = and($src1, $src2)",
            [(set IntRegs:$dst, (and IntRegs:$src1, IntRegs:$src2))]>;

def OR_ri : ALU32_ri<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s8Imm:$src2),
            "$dst = or($src1, #$src2)",
            [(set IntRegs:$dst, (or IntRegs:$src1, s8ImmPred:$src2))]>;

def NOT_rr : ALU32_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1),
            "$dst = not($src1)",
            [(set IntRegs:$dst, (not IntRegs:$src1))]>;

def AND_ri : ALU32_ri<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s10Imm:$src2),
            "$dst = and($src1, #$src2)",
            [(set IntRegs:$dst, (and IntRegs:$src1, s10ImmPred:$src2))]>;

let isPredicable = 1 in
def OR_rr : ALU32_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = or($src1, $src2)",
            [(set IntRegs:$dst, (or IntRegs:$src1, IntRegs:$src2))]>;

// Negate.
def NEG : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
          "$dst = neg($src1)",
          [(set IntRegs:$dst, (ineg IntRegs:$src1))]>;
// Nop.
let neverHasSideEffects = 1 in
def NOP : ALU32_rr<(outs), (ins),
          "nop",
          []>;

// Subtract.
let isPredicable = 1 in
def SUB_rr : ALU32_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sub($src1, $src2)",
            [(set IntRegs:$dst, (sub IntRegs:$src1, IntRegs:$src2))]>;

// Transfer immediate.
let isReMaterializable = 1, isPredicable = 1 in
def TFRI : ALU32_ri<(outs IntRegs:$dst), (ins s16Imm:$src1),
           "$dst = #$src1",
           [(set IntRegs:$dst, s16ImmPred:$src1)]>;

// Transfer register.
let neverHasSideEffects = 1, isPredicable = 1 in
def TFR : ALU32_ri<(outs IntRegs:$dst), (ins IntRegs:$src1),
          "$dst = $src1",
          []>;

// Transfer control register.
let neverHasSideEffects = 1 in
def TFCR : CRInst<(outs CRRegs:$dst), (ins IntRegs:$src1),
           "$dst = $src1",
           []>;
//===----------------------------------------------------------------------===//
// ALU32/ALU -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//

// Combine.
let isPredicable = 1, neverHasSideEffects = 1 in
def COMBINE_rr : ALU32_rr<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = combine($src1, $src2)",
            []>;

// Mux.
def VMUX_prr64 : ALU64_rr<(outs DoubleRegs:$dst), (ins PredRegs:$src1,
                                                   DoubleRegs:$src2,
                                                   DoubleRegs:$src3),
            "$dst = vmux($src1, $src2, $src3)",
            []>;

def MUX_rr : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                            IntRegs:$src2, IntRegs:$src3),
             "$dst = mux($src1, $src2, $src3)",
             [(set IntRegs:$dst, (select PredRegs:$src1, IntRegs:$src2,
                                         IntRegs:$src3))]>;

def MUX_ir : ALU32_ir<(outs IntRegs:$dst), (ins PredRegs:$src1, s8Imm:$src2,
                                                IntRegs:$src3),
             "$dst = mux($src1, #$src2, $src3)",
             [(set IntRegs:$dst, (select PredRegs:$src1,
                                         s8ImmPred:$src2, IntRegs:$src3))]>;

def MUX_ri : ALU32_ri<(outs IntRegs:$dst), (ins PredRegs:$src1, IntRegs:$src2,
                                                s8Imm:$src3),
             "$dst = mux($src1, $src2, #$src3)",
             [(set IntRegs:$dst, (select PredRegs:$src1, IntRegs:$src2,
                                         s8ImmPred:$src3))]>;

def MUX_ii : ALU32_ii<(outs IntRegs:$dst), (ins PredRegs:$src1, s8Imm:$src2,
                                                s8Imm:$src3),
             "$dst = mux($src1, #$src2, #$src3)",
             [(set IntRegs:$dst, (select PredRegs:$src1, s8ImmPred:$src2,
                                         s8ImmPred:$src3))]>;

// Shift halfword.
let isPredicable = 1 in
def ASLH : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
           "$dst = aslh($src1)",
           [(set IntRegs:$dst, (shl 16, IntRegs:$src1))]>;

let isPredicable = 1 in
def ASRH : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
           "$dst = asrh($src1)",
           [(set IntRegs:$dst, (sra 16, IntRegs:$src1))]>;

// Sign extend.
let isPredicable = 1 in
def SXTB : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
           "$dst = sxtb($src1)",
           [(set IntRegs:$dst, (sext_inreg IntRegs:$src1, i8))]>;

let isPredicable = 1 in
def SXTH : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
           "$dst = sxth($src1)",
           [(set IntRegs:$dst, (sext_inreg IntRegs:$src1, i16))]>;

// Zero extend.
let isPredicable = 1, neverHasSideEffects = 1 in
def ZXTB : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
           "$dst = zxtb($src1)",
           []>;

let isPredicable = 1, neverHasSideEffects = 1 in
def ZXTH : ALU32_rr<(outs IntRegs:$dst), (ins IntRegs:$src1),
                    "$dst = zxth($src1)",
                    []>;
//===----------------------------------------------------------------------===//
// ALU32/PERM -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PRED +
//===----------------------------------------------------------------------===//

// Conditional add.
let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_ri_cPt : ALU32_ri<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, s16Imm:$src3),
            "if ($src1) $dst = add($src2, #$src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_ri_cNotPt : ALU32_ri<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, s16Imm:$src3),
            "if (!$src1) $dst = add($src2, #$src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_ri_cdnPt : ALU32_ri<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, s16Imm:$src3),
            "if ($src1.new) $dst = add($src2, #$src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_ri_cdnNotPt : ALU32_ri<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, s16Imm:$src3),
            "if (!$src1.new) $dst = add($src2, #$src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_rr_cPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1) $dst = add($src2, $src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_rr_cNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1) $dst = add($src2, $src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_rr_cdnPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1.new) $dst = add($src2, $src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def ADD_rr_cdnNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1.new) $dst = add($src2, $src3)",
            []>;


// Conditional combine.

let neverHasSideEffects = 1, isPredicated = 1 in
def COMBINE_rr_cPt : ALU32_rr<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1) $dst = combine($src2, $src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def COMBINE_rr_cNotPt : ALU32_rr<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1) $dst = combine($src2, $src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def COMBINE_rr_cdnPt : ALU32_rr<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1.new) $dst = combine($src2, $src3)",
            []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def COMBINE_rr_cdnNotPt : ALU32_rr<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1.new) $dst = combine($src2, $src3)",
            []>;

// Conditional logical operations.

let isPredicated = 1 in
def XOR_rr_cPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1) $dst = xor($src2, $src3)",
            []>;

let isPredicated = 1 in
def XOR_rr_cNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1) $dst = xor($src2, $src3)",
            []>;

let isPredicated = 1 in
def XOR_rr_cdnPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1.new) $dst = xor($src2, $src3)",
            []>;

let isPredicated = 1 in
def XOR_rr_cdnNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1.new) $dst = xor($src2, $src3)",
            []>;

let isPredicated = 1 in
def AND_rr_cPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1) $dst = and($src2, $src3)",
            []>;

let isPredicated = 1 in
def AND_rr_cNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1) $dst = and($src2, $src3)",
            []>;

let isPredicated = 1 in
def AND_rr_cdnPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1.new) $dst = and($src2, $src3)",
            []>;

let isPredicated = 1 in
def AND_rr_cdnNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1.new) $dst = and($src2, $src3)",
            []>;

let isPredicated = 1 in
def OR_rr_cPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1) $dst = or($src2, $src3)",
            []>;

let isPredicated = 1 in
def OR_rr_cNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1) $dst = or($src2, $src3)",
            []>;

let isPredicated = 1 in
def OR_rr_cdnPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1.new) $dst = or($src2, $src3)",
            []>;

let isPredicated = 1 in
def OR_rr_cdnNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1.new) $dst = or($src2, $src3)",
            []>;


// Conditional subtract.

let isPredicated = 1 in
def SUB_rr_cPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1) $dst = sub($src2, $src3)",
            []>;

let isPredicated = 1 in
def SUB_rr_cNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1) $dst = sub($src2, $src3)",
            []>;

let isPredicated = 1 in
def SUB_rr_cdnPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if ($src1.new) $dst = sub($src2, $src3)",
            []>;

let isPredicated = 1 in
def SUB_rr_cdnNotPt : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "if (!$src1.new) $dst = sub($src2, $src3)",
            []>;


// Conditional transfer.

let neverHasSideEffects = 1, isPredicated = 1 in
def TFR_cPt : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1, IntRegs:$src2),
              "if ($src1) $dst = $src2",
              []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFR_cNotPt : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                    IntRegs:$src2),
                 "if (!$src1) $dst = $src2",
                 []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cPt : ALU32_ri<(outs IntRegs:$dst), (ins PredRegs:$src1, s12Imm:$src2),
               "if ($src1) $dst = #$src2",
               []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cNotPt : ALU32_ri<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                     s12Imm:$src2),
                  "if (!$src1) $dst = #$src2",
                  []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFR_cdnPt : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                   IntRegs:$src2),
                "if ($src1.new) $dst = $src2",
                []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFR_cdnNotPt : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                      IntRegs:$src2),
                   "if (!$src1.new) $dst = $src2",
                   []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cdnPt : ALU32_ri<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                    s12Imm:$src2),
                 "if ($src1.new) $dst = #$src2",
                 []>;

let neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cdnNotPt : ALU32_ri<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                       s12Imm:$src2),
                    "if (!$src1.new) $dst = #$src2",
                    []>;

// Compare.
defm CMPGTU : CMP32_rr_ri_u9<"cmp.gtu", setugt>;
defm CMPGT : CMP32_rr_ri_s10<"cmp.gt", setgt>;
defm CMPLT : CMP32_rr<"cmp.lt", setlt>;
defm CMPEQ : CMP32_rr_ri_s10<"cmp.eq", seteq>;
defm CMPGE : CMP32_ri_s8<"cmp.ge", setge>;
defm CMPGEU : CMP32_ri_u9<"cmp.geu", setuge>;
//===----------------------------------------------------------------------===//
// ALU32/PRED -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU32/VH +
//===----------------------------------------------------------------------===//
// Vector add halfwords

// Vector averagehalfwords

// Vector subtract halfwords
//===----------------------------------------------------------------------===//
// ALU32/VH -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU64/ALU +
//===----------------------------------------------------------------------===//
// Add.
def ADD64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = add($src1, $src2)",
               [(set DoubleRegs:$dst, (add DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

// Add halfword.

// Compare.
defm CMPEHexagon4 : CMP64_rr<"cmp.eq", seteq>;
defm CMPGT64 : CMP64_rr<"cmp.gt", setgt>;
defm CMPGTU64 : CMP64_rr<"cmp.gtu", setugt>;

// Logical operations.
def AND_rr64 : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = and($src1, $src2)",
               [(set DoubleRegs:$dst, (and DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

def OR_rr64 : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                    DoubleRegs:$src2),
              "$dst = or($src1, $src2)",
              [(set DoubleRegs:$dst, (or DoubleRegs:$src1, DoubleRegs:$src2))]>;

def XOR_rr64 : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = xor($src1, $src2)",
               [(set DoubleRegs:$dst, (xor DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

// Maximum.
def MAXw_rr : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
              "$dst = max($src2, $src1)",
              [(set IntRegs:$dst, (select (i1 (setlt IntRegs:$src2,
                                                     IntRegs:$src1)),
                                          IntRegs:$src1, IntRegs:$src2))]>;

// Minimum.
def MINw_rr : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
              "$dst = min($src2, $src1)",
              [(set IntRegs:$dst, (select (i1 (setgt IntRegs:$src2,
                                                     IntRegs:$src1)),
                                          IntRegs:$src1, IntRegs:$src2))]>;

// Subtract.
def SUB64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = sub($src1, $src2)",
               [(set DoubleRegs:$dst, (sub DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>;

// Subtract halfword.

// Transfer register.
let neverHasSideEffects = 1 in
def TFR_64 : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1),
             "$dst = $src1",
             []>;
//===----------------------------------------------------------------------===//
// ALU64/ALU -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/BIT +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/BIT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/PERM +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/PERM -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/VB +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/VB -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/VH +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/VH -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/VW +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/VW -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CR +
//===----------------------------------------------------------------------===//
// Logical reductions on predicates.

// Looping instructions.

// Pipelined looping instructions.

// Logical operations on predicates.
def AND_pp : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1, PredRegs:$src2),
             "$dst = and($src1, $src2)",
             [(set PredRegs:$dst, (and PredRegs:$src1, PredRegs:$src2))]>;

let neverHasSideEffects = 1 in
def AND_pnotp : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1,
                                                 PredRegs:$src2),
                "$dst = and($src1, !$src2)",
                []>;

def ANY_pp : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1),
             "$dst = any8($src1)",
             []>;

def ALL_pp : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1),
             "$dst = all8($src1)",
             []>;

def VITPACK_pp : SInst<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                 PredRegs:$src2),
             "$dst = vitpack($src1, $src2)",
             []>;

def VALIGN_rrp : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                    DoubleRegs:$src2,
                                                    PredRegs:$src3),
             "$dst = valignb($src1, $src2, $src3)",
             []>;

def VSPLICE_rrp : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2,
                                                     PredRegs:$src3),
             "$dst = vspliceb($src1, $src2, $src3)",
             []>;

def MASK_p : SInst<(outs DoubleRegs:$dst), (ins PredRegs:$src1),
             "$dst = mask($src1)",
             []>;

def NOT_p : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1),
             "$dst = not($src1)",
             [(set PredRegs:$dst, (not PredRegs:$src1))]>;

def OR_pp : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1, PredRegs:$src2),
            "$dst = or($src1, $src2)",
            [(set PredRegs:$dst, (or PredRegs:$src1, PredRegs:$src2))]>;

def XOR_pp : SInst<(outs PredRegs:$dst), (ins PredRegs:$src1, PredRegs:$src2),
             "$dst = xor($src1, $src2)",
             [(set PredRegs:$dst, (xor PredRegs:$src1, PredRegs:$src2))]>;


// User control register transfer.
//===----------------------------------------------------------------------===//
// CR -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// J +
//===----------------------------------------------------------------------===//
// Jump to address.
let isBranch = 1, isTerminator=1, isBarrier = 1, isPredicable = 1 in {
  def JMP : JInst< (outs),
            (ins brtarget:$offset),
            "jump $offset",
            [(br bb:$offset)]>;
}

// if (p0) jump
let isBranch = 1, isTerminator=1, Defs = [PC],
    isPredicated = 1 in {
  def JMP_c : JInst< (outs),
                 (ins PredRegs:$src, brtarget:$offset),
                 "if ($src) jump $offset",
                 [(brcond PredRegs:$src, bb:$offset)]>;
}

// if (!p0) jump
let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC],
    isPredicated = 1 in {
  def JMP_cNot : JInst< (outs),
                    (ins PredRegs:$src, brtarget:$offset),
                    "if (!$src) jump $offset",
                    []>;
}

let isTerminator = 1, isBranch = 1, neverHasSideEffects = 1, Defs = [PC],
    isPredicated = 1 in {
  def BRCOND : JInst < (outs), (ins PredRegs:$pred, brtarget:$dst),
               "if ($pred) jump $dst",
               []>;
}

// Jump to address conditioned on new predicate.
// if (p0) jump:t
let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC],
    isPredicated = 1 in {
  def JMP_cdnPt : JInst< (outs),
                   (ins PredRegs:$src, brtarget:$offset),
                   "if ($src.new) jump:t $offset",
                   []>;
}

// if (!p0) jump:t
let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC],
    isPredicated = 1 in {
  def JMP_cdnNotPt : JInst< (outs),
                      (ins PredRegs:$src, brtarget:$offset),
                      "if (!$src.new) jump:t $offset",
                      []>;
}

// Not taken.
let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC],
    isPredicated = 1 in {
  def JMP_cdnPnt : JInst< (outs),
                    (ins PredRegs:$src, brtarget:$offset),
                    "if ($src.new) jump:nt $offset",
                    []>;
}

// Not taken.
let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC],
    isPredicated = 1 in {
  def JMP_cdnNotPnt : JInst< (outs),
                       (ins PredRegs:$src, brtarget:$offset),
                       "if (!$src.new) jump:nt $offset",
                       []>;
}
//===----------------------------------------------------------------------===//
// J -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// JR +
//===----------------------------------------------------------------------===//
def retflag : SDNode<"HexagonISD::RET_FLAG", SDTNone,
                               [SDNPHasChain, SDNPOptInGlue]>;

// Jump to address from register.
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR: JRInst<(outs), (ins),
                   "jumpr r31",
                   [(retflag)]>;
}

// Jump to address from register.
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR_cPt: JRInst<(outs), (ins PredRegs:$src1),
                       "if ($src1) jumpr r31",
                       []>;
}

// Jump to address from register.
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR_cNotPt: JRInst<(outs), (ins PredRegs:$src1),
                          "if (!$src1) jumpr r31",
                          []>;
}

//===----------------------------------------------------------------------===//
// JR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LD +
//===----------------------------------------------------------------------===//
///
/// Make sure that in post increment load, the first operand is always the post
/// increment operand.
///
// Load doubleword.
let isPredicable = 1 in
def LDrid : LDInst<(outs DoubleRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memd($addr)",
            [(set DoubleRegs:$dst, (load ADDRriS11_3:$addr))]>;

let isPredicable = 1, AddedComplexity = 20 in
def LDrid_indexed : LDInst<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, s11_3Imm:$offset),
            "$dst=memd($src1+#$offset)",
            [(set DoubleRegs:$dst, (load (add IntRegs:$src1,
                                              s11_3ImmPred:$offset)))]>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_GP : LDInst<(outs DoubleRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memd(#$global+$offset)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDd_GP : LDInst<(outs DoubleRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memd(#$global)",
            []>;

let isPredicable = 1, mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrid : LDInstPI<(outs DoubleRegs:$dst, IntRegs:$dst2),
            (ins IntRegs:$src1, s4Imm:$offset),
            "$dst = memd($src1++#$offset)",
            [],
            "$src1 = $dst2">;

// Load doubleword conditionally.
let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_cPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1) $dst = memd($addr)",
            []>;


let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_cNotPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1) $dst = memd($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_indexed_cPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3),
            "if ($src1) $dst=memd($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_indexed_cNotPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3),
            "if (!$src1) $dst=memd($src2+#$src3)",
            []>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrid_cPt : LDInstPI<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_3Imm:$src3),
            "if ($src1) $dst1 = memd($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrid_cNotPt : LDInstPI<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_3Imm:$src3),
            "if (!$src1) $dst1 = memd($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_cdnPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1.new) $dst = memd($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_cdnNotPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1.new) $dst = memd($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_indexed_cdnPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3),
            "if ($src1.new) $dst=memd($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrid_indexed_cdnNotPt : LDInst<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3),
            "if (!$src1.new) $dst=memd($src2+#$src3)",
            []>;


// Load byte.
let isPredicable = 1 in
def LDrib : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memb($addr)",
            [(set IntRegs:$dst, (sextloadi8 ADDRriS11_0:$addr))]>;

def LDrib_ae : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memb($addr)",
            [(set IntRegs:$dst, (extloadi8 ADDRriS11_0:$addr))]>;

// Indexed load byte.
let isPredicable = 1, AddedComplexity = 20 in
def LDrib_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_0Imm:$offset),
            "$dst=memb($src1+#$offset)",
            [(set IntRegs:$dst, (sextloadi8 (add IntRegs:$src1,
                                                 s11_0ImmPred:$offset)))]>;


// Indexed load byte any-extend.
let AddedComplexity = 20 in
def LDrib_ae_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_0Imm:$offset),
            "$dst=memb($src1+#$offset)",
            [(set IntRegs:$dst, (extloadi8 (add IntRegs:$src1,
                                                s11_0ImmPred:$offset)))]>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memb(#$global+$offset)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDb_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memb(#$global)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDub_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memub(#$global)",
            []>;

let isPredicable = 1, mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrib : LDInstPI<(outs IntRegs:$dst, IntRegs:$dst2),
            (ins IntRegs:$src1, s4Imm:$offset),
            "$dst = memb($src1++#$offset)",
            [],
            "$src1 = $dst2">;

// Load byte conditionally.
let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1) $dst = memb($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1) $dst = memb($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_indexed_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if ($src1) $dst = memb($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_indexed_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if (!$src1) $dst = memb($src2+#$src3)",
            []>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrib_cPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if ($src1) $dst1 = memb($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrib_cNotPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if (!$src1) $dst1 = memb($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1.new) $dst = memb($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1.new) $dst = memb($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_indexed_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if ($src1.new) $dst = memb($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrib_indexed_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if (!$src1.new) $dst = memb($src2+#$src3)",
            []>;


// Load halfword.
let isPredicable = 1 in
def LDrih : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memh($addr)",
            [(set IntRegs:$dst, (sextloadi16 ADDRriS11_1:$addr))]>;

let isPredicable = 1, AddedComplexity = 20 in
def LDrih_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_1Imm:$offset),
            "$dst=memh($src1+#$offset)",
            [(set IntRegs:$dst, (sextloadi16 (add IntRegs:$src1,
                                                  s11_1ImmPred:$offset)))] >;

def LDrih_ae : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memh($addr)",
            [(set IntRegs:$dst, (extloadi16 ADDRriS11_1:$addr))]>;

let AddedComplexity = 20 in
def LDrih_ae_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_1Imm:$offset),
            "$dst=memh($src1+#$offset)",
            [(set IntRegs:$dst, (extloadi16 (add IntRegs:$src1,
                                                 s11_1ImmPred:$offset)))] >;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memh(#$global+$offset)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDh_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memh(#$global)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDuh_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memuh(#$global)",
            []>;


let isPredicable = 1, mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrih : LDInstPI<(outs IntRegs:$dst, IntRegs:$dst2),
            (ins IntRegs:$src1, s4Imm:$offset),
            "$dst = memh($src1++#$offset)",
            [],
            "$src1 = $dst2">;

// Load halfword conditionally.
let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1) $dst = memh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1) $dst = memh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_indexed_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if ($src1) $dst = memh($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_indexed_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if (!$src1) $dst = memh($src2+#$src3)",
            []>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrih_cPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if ($src1) $dst1 = memh($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrih_cNotPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if (!$src1) $dst1 = memh($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1.new) $dst = memh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1.new) $dst = memh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_indexed_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if ($src1.new) $dst = memh($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDrih_indexed_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if (!$src1.new) $dst = memh($src2+#$src3)",
            []>;

// Load unsigned byte.
let isPredicable = 1 in
def LDriub : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memub($addr)",
            [(set IntRegs:$dst, (zextloadi8 ADDRriS11_0:$addr))]>;

let isPredicable = 1 in
def LDriubit : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memub($addr)",
            [(set IntRegs:$dst, (zextloadi1 ADDRriS11_0:$addr))]>;

let isPredicable = 1, AddedComplexity = 20 in
def LDriub_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_0Imm:$offset),
            "$dst=memub($src1+#$offset)",
            [(set IntRegs:$dst, (zextloadi8 (add IntRegs:$src1,
                                                 s11_0ImmPred:$offset)))]>;

let AddedComplexity = 20 in
def LDriubit_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_0Imm:$offset),
            "$dst=memub($src1+#$offset)",
            [(set IntRegs:$dst, (zextloadi1 (add IntRegs:$src1,
                                                 s11_0ImmPred:$offset)))]>;

def LDriub_ae : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memub($addr)",
            [(set IntRegs:$dst, (extloadi8 ADDRriS11_0:$addr))]>;


let AddedComplexity = 20 in
def LDriub_ae_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_0Imm:$offset),
            "$dst=memub($src1+#$offset)",
            [(set IntRegs:$dst, (extloadi8 (add IntRegs:$src1,
                                                s11_0ImmPred:$offset)))]>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memub(#$global+$offset)",
            []>;

let isPredicable = 1, mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriub : LDInstPI<(outs IntRegs:$dst, IntRegs:$dst2),
            (ins IntRegs:$src1, s4Imm:$offset),
            "$dst = memub($src1++#$offset)",
            [],
            "$src1 = $dst2">;

// Load unsigned byte conditionally.
let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1) $dst = memub($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1) $dst = memub($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_indexed_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if ($src1) $dst = memub($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_indexed_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if (!$src1) $dst = memub($src2+#$src3)",
            []>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriub_cPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if ($src1) $dst1 = memub($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriub_cNotPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if (!$src1) $dst1 = memub($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1.new) $dst = memub($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1.new) $dst = memub($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_indexed_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if ($src1.new) $dst = memub($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriub_indexed_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3),
            "if (!$src1.new) $dst = memub($src2+#$src3)",
            []>;

// Load unsigned halfword.
let isPredicable = 1 in
def LDriuh : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memuh($addr)",
            [(set IntRegs:$dst, (zextloadi16 ADDRriS11_1:$addr))]>;

// Indexed load unsigned halfword.
let isPredicable = 1, AddedComplexity = 20 in
def LDriuh_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_1Imm:$offset),
            "$dst=memuh($src1+#$offset)",
            [(set IntRegs:$dst, (zextloadi16 (add IntRegs:$src1,
                                                  s11_1ImmPred:$offset)))]>;

def LDriuh_ae : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memuh($addr)",
            [(set IntRegs:$dst, (extloadi16 ADDRriS11_1:$addr))]>;


// Indexed load unsigned halfword any-extend.
let AddedComplexity = 20 in
def LDriuh_ae_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_1Imm:$offset),
            "$dst=memuh($src1+#$offset)",
            [(set IntRegs:$dst, (extloadi16 (add IntRegs:$src1,
                                                 s11_1ImmPred:$offset)))] >;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memuh(#$global+$offset)",
            []>;

let isPredicable = 1, mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriuh : LDInstPI<(outs IntRegs:$dst, IntRegs:$dst2),
            (ins IntRegs:$src1, s4Imm:$offset),
            "$dst = memuh($src1++#$offset)",
            [],
            "$src1 = $dst2">;

// Load unsigned halfword conditionally.
let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1) $dst = memuh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1) $dst = memuh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_indexed_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if ($src1) $dst = memuh($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_indexed_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if (!$src1) $dst = memuh($src2+#$src3)",
            []>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriuh_cPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if ($src1) $dst1 = memuh($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriuh_cNotPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if (!$src1) $dst1 = memuh($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1.new) $dst = memuh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1.new) $dst = memuh($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_indexed_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if ($src1.new) $dst = memuh($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriuh_indexed_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3),
            "if (!$src1.new) $dst = memuh($src2+#$src3)",
            []>;


// Load word.
let isPredicable = 1 in
def LDriw : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr), "$dst = memw($addr)",
            [(set IntRegs:$dst, (load ADDRriS11_2:$addr))]>;

// Load predicate.
let mayLoad = 1, Defs = [R10,R11] in
def LDriw_pred : LDInst<(outs PredRegs:$dst),
            (ins MEMri:$addr),
            "Error; should not emit",
            []>;

// Indexed load.
let isPredicable = 1, AddedComplexity = 20 in
def LDriw_indexed : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_2Imm:$offset),
            "$dst=memw($src1+#$offset)",
            [(set IntRegs:$dst, (load (add IntRegs:$src1,
                                           s11_2ImmPred:$offset)))]>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global, u16Imm:$offset),
            "$dst=memw(#$global+$offset)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDw_GP : LDInst<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memw(#$global)",
            []>;

let isPredicable = 1, mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriw : LDInstPI<(outs IntRegs:$dst, IntRegs:$dst2),
            (ins IntRegs:$src1, s4Imm:$offset),
            "$dst = memw($src1++#$offset)",
            [],
            "$src1 = $dst2">;

// Load word conditionally.

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1) $dst = memw($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1) $dst = memw($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_indexed_cPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3),
            "if ($src1) $dst=memw($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_indexed_cNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3),
            "if (!$src1) $dst=memw($src2+#$src3)",
            []>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriw_cPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_2Imm:$src3),
            "if ($src1) $dst1 = memw($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriw_cNotPt : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_2Imm:$src3),
            "if (!$src1) $dst1 = memw($src2++#$src3)",
            [],
            "$src2 = $dst2">;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if ($src1.new) $dst = memw($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, MEMri:$addr),
            "if (!$src1.new) $dst = memw($addr)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_indexed_cdnPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3),
            "if ($src1.new) $dst=memw($src2+#$src3)",
            []>;

let mayLoad = 1, neverHasSideEffects = 1 in
def LDriw_indexed_cdnNotPt : LDInst<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3),
            "if (!$src1.new) $dst=memw($src2+#$src3)",
            []>;

// Deallocate stack frame.
let Defs = [R29, R30, R31], Uses = [R29], neverHasSideEffects = 1 in {
  def DEALLOCFRAME : LDInst<(outs), (ins i32imm:$amt1),
                     "deallocframe",
                     []>;
}

// Load and unpack bytes to halfwords.
//===----------------------------------------------------------------------===//
// LD -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/ALU +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/ALU -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/COMPLEX +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/COMPLEX -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/MPYH +
//===----------------------------------------------------------------------===//
// Multiply and use lower result.
// Rd=+mpyi(Rs,#u8)
def MPYI_riu : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u8Imm:$src2),
              "$dst =+ mpyi($src1, #$src2)",
              [(set IntRegs:$dst, (mul IntRegs:$src1, u8ImmPred:$src2))]>;

// Rd=-mpyi(Rs,#u8)
def MPYI_rin : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, n8Imm:$src2),
              "$dst =- mpyi($src1, #$src2)",
              [(set IntRegs:$dst,
               (mul IntRegs:$src1, n8ImmPred:$src2))]>;

// Rd=mpyi(Rs,#m9)
// s9 is NOT the same as m9 - but it works.. so far.
// Assembler maps to either Rd=+mpyi(Rs,#u8 or Rd=-mpyi(Rs,#u8)
// depending on the value of m9. See Arch Spec.
def MPYI_ri : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, s9Imm:$src2),
              "$dst = mpyi($src1, #$src2)",
              [(set IntRegs:$dst, (mul IntRegs:$src1, s9ImmPred:$src2))]>;

// Rd=mpyi(Rs,Rt)
def MPYI : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
           "$dst = mpyi($src1, $src2)",
           [(set IntRegs:$dst, (mul IntRegs:$src1, IntRegs:$src2))]>;

// Rx+=mpyi(Rs,#u8)
def MPYI_acc_ri : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, u8Imm:$src3),
            "$dst += mpyi($src2, #$src3)",
            [(set IntRegs:$dst,
            (add (mul IntRegs:$src2, u8ImmPred:$src3), IntRegs:$src1))],
            "$src1 = $dst">;

// Rx+=mpyi(Rs,Rt)
def MPYI_acc_rr : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst += mpyi($src2, $src3)",
            [(set IntRegs:$dst,
            (add (mul IntRegs:$src2, IntRegs:$src3), IntRegs:$src1))],
            "$src1 = $dst">;

// Rx-=mpyi(Rs,#u8)
def MPYI_sub_ri : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, u8Imm:$src3),
            "$dst -= mpyi($src2, #$src3)",
            [(set IntRegs:$dst,
            (sub IntRegs:$src1, (mul IntRegs:$src2, u8ImmPred:$src3)))],
            "$src1 = $dst">;

// Multiply and use upper result.
// Rd=mpy(Rs,Rt.H):<<1:rnd:sat
// Rd=mpy(Rs,Rt.L):<<1:rnd:sat
// Rd=mpy(Rs,Rt)
def MPY : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
          "$dst = mpy($src1, $src2)",
          [(set IntRegs:$dst, (mulhs IntRegs:$src1, IntRegs:$src2))]>;

// Rd=mpy(Rs,Rt):rnd
// Rd=mpyu(Rs,Rt)
def MPYU : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
           "$dst = mpyu($src1, $src2)",
           [(set IntRegs:$dst, (mulhu IntRegs:$src1, IntRegs:$src2))]>;

// Multiply and use full result.
// Rdd=mpyu(Rs,Rt)
def MPYU64 : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             "$dst = mpyu($src1, $src2)",
             [(set DoubleRegs:$dst, (mul (i64 (anyext IntRegs:$src1)),
              (i64 (anyext IntRegs:$src2))))]>;

// Rdd=mpy(Rs,Rt)
def MPY64 : MInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             "$dst = mpy($src1, $src2)",
             [(set DoubleRegs:$dst, (mul (i64 (sext IntRegs:$src1)),
              (i64 (sext IntRegs:$src2))))]>;


// Multiply and accumulate, use full result.
// Rxx[+-]=mpy(Rs,Rt)
// Rxx+=mpy(Rs,Rt)
def MPY64_acc : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst += mpy($src2, $src3)",
            [(set DoubleRegs:$dst,
            (add (mul (i64 (sext IntRegs:$src2)), (i64 (sext IntRegs:$src3))),
               DoubleRegs:$src1))],
            "$src1 = $dst">;

// Rxx-=mpy(Rs,Rt)
def MPY64_sub : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst -= mpy($src2, $src3)",
            [(set DoubleRegs:$dst,
            (sub DoubleRegs:$src1,
                (mul (i64 (sext IntRegs:$src2)), (i64 (sext IntRegs:$src3)))))],
            "$src1 = $dst">;

// Rxx[+-]=mpyu(Rs,Rt)
// Rxx+=mpyu(Rs,Rt)
def MPYU64_acc : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                            IntRegs:$src2, IntRegs:$src3),
             "$dst += mpyu($src2, $src3)",
             [(set DoubleRegs:$dst, (add (mul (i64 (anyext IntRegs:$src2)),
              (i64 (anyext IntRegs:$src3))),
               DoubleRegs:$src1))],"$src1 = $dst">;

// Rxx-=mpyu(Rs,Rt)
def MPYU64_sub : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst += mpyu($src2, $src3)",
            [(set DoubleRegs:$dst,
            (sub DoubleRegs:$src1,
                    (mul (i64 (anyext IntRegs:$src2)),
                         (i64 (anyext IntRegs:$src3)))))],
            "$src1 = $dst">;


def ADDrr_acc : MInst_acc<(outs IntRegs: $dst), (ins IntRegs:$src1,
                            IntRegs:$src2, IntRegs:$src3),
             "$dst += add($src2, $src3)",
             [(set IntRegs:$dst, (add (add IntRegs:$src2, IntRegs:$src3),
                                      IntRegs:$src1))],
             "$src1 = $dst">;

def ADDri_acc : MInst_acc<(outs IntRegs: $dst), (ins IntRegs:$src1,
                            IntRegs:$src2, s8Imm:$src3),
             "$dst += add($src2, #$src3)",
             [(set IntRegs:$dst, (add (add IntRegs:$src2, s8ImmPred:$src3),
                                      IntRegs:$src1))],
             "$src1 = $dst">;

def SUBrr_acc : MInst_acc<(outs IntRegs: $dst), (ins IntRegs:$src1,
                            IntRegs:$src2, IntRegs:$src3),
             "$dst -= add($src2, $src3)",
             [(set IntRegs:$dst, (sub IntRegs:$src1, (add IntRegs:$src2,
                                                     IntRegs:$src3)))],
             "$src1 = $dst">;

def SUBri_acc : MInst_acc<(outs IntRegs: $dst), (ins IntRegs:$src1,
                            IntRegs:$src2, s8Imm:$src3),
             "$dst -= add($src2, #$src3)",
             [(set IntRegs:$dst, (sub IntRegs:$src1,
                                      (add IntRegs:$src2, s8ImmPred:$src3)))],
             "$src1 = $dst">;

//===----------------------------------------------------------------------===//
// MTYPE/MPYH -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/MPYS +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/MPYS -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/VB +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/VB -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/VH  +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/VH  -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ST +
//===----------------------------------------------------------------------===//
///
/// Assumptions::: ****** DO NOT IGNORE ********
/// 1. Make sure that in post increment store, the zero'th operand is always the
///    post increment operand.
/// 2. Make sure that the store value operand(Rt/Rtt) in a store is always the
///    last operand.
///
// Store doubleword.
let isPredicable = 1 in
def STrid : STInst<(outs),
            (ins MEMri:$addr, DoubleRegs:$src1),
            "memd($addr) = $src1",
            [(store DoubleRegs:$src1, ADDRriS11_3:$addr)]>;

// Indexed store double word.
let AddedComplexity = 10, isPredicable = 1 in
def STrid_indexed : STInst<(outs),
            (ins IntRegs:$src1, s11_3Imm:$src2,  DoubleRegs:$src3),
            "memd($src1+#$src2) = $src3",
            [(store DoubleRegs:$src3,
                                (add IntRegs:$src1, s11_3ImmPred:$src2))]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrid_GP : STInst<(outs),
            (ins globaladdress:$global, u16Imm:$offset, DoubleRegs:$src),
            "memd(#$global+$offset) = $src",
            []>;

let hasCtrlDep = 1, isPredicable = 1 in
def POST_STdri : STInstPI<(outs IntRegs:$dst),
            (ins DoubleRegs:$src1, IntRegs:$src2, s4Imm:$offset),
            "memd($src2++#$offset) = $src1",
            [(set IntRegs:$dst,
            (post_store DoubleRegs:$src1, IntRegs:$src2, s4_3ImmPred:$offset))],
            "$src2 = $dst">;

// Store doubleword conditionally.
// if ([!]Pv) memd(Rs+#u6:3)=Rtt
// if (Pv) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_cPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, DoubleRegs:$src2),
            "if ($src1) memd($addr) = $src2",
            []>;

// if (!Pv) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, DoubleRegs:$src2),
            "if (!$src1) memd($addr) = $src2",
            []>;

// if (Pv) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_cPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3,
                 DoubleRegs:$src4),
            "if ($src1) memd($src2+#$src3) = $src4",
            []>;

// if (!Pv) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3,
                 DoubleRegs:$src4),
            "if (!$src1) memd($src2+#$src3) = $src4",
            []>;

// if ([!]Pv) memd(Rx++#s4:3)=Rtt
// if (Pv) memd(Rx++#s4:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def POST_STdri_cPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, DoubleRegs:$src2, IntRegs:$src3,
                 s4_3Imm:$offset),
            "if ($src1) memd($src3++#$offset) = $src2",
            [],
            "$src3 = $dst">;

// if (!Pv) memd(Rx++#s4:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1,
    isPredicated = 1 in
def POST_STdri_cNotPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, DoubleRegs:$src2, IntRegs:$src3,
                 s4_3Imm:$offset),
            "if (!$src1) memd($src3++#$offset) = $src2",
            [],
            "$src3 = $dst">;


// Store byte.
// memb(Rs+#s11:0)=Rt
let isPredicable = 1 in
def STrib : STInst<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memb($addr) = $src1",
            [(truncstorei8 IntRegs:$src1, ADDRriS11_0:$addr)]>;

let AddedComplexity = 10, isPredicable = 1 in
def STrib_indexed : STInst<(outs),
            (ins IntRegs:$src1, s11_0Imm:$src2, IntRegs:$src3),
            "memb($src1+#$src2) = $src3",
            [(truncstorei8 IntRegs:$src3, (add IntRegs:$src1,
                                               s11_0ImmPred:$src2))]>;

// memb(gp+#u16:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP : STInst<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memb(#$global+$offset) = $src",
            []>;

let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP   : STInst<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memb(#$global) = $src",
            []>;

// memb(Rx++#s4:0)=Rt
let hasCtrlDep = 1, isPredicable = 1 in
def POST_STbri : STInstPI<(outs IntRegs:$dst), (ins IntRegs:$src1,
                                                    IntRegs:$src2,
                                                    s4Imm:$offset),
            "memb($src2++#$offset) = $src1",
            [(set IntRegs:$dst,
            (post_truncsti8 IntRegs:$src1, IntRegs:$src2,
                            s4_0ImmPred:$offset))],
            "$src2 = $dst">;

// Store byte conditionally.
// if ([!]Pv) memb(Rs+#u6:0)=Rt
// if (Pv) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memb($addr) = $src2",
            []>;

// if (!Pv) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memb($addr) = $src2",
            []>;

// if (Pv) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1) memb($src2+#$src3) = $src4",
            []>;

// if (!Pv) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1) memb($src2+#$src3) = $src4",
            []>;

// if ([!]Pv) memb(Rx++#s4:0)=Rt
// if (Pv) memb(Rx++#s4:0)=Rt
let mayStore = 1, hasCtrlDep = 1, isPredicated = 1 in
def POST_STbri_cPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1) memb($src3++#$offset) = $src2",
            [],"$src3 = $dst">;

// if (!Pv) memb(Rx++#s4:0)=Rt
let mayStore = 1, hasCtrlDep = 1, isPredicated = 1 in
def POST_STbri_cNotPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1) memb($src3++#$offset) = $src2",
            [],"$src3 = $dst">;


// Store halfword.
// memh(Rs+#s11:1)=Rt
let isPredicable = 1 in
def STrih : STInst<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memh($addr) = $src1",
            [(truncstorei16 IntRegs:$src1, ADDRriS11_1:$addr)]>;


let AddedComplexity = 10, isPredicable = 1 in
def STrih_indexed : STInst<(outs),
            (ins IntRegs:$src1, s11_1Imm:$src2,  IntRegs:$src3),
            "memh($src1+#$src2) = $src3",
            [(truncstorei16 IntRegs:$src3, (add IntRegs:$src1,
                                                s11_1ImmPred:$src2))]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP : STInst<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memh(#$global+$offset) = $src",
            []>;

let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP   : STInst<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memh(#$global) = $src",
            []>;

// memh(Rx++#s4:1)=Rt.H
// memh(Rx++#s4:1)=Rt
let hasCtrlDep = 1, isPredicable = 1 in
def POST_SThri : STInstPI<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4Imm:$offset),
            "memh($src2++#$offset) = $src1",
            [(set IntRegs:$dst,
            (post_truncsti16 IntRegs:$src1, IntRegs:$src2,
                             s4_1ImmPred:$offset))],
            "$src2 = $dst">;

// Store halfword conditionally.
// if ([!]Pv) memh(Rs+#u6:1)=Rt
// if (Pv) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memh($addr) = $src2",
            []>;

// if (!Pv) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memh($addr) = $src2",
            []>;

// if (Pv) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1) memh($src2+#$src3) = $src4",
            []>;

// if (!Pv) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1) memh($src2+#$src3) = $src4",
            []>;

// if ([!]Pv) memh(Rx++#s4:1)=Rt
// if (Pv) memh(Rx++#s4:1)=Rt
let mayStore = 1, hasCtrlDep = 1, isPredicated = 1 in
def POST_SThri_cPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1) memh($src3++#$offset) = $src2",
            [],"$src3 = $dst">;

// if (!Pv) memh(Rx++#s4:1)=Rt
let mayStore = 1, hasCtrlDep = 1, isPredicated = 1 in
def POST_SThri_cNotPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1) memh($src3++#$offset) = $src2",
            [],"$src3 = $dst">;


// Store word.
// Store predicate.
let Defs = [R10,R11] in
def STriw_pred : STInst<(outs),
            (ins MEMri:$addr, PredRegs:$src1),
            "Error; should not emit",
            []>;

// memw(Rs+#s11:2)=Rt
let isPredicable = 1 in
def STriw : STInst<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memw($addr) = $src1",
            [(store IntRegs:$src1, ADDRriS11_2:$addr)]>;

let AddedComplexity = 10, isPredicable = 1 in
def STriw_indexed : STInst<(outs),
            (ins IntRegs:$src1, s11_2Imm:$src2, IntRegs:$src3),
            "memw($src1+#$src2) = $src3",
            [(store IntRegs:$src3, (add IntRegs:$src1, s11_2ImmPred:$src2))]>;

let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP : STInst<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memw(#$global+$offset) = $src",
            []>;

let hasCtrlDep = 1, isPredicable = 1  in
def POST_STwri : STInstPI<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4Imm:$offset),
            "memw($src2++#$offset) = $src1",
            [(set IntRegs:$dst,
            (post_store IntRegs:$src1, IntRegs:$src2, s4_2ImmPred:$offset))],
            "$src2 = $dst">;

// Store word conditionally.
// if ([!]Pv) memw(Rs+#u6:2)=Rt
// if (Pv) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memw($addr) = $src2",
            []>;

// if (!Pv) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memw($addr) = $src2",
            []>;

// if (Pv) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1) memw($src2+#$src3) = $src4",
            []>;

// if (!Pv) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cNotPt : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1) memw($src2+#$src3) = $src4",
            []>;

// if ([!]Pv) memw(Rx++#s4:2)=Rt
// if (Pv) memw(Rx++#s4:2)=Rt
let mayStore = 1, hasCtrlDep = 1, isPredicated = 1 in
def POST_STwri_cPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1) memw($src3++#$offset) = $src2",
            [],"$src3 = $dst">;

// if (!Pv) memw(Rx++#s4:2)=Rt
let mayStore = 1, hasCtrlDep = 1, isPredicated = 1 in
def POST_STwri_cNotPt : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1) memw($src3++#$offset) = $src2",
            [],"$src3 = $dst">;



// Allocate stack frame.
let Defs = [R29, R30], Uses = [R31, R30], neverHasSideEffects = 1 in {
  def ALLOCFRAME : STInst<(outs),
             (ins i32imm:$amt),
             "allocframe(#$amt)",
             []>;
}
//===----------------------------------------------------------------------===//
// ST -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/ALU +
//===----------------------------------------------------------------------===//
// Logical NOT.
def NOT_rr64 : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1),
               "$dst = not($src1)",
               [(set DoubleRegs:$dst, (not DoubleRegs:$src1))]>;


// Sign extend word to doubleword.
def SXTW : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src1),
           "$dst = sxtw($src1)",
           [(set DoubleRegs:$dst, (sext IntRegs:$src1))]>;
//===----------------------------------------------------------------------===//
// STYPE/ALU -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/BIT +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/BIT -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// STYPE/COMPLEX +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/COMPLEX -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/PERM +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/PERM -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/PRED +
//===----------------------------------------------------------------------===//
// Predicate transfer.
let neverHasSideEffects = 1 in
def TFR_RsPd : SInst<(outs IntRegs:$dst), (ins PredRegs:$src1),
               "$dst = $src1  // Should almost never emit this",
               []>;

def TFR_PdRs : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1),
               "$dst = $src1  // Should almost never emit!",
               [(set PredRegs:$dst, (trunc IntRegs:$src1))]>;
//===----------------------------------------------------------------------===//
// STYPE/PRED -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/SHIFT +
//===----------------------------------------------------------------------===//
// Shift by immediate.
def ASR_ri : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
             "$dst = asr($src1, #$src2)",
             [(set IntRegs:$dst, (sra IntRegs:$src1, u5ImmPred:$src2))]>;

def ASRd_ri : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6Imm:$src2),
              "$dst = asr($src1, #$src2)",
              [(set DoubleRegs:$dst, (sra DoubleRegs:$src1, u6ImmPred:$src2))]>;

def ASL : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
          "$dst = asl($src1, #$src2)",
          [(set IntRegs:$dst, (shl IntRegs:$src1, u5ImmPred:$src2))]>;

def LSR_ri : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
             "$dst = lsr($src1, #$src2)",
             [(set IntRegs:$dst, (srl IntRegs:$src1, u5ImmPred:$src2))]>;

def LSRd_ri : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6Imm:$src2),
              "$dst = lsr($src1, #$src2)",
              [(set DoubleRegs:$dst, (srl DoubleRegs:$src1, u6ImmPred:$src2))]>;

def LSRd_ri_acc : SInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2,
                                                     u6Imm:$src3),
              "$dst += lsr($src2, #$src3)",
              [(set DoubleRegs:$dst, (add DoubleRegs:$src1,
                                          (srl DoubleRegs:$src2,
                                           u6ImmPred:$src3)))],
              "$src1 = $dst">;

// Shift by immediate and accumulate.
def ASR_rr_acc : SInst_acc<(outs IntRegs:$dst), (ins IntRegs:$src1,
                                                     IntRegs:$src2,
                                                     IntRegs:$src3),
                 "$dst += asr($src2, $src3)",
                 [], "$src1 = $dst">;

// Shift by immediate and add.
def ADDASL : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                             u3Imm:$src3),
             "$dst = addasl($src1, $src2, #$src3)",
             [(set IntRegs:$dst, (add IntRegs:$src1,
                                      (shl IntRegs:$src2,
                                           u3ImmPred:$src3)))]>;

// Shift by register.
def ASL_rr : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             "$dst = asl($src1, $src2)",
             [(set IntRegs:$dst, (shl IntRegs:$src1, IntRegs:$src2))]>;

def ASR_rr : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             "$dst = asr($src1, $src2)",
             [(set IntRegs:$dst, (sra IntRegs:$src1, IntRegs:$src2))]>;


def LSR_rr : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             "$dst = lsr($src1, $src2)",
             [(set IntRegs:$dst, (srl IntRegs:$src1, IntRegs:$src2))]>;

def LSLd : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, IntRegs:$src2),
           "$dst = lsl($src1, $src2)",
           [(set DoubleRegs:$dst, (shl DoubleRegs:$src1, IntRegs:$src2))]>;

def ASRd_rr : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                 IntRegs:$src2),
              "$dst = asr($src1, $src2)",
              [(set DoubleRegs:$dst, (sra DoubleRegs:$src1, IntRegs:$src2))]>;

def LSRd_rr : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                 IntRegs:$src2),
              "$dst = lsr($src1, $src2)",
              [(set DoubleRegs:$dst, (srl DoubleRegs:$src1, IntRegs:$src2))]>;

//===----------------------------------------------------------------------===//
// STYPE/SHIFT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/VH +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/VH -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/VW +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/VW -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SYSTEM/SUPER +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SYSTEM/USER +
//===----------------------------------------------------------------------===//
def SDHexagonBARRIER: SDTypeProfile<0, 0, []>;
def HexagonBARRIER: SDNode<"HexagonISD::BARRIER", SDHexagonBARRIER,
                           [SDNPHasChain]>;

let hasSideEffects = 1 in
def BARRIER : STInst<(outs), (ins),
                     "barrier",
                     [(HexagonBARRIER)]>;

//===----------------------------------------------------------------------===//
// SYSTEM/SUPER -
//===----------------------------------------------------------------------===//

// TFRI64 - assembly mapped.
let isReMaterializable = 1 in
def TFRI64 : ALU64_rr<(outs DoubleRegs:$dst), (ins s8Imm64:$src1),
             "$dst = #$src1",
             [(set DoubleRegs:$dst, s8Imm64Pred:$src1)]>;

// Pseudo instruction to encode a set of conditional transfers.
// This instruction is used instead of a mux and trades-off codesize
// for performance. We conduct this transformation optimistically in
// the hope that these instructions get promoted to dot-new transfers.
let AddedComplexity = 100 in
def TFR_condset_rr : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                        IntRegs:$src2,
                                                        IntRegs:$src3),
                     "Error; should not emit",
                     [(set IntRegs:$dst, (select PredRegs:$src1, IntRegs:$src2,
                                                 IntRegs:$src3))]>;

let AddedComplexity = 100 in
def TFR_condset_ri : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, s12Imm:$src3),
            "Error; should not emit",
            [(set IntRegs:$dst,
            (select PredRegs:$src1, IntRegs:$src2, s12ImmPred:$src3))]>;

let AddedComplexity = 100 in
def TFR_condset_ir : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, s12Imm:$src2, IntRegs:$src3),
            "Error; should not emit",
            [(set IntRegs:$dst,
            (select PredRegs:$src1, s12ImmPred:$src2, IntRegs:$src3))]>;

let AddedComplexity = 100 in
def TFR_condset_ii : ALU32_rr<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, s12Imm:$src2, s12Imm:$src3),
                     "Error; should not emit",
                     [(set IntRegs:$dst, (select PredRegs:$src1,
                                                 s12ImmPred:$src2,
                                                 s12ImmPred:$src3))]>;

// Generate frameindex addresses.
let isReMaterializable = 1 in
def TFR_FI : ALU32_ri<(outs IntRegs:$dst), (ins FrameIndex:$src1),
             "$dst = add($src1)",
             [(set IntRegs:$dst, ADDRri:$src1)]>;

//
// CR - Type.
//
let neverHasSideEffects = 1, Defs = [SA0, LC0] in {
def LOOP0_i : CRInst<(outs), (ins brtarget:$offset, u10Imm:$src2),
                      "loop0($offset, #$src2)",
                      []>;
}

let neverHasSideEffects = 1, Defs = [SA0, LC0] in {
def LOOP0_r : CRInst<(outs), (ins brtarget:$offset, IntRegs:$src2),
                      "loop0($offset, $src2)",
                      []>;
}

let isBranch = 1, isTerminator = 1, neverHasSideEffects = 1,
    Defs = [PC, LC0], Uses = [SA0, LC0] in {
def ENDLOOP0 : CRInst<(outs), (ins brtarget:$offset),
                      ":endloop0",
                      []>;
}

// Support for generating global address.
// Taken from X86InstrInfo.td.
def SDTHexagonCONST32 : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                             SDTCisPtrTy<0>]>;
def HexagonCONST32 : SDNode<"HexagonISD::CONST32",     SDTHexagonCONST32>;
def HexagonCONST32_GP : SDNode<"HexagonISD::CONST32_GP",     SDTHexagonCONST32>;

// This pattern is incorrect. When we add small data, we should change
// this pattern to use memw(#foo).
let isMoveImm = 1 in
def CONST32 : LDInst<(outs IntRegs:$dst), (ins globaladdress:$global),
              "$dst = CONST32(#$global)",
              [(set IntRegs:$dst,
              (load (HexagonCONST32 tglobaltlsaddr:$global)))]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32_set : LDInst<(outs IntRegs:$dst), (ins globaladdress:$global),
                  "$dst = CONST32(#$global)",
                  [(set IntRegs:$dst,
                  (HexagonCONST32 tglobaladdr:$global))]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32_set_jt : LDInst<(outs IntRegs:$dst), (ins jumptablebase:$jt),
                     "$dst = CONST32(#$jt)",
                     [(set IntRegs:$dst,
                     (HexagonCONST32 tjumptable:$jt))]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32GP_set : LDInst<(outs IntRegs:$dst), (ins globaladdress:$global),
                    "$dst = CONST32(#$global)",
                    [(set IntRegs:$dst,
                    (HexagonCONST32_GP tglobaladdr:$global))]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32_Int_Real : LDInst<(outs IntRegs:$dst), (ins i32imm:$global),
                       "$dst = CONST32(#$global)",
                       [(set IntRegs:$dst, imm:$global) ]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32_Label : LDInst<(outs IntRegs:$dst), (ins bblabel:$label),
                    "$dst = CONST32($label)",
                    [(set IntRegs:$dst, (HexagonCONST32 bbl:$label))]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST64_Int_Real : LDInst<(outs DoubleRegs:$dst), (ins i64imm:$global),
                       "$dst = CONST64(#$global)",
                       [(set DoubleRegs:$dst, imm:$global) ]>;

def TFR_PdFalse : SInst<(outs PredRegs:$dst), (ins),
                  "$dst = xor($dst, $dst)",
                  [(set PredRegs:$dst, 0)]>;

def MPY_trsext : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
                 "$dst = mpy($src1, $src2)",
                 [(set IntRegs:$dst,
                       (trunc (i64 (srl (i64 (mul (i64 (sext IntRegs:$src1)),
                                             (i64 (sext IntRegs:$src2)))),
                                        (i32 32)))))]>;

// Pseudo instructions.
def SDT_SPCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;

def SDT_SPCallSeqEnd : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def callseq_end : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeqEnd,
                  [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeqStart,
                    [SDNPHasChain, SDNPOutGlue]>;

def SDT_SPCall : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

def call : SDNode<"HexagonISD::CALL", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

// For tailcalls a HexagonTCRet SDNode has 3 SDNode Properties - a chain,
// Optional Flag and Variable Arguments.
// Its 1 Operand has pointer type.
def HexagonTCRet    : SDNode<"HexagonISD::TC_RETURN", SDT_SPCall,
                     [SDNPHasChain,  SDNPOptInGlue, SDNPVariadic]>;

let Defs = [R29, R30], Uses = [R31, R30, R29] in {
 def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt),
                        "Should never be emitted",
                        [(callseq_start timm:$amt)]>;
}

let Defs = [R29, R30, R31], Uses = [R29] in {
 def ADJCALLSTACKUP : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                      "Should never be emitted",
                      [(callseq_end timm:$amt1, timm:$amt2)]>;
}
// Call subroutine.
let isCall = 1, neverHasSideEffects = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
          R22, R23, R28, R31, P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def CALL : JInst<(outs), (ins calltarget:$dst, variable_ops),
             "call $dst", []>;
}

// Call subroutine from register.
let isCall = 1, neverHasSideEffects = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
          R22, R23, R28, R31, P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def CALLR : JRInst<(outs), (ins IntRegs:$dst, variable_ops),
              "callr $dst",
              []>;
 }

// Tail Calls.
let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
          R22, R23, R28, R31, P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def TCRETURNtg : JInst<(outs), (ins calltarget:$dst, variable_ops),
             "jump $dst // TAILCALL", []>;
}
let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
          R22, R23, R28, R31, P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def TCRETURNtext : JInst<(outs), (ins calltarget:$dst, variable_ops),
             "jump $dst // TAILCALL", []>;
}

let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10,
          R22, R23, R28, R31, P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def TCRETURNR : JInst<(outs), (ins IntRegs:$dst, variable_ops),
             "jumpr $dst // TAILCALL", []>;
}
// Map call instruction.
def : Pat<(call IntRegs:$dst),
      (CALLR IntRegs:$dst)>, Requires<[HasV2TOnly]>;
def : Pat<(call tglobaladdr:$dst),
      (CALL tglobaladdr:$dst)>, Requires<[HasV2TOnly]>;
def : Pat<(call texternalsym:$dst),
      (CALL texternalsym:$dst)>, Requires<[HasV2TOnly]>;
//Tail calls.
def : Pat<(HexagonTCRet tglobaladdr:$dst),
      (TCRETURNtg tglobaladdr:$dst)>;
def : Pat<(HexagonTCRet texternalsym:$dst),
      (TCRETURNtext texternalsym:$dst)>;
def : Pat<(HexagonTCRet IntRegs:$dst),
      (TCRETURNR IntRegs:$dst)>;

// Map from r0 = and(r1, 65535) to r0 = zxth(r1).
def : Pat <(and IntRegs:$src1, 65535),
      (ZXTH IntRegs:$src1)>;

// Map from r0 = and(r1, 255) to r0 = zxtb(r1).
def : Pat <(and IntRegs:$src1, 255),
      (ZXTB IntRegs:$src1)>;

// Map Add(p1, true) to p1 = not(p1).
//     Add(p1, false) should never be produced,
//     if it does, it got to be mapped to NOOP.
def : Pat <(add PredRegs:$src1, -1),
      (NOT_p PredRegs:$src1)>;

// Map from p0 = setlt(r0, r1) r2 = mux(p0, r3, r4) =>
//   p0 = cmp.lt(r0, r1), r0 = mux(p0, r2, r1).
def : Pat <(select (i1 (setlt IntRegs:$src1, IntRegs:$src2)), IntRegs:$src3,
                   IntRegs:$src4),
      (TFR_condset_rr (CMPLTrr IntRegs:$src1, IntRegs:$src2), IntRegs:$src4,
                      IntRegs:$src3)>, Requires<[HasV2TOnly]>;

// Map from p0 = pnot(p0); r0 = mux(p0, #i, #j) => r0 = mux(p0, #j, #i).
def : Pat <(select (not PredRegs:$src1), s8ImmPred:$src2, s8ImmPred:$src3),
      (TFR_condset_ii PredRegs:$src1, s8ImmPred:$src3, s8ImmPred:$src2)>;

// Map from p0 = pnot(p0); if (p0) jump => if (!p0) jump.
def : Pat <(brcond (not PredRegs:$src1), bb:$offset),
      (JMP_cNot PredRegs:$src1, bb:$offset)>;

// Map from p2 = pnot(p2); p1 = and(p0, p2) => p1 = and(p0, !p2).
def : Pat <(and PredRegs:$src1, (not PredRegs:$src2)),
      (AND_pnotp PredRegs:$src1, PredRegs:$src2)>;

// Map from store(globaladdress + x) -> memd(#foo + x).
let AddedComplexity = 100 in
def : Pat <(store DoubleRegs:$src1,
                  (add (HexagonCONST32_GP tglobaladdr:$global),
                       u16ImmPred:$offset)),
      (STrid_GP tglobaladdr:$global, u16ImmPred:$offset, DoubleRegs:$src1)>;

// Map from store(globaladdress) -> memd(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(store DoubleRegs:$src1, (HexagonCONST32_GP tglobaladdr:$global)),
      (STrid_GP tglobaladdr:$global, 0, DoubleRegs:$src1)>;

// Map from store(globaladdress + x) -> memw(#foo + x).
let AddedComplexity = 100 in
def : Pat <(store IntRegs:$src1, (add (HexagonCONST32_GP tglobaladdr:$global),
                                      u16ImmPred:$offset)),
      (STriw_GP tglobaladdr:$global, u16ImmPred:$offset, IntRegs:$src1)>;

// Map from store(globaladdress) -> memw(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(store IntRegs:$src1, (HexagonCONST32_GP tglobaladdr:$global)),
      (STriw_GP tglobaladdr:$global, 0, IntRegs:$src1)>;

// Map from store(globaladdress) -> memw(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(store IntRegs:$src1, (HexagonCONST32_GP tglobaladdr:$global)),
      (STriw_GP tglobaladdr:$global, 0, IntRegs:$src1)>;

// Map from store(globaladdress + x) -> memh(#foo + x).
let AddedComplexity = 100 in
def : Pat <(truncstorei16 IntRegs:$src1,
                          (add (HexagonCONST32_GP tglobaladdr:$global),
                               u16ImmPred:$offset)),
      (STrih_GP tglobaladdr:$global, u16ImmPred:$offset, IntRegs:$src1)>;

// Map from store(globaladdress) -> memh(#foo).
let AddedComplexity = 100 in
def : Pat <(truncstorei16 IntRegs:$src1,
                          (HexagonCONST32_GP tglobaladdr:$global)),
      (STh_GP tglobaladdr:$global, IntRegs:$src1)>;

// Map from store(globaladdress + x) -> memb(#foo + x).
let AddedComplexity = 100 in
def : Pat <(truncstorei8 IntRegs:$src1,
                         (add (HexagonCONST32_GP tglobaladdr:$global),
                              u16ImmPred:$offset)),
      (STrib_GP tglobaladdr:$global, u16ImmPred:$offset, IntRegs:$src1)>;

// Map from store(globaladdress) -> memb(#foo).
let AddedComplexity = 100 in
def : Pat <(truncstorei8 IntRegs:$src1,
                         (HexagonCONST32_GP tglobaladdr:$global)),
      (STb_GP tglobaladdr:$global, IntRegs:$src1)>;

// Map from load(globaladdress + x) -> memw(#foo + x).
let AddedComplexity = 100 in
def : Pat <(load (add (HexagonCONST32_GP tglobaladdr:$global),
                      u16ImmPred:$offset)),
      (LDriw_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memw(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(load (HexagonCONST32_GP tglobaladdr:$global)),
      (LDw_GP tglobaladdr:$global)>;

// Map from load(globaladdress + x) -> memd(#foo + x).
let AddedComplexity = 100 in
def : Pat <(i64 (load (add (HexagonCONST32_GP tglobaladdr:$global),
                           u16ImmPred:$offset))),
      (LDrid_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memw(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(i64 (load (HexagonCONST32_GP tglobaladdr:$global))),
      (LDd_GP tglobaladdr:$global)>;


// Map from Pd = load(globaladdress) -> Rd = memb(globaladdress + 0), Pd = Rd.
let AddedComplexity = 100 in
def : Pat <(i1 (load (HexagonCONST32_GP tglobaladdr:$global))),
      (TFR_PdRs (LDrib_GP tglobaladdr:$global, 0))>;

// Map from load(globaladdress + x) -> memh(#foo + x).
let AddedComplexity = 100 in
def : Pat <(sextloadi16 (add (HexagonCONST32_GP tglobaladdr:$global),
                             u16ImmPred:$offset)),
      (LDrih_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memh(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(sextloadi16 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDrih_GP tglobaladdr:$global, 0)>;

// Map from load(globaladdress + x) -> memuh(#foo + x).
let AddedComplexity = 100 in
def : Pat <(zextloadi16 (add (HexagonCONST32_GP tglobaladdr:$global),
                             u16ImmPred:$offset)),
      (LDriuh_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memuh(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(zextloadi16 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDriuh_GP tglobaladdr:$global, 0)>;

// Map from load(globaladdress + x) -> memuh(#foo + x).
let AddedComplexity = 100 in
def : Pat <(extloadi16 (add (HexagonCONST32_GP tglobaladdr:$global),
                            u16ImmPred:$offset)),
      (LDriuh_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memuh(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(extloadi16 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDriuh_GP tglobaladdr:$global, 0)>;
// Map from load(globaladdress + x) -> memub(#foo + x).
let AddedComplexity = 100 in
def : Pat <(zextloadi8 (add (HexagonCONST32_GP tglobaladdr:$global),
                            u16ImmPred:$offset)),
      (LDriub_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memuh(#foo + 0).
let AddedComplexity = 100 in
def : Pat <(zextloadi8 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDriub_GP tglobaladdr:$global, 0)>;

// Map from load(globaladdress + x) -> memb(#foo + x).
let AddedComplexity = 100 in
def : Pat <(sextloadi8 (add (HexagonCONST32_GP tglobaladdr:$global),
                            u16ImmPred:$offset)),
      (LDrib_GP tglobaladdr:$global, u16ImmPred:$offset)>;

// Map from load(globaladdress) -> memb(#foo).
let AddedComplexity = 100 in
def : Pat <(extloadi8 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDb_GP tglobaladdr:$global)>;

// Map from load(globaladdress) -> memb(#foo).
let AddedComplexity = 100 in
def : Pat <(sextloadi8 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDb_GP tglobaladdr:$global)>;

// Map from load(globaladdress) -> memub(#foo).
let AddedComplexity = 100 in
def : Pat <(zextloadi8 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDub_GP tglobaladdr:$global)>;

// When the Interprocedural Global Variable optimizer realizes that a
// certain global variable takes only two constant values, it shrinks the
// global to a boolean. Catch those loads here in the following 3 patterns.
let AddedComplexity = 100 in
def : Pat <(extloadi1 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDb_GP tglobaladdr:$global)>;

let AddedComplexity = 100 in
def : Pat <(sextloadi1 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDb_GP tglobaladdr:$global)>;

let AddedComplexity = 100 in
def : Pat <(zextloadi1 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDub_GP tglobaladdr:$global)>;

// Map from load(globaladdress) -> memh(#foo).
let AddedComplexity = 100 in
def : Pat <(extloadi16 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDh_GP tglobaladdr:$global)>;

// Map from load(globaladdress) -> memh(#foo).
let AddedComplexity = 100 in
def : Pat <(sextloadi16 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDh_GP tglobaladdr:$global)>;

// Map from load(globaladdress) -> memuh(#foo).
let AddedComplexity = 100 in
def : Pat <(zextloadi16 (HexagonCONST32_GP tglobaladdr:$global)),
      (LDuh_GP tglobaladdr:$global)>;

// Map from i1 loads to 32 bits. This assumes that the i1* is byte aligned.
def : Pat <(i32 (zextloadi1 ADDRriS11_0:$addr)),
      (AND_rr (LDrib ADDRriS11_0:$addr), (TFRI 0x1))>;

// Map from Rdd = sign_extend_inreg(Rss, i32) -> Rdd = SXTW(Rss.lo).
def : Pat <(i64 (sext_inreg DoubleRegs:$src1, i32)),
      (i64 (SXTW (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg)))>;

// Map from Rdd = sign_extend_inreg(Rss, i16) -> Rdd = SXTW(SXTH(Rss.lo)).
def : Pat <(i64 (sext_inreg DoubleRegs:$src1, i16)),
      (i64 (SXTW (SXTH (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg))))>;

// Map from Rdd = sign_extend_inreg(Rss, i8) -> Rdd = SXTW(SXTB(Rss.lo)).
def : Pat <(i64 (sext_inreg DoubleRegs:$src1, i8)),
      (i64 (SXTW (SXTB (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg))))>;

// We want to prevent emiting pnot's as much as possible.
// Map brcond with an unsupported setcc to a JMP_cNot.
def : Pat <(brcond (i1 (setne IntRegs:$src1, IntRegs:$src2)), bb:$offset),
      (JMP_cNot (CMPEQrr IntRegs:$src1, IntRegs:$src2), bb:$offset)>;

def : Pat <(brcond (i1 (setne IntRegs:$src1, s10ImmPred:$src2)), bb:$offset),
      (JMP_cNot (CMPEQri IntRegs:$src1, s10ImmPred:$src2), bb:$offset)>;

def : Pat <(brcond (i1 (setne PredRegs:$src1, (i1 -1))), bb:$offset),
      (JMP_cNot PredRegs:$src1, bb:$offset)>;

def : Pat <(brcond (i1 (setne PredRegs:$src1, (i1 0))), bb:$offset),
      (JMP_c PredRegs:$src1, bb:$offset)>;

def : Pat <(brcond (i1 (setlt IntRegs:$src1, s8ImmPred:$src2)), bb:$offset),
      (JMP_cNot (CMPGEri IntRegs:$src1, s8ImmPred:$src2), bb:$offset)>;

def : Pat <(brcond (i1 (setlt IntRegs:$src1, IntRegs:$src2)), bb:$offset),
      (JMP_c (CMPLTrr IntRegs:$src1, IntRegs:$src2), bb:$offset)>;

def : Pat <(brcond (i1 (setuge DoubleRegs:$src1, DoubleRegs:$src2)),
                   bb:$offset),
      (JMP_cNot (CMPGTU64rr DoubleRegs:$src2, DoubleRegs:$src1),
                   bb:$offset)>;

def : Pat <(brcond (i1 (setule IntRegs:$src1, IntRegs:$src2)), bb:$offset),
      (JMP_cNot (CMPGTUrr IntRegs:$src1, IntRegs:$src2), bb:$offset)>;

def : Pat <(brcond (i1 (setule DoubleRegs:$src1, DoubleRegs:$src2)),
                   bb:$offset),
      (JMP_cNot (CMPGTU64rr DoubleRegs:$src1, DoubleRegs:$src2),
                   bb:$offset)>;

// Map from a 64-bit select to an emulated 64-bit mux.
// Hexagon does not support 64-bit MUXes; so emulate with combines.
def : Pat <(select PredRegs:$src1, DoubleRegs:$src2, DoubleRegs:$src3),
      (COMBINE_rr
      (MUX_rr PredRegs:$src1,
      (EXTRACT_SUBREG DoubleRegs:$src2, subreg_hireg),
      (EXTRACT_SUBREG DoubleRegs:$src3, subreg_hireg)),
      (MUX_rr PredRegs:$src1,
      (EXTRACT_SUBREG DoubleRegs:$src2, subreg_loreg),
      (EXTRACT_SUBREG DoubleRegs:$src3, subreg_loreg)))>;

// Map from a 1-bit select to logical ops.
// From LegalizeDAG.cpp: (B1 ? B2 : B3) <=> (B1 & B2)|(!B1&B3).
def : Pat <(select PredRegs:$src1, PredRegs:$src2, PredRegs:$src3),
      (OR_pp (AND_pp PredRegs:$src1, PredRegs:$src2),
             (AND_pp (NOT_p PredRegs:$src1), PredRegs:$src3))>;

// Map Pd = load(addr) -> Rs = load(addr); Pd = Rs.
def : Pat<(i1 (load ADDRriS11_2:$addr)),
      (i1 (TFR_PdRs (i32 (LDrib ADDRriS11_2:$addr))))>;

// Map for truncating from 64 immediates to 32 bit immediates.
def : Pat<(i32 (trunc DoubleRegs:$src)),
      (i32 (EXTRACT_SUBREG DoubleRegs:$src, subreg_loreg))>;

// Map for truncating from i64 immediates to i1 bit immediates.
def :  Pat<(i1 (trunc DoubleRegs:$src)),
       (i1 (TFR_PdRs (i32(EXTRACT_SUBREG DoubleRegs:$src, subreg_loreg))))>;

// Map memb(Rs) = Rdd -> memb(Rs) = Rt.
def : Pat<(truncstorei8 DoubleRegs:$src, ADDRriS11_0:$addr),
      (STrib ADDRriS11_0:$addr, (i32 (EXTRACT_SUBREG DoubleRegs:$src,
                                                     subreg_loreg)))>;

// Map memh(Rs) = Rdd -> memh(Rs) = Rt.
def : Pat<(truncstorei16 DoubleRegs:$src, ADDRriS11_0:$addr),
      (STrih ADDRriS11_0:$addr, (i32 (EXTRACT_SUBREG DoubleRegs:$src,
                                                     subreg_loreg)))>;

// Map memw(Rs) = Rdd -> memw(Rs) = Rt.
def : Pat<(truncstorei32 DoubleRegs:$src, ADDRriS11_0:$addr),
      (STriw ADDRriS11_0:$addr, (i32 (EXTRACT_SUBREG DoubleRegs:$src,
                                                     subreg_loreg)))>;

// Map from i1 = constant<-1>; memw(addr) = i1 -> r0 = 1; memw(addr) = r0.
def : Pat<(store (i1 -1), ADDRriS11_2:$addr),
      (STrib ADDRriS11_2:$addr, (TFRI 1))>;

let AddedComplexity = 100 in
// Map from i1 = constant<-1>; memw(CONST32(#foo)) = i1 -> r0 = 1;
// memw(#foo) = r0
def : Pat<(store (i1 -1), (HexagonCONST32_GP tglobaladdr:$global)),
      (STb_GP tglobaladdr:$global, (TFRI 1))>;


// Map from i1 = constant<-1>; store i1 -> r0 = 1; store r0.
def : Pat<(store (i1 -1), ADDRriS11_2:$addr),
      (STrib ADDRriS11_2:$addr, (TFRI 1))>;

// Map from memb(Rs) = Pd -> Rt = mux(Pd, #0, #1); store Rt.
def : Pat<(store PredRegs:$src1, ADDRriS11_2:$addr),
      (STrib ADDRriS11_2:$addr, (i32 (MUX_ii PredRegs:$src1, 1, 0)) )>;

// Map Rdd = anyext(Rs) -> Rdd = sxtw(Rs).
// Hexagon_TODO: We can probably use combine but that will cost 2 instructions.
// Better way to do this?
def : Pat<(i64 (anyext IntRegs:$src1)),
      (i64 (SXTW IntRegs:$src1))>;

// Map cmple -> cmpgt.
// rs <= rt -> !(rs > rt).
def : Pat<(i1 (setle IntRegs:$src1, s10ImmPred:$src2)),
      (i1 (NOT_p (CMPGTri IntRegs:$src1, s10ImmPred:$src2)))>;

// rs <= rt -> !(rs > rt).
def : Pat<(i1 (setle IntRegs:$src1, IntRegs:$src2)),
      (i1 (NOT_p (CMPGTrr IntRegs:$src1, IntRegs:$src2)))>;

// Rss <= Rtt -> !(Rss > Rtt).
def : Pat<(i1 (setle DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (NOT_p (CMPGT64rr DoubleRegs:$src1, DoubleRegs:$src2)))>;

// Map cmpne -> cmpeq.
// Hexagon_TODO: We should improve on this.
// rs != rt -> !(rs == rt).
def : Pat <(i1 (setne IntRegs:$src1, s10ImmPred:$src2)),
      (i1 (NOT_p(i1 (CMPEQri IntRegs:$src1, s10ImmPred:$src2))))>;

// Map cmpne(Rs) -> !cmpeqe(Rs).
// rs != rt -> !(rs == rt).
def : Pat <(i1 (setne IntRegs:$src1, IntRegs:$src2)),
      (i1 (NOT_p(i1 (CMPEQrr IntRegs:$src1, IntRegs:$src2))))>;

// Convert setne back to xor for hexagon since we compute w/ pred registers.
def : Pat <(i1 (setne PredRegs:$src1, PredRegs:$src2)),
      (i1 (XOR_pp PredRegs:$src1, PredRegs:$src2))>;

// Map cmpne(Rss) -> !cmpew(Rss).
// rs != rt -> !(rs == rt).
def : Pat <(i1 (setne DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (NOT_p(i1 (CMPEHexagon4rr DoubleRegs:$src1, DoubleRegs:$src2))))>;

// Map cmpge(Rs, Rt) -> !(cmpgt(Rs, Rt).
// rs >= rt -> !(rt > rs).
def : Pat <(i1 (setge IntRegs:$src1, IntRegs:$src2)),
      (i1 (NOT_p(i1 (CMPGTrr IntRegs:$src2, IntRegs:$src1))))>;

def : Pat <(i1 (setge IntRegs:$src1, s8ImmPred:$src2)),
      (i1 (CMPGEri IntRegs:$src1, s8ImmPred:$src2))>;

// Map cmpge(Rss, Rtt) -> !cmpgt(Rtt, Rss).
// rss >= rtt -> !(rtt > rss).
def : Pat <(i1 (setge DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (NOT_p(i1 (CMPGT64rr DoubleRegs:$src2, DoubleRegs:$src1))))>;

// Map cmplt(Rs, Imm) -> !cmpge(Rs, Imm).
// rs < rt -> !(rs >= rt).
def : Pat <(i1 (setlt IntRegs:$src1, s8ImmPred:$src2)),
      (i1 (NOT_p (CMPGEri IntRegs:$src1, s8ImmPred:$src2)))>;

// Map cmplt(Rs, Rt) -> cmplt(Rs, Rt).
// rs < rt -> rs < rt. Let assembler map it.
def : Pat <(i1 (setlt IntRegs:$src1, IntRegs:$src2)),
      (i1 (CMPLTrr IntRegs:$src2, IntRegs:$src1))>;

// Map cmplt(Rss, Rtt) -> cmpgt(Rtt, Rss).
// rss < rtt -> (rtt > rss).
def : Pat <(i1 (setlt DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (CMPGT64rr DoubleRegs:$src2, DoubleRegs:$src1))>;

// Map from cmpltu(Rs, Rd) -> !cmpgtu(Rs, Rd - 1).
// rs < rt -> rt > rs.
def : Pat <(i1 (setult IntRegs:$src1, IntRegs:$src2)),
      (i1 (CMPGTUrr IntRegs:$src2, IntRegs:$src1))>;

// Map from cmpltu(Rss, Rdd) -> !cmpgtu(Rss, Rdd - 1).
// rs < rt -> rt > rs.
def : Pat <(i1 (setult DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (CMPGTU64rr DoubleRegs:$src2, DoubleRegs:$src1))>;

// Map from Rs >= Rt -> !(Rt > Rs).
// rs >= rt -> !(rt > rs).
def : Pat <(i1 (setuge IntRegs:$src1, IntRegs:$src2)),
      (i1 (NOT_p (CMPGTUrr IntRegs:$src2, IntRegs:$src1)))>;

// Map from Rs >= Rt -> !(Rt > Rs).
// rs >= rt -> !(rt > rs).
def : Pat <(i1 (setuge DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (NOT_p (CMPGTU64rr DoubleRegs:$src2, DoubleRegs:$src1)))>;

// Map from cmpleu(Rs, Rs) -> !cmpgtu(Rs, Rs).
// Map from (Rs <= Rt) -> !(Rs > Rt).
def : Pat <(i1 (setule IntRegs:$src1, IntRegs:$src2)),
      (i1 (NOT_p (CMPGTUrr IntRegs:$src1, IntRegs:$src2)))>;

// Map from cmpleu(Rss, Rtt) -> !cmpgtu(Rss, Rtt-1).
// Map from (Rs <= Rt) -> !(Rs > Rt).
def : Pat <(i1 (setule DoubleRegs:$src1, DoubleRegs:$src2)),
      (i1 (NOT_p (CMPGTU64rr DoubleRegs:$src1, DoubleRegs:$src2)))>;

// Sign extends.
// i1 -> i32
def : Pat <(i32 (sext PredRegs:$src1)),
      (i32 (MUX_ii PredRegs:$src1, -1, 0))>;

// Convert sign-extended load back to load and sign extend.
// i8 -> i64
def:  Pat <(i64 (sextloadi8 ADDRriS11_0:$src1)),
      (i64 (SXTW (LDrib ADDRriS11_0:$src1)))>;

// Convert any-extended load back to load and sign extend.
// i8 -> i64
def:  Pat <(i64 (extloadi8 ADDRriS11_0:$src1)),
      (i64 (SXTW (LDrib ADDRriS11_0:$src1)))>;

// Convert sign-extended load back to load and sign extend.
// i16 -> i64
def:  Pat <(i64 (sextloadi16 ADDRriS11_1:$src1)),
      (i64 (SXTW (LDrih ADDRriS11_1:$src1)))>;

// Convert sign-extended load back to load and sign extend.
// i32 -> i64
def:  Pat <(i64 (sextloadi32 ADDRriS11_2:$src1)),
      (i64 (SXTW (LDriw ADDRriS11_2:$src1)))>;


// Zero extends.
// i1 -> i32
def : Pat <(i32 (zext PredRegs:$src1)),
      (i32 (MUX_ii PredRegs:$src1, 1, 0))>;

// i1 -> i64
def : Pat <(i64 (zext PredRegs:$src1)),
      (i64 (COMBINE_rr (TFRI 0), (MUX_ii PredRegs:$src1, 1, 0)))>;

// i32 -> i64
def : Pat <(i64 (zext IntRegs:$src1)),
      (i64 (COMBINE_rr (TFRI 0), IntRegs:$src1))>;

// i8 -> i64
def:  Pat <(i64 (zextloadi8 ADDRriS11_0:$src1)),
      (i64 (COMBINE_rr (TFRI 0), (LDriub ADDRriS11_0:$src1)))>;

// i16 -> i64
def:  Pat <(i64 (zextloadi16 ADDRriS11_1:$src1)),
      (i64 (COMBINE_rr (TFRI 0), (LDriuh ADDRriS11_1:$src1)))>;

// i32 -> i64
def:  Pat <(i64 (zextloadi32 ADDRriS11_2:$src1)),
      (i64 (COMBINE_rr (TFRI 0), (LDriw ADDRriS11_2:$src1)))>;

def:  Pat <(i32 (zextloadi1 ADDRriS11_0:$src1)),
      (i32 (LDriw ADDRriS11_0:$src1))>;

// Map from Rs = Pd to Pd = mux(Pd, #1, #0)
def : Pat <(i32 (zext PredRegs:$src1)),
      (i32 (MUX_ii PredRegs:$src1, 1, 0))>;

// Map from Rs = Pd to Pd = mux(Pd, #1, #0)
def : Pat <(i32 (anyext PredRegs:$src1)),
      (i32 (MUX_ii PredRegs:$src1, 1, 0))>;

// Map from Rss = Pd to Rdd = sxtw (mux(Pd, #1, #0))
def : Pat <(i64 (anyext PredRegs:$src1)),
      (i64 (SXTW (i32 (MUX_ii PredRegs:$src1, 1, 0))))>;


// Any extended 64-bit load.
// anyext i32 -> i64
def:  Pat <(i64 (extloadi32 ADDRriS11_2:$src1)),
      (i64 (COMBINE_rr (TFRI 0), (LDriw ADDRriS11_2:$src1)))>;

// anyext i16 -> i64.
def:  Pat <(i64 (extloadi16 ADDRriS11_2:$src1)),
      (i64 (COMBINE_rr (TFRI 0), (LDrih ADDRriS11_2:$src1)))>;

// Map from Rdd = zxtw(Rs) -> Rdd = combine(0, Rs).
def : Pat<(i64 (zext IntRegs:$src1)),
      (i64 (COMBINE_rr (TFRI 0), IntRegs:$src1))>;

// Multiply 64-bit unsigned and use upper result.
def : Pat <(mulhu DoubleRegs:$src1, DoubleRegs:$src2),
      (MPYU64_acc(COMBINE_rr (TFRI 0),
                 (EXTRACT_SUBREG
                 (LSRd_ri(MPYU64_acc(MPYU64_acc(COMBINE_rr (TFRI 0),
                                 (EXTRACT_SUBREG (LSRd_ri(MPYU64
                                 (EXTRACT_SUBREG DoubleRegs:$src1,
                                                 subreg_loreg),
                                 (EXTRACT_SUBREG DoubleRegs:$src2,
                                                 subreg_loreg)),
                                  32) ,subreg_loreg)),
                                 (EXTRACT_SUBREG DoubleRegs:$src1,
                                                 subreg_hireg),
                                 (EXTRACT_SUBREG DoubleRegs:$src2,
                                                 subreg_loreg)),
                              (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg),
                              (EXTRACT_SUBREG DoubleRegs:$src2, subreg_hireg)),
                          32),subreg_loreg)),
                 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg),
                 (EXTRACT_SUBREG DoubleRegs:$src2, subreg_hireg)
      )>;

// Multiply 64-bit signed and use upper result.
def : Pat <(mulhs DoubleRegs:$src1, DoubleRegs:$src2),
      (MPY64_acc(COMBINE_rr (TFRI 0),
                 (EXTRACT_SUBREG
                 (LSRd_ri(MPY64_acc(MPY64_acc(COMBINE_rr (TFRI 0),
                                 (EXTRACT_SUBREG (LSRd_ri(MPYU64
                                 (EXTRACT_SUBREG DoubleRegs:$src1,
                                                 subreg_loreg),
                                 (EXTRACT_SUBREG DoubleRegs:$src2,
                                                 subreg_loreg)),
                                  32) ,subreg_loreg)),
                                 (EXTRACT_SUBREG DoubleRegs:$src1,
                                                 subreg_hireg),
                                 (EXTRACT_SUBREG DoubleRegs:$src2,
                                                 subreg_loreg)),
                              (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg),
                              (EXTRACT_SUBREG DoubleRegs:$src2, subreg_hireg)),
                          32),subreg_loreg)),
                 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg),
                 (EXTRACT_SUBREG DoubleRegs:$src2, subreg_hireg)
      )>;

// Hexagon specific ISD nodes.
def SDTHexagonADJDYNALLOC : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>]>;
def Hexagon_ADJDYNALLOC : SDNode<"HexagonISD::ADJDYNALLOC",
                                 SDTHexagonADJDYNALLOC>;
// Needed to tag these instructions for stack layout.
let usesCustomInserter = 1 in
def ADJDYNALLOC : ALU32_ri<(outs IntRegs:$dst), (ins IntRegs:$src1,
                                                     s16Imm:$src2),
                  "$dst = add($src1, #$src2)",
                  [(set IntRegs:$dst, (Hexagon_ADJDYNALLOC IntRegs:$src1,
                                                           s16ImmPred:$src2))]>;

def SDTHexagonARGEXTEND : SDTypeProfile<1, 1, []>;
def Hexagon_ARGEXTEND : SDNode<"HexagonISD::ARGEXTEND", SDTHexagonARGEXTEND>;
def ARGEXTEND : ALU32_rr <(outs IntRegs:$dst), (ins IntRegs:$src1),
                "$dst = $src1",
                [(set IntRegs:$dst, (Hexagon_ARGEXTEND IntRegs:$src1))]>;

let AddedComplexity = 100 in
def : Pat<(i32 (sext_inreg (Hexagon_ARGEXTEND IntRegs:$src1), i16)),
      (TFR IntRegs:$src1)>;


def SDHexagonBR_JT: SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def HexagonBR_JT: SDNode<"HexagonISD::BR_JT", SDHexagonBR_JT, [SDNPHasChain]>;

let isBranch=1, isIndirectBranch=1, isTerminator=1, isBarrier = 1 in
def BR_JT : JRInst<(outs), (ins IntRegs:$src),
                   "jumpr $src",
                   [(HexagonBR_JT IntRegs:$src)]>;
def HexagonWrapperJT: SDNode<"HexagonISD::WrapperJT", SDTIntUnaryOp>;

def : Pat<(HexagonWrapperJT tjumptable:$dst),
          (CONST32_set_jt tjumptable:$dst)>;


//===----------------------------------------------------------------------===//
// V3 Instructions +
//===----------------------------------------------------------------------===//

include "HexagonInstrInfoV3.td"

//===----------------------------------------------------------------------===//
// V3 Instructions -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// V4 Instructions +
//===----------------------------------------------------------------------===//

include "HexagonInstrInfoV4.td"

//===----------------------------------------------------------------------===//
// V4 Instructions -
//===----------------------------------------------------------------------===//
