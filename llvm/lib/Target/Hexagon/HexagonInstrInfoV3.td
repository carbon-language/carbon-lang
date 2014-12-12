//=- HexagonInstrInfoV3.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V3 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

def callv3 : SDNode<"HexagonISD::CALLv3", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

def callv3nr : SDNode<"HexagonISD::CALLv3nr", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// J +
//===----------------------------------------------------------------------===//
// Call subroutine.
let isCall = 1, hasSideEffects = 1, validSubTargets = HasV3SubT,
    Defs = VolatileV3.Regs, isPredicable = 1,
    isExtended = 0, isExtendable = 1, opExtendable = 0,
    isExtentSigned = 1, opExtentBits = 24, opExtentAlign = 2 in
class T_Call<string ExtStr>
  : JInst<(outs), (ins calltarget:$dst),
      "call " # ExtStr # "$dst", [], "", J_tc_2early_SLOT23> {
  let BaseOpcode = "call";
  bits<24> dst;

  let IClass = 0b0101;
  let Inst{27-25} = 0b101;
  let Inst{24-16,13-1} = dst{23-2};
  let Inst{0} = 0b0;
}

let isCall = 1, hasSideEffects = 1, validSubTargets = HasV3SubT,
    Defs = VolatileV3.Regs, isPredicated = 1,
    isExtended = 0, isExtendable = 1, opExtendable = 1,
    isExtentSigned = 1, opExtentBits = 17, opExtentAlign = 2 in
class T_CallPred<bit IfTrue, string ExtStr>
  : JInst<(outs), (ins PredRegs:$Pu, calltarget:$dst),
      CondStr<"$Pu", IfTrue, 0>.S # "call " # ExtStr # "$dst",
      [], "", J_tc_2early_SLOT23> {
  let BaseOpcode = "call";
  let isPredicatedFalse = !if(IfTrue,0,1);
  bits<2> Pu;
  bits<17> dst;

  let IClass = 0b0101;
  let Inst{27-24} = 0b1101;
  let Inst{23-22,20-16,13,7-1} = dst{16-2};
  let Inst{21} = !if(IfTrue,0,1);
  let Inst{11} = 0b0;
  let Inst{9-8} = Pu;
}

multiclass T_Calls<string ExtStr> {
  def NAME : T_Call<ExtStr>;
  def t    : T_CallPred<1, ExtStr>;
  def f    : T_CallPred<0, ExtStr>;
}

let isCodeGenOnly = 0 in
defm J2_call: T_Calls<"">, PredRel;

//===----------------------------------------------------------------------===//
// J -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// JR +
//===----------------------------------------------------------------------===//
// Call subroutine from register.
let isCall = 1, hasSideEffects = 0,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, R28, R31,
                P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def CALLRv3 : JRInst<(outs), (ins IntRegs:$dst),
              "callr $dst",
              []>, Requires<[HasV3TOnly]>;
 }

//===----------------------------------------------------------------------===//
// JR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/ALU +
//===----------------------------------------------------------------------===//

let AddedComplexity = 200 in
def MAXw_dd : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                    DoubleRegs:$src2),
              "$dst = max($src2, $src1)",
              [(set (i64 DoubleRegs:$dst),
                    (i64 (select (i1 (setlt (i64 DoubleRegs:$src2),
                                            (i64 DoubleRegs:$src1))),
                                 (i64 DoubleRegs:$src1),
                                 (i64 DoubleRegs:$src2))))]>,
Requires<[HasV3T]>;

let AddedComplexity = 200 in
def MINw_dd : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                    DoubleRegs:$src2),
              "$dst = min($src2, $src1)",
              [(set (i64 DoubleRegs:$dst),
                    (i64 (select (i1 (setgt (i64 DoubleRegs:$src2),
                                            (i64 DoubleRegs:$src1))),
                                 (i64 DoubleRegs:$src1),
                                 (i64 DoubleRegs:$src2))))]>,
Requires<[HasV3T]>;

//===----------------------------------------------------------------------===//
// ALU64/ALU -
//===----------------------------------------------------------------------===//




//def : Pat <(brcond (i1 (seteq (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegEzt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setne (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegNzt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setle (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegLezt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setge (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegGezt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setgt (i32 IntRegs:$src1), -1)), bb:$offset),
//      (JMP_RegGezt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;


// Map call instruction
def : Pat<(call (i32 IntRegs:$dst)),
      (J2_call (i32 IntRegs:$dst))>, Requires<[HasV3T]>;
def : Pat<(call tglobaladdr:$dst),
      (J2_call tglobaladdr:$dst)>, Requires<[HasV3T]>;
def : Pat<(call texternalsym:$dst),
      (J2_call texternalsym:$dst)>, Requires<[HasV3T]>;
