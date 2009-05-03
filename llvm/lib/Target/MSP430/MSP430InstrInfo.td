//===- MSP430InstrInfo.td - MSP430 Instruction defs -----------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the MSP430 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "MSP430InstrFormats.td"

//===----------------------------------------------------------------------===//
// Type Constraints.
//===----------------------------------------------------------------------===//
class SDTCisI8<int OpNum> : SDTCisVT<OpNum, i8>;
class SDTCisI16<int OpNum> : SDTCisVT<OpNum, i16>;

//===----------------------------------------------------------------------===//
// Type Profiles.
//===----------------------------------------------------------------------===//
def SDT_MSP430Call         : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def SDT_MSP430CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i16>]>;
def SDT_MSP430CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i16>, SDTCisVT<1, i16>]>;
def SDT_MSP430Wrapper      : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;
def SDT_MSP430SetCC        : SDTypeProfile<1, 2, [SDTCisVT<0, i8>,
                                                  SDTCisVT<1, i8>, SDTCisVT<2, i16>]>;
def SDT_MSP430Cmp          : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def SDT_MSP430BrCond       : SDTypeProfile<0, 3, [SDTCisVT<0, OtherVT>,
                                                  SDTCisVT<1, i8>, SDTCisVT<2, i16>]>;
def SDT_MSP430Select       : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, 
                                                  SDTCisVT<3, i8>, SDTCisVT<4, i16>]>;

//===----------------------------------------------------------------------===//
// MSP430 Specific Node Definitions.
//===----------------------------------------------------------------------===//
def MSP430retflag : SDNode<"MSP430ISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInFlag]>;

def MSP430rra     : SDNode<"MSP430ISD::RRA", SDTIntUnaryOp, []>;

def MSP430call    : SDNode<"MSP430ISD::CALL", SDT_MSP430Call,
                     [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;
def MSP430callseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_MSP430CallSeqStart,
                        [SDNPHasChain, SDNPOutFlag]>;
def MSP430callseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_MSP430CallSeqEnd,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def MSP430Wrapper : SDNode<"MSP430ISD::Wrapper", SDT_MSP430Wrapper>;
def MSP430setcc   : SDNode<"MSP430ISD::SETCC", SDT_MSP430SetCC>;
def MSP430cmp     : SDNode<"MSP430ISD::CMP", SDT_MSP430Cmp>;
def MSP430brcond  : SDNode<"MSP430ISD::BRCOND", SDT_MSP430BrCond, [SDNPHasChain]>;
def MSP430select  : SDNode<"MSP430ISD::SELECT", SDT_MSP430Select>;

//===----------------------------------------------------------------------===//
// MSP430 Operand Definitions.
//===----------------------------------------------------------------------===//

// Address operands
def memsrc : Operand<i16> {
  let PrintMethod = "printSrcMemOperand";
  let MIOperandInfo = (ops GR16, i16imm);
}

def memdst : Operand<i16> {
  let PrintMethod = "printSrcMemOperand";
  let MIOperandInfo = (ops GR16, i16imm);
}

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;

// Operand for printing out a condition code.
def cc : Operand<i8> {
  let PrintMethod = "printCCOperand";
}

//===----------------------------------------------------------------------===//
// MSP430 Complex Pattern Definitions.
//===----------------------------------------------------------------------===//

def addr : ComplexPattern<iPTR, 2, "SelectAddr", [], []>;

//===----------------------------------------------------------------------===//
// Pattern Fragments
def zextloadi16i8 : PatFrag<(ops node:$ptr), (i16 (zextloadi8 node:$ptr))>;
def  extloadi16i8 : PatFrag<(ops node:$ptr), (i16 ( extloadi8 node:$ptr))>;

//===----------------------------------------------------------------------===//
// Instruction list..

// ADJCALLSTACKDOWN/UP implicitly use/def SP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
// Pessimistically assume ADJCALLSTACKDOWN / ADJCALLSTACKUP will become
// sub / add which can clobber SRW.
let Defs = [SPW, SRW], Uses = [SPW] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i16imm:$amt),
                              "#ADJCALLSTACKDOWN",
                              [(MSP430callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i16imm:$amt1, i16imm:$amt2),
                              "#ADJCALLSTACKUP",
                              [(MSP430callseq_end timm:$amt1, timm:$amt2)]>;
}

let usesCustomDAGSchedInserter = 1 in {
  def Select16 : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2, i8imm:$cc),
                        "# Select16 PSEUDO",
                        [(set GR16:$dst,
                          (MSP430select GR16:$src1, GR16:$src2, imm:$cc, SRW))]>;
}

let neverHasSideEffects = 1 in
def NOP : Pseudo<(outs), (ins), "nop", []>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// FIXME: Provide proper encoding!
let isReturn = 1, isTerminator = 1 in {
  def RET : Pseudo<(outs), (ins), "ret", [(MSP430retflag)]>;
}

// Conditional branches
let isBranch = 1, isTerminator = 1, Uses = [SRW] in {
def JCC : Pseudo<(outs), (ins brtarget:$dst, cc:$cc),
                          "j$cc $dst",
                          [(MSP430brcond bb:$dst, imm:$cc, SRW)]>;
} // Uses = [SRW]

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers. SPW is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [R12W, R13W, R14W, R15W, SRW],
      Uses = [SPW] in {
    def CALLi     : Pseudo<(outs), (ins i16imm:$dst, variable_ops),
                           "call\t${dst:call}", [(MSP430call imm:$dst)]>;
    def CALLr     : Pseudo<(outs), (ins GR16:$dst, variable_ops),
                           "call\t$dst", [(MSP430call GR16:$dst)]>;
    def CALLm     : Pseudo<(outs), (ins memsrc:$dst, variable_ops),
                           "call\t${dst:mem}", [(MSP430call (load addr:$dst))]>;
  }


//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
let Defs = [SPW], Uses = [SPW], neverHasSideEffects=1 in {
let mayLoad = 1 in
def POP16r   : Pseudo<(outs GR16:$reg), (ins), "pop.w\t$reg", []>;

let mayStore = 1 in
def PUSH16r  : Pseudo<(outs), (ins GR16:$reg), "push.w\t$reg",[]>;
}

//===----------------------------------------------------------------------===//
// Move Instructions

// FIXME: Provide proper encoding!
let neverHasSideEffects = 1 in {
def MOV8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src),
                     "mov.b\t{$src, $dst}",
                     []>;
def MOV16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src),
                     "mov.w\t{$src, $dst}",
                     []>;
}

// FIXME: Provide proper encoding!
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV8ri  : Pseudo<(outs GR8:$dst), (ins i8imm:$src),
                     "mov.b\t{$src, $dst}",
                     [(set GR8:$dst, imm:$src)]>;
def MOV16ri : Pseudo<(outs GR16:$dst), (ins i16imm:$src),
                     "mov.w\t{$src, $dst}",
                     [(set GR16:$dst, imm:$src)]>;
}

let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def MOV8rm  : Pseudo<(outs GR8:$dst), (ins memsrc:$src),
                "mov.b\t{$src, $dst}",
                [(set GR8:$dst, (load addr:$src))]>;
def MOV16rm : Pseudo<(outs GR16:$dst), (ins memsrc:$src),
                "mov.w\t{$src, $dst}",
                [(set GR16:$dst, (load addr:$src))]>;
}

def MOVZX16rr8 : Pseudo<(outs GR16:$dst), (ins GR8:$src),
                "mov.b\t{$src, $dst}",
                [(set GR16:$dst, (zext GR8:$src))]>;
def MOVZX16rm8 : Pseudo<(outs GR16:$dst), (ins memsrc:$src),
                "mov.b\t{$src, $dst}",
                [(set GR16:$dst, (zextloadi16i8 addr:$src))]>;

def MOV8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "mov.b\t{$src, $dst}",
                [(store (i8 imm:$src), addr:$dst)]>;
def MOV16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "mov.w\t{$src, $dst}",
                [(store (i16 imm:$src), addr:$dst)]>;

def MOV8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "mov.b\t{$src, $dst}",
                [(store GR8:$src, addr:$dst)]>;
def MOV16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "mov.w\t{$src, $dst}",
                [(store GR16:$src, addr:$dst)]>;

//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let isTwoAddress = 1 in {

let Defs = [SRW] in {

let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
// FIXME: Provide proper encoding!
def ADD8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                     "add.b\t{$src2, $dst}",
                     [(set GR8:$dst, (add GR8:$src1, GR8:$src2)),
                      (implicit SRW)]>;
def ADD16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "add.w\t{$src2, $dst}",
                     [(set GR16:$dst, (add GR16:$src1, GR16:$src2)),
                      (implicit SRW)]>;
}

def ADD8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                     "add.b\t{$src2, $dst}",
                     [(set GR8:$dst, (add GR8:$src1, (load addr:$src2))),
                      (implicit SRW)]>;
def ADD16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                     "add.w\t{$src2, $dst}",
                     [(set GR16:$dst, (add GR16:$src1, (load addr:$src2))),
                      (implicit SRW)]>;

def ADD8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                     "add.b\t{$src2, $dst}",
                     [(set GR8:$dst, (add GR8:$src1, imm:$src2)),
                      (implicit SRW)]>;
def ADD16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                     "add.w\t{$src2, $dst}",
                     [(set GR16:$dst, (add GR16:$src1, imm:$src2)),
                      (implicit SRW)]>;

let isTwoAddress = 0 in {
def ADD8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "add.b\t{$src, $dst}",
                [(store (add (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def ADD16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "add.w\t{$src, $dst}",
                [(store (add (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def ADD8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "add.b\t{$src, $dst}",
                [(store (add (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def ADD16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "add.w\t{$src, $dst}",
                [(store (add (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def ADD8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "add.b\t{$src, $dst}",
                [(store (add (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def ADD16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "add.w\t{$src, $dst}",
                [(store (add (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}

let Uses = [SRW] in {

let isCommutable = 1 in { // X = ADDC Y, Z  == X = ADDC Z, Y
def ADC8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                     "addc.b\t{$src2, $dst}",
                     [(set GR8:$dst, (adde GR8:$src1, GR8:$src2)),
                      (implicit SRW)]>;
def ADC16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "addc.w\t{$src2, $dst}",
                     [(set GR16:$dst, (adde GR16:$src1, GR16:$src2)),
                      (implicit SRW)]>;
} // isCommutable

def ADC8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                     "addc.b\t{$src2, $dst}",
                     [(set GR8:$dst, (adde GR8:$src1, imm:$src2)),
                      (implicit SRW)]>;
def ADC16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                     "addc.w\t{$src2, $dst}",
                     [(set GR16:$dst, (adde GR16:$src1, imm:$src2)),
                      (implicit SRW)]>;

def ADC8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                     "addc.b\t{$src2, $dst}",
                     [(set GR8:$dst, (adde GR8:$src1, (load addr:$src2))),
                      (implicit SRW)]>;
def ADC16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                     "addc.w\t{$src2, $dst}",
                     [(set GR16:$dst, (adde GR16:$src1, (load addr:$src2))),
                      (implicit SRW)]>;

let isTwoAddress = 0 in {
def ADC8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "addc.b\t{$src, $dst}",
                [(store (adde (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def ADC16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "addc.w\t{$src, $dst}",
                [(store (adde (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def ADC8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "addc.b\t{$src, $dst}",
                [(store (adde (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def ADC16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "addc.w\t{$src, $dst}",
                [(store (adde (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def ADC8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "addc.b\t{$src, $dst}",
                [(store (adde (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def ADC16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "addc.w\t{$src, $dst}",
                [(store (adde (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}

} // Uses = [SRW]

let isCommutable = 1 in { // X = AND Y, Z  == X = AND Z, Y
def AND8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                     "and.b\t{$src2, $dst}",
                     [(set GR8:$dst, (and GR8:$src1, GR8:$src2)),
                      (implicit SRW)]>;
def AND16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "and.w\t{$src2, $dst}",
                     [(set GR16:$dst, (and GR16:$src1, GR16:$src2)),
                      (implicit SRW)]>;
}

def AND8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                     "and.b\t{$src2, $dst}",
                     [(set GR8:$dst, (and GR8:$src1, imm:$src2)),
                      (implicit SRW)]>;
def AND16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                     "and.w\t{$src2, $dst}",
                     [(set GR16:$dst, (and GR16:$src1, imm:$src2)),
                      (implicit SRW)]>;

def AND8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                     "and.b\t{$src2, $dst}",
                     [(set GR8:$dst, (and GR8:$src1, (load addr:$src2))),
                      (implicit SRW)]>;
def AND16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                     "and.w\t{$src2, $dst}",
                     [(set GR16:$dst, (and GR16:$src1, (load addr:$src2))),
                      (implicit SRW)]>;

let isTwoAddress = 0 in {
def AND8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "and.b\t{$src, $dst}",
                [(store (and (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def AND16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "and.w\t{$src, $dst}",
                [(store (and (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def AND8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "and.b\t{$src, $dst}",
                [(store (and (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def AND16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "and.w\t{$src, $dst}",
                [(store (and (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def AND8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "and.b\t{$src, $dst}",
                [(store (and (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def AND16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "and.w\t{$src, $dst}",
                [(store (and (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}


let isCommutable = 1 in { // X = XOR Y, Z  == X = XOR Z, Y
def XOR8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                     "xor.b\t{$src2, $dst}",
                     [(set GR8:$dst, (xor GR8:$src1, GR8:$src2)),
                      (implicit SRW)]>;
def XOR16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "xor.w\t{$src2, $dst}",
                     [(set GR16:$dst, (xor GR16:$src1, GR16:$src2)),
                      (implicit SRW)]>;
}

def XOR8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                     "xor.b\t{$src2, $dst}",
                     [(set GR8:$dst, (xor GR8:$src1, imm:$src2)),
                      (implicit SRW)]>;
def XOR16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                     "xor.w\t{$src2, $dst}",
                     [(set GR16:$dst, (xor GR16:$src1, imm:$src2)),
                      (implicit SRW)]>;

def XOR8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                     "xor.b\t{$src2, $dst}",
                     [(set GR8:$dst, (xor GR8:$src1, (load addr:$src2))),
                      (implicit SRW)]>;
def XOR16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                     "xor.w\t{$src2, $dst}",
                     [(set GR16:$dst, (xor GR16:$src1, (load addr:$src2))),
                      (implicit SRW)]>;

let isTwoAddress = 0 in {
def XOR8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "xor.b\t{$src, $dst}",
                [(store (xor (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def XOR16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "xor.w\t{$src, $dst}",
                [(store (xor (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def XOR8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "xor.b\t{$src, $dst}",
                [(store (xor (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def XOR16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "xor.w\t{$src, $dst}",
                [(store (xor (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def XOR8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "xor.b\t{$src, $dst}",
                [(store (xor (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def XOR16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "xor.w\t{$src, $dst}",
                [(store (xor (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}


def SUB8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                     "sub.b\t{$src2, $dst}",
                     [(set GR8:$dst, (sub GR8:$src1, GR8:$src2)),
                      (implicit SRW)]>;
def SUB16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "sub.w\t{$src2, $dst}",
                     [(set GR16:$dst, (sub GR16:$src1, GR16:$src2)),
                      (implicit SRW)]>;

def SUB8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                     "sub.b\t{$src2, $dst}",
                     [(set GR8:$dst, (sub GR8:$src1, imm:$src2)),
                      (implicit SRW)]>;
def SUB16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                     "sub.w\t{$src2, $dst}",
                     [(set GR16:$dst, (sub GR16:$src1, imm:$src2)),
                      (implicit SRW)]>;

def SUB8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                     "sub.b\t{$src2, $dst}",
                     [(set GR8:$dst, (sub GR8:$src1, (load addr:$src2))),
                      (implicit SRW)]>;
def SUB16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                     "sub.w\t{$src2, $dst}",
                     [(set GR16:$dst, (sub GR16:$src1, (load addr:$src2))),
                      (implicit SRW)]>;

let isTwoAddress = 0 in {
def SUB8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "sub.b\t{$src, $dst}",
                [(store (sub (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def SUB16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "sub.w\t{$src, $dst}",
                [(store (sub (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def SUB8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "sub.b\t{$src, $dst}",
                [(store (sub (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def SUB16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "sub.w\t{$src, $dst}",
                [(store (sub (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def SUB8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "sub.b\t{$src, $dst}",
                [(store (sub (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def SUB16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "sub.w\t{$src, $dst}",
                [(store (sub (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}

let Uses = [SRW] in {
def SBC8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                     "subc.b\t{$src2, $dst}",
                     [(set GR8:$dst, (sube GR8:$src1, GR8:$src2)),
                      (implicit SRW)]>;
def SBC16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                     "subc.w\t{$src2, $dst}",
                     [(set GR16:$dst, (sube GR16:$src1, GR16:$src2)),
                      (implicit SRW)]>;

def SBC8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                     "subc.b\t{$src2, $dst}",
                     [(set GR8:$dst, (sube GR8:$src1, imm:$src2)),
                      (implicit SRW)]>;
def SBC16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                     "subc.w\t{$src2, $dst}",
                     [(set GR16:$dst, (sube GR16:$src1, imm:$src2)),
                      (implicit SRW)]>;

def SBC8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                     "subc.b\t{$src2, $dst}",
                     [(set GR8:$dst, (sube GR8:$src1, (load addr:$src2))),
                      (implicit SRW)]>;
def SBC16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                     "subc.w\t{$src2, $dst}",
                     [(set GR16:$dst, (sube GR16:$src1, (load addr:$src2))),
                      (implicit SRW)]>;

let isTwoAddress = 0 in {
def SBC8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "subc.b\t{$src, $dst}",
                [(store (sube (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def SBC16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "subc.w\t{$src, $dst}",
                [(store (sube (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def SBC8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "subc.b\t{$src, $dst}",
                [(store (sube (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def SBC16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "subc.w\t{$src, $dst}",
                [(store (sube (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def SBC8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "subc.b\t{$src, $dst}",
                [(store (sube (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def SBC16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "subc.w\t{$src, $dst}",
                [(store (sube (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}

} // Uses = [SRW]

// FIXME: Provide proper encoding!
def SAR16r1 : Pseudo<(outs GR16:$dst), (ins GR16:$src),
                     "rra.w\t$dst",
                     [(set GR16:$dst, (MSP430rra GR16:$src)),
                      (implicit SRW)]>;

def SEXT16r : Pseudo<(outs GR16:$dst), (ins GR16:$src),
                     "sxt\t$dst",
                     [(set GR16:$dst, (sext_inreg GR16:$src, i8)),
                      (implicit SRW)]>;

//def SEXT16r : Pseudo<(outs GR16:$dst), (ins GR16:$src),
//                     "sxt\t$dst",
//                     [(set GR16:$dst, (sext_inreg GR16:$src, i8)),
//                      (implicit SRW)]>;

} // Defs = [SRW]

let isCommutable = 1 in { // X = OR Y, Z  == X = OR Z, Y
def OR8rr  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                    "bis.b\t{$src2, $dst}",
                    [(set GR8:$dst, (or GR8:$src1, GR8:$src2))]>;
def OR16rr : Pseudo<(outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                    "bis.w\t{$src2, $dst}",
                    [(set GR16:$dst, (or GR16:$src1, GR16:$src2))]>;
}

def OR8ri  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                    "bis.b\t{$src2, $dst}",
                    [(set GR8:$dst, (or GR8:$src1, imm:$src2))]>;
def OR16ri : Pseudo<(outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                    "bis.w\t{$src2, $dst}",
                    [(set GR16:$dst, (or GR16:$src1, imm:$src2))]>;

def OR8rm  : Pseudo<(outs GR8:$dst), (ins GR8:$src1, memsrc:$src2),
                    "bis.b\t{$src2, $dst}",
                    [(set GR8:$dst, (or GR8:$src1, (load addr:$src2)))]>;
def OR16rm : Pseudo<(outs GR16:$dst), (ins GR16:$src1, memsrc:$src2),
                    "bis.w\t{$src2, $dst}",
                    [(set GR16:$dst, (or GR16:$src1, (load addr:$src2)))]>;

let isTwoAddress = 0 in {
def OR8mr  : Pseudo<(outs), (ins memdst:$dst, GR8:$src),
                "bis.b\t{$src, $dst}",
                [(store (or (load addr:$dst), GR8:$src), addr:$dst),
                 (implicit SRW)]>;
def OR16mr : Pseudo<(outs), (ins memdst:$dst, GR16:$src),
                "bis.w\t{$src, $dst}",
                [(store (or (load addr:$dst), GR16:$src), addr:$dst),
                 (implicit SRW)]>;

def OR8mi  : Pseudo<(outs), (ins memdst:$dst, i8imm:$src),
                "bis.b\t{$src, $dst}",
                [(store (or (load addr:$dst), (i8 imm:$src)), addr:$dst),
                 (implicit SRW)]>;
def OR16mi : Pseudo<(outs), (ins memdst:$dst, i16imm:$src),
                "bis.w\t{$src, $dst}",
                [(store (or (load addr:$dst), (i16 imm:$src)), addr:$dst),
                 (implicit SRW)]>;

def OR8mm  : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "bis.b\t{$src, $dst}",
                [(store (or (load addr:$dst), (i8 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
def OR16mm : Pseudo<(outs), (ins memdst:$dst, memsrc:$src),
                "bis.w\t{$src, $dst}",
                [(store (or (load addr:$dst), (i16 (load addr:$src))), addr:$dst),
                 (implicit SRW)]>;
}

} // isTwoAddress = 1

// Integer comparisons
let Defs = [SRW] in {
def CMP8rr  : Pseudo<(outs), (ins GR8:$src1, GR8:$src2),
                     "cmp.b\t{$src1, $src2}",
                     [(MSP430cmp GR8:$src1, GR8:$src2), (implicit SRW)]>;
def CMP16rr : Pseudo<(outs), (ins GR16:$src1, GR16:$src2),
                     "cmp.w\t{$src1, $src2}",
                     [(MSP430cmp GR16:$src1, GR16:$src2), (implicit SRW)]>;

def CMP8ri  : Pseudo<(outs), (ins GR8:$src1, i8imm:$src2),
                     "cmp.b\t{$src1, $src2}",
                     [(MSP430cmp GR8:$src1, imm:$src2), (implicit SRW)]>;
def CMP16ri : Pseudo<(outs), (ins GR16:$src1, i16imm:$src2),
                     "cmp.w\t{$src1, $src2}",
                     [(MSP430cmp GR16:$src1, imm:$src2), (implicit SRW)]>;

def CMP8rm  : Pseudo<(outs), (ins GR8:$src1, memsrc:$src2),
                     "cmp.b\t{$src1, $src2}",
                     [(MSP430cmp GR8:$src1, (load addr:$src2)), (implicit SRW)]>;
def CMP16rm : Pseudo<(outs), (ins GR16:$src1, memsrc:$src2),
                     "cmp.w\t{$src1, $src2}",
                     [(MSP430cmp GR16:$src1, (load addr:$src2)), (implicit SRW)]>;

def CMP8mr  : Pseudo<(outs), (ins memsrc:$src1, GR8:$src2),
                "cmp.b\t{$src1, $src2}",
                [(MSP430cmp (load addr:$src1), GR8:$src2), (implicit SRW)]>;
def CMP16mr : Pseudo<(outs), (ins memsrc:$src1, GR16:$src2),
                "cmp.w\t{$src1, $src2}",
                [(MSP430cmp (load addr:$src1), GR16:$src2), (implicit SRW)]>;

def CMP8mi  : Pseudo<(outs), (ins memsrc:$src1, i8imm:$src2),
                "cmp.b\t{$src1, $src2}",
                [(MSP430cmp (load addr:$src1), (i8 imm:$src2)), (implicit SRW)]>;
def CMP16mi : Pseudo<(outs), (ins memsrc:$src1, i16imm:$src2),
                "cmp.w\t{$src1, $src2}",
                [(MSP430cmp (load addr:$src1), (i16 imm:$src2)), (implicit SRW)]>;

def CMP8mm  : Pseudo<(outs), (ins memsrc:$src1, memsrc:$src2),
                "cmp.b\t{$src1, $src2}",
                [(MSP430cmp (load addr:$src1), (i8 (load addr:$src2))), (implicit SRW)]>;
def CMP16mm : Pseudo<(outs), (ins memsrc:$src1, memsrc:$src2),
                "cmp.w\t{$src1, $src2}",
                [(MSP430cmp (load addr:$src1), (i16 (load addr:$src2))), (implicit SRW)]>;
} // Defs = [SRW]

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns

// extload
def : Pat<(extloadi16i8 addr:$src), (MOVZX16rm8 addr:$src)>;

// truncs
def : Pat<(i8 (trunc GR16:$src)),
          (EXTRACT_SUBREG GR16:$src, subreg_8bit)>;

// GlobalAddress
def : Pat<(i16 (MSP430Wrapper tglobaladdr:$dst)), (MOV16ri tglobaladdr:$dst)>;

def : Pat<(add GR16:$src1, (MSP430Wrapper tglobaladdr :$src2)),
          (ADD16ri GR16:$src1, tglobaladdr:$src2)>;

// calls
def : Pat<(MSP430call (i16 tglobaladdr:$dst)),
          (CALLi tglobaladdr:$dst)>;
