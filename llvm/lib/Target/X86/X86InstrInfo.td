//===- X86InstrInfo.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 instruction set, defining the instructions, and
// properties of the instructions which are needed for code generation, machine
// code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// X86 specific DAG Nodes.
//

def SDTIntShiftDOp: SDTypeProfile<1, 3,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                   SDTCisInt<0>, SDTCisInt<3>]>;

def SDTX86CmpTest : SDTypeProfile<1, 2, [SDTCisVT<0, FlagVT>, SDTCisSameAs<1, 2>]>;

def SDTX86Cmov    : SDTypeProfile<1, 4,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                   SDTCisVT<3, i8>, SDTCisVT<4, FlagVT>]>;

def SDTX86BrCond  : SDTypeProfile<0, 3,
                                  [SDTCisVT<0, OtherVT>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, FlagVT>]>;

def SDTX86SetCC   : SDTypeProfile<1, 2,
                                  [SDTCisVT<0, i8>, SDTCisVT<1, i8>,
                                   SDTCisVT<2, FlagVT>]>;

def SDTX86Ret     : SDTypeProfile<0, 1, [SDTCisVT<0, i16>]>;

def SDT_X86CallSeqStart : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;
def SDT_X86CallSeqEnd   : SDTypeProfile<0, 2, [ SDTCisVT<0, i32>,
                                                SDTCisVT<1, i32> ]>;

def SDT_X86Call   : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

def SDTX86FpGet   : SDTypeProfile<1, 0, [SDTCisVT<0, f64>]>;
def SDTX86FpSet   : SDTypeProfile<0, 1, [SDTCisFP<0>]>;

def SDTX86Fld     : SDTypeProfile<1, 2, [SDTCisVT<0, f64>,
                                         SDTCisPtrTy<1>, SDTCisVT<2, OtherVT>]>;
def SDTX86Fst     : SDTypeProfile<0, 3, [SDTCisFP<0>,
                                         SDTCisPtrTy<1>, SDTCisVT<2, OtherVT>]>;
def SDTX86Fild64m : SDTypeProfile<1, 1, [SDTCisVT<0, f64>, SDTCisPtrTy<1>]>;

def SDTX86RdTsc   : SDTypeProfile<0, 0, []>;

def X86addflag : SDNode<"X86ISD::ADD_FLAG", SDTIntBinOp ,
                        [SDNPCommutative, SDNPAssociative, SDNPOutFlag]>;
def X86subflag : SDNode<"X86ISD::SUB_FLAG", SDTIntBinOp,
                        [SDNPOutFlag]>;
def X86adc     : SDNode<"X86ISD::ADC" ,     SDTIntBinOp ,
                        [SDNPCommutative, SDNPAssociative]>;
def X86sbb     : SDNode<"X86ISD::SBB" ,     SDTIntBinOp>;

def X86shld    : SDNode<"X86ISD::SHLD",     SDTIntShiftDOp>;
def X86shrd    : SDNode<"X86ISD::SHRD",     SDTIntShiftDOp>;

def X86cmp     : SDNode<"X86ISD::CMP" ,     SDTX86CmpTest,  []>;
def X86test    : SDNode<"X86ISD::TEST",     SDTX86CmpTest,  []>;

def X86cmov    : SDNode<"X86ISD::CMOV",     SDTX86Cmov,    
                        [SDNPOutFlag]>;
def X86brcond  : SDNode<"X86ISD::BRCOND",   SDTX86BrCond,
                        [SDNPHasChain]>;
def X86setcc   : SDNode<"X86ISD::SETCC",    SDTX86SetCC,    []>;

def X86retflag : SDNode<"X86ISD::RET_FLAG", SDTX86Ret,
                        [SDNPHasChain, SDNPOptInFlag]>;

def X86callseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_X86CallSeqStart,
                        [SDNPHasChain]>;
def X86callseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_X86CallSeqEnd,
                        [SDNPHasChain]>;

def X86call    : SDNode<"X86ISD::CALL",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;

def X86fpget   : SDNode<"X86ISD::FP_GET_RESULT", SDTX86FpGet,
                        [SDNPHasChain, SDNPInFlag]>;
def X86fpset   : SDNode<"X86ISD::FP_SET_RESULT", SDTX86FpSet,
                        [SDNPHasChain, SDNPOutFlag]>;

def X86fld     : SDNode<"X86ISD::FLD",      SDTX86Fld,
                        [SDNPHasChain]>;
def X86fst     : SDNode<"X86ISD::FST",      SDTX86Fst,
                        [SDNPHasChain]>;
def X86fild64m : SDNode<"X86ISD::FILD64m",  SDTX86Fild64m,
                        [SDNPHasChain]>;

def X86rdtsc   : SDNode<"X86ISD::RDTSC_DAG",SDTX86RdTsc,
                        [SDNPHasChain, SDNPOutFlag]>;

//===----------------------------------------------------------------------===//
// X86 Operand Definitions.
//

// *mem - Operand definitions for the funky X86 addressing mode operands.
//
class X86MemOperand<string printMethod> : Operand<i32> {
  let PrintMethod = printMethod;
  let NumMIOperands = 4;
  let MIOperandInfo = (ops R32, i8imm, R32, i32imm);
}

def i8mem   : X86MemOperand<"printi8mem">;
def i16mem  : X86MemOperand<"printi16mem">;
def i32mem  : X86MemOperand<"printi32mem">;
def i64mem  : X86MemOperand<"printi64mem">;
def f32mem  : X86MemOperand<"printf32mem">;
def f64mem  : X86MemOperand<"printf64mem">;
def f80mem  : X86MemOperand<"printf80mem">;

def SSECC : Operand<i8> {
  let PrintMethod = "printSSECC";
}

// A couple of more descriptive operand definitions.
// 16-bits but only 8 bits are significant.
def i16i8imm  : Operand<i16>;
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32>;

// PCRelative calls need special operand formatting.
let PrintMethod = "printCallOperand" in
  def calltarget : Operand<i32>;

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;

//===----------------------------------------------------------------------===//
// X86 Complex Pattern Definitions.
//

// Define X86 specific addressing mode.
def addr    : ComplexPattern<i32, 4, "SelectAddr", []>;
def leaaddr : ComplexPattern<i32, 4, "SelectLEAAddr",
                             [add, frameindex, constpool,
                              globaladdr, tglobaladdr, externalsym]>;

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<5> val> {
  bits<5> Value = val;
}

def Pseudo     : Format<0>; def RawFrm     : Format<1>;
def AddRegFrm  : Format<2>; def MRMDestReg : Format<3>;
def MRMDestMem : Format<4>; def MRMSrcReg  : Format<5>;
def MRMSrcMem  : Format<6>;
def MRM0r  : Format<16>; def MRM1r  : Format<17>; def MRM2r  : Format<18>;
def MRM3r  : Format<19>; def MRM4r  : Format<20>; def MRM5r  : Format<21>;
def MRM6r  : Format<22>; def MRM7r  : Format<23>;
def MRM0m  : Format<24>; def MRM1m  : Format<25>; def MRM2m  : Format<26>;
def MRM3m  : Format<27>; def MRM4m  : Format<28>; def MRM5m  : Format<29>;
def MRM6m  : Format<30>; def MRM7m  : Format<31>;

//===----------------------------------------------------------------------===//
// X86 Instruction Predicate Definitions.
def HasSSE1 : Predicate<"X86Vector >= SSE">;
def HasSSE2 : Predicate<"X86Vector >= SSE2">;
def HasSSE3 : Predicate<"X86Vector >= SSE3">;
def FPStack : Predicate<"X86Vector < SSE2">;

//===----------------------------------------------------------------------===//
// X86 specific pattern fragments.
//

// ImmType - This specifies the immediate type used by an instruction. This is
// part of the ad-hoc solution used to emit machine instruction encodings by our
// machine code emitter.
class ImmType<bits<2> val> {
  bits<2> Value = val;
}
def NoImm  : ImmType<0>;
def Imm8   : ImmType<1>;
def Imm16  : ImmType<2>;
def Imm32  : ImmType<3>;

// FPFormat - This specifies what form this FP instruction has.  This is used by
// the Floating-Point stackifier pass.
class FPFormat<bits<3> val> {
  bits<3> Value = val;
}
def NotFP      : FPFormat<0>;
def ZeroArgFP  : FPFormat<1>;
def OneArgFP   : FPFormat<2>;
def OneArgFPRW : FPFormat<3>;
def TwoArgFP   : FPFormat<4>;
def CompareFP  : FPFormat<5>;
def CondMovFP  : FPFormat<6>;
def SpecialFP  : FPFormat<7>;


class X86Inst<bits<8> opcod, Format f, ImmType i, dag ops, string AsmStr>
  : Instruction {
  let Namespace = "X86";

  bits<8> Opcode = opcod;
  Format Form = f;
  bits<5> FormBits = Form.Value;
  ImmType ImmT = i;
  bits<2> ImmTypeBits = ImmT.Value;

  dag OperandList = ops;
  string AsmString = AsmStr;

  //
  // Attributes specific to X86 instructions...
  //
  bit hasOpSizePrefix = 0; // Does this inst have a 0x66 prefix?

  bits<4> Prefix = 0;       // Which prefix byte does this inst have?
  FPFormat FPForm;          // What flavor of FP instruction is this?
  bits<3> FPFormBits = 0;
}

class Imp<list<Register> uses, list<Register> defs> {
  list<Register> Uses = uses;
  list<Register> Defs = defs;
}


// Prefix byte classes which are used to indicate to the ad-hoc machine code
// emitter that various prefix bytes are required.
class OpSize { bit hasOpSizePrefix = 1; }
class TB     { bits<4> Prefix = 1; }
class REP    { bits<4> Prefix = 2; }
class D8     { bits<4> Prefix = 3; }
class D9     { bits<4> Prefix = 4; }
class DA     { bits<4> Prefix = 5; }
class DB     { bits<4> Prefix = 6; }
class DC     { bits<4> Prefix = 7; }
class DD     { bits<4> Prefix = 8; }
class DE     { bits<4> Prefix = 9; }
class DF     { bits<4> Prefix = 10; }
class XD     { bits<4> Prefix = 11; }
class XS     { bits<4> Prefix = 12; }


//===----------------------------------------------------------------------===//
// Pattern fragments...
//

// X86 specific condition code. These correspond to CondCode in
// X86ISelLowering.h. They must be kept in synch.
def X86_COND_A   : PatLeaf<(i8 0)>;
def X86_COND_AE  : PatLeaf<(i8 1)>;
def X86_COND_B   : PatLeaf<(i8 2)>;
def X86_COND_BE  : PatLeaf<(i8 3)>;
def X86_COND_E   : PatLeaf<(i8 4)>;
def X86_COND_G   : PatLeaf<(i8 5)>;
def X86_COND_GE  : PatLeaf<(i8 6)>;
def X86_COND_L   : PatLeaf<(i8 7)>;
def X86_COND_LE  : PatLeaf<(i8 8)>;
def X86_COND_NE  : PatLeaf<(i8 9)>;
def X86_COND_NO  : PatLeaf<(i8 10)>;
def X86_COND_NP  : PatLeaf<(i8 11)>;
def X86_COND_NS  : PatLeaf<(i8 12)>;
def X86_COND_O   : PatLeaf<(i8 13)>;
def X86_COND_P   : PatLeaf<(i8 14)>;
def X86_COND_S   : PatLeaf<(i8 15)>;

def i16immSExt8  : PatLeaf<(i16 imm), [{
  // i16immSExt8 predicate - True if the 16-bit immediate fits in a 8-bit
  // sign extended field.
  return (int)N->getValue() == (signed char)N->getValue();
}]>;

def i32immSExt8  : PatLeaf<(i32 imm), [{
  // i32immSExt8 predicate - True if the 32-bit immediate fits in a 8-bit
  // sign extended field.
  return (int)N->getValue() == (signed char)N->getValue();
}]>;

def i16immZExt8  : PatLeaf<(i16 imm), [{
  // i16immZExt8 predicate - True if the 16-bit immediate fits in a 8-bit zero
  // extended field.
  return (unsigned)N->getValue() == (unsigned char)N->getValue();
}]>;

def fp32imm0 : PatLeaf<(f32 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fp64imm0 : PatLeaf<(f64 fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fp64immneg0 : PatLeaf<(f64 fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

def fp64imm1 : PatLeaf<(f64 fpimm), [{
  return N->isExactlyValue(+1.0);
}]>;

def fp64immneg1 : PatLeaf<(f64 fpimm), [{
  return N->isExactlyValue(-1.0);
}]>;

// Helper fragments for loads.
def loadi8  : PatFrag<(ops node:$ptr), (i8  (load node:$ptr))>;
def loadi16 : PatFrag<(ops node:$ptr), (i16 (load node:$ptr))>;
def loadi32 : PatFrag<(ops node:$ptr), (i32 (load node:$ptr))>;
def loadf32 : PatFrag<(ops node:$ptr), (f32 (load node:$ptr))>;
def loadf64 : PatFrag<(ops node:$ptr), (f64 (load node:$ptr))>;

def sextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (sextload node:$ptr, i1))>;
def sextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (sextload node:$ptr, i1))>;
def sextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (sextload node:$ptr, i8))>;
def sextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (sextload node:$ptr, i8))>;
def sextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (sextload node:$ptr, i16))>;

def zextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (zextload node:$ptr, i1))>;
def zextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (zextload node:$ptr, i1))>;
def zextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (zextload node:$ptr, i8))>;
def zextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (zextload node:$ptr, i8))>;
def zextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (zextload node:$ptr, i16))>;

def extloadi8i1    : PatFrag<(ops node:$ptr), (i8  (extload node:$ptr, i1))>;
def extloadf64f32  : PatFrag<(ops node:$ptr), (f64 (extload node:$ptr, f32))>;

//===----------------------------------------------------------------------===//
// Instruction templates...

class I<bits<8> o, Format f, dag ops, string asm, list<dag> pattern>
  : X86Inst<o, f, NoImm, ops, asm> {
  let Pattern = pattern;
}
class Ii8 <bits<8> o, Format f, dag ops, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm8 , ops, asm> {
  let Pattern = pattern;
}
class Ii16<bits<8> o, Format f, dag ops, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm16, ops, asm> {
  let Pattern = pattern;
}
class Ii32<bits<8> o, Format f, dag ops, string asm, list<dag> pattern>
  : X86Inst<o, f, Imm32, ops, asm> {
  let Pattern = pattern;
}

//===----------------------------------------------------------------------===//
// Instruction list...
//

// Pseudo-instructions:
def PHI : I<0, Pseudo, (ops variable_ops), "PHINODE", []>;        // PHI node.

def ADJCALLSTACKDOWN : I<0, Pseudo, (ops i32imm:$amt), "#ADJCALLSTACKDOWN",
                         [(X86callseq_start imm:$amt)]>;
def ADJCALLSTACKUP   : I<0, Pseudo, (ops i32imm:$amt1, i32imm:$amt2),
                         "#ADJCALLSTACKUP",
                         [(X86callseq_end imm:$amt1, imm:$amt2)]>;
def IMPLICIT_USE     : I<0, Pseudo, (ops variable_ops), "#IMPLICIT_USE", []>;
def IMPLICIT_DEF     : I<0, Pseudo, (ops variable_ops), "#IMPLICIT_DEF", []>;
def IMPLICIT_DEF_R8  : I<0, Pseudo, (ops R8:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set R8:$dst, (undef))]>;
def IMPLICIT_DEF_R16  : I<0, Pseudo, (ops R16:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set R16:$dst, (undef))]>;
def IMPLICIT_DEF_R32  : I<0, Pseudo, (ops R32:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set R32:$dst, (undef))]>;
def IMPLICIT_DEF_FR32 : I<0, Pseudo, (ops FR32:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set FR32:$dst, (undef))]>, Requires<[HasSSE2]>;
def IMPLICIT_DEF_FR64 : I<0, Pseudo, (ops FR64:$dst),
                         "#IMPLICIT_DEF $dst",
                         [(set FR64:$dst, (undef))]>, Requires<[HasSSE2]>;


// CMOV* - Used to implement the SSE SELECT DAG operation.  Expanded by the
// scheduler into a branch sequence.
let usesCustomDAGSchedInserter = 1 in {  // Expanded by the scheduler.
  def CMOV_FR32 : I<0, Pseudo,
                    (ops FR32:$dst, FR32:$t, FR32:$f, i8imm:$cond),
                    "#CMOV PSEUDO!",
                    [(set FR32:$dst, (X86cmov FR32:$t, FR32:$f, imm:$cond,
                                      STATUS))]>;
  def CMOV_FR64 : I<0, Pseudo,
                    (ops FR64:$dst, FR64:$t, FR64:$f, i8imm:$cond),
                    "#CMOV PSEUDO!",
                    [(set FR64:$dst, (X86cmov FR64:$t, FR64:$f, imm:$cond,
                                      STATUS))]>;
}

let isTerminator = 1 in
  let Defs = [FP0, FP1, FP2, FP3, FP4, FP5, FP6] in
    def FP_REG_KILL  : I<0, Pseudo, (ops), "#FP_REG_KILL", []>;


// Nop
def NOOP : I<0x90, RawFrm, (ops), "nop", []>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// Return instructions.
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1, noResults = 1 in {
  def RET    : I<0xC3, RawFrm, (ops), "ret", [(X86retflag 0)]>;
  def RETI   : Ii16<0xC2, RawFrm, (ops i16imm:$amt), "ret $amt",
                    [(X86retflag imm:$amt)]>;
}

// All branches are RawFrm, Void, Branch, and Terminators
let isBranch = 1, isTerminator = 1, noResults = 1 in
  class IBr<bits<8> opcode, dag ops, string asm, list<dag> pattern> :
        I<opcode, RawFrm, ops, asm, pattern>;

// Conditional branches
let isBarrier = 1 in
  def JMP : IBr<0xE9, (ops brtarget:$dst), "jmp $dst", [(br bb:$dst)]>;

def JE  : IBr<0x84, (ops brtarget:$dst), "je $dst",
              [(X86brcond bb:$dst, X86_COND_E, STATUS)]>, Imp<[STATUS],[]>, TB;
def JNE : IBr<0x85, (ops brtarget:$dst), "jne $dst",
              [(X86brcond bb:$dst, X86_COND_NE, STATUS)]>, Imp<[STATUS],[]>, TB;
def JL  : IBr<0x8C, (ops brtarget:$dst), "jl $dst",
              [(X86brcond bb:$dst, X86_COND_L, STATUS)]>, Imp<[STATUS],[]>, TB;
def JLE : IBr<0x8E, (ops brtarget:$dst), "jle $dst",
              [(X86brcond bb:$dst, X86_COND_LE, STATUS)]>, Imp<[STATUS],[]>, TB;
def JG  : IBr<0x8F, (ops brtarget:$dst), "jg $dst",
              [(X86brcond bb:$dst, X86_COND_G, STATUS)]>, Imp<[STATUS],[]>, TB;
def JGE : IBr<0x8D, (ops brtarget:$dst), "jge $dst",
              [(X86brcond bb:$dst, X86_COND_GE, STATUS)]>, Imp<[STATUS],[]>, TB;

def JB  : IBr<0x82, (ops brtarget:$dst), "jb $dst",
              [(X86brcond bb:$dst, X86_COND_B, STATUS)]>, Imp<[STATUS],[]>, TB;
def JBE : IBr<0x86, (ops brtarget:$dst), "jbe $dst",
              [(X86brcond bb:$dst, X86_COND_BE, STATUS)]>, Imp<[STATUS],[]>, TB;
def JA  : IBr<0x87, (ops brtarget:$dst), "ja $dst",
              [(X86brcond bb:$dst, X86_COND_A, STATUS)]>, Imp<[STATUS],[]>, TB;
def JAE : IBr<0x83, (ops brtarget:$dst), "jae $dst",
              [(X86brcond bb:$dst, X86_COND_AE, STATUS)]>, Imp<[STATUS],[]>, TB;

def JS  : IBr<0x88, (ops brtarget:$dst), "js $dst",
              [(X86brcond bb:$dst, X86_COND_S, STATUS)]>, Imp<[STATUS],[]>, TB;
def JNS : IBr<0x89, (ops brtarget:$dst), "jns $dst",
              [(X86brcond bb:$dst, X86_COND_NS, STATUS)]>, Imp<[STATUS],[]>, TB;
def JP  : IBr<0x8A, (ops brtarget:$dst), "jp $dst",
              [(X86brcond bb:$dst, X86_COND_P, STATUS)]>, Imp<[STATUS],[]>, TB;
def JNP : IBr<0x8B, (ops brtarget:$dst), "jnp $dst",
              [(X86brcond bb:$dst, X86_COND_NP, STATUS)]>, Imp<[STATUS],[]>, TB;
def JO  : IBr<0x80, (ops brtarget:$dst), "jo $dst",
              [(X86brcond bb:$dst, X86_COND_O, STATUS)]>, Imp<[STATUS],[]>, TB;
def JNO : IBr<0x81, (ops brtarget:$dst), "jno $dst",
              [(X86brcond bb:$dst, X86_COND_NO, STATUS)]>, Imp<[STATUS],[]>, TB;

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1, noResults = 1 in
  // All calls clobber the non-callee saved registers...
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7] in {
    def CALLpcrel32 : I<0xE8, RawFrm, (ops calltarget:$dst), "call $dst",
                      []>;
    def CALL32r     : I<0xFF, MRM2r, (ops R32:$dst), "call {*}$dst",
                      [(X86call R32:$dst)]>;
    def CALL32m     : I<0xFF, MRM2m, (ops i32mem:$dst), "call {*}$dst",
                      [(X86call (loadi32 addr:$dst))]>;
  }

// Tail call stuff.
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, noResults = 1 in
  def TAILJMPd : IBr<0xE9, (ops calltarget:$dst), "jmp $dst  # TAIL CALL", []>;
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, noResults = 1 in
  def TAILJMPr : I<0xFF, MRM4r, (ops R32:$dst), "jmp {*}$dst  # TAIL CALL", []>;
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, noResults = 1 in
  def TAILJMPm : I<0xFF, MRM4m, (ops i32mem:$dst),
                   "jmp {*}$dst  # TAIL CALL", []>;

// ADJSTACKPTRri - This is a standard ADD32ri instruction, identical in every
// way, except that it is marked as being a terminator.  This causes the epilog
// inserter to insert reloads of callee saved registers BEFORE this.  We need
// this until we have a more accurate way of tracking where the stack pointer is
// within a function.
let isTerminator = 1, isTwoAddress = 1 in
  def ADJSTACKPTRri : Ii32<0x81, MRM0r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                           "add{l} {$src2, $dst|$dst, $src2}", []>;

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
def LEAVE    : I<0xC9, RawFrm,
                 (ops), "leave", []>, Imp<[EBP,ESP],[EBP,ESP]>;
def POP32r   : I<0x58, AddRegFrm,
                 (ops R32:$reg), "pop{l} $reg", []>, Imp<[ESP],[ESP]>;

let isTwoAddress = 1 in                               // R32 = bswap R32
  def BSWAP32r : I<0xC8, AddRegFrm,
                   (ops R32:$dst, R32:$src), "bswap{l} $dst", []>, TB;

def XCHG8rr  : I<0x86, MRMDestReg,                    // xchg R8, R8
                 (ops R8:$src1, R8:$src2),
                 "xchg{b} {$src2|$src1}, {$src1|$src2}", []>;
def XCHG16rr : I<0x87, MRMDestReg,                    // xchg R16, R16
                 (ops R16:$src1, R16:$src2),
                 "xchg{w} {$src2|$src1}, {$src1|$src2}", []>, OpSize;
def XCHG32rr : I<0x87, MRMDestReg,                    // xchg R32, R32
                 (ops R32:$src1, R32:$src2),
                 "xchg{l} {$src2|$src1}, {$src1|$src2}", []>;

def XCHG8mr  : I<0x86, MRMDestMem,
                 (ops i8mem:$src1, R8:$src2),
                 "xchg{b} {$src2|$src1}, {$src1|$src2}", []>;
def XCHG16mr : I<0x87, MRMDestMem,
                 (ops i16mem:$src1, R16:$src2),
                 "xchg{w} {$src2|$src1}, {$src1|$src2}", []>, OpSize;
def XCHG32mr : I<0x87, MRMDestMem,
                 (ops i32mem:$src1, R32:$src2),
                 "xchg{l} {$src2|$src1}, {$src1|$src2}", []>;
def XCHG8rm  : I<0x86, MRMSrcMem,
                 (ops R8:$src1, i8mem:$src2),
                 "xchg{b} {$src2|$src1}, {$src1|$src2}", []>;
def XCHG16rm : I<0x87, MRMSrcMem,
                 (ops R16:$src1, i16mem:$src2),
                 "xchg{w} {$src2|$src1}, {$src1|$src2}", []>, OpSize;
def XCHG32rm : I<0x87, MRMSrcMem,
                 (ops R32:$src1, i32mem:$src2),
                 "xchg{l} {$src2|$src1}, {$src1|$src2}", []>;

def LEA16r   : I<0x8D, MRMSrcMem,
                 (ops R16:$dst, i32mem:$src),
                 "lea{w} {$src|$dst}, {$dst|$src}", []>, OpSize;
def LEA32r   : I<0x8D, MRMSrcMem,
                 (ops R32:$dst, i32mem:$src),
                 "lea{l} {$src|$dst}, {$dst|$src}",
                 [(set R32:$dst, leaaddr:$src)]>;

def REP_MOVSB : I<0xA4, RawFrm, (ops), "{rep;movsb|rep movsb}", []>,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP;
def REP_MOVSW : I<0xA5, RawFrm, (ops), "{rep;movsw|rep movsw}", []>,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP, OpSize;
def REP_MOVSD : I<0xA5, RawFrm, (ops), "{rep;movsd|rep movsd}", []>,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP;

def REP_STOSB : I<0xAA, RawFrm, (ops), "{rep;stosb|rep stosb}", []>,
                Imp<[AL,ECX,EDI], [ECX,EDI]>, REP;
def REP_STOSW : I<0xAB, RawFrm, (ops), "{rep;stosw|rep stosw}", []>,
                Imp<[AX,ECX,EDI], [ECX,EDI]>, REP, OpSize;
def REP_STOSD : I<0xAB, RawFrm, (ops), "{rep;stosl|rep stosd}", []>,
                Imp<[EAX,ECX,EDI], [ECX,EDI]>, REP;


//===----------------------------------------------------------------------===//
//  Input/Output Instructions...
//
def IN8rr  : I<0xEC, RawFrm, (ops),
               "in{b} {%dx, %al|%AL, %DX}",
               [(set AL, (readport DX))]>,  Imp<[DX], [AL]>;
def IN16rr : I<0xED, RawFrm, (ops),
               "in{w} {%dx, %ax|%AX, %DX}",
               [(set AX, (readport DX))]>,  Imp<[DX], [AX]>, OpSize;
def IN32rr : I<0xED, RawFrm, (ops),
               "in{l} {%dx, %eax|%EAX, %DX}",
               [(set EAX, (readport DX))]>, Imp<[DX],[EAX]>;

def IN8ri  : Ii8<0xE4, RawFrm, (ops i16i8imm:$port),
                  "in{b} {$port, %al|%AL, $port}",
                 [(set AL, (readport i16immZExt8:$port))]>,
             Imp<[], [AL]>;
def IN16ri : Ii8<0xE5, RawFrm, (ops i16i8imm:$port),
                  "in{w} {$port, %ax|%AX, $port}",
                 [(set AX, (readport i16immZExt8:$port))]>,
             Imp<[], [AX]>, OpSize;
def IN32ri : Ii8<0xE5, RawFrm, (ops i16i8imm:$port),
                  "in{l} {$port, %eax|%EAX, $port}",
                 [(set EAX, (readport i16immZExt8:$port))]>,
             Imp<[],[EAX]>;

def OUT8rr  : I<0xEE, RawFrm, (ops),
                "out{b} {%al, %dx|%DX, %AL}",
                [(writeport AL, DX)]>,  Imp<[DX,  AL], []>;
def OUT16rr : I<0xEF, RawFrm, (ops),
                "out{w} {%ax, %dx|%DX, %AX}",
                [(writeport AX, DX)]>,  Imp<[DX,  AX], []>, OpSize;
def OUT32rr : I<0xEF, RawFrm, (ops),
                "out{l} {%eax, %dx|%DX, %EAX}",
                [(writeport EAX, DX)]>, Imp<[DX, EAX], []>;

def OUT8ir  : Ii8<0xE6, RawFrm, (ops i16i8imm:$port),
                   "out{b} {%al, $port|$port, %AL}",
                   [(writeport AL, i16immZExt8:$port)]>,
              Imp<[AL], []>;
def OUT16ir : Ii8<0xE7, RawFrm, (ops i16i8imm:$port),
                   "out{w} {%ax, $port|$port, %AX}",
                   [(writeport AX, i16immZExt8:$port)]>,
              Imp<[AX], []>, OpSize;
def OUT32ir : Ii8<0xE7, RawFrm, (ops i16i8imm:$port),
                   "out{l} {%eax, $port|$port, %EAX}",
                   [(writeport EAX, i16immZExt8:$port)]>,
              Imp<[EAX], []>;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//
def MOV8rr  : I<0x88, MRMDestReg, (ops R8 :$dst, R8 :$src),
                "mov{b} {$src, $dst|$dst, $src}", []>;
def MOV16rr : I<0x89, MRMDestReg, (ops R16:$dst, R16:$src),
                "mov{w} {$src, $dst|$dst, $src}", []>, OpSize;
def MOV32rr : I<0x89, MRMDestReg, (ops R32:$dst, R32:$src),
                "mov{l} {$src, $dst|$dst, $src}", []>;
def MOV8ri  : Ii8 <0xB0, AddRegFrm, (ops R8 :$dst, i8imm :$src),
                   "mov{b} {$src, $dst|$dst, $src}",
                   [(set R8:$dst, imm:$src)]>;
def MOV16ri : Ii16<0xB8, AddRegFrm, (ops R16:$dst, i16imm:$src),
                   "mov{w} {$src, $dst|$dst, $src}",
                   [(set R16:$dst, imm:$src)]>, OpSize;
def MOV32ri : Ii32<0xB8, AddRegFrm, (ops R32:$dst, i32imm:$src),
                   "mov{l} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, imm:$src)]>;
def MOV8mi  : Ii8 <0xC6, MRM0m, (ops i8mem :$dst, i8imm :$src),
                   "mov{b} {$src, $dst|$dst, $src}",
                   [(store (i8 imm:$src), addr:$dst)]>;
def MOV16mi : Ii16<0xC7, MRM0m, (ops i16mem:$dst, i16imm:$src),
                   "mov{w} {$src, $dst|$dst, $src}",
                   [(store (i16 imm:$src), addr:$dst)]>, OpSize;
def MOV32mi : Ii32<0xC7, MRM0m, (ops i32mem:$dst, i32imm:$src),
                   "mov{l} {$src, $dst|$dst, $src}",
                   [(store (i32 imm:$src), addr:$dst)]>;

def MOV8rm  : I<0x8A, MRMSrcMem, (ops R8 :$dst, i8mem :$src),
                "mov{b} {$src, $dst|$dst, $src}",
                [(set R8:$dst, (load addr:$src))]>;
def MOV16rm : I<0x8B, MRMSrcMem, (ops R16:$dst, i16mem:$src),
                "mov{w} {$src, $dst|$dst, $src}",
                [(set R16:$dst, (load addr:$src))]>, OpSize;
def MOV32rm : I<0x8B, MRMSrcMem, (ops R32:$dst, i32mem:$src),
                "mov{l} {$src, $dst|$dst, $src}",
                [(set R32:$dst, (load addr:$src))]>;

def MOV8mr  : I<0x88, MRMDestMem, (ops i8mem :$dst, R8 :$src),
                "mov{b} {$src, $dst|$dst, $src}",
                [(store R8:$src, addr:$dst)]>;
def MOV16mr : I<0x89, MRMDestMem, (ops i16mem:$dst, R16:$src),
                "mov{w} {$src, $dst|$dst, $src}",
                [(store R16:$src, addr:$dst)]>, OpSize;
def MOV32mr : I<0x89, MRMDestMem, (ops i32mem:$dst, R32:$src),
                "mov{l} {$src, $dst|$dst, $src}",
                [(store R32:$src, addr:$dst)]>;
                
//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication
def MUL8r  : I<0xF6, MRM4r, (ops R8:$src), "mul{b} $src", []>,
             Imp<[AL],[AX]>;               // AL,AH = AL*R8
def MUL16r : I<0xF7, MRM4r, (ops R16:$src), "mul{w} $src", []>,
             Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def MUL32r : I<0xF7, MRM4r, (ops R32:$src), "mul{l} $src", []>,
             Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def MUL8m  : I<0xF6, MRM4m, (ops i8mem :$src),
               "mul{b} $src", []>, Imp<[AL],[AX]>;          // AL,AH = AL*[mem8]
def MUL16m : I<0xF7, MRM4m, (ops i16mem:$src),
               "mul{w} $src", []>, Imp<[AX],[AX,DX]>,
               OpSize; // AX,DX = AX*[mem16]
def MUL32m : I<0xF7, MRM4m, (ops i32mem:$src),
               "mul{l} $src", []>, Imp<[EAX],[EAX,EDX]>;// EAX,EDX = EAX*[mem32]

def IMUL8r  : I<0xF6, MRM5r, (ops R8:$src), "imul{b} $src", []>,
              Imp<[AL],[AX]>;               // AL,AH = AL*R8
def IMUL16r : I<0xF7, MRM5r, (ops R16:$src), "imul{w} $src", []>,
              Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def IMUL32r : I<0xF7, MRM5r, (ops R32:$src), "imul{l} $src", []>,
              Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def IMUL8m  : I<0xF6, MRM5m, (ops i8mem :$src),
                "imul{b} $src", []>, Imp<[AL],[AX]>;        // AL,AH = AL*[mem8]
def IMUL16m : I<0xF7, MRM5m, (ops i16mem:$src),
                "imul{w} $src", []>, Imp<[AX],[AX,DX]>,
                OpSize; // AX,DX = AX*[mem16]
def IMUL32m : I<0xF7, MRM5m, (ops i32mem:$src),
                "imul{l} $src", []>,
                Imp<[EAX],[EAX,EDX]>;  // EAX,EDX = EAX*[mem32]

// unsigned division/remainder
def DIV8r  : I<0xF6, MRM6r, (ops R8:$src),          // AX/r8 = AL,AH
               "div{b} $src", []>, Imp<[AX],[AX]>;
def DIV16r : I<0xF7, MRM6r, (ops R16:$src),         // DX:AX/r16 = AX,DX
               "div{w} $src", []>, Imp<[AX,DX],[AX,DX]>, OpSize;
def DIV32r : I<0xF7, MRM6r, (ops R32:$src),         // EDX:EAX/r32 = EAX,EDX
               "div{l} $src", []>, Imp<[EAX,EDX],[EAX,EDX]>;
def DIV8m  : I<0xF6, MRM6m, (ops i8mem:$src),       // AX/[mem8] = AL,AH
               "div{b} $src", []>, Imp<[AX],[AX]>;
def DIV16m : I<0xF7, MRM6m, (ops i16mem:$src),      // DX:AX/[mem16] = AX,DX
               "div{w} $src", []>, Imp<[AX,DX],[AX,DX]>, OpSize;
def DIV32m : I<0xF7, MRM6m, (ops i32mem:$src),      // EDX:EAX/[mem32] = EAX,EDX
               "div{l} $src", []>, Imp<[EAX,EDX],[EAX,EDX]>;

// Signed division/remainder.
def IDIV8r : I<0xF6, MRM7r, (ops R8:$src),          // AX/r8 = AL,AH
               "idiv{b} $src", []>, Imp<[AX],[AX]>;
def IDIV16r: I<0xF7, MRM7r, (ops R16:$src),         // DX:AX/r16 = AX,DX
               "idiv{w} $src", []>, Imp<[AX,DX],[AX,DX]>, OpSize;
def IDIV32r: I<0xF7, MRM7r, (ops R32:$src),         // EDX:EAX/r32 = EAX,EDX
               "idiv{l} $src", []>, Imp<[EAX,EDX],[EAX,EDX]>;
def IDIV8m : I<0xF6, MRM7m, (ops i8mem:$src),      // AX/[mem8] = AL,AH
               "idiv{b} $src", []>, Imp<[AX],[AX]>;
def IDIV16m: I<0xF7, MRM7m, (ops i16mem:$src),     // DX:AX/[mem16] = AX,DX
               "idiv{w} $src", []>, Imp<[AX,DX],[AX,DX]>, OpSize;
def IDIV32m: I<0xF7, MRM7m, (ops i32mem:$src),     // EDX:EAX/[mem32] = EAX,EDX
               "idiv{l} $src", []>, Imp<[EAX,EDX],[EAX,EDX]>;

// Sign-extenders for division.
def CBW : I<0x98, RawFrm, (ops),
            "{cbtw|cbw}", []>, Imp<[AL],[AH]>;   // AX = signext(AL)
def CWD : I<0x99, RawFrm, (ops),
            "{cwtd|cwd}", []>, Imp<[AX],[DX]>;   // DX:AX = signext(AX)
def CDQ : I<0x99, RawFrm, (ops),
            "{cltd|cdq}", []>, Imp<[EAX],[EDX]>; // EDX:EAX = signext(EAX)
          

//===----------------------------------------------------------------------===//
//  Two address Instructions...
//
let isTwoAddress = 1 in {

// Conditional moves
def CMOVB16rr : I<0x42, MRMSrcReg,       // if <u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_B, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVB16rm : I<0x42, MRMSrcMem,       // if <u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_B, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVB32rr : I<0x42, MRMSrcReg,       // if <u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_B, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVB32rm : I<0x42, MRMSrcMem,       // if <u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_B, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVAE16rr: I<0x43, MRMSrcReg,       // if >=u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_AE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVAE16rm: I<0x43, MRMSrcMem,       // if >=u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_AE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVAE32rr: I<0x43, MRMSrcReg,       // if >=u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_AE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVAE32rm: I<0x43, MRMSrcMem,       // if >=u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_AE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVE16rr : I<0x44, MRMSrcReg,       // if ==, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_E, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVE16rm : I<0x44, MRMSrcMem,       // if ==, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_E, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVE32rr : I<0x44, MRMSrcReg,       // if ==, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_E, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVE32rm : I<0x44, MRMSrcMem,       // if ==, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_E, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVNE16rr: I<0x45, MRMSrcReg,       // if !=, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_NE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVNE16rm: I<0x45, MRMSrcMem,       // if !=, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVNE32rr: I<0x45, MRMSrcReg,       // if !=, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_NE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVNE32rm: I<0x45, MRMSrcMem,       // if !=, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVBE16rr: I<0x46, MRMSrcReg,       // if <=u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_BE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVBE16rm: I<0x46, MRMSrcMem,       // if <=u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_BE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVBE32rr: I<0x46, MRMSrcReg,       // if <=u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_BE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVBE32rm: I<0x46, MRMSrcMem,       // if <=u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_BE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVA16rr : I<0x47, MRMSrcReg,       // if >u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_A, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVA16rm : I<0x47, MRMSrcMem,       // if >u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_A, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVA32rr : I<0x47, MRMSrcReg,       // if >u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_A, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVA32rm : I<0x47, MRMSrcMem,       // if >u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_A, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVL16rr : I<0x4C, MRMSrcReg,       // if <s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_L, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVL16rm : I<0x4C, MRMSrcMem,       // if <s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_L, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVL32rr : I<0x4C, MRMSrcReg,       // if <s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_L, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVL32rm : I<0x4C, MRMSrcMem,       // if <s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_L, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVGE16rr: I<0x4D, MRMSrcReg,       // if >=s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_GE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVGE16rm: I<0x4D, MRMSrcMem,       // if >=s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_GE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVGE32rr: I<0x4D, MRMSrcReg,       // if >=s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_GE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVGE32rm: I<0x4D, MRMSrcMem,       // if >=s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_GE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVLE16rr: I<0x4E, MRMSrcReg,       // if <=s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_LE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVLE16rm: I<0x4E, MRMSrcMem,       // if <=s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_LE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVLE32rr: I<0x4E, MRMSrcReg,       // if <=s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_LE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVLE32rm: I<0x4E, MRMSrcMem,       // if <=s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_LE, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVG16rr : I<0x4F, MRMSrcReg,       // if >s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_G, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVG16rm : I<0x4F, MRMSrcMem,       // if >s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_G, STATUS))]>,
                  Imp<[STATUS],[]>,  TB, OpSize;
def CMOVG32rr : I<0x4F, MRMSrcReg,       // if >s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_G, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;
def CMOVG32rm : I<0x4F, MRMSrcMem,       // if >s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_G, STATUS))]>,
                  Imp<[STATUS],[]>,  TB;

def CMOVS16rr : I<0x48, MRMSrcReg,       // if signed, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_S, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVS16rm : I<0x48, MRMSrcMem,       // if signed, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_S, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVS32rr : I<0x48, MRMSrcReg,       // if signed, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_S, STATUS))]>,
                  Imp<[STATUS],[]>, TB;
def CMOVS32rm : I<0x48, MRMSrcMem,       // if signed, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_S, STATUS))]>,
                  Imp<[STATUS],[]>, TB;

def CMOVNS16rr: I<0x49, MRMSrcReg,       // if !signed, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_NS, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVNS16rm: I<0x49, MRMSrcMem,       // if !signed, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NS, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVNS32rr: I<0x49, MRMSrcReg,       // if !signed, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_NS, STATUS))]>,
                  Imp<[STATUS],[]>, TB;
def CMOVNS32rm: I<0x49, MRMSrcMem,       // if !signed, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NS, STATUS))]>,
                  Imp<[STATUS],[]>, TB;

def CMOVP16rr : I<0x4A, MRMSrcReg,       // if parity, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_P, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVP16rm : I<0x4A, MRMSrcMem,       // if parity, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_P, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVP32rr : I<0x4A, MRMSrcReg,       // if parity, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_P, STATUS))]>,
                  Imp<[STATUS],[]>, TB;
def CMOVP32rm : I<0x4A, MRMSrcMem,       // if parity, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_P, STATUS))]>,
                  Imp<[STATUS],[]>, TB;

def CMOVNP16rr : I<0x4B, MRMSrcReg,       // if !parity, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                    X86_COND_NP, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVNP16rm : I<0x4B, MRMSrcMem,       // if !parity, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NP, STATUS))]>,
                  Imp<[STATUS],[]>, TB, OpSize;
def CMOVNP32rr : I<0x4B, MRMSrcReg,       // if !parity, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                    X86_COND_NP, STATUS))]>,
                  Imp<[STATUS],[]>, TB;
def CMOVNP32rm : I<0x4B, MRMSrcMem,       // if !parity, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NP, STATUS))]>,
                  Imp<[STATUS],[]>, TB;


// unary instructions
def NEG8r  : I<0xF6, MRM3r, (ops R8 :$dst, R8 :$src), "neg{b} $dst",
               [(set R8:$dst, (ineg R8:$src))]>;
def NEG16r : I<0xF7, MRM3r, (ops R16:$dst, R16:$src), "neg{w} $dst",
               [(set R16:$dst, (ineg R16:$src))]>, OpSize;
def NEG32r : I<0xF7, MRM3r, (ops R32:$dst, R32:$src), "neg{l} $dst",
               [(set R32:$dst, (ineg R32:$src))]>;
let isTwoAddress = 0 in {
  def NEG8m  : I<0xF6, MRM3m, (ops i8mem :$dst), "neg{b} $dst",
                 [(store (ineg (loadi8 addr:$dst)), addr:$dst)]>;
  def NEG16m : I<0xF7, MRM3m, (ops i16mem:$dst), "neg{w} $dst",
                 [(store (ineg (loadi16 addr:$dst)), addr:$dst)]>, OpSize;
  def NEG32m : I<0xF7, MRM3m, (ops i32mem:$dst), "neg{l} $dst",
                 [(store (ineg (loadi32 addr:$dst)), addr:$dst)]>;

}

def NOT8r  : I<0xF6, MRM2r, (ops R8 :$dst, R8 :$src), "not{b} $dst",
               [(set R8:$dst, (not R8:$src))]>;
def NOT16r : I<0xF7, MRM2r, (ops R16:$dst, R16:$src), "not{w} $dst",
               [(set R16:$dst, (not R16:$src))]>, OpSize;
def NOT32r : I<0xF7, MRM2r, (ops R32:$dst, R32:$src), "not{l} $dst",
               [(set R32:$dst, (not R32:$src))]>;
let isTwoAddress = 0 in {
  def NOT8m  : I<0xF6, MRM2m, (ops i8mem :$dst), "not{b} $dst",
                 [(store (not (loadi8 addr:$dst)), addr:$dst)]>;
  def NOT16m : I<0xF7, MRM2m, (ops i16mem:$dst), "not{w} $dst",
                 [(store (not (loadi16 addr:$dst)), addr:$dst)]>, OpSize;
  def NOT32m : I<0xF7, MRM2m, (ops i32mem:$dst), "not{l} $dst",
                 [(store (not (loadi32 addr:$dst)), addr:$dst)]>;
}

// TODO: inc/dec is slow for P4, but fast for Pentium-M.
def INC8r  : I<0xFE, MRM0r, (ops R8 :$dst, R8 :$src), "inc{b} $dst",
               [(set R8:$dst, (add R8:$src, 1))]>;
let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def INC16r : I<0xFF, MRM0r, (ops R16:$dst, R16:$src), "inc{w} $dst",
               [(set R16:$dst, (add R16:$src, 1))]>, OpSize;
def INC32r : I<0xFF, MRM0r, (ops R32:$dst, R32:$src), "inc{l} $dst",
               [(set R32:$dst, (add R32:$src, 1))]>;
}
let isTwoAddress = 0 in {
  def INC8m  : I<0xFE, MRM0m, (ops i8mem :$dst), "inc{b} $dst",
               [(store (add (loadi8 addr:$dst), 1), addr:$dst)]>;
  def INC16m : I<0xFF, MRM0m, (ops i16mem:$dst), "inc{w} $dst",
               [(store (add (loadi16 addr:$dst), 1), addr:$dst)]>, OpSize;
  def INC32m : I<0xFF, MRM0m, (ops i32mem:$dst), "inc{l} $dst",
               [(store (add (loadi32 addr:$dst), 1), addr:$dst)]>;
}

def DEC8r  : I<0xFE, MRM1r, (ops R8 :$dst, R8 :$src), "dec{b} $dst",
               [(set R8:$dst, (add R8:$src, -1))]>;
let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def DEC16r : I<0xFF, MRM1r, (ops R16:$dst, R16:$src), "dec{w} $dst",
               [(set R16:$dst, (add R16:$src, -1))]>, OpSize;
def DEC32r : I<0xFF, MRM1r, (ops R32:$dst, R32:$src), "dec{l} $dst",
               [(set R32:$dst, (add R32:$src, -1))]>;
}

let isTwoAddress = 0 in {
  def DEC8m  : I<0xFE, MRM1m, (ops i8mem :$dst), "dec{b} $dst",
               [(store (add (loadi8 addr:$dst), -1), addr:$dst)]>;
  def DEC16m : I<0xFF, MRM1m, (ops i16mem:$dst), "dec{w} $dst",
               [(store (add (loadi16 addr:$dst), -1), addr:$dst)]>, OpSize;
  def DEC32m : I<0xFF, MRM1m, (ops i32mem:$dst), "dec{l} $dst",
               [(store (add (loadi32 addr:$dst), -1), addr:$dst)]>;
}

// Logical operators...
let isCommutable = 1 in {   // X = AND Y, Z   --> X = AND Z, Y
def AND8rr   : I<0x20, MRMDestReg,
                (ops R8 :$dst, R8 :$src1, R8 :$src2),
                "and{b} {$src2, $dst|$dst, $src2}",
                [(set R8:$dst, (and R8:$src1, R8:$src2))]>;
def AND16rr  : I<0x21, MRMDestReg,
                 (ops R16:$dst, R16:$src1, R16:$src2),
                 "and{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (and R16:$src1, R16:$src2))]>, OpSize;
def AND32rr  : I<0x21, MRMDestReg, 
                 (ops R32:$dst, R32:$src1, R32:$src2),
                 "and{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (and R32:$src1, R32:$src2))]>;
}

def AND8rm   : I<0x22, MRMSrcMem, 
                 (ops R8 :$dst, R8 :$src1, i8mem :$src2),
                 "and{b} {$src2, $dst|$dst, $src2}",
                [(set R8:$dst, (and R8:$src1, (load addr:$src2)))]>;
def AND16rm  : I<0x23, MRMSrcMem, 
                 (ops R16:$dst, R16:$src1, i16mem:$src2),
                 "and{w} {$src2, $dst|$dst, $src2}",
                [(set R16:$dst, (and R16:$src1, (load addr:$src2)))]>, OpSize;
def AND32rm  : I<0x23, MRMSrcMem,
                 (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "and{l} {$src2, $dst|$dst, $src2}",
                [(set R32:$dst, (and R32:$src1, (load addr:$src2)))]>;

def AND8ri   : Ii8<0x80, MRM4r, 
                   (ops R8 :$dst, R8 :$src1, i8imm :$src2),
                   "and{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (and R8:$src1, imm:$src2))]>;
def AND16ri  : Ii16<0x81, MRM4r, 
                    (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "and{w} {$src2, $dst|$dst, $src2}",
                    [(set R16:$dst, (and R16:$src1, imm:$src2))]>, OpSize;
def AND32ri  : Ii32<0x81, MRM4r, 
                    (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "and{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (and R32:$src1, imm:$src2))]>;
def AND16ri8 : Ii8<0x83, MRM4r, 
                   (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                   "and{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (and R16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def AND32ri8 : Ii8<0x83, MRM4r, 
                   (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "and{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (and R32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def AND8mr   : I<0x20, MRMDestMem,
                   (ops i8mem :$dst, R8 :$src),
                   "and{b} {$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), R8:$src), addr:$dst)]>;
  def AND16mr  : I<0x21, MRMDestMem,
                   (ops i16mem:$dst, R16:$src),
                   "and{w} {$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), R16:$src), addr:$dst)]>,
                   OpSize;
  def AND32mr  : I<0x21, MRMDestMem,
                   (ops i32mem:$dst, R32:$src),
                   "and{l} {$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), R32:$src), addr:$dst)]>;
  def AND8mi   : Ii8<0x80, MRM4m,
                     (ops i8mem :$dst, i8imm :$src),
                     "and{b} {$src, $dst|$dst, $src}",
                      [(store (and (loadi8 addr:$dst), imm:$src), addr:$dst)]>;
  def AND16mi  : Ii16<0x81, MRM4m,
                      (ops i16mem:$dst, i16imm:$src),
                      "and{w} {$src, $dst|$dst, $src}",
                      [(store (and (loadi16 addr:$dst), imm:$src), addr:$dst)]>,
                      OpSize;
  def AND32mi  : Ii32<0x81, MRM4m,
                      (ops i32mem:$dst, i32imm:$src),
                      "and{l} {$src, $dst|$dst, $src}",
                      [(store (and (loadi32 addr:$dst), imm:$src), addr:$dst)]>;
  def AND16mi8 : Ii8<0x83, MRM4m,
                     (ops i16mem:$dst, i16i8imm :$src),
                     "and{w} {$src, $dst|$dst, $src}",
                [(store (and (load addr:$dst), i16immSExt8:$src), addr:$dst)]>,
                     OpSize;
  def AND32mi8 : Ii8<0x83, MRM4m,
                     (ops i32mem:$dst, i32i8imm :$src),
                     "and{l} {$src, $dst|$dst, $src}",
                [(store (add (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
}


let isCommutable = 1 in {   // X = OR Y, Z   --> X = OR Z, Y
def OR8rr    : I<0x08, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2),
                 "or{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (or R8:$src1, R8:$src2))]>;
def OR16rr   : I<0x09, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2),
                 "or{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (or R16:$src1, R16:$src2))]>, OpSize;
def OR32rr   : I<0x09, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "or{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (or R32:$src1, R32:$src2))]>;
}
def OR8rm    : I<0x0A, MRMSrcMem , (ops R8 :$dst, R8 :$src1, i8mem :$src2),
                 "or{b} {$src2, $dst|$dst, $src2}",
                [(set R8:$dst, (or R8:$src1, (load addr:$src2)))]>;
def OR16rm   : I<0x0B, MRMSrcMem , (ops R16:$dst, R16:$src1, i16mem:$src2),
                 "or{w} {$src2, $dst|$dst, $src2}",
                [(set R16:$dst, (or R16:$src1, (load addr:$src2)))]>, OpSize;
def OR32rm   : I<0x0B, MRMSrcMem , (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "or{l} {$src2, $dst|$dst, $src2}",
                [(set R32:$dst, (or R32:$src1, (load addr:$src2)))]>;

def OR8ri    : Ii8 <0x80, MRM1r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                    "or{b} {$src2, $dst|$dst, $src2}",
                    [(set R8:$dst, (or R8:$src1, imm:$src2))]>;
def OR16ri   : Ii16<0x81, MRM1r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "or{w} {$src2, $dst|$dst, $src2}", 
                    [(set R16:$dst, (or R16:$src1, imm:$src2))]>, OpSize;
def OR32ri   : Ii32<0x81, MRM1r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "or{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (or R32:$src1, imm:$src2))]>;

def OR16ri8  : Ii8<0x83, MRM1r, (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                   "or{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (or R16:$src1, i16immSExt8:$src2))]>, OpSize;
def OR32ri8  : Ii8<0x83, MRM1r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "or{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (or R32:$src1, i32immSExt8:$src2))]>;
let isTwoAddress = 0 in {
  def OR8mr  : I<0x08, MRMDestMem, (ops i8mem:$dst, R8:$src),
                 "or{b} {$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), R8:$src), addr:$dst)]>;
  def OR16mr : I<0x09, MRMDestMem, (ops i16mem:$dst, R16:$src),
                 "or{w} {$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), R16:$src), addr:$dst)]>, OpSize;
  def OR32mr : I<0x09, MRMDestMem, (ops i32mem:$dst, R32:$src),
                 "or{l} {$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), R32:$src), addr:$dst)]>;
  def OR8mi    : Ii8<0x80, MRM1m, (ops i8mem :$dst, i8imm:$src),
                 "or{b} {$src, $dst|$dst, $src}",
                 [(store (or (loadi8 addr:$dst), imm:$src), addr:$dst)]>;
  def OR16mi   : Ii16<0x81, MRM1m, (ops i16mem:$dst, i16imm:$src),
                 "or{w} {$src, $dst|$dst, $src}",
                 [(store (or (loadi16 addr:$dst), imm:$src), addr:$dst)]>,
                 OpSize;
  def OR32mi   : Ii32<0x81, MRM1m, (ops i32mem:$dst, i32imm:$src),
                 "or{l} {$src, $dst|$dst, $src}",
                 [(store (or (loadi32 addr:$dst), imm:$src), addr:$dst)]>;
  def OR16mi8  : Ii8<0x83, MRM1m, (ops i16mem:$dst, i16i8imm:$src),
                 "or{w} {$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), i16immSExt8:$src), addr:$dst)]>,
                     OpSize;
  def OR32mi8  : Ii8<0x83, MRM1m, (ops i32mem:$dst, i32i8imm:$src),
                 "or{l} {$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
}


let isCommutable = 1 in {   // X = XOR Y, Z   --> X = XOR Z, Y
def XOR8rr   : I<0x30, MRMDestReg,
                 (ops R8 :$dst, R8 :$src1, R8 :$src2),
                 "xor{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (xor R8:$src1, R8:$src2))]>;
def XOR16rr  : I<0x31, MRMDestReg, 
                 (ops R16:$dst, R16:$src1, R16:$src2), 
                 "xor{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (xor R16:$src1, R16:$src2))]>, OpSize;
def XOR32rr  : I<0x31, MRMDestReg, 
                 (ops R32:$dst, R32:$src1, R32:$src2), 
                 "xor{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (xor R32:$src1, R32:$src2))]>;
}

def XOR8rm   : I<0x32, MRMSrcMem , 
                 (ops R8 :$dst, R8:$src1, i8mem :$src2), 
                 "xor{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (xor R8:$src1, (load addr:$src2)))]>;
def XOR16rm  : I<0x33, MRMSrcMem , 
                 (ops R16:$dst, R16:$src1, i16mem:$src2), 
                 "xor{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (xor R16:$src1, (load addr:$src2)))]>, OpSize;
def XOR32rm  : I<0x33, MRMSrcMem , 
                 (ops R32:$dst, R32:$src1, i32mem:$src2), 
                 "xor{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (xor R32:$src1, (load addr:$src2)))]>;

def XOR8ri   : Ii8<0x80, MRM6r, 
                   (ops R8:$dst, R8:$src1, i8imm:$src2), 
                   "xor{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (xor R8:$src1, imm:$src2))]>;
def XOR16ri  : Ii16<0x81, MRM6r, 
                    (ops R16:$dst, R16:$src1, i16imm:$src2), 
                    "xor{w} {$src2, $dst|$dst, $src2}",
                    [(set R16:$dst, (xor R16:$src1, imm:$src2))]>, OpSize;
def XOR32ri  : Ii32<0x81, MRM6r, 
                    (ops R32:$dst, R32:$src1, i32imm:$src2), 
                    "xor{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (xor R32:$src1, imm:$src2))]>;
def XOR16ri8 : Ii8<0x83, MRM6r, 
                   (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                   "xor{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (xor R16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def XOR32ri8 : Ii8<0x83, MRM6r, 
                   (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "xor{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (xor R32:$src1, i32immSExt8:$src2))]>;
let isTwoAddress = 0 in {
  def XOR8mr   : I<0x30, MRMDestMem,
                   (ops i8mem :$dst, R8 :$src),
                   "xor{b} {$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), R8:$src), addr:$dst)]>;
  def XOR16mr  : I<0x31, MRMDestMem,
                   (ops i16mem:$dst, R16:$src),
                   "xor{w} {$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), R16:$src), addr:$dst)]>,
                   OpSize;
  def XOR32mr  : I<0x31, MRMDestMem,
                   (ops i32mem:$dst, R32:$src),
                   "xor{l} {$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), R32:$src), addr:$dst)]>;
  def XOR8mi   : Ii8<0x80, MRM6m,
                     (ops i8mem :$dst, i8imm :$src),
                     "xor{b} {$src, $dst|$dst, $src}",
                    [(store (xor (loadi8 addr:$dst), imm:$src), addr:$dst)]>;
  def XOR16mi  : Ii16<0x81, MRM6m,
                      (ops i16mem:$dst, i16imm:$src),
                      "xor{w} {$src, $dst|$dst, $src}",
                   [(store (xor (loadi16 addr:$dst), imm:$src), addr:$dst)]>,
                      OpSize;
  def XOR32mi  : Ii32<0x81, MRM6m,
                      (ops i32mem:$dst, i32imm:$src),
                      "xor{l} {$src, $dst|$dst, $src}",
                   [(store (xor (loadi32 addr:$dst), imm:$src), addr:$dst)]>;
  def XOR16mi8 : Ii8<0x83, MRM6m,
                     (ops i16mem:$dst, i16i8imm :$src),
                     "xor{w} {$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i16immSExt8:$src), addr:$dst)]>,
                     OpSize;
  def XOR32mi8 : Ii8<0x83, MRM6m,
                     (ops i32mem:$dst, i32i8imm :$src),
                     "xor{l} {$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
}

// Shift instructions
def SHL8rCL  : I<0xD2, MRM4r, (ops R8 :$dst, R8 :$src),
                 "shl{b} {%cl, $dst|$dst, %CL}",
                 [(set R8:$dst, (shl R8:$src, CL))]>, Imp<[CL],[]>;
def SHL16rCL : I<0xD3, MRM4r, (ops R16:$dst, R16:$src),
                 "shl{w} {%cl, $dst|$dst, %CL}",
                 [(set R16:$dst, (shl R16:$src, CL))]>, Imp<[CL],[]>, OpSize;
def SHL32rCL : I<0xD3, MRM4r, (ops R32:$dst, R32:$src),
                 "shl{l} {%cl, $dst|$dst, %CL}",
                 [(set R32:$dst, (shl R32:$src, CL))]>, Imp<[CL],[]>;

def SHL8ri   : Ii8<0xC0, MRM4r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "shl{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (shl R8:$src1, (i8 imm:$src2)))]>;
let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def SHL16ri  : Ii8<0xC1, MRM4r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "shl{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (shl R16:$src1, (i8 imm:$src2)))]>, OpSize;
def SHL32ri  : Ii8<0xC1, MRM4r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "shl{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (shl R32:$src1, (i8 imm:$src2)))]>;
}

let isTwoAddress = 0 in {
  def SHL8mCL  : I<0xD2, MRM4m, (ops i8mem :$dst),
                   "shl{b} {%cl, $dst|$dst, %CL}",
                   [(store (shl (loadi8 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def SHL16mCL : I<0xD3, MRM4m, (ops i16mem:$dst),
                   "shl{w} {%cl, $dst|$dst, %CL}",
                   [(store (shl (loadi16 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>, OpSize;
  def SHL32mCL : I<0xD3, MRM4m, (ops i32mem:$dst),
                   "shl{l} {%cl, $dst|$dst, %CL}",
                   [(store (shl (loadi32 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def SHL8mi   : Ii8<0xC0, MRM4m, (ops i8mem :$dst, i8imm:$src),
                     "shl{b} {$src, $dst|$dst, $src}",
                  [(store (shl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SHL16mi  : Ii8<0xC1, MRM4m, (ops i16mem:$dst, i8imm:$src),
                     "shl{w} {$src, $dst|$dst, $src}",
                 [(store (shl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SHL32mi  : Ii8<0xC1, MRM4m, (ops i32mem:$dst, i8imm:$src),
                     "shl{l} {$src, $dst|$dst, $src}",
                 [(store (shl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
}

def SHR8rCL  : I<0xD2, MRM5r, (ops R8 :$dst, R8 :$src),
                 "shr{b} {%cl, $dst|$dst, %CL}",
                 [(set R8:$dst, (srl R8:$src, CL))]>, Imp<[CL],[]>;
def SHR16rCL : I<0xD3, MRM5r, (ops R16:$dst, R16:$src),
                 "shr{w} {%cl, $dst|$dst, %CL}",
                 [(set R16:$dst, (srl R16:$src, CL))]>, Imp<[CL],[]>, OpSize;
def SHR32rCL : I<0xD3, MRM5r, (ops R32:$dst, R32:$src),
                 "shr{l} {%cl, $dst|$dst, %CL}",
                 [(set R32:$dst, (srl R32:$src, CL))]>, Imp<[CL],[]>;

def SHR8ri   : Ii8<0xC0, MRM5r, (ops R8:$dst, R8:$src1, i8imm:$src2),
                   "shr{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (srl R8:$src1, (i8 imm:$src2)))]>;
def SHR16ri  : Ii8<0xC1, MRM5r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "shr{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (srl R16:$src1, (i8 imm:$src2)))]>, OpSize;
def SHR32ri  : Ii8<0xC1, MRM5r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "shr{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (srl R32:$src1, (i8 imm:$src2)))]>;

let isTwoAddress = 0 in {
  def SHR8mCL  : I<0xD2, MRM5m, (ops i8mem :$dst),
                   "shr{b} {%cl, $dst|$dst, %CL}",
                   [(store (srl (loadi8 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def SHR16mCL : I<0xD3, MRM5m, (ops i16mem:$dst),
                   "shr{w} {%cl, $dst|$dst, %CL}",
                   [(store (srl (loadi16 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>, OpSize;
  def SHR32mCL : I<0xD3, MRM5m, (ops i32mem:$dst),
                   "shr{l} {%cl, $dst|$dst, %CL}",
                   [(store (srl (loadi32 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def SHR8mi   : Ii8<0xC0, MRM5m, (ops i8mem :$dst, i8imm:$src),
                     "shr{b} {$src, $dst|$dst, $src}",
                  [(store (srl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SHR16mi  : Ii8<0xC1, MRM5m, (ops i16mem:$dst, i8imm:$src),
                     "shr{w} {$src, $dst|$dst, $src}",
                 [(store (srl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SHR32mi  : Ii8<0xC1, MRM5m, (ops i32mem:$dst, i8imm:$src),
                     "shr{l} {$src, $dst|$dst, $src}",
                 [(store (srl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
}

def SAR8rCL  : I<0xD2, MRM7r, (ops R8 :$dst, R8 :$src),
                 "sar{b} {%cl, $dst|$dst, %CL}",
                 [(set R8:$dst, (sra R8:$src, CL))]>, Imp<[CL],[]>;
def SAR16rCL : I<0xD3, MRM7r, (ops R16:$dst, R16:$src),
                 "sar{w} {%cl, $dst|$dst, %CL}",
                 [(set R16:$dst, (sra R16:$src, CL))]>, Imp<[CL],[]>, OpSize;
def SAR32rCL : I<0xD3, MRM7r, (ops R32:$dst, R32:$src),
                 "sar{l} {%cl, $dst|$dst, %CL}",
                 [(set R32:$dst, (sra R32:$src, CL))]>, Imp<[CL],[]>;

def SAR8ri   : Ii8<0xC0, MRM7r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "sar{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (sra R8:$src1, (i8 imm:$src2)))]>;
def SAR16ri  : Ii8<0xC1, MRM7r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "sar{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (sra R16:$src1, (i8 imm:$src2)))]>,
                   OpSize;
def SAR32ri  : Ii8<0xC1, MRM7r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "sar{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (sra R32:$src1, (i8 imm:$src2)))]>;
let isTwoAddress = 0 in {
  def SAR8mCL  : I<0xD2, MRM7m, (ops i8mem :$dst),
                   "sar{b} {%cl, $dst|$dst, %CL}",
                   [(store (sra (loadi8 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def SAR16mCL : I<0xD3, MRM7m, (ops i16mem:$dst),
                   "sar{w} {%cl, $dst|$dst, %CL}",
                   [(store (sra (loadi16 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>, OpSize;
  def SAR32mCL : I<0xD3, MRM7m, (ops i32mem:$dst), 
                   "sar{l} {%cl, $dst|$dst, %CL}",
                   [(store (sra (loadi32 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def SAR8mi   : Ii8<0xC0, MRM7m, (ops i8mem :$dst, i8imm:$src),
                     "sar{b} {$src, $dst|$dst, $src}",
                  [(store (sra (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SAR16mi  : Ii8<0xC1, MRM7m, (ops i16mem:$dst, i8imm:$src),
                     "sar{w} {$src, $dst|$dst, $src}",
                 [(store (sra (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SAR32mi  : Ii8<0xC1, MRM7m, (ops i32mem:$dst, i8imm:$src),
                     "sar{l} {$src, $dst|$dst, $src}",
                 [(store (sra (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
}

// Rotate instructions
// FIXME: provide shorter instructions when imm8 == 1
def ROL8rCL  : I<0xD2, MRM0r, (ops R8 :$dst, R8 :$src),
                 "rol{b} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;
def ROL16rCL : I<0xD3, MRM0r, (ops R16:$dst, R16:$src),
                 "rol{w} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>, OpSize;
def ROL32rCL : I<0xD3, MRM0r, (ops R32:$dst, R32:$src),
                 "rol{l} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;

def ROL8ri   : Ii8<0xC0, MRM0r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "rol{b} {$src2, $dst|$dst, $src2}", []>;
def ROL16ri  : Ii8<0xC1, MRM0r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "rol{w} {$src2, $dst|$dst, $src2}", []>, OpSize;
def ROL32ri  : Ii8<0xC1, MRM0r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "rol{l} {$src2, $dst|$dst, $src2}", []>;

let isTwoAddress = 0 in {
  def ROL8mCL  : I<0xD2, MRM0m, (ops i8mem :$dst),
                   "rol{b} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;
  def ROL16mCL : I<0xD3, MRM0m, (ops i16mem:$dst),
                   "rol{w} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>, OpSize;
  def ROL32mCL : I<0xD3, MRM0m, (ops i32mem:$dst),
                   "rol{l} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;
  def ROL8mi   : Ii8<0xC0, MRM0m, (ops i8mem :$dst, i8imm:$src),
                     "rol{b} {$src, $dst|$dst, $src}", []>;
  def ROL16mi  : Ii8<0xC1, MRM0m, (ops i16mem:$dst, i8imm:$src),
                     "rol{w} {$src, $dst|$dst, $src}", []>, OpSize;
  def ROL32mi  : Ii8<0xC1, MRM0m, (ops i32mem:$dst, i8imm:$src),
                     "rol{l} {$src, $dst|$dst, $src}", []>;
}

def ROR8rCL  : I<0xD2, MRM1r, (ops R8 :$dst, R8 :$src),
                 "ror{b} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;
def ROR16rCL : I<0xD3, MRM1r, (ops R16:$dst, R16:$src),
                 "ror{w} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>, OpSize;
def ROR32rCL : I<0xD3, MRM1r, (ops R32:$dst, R32:$src),
                 "ror{l} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;

def ROR8ri   : Ii8<0xC0, MRM1r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "ror{b} {$src2, $dst|$dst, $src2}", []>;
def ROR16ri  : Ii8<0xC1, MRM1r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "ror{w} {$src2, $dst|$dst, $src2}", []>, OpSize;
def ROR32ri  : Ii8<0xC1, MRM1r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "ror{l} {$src2, $dst|$dst, $src2}", []>;
let isTwoAddress = 0 in {
  def ROR8mCL  : I<0xD2, MRM1m, (ops i8mem :$dst),
                   "ror{b} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;
  def ROR16mCL : I<0xD3, MRM1m, (ops i16mem:$dst),
                   "ror{w} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>, OpSize;
  def ROR32mCL : I<0xD3, MRM1m, (ops i32mem:$dst), 
                   "ror{l} {%cl, $dst|$dst, %CL}", []>, Imp<[CL],[]>;
  def ROR8mi   : Ii8<0xC0, MRM1m, (ops i8mem :$dst, i8imm:$src),
                     "ror{b} {$src, $dst|$dst, $src}", []>;
  def ROR16mi  : Ii8<0xC1, MRM1m, (ops i16mem:$dst, i8imm:$src),
                     "ror{w} {$src, $dst|$dst, $src}", []>, OpSize;
  def ROR32mi  : Ii8<0xC1, MRM1m, (ops i32mem:$dst, i8imm:$src),
                     "ror{l} {$src, $dst|$dst, $src}", []>;
}



// Double shift instructions (generalizations of rotate)

def SHLD32rrCL : I<0xA5, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                   "shld{l} {%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set R32:$dst, (X86shld R32:$src1, R32:$src2, CL))]>,
                   Imp<[CL],[]>, TB;
def SHRD32rrCL : I<0xAD, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                   "shrd{l} {%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set R32:$dst, (X86shrd R32:$src1, R32:$src2, CL))]>,
                   Imp<[CL],[]>, TB;
def SHLD16rrCL : I<0xA5, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2),
                   "shld{w} {%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set R16:$dst, (X86shld R16:$src1, R16:$src2, CL))]>,
                   Imp<[CL],[]>, TB, OpSize;
def SHRD16rrCL : I<0xAD, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2),
                   "shrd{w} {%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set R16:$dst, (X86shrd R16:$src1, R16:$src2, CL))]>,
                   Imp<[CL],[]>, TB, OpSize;

let isCommutable = 1 in {  // These instructions commute to each other.
def SHLD32rri8 : Ii8<0xA4, MRMDestReg,
                     (ops R32:$dst, R32:$src1, R32:$src2, i8imm:$src3),
                     "shld{l} {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set R32:$dst, (X86shld R32:$src1, R32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB;
def SHRD32rri8 : Ii8<0xAC, MRMDestReg,
                     (ops R32:$dst, R32:$src1, R32:$src2, i8imm:$src3),
                     "shrd{l} {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set R32:$dst, (X86shrd R32:$src1, R32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB;
def SHLD16rri8 : Ii8<0xA4, MRMDestReg,
                     (ops R16:$dst, R16:$src1, R16:$src2, i8imm:$src3),
                     "shld{w} {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set R16:$dst, (X86shld R16:$src1, R16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize;
def SHRD16rri8 : Ii8<0xAC, MRMDestReg,
                     (ops R16:$dst, R16:$src1, R16:$src2, i8imm:$src3),
                     "shrd{w} {$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set R16:$dst, (X86shrd R16:$src1, R16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize;
}

let isTwoAddress = 0 in {
  def SHLD32mrCL : I<0xA5, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                     "shld{l} {%cl, $src2, $dst|$dst, $src2, %CL}",
                     [(store (X86shld (loadi32 addr:$dst), R32:$src2, CL),
                       addr:$dst)]>,
                     Imp<[CL],[]>, TB;
  def SHRD32mrCL : I<0xAD, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                    "shrd{l} {%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shrd (loadi32 addr:$dst), R32:$src2, CL),
                      addr:$dst)]>,
                    Imp<[CL],[]>, TB;
  def SHLD32mri8 : Ii8<0xA4, MRMDestMem,
                      (ops i32mem:$dst, R32:$src2, i8imm:$src3),
                      "shld{l} {$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi32 addr:$dst), R32:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                      TB;
  def SHRD32mri8 : Ii8<0xAC, MRMDestMem, 
                       (ops i32mem:$dst, R32:$src2, i8imm:$src3),
                       "shrd{l} {$src3, $src2, $dst|$dst, $src2, $src3}",
                       [(store (X86shrd (loadi32 addr:$dst), R32:$src2,
                                         (i8 imm:$src3)), addr:$dst)]>,
                       TB;

  def SHLD16mrCL : I<0xA5, MRMDestMem, (ops i16mem:$dst, R16:$src2),
                     "shld{w} {%cl, $src2, $dst|$dst, $src2, %CL}",
                     [(store (X86shld (loadi16 addr:$dst), R16:$src2, CL),
                       addr:$dst)]>,
                     Imp<[CL],[]>, TB, OpSize;
  def SHRD16mrCL : I<0xAD, MRMDestMem, (ops i16mem:$dst, R16:$src2),
                    "shrd{w} {%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shrd (loadi16 addr:$dst), R16:$src2, CL),
                      addr:$dst)]>,
                    Imp<[CL],[]>, TB, OpSize;
  def SHLD16mri8 : Ii8<0xA4, MRMDestMem,
                      (ops i16mem:$dst, R16:$src2, i8imm:$src3),
                      "shld{w} {$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi16 addr:$dst), R16:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                      TB, OpSize;
  def SHRD16mri8 : Ii8<0xAC, MRMDestMem, 
                       (ops i16mem:$dst, R16:$src2, i8imm:$src3),
                       "shrd{w} {$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shrd (loadi16 addr:$dst), R16:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                       TB, OpSize;
}


// Arithmetic.
let isCommutable = 1 in {   // X = ADD Y, Z   --> X = ADD Z, Y
def ADD8rr   : I<0x00, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2),
                 "add{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (add R8:$src1, R8:$src2))]>;
let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def ADD16rr  : I<0x01, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2),
                 "add{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (add R16:$src1, R16:$src2))]>, OpSize;
def ADD32rr  : I<0x01, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "add{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (add R32:$src1, R32:$src2))]>;
} // end isConvertibleToThreeAddress
} // end isCommutable
def ADD8rm   : I<0x02, MRMSrcMem, (ops R8 :$dst, R8 :$src1, i8mem :$src2),
                 "add{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (add R8:$src1, (load addr:$src2)))]>;
def ADD16rm  : I<0x03, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                 "add{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (add R16:$src1, (load addr:$src2)))]>, OpSize;
def ADD32rm  : I<0x03, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "add{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (add R32:$src1, (load addr:$src2)))]>;

def ADD8ri   : Ii8<0x80, MRM0r, (ops R8:$dst, R8:$src1, i8imm:$src2),
                   "add{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (add R8:$src1, imm:$src2))]>;

let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def ADD16ri  : Ii16<0x81, MRM0r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "add{w} {$src2, $dst|$dst, $src2}",
                    [(set R16:$dst, (add R16:$src1, imm:$src2))]>, OpSize;
def ADD32ri  : Ii32<0x81, MRM0r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "add{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (add R32:$src1, imm:$src2))]>;
}

// FIXME: move ADD16ri8 above ADD16ri to optimize for space.
def ADD16ri8 : Ii8<0x83, MRM0r, (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                   "add{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (add R16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def ADD32ri8 : Ii8<0x83, MRM0r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "add{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (add R32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def ADD8mr   : I<0x00, MRMDestMem, (ops i8mem :$dst, R8 :$src2),
                   "add{b} {$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), R8:$src2), addr:$dst)]>;
  def ADD16mr  : I<0x01, MRMDestMem, (ops i16mem:$dst, R16:$src2),
                   "add{w} {$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), R16:$src2), addr:$dst)]>,
                   OpSize;
  def ADD32mr  : I<0x01, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                   "add{l} {$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), R32:$src2), addr:$dst)]>;
  def ADD8mi   : Ii8<0x80, MRM0m, (ops i8mem :$dst, i8imm :$src2),
                     "add{b} {$src2, $dst|$dst, $src2}",
                   [(store (add (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADD16mi  : Ii16<0x81, MRM0m, (ops i16mem:$dst, i16imm:$src2),
                      "add{w} {$src2, $dst|$dst, $src2}",
                  [(store (add (loadi16 addr:$dst), imm:$src2), addr:$dst)]>,
                   OpSize;
  def ADD32mi  : Ii32<0x81, MRM0m, (ops i32mem:$dst, i32imm:$src2),
                      "add{l} {$src2, $dst|$dst, $src2}",
                  [(store (add (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADD16mi8 : Ii8<0x83, MRM0m, (ops i16mem:$dst, i16i8imm :$src2),
                     "add{w} {$src2, $dst|$dst, $src2}",
                [(store (add (load addr:$dst), i16immSExt8:$src2), addr:$dst)]>,
                   OpSize;
  def ADD32mi8 : Ii8<0x83, MRM0m, (ops i32mem:$dst, i32i8imm :$src2),
                     "add{l} {$src2, $dst|$dst, $src2}",
                [(store (add (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}

let isCommutable = 1 in {  // X = ADC Y, Z --> X = ADC Z, Y
def ADC32rr  : I<0x11, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (X86adc R32:$src1, R32:$src2))]>;
}
def ADC32rm  : I<0x13, MRMSrcMem , (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (X86adc R32:$src1, (load addr:$src2)))]>;
def ADC32ri  : Ii32<0x81, MRM2r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (X86adc R32:$src1, imm:$src2))]>;
def ADC32ri8 : Ii8<0x83, MRM2r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (X86adc R32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def ADC32mr  : I<0x11, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                   "adc{l} {$src2, $dst|$dst, $src2}",
                   [(store (X86adc (load addr:$dst), R32:$src2), addr:$dst)]>;
  def ADC32mi  : Ii32<0x81, MRM2m, (ops i32mem:$dst, i32imm:$src2),
                      "adc{l} {$src2, $dst|$dst, $src2}",
                  [(store (X86adc (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADC32mi8 : Ii8<0x83, MRM2m, (ops i32mem:$dst, i32i8imm :$src2),
                     "adc{l} {$src2, $dst|$dst, $src2}",
             [(store (X86adc (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}

def SUB8rr   : I<0x28, MRMDestReg, (ops R8 :$dst, R8 :$src1, R8 :$src2),
                 "sub{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (sub R8:$src1, R8:$src2))]>;
def SUB16rr  : I<0x29, MRMDestReg, (ops R16:$dst, R16:$src1, R16:$src2),
                 "sub{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (sub R16:$src1, R16:$src2))]>, OpSize;
def SUB32rr  : I<0x29, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "sub{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (sub R32:$src1, R32:$src2))]>;
def SUB8rm   : I<0x2A, MRMSrcMem, (ops R8 :$dst, R8 :$src1, i8mem :$src2),
                 "sub{b} {$src2, $dst|$dst, $src2}",
                 [(set R8:$dst, (sub R8:$src1, (load addr:$src2)))]>;
def SUB16rm  : I<0x2B, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                 "sub{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (sub R16:$src1, (load addr:$src2)))]>, OpSize;
def SUB32rm  : I<0x2B, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "sub{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (sub R32:$src1, (load addr:$src2)))]>;

def SUB8ri   : Ii8 <0x80, MRM5r, (ops R8:$dst, R8:$src1, i8imm:$src2),
                    "sub{b} {$src2, $dst|$dst, $src2}",
                    [(set R8:$dst, (sub R8:$src1, imm:$src2))]>;
def SUB16ri  : Ii16<0x81, MRM5r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "sub{w} {$src2, $dst|$dst, $src2}",
                    [(set R16:$dst, (sub R16:$src1, imm:$src2))]>, OpSize;
def SUB32ri  : Ii32<0x81, MRM5r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "sub{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (sub R32:$src1, imm:$src2))]>;
def SUB16ri8 : Ii8<0x83, MRM5r, (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                   "sub{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (sub R16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def SUB32ri8 : Ii8<0x83, MRM5r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "sub{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (sub R32:$src1, i32immSExt8:$src2))]>;
let isTwoAddress = 0 in {
  def SUB8mr   : I<0x28, MRMDestMem, (ops i8mem :$dst, R8 :$src2),
                   "sub{b} {$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), R8:$src2), addr:$dst)]>;
  def SUB16mr  : I<0x29, MRMDestMem, (ops i16mem:$dst, R16:$src2),
                   "sub{w} {$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), R16:$src2), addr:$dst)]>,
                   OpSize;
  def SUB32mr  : I<0x29, MRMDestMem, (ops i32mem:$dst, R32:$src2), 
                   "sub{l} {$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), R32:$src2), addr:$dst)]>;
  def SUB8mi   : Ii8<0x80, MRM5m, (ops i8mem :$dst, i8imm:$src2), 
                     "sub{b} {$src2, $dst|$dst, $src2}",
                   [(store (sub (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def SUB16mi  : Ii16<0x81, MRM5m, (ops i16mem:$dst, i16imm:$src2), 
                      "sub{w} {$src2, $dst|$dst, $src2}",
                  [(store (sub (loadi16 addr:$dst), imm:$src2), addr:$dst)]>,
                   OpSize;
  def SUB32mi  : Ii32<0x81, MRM5m, (ops i32mem:$dst, i32imm:$src2), 
                      "sub{l} {$src2, $dst|$dst, $src2}",
                  [(store (sub (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def SUB16mi8 : Ii8<0x83, MRM5m, (ops i16mem:$dst, i16i8imm :$src2), 
                     "sub{w} {$src2, $dst|$dst, $src2}",
                [(store (sub (load addr:$dst), i16immSExt8:$src2), addr:$dst)]>,
                   OpSize;
  def SUB32mi8 : Ii8<0x83, MRM5m, (ops i32mem:$dst, i32i8imm :$src2), 
                     "sub{l} {$src2, $dst|$dst, $src2}",
                [(store (sub (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}

def SBB32rr    : I<0x19, MRMDestReg, (ops R32:$dst, R32:$src1, R32:$src2),
                  "sbb{l} {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86sbb R32:$src1, R32:$src2))]>;

let isTwoAddress = 0 in {
  def SBB32mr  : I<0x19, MRMDestMem, (ops i32mem:$dst, R32:$src2), 
                   "sbb{l} {$src2, $dst|$dst, $src2}",
                   [(store (X86sbb (load addr:$dst), R32:$src2), addr:$dst)]>;
  def SBB8mi  : Ii32<0x80, MRM3m, (ops i8mem:$dst, i8imm:$src2), 
                      "sbb{b} {$src2, $dst|$dst, $src2}",
                   [(store (X86sbb (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB16mi  : Ii32<0x81, MRM3m, (ops i16mem:$dst, i16imm:$src2), 
                      "sbb{w} {$src2, $dst|$dst, $src2}",
                  [(store (X86sbb (loadi16 addr:$dst), imm:$src2), addr:$dst)]>,
                     OpSize;
  def SBB32mi  : Ii32<0x81, MRM3m, (ops i32mem:$dst, i32imm:$src2), 
                      "sbb{l} {$src2, $dst|$dst, $src2}",
                  [(store (X86sbb (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB16mi8 : Ii8<0x83, MRM3m, (ops i16mem:$dst, i16i8imm :$src2), 
                     "sbb{w} {$src2, $dst|$dst, $src2}",
             [(store (X86sbb (load addr:$dst), i16immSExt8:$src2), addr:$dst)]>,
                     OpSize;
  def SBB32mi8 : Ii8<0x83, MRM3m, (ops i32mem:$dst, i32i8imm :$src2), 
                     "sbb{l} {$src2, $dst|$dst, $src2}",
             [(store (X86sbb (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}
def SBB8ri   : Ii8<0x80, MRM3r, (ops R8:$dst, R8:$src1, i8imm:$src2),
                    "sbb{b} {$src2, $dst|$dst, $src2}",
                    [(set R8:$dst, (X86sbb R8:$src1, imm:$src2))]>;
def SBB16ri  : Ii16<0x81, MRM3r, (ops R16:$dst, R16:$src1, i16imm:$src2),
                    "sbb{w} {$src2, $dst|$dst, $src2}",
                    [(set R16:$dst, (X86sbb R16:$src1, imm:$src2))]>, OpSize;

def SBB32rm  : I<0x1B, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                    "sbb{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (X86sbb R32:$src1, (load addr:$src2)))]>;
def SBB32ri  : Ii32<0x81, MRM3r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "sbb{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (X86sbb R32:$src1, imm:$src2))]>;

def SBB16ri8 : Ii8<0x83, MRM3r, (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                   "sbb{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (X86sbb R16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def SBB32ri8 : Ii8<0x83, MRM3r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "sbb{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (X86sbb R32:$src1, i32immSExt8:$src2))]>;

let isCommutable = 1 in {  // X = IMUL Y, Z --> X = IMUL Z, Y
def IMUL16rr : I<0xAF, MRMSrcReg, (ops R16:$dst, R16:$src1, R16:$src2),
                 "imul{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (mul R16:$src1, R16:$src2))]>, TB, OpSize;
def IMUL32rr : I<0xAF, MRMSrcReg, (ops R32:$dst, R32:$src1, R32:$src2),
                 "imul{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (mul R32:$src1, R32:$src2))]>, TB;
}
def IMUL16rm : I<0xAF, MRMSrcMem, (ops R16:$dst, R16:$src1, i16mem:$src2),
                 "imul{w} {$src2, $dst|$dst, $src2}",
                 [(set R16:$dst, (mul R16:$src1, (load addr:$src2)))]>,
                 TB, OpSize;
def IMUL32rm : I<0xAF, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "imul{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (mul R32:$src1, (load addr:$src2)))]>, TB;

} // end Two Address instructions

// Suprisingly enough, these are not two address instructions!
def IMUL16rri  : Ii16<0x69, MRMSrcReg,                      // R16 = R16*I16
                      (ops R16:$dst, R16:$src1, i16imm:$src2),
                      "imul{w} {$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set R16:$dst, (mul R16:$src1, imm:$src2))]>, OpSize;
def IMUL32rri  : Ii32<0x69, MRMSrcReg,                      // R32 = R32*I32
                      (ops R32:$dst, R32:$src1, i32imm:$src2),
                      "imul{l} {$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set R32:$dst, (mul R32:$src1, imm:$src2))]>;
def IMUL16rri8 : Ii8<0x6B, MRMSrcReg,                       // R16 = R16*I8
                     (ops R16:$dst, R16:$src1, i16i8imm:$src2),
                     "imul{w} {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set R16:$dst, (mul R16:$src1, i16immSExt8:$src2))]>,
                     OpSize;
def IMUL32rri8 : Ii8<0x6B, MRMSrcReg,                       // R32 = R32*I8
                     (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                     "imul{l} {$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set R32:$dst, (mul R32:$src1, i32immSExt8:$src2))]>;

def IMUL16rmi  : Ii16<0x69, MRMSrcMem,                      // R16 = [mem16]*I16
                      (ops R16:$dst, i16mem:$src1, i16imm:$src2),
                      "imul{w} {$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set R16:$dst, (mul (load addr:$src1), imm:$src2))]>,
                      OpSize;
def IMUL32rmi  : Ii32<0x69, MRMSrcMem,                      // R32 = [mem32]*I32
                      (ops R32:$dst, i32mem:$src1, i32imm:$src2),
                      "imul{l} {$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set R32:$dst, (mul (load addr:$src1), imm:$src2))]>;
def IMUL16rmi8 : Ii8<0x6B, MRMSrcMem,                       // R16 = [mem16]*I8
                     (ops R16:$dst, i16mem:$src1, i16i8imm :$src2),
                     "imul{w} {$src2, $src1, $dst|$dst, $src1, $src2}",
                  [(set R16:$dst, (mul (load addr:$src1), i16immSExt8:$src2))]>,
                     OpSize;
def IMUL32rmi8 : Ii8<0x6B, MRMSrcMem,                       // R32 = [mem32]*I8
                     (ops R32:$dst, i32mem:$src1, i32i8imm: $src2),
                     "imul{l} {$src2, $src1, $dst|$dst, $src1, $src2}",
                  [(set R32:$dst, (mul (load addr:$src1), i32immSExt8:$src2))]>;

//===----------------------------------------------------------------------===//
// Test instructions are just like AND, except they don't generate a result.
//
let isCommutable = 1 in {   // TEST X, Y   --> TEST Y, X
def TEST8rr  : I<0x84, MRMDestReg, (ops R8:$src1, R8:$src2),
                 "test{b} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test R8:$src1, R8:$src2))]>,
               Imp<[],[STATUS]>;
def TEST16rr : I<0x85, MRMDestReg, (ops R16:$src1, R16:$src2),
                 "test{w} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test R16:$src1, R16:$src2))]>,
               Imp<[],[STATUS]>, OpSize;
def TEST32rr : I<0x85, MRMDestReg, (ops R32:$src1, R32:$src2),
                 "test{l} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test R32:$src1, R32:$src2))]>,
               Imp<[],[STATUS]>;
}
def TEST8mr  : I<0x84, MRMDestMem, (ops i8mem :$src1, R8 :$src2),
                 "test{b} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test (loadi8 addr:$src1), R8:$src2))]>,
               Imp<[],[STATUS]>;
def TEST16mr : I<0x85, MRMDestMem, (ops i16mem:$src1, R16:$src2),
                 "test{w} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test (loadi16 addr:$src1), R16:$src2))]>,
               Imp<[],[STATUS]>, OpSize;
def TEST32mr : I<0x85, MRMDestMem, (ops i32mem:$src1, R32:$src2),
                 "test{l} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test (loadi32 addr:$src1), R32:$src2))]>,
               Imp<[],[STATUS]>;
def TEST8rm  : I<0x84, MRMSrcMem, (ops R8 :$src1, i8mem :$src2),
                 "test{b} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test R8:$src1, (loadi8 addr:$src2)))]>,
               Imp<[],[STATUS]>;
def TEST16rm : I<0x85, MRMSrcMem, (ops R16:$src1, i16mem:$src2),
                 "test{w} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test R16:$src1, (loadi16 addr:$src2)))]>,
               Imp<[],[STATUS]>, OpSize;
def TEST32rm : I<0x85, MRMSrcMem, (ops R32:$src1, i32mem:$src2),
                 "test{l} {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86test R32:$src1, (loadi32 addr:$src2)))]>,
               Imp<[],[STATUS]>;

def TEST8ri  : Ii8 <0xF6, MRM0r,                     // flags = R8  & imm8
                    (ops R8:$src1, i8imm:$src2),
                    "test{b} {$src2, $src1|$src1, $src2}",
                    [(set STATUS, (X86test R8:$src1, imm:$src2))]>,
                   Imp<[],[STATUS]>;
def TEST16ri : Ii16<0xF7, MRM0r,                     // flags = R16 & imm16
                    (ops R16:$src1, i16imm:$src2),
                    "test{w} {$src2, $src1|$src1, $src2}",
                    [(set STATUS, (X86test R16:$src1, imm:$src2))]>,
                   Imp<[],[STATUS]>, OpSize;
def TEST32ri : Ii32<0xF7, MRM0r,                     // flags = R32 & imm32
                    (ops R32:$src1, i32imm:$src2),
                    "test{l} {$src2, $src1|$src1, $src2}",
                    [(set STATUS, (X86test R32:$src1, imm:$src2))]>,
                   Imp<[],[STATUS]>;
def TEST8mi  : Ii8 <0xF6, MRM0m,                     // flags = [mem8]  & imm8
                    (ops i8mem:$src1, i8imm:$src2),
                    "test{b} {$src2, $src1|$src1, $src2}",
                    [(set STATUS, (X86test (loadi8 addr:$src1), imm:$src2))]>,
                   Imp<[],[STATUS]>;
def TEST16mi : Ii16<0xF7, MRM0m,                     // flags = [mem16] & imm16
                    (ops i16mem:$src1, i16imm:$src2),
                    "test{w} {$src2, $src1|$src1, $src2}",
                    [(set STATUS, (X86test (loadi16 addr:$src1), imm:$src2))]>,
                   Imp<[],[STATUS]>, OpSize;
def TEST32mi : Ii32<0xF7, MRM0m,                     // flags = [mem32] & imm32
                    (ops i32mem:$src1, i32imm:$src2),
                    "test{l} {$src2, $src1|$src1, $src2}",
                    [(set STATUS, (X86test (loadi32 addr:$src1), imm:$src2))]>,
                   Imp<[],[STATUS]>;


// Condition code ops, incl. set if equal/not equal/...
def SAHF     : I<0x9E, RawFrm, (ops), "sahf", []>, Imp<[AH],[]>;  // flags = AH
def LAHF     : I<0x9F, RawFrm, (ops), "lahf", []>, Imp<[],[AH]>;  // AH = flags

def SETEr    : I<0x94, MRM0r, 
                 (ops R8   :$dst),
                 "sete $dst",
                 [(set R8:$dst, (X86setcc X86_COND_E, STATUS))]>,
               TB;                        // R8 = ==
def SETEm    : I<0x94, MRM0m, 
                 (ops i8mem:$dst),
                 "sete $dst",
                 [(store (X86setcc X86_COND_E, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = ==
def SETNEr   : I<0x95, MRM0r, 
                 (ops R8   :$dst),
                 "setne $dst",
                 [(set R8:$dst, (X86setcc X86_COND_NE, STATUS))]>,
               TB;                        // R8 = !=
def SETNEm   : I<0x95, MRM0m, 
                 (ops i8mem:$dst),
                 "setne $dst",
                 [(store (X86setcc X86_COND_NE, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = !=
def SETLr    : I<0x9C, MRM0r, 
                 (ops R8   :$dst),
                 "setl $dst",
                 [(set R8:$dst, (X86setcc X86_COND_L, STATUS))]>,
               TB;                        // R8 = <  signed
def SETLm    : I<0x9C, MRM0m, 
                 (ops i8mem:$dst),
                 "setl $dst",
                 [(store (X86setcc X86_COND_L, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = <  signed
def SETGEr   : I<0x9D, MRM0r, 
                 (ops R8   :$dst),
                 "setge $dst",
                 [(set R8:$dst, (X86setcc X86_COND_GE, STATUS))]>,
               TB;                        // R8 = >= signed
def SETGEm   : I<0x9D, MRM0m, 
                 (ops i8mem:$dst),
                 "setge $dst",
                 [(store (X86setcc X86_COND_GE, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = >= signed
def SETLEr   : I<0x9E, MRM0r, 
                 (ops R8   :$dst),
                 "setle $dst",
                 [(set R8:$dst, (X86setcc X86_COND_LE, STATUS))]>,
               TB;                        // R8 = <= signed
def SETLEm   : I<0x9E, MRM0m, 
                 (ops i8mem:$dst),
                 "setle $dst",
                 [(store (X86setcc X86_COND_LE, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = <= signed
def SETGr    : I<0x9F, MRM0r, 
                 (ops R8   :$dst),
                 "setg $dst",
                 [(set R8:$dst, (X86setcc X86_COND_G, STATUS))]>,
               TB;                        // R8 = >  signed
def SETGm    : I<0x9F, MRM0m, 
                 (ops i8mem:$dst),
                 "setg $dst",
                 [(store (X86setcc X86_COND_G, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETBr    : I<0x92, MRM0r,
                 (ops R8   :$dst),
                 "setb $dst",
                 [(set R8:$dst, (X86setcc X86_COND_B, STATUS))]>,
               TB;                        // R8 = <  unsign
def SETBm    : I<0x92, MRM0m,
                 (ops i8mem:$dst),
                 "setb $dst",
                 [(store (X86setcc X86_COND_B, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = <  unsign
def SETAEr   : I<0x93, MRM0r, 
                 (ops R8   :$dst),
                 "setae $dst",
                 [(set R8:$dst, (X86setcc X86_COND_AE, STATUS))]>,
               TB;                        // R8 = >= unsign
def SETAEm   : I<0x93, MRM0m, 
                 (ops i8mem:$dst),
                 "setae $dst",
                 [(store (X86setcc X86_COND_AE, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = >= unsign
def SETBEr   : I<0x96, MRM0r, 
                 (ops R8   :$dst),
                 "setbe $dst",
                 [(set R8:$dst, (X86setcc X86_COND_BE, STATUS))]>,
               TB;                        // R8 = <= unsign
def SETBEm   : I<0x96, MRM0m, 
                 (ops i8mem:$dst),
                 "setbe $dst",
                 [(store (X86setcc X86_COND_BE, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = <= unsign
def SETAr    : I<0x97, MRM0r, 
                 (ops R8   :$dst),
                 "seta $dst",
                 [(set R8:$dst, (X86setcc X86_COND_A, STATUS))]>,
               TB;                        // R8 = >  signed
def SETAm    : I<0x97, MRM0m, 
                 (ops i8mem:$dst),
                 "seta $dst",
                 [(store (X86setcc X86_COND_A, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETSr    : I<0x98, MRM0r, 
                 (ops R8   :$dst),
                 "sets $dst",
                 [(set R8:$dst, (X86setcc X86_COND_S, STATUS))]>,
               TB;                        // R8 = <sign bit>
def SETSm    : I<0x98, MRM0m, 
                 (ops i8mem:$dst),
                 "sets $dst",
                 [(store (X86setcc X86_COND_S, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = <sign bit>
def SETNSr   : I<0x99, MRM0r, 
                 (ops R8   :$dst),
                 "setns $dst",
                 [(set R8:$dst, (X86setcc X86_COND_NS, STATUS))]>,
               TB;                        // R8 = !<sign bit>
def SETNSm   : I<0x99, MRM0m, 
                 (ops i8mem:$dst),
                 "setns $dst",
                 [(store (X86setcc X86_COND_NS, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = !<sign bit>
def SETPr    : I<0x9A, MRM0r, 
                 (ops R8   :$dst),
                 "setp $dst",
                 [(set R8:$dst, (X86setcc X86_COND_P, STATUS))]>,
               TB;                        // R8 = parity
def SETPm    : I<0x9A, MRM0m, 
                 (ops i8mem:$dst),
                 "setp $dst",
                 [(store (X86setcc X86_COND_P, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = parity
def SETNPr   : I<0x9B, MRM0r, 
                 (ops R8   :$dst),
                 "setnp $dst",
                 [(set R8:$dst, (X86setcc X86_COND_NP, STATUS))]>,
               TB;                        // R8 = not parity
def SETNPm   : I<0x9B, MRM0m, 
                 (ops i8mem:$dst),
                 "setnp $dst",
                 [(store (X86setcc X86_COND_NP, STATUS), addr:$dst)]>,
               TB;                        // [mem8] = not parity

// Integer comparisons
def CMP8rr  : I<0x38, MRMDestReg,
                (ops R8 :$src1, R8 :$src2),
                "cmp{b} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp R8:$src1, R8:$src2))]>,
              Imp<[],[STATUS]>;
def CMP16rr : I<0x39, MRMDestReg,
                (ops R16:$src1, R16:$src2),
                "cmp{w} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp R16:$src1, R16:$src2))]>,
              Imp<[],[STATUS]>, OpSize;
def CMP32rr : I<0x39, MRMDestReg,
                (ops R32:$src1, R32:$src2),
                "cmp{l} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp R32:$src1, R32:$src2))]>,
              Imp<[],[STATUS]>;
def CMP8mr  : I<0x38, MRMDestMem,
                (ops i8mem :$src1, R8 :$src2),
                "cmp{b} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp (loadi8 addr:$src1), R8:$src2))]>,
              Imp<[],[STATUS]>;
def CMP16mr : I<0x39, MRMDestMem,
                (ops i16mem:$src1, R16:$src2),
                "cmp{w} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp (loadi16 addr:$src1), R16:$src2))]>,
              Imp<[],[STATUS]>, OpSize;
def CMP32mr : I<0x39, MRMDestMem,
                (ops i32mem:$src1, R32:$src2),
                "cmp{l} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp (loadi32 addr:$src1), R32:$src2))]>,
              Imp<[],[STATUS]>;
def CMP8rm  : I<0x3A, MRMSrcMem,
                (ops R8 :$src1, i8mem :$src2),
                "cmp{b} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp R8:$src1, (loadi8 addr:$src2)))]>,
              Imp<[],[STATUS]>;
def CMP16rm : I<0x3B, MRMSrcMem,
                (ops R16:$src1, i16mem:$src2),
                "cmp{w} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp R16:$src1, (loadi16 addr:$src2)))]>,
                Imp<[],[STATUS]>, OpSize;
def CMP32rm : I<0x3B, MRMSrcMem,
                (ops R32:$src1, i32mem:$src2),
                "cmp{l} {$src2, $src1|$src1, $src2}",
                [(set STATUS, (X86cmp R32:$src1, (loadi32 addr:$src2)))]>,
              Imp<[],[STATUS]>;
def CMP8ri  : Ii8<0x80, MRM7r,
                  (ops R8:$src1, i8imm:$src2),
                  "cmp{b} {$src2, $src1|$src1, $src2}",
                  [(set STATUS, (X86cmp R8:$src1, imm:$src2))]>,
              Imp<[],[STATUS]>;
def CMP16ri : Ii16<0x81, MRM7r,
                   (ops R16:$src1, i16imm:$src2),
                   "cmp{w} {$src2, $src1|$src1, $src2}",
                   [(set STATUS, (X86cmp R16:$src1, imm:$src2))]>,
              Imp<[],[STATUS]>, OpSize;
def CMP32ri : Ii32<0x81, MRM7r,
                   (ops R32:$src1, i32imm:$src2),
                   "cmp{l} {$src2, $src1|$src1, $src2}",
                   [(set STATUS, (X86cmp R32:$src1, imm:$src2))]>,
              Imp<[],[STATUS]>;
def CMP8mi  : Ii8 <0x80, MRM7m,
                   (ops i8mem :$src1, i8imm :$src2),
                   "cmp{b} {$src2, $src1|$src1, $src2}",
                   [(set STATUS, (X86cmp (loadi8 addr:$src1), imm:$src2))]>,
              Imp<[],[STATUS]>;
def CMP16mi : Ii16<0x81, MRM7m,
                   (ops i16mem:$src1, i16imm:$src2),
                   "cmp{w} {$src2, $src1|$src1, $src2}",
                   [(set STATUS, (X86cmp (loadi16 addr:$src1), imm:$src2))]>,
              Imp<[],[STATUS]>, OpSize;
def CMP32mi : Ii32<0x81, MRM7m,
                   (ops i32mem:$src1, i32imm:$src2),
                   "cmp{l} {$src2, $src1|$src1, $src2}",
                   [(set STATUS, (X86cmp (loadi32 addr:$src1), imm:$src2))]>,
              Imp<[],[STATUS]>;

// Sign/Zero extenders
def MOVSX16rr8 : I<0xBE, MRMSrcReg, (ops R16:$dst, R8 :$src),
                   "movs{bw|x} {$src, $dst|$dst, $src}",
                   [(set R16:$dst, (sext R8:$src))]>, TB, OpSize;
def MOVSX16rm8 : I<0xBE, MRMSrcMem, (ops R16:$dst, i8mem :$src),
                   "movs{bw|x} {$src, $dst|$dst, $src}",
                   [(set R16:$dst, (sextloadi16i8 addr:$src))]>, TB, OpSize;
def MOVSX32rr8 : I<0xBE, MRMSrcReg, (ops R32:$dst, R8 :$src),
                   "movs{bl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (sext R8:$src))]>, TB;
def MOVSX32rm8 : I<0xBE, MRMSrcMem, (ops R32:$dst, i8mem :$src),
                   "movs{bl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (sextloadi32i8 addr:$src))]>, TB;
def MOVSX32rr16: I<0xBF, MRMSrcReg, (ops R32:$dst, R16:$src),
                   "movs{wl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (sext R16:$src))]>, TB;
def MOVSX32rm16: I<0xBF, MRMSrcMem, (ops R32:$dst, i16mem:$src),
                   "movs{wl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (sextloadi32i16 addr:$src))]>, TB;

def MOVZX16rr8 : I<0xB6, MRMSrcReg, (ops R16:$dst, R8 :$src),
                   "movz{bw|x} {$src, $dst|$dst, $src}",
                   [(set R16:$dst, (zext R8:$src))]>, TB, OpSize;
def MOVZX16rm8 : I<0xB6, MRMSrcMem, (ops R16:$dst, i8mem :$src),
                   "movz{bw|x} {$src, $dst|$dst, $src}",
                   [(set R16:$dst, (zextloadi16i8 addr:$src))]>, TB, OpSize;
def MOVZX32rr8 : I<0xB6, MRMSrcReg, (ops R32:$dst, R8 :$src),
                   "movz{bl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (zext R8:$src))]>, TB;
def MOVZX32rm8 : I<0xB6, MRMSrcMem, (ops R32:$dst, i8mem :$src),
                   "movz{bl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (zextloadi32i8 addr:$src))]>, TB;
def MOVZX32rr16: I<0xB7, MRMSrcReg, (ops R32:$dst, R16:$src),
                   "movz{wl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (zext R16:$src))]>, TB;
def MOVZX32rm16: I<0xB7, MRMSrcMem, (ops R32:$dst, i16mem:$src),
                   "movz{wl|x} {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (zextloadi32i16 addr:$src))]>, TB;

//===----------------------------------------------------------------------===//
// XMM Floating point support (requires SSE / SSE2)
//===----------------------------------------------------------------------===//

def MOVSSrr : I<0x10, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XS;
def MOVSDrr : I<0x10, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE2]>, XD;

def MOVSSrm : I<0x10, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                "movss {$src, $dst|$dst, $src}",
                [(set FR32:$dst, (loadf32 addr:$src))]>,
              Requires<[HasSSE1]>, XS;
def MOVSSmr : I<0x11, MRMDestMem, (ops f32mem:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}",
                [(store FR32:$src, addr:$dst)]>,
              Requires<[HasSSE1]>, XS;
def MOVSDrm : I<0x10, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(set FR64:$dst, (loadf64 addr:$src))]>,
              Requires<[HasSSE2]>, XD;
def MOVSDmr : I<0x11, MRMDestMem, (ops f64mem:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(store FR64:$src, addr:$dst)]>,
              Requires<[HasSSE2]>, XD;

def CVTTSD2SIrr: I<0x2C, MRMSrcReg, (ops R32:$dst, FR64:$src),
                   "cvttsd2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint FR64:$src))]>,
                 Requires<[HasSSE2]>, XD;
def CVTTSD2SIrm: I<0x2C, MRMSrcMem, (ops R32:$dst, f64mem:$src),
                   "cvttsd2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint (loadf64 addr:$src)))]>,
                 Requires<[HasSSE2]>, XD;
def CVTTSS2SIrr: I<0x2C, MRMSrcReg, (ops R32:$dst, FR32:$src),
                   "cvttss2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint FR32:$src))]>,
                 Requires<[HasSSE1]>, XS;
def CVTTSS2SIrm: I<0x2C, MRMSrcMem, (ops R32:$dst, f32mem:$src),
                   "cvttss2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint (loadf32 addr:$src)))]>,
                 Requires<[HasSSE1]>, XS;
def CVTSD2SSrr: I<0x5A, MRMSrcReg, (ops FR32:$dst, FR64:$src),
                  "cvtsd2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (fround FR64:$src))]>,
                Requires<[HasSSE2]>, XS;
def CVTSD2SSrm: I<0x5A, MRMSrcMem, (ops FR32:$dst, f64mem:$src), 
                  "cvtsd2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (fround (loadf64 addr:$src)))]>,
                Requires<[HasSSE2]>, XS;
def CVTSS2SDrr: I<0x5A, MRMSrcReg, (ops FR64:$dst, FR32:$src),
                  "cvtss2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (fextend FR32:$src))]>,
                Requires<[HasSSE2]>, XD;
def CVTSS2SDrm: I<0x5A, MRMSrcMem, (ops FR64:$dst, f32mem:$src),
                  "cvtss2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (fextend (loadf32 addr:$src)))]>,
                Requires<[HasSSE2]>, XD;
def CVTSI2SSrr: I<0x2A, MRMSrcReg, (ops FR32:$dst, R32:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp R32:$src))]>,
                Requires<[HasSSE2]>, XS;
def CVTSI2SSrm: I<0x2A, MRMSrcMem, (ops FR32:$dst, i32mem:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp (loadi32 addr:$src)))]>,
                Requires<[HasSSE2]>, XS;
def CVTSI2SDrr: I<0x2A, MRMSrcReg, (ops FR64:$dst, R32:$src),
                  "cvtsi2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (sint_to_fp R32:$src))]>,
                Requires<[HasSSE2]>, XD;
def CVTSI2SDrm: I<0x2A, MRMSrcMem, (ops FR64:$dst, i32mem:$src),
                  "cvtsi2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (sint_to_fp (loadi32 addr:$src)))]>,
                Requires<[HasSSE2]>, XD;

def SQRTSSrm : I<0x51, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt (loadf32 addr:$src)))]>,
               Requires<[HasSSE1]>, XS;
def SQRTSSrr : I<0x51, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt FR32:$src))]>,
               Requires<[HasSSE1]>, XS;
def SQRTSDrm : I<0x51, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt (loadf64 addr:$src)))]>,
               Requires<[HasSSE2]>, XD;
def SQRTSDrr : I<0x51, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt FR64:$src))]>,
               Requires<[HasSSE2]>, XD;

def UCOMISDrr: I<0x2E, MRMSrcReg, (ops FR64:$src1, FR64:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86cmp FR64:$src1, FR64:$src2))]>,
               Requires<[HasSSE2]>, TB, OpSize;
def UCOMISDrm: I<0x2E, MRMSrcMem, (ops FR64:$src1, f64mem:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86cmp FR64:$src1, (loadf64 addr:$src2)))]>,
               Imp<[],[STATUS]>, Requires<[HasSSE2]>, TB, OpSize;
def UCOMISSrr: I<0x2E, MRMSrcReg, (ops FR32:$src1, FR32:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86cmp FR32:$src1, FR32:$src2))]>,
               Imp<[],[STATUS]>, Requires<[HasSSE1]>, TB;
def UCOMISSrm: I<0x2E, MRMSrcMem, (ops FR32:$src1, f32mem:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(set STATUS, (X86cmp FR32:$src1, (loadf32 addr:$src2)))]>,
               Imp<[],[STATUS]>, Requires<[HasSSE1]>, TB;

// Pseudo-instructions that map fld0 to xorps/xorpd for sse.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
def FLD0SS : I<0x57, MRMSrcReg, (ops FR32:$dst),
               "xorps $dst, $dst", [(set FR32:$dst, fp32imm0)]>,
             Requires<[HasSSE1]>, TB;
def FLD0SD : I<0x57, MRMSrcReg, (ops FR64:$dst),
               "xorpd $dst, $dst", [(set FR64:$dst, fp64imm0)]>,
             Requires<[HasSSE2]>, TB, OpSize;

let isTwoAddress = 1 in {
// SSE Scalar Arithmetic
let isCommutable = 1 in {
def ADDSSrr : I<0x58, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fadd FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def ADDSDrr : I<0x58, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fadd FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
def MULSSrr : I<0x59, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fmul FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def MULSDrr : I<0x59, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fmul FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
}

def ADDSSrm : I<0x58, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fadd FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def ADDSDrm : I<0x58, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fadd FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;
def MULSSrm : I<0x59, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fmul FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def MULSDrm : I<0x59, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fmul FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;

def DIVSSrr : I<0x5E, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fdiv FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def DIVSSrm : I<0x5E, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fdiv FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def DIVSDrr : I<0x5E, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fdiv FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
def DIVSDrm : I<0x5E, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fdiv FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;

def SUBSSrr : I<0x5C, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fsub FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def SUBSSrm : I<0x5C, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fsub FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def SUBSDrr : I<0x5C, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fsub FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
def SUBSDrm : I<0x5C, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fsub FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;

// SSE Logical
let isCommutable = 1 in {
def ANDPSrr : I<0x54, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "andps {$src2, $dst|$dst, $src2}", []>,
              Requires<[HasSSE1]>, TB;
def ANDPDrr : I<0x54, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "andpd {$src2, $dst|$dst, $src2}", []>,
              Requires<[HasSSE2]>, TB, OpSize;
def ORPSrr : I<0x56, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "orps {$src2, $dst|$dst, $src2}", []>,
             Requires<[HasSSE1]>, TB;
def ORPDrr : I<0x56, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "orpd {$src2, $dst|$dst, $src2}", []>,
             Requires<[HasSSE2]>, TB, OpSize;
def XORPSrr : I<0x57, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "xorps {$src2, $dst|$dst, $src2}", []>,
              Requires<[HasSSE1]>, TB;
def XORPDrr : I<0x57, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "xorpd {$src2, $dst|$dst, $src2}", []>,
              Requires<[HasSSE2]>, TB, OpSize;
}
def ANDNPSrr : I<0x55, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "andnps {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE1]>, TB;
def ANDNPDrr : I<0x55, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "andnpd {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE2]>, TB, OpSize;

def CMPSSrr : I<0xC2, MRMSrcReg, 
                (ops FR32:$dst, FR32:$src1, FR32:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XS;
def CMPSSrm : I<0xC2, MRMSrcMem, 
                (ops FR32:$dst, FR32:$src1, f32mem:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XS;
def CMPSDrr : I<0xC2, MRMSrcReg, 
                (ops FR64:$dst, FR64:$src1, FR64:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XD;
def CMPSDrm : I<0xC2, MRMSrcMem, 
                (ops FR64:$dst, FR64:$src1, f64mem:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE2]>, XD;
}

//===----------------------------------------------------------------------===//
// Floating Point Stack Support
//===----------------------------------------------------------------------===//

// Floating point support.  All FP Stack operations are represented with two 
// instructions here.  The first instruction, generated by the instruction
// selector, uses "RFP" registers: a traditional register file to reference
// floating point values.  These instructions are all psuedo instructions and
// use the "Fp" prefix.  The second instruction is defined with FPI, which is
// the actual instruction emitted by the assembler.  The FP stackifier pass
// converts one to the other after register allocation occurs.
//
// Note that the FpI instruction should have instruction selection info (e.g.
// a pattern) and the FPI instruction should have emission info (e.g. opcode
// encoding and asm printing info).

// FPI - Floating Point Instruction template.
class FPI<bits<8> o, Format F, dag ops, string asm> : I<o, F, ops, asm, []> {}

// FpI_ - Floating Point Psuedo Instruction template. Not Predicated.
class FpI_<dag ops, FPFormat fp, list<dag> pattern>
  : X86Inst<0, Pseudo, NoImm, ops, ""> {
  let FPForm = fp; let FPFormBits = FPForm.Value;
  let Pattern = pattern;
}

// Random Pseudo Instructions.
def FpGETRESULT : FpI_<(ops RFP:$dst), SpecialFP,
                  [(set RFP:$dst, X86fpget)]>;                    // FPR = ST(0)

let noResults = 1 in 
  def FpSETRESULT : FpI_<(ops RFP:$src), SpecialFP,
                        [(X86fpset RFP:$src)]>, Imp<[], [ST0]>;   // ST(0) = FPR

// FpI - Floating Point Psuedo Instruction template. Predicated on FPStack.
class FpI<dag ops, FPFormat fp, list<dag> pattern> :
  FpI_<ops, fp, pattern>, Requires<[FPStack]>;


def FpMOV       : FpI<(ops RFP:$dst, RFP:$src), SpecialFP, []>; // f1 = fmov f2

// Arithmetic
// Add, Sub, Mul, Div.
def FpADD : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), TwoArgFP,
                [(set RFP:$dst, (fadd RFP:$src1, RFP:$src2))]>;
def FpSUB : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), TwoArgFP,
                [(set RFP:$dst, (fsub RFP:$src1, RFP:$src2))]>;
def FpMUL : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), TwoArgFP,
                [(set RFP:$dst, (fmul RFP:$src1, RFP:$src2))]>;
def FpDIV : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), TwoArgFP,
                [(set RFP:$dst, (fdiv RFP:$src1, RFP:$src2))]>;

class FPST0rInst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (ops RST:$op), asm>, D8;
class FPrST0Inst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (ops RST:$op), asm>, DC;
class FPrST0PInst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (ops RST:$op), asm>, DE;

// Binary Ops with a memory source.
def FpADD32m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fadd RFP:$src1,
                                     (extloadf64f32 addr:$src2)))]>;
                // ST(0) = ST(0) + [mem32]
def FpADD64m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fadd RFP:$src1, (loadf64 addr:$src2)))]>;
                // ST(0) = ST(0) + [mem64]
def FpMUL32m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fmul RFP:$src1,
                                     (extloadf64f32 addr:$src2)))]>;
                // ST(0) = ST(0) * [mem32]
def FpMUL64m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fmul RFP:$src1, (loadf64 addr:$src2)))]>;
                // ST(0) = ST(0) * [mem64]
def FpSUB32m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fsub RFP:$src1,
                                    (extloadf64f32 addr:$src2)))]>;
                // ST(0) = ST(0) - [mem32]
def FpSUB64m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fsub RFP:$src1, (loadf64 addr:$src2)))]>;
                // ST(0) = ST(0) - [mem64]
def FpSUBR32m : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fsub (extloadf64f32 addr:$src2),
                                     RFP:$src1))]>;
                // ST(0) = [mem32] - ST(0)
def FpSUBR64m : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fsub (loadf64 addr:$src2), RFP:$src1))]>;
                // ST(0) = [mem64] - ST(0)
def FpDIV32m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fdiv RFP:$src1,
                                    (extloadf64f32 addr:$src2)))]>;
                // ST(0) = ST(0) / [mem32]
def FpDIV64m  : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fdiv RFP:$src1, (loadf64 addr:$src2)))]>;
                // ST(0) = ST(0) / [mem64]
def FpDIVR32m : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fdiv (extloadf64f32 addr:$src2),
                                     RFP:$src1))]>;
                // ST(0) = [mem32] / ST(0)
def FpDIVR64m : FpI<(ops RFP:$dst, RFP:$src1, f32mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fdiv (loadf64 addr:$src2), RFP:$src1))]>;
                // ST(0) = [mem64] / ST(0)


def FADD32m  : FPI<0xD8, MRM0m, (ops f32mem:$src), "fadd{s} $src">;
def FADD64m  : FPI<0xDC, MRM0m, (ops f64mem:$src), "fadd{l} $src">;
def FMUL32m  : FPI<0xD8, MRM1m, (ops f32mem:$src), "fmul{s} $src">;
def FMUL64m  : FPI<0xDC, MRM1m, (ops f64mem:$src), "fmul{l} $src">;
def FSUB32m  : FPI<0xD8, MRM4m, (ops f32mem:$src), "fsub{s} $src">;
def FSUB64m  : FPI<0xDC, MRM4m, (ops f64mem:$src), "fsub{l} $src">;
def FSUBR32m : FPI<0xD8, MRM5m, (ops f32mem:$src), "fsubr{s} $src">;
def FSUBR64m : FPI<0xDC, MRM5m, (ops f64mem:$src), "fsubr{l} $src">;
def FDIV32m  : FPI<0xD8, MRM6m, (ops f32mem:$src), "fdiv{s} $src">;
def FDIV64m  : FPI<0xDC, MRM6m, (ops f64mem:$src), "fdiv{l} $src">;
def FDIVR32m : FPI<0xD8, MRM7m, (ops f32mem:$src), "fdivr{s} $src">;
def FDIVR64m : FPI<0xDC, MRM7m, (ops f64mem:$src), "fdivr{l} $src">;

// FIXME: Implement these when we have a dag-dag isel!
def FpIADD16m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fadd RFP:$src1,
                                     (sint_to_fp (loadi16 addr:$src2))))]>;
                // ST(0) = ST(0) + [mem16int]
def FpIADD32m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fadd RFP:$src1,
                                     (sint_to_fp (loadi32 addr:$src2))))]>;
                // ST(0) = ST(0) + [mem32int]
def FpIMUL16m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fmul RFP:$src1,
                                     (sint_to_fp (loadi16 addr:$src2))))]>;
                // ST(0) = ST(0) * [mem16int]
def FpIMUL32m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fmul RFP:$src1,
                                     (sint_to_fp (loadi32 addr:$src2))))]>;
                // ST(0) = ST(0) * [mem32int]
def FpISUB16m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fsub RFP:$src1,
                                     (sint_to_fp (loadi16 addr:$src2))))]>;
                // ST(0) = ST(0) - [mem16int]
def FpISUB32m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fsub RFP:$src1,
                                     (sint_to_fp (loadi32 addr:$src2))))]>;
                // ST(0) = ST(0) - [mem32int]
def FpISUBR16m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                     [(set RFP:$dst, (fsub (sint_to_fp (loadi16 addr:$src2)),
                                      RFP:$src1))]>;
                // ST(0) = [mem16int] - ST(0)
def FpISUBR32m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                     [(set RFP:$dst, (fsub (sint_to_fp (loadi32 addr:$src2)),
                                      RFP:$src1))]>;
                // ST(0) = [mem32int] - ST(0)
def FpIDIV16m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fdiv RFP:$src1,
                                     (sint_to_fp (loadi16 addr:$src2))))]>;
                // ST(0) = ST(0) / [mem16int]
def FpIDIV32m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                    [(set RFP:$dst, (fdiv RFP:$src1,
                                     (sint_to_fp (loadi32 addr:$src2))))]>;
                // ST(0) = ST(0) / [mem32int]
def FpIDIVR16m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                     [(set RFP:$dst, (fdiv (sint_to_fp (loadi16 addr:$src2)),
                                      RFP:$src1))]>;
                // ST(0) = [mem16int] / ST(0)
def FpIDIVR32m : FpI<(ops RFP:$dst, RFP:$src1, i16mem:$src2), OneArgFPRW,
                     [(set RFP:$dst, (fdiv (sint_to_fp (loadi32 addr:$src2)),
                                      RFP:$src1))]>;
                // ST(0) = [mem32int] / ST(0)

def FIADD16m  : FPI<0xDE, MRM0m, (ops i16mem:$src), "fiadd{s} $src">;
def FIADD32m  : FPI<0xDA, MRM0m, (ops i32mem:$src), "fiadd{l} $src">;
def FIMUL16m  : FPI<0xDE, MRM1m, (ops i16mem:$src), "fimul{s} $src">;
def FIMUL32m  : FPI<0xDA, MRM1m, (ops i32mem:$src), "fimul{l} $src">;
def FISUB16m  : FPI<0xDE, MRM4m, (ops i16mem:$src), "fisub{s} $src">;
def FISUB32m  : FPI<0xDA, MRM4m, (ops i32mem:$src), "fisub{l} $src">;
def FISUBR16m : FPI<0xDE, MRM5m, (ops i16mem:$src), "fisubr{s} $src">;
def FISUBR32m : FPI<0xDA, MRM5m, (ops i32mem:$src), "fisubr{l} $src">;
def FIDIV16m  : FPI<0xDE, MRM6m, (ops i16mem:$src), "fidiv{s} $src">;
def FIDIV32m  : FPI<0xDA, MRM6m, (ops i32mem:$src), "fidiv{s} $src">;
def FIDIVR16m : FPI<0xDE, MRM7m, (ops i16mem:$src), "fidivr{s} $src">;
def FIDIVR32m : FPI<0xDA, MRM7m, (ops i32mem:$src), "fidivr{s} $src">;

// NOTE: GAS and apparently all other AT&T style assemblers have a broken notion
// of some of the 'reverse' forms of the fsub and fdiv instructions.  As such,
// we have to put some 'r's in and take them out of weird places.
def FADDST0r   : FPST0rInst <0xC0, "fadd $op">;
def FADDrST0   : FPrST0Inst <0xC0, "fadd {%ST(0), $op|$op, %ST(0)}">;
def FADDPrST0  : FPrST0PInst<0xC0, "faddp $op">;
def FSUBRST0r  : FPST0rInst <0xE8, "fsubr $op">;
def FSUBrST0   : FPrST0Inst <0xE8, "fsub{r} {%ST(0), $op|$op, %ST(0)}">;
def FSUBPrST0  : FPrST0PInst<0xE8, "fsub{r}p $op">;
def FSUBST0r   : FPST0rInst <0xE0, "fsub $op">;
def FSUBRrST0  : FPrST0Inst <0xE0, "fsub{|r} {%ST(0), $op|$op, %ST(0)}">;
def FSUBRPrST0 : FPrST0PInst<0xE0, "fsub{|r}p $op">;
def FMULST0r   : FPST0rInst <0xC8, "fmul $op">;
def FMULrST0   : FPrST0Inst <0xC8, "fmul {%ST(0), $op|$op, %ST(0)}">;
def FMULPrST0  : FPrST0PInst<0xC8, "fmulp $op">;
def FDIVRST0r  : FPST0rInst <0xF8, "fdivr $op">;
def FDIVrST0   : FPrST0Inst <0xF8, "fdiv{r} {%ST(0), $op|$op, %ST(0)}">;
def FDIVPrST0  : FPrST0PInst<0xF8, "fdiv{r}p $op">;
def FDIVST0r   : FPST0rInst <0xF0, "fdiv $op">;
def FDIVRrST0  : FPrST0Inst <0xF0, "fdiv{|r} {%ST(0), $op|$op, %ST(0)}">;
def FDIVRPrST0 : FPrST0PInst<0xF0, "fdiv{|r}p $op">;


// Unary operations.
def FpCHS  : FpI<(ops RFP:$dst, RFP:$src), OneArgFPRW,
                 [(set RFP:$dst, (fneg RFP:$src))]>;
def FpABS  : FpI<(ops RFP:$dst, RFP:$src), OneArgFPRW,
                 [(set RFP:$dst, (fabs RFP:$src))]>;
def FpSQRT : FpI<(ops RFP:$dst, RFP:$src), OneArgFPRW,
                 [(set RFP:$dst, (fsqrt RFP:$src))]>;
def FpSIN  : FpI<(ops RFP:$dst, RFP:$src), OneArgFPRW,
                 [(set RFP:$dst, (fsin RFP:$src))]>;
def FpCOS  : FpI<(ops RFP:$dst, RFP:$src), OneArgFPRW,
                 [(set RFP:$dst, (fcos RFP:$src))]>;
def FpTST  : FpI<(ops RFP:$src), OneArgFP,
                 []>;

def FCHS  : FPI<0xE0, RawFrm, (ops), "fchs">, D9;
def FABS  : FPI<0xE1, RawFrm, (ops), "fabs">, D9;
def FSQRT : FPI<0xFA, RawFrm, (ops), "fsqrt">, D9;
def FSIN  : FPI<0xFE, RawFrm, (ops), "fsin">, D9;
def FCOS  : FPI<0xFF, RawFrm, (ops), "fcos">, D9;
def FTST  : FPI<0xE4, RawFrm, (ops), "ftst">, D9;


// Floating point cmovs.
let isTwoAddress = 1 in {
  def FpCMOVB  : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_B, STATUS))]>;
  def FpCMOVBE : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_BE, STATUS))]>;
  def FpCMOVE  : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_E, STATUS))]>;
  def FpCMOVP  : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_P, STATUS))]>;
  def FpCMOVAE : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_AE, STATUS))]>;
  def FpCMOVA  : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_A, STATUS))]>;
  def FpCMOVNE : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_NE, STATUS))]>;
  def FpCMOVNP : FpI<(ops RFP:$dst, RFP:$src1, RFP:$src2), CondMovFP,
                     [(set RFP:$dst, (X86cmov RFP:$src1, RFP:$src2,
                                      X86_COND_NP, STATUS))]>;
}

def FCMOVB  : FPI<0xC0, AddRegFrm, (ops RST:$op),
                  "fcmovb {$op, %ST(0)|%ST(0), $op}">, DA;
def FCMOVBE : FPI<0xD0, AddRegFrm, (ops RST:$op),
                  "fcmovbe {$op, %ST(0)|%ST(0), $op}">, DA;
def FCMOVE  : FPI<0xC8, AddRegFrm, (ops RST:$op),
                  "fcmove {$op, %ST(0)|%ST(0), $op}">, DA;
def FCMOVP  : FPI<0xD8, AddRegFrm, (ops RST:$op),
                  "fcmovu  {$op, %ST(0)|%ST(0), $op}">, DA;
def FCMOVAE : FPI<0xC0, AddRegFrm, (ops RST:$op),
                  "fcmovae {$op, %ST(0)|%ST(0), $op}">, DB;
def FCMOVA  : FPI<0xD0, AddRegFrm, (ops RST:$op),
                  "fcmova {$op, %ST(0)|%ST(0), $op}">, DB;
def FCMOVNE : FPI<0xC8, AddRegFrm, (ops RST:$op),
                  "fcmovne {$op, %ST(0)|%ST(0), $op}">, DB;
def FCMOVNP : FPI<0xD8, AddRegFrm, (ops RST:$op),
                  "fcmovnu {$op, %ST(0)|%ST(0), $op}">, DB;

// Floating point loads & stores.
def FpLD32m  : FpI<(ops RFP:$dst, f32mem:$src), ZeroArgFP,
                   [(set RFP:$dst, (extloadf64f32 addr:$src))]>;
def FpLD64m  : FpI<(ops RFP:$dst, f64mem:$src), ZeroArgFP,
                   [(set RFP:$dst, (loadf64 addr:$src))]>;
def FpILD16m : FpI<(ops RFP:$dst, i16mem:$src), ZeroArgFP,
                   [(set RFP:$dst, (sint_to_fp (loadi16 addr:$src)))]>;
def FpILD32m : FpI<(ops RFP:$dst, i32mem:$src), ZeroArgFP,
                   [(set RFP:$dst, (sint_to_fp (loadi32 addr:$src)))]>;
def FpILD64m : FpI<(ops RFP:$dst, i64mem:$src), ZeroArgFP,
                   [(set RFP:$dst, (X86fild64m addr:$src))]>;

def FpST32m   : FpI<(ops f32mem:$op, RFP:$src), OneArgFP,
                [(truncstore RFP:$src, addr:$op, f32)]>;
def FpST64m   : FpI<(ops f64mem:$op, RFP:$src), OneArgFP,
                [(store RFP:$src, addr:$op)]>;

def FpSTP32m  : FpI<(ops f32mem:$op, RFP:$src), OneArgFP, []>;
def FpSTP64m  : FpI<(ops f64mem:$op, RFP:$src), OneArgFP, []>;
def FpIST16m  : FpI<(ops i16mem:$op, RFP:$src), OneArgFP, []>;
def FpIST32m  : FpI<(ops i32mem:$op, RFP:$src), OneArgFP, []>;
def FpIST64m  : FpI<(ops i64mem:$op, RFP:$src), OneArgFP, []>;

def FLD32m   : FPI<0xD9, MRM0m, (ops f32mem:$src), "fld{s} $src">;
def FLD64m   : FPI<0xDD, MRM0m, (ops f64mem:$src), "fld{l} $src">;
def FILD16m  : FPI<0xDF, MRM0m, (ops i16mem:$src), "fild{s} $src">;
def FILD32m  : FPI<0xDB, MRM0m, (ops i32mem:$src), "fild{l} $src">;
def FILD64m  : FPI<0xDF, MRM5m, (ops i64mem:$src), "fild{ll} $src">;
def FST32m   : FPI<0xD9, MRM2m, (ops f32mem:$dst), "fst{s} $dst">;
def FST64m   : FPI<0xDD, MRM2m, (ops f64mem:$dst), "fst{l} $dst">;
def FSTP32m  : FPI<0xD9, MRM3m, (ops f32mem:$dst), "fstp{s} $dst">;
def FSTP64m  : FPI<0xDD, MRM3m, (ops f64mem:$dst), "fstp{l} $dst">;
def FIST16m  : FPI<0xDF, MRM2m, (ops i16mem:$dst), "fist{s} $dst">;
def FIST32m  : FPI<0xDB, MRM2m, (ops i32mem:$dst), "fist{l} $dst">;
def FISTP16m : FPI<0xDF, MRM3m, (ops i16mem:$dst), "fistp{s} $dst">;
def FISTP32m : FPI<0xDB, MRM3m, (ops i32mem:$dst), "fistp{l} $dst">;
def FISTP64m : FPI<0xDF, MRM7m, (ops i64mem:$dst), "fistp{ll} $dst">;

// FP Stack manipulation instructions.
def FLDrr   : FPI<0xC0, AddRegFrm, (ops RST:$op), "fld $op">, D9;
def FSTrr   : FPI<0xD0, AddRegFrm, (ops RST:$op), "fst $op">, DD;
def FSTPrr  : FPI<0xD8, AddRegFrm, (ops RST:$op), "fstp $op">, DD;
def FXCH    : FPI<0xC8, AddRegFrm, (ops RST:$op), "fxch $op">, D9;

// Floating point constant loads.
def FpLD0 : FpI<(ops RFP:$dst), ZeroArgFP,
                [(set RFP:$dst, fp64imm0)]>;
def FpLD1 : FpI<(ops RFP:$dst), ZeroArgFP,
                [(set RFP:$dst, fp64imm1)]>;

def FLD0 : FPI<0xEE, RawFrm, (ops), "fldz">, D9;
def FLD1 : FPI<0xE8, RawFrm, (ops), "fld1">, D9;


// Floating point compares.
def FpUCOMr   : FpI<(ops RFP:$lhs, RFP:$rhs), CompareFP,
                    []>;  // FPSW = cmp ST(0) with ST(i)
def FpUCOMIr  : FpI<(ops RFP:$lhs, RFP:$rhs), CompareFP,
                    [(set STATUS, (X86cmp RFP:$lhs, RFP:$rhs))]>,
                Imp<[],[STATUS]>;       // CC = cmp ST(0) with ST(i)

def FUCOMr    : FPI<0xE0, AddRegFrm,    // FPSW = cmp ST(0) with ST(i)
                    (ops RST:$reg),
                    "fucom $reg">, DD, Imp<[ST0],[]>;
def FUCOMPr   : FPI<0xE8, AddRegFrm,    // FPSW = cmp ST(0) with ST(i), pop
                  (ops RST:$reg),
                  "fucomp $reg">, DD, Imp<[ST0],[]>;
def FUCOMPPr  : FPI<0xE9, RawFrm,       // cmp ST(0) with ST(1), pop, pop
                  (ops),
                  "fucompp">, DA, Imp<[ST0],[]>;

def FUCOMIr  : FPI<0xE8, AddRegFrm,     // CC = cmp ST(0) with ST(i)
                   (ops RST:$reg),
                   "fucomi {$reg, %ST(0)|%ST(0), $reg}">, DB, Imp<[ST0],[]>;
def FUCOMIPr : FPI<0xE8, AddRegFrm,     // CC = cmp ST(0) with ST(i), pop
                 (ops RST:$reg),
                 "fucomip {$reg, %ST(0)|%ST(0), $reg}">, DF, Imp<[ST0],[]>;


// Floating point flag ops.
def FNSTSW8r  : I<0xE0, RawFrm,                  // AX = fp flags
                  (ops), "fnstsw", []>, DF, Imp<[],[AX]>;

def FNSTCW16m : I<0xD9, MRM7m,                   // [mem16] = X87 control world
                  (ops i16mem:$dst), "fnstcw $dst", []>;
def FLDCW16m  : I<0xD9, MRM5m,                   // X87 control world = [mem16]
                  (ops i16mem:$dst), "fldcw $dst", []>;


//===----------------------------------------------------------------------===//
// Miscellaneous Instructions
//===----------------------------------------------------------------------===//

def RDTSC : I<0x31, RawFrm, (ops), "rdtsc", [(X86rdtsc)]>,
            TB, Imp<[],[EAX,EDX]>;


//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Calls
def : Pat<(X86call tglobaladdr:$dst),
          (CALLpcrel32 tglobaladdr:$dst)>;
def : Pat<(X86call externalsym:$dst),
          (CALLpcrel32 externalsym:$dst)>;

// X86 specific add which produces a flag.
def : Pat<(X86addflag R32:$src1, R32:$src2),
          (ADD32rr R32:$src1, R32:$src2)>;
def : Pat<(X86addflag R32:$src1, (load addr:$src2)),
          (ADD32rm R32:$src1, addr:$src2)>;
def : Pat<(X86addflag R32:$src1, imm:$src2),
          (ADD32ri R32:$src1, imm:$src2)>;
def : Pat<(X86addflag R32:$src1, i32immSExt8:$src2),
          (ADD32ri8 R32:$src1, i32immSExt8:$src2)>;

def : Pat<(X86subflag R32:$src1, R32:$src2),
          (SUB32rr R32:$src1, R32:$src2)>;
def : Pat<(X86subflag R32:$src1, (load addr:$src2)),
          (SUB32rm R32:$src1, addr:$src2)>;
def : Pat<(X86subflag R32:$src1, imm:$src2),
          (SUB32ri R32:$src1, imm:$src2)>;
def : Pat<(X86subflag R32:$src1, i32immSExt8:$src2),
          (SUB32ri8 R32:$src1, i32immSExt8:$src2)>;

// {s|z}extload bool -> {s|z}extload byte
def : Pat<(sextloadi16i1 addr:$src), (MOVSX16rm8 addr:$src)>;
def : Pat<(sextloadi32i1 addr:$src), (MOVSX32rm8 addr:$src)>;
def : Pat<(zextloadi16i1 addr:$src), (MOVZX16rm8 addr:$src)>;
def : Pat<(zextloadi32i1 addr:$src), (MOVZX32rm8 addr:$src)>;

// extload bool -> extload byte
def : Pat<(extloadi8i1 addr:$src), (MOV8rm addr:$src)>;

// anyext -> zext
def : Pat<(i16 (anyext R8 :$src)), (MOVZX16rr8  R8 :$src)>;
def : Pat<(i32 (anyext R8 :$src)), (MOVZX32rr8  R8 :$src)>;
def : Pat<(i32 (anyext R16:$src)), (MOVZX32rr16 R16:$src)>;

// Required for RET of f32 / f64 values.
def : Pat<(X86fld addr:$src, f32), (FpLD32m addr:$src)>;
def : Pat<(X86fld addr:$src, f64), (FpLD64m addr:$src)>;

// Required for CALL which return f32 / f64 values.
def : Pat<(X86fst RFP:$src, addr:$op, f32), (FpST32m addr:$op, RFP:$src)>;
def : Pat<(X86fst RFP:$src, addr:$op, f64), (FpST64m addr:$op, RFP:$src)>;

// Floatin point constant -0.0 and -1.0
def : Pat<(f64 fp64immneg0), (FpCHS (FpLD0))>, Requires<[FPStack]>;
def : Pat<(f64 fp64immneg1), (FpCHS (FpLD1))>, Requires<[FPStack]>;

// RFP undef
def : Pat<(f64 (undef)), (FpLD0)>,  Requires<[FPStack]>;


//===----------------------------------------------------------------------===//
// Some peepholes
//===----------------------------------------------------------------------===//

// (shl x, 1) ==> (add x, x)
def : Pat<(shl R8 :$src1, (i8 1)), (ADD8rr  R8 :$src1, R8 :$src1)>;
def : Pat<(shl R16:$src1, (i8 1)), (ADD16rr R16:$src1, R16:$src1)>;
def : Pat<(shl R32:$src1, (i8 1)), (ADD32rr R32:$src1, R32:$src1)>;
