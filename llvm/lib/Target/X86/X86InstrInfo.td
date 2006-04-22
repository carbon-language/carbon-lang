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

def SDTX86CmpTest : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;

def SDTX86Cmov    : SDTypeProfile<1, 3,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                   SDTCisVT<3, i8>]>;

def SDTX86BrCond  : SDTypeProfile<0, 2,
                                  [SDTCisVT<0, OtherVT>, SDTCisVT<1, i8>]>;

def SDTX86SetCC   : SDTypeProfile<1, 1,
                                  [SDTCisVT<0, i8>, SDTCisVT<1, i8>]>;

def SDTX86Ret     : SDTypeProfile<0, 1, [SDTCisVT<0, i16>]>;

def SDT_X86CallSeqStart : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;
def SDT_X86CallSeqEnd   : SDTypeProfile<0, 2, [ SDTCisVT<0, i32>,
                                                SDTCisVT<1, i32> ]>;

def SDT_X86Call   : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

def SDTX86RepStr  : SDTypeProfile<0, 1, [SDTCisVT<0, OtherVT>]>;

def SDTX86RdTsc   : SDTypeProfile<0, 0, []>;

def SDTX86Wrapper : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

def X86shld    : SDNode<"X86ISD::SHLD",     SDTIntShiftDOp>;
def X86shrd    : SDNode<"X86ISD::SHRD",     SDTIntShiftDOp>;

def X86cmp     : SDNode<"X86ISD::CMP" ,     SDTX86CmpTest,
                        [SDNPOutFlag]>;
def X86test    : SDNode<"X86ISD::TEST",     SDTX86CmpTest,
                        [SDNPOutFlag]>;

def X86cmov    : SDNode<"X86ISD::CMOV",     SDTX86Cmov,    
                        [SDNPInFlag, SDNPOutFlag]>;
def X86brcond  : SDNode<"X86ISD::BRCOND",   SDTX86BrCond,
                        [SDNPHasChain, SDNPInFlag]>;
def X86setcc   : SDNode<"X86ISD::SETCC",    SDTX86SetCC,
                        [SDNPInFlag, SDNPOutFlag]>;

def X86retflag : SDNode<"X86ISD::RET_FLAG", SDTX86Ret,
                        [SDNPHasChain, SDNPOptInFlag]>;

def X86callseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_X86CallSeqStart,
                        [SDNPHasChain]>;
def X86callseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_X86CallSeqEnd,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

def X86call    : SDNode<"X86ISD::CALL",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;

def X86rep_stos: SDNode<"X86ISD::REP_STOS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;
def X86rep_movs: SDNode<"X86ISD::REP_MOVS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag]>;

def X86rdtsc   : SDNode<"X86ISD::RDTSC_DAG",SDTX86RdTsc,
                        [SDNPHasChain, SDNPOutFlag]>;

def X86Wrapper : SDNode<"X86ISD::Wrapper",  SDTX86Wrapper>;

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
def i128mem : X86MemOperand<"printi128mem">;
def f32mem  : X86MemOperand<"printf32mem">;
def f64mem  : X86MemOperand<"printf64mem">;
def f128mem : X86MemOperand<"printf128mem">;

def SSECC : Operand<i8> {
  let PrintMethod = "printSSECC";
}

def piclabel: Operand<i32> {
  let PrintMethod = "printPICLabel";
}

// A couple of more descriptive operand definitions.
// 16-bits but only 8 bits are significant.
def i16i8imm  : Operand<i16>;
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32>;

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;

//===----------------------------------------------------------------------===//
// X86 Complex Pattern Definitions.
//

// Define X86 specific addressing mode.
def addr    : ComplexPattern<i32, 4, "SelectAddr", []>;
def leaaddr : ComplexPattern<i32, 4, "SelectLEAAddr",
                             [add, mul, shl, frameindex]>;

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<6> val> {
  bits<6> Value = val;
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
def MRMInitReg : Format<32>;

//===----------------------------------------------------------------------===//
// X86 Instruction Predicate Definitions.
def HasMMX  : Predicate<"Subtarget->hasMMX()">;
def HasSSE1 : Predicate<"Subtarget->hasSSE1()">;
def HasSSE2 : Predicate<"Subtarget->hasSSE2()">;
def HasSSE3 : Predicate<"Subtarget->hasSSE3()">;
def FPStack : Predicate<"!Subtarget->hasSSE2()">;

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
  bits<6> FormBits = Form.Value;
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

// Helper fragments for loads.
def loadi8  : PatFrag<(ops node:$ptr), (i8  (load node:$ptr))>;
def loadi16 : PatFrag<(ops node:$ptr), (i16 (load node:$ptr))>;
def loadi32 : PatFrag<(ops node:$ptr), (i32 (load node:$ptr))>;
def loadi64 : PatFrag<(ops node:$ptr), (i64 (load node:$ptr))>;

def loadf32 : PatFrag<(ops node:$ptr), (f32 (load node:$ptr))>;
def loadf64 : PatFrag<(ops node:$ptr), (f64 (load node:$ptr))>;

def sextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (sextload node:$ptr, i1))>;
def sextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (sextload node:$ptr, i1))>;
def sextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (sextload node:$ptr, i8))>;
def sextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (sextload node:$ptr, i8))>;
def sextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (sextload node:$ptr, i16))>;

def zextloadi8i1   : PatFrag<(ops node:$ptr), (i8  (zextload node:$ptr, i1))>;
def zextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (zextload node:$ptr, i1))>;
def zextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (zextload node:$ptr, i1))>;
def zextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (zextload node:$ptr, i8))>;
def zextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (zextload node:$ptr, i8))>;
def zextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (zextload node:$ptr, i16))>;

def extloadi8i1    : PatFrag<(ops node:$ptr), (i8  (extload node:$ptr, i1))>;

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

// Indirect branches
let isBarrier = 1 in
  def JMP : IBr<0xE9, (ops brtarget:$dst), "jmp $dst", [(br bb:$dst)]>;

let isBranch = 1, isTerminator = 1, noResults = 1, isBarrier = 1 in {
  def JMP32r     : I<0xFF, MRM4r, (ops R32:$dst), "jmp{l} {*}$dst",
                     [(brind R32:$dst)]>;
  def JMP32m     : I<0xFF, MRM4m, (ops i32mem:$dst), "jmp{l} {*}$dst",
                     [(brind (loadi32 addr:$dst))]>;
}

// Conditional branches
def JE  : IBr<0x84, (ops brtarget:$dst), "je $dst",
              [(X86brcond bb:$dst, X86_COND_E)]>, TB;
def JNE : IBr<0x85, (ops brtarget:$dst), "jne $dst",
              [(X86brcond bb:$dst, X86_COND_NE)]>, TB;
def JL  : IBr<0x8C, (ops brtarget:$dst), "jl $dst",
              [(X86brcond bb:$dst, X86_COND_L)]>, TB;
def JLE : IBr<0x8E, (ops brtarget:$dst), "jle $dst",
              [(X86brcond bb:$dst, X86_COND_LE)]>, TB;
def JG  : IBr<0x8F, (ops brtarget:$dst), "jg $dst",
              [(X86brcond bb:$dst, X86_COND_G)]>, TB;
def JGE : IBr<0x8D, (ops brtarget:$dst), "jge $dst",
              [(X86brcond bb:$dst, X86_COND_GE)]>, TB;

def JB  : IBr<0x82, (ops brtarget:$dst), "jb $dst",
              [(X86brcond bb:$dst, X86_COND_B)]>, TB;
def JBE : IBr<0x86, (ops brtarget:$dst), "jbe $dst",
              [(X86brcond bb:$dst, X86_COND_BE)]>, TB;
def JA  : IBr<0x87, (ops brtarget:$dst), "ja $dst",
              [(X86brcond bb:$dst, X86_COND_A)]>, TB;
def JAE : IBr<0x83, (ops brtarget:$dst), "jae $dst",
              [(X86brcond bb:$dst, X86_COND_AE)]>, TB;

def JS  : IBr<0x88, (ops brtarget:$dst), "js $dst",
              [(X86brcond bb:$dst, X86_COND_S)]>, TB;
def JNS : IBr<0x89, (ops brtarget:$dst), "jns $dst",
              [(X86brcond bb:$dst, X86_COND_NS)]>, TB;
def JP  : IBr<0x8A, (ops brtarget:$dst), "jp $dst",
              [(X86brcond bb:$dst, X86_COND_P)]>, TB;
def JNP : IBr<0x8B, (ops brtarget:$dst), "jnp $dst",
              [(X86brcond bb:$dst, X86_COND_NP)]>, TB;
def JO  : IBr<0x80, (ops brtarget:$dst), "jo $dst",
              [(X86brcond bb:$dst, X86_COND_O)]>, TB;
def JNO : IBr<0x81, (ops brtarget:$dst), "jno $dst",
              [(X86brcond bb:$dst, X86_COND_NO)]>, TB;

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1, noResults = 1 in
  // All calls clobber the non-callee saved registers...
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7] in {
    def CALLpcrel32 : I<0xE8, RawFrm, (ops i32imm:$dst), "call ${dst:call}",
                      []>;
    def CALL32r     : I<0xFF, MRM2r, (ops R32:$dst), "call {*}$dst",
                      [(X86call R32:$dst)]>;
    def CALL32m     : I<0xFF, MRM2m, (ops i32mem:$dst), "call {*}$dst",
                      [(X86call (loadi32 addr:$dst))]>;
  }

// Tail call stuff.
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, noResults = 1 in
  def TAILJMPd : IBr<0xE9, (ops i32imm:$dst), "jmp ${dst:call}  # TAIL CALL", []>;
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

def MovePCtoStack : I<0, Pseudo, (ops piclabel:$label),
                      "call $label", []>;

let isTwoAddress = 1 in                               // R32 = bswap R32
  def BSWAP32r : I<0xC8, AddRegFrm,
                   (ops R32:$dst, R32:$src),
                   "bswap{l} $dst", 
                   [(set R32:$dst, (bswap R32:$src))]>, TB;

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

def REP_MOVSB : I<0xA4, RawFrm, (ops), "{rep;movsb|rep movsb}",
                  [(X86rep_movs i8)]>,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP;
def REP_MOVSW : I<0xA5, RawFrm, (ops), "{rep;movsw|rep movsw}",
                  [(X86rep_movs i16)]>,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP, OpSize;
def REP_MOVSD : I<0xA5, RawFrm, (ops), "{rep;movsd|rep movsd}",
                  [(X86rep_movs i32)]>,
                Imp<[ECX,EDI,ESI], [ECX,EDI,ESI]>, REP;

def REP_STOSB : I<0xAA, RawFrm, (ops), "{rep;stosb|rep stosb}",
                  [(X86rep_stos i8)]>,
                Imp<[AL,ECX,EDI], [ECX,EDI]>, REP;
def REP_STOSW : I<0xAB, RawFrm, (ops), "{rep;stosw|rep stosw}",
                  [(X86rep_stos i16)]>,
                Imp<[AX,ECX,EDI], [ECX,EDI]>, REP, OpSize;
def REP_STOSD : I<0xAB, RawFrm, (ops), "{rep;stosl|rep stosd}",
                  [(X86rep_stos i32)]>,
                Imp<[EAX,ECX,EDI], [ECX,EDI]>, REP;


//===----------------------------------------------------------------------===//
//  Input/Output Instructions...
//
def IN8rr  : I<0xEC, RawFrm, (ops),
               "in{b} {%dx, %al|%AL, %DX}",
               []>,  Imp<[DX], [AL]>;
def IN16rr : I<0xED, RawFrm, (ops),
               "in{w} {%dx, %ax|%AX, %DX}",
               []>,  Imp<[DX], [AX]>, OpSize;
def IN32rr : I<0xED, RawFrm, (ops),
               "in{l} {%dx, %eax|%EAX, %DX}",
               []>, Imp<[DX],[EAX]>;

def IN8ri  : Ii8<0xE4, RawFrm, (ops i16i8imm:$port),
                  "in{b} {$port, %al|%AL, $port}",
                 []>,
             Imp<[], [AL]>;
def IN16ri : Ii8<0xE5, RawFrm, (ops i16i8imm:$port),
                  "in{w} {$port, %ax|%AX, $port}",
                 []>,
             Imp<[], [AX]>, OpSize;
def IN32ri : Ii8<0xE5, RawFrm, (ops i16i8imm:$port),
                  "in{l} {$port, %eax|%EAX, $port}",
                 []>,
             Imp<[],[EAX]>;

def OUT8rr  : I<0xEE, RawFrm, (ops),
                "out{b} {%al, %dx|%DX, %AL}",
                []>,  Imp<[DX,  AL], []>;
def OUT16rr : I<0xEF, RawFrm, (ops),
                "out{w} {%ax, %dx|%DX, %AX}",
                []>,  Imp<[DX,  AX], []>, OpSize;
def OUT32rr : I<0xEF, RawFrm, (ops),
                "out{l} {%eax, %dx|%DX, %EAX}",
                []>, Imp<[DX, EAX], []>;

def OUT8ir  : Ii8<0xE6, RawFrm, (ops i16i8imm:$port),
                   "out{b} {%al, $port|$port, %AL}",
                   []>,
              Imp<[AL], []>;
def OUT16ir : Ii8<0xE7, RawFrm, (ops i16i8imm:$port),
                   "out{w} {%ax, $port|$port, %AX}",
                   []>,
              Imp<[AX], []>, OpSize;
def OUT32ir : Ii8<0xE7, RawFrm, (ops i16i8imm:$port),
                   "out{l} {%eax, $port|$port, %EAX}",
                   []>,
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
def MUL8r  : I<0xF6, MRM4r, (ops R8:$src), "mul{b} $src",
               // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
               // This probably ought to be moved to a def : Pat<> if the
               // syntax can be accepted.
               [(set AL, (mul AL, R8:$src))]>,
             Imp<[AL],[AX]>;               // AL,AH = AL*R8
def MUL16r : I<0xF7, MRM4r, (ops R16:$src), "mul{w} $src", []>,
             Imp<[AX],[AX,DX]>, OpSize;    // AX,DX = AX*R16
def MUL32r : I<0xF7, MRM4r, (ops R32:$src), "mul{l} $src", []>,
             Imp<[EAX],[EAX,EDX]>;         // EAX,EDX = EAX*R32
def MUL8m  : I<0xF6, MRM4m, (ops i8mem :$src),
               "mul{b} $src",
               // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
               // This probably ought to be moved to a def : Pat<> if the
               // syntax can be accepted.
               [(set AL, (mul AL, (loadi8 addr:$src)))]>,
             Imp<[AL],[AX]>;          // AL,AH = AL*[mem8]
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
                                   X86_COND_B))]>,
                  TB, OpSize;
def CMOVB16rm : I<0x42, MRMSrcMem,       // if <u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_B))]>,
                  TB, OpSize;
def CMOVB32rr : I<0x42, MRMSrcReg,       // if <u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_B))]>,
                   TB;
def CMOVB32rm : I<0x42, MRMSrcMem,       // if <u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovb {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_B))]>,
                   TB;

def CMOVAE16rr: I<0x43, MRMSrcReg,       // if >=u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_AE))]>,
                   TB, OpSize;
def CMOVAE16rm: I<0x43, MRMSrcMem,       // if >=u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_AE))]>,
                   TB, OpSize;
def CMOVAE32rr: I<0x43, MRMSrcReg,       // if >=u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_AE))]>,
                   TB;
def CMOVAE32rm: I<0x43, MRMSrcMem,       // if >=u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovae {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_AE))]>,
                   TB;

def CMOVE16rr : I<0x44, MRMSrcReg,       // if ==, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_E))]>,
                   TB, OpSize;
def CMOVE16rm : I<0x44, MRMSrcMem,       // if ==, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_E))]>,
                   TB, OpSize;
def CMOVE32rr : I<0x44, MRMSrcReg,       // if ==, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_E))]>,
                   TB;
def CMOVE32rm : I<0x44, MRMSrcMem,       // if ==, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmove {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_E))]>,
                   TB;

def CMOVNE16rr: I<0x45, MRMSrcReg,       // if !=, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_NE))]>,
                   TB, OpSize;
def CMOVNE16rm: I<0x45, MRMSrcMem,       // if !=, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NE))]>,
                   TB, OpSize;
def CMOVNE32rr: I<0x45, MRMSrcReg,       // if !=, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_NE))]>,
                   TB;
def CMOVNE32rm: I<0x45, MRMSrcMem,       // if !=, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovne {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NE))]>,
                   TB;

def CMOVBE16rr: I<0x46, MRMSrcReg,       // if <=u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_BE))]>,
                   TB, OpSize;
def CMOVBE16rm: I<0x46, MRMSrcMem,       // if <=u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_BE))]>,
                   TB, OpSize;
def CMOVBE32rr: I<0x46, MRMSrcReg,       // if <=u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_BE))]>,
                   TB;
def CMOVBE32rm: I<0x46, MRMSrcMem,       // if <=u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovbe {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_BE))]>,
                   TB;

def CMOVA16rr : I<0x47, MRMSrcReg,       // if >u, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_A))]>,
                   TB, OpSize;
def CMOVA16rm : I<0x47, MRMSrcMem,       // if >u, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_A))]>,
                   TB, OpSize;
def CMOVA32rr : I<0x47, MRMSrcReg,       // if >u, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_A))]>,
                   TB;
def CMOVA32rm : I<0x47, MRMSrcMem,       // if >u, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmova {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_A))]>,
                   TB;

def CMOVL16rr : I<0x4C, MRMSrcReg,       // if <s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_L))]>,
                   TB, OpSize;
def CMOVL16rm : I<0x4C, MRMSrcMem,       // if <s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_L))]>,
                   TB, OpSize;
def CMOVL32rr : I<0x4C, MRMSrcReg,       // if <s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_L))]>,
                   TB;
def CMOVL32rm : I<0x4C, MRMSrcMem,       // if <s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovl {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_L))]>,
                   TB;

def CMOVGE16rr: I<0x4D, MRMSrcReg,       // if >=s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_GE))]>,
                   TB, OpSize;
def CMOVGE16rm: I<0x4D, MRMSrcMem,       // if >=s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_GE))]>,
                   TB, OpSize;
def CMOVGE32rr: I<0x4D, MRMSrcReg,       // if >=s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_GE))]>,
                   TB;
def CMOVGE32rm: I<0x4D, MRMSrcMem,       // if >=s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovge {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_GE))]>,
                   TB;

def CMOVLE16rr: I<0x4E, MRMSrcReg,       // if <=s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_LE))]>,
                   TB, OpSize;
def CMOVLE16rm: I<0x4E, MRMSrcMem,       // if <=s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_LE))]>,
                   TB, OpSize;
def CMOVLE32rr: I<0x4E, MRMSrcReg,       // if <=s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_LE))]>,
                   TB;
def CMOVLE32rm: I<0x4E, MRMSrcMem,       // if <=s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovle {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_LE))]>,
                   TB;

def CMOVG16rr : I<0x4F, MRMSrcReg,       // if >s, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_G))]>,
                   TB, OpSize;
def CMOVG16rm : I<0x4F, MRMSrcMem,       // if >s, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_G))]>,
                   TB, OpSize;
def CMOVG32rr : I<0x4F, MRMSrcReg,       // if >s, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_G))]>,
                   TB;
def CMOVG32rm : I<0x4F, MRMSrcMem,       // if >s, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovg {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_G))]>,
                   TB;

def CMOVS16rr : I<0x48, MRMSrcReg,       // if signed, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_S))]>,
                  TB, OpSize;
def CMOVS16rm : I<0x48, MRMSrcMem,       // if signed, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_S))]>,
                  TB, OpSize;
def CMOVS32rr : I<0x48, MRMSrcReg,       // if signed, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_S))]>,
                  TB;
def CMOVS32rm : I<0x48, MRMSrcMem,       // if signed, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovs {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_S))]>,
                  TB;

def CMOVNS16rr: I<0x49, MRMSrcReg,       // if !signed, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_NS))]>,
                  TB, OpSize;
def CMOVNS16rm: I<0x49, MRMSrcMem,       // if !signed, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NS))]>,
                  TB, OpSize;
def CMOVNS32rr: I<0x49, MRMSrcReg,       // if !signed, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_NS))]>,
                  TB;
def CMOVNS32rm: I<0x49, MRMSrcMem,       // if !signed, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovns {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NS))]>,
                  TB;

def CMOVP16rr : I<0x4A, MRMSrcReg,       // if parity, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                   X86_COND_P))]>,
                  TB, OpSize;
def CMOVP16rm : I<0x4A, MRMSrcMem,       // if parity, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                   X86_COND_P))]>,
                  TB, OpSize;
def CMOVP32rr : I<0x4A, MRMSrcReg,       // if parity, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                   X86_COND_P))]>,
                  TB;
def CMOVP32rm : I<0x4A, MRMSrcMem,       // if parity, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovp {$src2, $dst|$dst, $src2}",
                  [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                   X86_COND_P))]>,
                  TB;

def CMOVNP16rr : I<0x4B, MRMSrcReg,       // if !parity, R16 = R16
                  (ops R16:$dst, R16:$src1, R16:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (X86cmov R16:$src1, R16:$src2,
                                    X86_COND_NP))]>,
                  TB, OpSize;
def CMOVNP16rm : I<0x4B, MRMSrcMem,       // if !parity, R16 = [mem16]
                  (ops R16:$dst, R16:$src1, i16mem:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (X86cmov R16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NP))]>,
                  TB, OpSize;
def CMOVNP32rr : I<0x4B, MRMSrcReg,       // if !parity, R32 = R32
                  (ops R32:$dst, R32:$src1, R32:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (X86cmov R32:$src1, R32:$src2,
                                    X86_COND_NP))]>,
                  TB;
def CMOVNP32rm : I<0x4B, MRMSrcMem,       // if !parity, R32 = [mem32]
                  (ops R32:$dst, R32:$src1, i32mem:$src2),
                  "cmovnp {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (X86cmov R32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NP))]>,
                  TB;


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
                [(store (and (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
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
                 "rol{b} {%cl, $dst|$dst, %CL}",
                 [(set R8:$dst, (rotl R8:$src, CL))]>, Imp<[CL],[]>;
def ROL16rCL : I<0xD3, MRM0r, (ops R16:$dst, R16:$src),
                 "rol{w} {%cl, $dst|$dst, %CL}",
                 [(set R16:$dst, (rotl R16:$src, CL))]>, Imp<[CL],[]>, OpSize;
def ROL32rCL : I<0xD3, MRM0r, (ops R32:$dst, R32:$src),
                 "rol{l} {%cl, $dst|$dst, %CL}",
                 [(set R32:$dst, (rotl R32:$src, CL))]>, Imp<[CL],[]>;

def ROL8ri   : Ii8<0xC0, MRM0r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "rol{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (rotl R8:$src1, (i8 imm:$src2)))]>;
def ROL16ri  : Ii8<0xC1, MRM0r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "rol{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (rotl R16:$src1, (i8 imm:$src2)))]>, OpSize;
def ROL32ri  : Ii8<0xC1, MRM0r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "rol{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (rotl R32:$src1, (i8 imm:$src2)))]>;

let isTwoAddress = 0 in {
  def ROL8mCL  : I<0xD2, MRM0m, (ops i8mem :$dst),
                   "rol{b} {%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi8 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def ROL16mCL : I<0xD3, MRM0m, (ops i16mem:$dst),
                   "rol{w} {%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi16 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>, OpSize;
  def ROL32mCL : I<0xD3, MRM0m, (ops i32mem:$dst),
                   "rol{l} {%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi32 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def ROL8mi   : Ii8<0xC0, MRM0m, (ops i8mem :$dst, i8imm:$src),
                     "rol{b} {$src, $dst|$dst, $src}",
                 [(store (rotl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def ROL16mi  : Ii8<0xC1, MRM0m, (ops i16mem:$dst, i8imm:$src),
                     "rol{w} {$src, $dst|$dst, $src}",
                [(store (rotl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def ROL32mi  : Ii8<0xC1, MRM0m, (ops i32mem:$dst, i8imm:$src),
                     "rol{l} {$src, $dst|$dst, $src}",
                [(store (rotl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
}

def ROR8rCL  : I<0xD2, MRM1r, (ops R8 :$dst, R8 :$src),
                 "ror{b} {%cl, $dst|$dst, %CL}",
                 [(set R8:$dst, (rotr R8:$src, CL))]>, Imp<[CL],[]>;
def ROR16rCL : I<0xD3, MRM1r, (ops R16:$dst, R16:$src),
                 "ror{w} {%cl, $dst|$dst, %CL}",
                 [(set R16:$dst, (rotr R16:$src, CL))]>, Imp<[CL],[]>, OpSize;
def ROR32rCL : I<0xD3, MRM1r, (ops R32:$dst, R32:$src),
                 "ror{l} {%cl, $dst|$dst, %CL}",
                 [(set R32:$dst, (rotr R32:$src, CL))]>, Imp<[CL],[]>;

def ROR8ri   : Ii8<0xC0, MRM1r, (ops R8 :$dst, R8 :$src1, i8imm:$src2),
                   "ror{b} {$src2, $dst|$dst, $src2}",
                   [(set R8:$dst, (rotr R8:$src1, (i8 imm:$src2)))]>;
def ROR16ri  : Ii8<0xC1, MRM1r, (ops R16:$dst, R16:$src1, i8imm:$src2),
                   "ror{w} {$src2, $dst|$dst, $src2}",
                   [(set R16:$dst, (rotr R16:$src1, (i8 imm:$src2)))]>, OpSize;
def ROR32ri  : Ii8<0xC1, MRM1r, (ops R32:$dst, R32:$src1, i8imm:$src2),
                   "ror{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (rotr R32:$src1, (i8 imm:$src2)))]>;
let isTwoAddress = 0 in {
  def ROR8mCL  : I<0xD2, MRM1m, (ops i8mem :$dst),
                   "ror{b} {%cl, $dst|$dst, %CL}",
                   [(store (rotr (loadi8 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def ROR16mCL : I<0xD3, MRM1m, (ops i16mem:$dst),
                   "ror{w} {%cl, $dst|$dst, %CL}",
                   [(store (rotr (loadi16 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>, OpSize;
  def ROR32mCL : I<0xD3, MRM1m, (ops i32mem:$dst), 
                   "ror{l} {%cl, $dst|$dst, %CL}",
                   [(store (rotr (loadi32 addr:$dst), CL), addr:$dst)]>,
                   Imp<[CL],[]>;
  def ROR8mi   : Ii8<0xC0, MRM1m, (ops i8mem :$dst, i8imm:$src),
                     "ror{b} {$src, $dst|$dst, $src}",
                 [(store (rotr (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def ROR16mi  : Ii8<0xC1, MRM1m, (ops i16mem:$dst, i8imm:$src),
                     "ror{w} {$src, $dst|$dst, $src}",
                [(store (rotr (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def ROR32mi  : Ii8<0xC1, MRM1m, (ops i32mem:$dst, i8imm:$src),
                     "ror{l} {$src, $dst|$dst, $src}",
                [(store (rotr (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
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
                 [(set R32:$dst, (adde R32:$src1, R32:$src2))]>;
}
def ADC32rm  : I<0x13, MRMSrcMem , (ops R32:$dst, R32:$src1, i32mem:$src2),
                 "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (adde R32:$src1, (load addr:$src2)))]>;
def ADC32ri  : Ii32<0x81, MRM2r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (adde R32:$src1, imm:$src2))]>;
def ADC32ri8 : Ii8<0x83, MRM2r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "adc{l} {$src2, $dst|$dst, $src2}",
                 [(set R32:$dst, (adde R32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def ADC32mr  : I<0x11, MRMDestMem, (ops i32mem:$dst, R32:$src2),
                   "adc{l} {$src2, $dst|$dst, $src2}",
                   [(store (adde (load addr:$dst), R32:$src2), addr:$dst)]>;
  def ADC32mi  : Ii32<0x81, MRM2m, (ops i32mem:$dst, i32imm:$src2),
                      "adc{l} {$src2, $dst|$dst, $src2}",
                  [(store (adde (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADC32mi8 : Ii8<0x83, MRM2m, (ops i32mem:$dst, i32i8imm :$src2),
                     "adc{l} {$src2, $dst|$dst, $src2}",
             [(store (adde (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
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
                  [(set R32:$dst, (sube R32:$src1, R32:$src2))]>;

let isTwoAddress = 0 in {
  def SBB32mr  : I<0x19, MRMDestMem, (ops i32mem:$dst, R32:$src2), 
                   "sbb{l} {$src2, $dst|$dst, $src2}",
                   [(store (sube (load addr:$dst), R32:$src2), addr:$dst)]>;
  def SBB8mi  : Ii32<0x80, MRM3m, (ops i8mem:$dst, i8imm:$src2), 
                      "sbb{b} {$src2, $dst|$dst, $src2}",
                   [(store (sube (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB32mi  : Ii32<0x81, MRM3m, (ops i32mem:$dst, i32imm:$src2), 
                      "sbb{l} {$src2, $dst|$dst, $src2}",
                  [(store (sube (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB32mi8 : Ii8<0x83, MRM3m, (ops i32mem:$dst, i32i8imm :$src2), 
                     "sbb{l} {$src2, $dst|$dst, $src2}",
             [(store (sube (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}
def SBB32rm  : I<0x1B, MRMSrcMem, (ops R32:$dst, R32:$src1, i32mem:$src2),
                    "sbb{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (sube R32:$src1, (load addr:$src2)))]>;
def SBB32ri  : Ii32<0x81, MRM3r, (ops R32:$dst, R32:$src1, i32imm:$src2),
                    "sbb{l} {$src2, $dst|$dst, $src2}",
                    [(set R32:$dst, (sube R32:$src1, imm:$src2))]>;
def SBB32ri8 : Ii8<0x83, MRM3r, (ops R32:$dst, R32:$src1, i32i8imm:$src2),
                   "sbb{l} {$src2, $dst|$dst, $src2}",
                   [(set R32:$dst, (sube R32:$src1, i32immSExt8:$src2))]>;

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
                 [(X86test R8:$src1, R8:$src2)]>;
def TEST16rr : I<0x85, MRMDestReg, (ops R16:$src1, R16:$src2),
                 "test{w} {$src2, $src1|$src1, $src2}",
                 [(X86test R16:$src1, R16:$src2)]>, OpSize;
def TEST32rr : I<0x85, MRMDestReg, (ops R32:$src1, R32:$src2),
                 "test{l} {$src2, $src1|$src1, $src2}",
                 [(X86test R32:$src1, R32:$src2)]>;
}
def TEST8mr  : I<0x84, MRMDestMem, (ops i8mem :$src1, R8 :$src2),
                 "test{b} {$src2, $src1|$src1, $src2}",
                 [(X86test (loadi8 addr:$src1), R8:$src2)]>;
def TEST16mr : I<0x85, MRMDestMem, (ops i16mem:$src1, R16:$src2),
                 "test{w} {$src2, $src1|$src1, $src2}",
                 [(X86test (loadi16 addr:$src1), R16:$src2)]>,
               OpSize;
def TEST32mr : I<0x85, MRMDestMem, (ops i32mem:$src1, R32:$src2),
                 "test{l} {$src2, $src1|$src1, $src2}",
                 [(X86test (loadi32 addr:$src1), R32:$src2)]>;
def TEST8rm  : I<0x84, MRMSrcMem, (ops R8 :$src1, i8mem :$src2),
                 "test{b} {$src2, $src1|$src1, $src2}",
                 [(X86test R8:$src1, (loadi8 addr:$src2))]>;
def TEST16rm : I<0x85, MRMSrcMem, (ops R16:$src1, i16mem:$src2),
                 "test{w} {$src2, $src1|$src1, $src2}",
                 [(X86test R16:$src1, (loadi16 addr:$src2))]>,
               OpSize;
def TEST32rm : I<0x85, MRMSrcMem, (ops R32:$src1, i32mem:$src2),
                 "test{l} {$src2, $src1|$src1, $src2}",
                 [(X86test R32:$src1, (loadi32 addr:$src2))]>;

def TEST8ri  : Ii8 <0xF6, MRM0r,                     // flags = R8  & imm8
                    (ops R8:$src1, i8imm:$src2),
                    "test{b} {$src2, $src1|$src1, $src2}",
                    [(X86test R8:$src1, imm:$src2)]>;
def TEST16ri : Ii16<0xF7, MRM0r,                     // flags = R16 & imm16
                    (ops R16:$src1, i16imm:$src2),
                    "test{w} {$src2, $src1|$src1, $src2}",
                    [(X86test R16:$src1, imm:$src2)]>, OpSize;
def TEST32ri : Ii32<0xF7, MRM0r,                     // flags = R32 & imm32
                    (ops R32:$src1, i32imm:$src2),
                    "test{l} {$src2, $src1|$src1, $src2}",
                    [(X86test R32:$src1, imm:$src2)]>;
def TEST8mi  : Ii8 <0xF6, MRM0m,                     // flags = [mem8]  & imm8
                    (ops i8mem:$src1, i8imm:$src2),
                    "test{b} {$src2, $src1|$src1, $src2}",
                    [(X86test (loadi8 addr:$src1), imm:$src2)]>;
def TEST16mi : Ii16<0xF7, MRM0m,                     // flags = [mem16] & imm16
                    (ops i16mem:$src1, i16imm:$src2),
                    "test{w} {$src2, $src1|$src1, $src2}",
                    [(X86test (loadi16 addr:$src1), imm:$src2)]>,
               OpSize;
def TEST32mi : Ii32<0xF7, MRM0m,                     // flags = [mem32] & imm32
                    (ops i32mem:$src1, i32imm:$src2),
                    "test{l} {$src2, $src1|$src1, $src2}",
                    [(X86test (loadi32 addr:$src1), imm:$src2)]>;


// Condition code ops, incl. set if equal/not equal/...
def SAHF     : I<0x9E, RawFrm, (ops), "sahf", []>, Imp<[AH],[]>;  // flags = AH
def LAHF     : I<0x9F, RawFrm, (ops), "lahf", []>, Imp<[],[AH]>;  // AH = flags

def SETEr    : I<0x94, MRM0r, 
                 (ops R8   :$dst),
                 "sete $dst",
                 [(set R8:$dst, (X86setcc X86_COND_E))]>,
               TB;                        // R8 = ==
def SETEm    : I<0x94, MRM0m, 
                 (ops i8mem:$dst),
                 "sete $dst",
                 [(store (X86setcc X86_COND_E), addr:$dst)]>,
               TB;                        // [mem8] = ==
def SETNEr   : I<0x95, MRM0r, 
                 (ops R8   :$dst),
                 "setne $dst",
                 [(set R8:$dst, (X86setcc X86_COND_NE))]>,
               TB;                        // R8 = !=
def SETNEm   : I<0x95, MRM0m, 
                 (ops i8mem:$dst),
                 "setne $dst",
                 [(store (X86setcc X86_COND_NE), addr:$dst)]>,
               TB;                        // [mem8] = !=
def SETLr    : I<0x9C, MRM0r, 
                 (ops R8   :$dst),
                 "setl $dst",
                 [(set R8:$dst, (X86setcc X86_COND_L))]>,
               TB;                        // R8 = <  signed
def SETLm    : I<0x9C, MRM0m, 
                 (ops i8mem:$dst),
                 "setl $dst",
                 [(store (X86setcc X86_COND_L), addr:$dst)]>,
               TB;                        // [mem8] = <  signed
def SETGEr   : I<0x9D, MRM0r, 
                 (ops R8   :$dst),
                 "setge $dst",
                 [(set R8:$dst, (X86setcc X86_COND_GE))]>,
               TB;                        // R8 = >= signed
def SETGEm   : I<0x9D, MRM0m, 
                 (ops i8mem:$dst),
                 "setge $dst",
                 [(store (X86setcc X86_COND_GE), addr:$dst)]>,
               TB;                        // [mem8] = >= signed
def SETLEr   : I<0x9E, MRM0r, 
                 (ops R8   :$dst),
                 "setle $dst",
                 [(set R8:$dst, (X86setcc X86_COND_LE))]>,
               TB;                        // R8 = <= signed
def SETLEm   : I<0x9E, MRM0m, 
                 (ops i8mem:$dst),
                 "setle $dst",
                 [(store (X86setcc X86_COND_LE), addr:$dst)]>,
               TB;                        // [mem8] = <= signed
def SETGr    : I<0x9F, MRM0r, 
                 (ops R8   :$dst),
                 "setg $dst",
                 [(set R8:$dst, (X86setcc X86_COND_G))]>,
               TB;                        // R8 = >  signed
def SETGm    : I<0x9F, MRM0m, 
                 (ops i8mem:$dst),
                 "setg $dst",
                 [(store (X86setcc X86_COND_G), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETBr    : I<0x92, MRM0r,
                 (ops R8   :$dst),
                 "setb $dst",
                 [(set R8:$dst, (X86setcc X86_COND_B))]>,
               TB;                        // R8 = <  unsign
def SETBm    : I<0x92, MRM0m,
                 (ops i8mem:$dst),
                 "setb $dst",
                 [(store (X86setcc X86_COND_B), addr:$dst)]>,
               TB;                        // [mem8] = <  unsign
def SETAEr   : I<0x93, MRM0r, 
                 (ops R8   :$dst),
                 "setae $dst",
                 [(set R8:$dst, (X86setcc X86_COND_AE))]>,
               TB;                        // R8 = >= unsign
def SETAEm   : I<0x93, MRM0m, 
                 (ops i8mem:$dst),
                 "setae $dst",
                 [(store (X86setcc X86_COND_AE), addr:$dst)]>,
               TB;                        // [mem8] = >= unsign
def SETBEr   : I<0x96, MRM0r, 
                 (ops R8   :$dst),
                 "setbe $dst",
                 [(set R8:$dst, (X86setcc X86_COND_BE))]>,
               TB;                        // R8 = <= unsign
def SETBEm   : I<0x96, MRM0m, 
                 (ops i8mem:$dst),
                 "setbe $dst",
                 [(store (X86setcc X86_COND_BE), addr:$dst)]>,
               TB;                        // [mem8] = <= unsign
def SETAr    : I<0x97, MRM0r, 
                 (ops R8   :$dst),
                 "seta $dst",
                 [(set R8:$dst, (X86setcc X86_COND_A))]>,
               TB;                        // R8 = >  signed
def SETAm    : I<0x97, MRM0m, 
                 (ops i8mem:$dst),
                 "seta $dst",
                 [(store (X86setcc X86_COND_A), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETSr    : I<0x98, MRM0r, 
                 (ops R8   :$dst),
                 "sets $dst",
                 [(set R8:$dst, (X86setcc X86_COND_S))]>,
               TB;                        // R8 = <sign bit>
def SETSm    : I<0x98, MRM0m, 
                 (ops i8mem:$dst),
                 "sets $dst",
                 [(store (X86setcc X86_COND_S), addr:$dst)]>,
               TB;                        // [mem8] = <sign bit>
def SETNSr   : I<0x99, MRM0r, 
                 (ops R8   :$dst),
                 "setns $dst",
                 [(set R8:$dst, (X86setcc X86_COND_NS))]>,
               TB;                        // R8 = !<sign bit>
def SETNSm   : I<0x99, MRM0m, 
                 (ops i8mem:$dst),
                 "setns $dst",
                 [(store (X86setcc X86_COND_NS), addr:$dst)]>,
               TB;                        // [mem8] = !<sign bit>
def SETPr    : I<0x9A, MRM0r, 
                 (ops R8   :$dst),
                 "setp $dst",
                 [(set R8:$dst, (X86setcc X86_COND_P))]>,
               TB;                        // R8 = parity
def SETPm    : I<0x9A, MRM0m, 
                 (ops i8mem:$dst),
                 "setp $dst",
                 [(store (X86setcc X86_COND_P), addr:$dst)]>,
               TB;                        // [mem8] = parity
def SETNPr   : I<0x9B, MRM0r, 
                 (ops R8   :$dst),
                 "setnp $dst",
                 [(set R8:$dst, (X86setcc X86_COND_NP))]>,
               TB;                        // R8 = not parity
def SETNPm   : I<0x9B, MRM0m, 
                 (ops i8mem:$dst),
                 "setnp $dst",
                 [(store (X86setcc X86_COND_NP), addr:$dst)]>,
               TB;                        // [mem8] = not parity

// Integer comparisons
def CMP8rr  : I<0x38, MRMDestReg,
                (ops R8 :$src1, R8 :$src2),
                "cmp{b} {$src2, $src1|$src1, $src2}",
                [(X86cmp R8:$src1, R8:$src2)]>;
def CMP16rr : I<0x39, MRMDestReg,
                (ops R16:$src1, R16:$src2),
                "cmp{w} {$src2, $src1|$src1, $src2}",
                [(X86cmp R16:$src1, R16:$src2)]>, OpSize;
def CMP32rr : I<0x39, MRMDestReg,
                (ops R32:$src1, R32:$src2),
                "cmp{l} {$src2, $src1|$src1, $src2}",
                [(X86cmp R32:$src1, R32:$src2)]>;
def CMP8mr  : I<0x38, MRMDestMem,
                (ops i8mem :$src1, R8 :$src2),
                "cmp{b} {$src2, $src1|$src1, $src2}",
                [(X86cmp (loadi8 addr:$src1), R8:$src2)]>;
def CMP16mr : I<0x39, MRMDestMem,
                (ops i16mem:$src1, R16:$src2),
                "cmp{w} {$src2, $src1|$src1, $src2}",
                [(X86cmp (loadi16 addr:$src1), R16:$src2)]>, OpSize;
def CMP32mr : I<0x39, MRMDestMem,
                (ops i32mem:$src1, R32:$src2),
                "cmp{l} {$src2, $src1|$src1, $src2}",
                [(X86cmp (loadi32 addr:$src1), R32:$src2)]>;
def CMP8rm  : I<0x3A, MRMSrcMem,
                (ops R8 :$src1, i8mem :$src2),
                "cmp{b} {$src2, $src1|$src1, $src2}",
                [(X86cmp R8:$src1, (loadi8 addr:$src2))]>;
def CMP16rm : I<0x3B, MRMSrcMem,
                (ops R16:$src1, i16mem:$src2),
                "cmp{w} {$src2, $src1|$src1, $src2}",
                [(X86cmp R16:$src1, (loadi16 addr:$src2))]>, OpSize;
def CMP32rm : I<0x3B, MRMSrcMem,
                (ops R32:$src1, i32mem:$src2),
                "cmp{l} {$src2, $src1|$src1, $src2}",
                [(X86cmp R32:$src1, (loadi32 addr:$src2))]>;
def CMP8ri  : Ii8<0x80, MRM7r,
                  (ops R8:$src1, i8imm:$src2),
                  "cmp{b} {$src2, $src1|$src1, $src2}",
                  [(X86cmp R8:$src1, imm:$src2)]>;
def CMP16ri : Ii16<0x81, MRM7r,
                   (ops R16:$src1, i16imm:$src2),
                   "cmp{w} {$src2, $src1|$src1, $src2}",
                   [(X86cmp R16:$src1, imm:$src2)]>, OpSize;
def CMP32ri : Ii32<0x81, MRM7r,
                   (ops R32:$src1, i32imm:$src2),
                   "cmp{l} {$src2, $src1|$src1, $src2}",
                   [(X86cmp R32:$src1, imm:$src2)]>;
def CMP8mi  : Ii8 <0x80, MRM7m,
                   (ops i8mem :$src1, i8imm :$src2),
                   "cmp{b} {$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi8 addr:$src1), imm:$src2)]>;
def CMP16mi : Ii16<0x81, MRM7m,
                   (ops i16mem:$src1, i16imm:$src2),
                   "cmp{w} {$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi16 addr:$src1), imm:$src2)]>, OpSize;
def CMP32mi : Ii32<0x81, MRM7m,
                   (ops i32mem:$src1, i32imm:$src2),
                   "cmp{l} {$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi32 addr:$src1), imm:$src2)]>;
def CMP16ri8 : Ii8<0x83, MRM7r,
                   (ops R16:$src1, i16i8imm:$src2),
                   "cmp{w} {$src2, $src1|$src1, $src2}",
                   [(X86cmp R16:$src1, i16immSExt8:$src2)]>, OpSize;
def CMP16mi8 : Ii8<0x83, MRM7m,
                   (ops i16mem:$src1, i16i8imm:$src2),
                   "cmp{w} {$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi16 addr:$src1), i16immSExt8:$src2)]>, OpSize;
def CMP32mi8 : Ii8<0x83, MRM7m,
                   (ops i32mem:$src1, i32i8imm:$src2),
                   "cmp{l} {$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi32 addr:$src1), i32immSExt8:$src2)]>;
def CMP32ri8 : Ii8<0x83, MRM7r,
                   (ops R32:$src1, i32i8imm:$src2),
                   "cmp{l} {$src2, $src1|$src1, $src2}",
                   [(X86cmp R32:$src1, i32immSExt8:$src2)]>;

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
// Miscellaneous Instructions
//===----------------------------------------------------------------------===//

def RDTSC : I<0x31, RawFrm, (ops), "rdtsc", [(X86rdtsc)]>,
            TB, Imp<[],[EAX,EDX]>;

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map movr0 to xor.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
def MOV8r0   : I<0x30, MRMInitReg, (ops R8 :$dst),
                 "xor{b} $dst, $dst",
                 [(set R8:$dst, 0)]>;
def MOV16r0  : I<0x31, MRMInitReg,  (ops R16:$dst), 
                 "xor{w} $dst, $dst",
                 [(set R16:$dst, 0)]>, OpSize;
def MOV32r0  : I<0x31, MRMInitReg,  (ops R32:$dst), 
                 "xor{l} $dst, $dst",
                 [(set R32:$dst, 0)]>;

//===----------------------------------------------------------------------===//
// DWARF Pseudo Instructions
//

def DWARF_LOC   : I<0, Pseudo, (ops i32imm:$line, i32imm:$col, i32imm:$file),
                    "; .loc $file, $line, $col",
                    [(dwarf_loc (i32 imm:$line), (i32 imm:$col),
                      (i32 imm:$file))]>;

def DWARF_LABEL : I<0, Pseudo, (ops i32imm:$id),
                   "\nLdebug_loc${id:debug}:",
                   [(dwarf_label (i32 imm:$id))]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// ConstantPool GlobalAddress, ExternalSymbol
def : Pat<(i32 (X86Wrapper tconstpool  :$dst)), (MOV32ri tconstpool  :$dst)>;
def : Pat<(i32 (X86Wrapper tjumptable  :$dst)), (MOV32ri tjumptable  :$dst)>;
def : Pat<(i32 (X86Wrapper tglobaladdr :$dst)), (MOV32ri tglobaladdr :$dst)>;
def : Pat<(i32 (X86Wrapper texternalsym:$dst)), (MOV32ri texternalsym:$dst)>;

def : Pat<(add R32:$src1, (X86Wrapper tconstpool:$src2)),
          (ADD32ri R32:$src1, tconstpool:$src2)>;
def : Pat<(add R32:$src1, (X86Wrapper tjumptable:$src2)),
          (ADD32ri R32:$src1, tjumptable:$src2)>;
def : Pat<(add R32:$src1, (X86Wrapper tglobaladdr :$src2)),
          (ADD32ri R32:$src1, tglobaladdr:$src2)>;
def : Pat<(add R32:$src1, (X86Wrapper texternalsym:$src2)),
          (ADD32ri R32:$src1, texternalsym:$src2)>;

def : Pat<(store (X86Wrapper tglobaladdr:$src), addr:$dst),
          (MOV32mi addr:$dst, tglobaladdr:$src)>;
def : Pat<(store (X86Wrapper texternalsym:$src), addr:$dst),
          (MOV32mi addr:$dst, texternalsym:$src)>;

// Calls
def : Pat<(X86call tglobaladdr:$dst),
          (CALLpcrel32 tglobaladdr:$dst)>;
def : Pat<(X86call texternalsym:$dst),
          (CALLpcrel32 texternalsym:$dst)>;

// X86 specific add which produces a flag.
def : Pat<(addc R32:$src1, R32:$src2),
          (ADD32rr R32:$src1, R32:$src2)>;
def : Pat<(addc R32:$src1, (load addr:$src2)),
          (ADD32rm R32:$src1, addr:$src2)>;
def : Pat<(addc R32:$src1, imm:$src2),
          (ADD32ri R32:$src1, imm:$src2)>;
def : Pat<(addc R32:$src1, i32immSExt8:$src2),
          (ADD32ri8 R32:$src1, i32immSExt8:$src2)>;

def : Pat<(subc R32:$src1, R32:$src2),
          (SUB32rr R32:$src1, R32:$src2)>;
def : Pat<(subc R32:$src1, (load addr:$src2)),
          (SUB32rm R32:$src1, addr:$src2)>;
def : Pat<(subc R32:$src1, imm:$src2),
          (SUB32ri R32:$src1, imm:$src2)>;
def : Pat<(subc R32:$src1, i32immSExt8:$src2),
          (SUB32ri8 R32:$src1, i32immSExt8:$src2)>;

def : Pat<(truncstore (i8 imm:$src), addr:$dst, i1), 
          (MOV8mi addr:$dst, imm:$src)>;
def : Pat<(truncstore R8:$src, addr:$dst, i1), 
          (MOV8mr addr:$dst, R8:$src)>;

// {s|z}extload bool -> {s|z}extload byte
def : Pat<(sextloadi16i1 addr:$src), (MOVSX16rm8 addr:$src)>;
def : Pat<(sextloadi32i1 addr:$src), (MOVSX32rm8 addr:$src)>;
def : Pat<(zextloadi8i1  addr:$src), (MOV8rm     addr:$src)>;
def : Pat<(zextloadi16i1 addr:$src), (MOVZX16rm8 addr:$src)>;
def : Pat<(zextloadi32i1 addr:$src), (MOVZX32rm8 addr:$src)>;

// extload bool -> extload byte
def : Pat<(extloadi8i1 addr:$src), (MOV8rm addr:$src)>;

// anyext -> zext
def : Pat<(i16 (anyext R8 :$src)), (MOVZX16rr8  R8 :$src)>;
def : Pat<(i32 (anyext R8 :$src)), (MOVZX32rr8  R8 :$src)>;
def : Pat<(i32 (anyext R16:$src)), (MOVZX32rr16 R16:$src)>;
def : Pat<(i16 (anyext (loadi8  addr:$src))), (MOVZX16rm8  addr:$src)>;
def : Pat<(i32 (anyext (loadi8  addr:$src))), (MOVZX32rm8  addr:$src)>;
def : Pat<(i32 (anyext (loadi16 addr:$src))), (MOVZX32rm16 addr:$src)>;

//===----------------------------------------------------------------------===//
// Some peepholes
//===----------------------------------------------------------------------===//

// (shl x, 1) ==> (add x, x)
def : Pat<(shl R8 :$src1, (i8 1)), (ADD8rr  R8 :$src1, R8 :$src1)>;
def : Pat<(shl R16:$src1, (i8 1)), (ADD16rr R16:$src1, R16:$src1)>;
def : Pat<(shl R32:$src1, (i8 1)), (ADD32rr R32:$src1, R32:$src1)>;

// (or (x >> c) | (y << (32 - c))) ==> (shrd32 x, y, c)
def : Pat<(or (srl R32:$src1, CL:$amt),
              (shl R32:$src2, (sub 32, CL:$amt))),
          (SHRD32rrCL R32:$src1, R32:$src2)>;

def : Pat<(store (or (srl (loadi32 addr:$dst), CL:$amt),
                     (shl R32:$src2, (sub 32, CL:$amt))), addr:$dst),
          (SHRD32mrCL addr:$dst, R32:$src2)>;

// (or (x << c) | (y >> (32 - c))) ==> (shld32 x, y, c)
def : Pat<(or (shl R32:$src1, CL:$amt),
              (srl R32:$src2, (sub 32, CL:$amt))),
          (SHLD32rrCL R32:$src1, R32:$src2)>;

def : Pat<(store (or (shl (loadi32 addr:$dst), CL:$amt),
                     (srl R32:$src2, (sub 32, CL:$amt))), addr:$dst),
          (SHLD32mrCL addr:$dst, R32:$src2)>;

// (or (x >> c) | (y << (16 - c))) ==> (shrd16 x, y, c)
def : Pat<(or (srl R16:$src1, CL:$amt),
              (shl R16:$src2, (sub 16, CL:$amt))),
          (SHRD16rrCL R16:$src1, R16:$src2)>;

def : Pat<(store (or (srl (loadi16 addr:$dst), CL:$amt),
                     (shl R16:$src2, (sub 16, CL:$amt))), addr:$dst),
          (SHRD16mrCL addr:$dst, R16:$src2)>;

// (or (x << c) | (y >> (16 - c))) ==> (shld16 x, y, c)
def : Pat<(or (shl R16:$src1, CL:$amt),
              (srl R16:$src2, (sub 16, CL:$amt))),
          (SHLD16rrCL R16:$src1, R16:$src2)>;

def : Pat<(store (or (shl (loadi16 addr:$dst), CL:$amt),
                     (srl R16:$src2, (sub 16, CL:$amt))), addr:$dst),
          (SHLD16mrCL addr:$dst, R16:$src2)>;


//===----------------------------------------------------------------------===//
// Floating Point Stack Support
//===----------------------------------------------------------------------===//

include "X86InstrFPStack.td"

//===----------------------------------------------------------------------===//
// MMX and XMM Packed Integer support (requires MMX, SSE, and SSE2)
//===----------------------------------------------------------------------===//

include "X86InstrMMX.td"

//===----------------------------------------------------------------------===//
// XMM Floating point support (requires SSE / SSE2)
//===----------------------------------------------------------------------===//

include "X86InstrSSE.td"
