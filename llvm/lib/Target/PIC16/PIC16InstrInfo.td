//===- PIC16InstrInfo.td - PIC16 Register defs ----------------*- tblgen-*-===//
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

include "PIC16InstrFormats.td"

//===----------------------------------------------------------------------===//
// PIC16 profiles and nodes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PIC16 addressing mode. 
//===----------------------------------------------------------------------===//
// It matches address of globals as well as the stack slots
// that are created for locals and temporaries. This addressing mode
// converts the GlobalAddress and FrameIndex nodes to TargetGlobalAddress
// and TargetFrameIndex nodes.
def diraddrmode : ComplexPattern<i16, 2, "SelectDirectAM", [frameindex], []>;
def dirloadmode : ComplexPattern<i16, 2, "LoadNothing", [frameindex], []>;
def indirloadmode : ComplexPattern<i16, 2, "LoadFSR", [frameindex], []>;


// Address operand.
def mem : Operand<i16> {
  let PrintMethod = "printAddrModeOperand";
  let MIOperandInfo = (ops i16imm, PTRRegs);
}

// Instruction operand types
def simm8      : Operand<i8>;


// These are target-independent nodes, but have target-specific formats.
def SDT_PIC16CallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i8> ]>;
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_PIC16CallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_PIC16CallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;

def PIC16Wrapper  : SDNode<"PIC16ISD::Wrapper", SDTIntUnaryOp>;

// so_imm_XFORM - Return a so_imm value packed into the format described for
// so_imm def below.
def so_imm_XFORM : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((int8_t)N->getZExtValue(), MVT::i32);
}]>;

def so_imm : Operand<i8>,
             PatLeaf<(imm), [{}]> {
  let PrintMethod = "printSOImmOperand";
}



// PIC16 Address Mode! SDNode frameindex could possibily be a match 
// since load and store instructions from stack used it.
def addr : Operand<i16>;

// Arithmetic 2 register operands
class ArithI<bits<6> op, string instr_asm, SDNode OpNode,
             Operand Od> :
  LiteralFormat< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, Od:$c),
      !strconcat(instr_asm, " $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, Od:$c))]>;

// Memory Load/Store. 
class LoadDirect<bits<6> op, string instr_asm, PatFrag OpNode>:
  ByteFormat< op,
      (outs CPURegs:$dst),
      (ins mem:$addr),
      !strconcat(instr_asm, " $addr"),
      [(set CPURegs:$dst, (OpNode diraddrmode:$addr))]>;

class LoadInDirect<bits<6> op, string instr_asm, PatFrag OpNode>:
  ByteFormat< op,
      (outs PTRRegs:$dst),
      (ins mem:$addr),
      !strconcat(instr_asm, " $addr, $dst"),
      [(set PTRRegs:$dst, (OpNode indirloadmode:$addr))]>;

class StoreDirect<bits<6> op, string instr_asm, PatFrag OpNode>:
  ByteFormat< op,
      (outs),
      (ins CPURegs:$src, mem:$addr),
      !strconcat(instr_asm, " $addr"),
      [(OpNode CPURegs:$src, diraddrmode:$addr)]>;

class StoreInDirect<bits<6> op, string instr_asm, PatFrag OpNode>:
  ByteFormat< op,
      (outs),
      (ins CPURegs:$src, PTRRegs:$fsr),
      !strconcat(instr_asm, " $fsr"),
      [(OpNode CPURegs:$src, PTRRegs:$fsr)]>;

// Move.
class MovLit<bits<6> op, string instr_asm>:
  LiteralFormat< op,
      (outs CPURegs:$dst),
      (ins i8imm:$src),
      !strconcat(instr_asm, " $src"),
      [(set CPURegs:$dst, imm:$src)]>;


// Arithmetic with memory store.
// Arithmetic instrunctions involving W and memory location.
// Since W is implicit, we only print the memory operand.
class Arith1M<bits<6> op, string instr_asm, SDNode OpNode>:
  ByteFormat< op,
      (outs),
      (ins CPURegs:$b, mem:$dst),
      !strconcat(instr_asm, " $dst"),
      [(store (OpNode (load diraddrmode:$dst), CPURegs:$b), diraddrmode:$dst),
       (store (OpNode CPURegs:$b, (load diraddrmode:$dst)), diraddrmode:$dst)]>;

// Arithmetic with memory load.
// Arithmetic instrunctions involving W and memory location.
// Since W is implicit, we only print the memory operand.
class Arith1R<bits<6> op, string instr_asm, SDNode OpNode>:
  ByteFormat< op,
      (outs CPURegs:$dst),
      (ins mem:$src1, CPURegs:$src2),
      !strconcat(instr_asm, " $src1"),
      [(set CPURegs:$dst, (OpNode (load diraddrmode:$src1), CPURegs:$src2))]>;

// Arithmetic with memory load.
// Arithmetic instrunctions involving W and memory location.
// Since W is implicit, we only print the memory operand.
class Arith2R<bits<6> op, string instr_asm, SDNode OpNode>:
  ByteFormat< op,
      (outs CPURegs:$dst),
      (ins mem:$src1, CPURegs:$src2),
      !strconcat(instr_asm, " $src1"),
      [(set CPURegs:$dst, (OpNode CPURegs:$src2, (load diraddrmode:$src1)))]>;

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PIC16I Instructions
//===----------------------------------------------------------------------===//

// Arithmetic

// ADDiu just accept 16-bit immediates but we handle this on Pat's.
// immZExt32 is used here so it can match GlobalAddress immediates.
// def ADDLW   : ArithI<0x09, "addlw", add, so_imm>;

let isReMaterializable = 1 in {
def MOVLW : MovLit<0x24, "movlw">;
}

// Load/Store
def LFSR1      : LoadInDirect        <0x4, "lfsr",  load>;

let isReMaterializable = 1 in {
def MOVF       : LoadDirect <0x23, "movf",  load>;
}

def MOVWF      : StoreDirect <0x2b, "movwf", store>;

def MOVFSRINC  : StoreInDirect <0x5, "movfsrinc", store>;

def RETURN     : ControlFormat<0x03, (outs), (ins), "return", []>;

def ADDWF      : Arith1M<0x01, "addwf", add>; 
def ADDFW      : Arith1R<0x02, "addfw", add>; 

def ADDWFE     : Arith1M<0x03, "addwfe", adde>; 
def ADDFWE     : Arith1R<0x04, "addfwe", adde>; 

def ADDWFC     : Arith1M<0x05, "addwfc", addc>; 
def ADDFWC     : Arith1R<0x06, "addfwc", addc>; 

def SUBWF      : Arith1M<0x07, "subwf", sub>; 
def SUBFW      : Arith1R<0x08, "subfw", sub>; 

def SUBWFE     : Arith1M<0x09, "subwfe", sube>; 
def SUBFWE     : Arith1R<0x0a, "subfwe", sube>; 

def SUBWFC     : Arith1M<0x0b, "subwfc", subc>; 
def SUBFWC     : Arith1R<0x0d, "subfwc", subc>; 

def SUBRFW     : Arith2R<0x08, "subfw", sub>; 

def SUBRFWE    : Arith2R<0x0a, "subfwe", sube>; 

def SUBRFWC    : Arith2R<0x0d, "subfwc", subc>; 

def brtarget   : Operand<OtherVT>;

class UncondJump< bits<4> op, string instr_asm>:
  BitFormat< op,
             (outs),
             (ins brtarget:$target),
             !strconcat(instr_asm, " $target"),
             [(br bb:$target)]>;

def GOTO       : UncondJump<0x1, "goto">;

class LogicM<bits<6> op, string instr_asm, SDNode OpNode> :
  ByteFormat< op,
      (outs),
      (ins CPURegs:$b, mem:$dst),
      !strconcat(instr_asm, " $dst"),
      [(store (OpNode (load diraddrmode:$dst), CPURegs:$b), diraddrmode:$dst)]>;

class LogicR<bits<6> op, string instr_asm, SDNode OpNode> :
  ByteFormat< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, mem:$c),
      !strconcat(instr_asm, " $c"),
      [(set CPURegs:$dst, (OpNode (load diraddrmode:$c), CPURegs:$b))]>;

class LogicI<bits<6> op, string instr_asm, SDNode OpNode, Operand Od> :
  LiteralFormat< op,
      (outs CPURegs:$dst),
      (ins CPURegs:$b, Od:$c),
      !strconcat(instr_asm, " $c"),
      [(set CPURegs:$dst, (OpNode CPURegs:$b, Od:$c ))]>;

def XORWF : LogicM<0x1,"xorwf",xor>; 
def XORFW : LogicR<0x1,"xorfw",xor>; 
def XORLW : LogicI<0x1,"xorlw",xor, so_imm>;

def ANDWF : LogicM<0x1,"andwf",and>; 
def ANDFW : LogicR<0x1,"andfw",and>; 
def ANDLW : LogicI<0x1,"andlw",and, so_imm>;

def IORWF : LogicM<0x1,"iorwf",or>; 
def IORFW : LogicR<0x1,"iorfw",or>; 
def IORLW : LogicI<0x1,"iorlw",or, so_imm>;


/* For comparison before branch */
def SDT_PIC16Cmp  : SDTypeProfile<1, 3, [SDTCisSameAs<0,1>]>;
def SDTIntBinOpPIC16 : SDTypeProfile<1, 2, [SDTCisSameAs<0,1>, 
                                     SDTCisSameAs<1,2>, SDTCisInt<1>]>;

def PIC16Cmp : SDNode<"PIC16ISD::Cmp",SDTIntBinOpPIC16, [SDNPOutFlag]>; 
def PIC16XORCC : SDNode<"PIC16ISD::XORCC",SDTIntBinOpPIC16, [SDNPOutFlag]>; 
def PIC16SUBCC : SDNode<"PIC16ISD::SUBCC",SDTIntBinOpPIC16, [SDNPOutFlag]>; 

def XORFWCC : LogicR<0x1,"xorfw",PIC16XORCC>; 
def XORLWCC : LogicI<0x1,"xorlw",PIC16XORCC, so_imm>; 
def SUBFWCC : Arith1R<0x1,"subfw",PIC16SUBCC>; 
def SUBLWCC : ArithI<0x1,"sublw",PIC16SUBCC, so_imm>; 


/* For branch conditions */
def SDT_PIC16Branch  : SDTypeProfile<0, 3, [SDTCisVT<0, OtherVT>, 
                                     SDTCisVT<1,i8>, SDTCisVT<2,i8>]>;

def PIC16Branch : SDNode<"PIC16ISD::Branch",SDT_PIC16Branch, 
                         [SDNPHasChain, SDNPInFlag]>; 

def PIC16BTFSS  : SDNode<"PIC16ISD::BTFSS",SDT_PIC16Branch, 
                         [SDNPHasChain, SDNPInFlag]>; 

def PIC16BTFSC  : SDNode<"PIC16ISD::BTFSC",SDT_PIC16Branch, 
                         [SDNPHasChain, SDNPInFlag]>; 

class InstrBitTestCC<bits<4> op, string instr_asm,SDNode OpNode>:
  BitFormat< op,
             (outs),
             (ins brtarget:$target ,so_imm:$i, STATUSRegs:$s ),
                   !strconcat(instr_asm, " $s, $i, $target"),
                   [(OpNode bb:$target, so_imm:$i, STATUSRegs:$s )]>;

def BTFSS : InstrBitTestCC<0x1,"btfss",PIC16BTFSS>;
def BTFSC : InstrBitTestCC<0x1,"btfsc",PIC16BTFSC>;


//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

let Defs = [STKPTR], Uses = [STKPTR] in {
def ADJCALLSTACKDOWN : Pseudo<255, (outs), (ins i8imm:$amt),
                               "!ADJCALLSTACKDOWN $amt",
                               [(callseq_start imm:$amt)]>;
def ADJCALLSTACKUP : Pseudo<254, (outs), (ins i8imm:$amt),
                            "!ADJCALLSTACKUP $amt",
                            [(callseq_end imm:$amt)]>;
}


//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//
def : Pat<(ret), (RETURN)>;
