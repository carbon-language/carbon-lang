//===- AlphaInstrFormats.td - Alpha Instruction Formats ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//3.3:
//Memory
//Branch
//Operate
//Floating-point
//PALcode

def u8imm   : Operand<i64>;
def s14imm  : Operand<i64>;
def s16imm  : Operand<i64>;
def s21imm  : Operand<i64>;
def s64imm  : Operand<i64>;

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
// Alpha instruction baseline
class InstAlphaAlt<bits<6> op, string asmstr> : Instruction {
  field bits<32> Inst;
  let Namespace = "Alpha";
  let AsmString = asmstr;
  let Inst{31-26} = op;
}

class InstAlpha<bits<6> op, dag OL, string asmstr> 
: InstAlphaAlt<op, asmstr> { // Alpha instruction baseline
  let OperandList = OL;
}

//3.3.1
class MFormAlt<bits<6> opcode, string asmstr> 
        : InstAlphaAlt<opcode, asmstr> {
  bits<5> Ra;
  bits<16> disp;
  bits<5> Rb;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-0} = disp;
}
class MForm<bits<6> opcode, string asmstr> 
        : InstAlpha<opcode, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), asmstr> {
  bits<5> Ra;
  bits<16> disp;
  bits<5> Rb;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-0} = disp;
}
class MfcForm<bits<6> opcode, bits<16> fc, string asmstr> 
        : InstAlpha<opcode, (ops GPRC:$RA, GPRC:$RB), asmstr> {
  bits<5> Ra;
  bits<5> Rb;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-0} = fc;
}

class MgForm<bits<6> opcode, string asmstr> 
        : InstAlpha<opcode, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB, s16imm:$NUM), asmstr> {
  bits<5> Ra;
  bits<16> disp;
  bits<5> Rb;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-0} = disp;
}

class MbrForm<bits<6> opcode, bits<2> TB, dag OL, string asmstr> : InstAlpha<opcode, OL, asmstr> {
  bits<5> Ra;
  bits<5> Rb;
  bits<14> disp;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-14} = TB;
  let Inst{13-0} = disp;
}

//3.3.2
let isBranch = 1, isTerminator = 1 in
class BForm<bits<6> opcode, string asmstr> 
    : InstAlpha<opcode, (ops GPRC:$RA, s21imm:$DISP), asmstr> {
  bits<5> Ra;
  bits<21> disp;

  let Inst{25-21} = Ra;
  let Inst{20-0} = disp;
}
class BFormD<bits<6> opcode, string asmstr> 
    : InstAlpha<opcode, (ops s21imm:$DISP), asmstr> {
  bits<5> Ra = 31;
  bits<21> disp;

  let Inst{25-21} = Ra;
  let Inst{20-0} = disp;
}

let isBranch = 1, isTerminator = 1 in
class FBForm<bits<6> opcode, string asmstr> 
    : InstAlpha<opcode, (ops F8RC:$RA, s21imm:$DISP), asmstr> {
  bits<5> Ra;
  bits<21> disp;

  let Inst{25-21} = Ra;
  let Inst{20-0} = disp;
}

//3.3.3
class OForm<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern> 
        : InstAlpha<opcode, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), asmstr> {
  let Pattern = pattern;

  bits<5> Rc;
  bits<5> Ra;
  bits<5> Rb;
  bits<7> Function = fun;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-13} = 0;
  let Inst{12} = 0;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

class OForm2<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern> 
        : InstAlpha<opcode, (ops GPRC:$RC, GPRC:$RB), asmstr> {
  let Pattern = pattern;

  bits<5> Rc;
  bits<5> Rb;
  bits<7> Function = fun;

  let Inst{25-21} = 31;
  let Inst{20-16} = Rb;
  let Inst{15-13} = 0;
  let Inst{12} = 0;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

class OForm4<bits<6> opcode, bits<7> fun, string asmstr> 
        : InstAlpha<opcode, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND), asmstr> {
  bits<5> Rc;
  bits<5> Rb;
  bits<5> Ra;
  bits<7> Function = fun;

  let isTwoAddress = 1;
  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-13} = 0;
  let Inst{12} = 0;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}


class OFormL<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern> 
        : InstAlpha<opcode, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), asmstr> {
  let Pattern = pattern;

  bits<5> Rc;
  bits<5> Ra;
  bits<8> LIT;
  bits<7> Function = fun;

  let Inst{25-21} = Ra;
  let Inst{20-13} = LIT;
  let Inst{12} = 1;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

class OForm4L<bits<6> opcode, bits<7> fun, string asmstr> 
        : InstAlpha<opcode, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND), asmstr> {
  bits<5> Rc;
  bits<8> LIT;
  bits<5> Ra;
  bits<7> Function = fun;

  let isTwoAddress = 1;
  let Inst{25-21} = Ra;
  let Inst{20-13} = LIT;
  let Inst{12} = 1;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

//3.3.4
class FPForm<bits<6> opcode, bits<11> fun, string asmstr, list<dag> pattern> 
        : InstAlphaAlt<opcode, asmstr> {
  let Pattern = pattern;

  bits<5> Fc;
  bits<5> Fa;
  bits<5> Fb;
  bits<11> Function = fun;

  let Inst{25-21} = Fa;
  let Inst{20-16} = Fb;
  let Inst{15-5} = Function;
  let Inst{4-0} = Fc;
}

//3.3.5
class PALForm<bits<6> opcode, dag OL, string asmstr> : InstAlpha<opcode, OL, asmstr> {
  bits<26> Function;

  let Inst{25-0} = Function;
}


// Pseudo instructions.
class PseudoInstAlpha<dag OL, string nm, list<dag> pattern> : InstAlpha<0, OL, nm>  {
  let Pattern = pattern;

}
