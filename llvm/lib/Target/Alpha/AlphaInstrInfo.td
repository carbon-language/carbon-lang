//===- AlphaInstrInfo.td - The Alpha Instruction Set -------*- tablegen -*-===//
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

include "AlphaInstrFormats.td"

//********************
//Paterns for matching
//********************
def immUExt8  : PatLeaf<(imm), [{
  // immUExt8 predicate - True if the immediate fits in a 8-bit zero extended
  // field.  Used by instructions like 'addi'.
  return (unsigned long)N->getValue() == (unsigned char)N->getValue();
}]>;
def intop : PatFrag<(ops node:$op), (sext_inreg node:$op, i32)>;


  // //#define FP    $15
  // //#define RA    $26
  // //#define PV    $27
  // //#define GP    $29
  // //#define SP    $30

def PHI : PseudoInstAlpha<(ops variable_ops), "#phi">;
def IDEF : PseudoInstAlpha<(ops GPRC:$RA), "#idef $RA">;
def WTF : PseudoInstAlpha<(ops variable_ops), "#wtf">;
def ADJUSTSTACKUP : PseudoInstAlpha<(ops variable_ops), "ADJUP">;
def ADJUSTSTACKDOWN : PseudoInstAlpha<(ops variable_ops), "ADJDOWN">;
def ALTENT : PseudoInstAlpha<(ops s64imm:$TARGET), "$TARGET:\n">;
def PCLABEL : PseudoInstAlpha<(ops s64imm:$num), "PCMARKER_$num:\n">;
def MEMLABEL : PseudoInstAlpha<(ops s64imm:$i, s64imm:$j, s64imm:$k, s64imm:$m),
         "LSMARKER$$$i$$$j$$$k$$$m:\n">;

//*****************
//These are shortcuts, the assembler expands them
//*****************
//AT = R28
//T0-T7 = R1 - R8
//T8-T11 = R22-R25

//An even better improvement on the Int = SetCC(FP):  SelectCC!
//These are evil because they hide control flow in a MBB
//really the ISel should emit multiple MBB
let isTwoAddress = 1 in {
//Conditional move of an int based on a FP CC
  def CMOVEQ_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, GPRC:$RSRC_T, FPRC:$RCOND),
                                  "fbne $RCOND, 42f\n\tbis $RSRC_T,$RSRC_T,$RDEST\n42:\n">;
  def CMOVEQi_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, u8imm:$L, FPRC:$RCOND),
                                  "fbne $RCOND, 42f\n\taddq $$31,$L,$RDEST\n42:\n">;

  def CMOVNE_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, GPRC:$RSRC_T, FPRC:$RCOND),
                                  "fbeq $RCOND, 42f\n\tbis $RSRC_T,$RSRC_T,$RDEST\n42:\n">;
  def CMOVNEi_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, u8imm:$L, FPRC:$RCOND),
                                  "fbeq $RCOND, 42f\n\taddq $$31,$L,$RDEST\n42:\n">;
//Conditional move of an FP based on a Int CC
  def FCMOVEQ_INT : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, GPRC:$RSRC_T, FPRC:$RCOND),
                                  "bne $RCOND, 42f\n\tcpys $RSRC_T,$RSRC_T,$RDEST\n42:\n">;
  def FCMOVNE_INT : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, GPRC:$RSRC_T, FPRC:$RCOND),
                                  "beq $RCOND, 42f\n\tcpys $RSRC_T,$RSRC_T,$RDEST\n42:\n">;
}

//***********************
//Real instructions
//***********************

//Operation Form:

let isTwoAddress = 1 in {
//conditional moves, int
 def CMOVEQ   : OcmForm<  0x11, 0x24, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmoveq $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND =  zero
 def CMOVEQi  : OcmFormL< 0x11, 0x24, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmoveq $RCOND,$L,$RDEST">; //CMOVE if RCOND =  zero
 def CMOVGE   : OcmForm<  0x11, 0x46, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovge $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND >= zero
 def CMOVGEi  : OcmFormL< 0x11, 0x46, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovge $RCOND,$L,$RDEST">; //CMOVE if RCOND >= zero
 def CMOVGT   : OcmForm<  0x11, 0x66, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovgt $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND > zero
 def CMOVGTi  : OcmFormL< 0x11, 0x66, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovgt $RCOND,$L,$RDEST">; //CMOVE if RCOND > zero
 def CMOVLBC  : OcmForm<  0x11, 0x16, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovlbc $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND low bit clear
 def CMOVLBCi : OcmFormL< 0x11, 0x16, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovlbc $RCOND,$L,$RDEST">; //CMOVE if RCOND low bit clear
 def CMOVLBS  : OcmForm<  0x11, 0x14, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovlbs $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND low bit set
 def CMOVLBSi : OcmFormL< 0x11, 0x14, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovlbs $RCOND,$L,$RDEST">; //CMOVE if RCOND low bit set
 def CMOVLE   : OcmForm<  0x11, 0x64, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovle $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND <= zero
 def CMOVLEi  : OcmFormL< 0x11, 0x64, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovle $RCOND,$L,$RDEST">; //CMOVE if RCOND <= zero
 def CMOVLT   : OcmForm<  0x11, 0x44, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovlt $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND < zero
 def CMOVLTi  : OcmFormL< 0x11, 0x44, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovlt $RCOND,$L,$RDEST">; //CMOVE if RCOND < zero
 def CMOVNE   : OcmForm<  0x11, 0x26, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovne $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND != zero
 def CMOVNEi  : OcmFormL< 0x11, 0x26, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovne $RCOND,$L,$RDEST">; //CMOVE if RCOND != zero

//conditional moves, fp
 def FCMOVEQ : FPFormCM<0x17, 0x02A, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmoveq $RCOND,$RSRC,$RDEST">; //FCMOVE if = zero
 def FCMOVGE : FPFormCM<0x17, 0x02D, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovge $RCOND,$RSRC,$RDEST">; //FCMOVE if >= zero
 def FCMOVGT : FPFormCM<0x17, 0x02F, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovgt $RCOND,$RSRC,$RDEST">; //FCMOVE if > zero
 def FCMOVLE : FPFormCM<0x17, 0x02E, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovle $RCOND,$RSRC,$RDEST">; //FCMOVE if <= zero
 def FCMOVLT : FPFormCM<0x17, 0x02, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovlt $RCOND,$RSRC,$RDEST">; // FCMOVE if < zero
 def FCMOVNE : FPFormCM<0x17, 0x02B, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovne $RCOND,$RSRC,$RDEST">; //FCMOVE if != zero
}

def ADDL     : OForm< 0x10, 0x00, "addl $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (add GPRC:$RA, GPRC:$RB)))]>;
def ADDLi    : OFormL<0x10, 0x00, "addl $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (add GPRC:$RA, immUExt8:$L)))]>;
def ADDQ     : OForm< 0x10, 0x20, "addq $RA,$RB,$RC",
                      [(set GPRC:$RC, (add GPRC:$RA, GPRC:$RB))]>;
def ADDQi    : OFormL<0x10, 0x20, "addq $RA,$L,$RC",
                      [(set GPRC:$RC, (add GPRC:$RA, immUExt8:$L))]>;
//def AMASK    : OForm< 0x11, 0x61, "AMASK $RA,$RB,$RC", []>; //Architecture mask
//def AMASKi   : OFormL<0x11, 0x61, "AMASK $RA,$L,$RC", []>; //Architecture mask
def AND      : OForm< 0x11, 0x00, "and $RA,$RB,$RC",
                      [(set GPRC:$RC, (and GPRC:$RA, GPRC:$RB))]>;
def ANDi     : OFormL<0x11, 0x00, "and $RA,$L,$RC",
                      [(set GPRC:$RC, (and GPRC:$RA, immUExt8:$L))]>;
def BIC      : OForm< 0x11, 0x08, "bic $RA,$RB,$RC",
                      [(set GPRC:$RC, (and GPRC:$RA, (not GPRC:$RB)))]>;
def BICi     : OFormL<0x11, 0x08, "bic $RA,$L,$RC", []>;
//                      [(set GPRC:$RC, (and GPRC:$RA, (not immUExt8:$L)))]>; //FIXME?
def BIS      : OForm< 0x11, 0x20, "bis $RA,$RB,$RC",
                      [(set GPRC:$RC, (or GPRC:$RA, GPRC:$RB))]>;
def BISi     : OFormL<0x11, 0x20, "bis $RA,$L,$RC",
                      [(set GPRC:$RC, (or GPRC:$RA, immUExt8:$L))]>;
def CTLZ     : OFormT<0x1C, 0x32, "CTLZ $RB,$RC", 
                      [(set GPRC:$RC, (ctlz GPRC:$RB))]>;
def CTPOP    : OFormT<0x1C, 0x30, "CTPOP $RB,$RC", 
                      [(set GPRC:$RC, (ctpop GPRC:$RB))]>;
def CTTZ     : OFormT<0x1C, 0x33, "CTTZ $RB,$RC", 
                      [(set GPRC:$RC, (cttz GPRC:$RB))]>;
def EQV      : OForm< 0x11, 0x48, "eqv $RA,$RB,$RC",
                      [(set GPRC:$RC, (xor GPRC:$RA, (not GPRC:$RB)))]>;
def EQVi     : OFormL<0x11, 0x48, "eqv $RA,$L,$RC", []>;
//                      [(set GPRC:$RC, (xor GPRC:$RA, (not immUExt8:$L)))]>;
//def EXTBL    : OForm< 0x12, 0x06, "EXTBL $RA,$RB,$RC", []>; //Extract byte low
//def EXTBLi   : OFormL<0x12, 0x06, "EXTBL $RA,$L,$RC", []>; //Extract byte low
//def EXTLH    : OForm< 0x12, 0x6A, "EXTLH $RA,$RB,$RC", []>; //Extract longword high
//def EXTLHi   : OFormL<0x12, 0x6A, "EXTLH $RA,$L,$RC", []>; //Extract longword high
//def EXTLL    : OForm< 0x12, 0x26, "EXTLL $RA,$RB,$RC", []>; //Extract longword low
//def EXTLLi   : OFormL<0x12, 0x26, "EXTLL $RA,$L,$RC", []>; //Extract longword low
//def EXTQH    : OForm< 0x12, 0x7A, "EXTQH $RA,$RB,$RC", []>; //Extract quadword high
//def EXTQHi   : OFormL<0x12, 0x7A, "EXTQH $RA,$L,$RC", []>; //Extract quadword high
//def EXTQ     : OForm< 0x12, 0x36, "EXTQ $RA,$RB,$RC", []>; //Extract quadword low
//def EXTQi    : OFormL<0x12, 0x36, "EXTQ $RA,$L,$RC", []>; //Extract quadword low
//def EXTWH    : OForm< 0x12, 0x5A, "EXTWH $RA,$RB,$RC", []>; //Extract word high
//def EXTWHi   : OFormL<0x12, 0x5A, "EXTWH $RA,$L,$RC", []>; //Extract word high
//def EXTWL    : OForm< 0x12, 0x16, "EXTWL $RA,$RB,$RC", []>; //Extract word low
//def EXTWLi   : OFormL<0x12, 0x16, "EXTWL $RA,$L,$RC", []>; //Extract word low
//def IMPLVER  : OForm< 0x11, 0x6C, "IMPLVER $RA,$RB,$RC", []>; //Implementation version
//def IMPLVERi : OFormL<0x11, 0x6C, "IMPLVER $RA,$L,$RC", []>; //Implementation version
//def INSBL    : OForm< 0x12, 0x0B, "INSBL $RA,$RB,$RC", []>; //Insert byte low
//def INSBLi   : OFormL<0x12, 0x0B, "INSBL $RA,$L,$RC", []>; //Insert byte low
//def INSLH    : OForm< 0x12, 0x67, "INSLH $RA,$RB,$RC", []>; //Insert longword high
//def INSLHi   : OFormL<0x12, 0x67, "INSLH $RA,$L,$RC", []>; //Insert longword high
//def INSLL    : OForm< 0x12, 0x2B, "INSLL $RA,$RB,$RC", []>; //Insert longword low
//def INSLLi   : OFormL<0x12, 0x2B, "INSLL $RA,$L,$RC", []>; //Insert longword low
//def INSQH    : OForm< 0x12, 0x77, "INSQH $RA,$RB,$RC", []>; //Insert quadword high
//def INSQHi   : OFormL<0x12, 0x77, "INSQH $RA,$L,$RC", []>; //Insert quadword high
//def INSQL    : OForm< 0x12, 0x3B, "INSQL $RA,$RB,$RC", []>; //Insert quadword low
//def INSQLi   : OFormL<0x12, 0x3B, "INSQL $RA,$L,$RC", []>; //Insert quadword low
//def INSWH    : OForm< 0x12, 0x57, "INSWH $RA,$RB,$RC", []>; //Insert word high
//def INSWHi   : OFormL<0x12, 0x57, "INSWH $RA,$L,$RC", []>; //Insert word high
//def INSWL    : OForm< 0x12, 0x1B, "INSWL $RA,$RB,$RC", []>; //Insert word low
//def INSWLi   : OFormL<0x12, 0x1B, "INSWL $RA,$L,$RC", []>; //Insert word low
//def MSKBL    : OForm< 0x12, 0x02, "MSKBL $RA,$RB,$RC", []>; //Mask byte low
//def MSKBLi   : OFormL<0x12, 0x02, "MSKBL $RA,$L,$RC", []>; //Mask byte low
//def MSKLH    : OForm< 0x12, 0x62, "MSKLH $RA,$RB,$RC", []>; //Mask longword high
//def MSKLHi   : OFormL<0x12, 0x62, "MSKLH $RA,$L,$RC", []>; //Mask longword high
//def MSKLL    : OForm< 0x12, 0x22, "MSKLL $RA,$RB,$RC", []>; //Mask longword low
//def MSKLLi   : OFormL<0x12, 0x22, "MSKLL $RA,$L,$RC", []>; //Mask longword low
//def MSKQH    : OForm< 0x12, 0x72, "MSKQH $RA,$RB,$RC", []>; //Mask quadword high
//def MSKQHi   : OFormL<0x12, 0x72, "MSKQH $RA,$L,$RC", []>; //Mask quadword high
//def MSKQL    : OForm< 0x12, 0x32, "MSKQL $RA,$RB,$RC", []>; //Mask quadword low
//def MSKQLi   : OFormL<0x12, 0x32, "MSKQL $RA,$L,$RC", []>; //Mask quadword low
//def MSKWH    : OForm< 0x12, 0x52, "MSKWH $RA,$RB,$RC", []>; //Mask word high
//def MSKWHi   : OFormL<0x12, 0x52, "MSKWH $RA,$L,$RC", []>; //Mask word high
//def MSKWL    : OForm< 0x12, 0x12, "MSKWL $RA,$RB,$RC", []>; //Mask word low
//def MSKWLi   : OFormL<0x12, 0x12, "MSKWL $RA,$L,$RC", []>; //Mask word low

// Some Alpha pattern fragments to make things more terse and easier to read.
def add4  : PatFrag<(ops node:$op1, node:$op2),
                    (add (shl node:$op1, 2), node:$op2)>;
def sub4  : PatFrag<(ops node:$op1, node:$op2),
                    (sub (shl node:$op1, 2), node:$op2)>;
def add8  : PatFrag<(ops node:$op1, node:$op2),
                    (add (shl node:$op1, 3), node:$op2)>;
def sub8  : PatFrag<(ops node:$op1, node:$op2),
                    (sub (shl node:$op1, 3), node:$op2)>;
                        
def MULL     : OForm< 0x13, 0x00, "mull $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (mul GPRC:$RA, GPRC:$RB)))]>;
def MULLi    : OFormL<0x13, 0x00, "mull $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (mul GPRC:$RA, immUExt8:$L)))]>;
def MULQ     : OForm< 0x13, 0x20, "mulq $RA,$RB,$RC",
                      [(set GPRC:$RC, (mul GPRC:$RA, GPRC:$RB))]>;
def MULQi    : OFormL<0x13, 0x20, "mulq $RA,$L,$RC",
                      [(set GPRC:$RC, (mul GPRC:$RA, immUExt8:$L))]>;
def ORNOT    : OForm< 0x11, 0x28, "ornot $RA,$RB,$RC",
                      [(set GPRC:$RC, (or GPRC:$RA, (not GPRC:$RB)))]>;
def ORNOTi   : OFormL<0x11, 0x28, "ornot $RA,$L,$RC", []>;
//                      [(set GPRC:$RC, (or GPRC:$RA, (not immUExt8:$L)))]>;
def S4ADDL   : OForm< 0x10, 0x02, "s4addl $RA,$RB,$RC", 
                      [(set GPRC:$RC, (intop (add4 GPRC:$RA, GPRC:$RB)))]>;
def S4ADDLi  : OFormL<0x10, 0x02, "s4addl $RA,$L,$RC", 
                      [(set GPRC:$RC, (intop (add4 GPRC:$RA, immUExt8:$L)))]>;
def S4ADDQ   : OForm< 0x10, 0x22, "s4addq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (add4 GPRC:$RA, GPRC:$RB))]>;
def S4ADDQi  : OFormL<0x10, 0x22, "s4addq $RA,$L,$RC", 
                      [(set GPRC:$RC, (add4 GPRC:$RA, immUExt8:$L))]>;
def S4SUBL   : OForm< 0x10, 0x0B, "s4subl $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (sub4 GPRC:$RA, GPRC:$RB)))]>;
def S4SUBLi  : OFormL<0x10, 0x0B, "s4subl $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (sub4 GPRC:$RA, immUExt8:$L)))]>;
def S4SUBQ   : OForm< 0x10, 0x2B, "s4subq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (sub4 GPRC:$RA, GPRC:$RB))]>;
def S4SUBQi  : OFormL<0x10, 0x2B, "s4subq $RA,$L,$RC", 
                      [(set GPRC:$RC, (sub4 GPRC:$RA, immUExt8:$L))]>;
def S8ADDL   : OForm< 0x10, 0x12, "s8addl $RA,$RB,$RC", 
                      [(set GPRC:$RC, (intop (add8 GPRC:$RA, GPRC:$RB)))]>;
def S8ADDLi  : OFormL<0x10, 0x12, "s8addl $RA,$L,$RC", 
                      [(set GPRC:$RC, (intop (add8 GPRC:$RA, immUExt8:$L)))]>;
def S8ADDQ   : OForm< 0x10, 0x32, "s8addq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (add8 GPRC:$RA, GPRC:$RB))]>;
def S8ADDQi  : OFormL<0x10, 0x32, "s8addq $RA,$L,$RC", 
                      [(set GPRC:$RC, (add8 GPRC:$RA, immUExt8:$L))]>;
def S8SUBL   : OForm< 0x10, 0x1B, "s8subl $RA,$RB,$RC", 
                      [(set GPRC:$RC, (intop (sub8 GPRC:$RA, GPRC:$RB)))]>;
def S8SUBLi  : OFormL<0x10, 0x1B, "s8subl $RA,$L,$RC", 
                      [(set GPRC:$RC, (intop (sub8 GPRC:$RA, immUExt8:$L)))]>;
def S8SUBQ   : OForm< 0x10, 0x3B, "s8subq $RA,$RB,$RC", 
                      [(set GPRC:$RC, (sub8 GPRC:$RA, GPRC:$RB))]>;
def S8SUBQi  : OFormL<0x10, 0x3B, "s8subq $RA,$L,$RC", 
                      [(set GPRC:$RC, (sub8 GPRC:$RA, immUExt8:$L))]>;
def SEXTB    : OFormT<0x1C, 0x00, "sextb $RB,$RC", 
                      [(set GPRC:$RC, (sext_inreg GPRC:$RB, i8))]>;
def SEXTW    : OFormT<0x1C, 0x01, "sextw $RB,$RC", 
                      [(set GPRC:$RC, (sext_inreg GPRC:$RB, i16))]>;
def SL       : OForm< 0x12, 0x39, "sll $RA,$RB,$RC",
                      [(set GPRC:$RC, (shl GPRC:$RA, GPRC:$RB))]>;
def SLi      : OFormL<0x12, 0x39, "sll $RA,$L,$RC",
                      [(set GPRC:$RC, (shl GPRC:$RA, immUExt8:$L))]>;
def SRA      : OForm< 0x12, 0x3C, "sra $RA,$RB,$RC",
                      [(set GPRC:$RC, (sra GPRC:$RA, GPRC:$RB))]>;
def SRAi     : OFormL<0x12, 0x3C, "sra $RA,$L,$RC",
                      [(set GPRC:$RC, (sra GPRC:$RA, immUExt8:$L))]>;
def SRL      : OForm< 0x12, 0x34, "srl $RA,$RB,$RC",
                      [(set GPRC:$RC, (srl GPRC:$RA, GPRC:$RB))]>;
def SRLi     : OFormL<0x12, 0x34, "srl $RA,$L,$RC",
                      [(set GPRC:$RC, (srl GPRC:$RA, immUExt8:$L))]>;
def SUBL     : OForm< 0x10, 0x09, "subl $RA,$RB,$RC",
                      [(set GPRC:$RC, (intop (sub GPRC:$RA, GPRC:$RB)))]>;
def SUBLi    : OFormL<0x10, 0x09, "subl $RA,$L,$RC",
                      [(set GPRC:$RC, (intop (sub GPRC:$RA, immUExt8:$L)))]>;
def SUBQ     : OForm< 0x10, 0x29, "subq $RA,$RB,$RC",
                      [(set GPRC:$RC, (sub GPRC:$RA, GPRC:$RB))]>;
def SUBQi    : OFormL<0x10, 0x29, "subq $RA,$L,$RC",
                      [(set GPRC:$RC, (sub GPRC:$RA, immUExt8:$L))]>;
def UMULH    : OForm< 0x13, 0x30, "umulh $RA,$RB,$RC",
                      [(set GPRC:$RC, (mulhu GPRC:$RA, GPRC:$RB))]>;                     
def UMULHi   : OFormL<0x13, 0x30, "umulh $RA,$L,$RC", 
                      [(set GPRC:$RC, (mulhu GPRC:$RA, immUExt8:$L))]>;
def XOR      : OForm< 0x11, 0x40, "xor $RA,$RB,$RC",
                      [(set GPRC:$RC, (xor GPRC:$RA, GPRC:$RB))]>;
def XORi     : OFormL<0x11, 0x40, "xor $RA,$L,$RC",
                      [(set GPRC:$RC, (xor GPRC:$RA, immUExt8:$L))]>;
def ZAP      : OForm< 0x12, 0x30, "zap $RA,$RB,$RC", []>; //Zero bytes
def ZAPi     : OFormL<0x12, 0x30, "zap $RA,$L,$RC", []>; //Zero bytes
def ZAPNOT   : OForm< 0x12, 0x31, "zapnot $RA,$RB,$RC", []>; //Zero bytes not
def ZAPNOTi  : OFormL<0x12, 0x31, "zapnot $RA,$L,$RC", []>; //Zero bytes not

//Comparison, int
def CMPBGE   : OForm< 0x10, 0x0F, "cmpbge $RA,$RB,$RC", []>; //Compare byte
def CMPBGEi  : OFormL<0x10, 0x0F, "cmpbge $RA,$L,$RC", []>; //Compare byte
def CMPEQ    : OForm< 0x10, 0x2D, "cmpeq $RA,$RB,$RC", []>; //Compare signed quadword equal
def CMPEQi   : OFormL<0x10, 0x2D, "cmpeq $RA,$L,$RC", []>; //Compare signed quadword equal
def CMPLE    : OForm< 0x10, 0x6D, "cmple $RA,$RB,$RC", []>; //Compare signed quadword less than or equal
def CMPLEi   : OFormL<0x10, 0x6D, "cmple $RA,$L,$RC", []>; //Compare signed quadword less than or equal
def CMPLT    : OForm< 0x10, 0x4D, "cmplt $RA,$RB,$RC", []>; //Compare signed quadword less than
def CMPLTi   : OFormL<0x10, 0x4D, "cmplt $RA,$L,$RC", []>; //Compare signed quadword less than
def CMPULE   : OForm< 0x10, 0x3D, "cmpule $RA,$RB,$RC", []>; //Compare unsigned quadword less than or equal
def CMPULEi  : OFormL<0x10, 0x3D, "cmpule $RA,$L,$RC", []>; //Compare unsigned quadword less than or equal
def CMPULT   : OForm< 0x10, 0x1D, "cmpult $RA,$RB,$RC", []>; //Compare unsigned quadword less than
def CMPULTi  : OFormL<0x10, 0x1D, "cmpult $RA,$L,$RC", []>; //Compare unsigned quadword less than

//Comparison, FP
def CMPTEQ : FPForm<0x16, 0x0A5, "cmpteq/su $RA,$RB,$RC">;  //Compare T_floating equal
def CMPTLE : FPForm<0x16, 0x0A7, "cmptle/su $RA,$RB,$RC">;  //Compare T_floating less than or equal
def CMPTLT : FPForm<0x16, 0x0A6, "cmptlt/su $RA,$RB,$RC">;  //Compare T_floating less than
def CMPTUN : FPForm<0x16, 0x0A4, "cmptun/su $RA,$RB,$RC">;  //Compare T_floating unordered

//There are in the Multimedia extentions, so let's not use them yet
//def MAXSB8  : OForm<0x1C, 0x3E, "MAXSB8 $RA,$RB,$RC">; //Vector signed byte maximum
//def MAXSW4 : OForm< 0x1C, 0x3F, "MAXSW4 $RA,$RB,$RC">; //Vector signed word maximum
//def MAXUB8  : OForm<0x1C, 0x3C, "MAXUB8 $RA,$RB,$RC">; //Vector unsigned byte maximum
//def MAXUW4 : OForm< 0x1C, 0x3D, "MAXUW4 $RA,$RB,$RC">; //Vector unsigned word maximum
//def MINSB8 : OForm< 0x1C, 0x38, "MINSB8 $RA,$RB,$RC">; //Vector signed byte minimum
//def MINSW4 : OForm< 0x1C, 0x39, "MINSW4 $RA,$RB,$RC">; //Vector signed word minimum
//def MINUB8 : OForm< 0x1C, 0x3A, "MINUB8 $RA,$RB,$RC">; //Vector unsigned byte minimum
//def MINUW4 : OForm< 0x1C, 0x3B, "MINUW4 $RA,$RB,$RC">; //Vector unsigned word minimum
//def PERR : OForm< 0x1C, 0x31, "PERR $RA,$RB,$RC">; //Pixel error
//def PKLB : OForm< 0x1C, 0x37, "PKLB $RA,$RB,$RC">; //Pack longwords to bytes
//def PKWB  : OForm<0x1C, 0x36, "PKWB $RA,$RB,$RC">; //Pack words to bytes
//def UNPKBL : OForm< 0x1C, 0x35, "UNPKBL $RA,$RB,$RC">; //Unpack bytes to longwords
//def UNPKBW : OForm< 0x1C, 0x34, "UNPKBW $RA,$RB,$RC">; //Unpack bytes to words

//End operate

let isReturn = 1, isTerminator = 1 in 
  def RET : MbrForm< 0x1A, 0x02, (ops GPRC:$RD, GPRC:$RS, s64imm:$DISP), "ret $RD,($RS),$DISP">; //Return from subroutine
//DAG Version:
let isReturn = 1, isTerminator = 1, Ra = 31, Rb = 26, disp = 1, Uses = [R26] in 
  def RETDAG : MbrForm< 0x1A, 0x02, (ops), "ret $$31,($$26),1">; //Return from subroutine

def JMP : MbrForm< 0x1A, 0x00, (ops GPRC:$RD, GPRC:$RS, GPRC:$DISP), "jmp $RD,($RS),$DISP">; //Jump
let isCall = 1,
    Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19,
            R20, R21, R22, R23, R24, R25, R27, R28, R29,
            F0, F1,
            F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30], Uses = [R29] in {
    def JSR : MbrForm< 0x1A, 0x01, (ops GPRC:$RD, GPRC:$RS, s14imm:$DISP), "jsr $RD,($RS),$DISP">; //Jump to subroutine
    def BSR : BForm<0x34, "bsr $RA,$DISP">; //Branch to subroutine
}
let isCall = 1, Defs = [R24, R25, R27, R28], Uses = [R24, R25] in
  def JSRs : MbrForm< 0x1A, 0x01, (ops GPRC:$RD, GPRC:$RS, s14imm:$DISP), "jsr $RD,($RS),$DISP">; //Jump to div or rem

def JSR_COROUTINE : MbrForm< 0x1A, 0x03, (ops GPRC:$RD, GPRC:$RS, s14imm:$DISP), "jsr_coroutine $RD,($RS),$DISP">; //Jump to subroutine return
def BR : BForm<0x30, "br $RA,$DISP">; //Branch

//Stores, int
def STB : MForm<0x0E, "stb $RA,$DISP($RB)">; // Store byte
def STW : MForm<0x0D, "stw $RA,$DISP($RB)">; // Store word
def STL : MForm<0x2C, "stl $RA,$DISP($RB)">; // Store longword
def STQ : MForm<0x2D, "stq $RA,$DISP($RB)">; //Store quadword

//Loads, int
def LDL : MForm<0x28,  "ldl $RA,$DISP($RB)">; // Load sign-extended longword
def LDQ : MForm<0x29,  "ldq $RA,$DISP($RB)">; //Load quadword
def LDBU : MForm<0x0A, "ldbu $RA,$DISP($RB)">; //Load zero-extended byte
def LDWU : MForm<0x0C, "ldwu $RA,$DISP($RB)">; //Load zero-extended word

//Stores, float
def STS : MForm<0x26, "sts $RA,$DISP($RB)">; //Store S_floating
def STT : MForm<0x27, "stt $RA,$DISP($RB)">; //Store T_floating

//Loads, float
def LDS : MForm<0x22, "lds $RA,$DISP($RB)">; //Load S_floating
def LDT : MForm<0x23, "ldt $RA,$DISP($RB)">; //Load T_floating

//Load address
def LDA : MForm<0x08,  "lda $RA,$DISP($RB)">;  //Load address
def LDAH : MForm<0x09, "ldah $RA,$DISP($RB)">;  //Load address high


//Loads, int, Rellocated Low form
def LDLr : MForm<0x28,  "ldl $RA,$DISP($RB)\t\t!gprellow">; // Load sign-extended longword
def LDQr : MForm<0x29,  "ldq $RA,$DISP($RB)\t\t!gprellow">; //Load quadword
def LDBUr : MForm<0x0A, "ldbu $RA,$DISP($RB)\t\t!gprellow">; //Load zero-extended byte
def LDWUr : MForm<0x0C, "ldwu $RA,$DISP($RB)\t\t!gprellow">; //Load zero-extended word

//Loads, float, Rellocated Low form
def LDSr : MForm<0x22, "lds $RA,$DISP($RB)\t\t!gprellow">; //Load S_floating
def LDTr : MForm<0x23, "ldt $RA,$DISP($RB)\t\t!gprellow">; //Load T_floating

//Load address, rellocated low and high form
def LDAr : MForm<0x08,  "lda $RA,$DISP($RB)\t\t!gprellow">;  //Load address
def LDAHr : MForm<0x09, "ldah $RA,$DISP($RB)\t\t!gprelhigh">;  //Load address high

//load address, rellocated gpdist form
def LDAg : MgForm<0x08,  "lda $RA,0($RB)\t\t!gpdisp!$NUM">;  //Load address
def LDAHg : MgForm<0x09, "ldah $RA,0($RB)\t\t!gpdisp!$NUM">;  //Load address


//Load quad, rellocated literal form
def LDQl : MForm<0x29, "ldq $RA,$DISP($RB)\t\t!literal">; //Load quadword

//Stores, int
def STBr : MForm<0x0E, "stb $RA,$DISP($RB)\t\t!gprellow">; // Store byte
def STWr : MForm<0x0D, "stw $RA,$DISP($RB)\t\t!gprellow">; // Store word
def STLr : MForm<0x2C, "stl $RA,$DISP($RB)\t\t!gprellow">; // Store longword
def STQr : MForm<0x2D, "stq $RA,$DISP($RB)\t\t!gprellow">; //Store quadword

//Stores, float
def STSr : MForm<0x26, "sts $RA,$DISP($RB)\t\t!gprellow">; //Store S_floating
def STTr : MForm<0x27, "stt $RA,$DISP($RB)\t\t!gprellow">; //Store T_floating


//Branches, int
def BEQ : BForm<0x39,  "beq $RA,$DISP">; //Branch if = zero
def BGE : BForm<0x3E,  "bge $RA,$DISP">; //Branch if >= zero
def BGT : BForm<0x3F,  "bgt $RA,$DISP">; //Branch if > zero
def BLBC : BForm<0x38, "blbc $RA,$DISP">; //Branch if low bit clear
def BLBS : BForm<0x3C, "blbs $RA,$DISP">; //Branch if low bit set
def BLE : BForm<0x3B,  "ble $RA,$DISP">; //Branch if <= zero
def BLT : BForm<0x3A,  "blt $RA,$DISP">; //Branch if < zero
def BNE : BForm<0x3D,  "bne $RA,$DISP">; //Branch if != zero

//Branches, float
def FBEQ : FBForm<0x31, "fbeq $RA,$DISP">; //Floating branch if =  zero
def FBGE : FBForm<0x36, "fbge $RA,$DISP">; //Floating branch if >= zero
def FBGT : FBForm<0x37, "fbgt $RA,$DISP">; //Floating branch if > zero
def FBLE : FBForm<0x33, "fble $RA,$DISP">; //Floating branch if <= zero
def FBLT : FBForm<0x32, "fblt $RA,$DISP">; //Floating branch if < zero
def FBNE : FBForm<0x35, "fbne $RA,$DISP">; //Floating branch if != zero

//Funky Floating point ops
def CPYS  : FPForm<0x17, 0x020, "cpys $RA,$RB,$RC">;  //Copy sign
def CPYSE : FPForm<0x17, 0x022, "cpyse $RA,$RB,$RC">; //Copy sign and exponent
def CPYSN : FPForm<0x17, 0x021, "cpysn $RA,$RB,$RC">; //Copy sign negate

//Basic Floating point ops
def ADDS  : FPForm<0x16, 0x580, "adds/su $RA,$RB,$RC">;  //Add S_floating
def ADDT  : FPForm<0x16, 0x5A0, "addt/su $RA,$RB,$RC">;  //Add T_floating
def SUBS  : FPForm<0x16, 0x581, "subs/su $RA,$RB,$RC">;  //Subtract S_floating
def SUBT  : FPForm<0x16, 0x5A1, "subt/su $RA,$RB,$RC">;  //Subtract T_floating
def DIVS  : FPForm<0x16, 0x583, "divs/su $RA,$RB,$RC">;  //Divide S_floating
def DIVT  : FPForm<0x16, 0x5A3, "divt/su $RA,$RB,$RC">;  //Divide T_floating
def MULS  : FPForm<0x16, 0x582, "muls/su $RA,$RB,$RC">;  //Multiply S_floating
def MULT  : FPForm<0x16, 0x5A2, "mult/su $RA,$RB,$RC">;  //Multiply T_floating
def SQRTS : FPForm<0x14, 0x58B, "sqrts/su $RA,$RB,$RC">;  //Square root S_floating
def SQRTT : FPForm<0x14, 0x5AB, "sqrtt/su $RA,$RB,$RC">;  //Square root T_floating

//INT reg to FP reg and back again
//not supported on 21164
def FTOIS : FPForm<0x1C, 0x078, "ftois $RA,$RC">; //Floating to integer move, S_floating
def FTOIT : FPForm<0x1C, 0x070, "ftoit $RA,$RC">; //Floating to integer move, T_floating
def ITOFS : FPForm<0x14, 0x004, "itofs $RA,$RC">; //Integer to floating move, S_floating
def ITOFT : FPForm<0x14, 0x024, "itoft $RA,$RC">; //Integer to floating move, T_floating

//CVTLQ F-P 17.010 Convert longword to quadword
//CVTQL F-P 17.030 Convert quadword to longword
//These use SW completion, may not have function code for that set right (matters for JIT)
def CVTQS : FPForm<0x16, 0x0BC, "cvtqs $RB,$RC">; //Convert quadword to S_floating
def CVTQT : FPForm<0x16, 0x0BE, "cvtqt $RB,$RC">; //Convert quadword to T_floating
def CVTST : FPForm<0x16, 0x2AC, "cvtsts $RB,$RC">; //Convert S_floating to T_floating
def CVTTQ : FPForm<0x16, 0x52F, "cvttq/svc $RB,$RC">; //Convert T_floating to quadword
def CVTTS : FPForm<0x16, 0x5AC, "cvtts/su $RB,$RC">; //Convert T_floating to S_floating

//S_floating : IEEE Single
//T_floating : IEEE Double

//Mnemonic Format Opcode Description

//CALL_PAL Pcd 00 Trap to PALcode
//ECB Mfc 18.E800 Evict cache block
//EXCB Mfc 18.0400 Exception barrier
//FETCH Mfc 18.8000 Prefetch data
//FETCH_M Mfc 18.A000 Prefetch data, modify intent

//LDL_L Mem 2A Load sign-extended longword locked
//LDQ_L Mem 2B Load quadword locked
//LDQ_U Mem 0B Load unaligned quadword
//MB Mfc 18.4000 Memory barrier
//RPCC Mfc 18.C000 Read process cycle counter

//STL_C Mem 2E Store longword conditional
//STQ_C Mem 2F Store quadword conditional
//STQ_U Mem 0F Store unaligned quadword

//TRAPB Mfc 18.0000 Trap barrier
//WH64 Mfc 18.F800 Write hint  64 bytes
//WMB Mfc 18.4400 Write memory barrier


//MF_FPCR F-P 17.025 Move from FPCR
//MT_FPCR F-P 17.024 Move to FPCR
