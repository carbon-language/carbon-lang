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

  // //#define FP    $15
  // //#define RA    $26
  // //#define PV    $27
  // //#define GP    $29
  // //#define SP    $30

def u8imm   : Operand<i8>;
def s14imm  : Operand<i16>;
def s16imm  : Operand<i16>;
def s21imm  : Operand<i32>;
def s64imm  : Operand<i64>;

def PHI : PseudoInstAlpha<(ops ), "#phi">;
def IDEF : PseudoInstAlpha<(ops GPRC:$RA), "#idef $RA">;
def WTF : PseudoInstAlpha<(ops ), "#wtf">;
def ADJUSTSTACKUP : PseudoInstAlpha<(ops ), "ADJUP">;
def ADJUSTSTACKDOWN : PseudoInstAlpha<(ops ), "ADJDOWN">;

//*****************
//These are shortcuts, the assembler expands them
//*****************
//AT = R28
//T0-T7 = R1 - R8
//T8-T11 = R22-R25

let Defs = [R29] in 
  let Uses = [R27] in
    def LDGP : PseudoInstAlpha<(ops), "ldgp $$29, 0($$27)">;

let isCall = 1,
    Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19,
            R20, R21, R22, R23, R24, R25, R26, R27, R28, R29,
            F0, F1,
            F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30],
    Uses = [R29] in
      def CALL : PseudoInstAlpha< (ops s64imm:$TARGET), "jsr $TARGET">; //Jump to subroutine

let isReturn = 1, isTerminator = 1 in
  def RETURN : PseudoInstAlpha<(ops ), "ret $$31,($$26),1">; //Return from subroutine

let Uses = [R29], Defs = [R28] in {
  def LOAD_ADDR : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "lda $RA,$DISP">;  //Load address
  def LDQ_SYM : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "ldq $RA,$DISP">; //Load quadword
  def LDS_SYM : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "lds $RA,$DISP">; //Load float
  def LDT_SYM : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "ldt $RA,$DISP">; //Load double
  def LDL_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "ldl $RA,$DISP">; // Load sign-extended longword
  def LDBU_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "ldbu $RA,$DISP">; //Load zero-extended byte
  def LDWU_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "ldwu $RA,$DISP">; //Load zero-extended word
  def LDW_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "ldw $RA,$DISP">; // Load sign-extended word
  def LDB_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "ldb $RA,$DISP">; //Load byte

  def LDW : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldw $RA,$DISP($RB)">; // Load sign-extended word
  def LDB : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldb $RA,$DISP($RB)">; //Load byte

  def STB_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "stb $RA,$DISP">; // Store byte
  def STW_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "stw $RA,$DISP">; // Store word
  def STL_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "stl $RA,$DISP">; // Store longword
  def STQ_SYM : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP), "stq $RA,$DISP">; //Store quadword

  def STS_SYM : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "sts $RA,$DISP">; //store float
  def STT_SYM : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "stt $RA,$DISP">; //store double
}


//RESULTS of these go to R27
let Uses = [R29], 
    Defs = [R28, R23, R24, R25, R27] in
{
  def REMQU : PseudoInstAlpha<(ops GPRC:$RA, GPRC:$RB), "remqu $RA,$RB,$$27">; //unsigned remander
  def REMQ  : PseudoInstAlpha<(ops GPRC:$RA, GPRC:$RB), "remq $RA,$RB,$$27">; //signed remander
  def DIVQU : PseudoInstAlpha<(ops GPRC:$RA, GPRC:$RB), "divqu $RA,$RB,$$27">; //unsigned division
  def DIVQ  : PseudoInstAlpha<(ops GPRC:$RA, GPRC:$RB), "divq $RA,$RB,$$27">; //signed division
}

//This is an improvement on the old style setcc (FP)
//def CC2INT_INV : PseudoInstAlpha<(ops GPRC:$RES, FPRC:$COND), 
//  "lda $RES,1($$31)\n\tfbeq $COND, 42f\n\tbis $$31,$$31,$RES\n42:\n">;
//def CC2INT : PseudoInstAlpha<(ops GPRC:$RES, FPRC:$COND), 
//  "lda $RES,1($$31)\n\tfbne $COND, 42f\n\tbis $$31,$$31,$RES\n42:\n">;

//An even better improvement on the Int = SetCC(FP):  SelectCC!
let isTwoAddress = 1 in {
  def CMOVEQ_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, GPRC:$RSRC_T, FPRC:$RCOND),
                                  "fbne $RCOND, 42f\n\tbis $RSRC_T,$RSRC_T,$RDEST\n42:\n">;
  def CMOVEQi_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, u8imm:$L, FPRC:$RCOND),
                                  "fbne $RCOND, 42f\n\taddq $$31,$L,$RDEST\n42:\n">;

  def CMOVNE_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, GPRC:$RSRC_T, FPRC:$RCOND),
                                  "fbeq $RCOND, 42f\n\tbis $RSRC_T,$RSRC_T,$RDEST\n42:\n">;
  def CMOVNEi_FP : PseudoInstAlpha<(ops GPRC:$RDEST, GPRC:$RSRC_F, u8imm:$L, FPRC:$RCOND),
                                  "fbeq $RCOND, 42f\n\taddq $$31,$L,$RDEST\n42:\n">;

}

//***********************
//Real instructions
//***********************

//Operation Form:

let isTwoAddress = 1 in {
//conditional moves, int
 def CMOVEQ   : OForm<  0x11, 0x24, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmoveq $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND =  zero
 def CMOVEQi  : OFormL< 0x11, 0x24, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmoveq $RCOND,$L,$RDEST">; //CMOVE if RCOND =  zero
 def CMOVGE   : OForm<  0x11, 0x46, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "CMOVGE $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND >= zero
 def CMOVGEi  : OFormL< 0x11, 0x46, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "CMOVGE $RCOND,$L,$RDEST">; //CMOVE if RCOND >= zero
 def CMOVGT   : OForm<  0x11, 0x66, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "CMOVGT $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND > zero
 def CMOVGTi  : OFormL< 0x11, 0x66, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "CMOVGT $RCOND,$L,$RDEST">; //CMOVE if RCOND > zero
 def CMOVLBC  : OForm<  0x11, 0x16, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "CMOVLBC $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND low bit clear
 def CMOVLBCi : OFormL< 0x11, 0x16, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "CMOVLBC $RCOND,$L,$RDEST">; //CMOVE if RCOND low bit clear
 def CMOVLBS  : OForm<  0x11, 0x14, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "CMOVLBS $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND low bit set
 def CMOVLBSi : OFormL< 0x11, 0x14, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "CMOVLBS $RCOND,$L,$RDEST">; //CMOVE if RCOND low bit set
 def CMOVLE   : OForm<  0x11, 0x64, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "CMOVLE $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND <= zero
 def CMOVLEi  : OFormL< 0x11, 0x64, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "CMOVLE $RCOND,$L,$RDEST">; //CMOVE if RCOND <= zero
 def CMOVLT   : OForm<  0x11, 0x44, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "CMOVLT $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND < zero
 def CMOVLTi  : OFormL< 0x11, 0x44, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "CMOVLT $RCOND,$L,$RDEST">; //CMOVE if RCOND < zero
 def CMOVNE   : OForm<  0x11, 0x26, (ops GPRC:$RDEST, GPRC:$RSRC2, GPRC:$RSRC, GPRC:$RCOND),
        "cmovne $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND != zero
 def CMOVNEi  : OFormL< 0x11, 0x26, (ops GPRC:$RDEST, GPRC:$RSRC2, u8imm:$L, GPRC:$RCOND),
        "cmovne $RCOND,$L,$RDEST">; //CMOVE if RCOND != zero

//conditional moves, fp
 def FCMOVEQ  : FPForm<0x17, 0x02A, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmoveq $RCOND,$RSRC,$RDEST">; //FCMOVE if = zero
 def FCMOVGE  : FPForm<0x17, 0x02D, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovge $RCOND,$RSRC,$RDEST">; //FCMOVE if >= zero
 def FCMOVGT  : FPForm<0x17, 0x02F, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovgt $RCOND,$RSRC,$RDEST">; //FCMOVE if > zero
 def FCMOVLE  : FPForm<0x17, 0x02E, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovle $RCOND,$RSRC,$RDEST">; //FCMOVE if <= zero
 def FCMOVLT  : FPForm<0x17, 0x02, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovlt $RCOND,$RSRC,$RDEST">; // FCMOVE if < zero
 def FCMOVNE : FPForm<0x17, 0x02B, (ops FPRC:$RDEST, FPRC:$RSRC2, FPRC:$RSRC, FPRC:$RCOND),
        "fcmovne $RCOND,$RSRC,$RDEST">; //FCMOVE if != zero
}

def ADDL     : OForm< 0x10, 0x00, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "addl $RA,$RB,$RC">; //Add longword
def ADDLi    : OFormL<0x10, 0x00, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "addl $RA,$L,$RC">; //Add longword
def ADDQ     : OForm< 0x10, 0x20, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "addq $RA,$RB,$RC">; //Add quadword
def ADDQi    : OFormL<0x10, 0x20, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "addq $RA,$L,$RC">; //Add quadword
def AMASK    : OForm< 0x11, 0x61, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "AMASK $RA,$RB,$RC">; //Architecture mask
def AMASKi   : OFormL<0x11, 0x61, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "AMASK $RA,$L,$RC">; //Architecture mask
def AND      : OForm< 0x11, 0x00, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "AND $RA,$RB,$RC">; //Logical product
def ANDi     : OFormL<0x11, 0x00, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "AND $RA,$L,$RC">; //Logical product
def BIC      : OForm< 0x11, 0x08, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "BIC $RA,$RB,$RC">; //Bit clear
def BICi     : OFormL<0x11, 0x08, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "BIC $RA,$L,$RC">; //Bit clear
def BIS      : OForm< 0x11, 0x20, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "bis $RA,$RB,$RC">; //Logical sum
def BISi     : OFormL<0x11, 0x20, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "bis $RA,$L,$RC">; //Logical sum
def CTLZ     : OForm< 0x1C, 0x32, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CTLZ $RA,$RB,$RC">; //Count leading zero
def CTLZi    : OFormL<0x1C, 0x32, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CTLZ $RA,$L,$RC">; //Count leading zero
def CTPOP    : OForm< 0x1C, 0x30, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CTPOP $RA,$RB,$RC">; //Count population
def CTPOPi   : OFormL<0x1C, 0x30, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CTPOP $RA,$L,$RC">; //Count population
def CTTZ     : OForm< 0x1C, 0x33, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CTTZ $RA,$RB,$RC">; //Count trailing zero
def CTTZi    : OFormL<0x1C, 0x33, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CTTZ $RA,$L,$RC">; //Count trailing zero
def EQV      : OForm< 0x11, 0x48, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EQV $RA,$RB,$RC">; //Logical equivalence
def EQVi     : OFormL<0x11, 0x48, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EQV $RA,$L,$RC">; //Logical equivalence
def EXTBL    : OForm< 0x12, 0x06, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTBL $RA,$RB,$RC">; //Extract byte low
def EXTBLi   : OFormL<0x12, 0x06, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTBL $RA,$L,$RC">; //Extract byte low
def EXTLH    : OForm< 0x12, 0x6A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTLH $RA,$RB,$RC">; //Extract longword high
def EXTLHi   : OFormL<0x12, 0x6A, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTLH $RA,$L,$RC">; //Extract longword high
def EXTLL    : OForm< 0x12, 0x26, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTLL $RA,$RB,$RC">; //Extract longword low
def EXTLLi   : OFormL<0x12, 0x26, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTLL $RA,$L,$RC">; //Extract longword low
def EXTQH    : OForm< 0x12, 0x7A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTQH $RA,$RB,$RC">; //Extract quadword high
def EXTQHi   : OFormL<0x12, 0x7A, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTQH $RA,$L,$RC">; //Extract quadword high
def EXTQ     : OForm< 0x12, 0x36, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTQ $RA,$RB,$RC">; //Extract quadword low
def EXTQi    : OFormL<0x12, 0x36, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTQ $RA,$L,$RC">; //Extract quadword low
def EXTWH    : OForm< 0x12, 0x5A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTWH $RA,$RB,$RC">; //Extract word high
def EXTWHi   : OFormL<0x12, 0x5A, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTWH $RA,$L,$RC">; //Extract word high
def EXTWL    : OForm< 0x12, 0x16, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTWL $RA,$RB,$RC">; //Extract word low
def EXTWLi   : OFormL<0x12, 0x16, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "EXTWL $RA,$L,$RC">; //Extract word low
def IMPLVER  : OForm< 0x11, 0x6C, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "IMPLVER $RA,$RB,$RC">; //Implementation version
def IMPLVERi : OFormL<0x11, 0x6C, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "IMPLVER $RA,$L,$RC">; //Implementation version
def INSBL    : OForm< 0x12, 0x0B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSBL $RA,$RB,$RC">; //Insert byte low
def INSBLi   : OFormL<0x12, 0x0B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSBL $RA,$L,$RC">; //Insert byte low
def INSLH    : OForm< 0x12, 0x67, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSLH $RA,$RB,$RC">; //Insert longword high
def INSLHi   : OFormL<0x12, 0x67, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSLH $RA,$L,$RC">; //Insert longword high
def INSLL    : OForm< 0x12, 0x2B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSLL $RA,$RB,$RC">; //Insert longword low
def INSLLi   : OFormL<0x12, 0x2B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSLL $RA,$L,$RC">; //Insert longword low
def INSQH    : OForm< 0x12, 0x77, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSQH $RA,$RB,$RC">; //Insert quadword high
def INSQHi   : OFormL<0x12, 0x77, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSQH $RA,$L,$RC">; //Insert quadword high
def INSQL    : OForm< 0x12, 0x3B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSQL $RA,$RB,$RC">; //Insert quadword low
def INSQLi   : OFormL<0x12, 0x3B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSQL $RA,$L,$RC">; //Insert quadword low
def INSWH    : OForm< 0x12, 0x57, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSWH $RA,$RB,$RC">; //Insert word high
def INSWHi   : OFormL<0x12, 0x57, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSWH $RA,$L,$RC">; //Insert word high
def INSWL    : OForm< 0x12, 0x1B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSWL $RA,$RB,$RC">; //Insert word low
def INSWLi   : OFormL<0x12, 0x1B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "INSWL $RA,$L,$RC">; //Insert word low
def MSKBL    : OForm< 0x12, 0x02, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKBL $RA,$RB,$RC">; //Mask byte low
def MSKBLi   : OFormL<0x12, 0x02, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKBL $RA,$L,$RC">; //Mask byte low
def MSKLH    : OForm< 0x12, 0x62, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKLH $RA,$RB,$RC">; //Mask longword high
def MSKLHi   : OFormL<0x12, 0x62, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKLH $RA,$L,$RC">; //Mask longword high
def MSKLL    : OForm< 0x12, 0x22, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKLL $RA,$RB,$RC">; //Mask longword low
def MSKLLi   : OFormL<0x12, 0x22, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKLL $RA,$L,$RC">; //Mask longword low
def MSKQH    : OForm< 0x12, 0x72, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKQH $RA,$RB,$RC">; //Mask quadword high
def MSKQHi   : OFormL<0x12, 0x72, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKQH $RA,$L,$RC">; //Mask quadword high
def MSKQL    : OForm< 0x12, 0x32, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKQL $RA,$RB,$RC">; //Mask quadword low
def MSKQLi   : OFormL<0x12, 0x32, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKQL $RA,$L,$RC">; //Mask quadword low
def MSKWH    : OForm< 0x12, 0x52, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKWH $RA,$RB,$RC">; //Mask word high
def MSKWHi   : OFormL<0x12, 0x52, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKWH $RA,$L,$RC">; //Mask word high
def MSKWL    : OForm< 0x12, 0x12, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKWL $RA,$RB,$RC">; //Mask word low
def MSKWLi   : OFormL<0x12, 0x12, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MSKWL $RA,$L,$RC">; //Mask word low
def MULL     : OForm< 0x13, 0x00, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MULL $RA,$RB,$RC">; //Multiply longword
def MULLi    : OFormL<0x13, 0x00, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MULL $RA,$L,$RC">; //Multiply longword
def MULQ     : OForm< 0x13, 0x20, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MULQ $RA,$RB,$RC">; //Multiply quadword
def MULQi    : OFormL<0x13, 0x20, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "MULQ $RA,$L,$RC">; //Multiply quadword
def ORNOT    : OForm< 0x11, 0x28, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ORNOT $RA,$RB,$RC">; //Logical sum with complement
def ORNOTi   : OFormL<0x11, 0x28, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "ORNOT $RA,$L,$RC">; //Logical sum with complement
def S4ADDL   : OForm< 0x10, 0x02, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4ADDL $RA,$RB,$RC">; //Scaled add longword by 4
def S4ADDLi  : OFormL<0x10, 0x02, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S4ADDL $RA,$L,$RC">; //Scaled add longword by 4
def S4ADDQ   : OForm< 0x10, 0x22, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4ADDQ $RA,$RB,$RC">; //Scaled add quadword by 4
def S4ADDQi  : OFormL<0x10, 0x22, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S4ADDQ $RA,$L,$RC">; //Scaled add quadword by 4
def S4SUBL   : OForm< 0x10, 0x0B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4SUBL $RA,$RB,$RC">; //Scaled subtract longword by 4
def S4SUBLi  : OFormL<0x10, 0x0B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S4SUBL $RA,$L,$RC">; //Scaled subtract longword by 4
def S4SUBQ   : OForm< 0x10, 0x2B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4SUBQ $RA,$RB,$RC">; //Scaled subtract quadword by 4
def S4SUBQi  : OFormL<0x10, 0x2B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S4SUBQ $RA,$L,$RC">; //Scaled subtract quadword by 4
def S8ADDL   : OForm< 0x10, 0x12, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8ADDL $RA,$RB,$RC">; //Scaled add longword by 8
def S8ADDLi  : OFormL<0x10, 0x12, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S8ADDL $RA,$L,$RC">; //Scaled add longword by 8
def S8ADDQ   : OForm< 0x10, 0x32, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8ADDQ $RA,$RB,$RC">; //Scaled add quadword by 8
def S8ADDQi  : OFormL<0x10, 0x32, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S8ADDQ $RA,$L,$RC">; //Scaled add quadword by 8
def S8SUBL   : OForm< 0x10, 0x1B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8SUBL $RA,$RB,$RC">; //Scaled subtract longword by 8
def S8SUBLi  : OFormL<0x10, 0x1B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S8SUBL $RA,$L,$RC">; //Scaled subtract longword by 8
def S8SUBQ   : OForm< 0x10, 0x3B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8SUBQ $RA,$RB,$RC">; //Scaled subtract quadword by 8
def S8SUBQi  : OFormL<0x10, 0x3B, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "S8SUBQ $RA,$L,$RC">; //Scaled subtract quadword by 8
def SEXTB    : OForm< 0x1C, 0x00, (ops GPRC:$RC, GPRC:$RB), "sextb $RB,$RC">; //Sign extend byte
def SEXTBi   : OFormL<0x1C, 0x00, (ops GPRC:$RC, u8imm:$L), "sextb $L,$RC">; //Sign extend byte
def SEXTW    : OForm< 0x1C, 0x01, (ops GPRC:$RC, GPRC:$RB), "sextw $RB,$RC">; //Sign extend word
def SEXTWi   : OFormL<0x1C, 0x01, (ops GPRC:$RC, u8imm:$L), "sextw $L,$RC">; //Sign extend word
def SL       : OForm< 0x12, 0x39, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SLL $RA,$RB,$RC">; //Shift left logical
def SLi      : OFormL<0x12, 0x39, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "SLL $RA,$L,$RC">; //Shift left logical
def SRA      : OForm< 0x12, 0x3C, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SRA $RA,$RB,$RC">; //Shift right arithmetic
def SRAi     : OFormL<0x12, 0x3C, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "SRA $RA,$L,$RC">; //Shift right arithmetic
def SRL      : OForm< 0x12, 0x34, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SRL $RA,$RB,$RC">; //Shift right logical

def SRLi     : OFormL<0x12, 0x34, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "SRL $RA,$L,$RC">; //Shift right logical
def SUBL     : OForm< 0x10, 0x09, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SUBL $RA,$RB,$RC">; //Subtract longword
def SUBLi    : OFormL<0x10, 0x09, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "SUBL $RA,$L,$RC">; //Subtract longword
def SUBQ     : OForm< 0x10, 0x29, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SUBQ $RA,$RB,$RC">; //Subtract quadword
def SUBQi    : OFormL<0x10, 0x29, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "SUBQ $RA,$L,$RC">; //Subtract quadword
def UMULH    : OForm< 0x13, 0x30, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "UMULH $RA,$RB,$RC">; //Unsigned multiply quadword high
def UMULHi   : OFormL<0x13, 0x30, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "UMULH $RA,$L,$RC">; //Unsigned multiply quadword high
def XOR      : OForm< 0x11, 0x40, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "XOR $RA,$RB,$RC">; //Logical difference
def XORi     : OFormL<0x11, 0x40, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "XOR $RA,$L,$RC">; //Logical difference
def ZAP      : OForm< 0x12, 0x30, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ZAP $RA,$RB,$RC">; //Zero bytes
def ZAPi     : OFormL<0x12, 0x30, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "ZAP $RA,$L,$RC">; //Zero bytes
def ZAPNOT   : OForm< 0x12, 0x31, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ZAPNOT $RA,$RB,$RC">; //Zero bytes not
def ZAPNOTi  : OFormL<0x12, 0x31, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "ZAPNOT $RA,$L,$RC">; //Zero bytes not

//Comparison, int
def CMPBGE   : OForm< 0x10, 0x0F, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPBGE $RA,$RB,$RC">; //Compare byte
def CMPBGEi  : OFormL<0x10, 0x0F, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CMPBGE $RA,$L,$RC">; //Compare byte
def CMPEQ    : OForm< 0x10, 0x2D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPEQ $RA,$RB,$RC">; //Compare signed quadword equal
def CMPEQi   : OFormL<0x10, 0x2D, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CMPEQ $RA,$L,$RC">; //Compare signed quadword equal
def CMPLE    : OForm< 0x10, 0x6D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPLE $RA,$RB,$RC">; //Compare signed quadword less than or equal
def CMPLEi   : OFormL<0x10, 0x6D, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CMPLE $RA,$L,$RC">; //Compare signed quadword less than or equal
def CMPLT    : OForm< 0x10, 0x4D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPLT $RA,$RB,$RC">; //Compare signed quadword less than
def CMPLTi   : OFormL<0x10, 0x4D, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CMPLT $RA,$L,$RC">; //Compare signed quadword less than
def CMPULE   : OForm< 0x10, 0x3D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPULE $RA,$RB,$RC">; //Compare unsigned quadword less than or equal
def CMPULEi  : OFormL<0x10, 0x3D, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CMPULE $RA,$L,$RC">; //Compare unsigned quadword less than or equal
def CMPULT   : OForm< 0x10, 0x1D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPULT $RA,$RB,$RC">; //Compare unsigned quadword less than
def CMPULTi  : OFormL<0x10, 0x1D, (ops GPRC:$RC, GPRC:$RA, u8imm:$L), "CMPULT $RA,$L,$RC">; //Compare unsigned quadword less than

//Comparison, FP
def CMPTEQ : FPForm<0x16, 0x0A5, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cmpteq/su $RA,$RB,$RC">;  //Compare T_floating equal
def CMPTLE : FPForm<0x16, 0x0A7, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cmptle/su $RA,$RB,$RC">;  //Compare T_floating less than or equal
def CMPTLT : FPForm<0x16, 0x0A6, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cmptlt/su $RA,$RB,$RC">;  //Compare T_floating less than
def CMPTUN : FPForm<0x16, 0x0A4, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cmptun/su $RA,$RB,$RC">;  //Compare T_floating unordered

//There are in the Multimedia extentions, so let's not use them yet
def MAXSB8  : OForm<0x1C, 0x3E, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXSB8 $RA,$RB,$RC">; //Vector signed byte maximum
def MAXSW4 : OForm< 0x1C, 0x3F, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXSW4 $RA,$RB,$RC">; //Vector signed word maximum
def MAXUB8  : OForm<0x1C, 0x3C, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXUB8 $RA,$RB,$RC">; //Vector unsigned byte maximum
def MAXUW4 : OForm< 0x1C, 0x3D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXUW4 $RA,$RB,$RC">; //Vector unsigned word maximum
def MINSB8 : OForm< 0x1C, 0x38, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINSB8 $RA,$RB,$RC">; //Vector signed byte minimum
def MINSW4 : OForm< 0x1C, 0x39, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINSW4 $RA,$RB,$RC">; //Vector signed word minimum
def MINUB8 : OForm< 0x1C, 0x3A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINUB8 $RA,$RB,$RC">; //Vector unsigned byte minimum
def MINUW4 : OForm< 0x1C, 0x3B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINUW4 $RA,$RB,$RC">; //Vector unsigned word minimum
def PERR : OForm< 0x1C, 0x31, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "PERR $RA,$RB,$RC">; //Pixel error
def PKLB : OForm< 0x1C, 0x37, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "PKLB $RA,$RB,$RC">; //Pack longwords to bytes
def PKWB  : OForm<0x1C, 0x36, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "PKWB $RA,$RB,$RC">; //Pack words to bytes
def UNPKBL : OForm< 0x1C, 0x35, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "UNPKBL $RA,$RB,$RC">; //Unpack bytes to longwords
def UNPKBW : OForm< 0x1C, 0x34, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "UNPKBW $RA,$RB,$RC">; //Unpack bytes to words

//End operate

let isReturn = 1, isTerminator = 1 in
  def RET : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS), "ret $RD,($RS),1">; //Return from subroutine

def JMP : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS), "jmp $RD,($RS),0">; //Jump
let isCall = 1,
    Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19,
            R20, R21, R22, R23, R24, R25, R27, R28, R29,
            F0, F1,
            F10, F11, F12, F13, F14, F15, F16, F17, F18, F19,
            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30] in {
    def JSR : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS, s14imm:$DISP), "jsr $RD,($RS),$DISP">; //Jump to subroutine
    def BSR : BForm<0x34, (ops GPRC:$RD, s21imm:$DISP), "bsr $RD,$DISP">; //Branch to subroutine
}

def JSR_COROUTINE : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS), "jsr_coroutine $RD,($RS),1">; //Jump to subroutine return
def BR : BForm<0x30, (ops GPRC:$RD, s21imm:$DISP), "br $RD,$DISP">; //Branch

let Uses = [R28] in {
//Stores, int
def STB : MForm<0x0E, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stb $RA,$DISP($RB)">; // Store byte
def STW : MForm<0x0D, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stw $RA,$DISP($RB)">; // Store word
def STL : MForm<0x2C, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stl $RA,$DISP($RB)">; // Store longword
def STQ : MForm<0x2D, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stq $RA,$DISP($RB)">; //Store quadword

//Loads, int
def LDL : MForm<0x28, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldl $RA,$DISP($RB)">; // Load sign-extended longword
def LDQ : MForm<0x29, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldq $RA,$DISP($RB)">; //Load quadword
def LDBU : MForm<0x0A, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldbu $RA,$DISP($RB)">; //Load zero-extended byte
def LDWU : MForm<0x0C, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldwu $RA,$DISP($RB)">; //Load zero-extended word

//Stores, float
def STS : MForm<0x26, (ops FPRC:$RA, s16imm:$DISP, GPRC:$RB), "sts $RA,$DISP($RB)">; //Store S_floating
def STT : MForm<0x27, (ops FPRC:$RA, s16imm:$DISP, GPRC:$RB), "stt $RA,$DISP($RB)">; //Store T_floating

//Loads, float
def LDS : MForm<0x22, (ops FPRC:$RA, s16imm:$DISP, GPRC:$RB), "lds $RA,$DISP($RB)">; //Load S_floating
def LDT : MForm<0x23, (ops FPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldt $RA,$DISP($RB)">; //Load T_floating
}

//Load address
def LDA : MForm<0x08, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "lda $RA,$DISP($RB)">;  //Load address
def LDAH : MForm<0x08, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldah $RA,$DISP($RB)">;  //Load address high


//Branches, int
def BEQ : BForm<0x39, (ops GPRC:$RA, s21imm:$DISP), "beq $RA,$DISP">; //Branch if = zero
def BGE : BForm<0x3E, (ops GPRC:$RA, s21imm:$DISP), "bge $RA,$DISP">; //Branch if >= zero
def BGT : BForm<0x3F, (ops GPRC:$RA, s21imm:$DISP), "bgt $RA,$DISP">; //Branch if > zero
def BLBC : BForm<0x38, (ops GPRC:$RA, s21imm:$DISP), "blbc $RA,$DISP">; //Branch if low bit clear
def BLBS : BForm<0x3C, (ops GPRC:$RA, s21imm:$DISP), "blbs $RA,$DISP">; //Branch if low bit set
def BLE : BForm<0x3B, (ops GPRC:$RA, s21imm:$DISP), "ble $RA,$DISP">; //Branch if <= zero
def BLT : BForm<0x3A, (ops GPRC:$RA, s21imm:$DISP), "blt $RA,$DISP">; //Branch if < zero
def BNE : BForm<0x3D, (ops GPRC:$RA, s21imm:$DISP), "bne $RA,$DISP">; //Branch if != zero

//Branches, float
def FBEQ : BForm<0x31, (ops FPRC:$RA, s21imm:$DISP), "fbeq $RA,$DISP">; //Floating branch if =  zero
def FBGE : BForm<0x36, (ops FPRC:$RA, s21imm:$DISP), "fbge $RA,$DISP">; //Floating branch if >= zero
def FBGT : BForm<0x37, (ops FPRC:$RA, s21imm:$DISP), "fbgt $RA,$DISP">; //Floating branch if > zero
def FBLE : BForm<0x33, (ops FPRC:$RA, s21imm:$DISP), "fble $RA,$DISP">; //Floating branch if <= zero
def FBLT : BForm<0x32, (ops FPRC:$RA, s21imm:$DISP), "fblt $RA,$DISP">; //Floating branch if < zero
def FBNE : BForm<0x35, (ops FPRC:$RA, s21imm:$DISP), "fbne $RA,$DISP">; //Floating branch if != zero

//Funky Floating point ops
def CPYS  : FPForm<0x17, 0x020, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cpys $RA,$RB,$RC">;  //Copy sign
def CPYSE : FPForm<0x17, 0x022, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cpyse $RA,$RB,$RC">; //Copy sign and exponent
def CPYSN : FPForm<0x17, 0x021, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "cpysn $RA,$RB,$RC">; //Copy sign negate

//Basic Floating point ops
def ADDS  : FPForm<0x16, 0x080, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "adds/sui $RA,$RB,$RC">;  //Add S_floating
def ADDT  : FPForm<0x16, 0x0A0, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "addt/sui $RA,$RB,$RC">;  //Add T_floating
def SUBS  : FPForm<0x16, 0x081, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "subs/sui $RA,$RB,$RC">;  //Subtract S_floating
def SUBT  : FPForm<0x16, 0x0A1, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "subt/sui $RA,$RB,$RC">;  //Subtract T_floating
def DIVS  : FPForm<0x16, 0x083, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "divs/sui $RA,$RB,$RC">;  //Divide S_floating
def DIVT  : FPForm<0x16, 0x0A3, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "divt/sui $RA,$RB,$RC">;  //Divide T_floating
def MULS  : FPForm<0x16, 0x082, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "muls/sui $RA,$RB,$RC">;  //Multiply S_floating
def MULT  : FPForm<0x16, 0x0A2, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "mult/sui $RA,$RB,$RC">;  //Multiply T_floating
def SQRTS : FPForm<0x14, 0x08B, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "sqrts $RA,$RB,$RC">;  //Square root S_floating
def SQRTT : FPForm<0x14, 0x0AB, (ops FPRC:$RC, FPRC:$RA, FPRC:$RB), "sqrtt $RA,$RB,$RC">;  //Square root T_floating

//INT reg to FP reg and back again
//not supported on 21164
def FTOIS : FPForm<0x1C, 0x078, (ops FPRC:$RC, GPRC:$RA), "ftois $RA,$RC">; //Floating to integer move, S_floating
def FTOIT : FPForm<0x1C, 0x070, (ops FPRC:$RC, GPRC:$RA), "ftoit $RA,$RC">; //Floating to integer move, T_floating
def ITOFS : FPForm<0x14, 0x004, (ops FPRC:$RC, GPRC:$RA), "itofs $RA,$RC">; //Integer to floating move, S_floating
def ITOFT : FPForm<0x14, 0x024, (ops FPRC:$RC, GPRC:$RA), "itoft $RA,$RC">; //Integer to floating move, T_floating

//CVTLQ F-P 17.010 Convert longword to quadword
//CVTQL F-P 17.030 Convert quadword to longword
//These use SW completion, may not have function code for that set right (matters for JIT)
def CVTQS : FPForm<0x16, 0x0BC,  (ops FPRC:$RC, FPRC:$RA), "cvtqs $RA,$RC">; //Convert quadword to S_floating
def CVTQT : FPForm<0x16, 0x0BE,  (ops FPRC:$RC, FPRC:$RA), "cvtqt $RA,$RC">; //Convert quadword to T_floating
def CVTST : FPForm<0x16, 0x2AC, (ops FPRC:$RC, FPRC:$RA), "cvtsts $RA,$RC">; //Convert S_floating to T_floating
def CVTTQ : FPForm<0x16, 0x0AF,  (ops FPRC:$RC, FPRC:$RA), "cvttq/svc $RA,$RC">; //Convert T_floating to quadword
def CVTTS : FPForm<0x16, 0x2AC, (ops FPRC:$RC, FPRC:$RA), "cvtts/su $RA,$RC">; //Convert T_floating to S_floating

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
//RC Mfc 18.E000 Read and clear
//RPCC Mfc 18.C000 Read process cycle counter
//RS Mfc 18.F000 Read and set

//STL_C Mem 2E Store longword conditional
//STQ_C Mem 2F Store quadword conditional
//STQ_U Mem 0F Store unaligned quadword

//TRAPB Mfc 18.0000 Trap barrier
//WH64 Mfc 18.F800 Write hint  64 bytes
//WMB Mfc 18.4400 Write memory barrier


//MF_FPCR F-P 17.025 Move from FPCR
//MT_FPCR F-P 17.024 Move to FPCR
