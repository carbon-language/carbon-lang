//===- AlphaInstrInfo.td - The Alpha Instruction Set -----*- tablegen -*-=//
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

def s14imm  : Operand<i16>;
def s16imm  : Operand<i16>;
def s21imm  : Operand<i32>;
def s64imm  : Operand<i64>;

def PHI : PseudoInstAlpha<(ops ), "#phi">;
def IDEF : PseudoInstAlpha<(ops ), "#idef">;
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
    Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R29],
    Uses = [R27, R29] in
      def CALL : PseudoInstAlpha< (ops s64imm:$TARGET), "jsr $TARGET">; //Jump to subroutine

let isReturn = 1, isTerminator = 1 in
  def RETURN : PseudoInstAlpha<(ops ), "ret $$31,($$26),1">; //Return from subroutine

def LOAD_IMM : PseudoInstAlpha<(ops GPRC:$RC, s64imm:$IMM), "ldiq $RC,$IMM">; //Load Immediate Quadword

let Uses = [R29] in
  def STORE : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "stq $RA,$DISP">; //Store quadword

let Uses = [R29] in
  def LOAD_ADDR : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "lda $RA,$DISP">;  //Load address

let Uses = [R29] in
  def LOAD : PseudoInstAlpha<(ops GPRC:$RA, s64imm:$DISP), "ldq $RA,$DISP">; //Load quadword

def LDW : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldw $RA,$DISP($RB)">; // Load sign-extended word
def LDB : PseudoInstAlpha<(ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldb $RA,$DISP($RB)">; //Load byte

let Uses = [R28, R23, R24, R25, R26] in
  def REMQU : PseudoInstAlpha<(ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "remqu $RA,$RB,$RC">; //unsigned remander

//***********************
//Real instructions
//***********************

//Operation Form:
def ADDL : OForm<0x10, 0x00, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "addl $RA,$RB,$RC">; //Add longword
def ADDL_V : OForm< 0x10, 0x40, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ADDL/V $RA,$RB,$RC">;
def ADDQ : OForm< 0x10, 0x20, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "addq $RA,$RB,$RC">; //Add quadword
def ADDQ_V : OForm< 0x10, 0x60, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ADDQ/V $RA,$RB,$RC">; 
def AMASK : OForm< 0x11, 0x61, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "AMASK $RA,$RB,$RC">; //Architecture mask
def AND : OForm< 0x11, 0x00, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "AND $RA,$RB,$RC">; //Logical product
def BIC : OForm< 0x11, 0x08, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "BIC $RA,$RB,$RC">; //Bit clear
def BIS  : OForm<0x11, 0x20, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "bis $RA,$RB,$RC">; //Logical sum

//let isTwoAddress = 1 in {
 def CMOVEQ : OForm< 0x11, 0x24, 
	(ops GPRC:$RDEST, GPRC:$RSRC, GPRC:$RCOND), 
	"cmoveq $RCOND,$RSRC,$RDEST">; //CMOVE if RCOND =  zero
 def CMOVGE : OForm< 0x11, 0x46, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMOVGE $RA,$RB,$RC">; //CMOVE if ³ zero
 def CMOVGT  : OForm<0x11, 0x66, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMOVGT $RA,$RB,$RC">; //CMOVE if > zero
 def CMOVLBC : OForm< 0x11, 0x16, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMOVLBC $RA,$RB,$RC">; //CMOVE if low bit clear
 def CMOVLBS : OForm< 0x11, 0x14, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMOVLBS $RA,$RB,$RC">; //CMOVE if low bit set
 def CMOVLE  : OForm<0x11, 0x64, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMOVLE $RA,$RB,$RC">; //CMOVE if £ zero
 def CMOVLT : OForm< 0x11, 0x44, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMOVLT $RA,$RB,$RC">; //CMOVE if < zero
 def CMOVNE : OForm< 0x11, 0x26, 
	(ops GPRC:$RC, GPRC:$DUMMY, GPRC:$RA, GPRC:$RB), 
	"cmovne $RA,$RB,$RC">; //CMOVE if ¹ zero
//}

def CMPBGE : OForm< 0x10, 0x0F, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPBGE $RA,$RB,$RC">; //Compare byte
def CMPEQ : OForm< 0x10, 0x2D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPEQ $RA,$RB,$RC">; //Compare signed quadword equal
def CMPLE : OForm< 0x10, 0x6D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPLE $RA,$RB,$RC">; //Compare signed quadword less than or equal
def CMPLT : OForm< 0x10, 0x4D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPLT $RA,$RB,$RC">; //Compare signed quadword less than
def CMPULE : OForm< 0x10, 0x3D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPULE $RA,$RB,$RC">; //Compare unsigned quadword less than or equal
def CMPULT : OForm< 0x10, 0x1D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CMPULT $RA,$RB,$RC">; //Compare unsigned quadword less than
def CTLZ : OForm< 0x1C, 0x32, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CTLZ $RA,$RB,$RC">; //Count leading zero
def CTPOP : OForm< 0x1C, 0x30, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CTPOP $RA,$RB,$RC">; //Count population
def CTTZ  : OForm<0x1C, 0x33, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "CTTZ $RA,$RB,$RC">; //Count trailing zero
def EQV : OForm< 0x11, 0x48, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EQV $RA,$RB,$RC">; //Logical equivalence
def EXTBL : OForm< 0x12, 0x06, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTBL $RA,$RB,$RC">; //Extract byte low
def EXTLH : OForm< 0x12, 0x6A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTLH $RA,$RB,$RC">; //Extract longword high
def EXTLL : OForm< 0x12, 0x26, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTLL $RA,$RB,$RC">; //Extract longword low
def EXTQH : OForm< 0x12, 0x7A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTQH $RA,$RB,$RC">; //Extract quadword high
def EXTQ : OForm< 0x12, 0x36, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTQ $RA,$RB,$RC">; //Extract quadword low
def EXTWH : OForm< 0x12, 0x5A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTWH $RA,$RB,$RC">; //Extract word high
def EXTWL : OForm< 0x12, 0x16, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "EXTWL $RA,$RB,$RC">; //Extract word low
def IMPLVER : OForm< 0x11, 0x6C, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "IMPLVER $RA,$RB,$RC">; //Implementation version
def INSBL : OForm< 0x12, 0x0B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSBL $RA,$RB,$RC">; //Insert byte low
def INSLH : OForm< 0x12, 0x67, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSLH $RA,$RB,$RC">; //Insert longword high
def INSLL : OForm< 0x12, 0x2B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSLL $RA,$RB,$RC">; //Insert longword low
def INSQH : OForm< 0x12, 0x77, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSQH $RA,$RB,$RC">; //Insert quadword high
def INSQL : OForm< 0x12, 0x3B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSQL $RA,$RB,$RC">; //Insert quadword low
def INSWH : OForm< 0x12, 0x57, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSWH $RA,$RB,$RC">; //Insert word high
def INSWL : OForm< 0x12, 0x1B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "INSWL $RA,$RB,$RC">; //Insert word low
def MAXSB8  : OForm<0x1C, 0x3E, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXSB8 $RA,$RB,$RC">; //Vector signed byte maximum
def MAXSW4 : OForm< 0x1C, 0x3F, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXSW4 $RA,$RB,$RC">; //Vector signed word maximum
def MAXUB8  : OForm<0x1C, 0x3C, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXUB8 $RA,$RB,$RC">; //Vector unsigned byte maximum
def MAXUW4 : OForm< 0x1C, 0x3D, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MAXUW4 $RA,$RB,$RC">; //Vector unsigned word maximum
def MINSB8 : OForm< 0x1C, 0x38, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINSB8 $RA,$RB,$RC">; //Vector signed byte minimum
def MINSW4 : OForm< 0x1C, 0x39, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINSW4 $RA,$RB,$RC">; //Vector signed word minimum
def MINUB8 : OForm< 0x1C, 0x3A, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINUB8 $RA,$RB,$RC">; //Vector unsigned byte minimum
def MINUW4 : OForm< 0x1C, 0x3B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MINUW4 $RA,$RB,$RC">; //Vector unsigned word minimum
def MSKBL : OForm< 0x12, 0x02, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKBL $RA,$RB,$RC">; //Mask byte low
def MSKLH : OForm< 0x12, 0x62, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKLH $RA,$RB,$RC">; //Mask longword high
def MSKLL : OForm< 0x12, 0x22, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKLL $RA,$RB,$RC">; //Mask longword low
def MSKQH : OForm< 0x12, 0x72, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKQH $RA,$RB,$RC">; //Mask quadword high
def MSKQL : OForm< 0x12, 0x32, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKQL $RA,$RB,$RC">; //Mask quadword low
def MSKWH : OForm< 0x12, 0x52, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKWH $RA,$RB,$RC">; //Mask word high
def MSKWL : OForm< 0x12, 0x12, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MSKWL $RA,$RB,$RC">; //Mask word low
def MULL : OForm< 0x13, 0x00, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MULL $RA,$RB,$RC">; //Multiply longword
def MULL_V : OForm< 0x13, 0x40, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MULL/V $RA,$RB,$RC">; 
def MULQ : OForm< 0x13, 0x20, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MULQ $RA,$RB,$RC">; //Multiply quadword
def MULQ_V : OForm< 0x13, 0x60, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "MULQ/V $RA,$RB,$RC">; 
def ORNOT : OForm< 0x11, 0x28, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ORNOT $RA,$RB,$RC">; //Logical sum with complement
def PERR : OForm< 0x1C, 0x31, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "PERR $RA,$RB,$RC">; //Pixel error
def PKLB : OForm< 0x1C, 0x37, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "PKLB $RA,$RB,$RC">; //Pack longwords to bytes
def PKWB  : OForm<0x1C, 0x36, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "PKWB $RA,$RB,$RC">; //Pack words to bytes
def S4ADDL : OForm< 0x10, 0x02, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4ADDL $RA,$RB,$RC">; //Scaled add longword by 4
def S4ADDQ : OForm< 0x10, 0x22, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4ADDQ $RA,$RB,$RC">; //Scaled add quadword by 4
def S4SUBL : OForm< 0x10, 0x0B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4SUBL $RA,$RB,$RC">; //Scaled subtract longword by 4
def S4SUBQ : OForm< 0x10, 0x2B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S4SUBQ $RA,$RB,$RC">; //Scaled subtract quadword by 4
def S8ADDL : OForm< 0x10, 0x12, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8ADDL $RA,$RB,$RC">; //Scaled add longword by 8
def S8ADDQ : OForm< 0x10, 0x32, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8ADDQ $RA,$RB,$RC">; //Scaled add quadword by 8
def S8SUBL : OForm< 0x10, 0x1B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8SUBL $RA,$RB,$RC">; //Scaled subtract longword by 8
def S8SUBQ : OForm< 0x10, 0x3B, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "S8SUBQ $RA,$RB,$RC">; //Scaled subtract quadword by 8
def SEXTB : OForm< 0x1C, 0x00, (ops GPRC:$RC, GPRC:$RB), "sextb $RB,$RC">; //Sign extend byte
def SEXTW : OForm< 0x1C, 0x01, (ops GPRC:$RC, GPRC:$RB), "sextw $RB,$RC">; //Sign extend word
def SL : OForm< 0x12, 0x39, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SLL $RA,$RB,$RC">; //Shift left logical
def SRA : OForm< 0x12, 0x3C, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SRA $RA,$RB,$RC">; //Shift right arithmetic
def SRL : OForm< 0x12, 0x34, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SRL $RA,$RB,$RC">; //Shift right logical
def SUBL : OForm< 0x10, 0x09, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SUBL $RA,$RB,$RC">; //Subtract longword
def SUBL_V : OForm< 0x10, 0x49, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SUBL/V $RA,$RB,$RC">; 
def SUBQ : OForm< 0x10, 0x29, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SUBQ $RA,$RB,$RC">; //Subtract quadword
def SUBQ_V : OForm< 0x10, 0x69, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "SUBQ/V $RA,$RB,$RC">; 
def UMULH : OForm< 0x13, 0x30, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "UMULH $RA,$RB,$RC">; //Unsigned multiply quadword high
def UNPKBL : OForm< 0x1C, 0x35, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "UNPKBL $RA,$RB,$RC">; //Unpack bytes to longwords
def UNPKBW : OForm< 0x1C, 0x34, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "UNPKBW $RA,$RB,$RC">; //Unpack bytes to words
def XOR : OForm< 0x11, 0x40, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "XOR $RA,$RB,$RC">; //Logical difference
def ZAP : OForm< 0x12, 0x30, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ZAP $RA,$RB,$RC">; //Zero bytes
def ZAPNOT : OForm< 0x12, 0x31, (ops GPRC:$RC, GPRC:$RA, GPRC:$RB), "ZAPNOT $RA,$RB,$RC">; //Zero bytes not

let isReturn = 1, isTerminator = 1 in
  def RET : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS), "ret $RD,($RS),1">; //Return from subroutine

def JMP : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS), "jmp $RD,($RS),0">; //Jump
let isCall = 1 in
  let Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R27, R29] in 
    def JSR : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS, s14imm:$DISP), "jsr $RD,($RS),$DISP">; //Jump to subroutine
def JSR_COROUTINE : MForm< 0x1A, (ops GPRC:$RD, GPRC:$RS), "jsr_coroutine $RD,($RS),1">; //Jump to subroutine return

def BR : BForm<0x30, (ops GPRC:$RD, s21imm:$DISP), "br $RD,$DISP">; //Branch
let isCall = 1 in
  let Defs = [R0, R1, R2, R3, R4, R5, R6, R7, R8, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R27, R29] in 
    def BSR : BForm<0x34, (ops GPRC:$RD, s21imm:$DISP), "bsr $RD,$DISP">; //Branch to subroutine

def STB : MForm<0x0E, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stb $RA,$DISP($RB)">; // Store byte
def STW : MForm<0x0D, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stw $RA,$DISP($RB)">; // Store word
def STL : MForm<0x2C, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stl $RA,$DISP($RB)">; // Store longword
def STQ : MForm<0x2D, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "stq $RA,$DISP($RB)">; //Store quadword

def LDA : MForm<0x08, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "lda $RA,$DISP($RB)">;  //Load address

def LDL : MForm<0x28, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldq $RA,$DISP($RB)">; // Load sign-extended longword
def LDQ : MForm<0x29, (ops GPRC:$RA, s16imm:$DISP, GPRC:$RB), "ldq $RA,$DISP($RB)">; //Load quadword

def BEQ : BForm<0x39, (ops GPRC:$RA, s21imm:$DISP), "beq $RA,$DISP">; //Branch if = zero
def BGE : BForm<0x3E, (ops GPRC:$RA, s21imm:$DISP), "bge $RA,$DISP">; //Branch if >= zero
def BGT : BForm<0x3F, (ops GPRC:$RA, s21imm:$DISP), "bgt $RA,$DISP">; //Branch if > zero
def BLBC : BForm<0x38, (ops GPRC:$RA, s21imm:$DISP), "blbc $RA,$DISP">; //Branch if low bit clear
def BLBS : BForm<0x3C, (ops GPRC:$RA, s21imm:$DISP), "blbs $RA,$DISP">; //Branch if low bit set
def BLE : BForm<0x3B, (ops GPRC:$RA, s21imm:$DISP), "ble $RA,$DISP">; //Branch if <= zero
def BLT : BForm<0x3A, (ops GPRC:$RA, s21imm:$DISP), "blt $RA,$DISP">; //Branch if < zero
def BNE : BForm<0x3D, (ops GPRC:$RA, s21imm:$DISP), "bne $RA,$DISP">; //Branch if != zero

//Mnemonic Format Opcode Description
//ADDF F-P 15.080 Add F_floating
//ADDG F-P 15.0A0 Add G_floating
//ADDS F-P 16.080 Add S_floating
//ADDT F-P 16.0A0 Add T_floating
//CALL_PAL Pcd 00 Trap to PALcode
//ECB Mfc 18.E800 Evict cache block
//EXCB Mfc 18.0400 Exception barrier
//FETCH Mfc 18.8000 Prefetch data
//FETCH_M Mfc 18.A000 Prefetch data, modify intent
//LDAH Mem 09 Load address high
//LDBU Mem 0A Load zero-extended byte
//LDWU Mem 0C Load zero-extended word
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


//CMPGEQ F-P 15.0A5 Compare G_floating equal
//CMPGLE F-P 15.0A7 Compare G_floating less than or equal
//CMPGLT F-P 15.0A6 Compare G_floating less than
//CMPTEQ F-P 16.0A5 Compare T_floating equal
//CMPTLE F-P 16.0A7 Compare T_floating less than or equal
//CMPTLT F-P 16.0A6 Compare T_floating less than
//CMPTUN F-P 16.0A4 Compare T_floating unordered
//CPYS F-P 17.020 Copy sign
//CPYSE F-P 17.022 Copy sign and exponent
//CPYSN F-P 17.021 Copy sign negate
//CVTDG F-P 15.09E Convert D_floating to G_floating
//CVTGD F-P 15.0AD Convert G_floating to D_floating
//CVTGF F-P 15.0AC Convert G_floating to F_floating
//CVTGQ F-P 15.0AF Convert G_floating to quadword
//CVTLQ F-P 17.010 Convert longword to quadword
//CVTQF F-P 15.0BC Convert quadword to F_floating
//CVTQG F-P 15.0BE Convert quadword to G_floating
//CVTQL F-P 17.030 Convert quadword to longword
//CVTQS F-P 16.0BC Convert quadword to S_floating
//CVTQT F-P 16.0BE Convert quadword to T_floating
//CVTST F-P 16.2AC Convert S_floating to T_floating
//CVTTQ F-P 16.0AF Convert T_floating to quadword
//CVTTS F-P 16.0AC Convert T_floating to S_floating
//DIVF F-P 15.083 Divide F_floating
//DIVG F-P 15.0A3 Divide G_floating
//DIVS F-P 16.083 Divide S_floating
//DIVT F-P 16.0A3 Divide T_floating
//FBEQ Bra 31 Floating branch if =  zero
//FBGE Bra 36 Floating branch if ³ zero
//FBGT Bra 37 Floating branch if > zero
//FBLE Bra 33 Floating branch if £ zero
//FBLT Bra 32 Floating branch if < zero
//FBNE Bra 35 Floating branch if ¹ zero
//FCMOVEQ F-P 17.02A FCMOVE if = zero
//FCMOVGE F-P 17.02D FCMOVE if ³ zero
//FCMOVGT F-P 17.02F FCMOVE if > zero
//FCMOVLE F-P 17.02E FCMOVE if £ zero
//FCMOVLT F-P 17.02C FCMOVE if < zero
//FCMOVNE F-P 17.02B FCMOVE if ¹ zero
//FTOIS F-P 1C.78 Floating to integer move, S_floating
//FTOIT F-P 1C.70 Floating to integer move, T_floating
//ITOFF F-P 14.014 Integer to floating move, F_floating
//ITOFS F-P 14.004 Integer to floating move, S_floating
//ITOFT F-P 14.024 Integer to floating move, T_floating
//LDF Mem 20 Load F_floating
//LDG Mem 21 Load G_floating
//LDS Mem 22 Load S_floating
//LDT Mem 23 Load T_floating
//MF_FPCR F-P 17.025 Move from FPCR
//MT_FPCR F-P 17.024 Move to FPCR
//MULF F-P 15.082 Multiply F_floating
//MULG F-P 15.0A2 Multiply G_floating
//MULS F-P 16.082 Multiply S_floating
//MULT F-P 16.0A2 Multiply T_floating
//SQRTF F-P 14.08A Square root F_floating
//SQRTG F-P 14.0AA Square root G_floating
//SQRTS F-P 14.08B Square root S_floating
//SQRTT F-P 14.0AB Square root T_floating
//STF Mem 24 Store F_floating
//STG Mem 25 Store G_floating
//STS Mem 26 Store S_floating
//STT Mem 27 Store T_floating
//SUBF F-P 15.081 Subtract F_floating
//SUBG F-P 15.0A1 Subtract G_floating
//SUBS F-P 16.081 Subtract S_floating
//SUBT F-P 16.0A1 Subtract T_floating
