//===- PowerPCInstrInfo.td - The PowerPC Instruction Set -----*- tablegen -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the subset of the 32-bit PowerPC instruction set, as used
// by the PowerPC instruction selector.
//
//===----------------------------------------------------------------------===//

include "PowerPCInstrFormats.td"

let isTerminator = 1, isReturn = 1 in
  def BLR : XLForm_2_ext<"blr", 19, 16, 20, 31, 1, 0, 0>;

class II<dag OL, string asmstr> {
  dag OperandList = OL;
  string AsmString = asmstr;
}

def u5imm   : Operand<i8> {
  let PrintMethod = "printU5ImmOperand";
}
def u16imm  : Operand<i16> {
  let PrintMethod = "printU16ImmOperand";
}


// Pseudo-instructions:
def PHI : Pseudo<"PHI">;          // PHI node...
def ADJCALLSTACKDOWN : Pseudo<"ADJCALLSTACKDOWN">;
def ADJCALLSTACKUP : Pseudo<"ADJCALLSTACKUP">;
let Defs = [LR] in
  def MovePCtoLR : Pseudo<"MovePCtoLR">;
def IMPLICIT_DEF : Pseudo<"IMPLICIT_DEF">;

def LA : DForm_2<"la", 14, 0, 0>;
def LOADHiAddr : DForm_2_r0<"addis", 15, 0, 0>;

def ADDI : DForm_2<"addi", 14, 0, 0>;
def ADDIS : DForm_2<"addis", 15, 0, 0>;
def SUBI : DForm_2<"subi", 14, 0, 0>;
def LI : DForm_2_r0<"li", 14, 0, 0>;
def LIS : DForm_2_r0<"lis", 15, 0, 0>;
def ADDIC : DForm_2<"addic",  12, 0, 0>;
def ADD : XOForm_1<"add", 31, 266, 0, 0, 0, 0>;
def ADDC : XOForm_1<"addc", 31, 10, 0, 0, 0, 0>;
def ADDE : XOForm_1<"adde", 31, 138, 0, 0, 0, 0>;
def ADDZE : XOForm_3<"addze", 31, 202, 0, 0, 0, 0>;
def ANDIo : DForm_4<28, 0, 0,
                   (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                   "andi. $dst, $src1, $src2">;

let isBranch = 1, isTerminator = 1 in {
  def COND_BRANCH : Pseudo<"COND_BRANCH">;
  def B : IForm<"b", 18, 0, 0, 0, 0>;
  // FIXME: 4*CR# needs to be added to the BI field!
  // This will only work for CR0 as it stands now
  def BLT : BForm_ext<"blt", 16, 0, 0, 12, 0, 0, 0>;
  def BLE : BForm_ext<"ble", 16, 0, 0, 4,  1, 0, 0>;
  def BEQ : BForm_ext<"beq", 16, 0, 0, 12, 2, 0, 0>;
  def BGE : BForm_ext<"bge", 16, 0, 0, 4,  0, 0, 0>;
  def BGT : BForm_ext<"bgt", 16, 0, 0, 12, 1, 0, 0>;
  def BNE : BForm_ext<"bne", 16, 0, 0, 4,  2, 0, 0>;
}

let isBranch = 1, isTerminator = 1, isCall = 1, 
  // All calls clobber the non-callee saved registers...
  Defs = [R0,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,
          F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,
          LR,XER,CTR,
          CR0,CR1,CR5,CR6,CR7] in {
  // Convenient aliases for call instructions
  def CALLpcrel : IForm<"bl", 18, 0, 1, 0, 0>;
  def CALLindirect : XLForm_2_ext<"bctrl", 19, 528, 20, 31, 1, 0, 0>;
}

def CMPI : DForm_5<"cmpi", 11, 0, 0>;
def CMPWI : DForm_5_ext<"cmpwi", 11, 0, 0>;
def CMPDI : DForm_5_ext<"cmpdi", 11, 1, 0>;
def CMP : XForm_16<"cmp", 31, 0, 0, 0>;
def CMPW : XForm_16_ext<"cmpw", 31, 0, 0, 0>;
def CMPD : XForm_16_ext<"cmpd", 31, 0, 1, 0>;
def CMPLI : DForm_6<10, 0, 0,
                    (ops CRRC:$dst, i1imm:$size, GPRC:$src1, u16imm:$src2),
                         "cmpli $dst, $size, $src1, $src2">;
def CMPLWI : DForm_6_ext<10, 0, 0,
                         (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmplwi $dst, $src1, $src2">;
def CMPLDI : DForm_6_ext<10, 1, 0,
                         (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2">;
def CMPL : XForm_16<"cmpl", 31, 32, 0, 0>;
def CMPLW : XForm_16_ext<"cmplw", 31, 32, 0, 0>;
def CMPLD : XForm_16_ext<"cmpld", 31, 32, 1, 0>;
def DIVW : XOForm_1<"divw", 31, 491, 0, 0, 0, 0>;
def DIVWU : XOForm_1<"divwu", 31, 459, 0, 0, 0, 0>;
def FADD : AForm_2<"fadd", 63, 21, 0, 0, 0>;
def FADDS : AForm_2<"fadds", 59, 21, 0, 0, 0>;
def FSUB : AForm_2<"fsub", 63, 20, 0, 0, 0>;
def FSUBS : AForm_2<"fsubs", 59, 20, 0, 0, 0>;
def FMUL : AForm_3<"fmul", 63, 25, 0, 0, 0>;
def FMULS : AForm_3<"fmuls", 59, 25, 0, 0, 0>;
def FDIV : AForm_2<"fdiv", 63, 18, 0, 0, 0>;
def FDIVS : AForm_2<"fdivs", 59, 18, 0, 0, 0>;
def FSEL : AForm_1<"fsel", 63, 23, 0, 0, 0>;
def FCMPU : XForm_17<"fcmpu", 63, 0, 0, 0>;
def LBZ : DForm_1<"lbz", 35, 0, 0>;
def LHA : DForm_1<"lha", 42, 0, 0>;
def LHZ : DForm_1<"lhz", 40, 0, 0>;
def LWZ : DForm_1<"lwz", 32, 0, 0>;
def LWA : DSForm_1<"lwa", 58, 2, 1, 0>;
def LD : DSForm_2<"ld", 58, 0, 1, 0>;
def LMW : DForm_1<"lmw", 46, 0, 0>;
def STMW : DForm_3<"stmw", 47, 0, 0>;
def LFS : DForm_8<"lfs", 48, 0, 0>;
def LFD : DForm_8<"lfd", 50, 0, 0>;
def MFLR : XFXForm_1_ext<"", 31, 399, 8, 0, 0>,
                      II<(ops GPRC:$reg), "mflr $reg">;
def MFCTR : XFXForm_1_ext<"mfctr", 31, 399, 9, 0, 0>;
def MTLR : XFXForm_7_ext<"mtlr", 31, 467, 8, 0, 0>;
def MTCTR : XFXForm_7_ext<"mtctr", 31, 467, 9, 0, 0>;
def MULLD : XOForm_1<"mulld", 31, 233, 0, 0, 1, 0>;
def MULLW : XOForm_1<"mullw", 31, 235, 0, 0, 0, 0>;
def MULHWU : XOForm_2<"mulhwu", 31, 11, 0, 0, 0>;
def NEG : XOForm_3<"neg", 31, 104, 0, 0, 0, 0>;
def NOP : DForm_4_zero<"nop", 24, 0, 0, (ops), "nop">;
def ORI  : DForm_4<24, 0, 0,
                   (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                   "ori $dst, $src1, $src2">;
def ORIS : DForm_4<25, 0, 0,
                   (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                   "oris $dst, $src1, $src2">;
def RLDICL : MDForm_1<"rldicl", 30, 0, 0, 1, 0>;
def RLDICR : MDForm_1<"rldicr", 30, 1, 0, 1, 0>;
def RLWINM : MForm_2<"rlwinm", 21, 0, 0, 0>;
def RLWNM : MForm_1<"rlwnm", 23, 0, 0, 0>;
def RLWIMI : MForm_2<"rlwimi", 20, 0, 0, 0>;
def SRADI  : XSForm_1<"sradi",  31, 413, 0, 1, 0>;
def SRWI : MForm_2<"srwi", 21, 0, 0, 0>;
def STB : DForm_3<"stb", 38, 0, 0>;
def STBU : DForm_3<"stbu", 39, 0, 0>;
def STH : DForm_3<"sth", 44, 0, 0>;
def STHU : DForm_3<"sthu", 45, 0, 0>;
def STW : DForm_3<"stw", 36, 0, 0>;
def STWU : DForm_3<"stwu", 37, 0, 0>;
def STD : DSForm_2<"std", 62, 0, 1, 0>;
def STDU : DSForm_2<"stdu", 62, 1, 1, 0>;
def STFS : DForm_9<"stfs", 52, 0, 0>;
def STFD : DForm_9<"stfd", 54, 0, 0>;
def SUBFIC : DForm_2<"subfic", 8, 0, 0>;
def SUB : XOForm_1_rev<"sub", 31, 40, 0, 0, 0, 0>;
def SUBF : XOForm_1<"subf", 31, 40, 0, 0, 0, 0>;
def SUBC : XOForm_1_rev<"subc", 31, 8, 0, 0, 0, 0>;
def SUBFC : XOForm_1<"subfc", 31, 8, 0, 0, 0, 0>;
def SUBFE : XOForm_1<"subfe", 31, 136, 0, 0, 0, 0>;
def SUBFZE : XOForm_3<"subfze", 31, 200, 0, 0, 0, 0>;
def XORI  : DForm_4<26, 0, 0,
                   (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                   "xori $dst, $src1, $src2">;
def XORIS : DForm_4<27, 0, 0,
                   (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                   "xoris $dst, $src1, $src2">;
def MULLI : DForm_2<"mulli", 7, 0, 0>;


def LBZX : XForm_1<31,  87, 0, 0, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lbzx $dst, $base, $index">;
def LHAX : XForm_1<31, 343, 0, 0, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lhax $dst, $base, $index">;
def LHZX : XForm_1<31, 279, 0, 0, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lhzx $dst, $base, $index">;
def LWAX : XForm_1<31, 341, 1, 0, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lwax $dst, $base, $index">;
def LWZX : XForm_1<31,  23, 0, 0, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lwzx $dst, $base, $index">;
def LDX  : XForm_1<31,  21, 1, 0, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "ldx $dst, $base, $index">;
def MFCR : XForm_5<31,  19, 0, 0, (ops GPRC:$dst), "mfcr $dst">;
def AND  : XForm_6<31,  28, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "and $rA, $rS, $rB">;
def ANDC : XForm_6<31,  60, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "andc $rA, $rS, $rB">;
def EQV  : XForm_6<31, 284, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "eqv $rA, $rS, $rB">;
def NAND : XForm_6<31, 476, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "nand $rA, $rS, $rB">;
def NOR  : XForm_6<31, 124, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "nor $rA, $rS, $rB">;
def OR   : XForm_6<31, 444, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "or $rA, $rS, $rB">;
def ORo  : XForm_6<31, 444, 1, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "or. $rA, $rS, $rB">;
def ORC  : XForm_6<31, 412, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "orc $rA, $rS, $rB">;
def SLD  : XForm_6<31,  27, 0, 1, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "sld $rA, $rS, $rB">;
def SLW  : XForm_6<31,  24, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "slw $rA, $rS, $rB">;
def SRD  : XForm_6<31, 539, 0, 1, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srd $rA, $rS, $rB">;
def SRW  : XForm_6<31, 536, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srw $rA, $rS, $rB">;
def SRAD : XForm_6<31, 794, 0, 1, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srad $rA, $rS, $rB">;
def SRAW : XForm_6<31, 792, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "sraw $rA, $rS, $rB">;
def XOR  : XForm_6<31, 316, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "xor $rA, $rS, $rB">;
def STBX  : XForm_8<31, 215, 0, 0, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stbx $rS, $rA, $rB">;
def STHX  : XForm_8<31, 407, 0, 0, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "sthx $rS, $rA, $rB">;
def STWX  : XForm_8<31, 151, 0, 0, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stwx $rS, $rA, $rB">;
def STWUX : XForm_8<31, 183, 0, 0, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stwux $rS, $rA, $rB">;
def STDX  : XForm_8<31, 149, 1, 0, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stdx $rS, $rA, $rB">;
def STDUX : XForm_8<31, 181, 1, 0, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stdux $rS, $rA, $rB">;
def SRAWI : XForm_10<31, 824, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS, u5imm:$SH), 
                     "srawi $rA, $rS, $SH">;
def CNTLZW : XForm_11<31,  26, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS),
                      "cntlzw $rA, $rS">;
def EXTSB  : XForm_11<31, 954, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS),
                      "extsb $rA, $rS">;
def EXTSH  : XForm_11<31, 922, 0, 0, 0, (ops GPRC:$rA, GPRC:$rS),
                      "extsh $rA, $rS">;
def LFSX : XForm_25<31, 535, 0, 0, (ops FPRC:$dst, GPRC:$base, GPRC:$index),
                   "lfsx $dst, $base, $index">;
def LFDX : XForm_25<31, 599, 0, 0, (ops FPRC:$dst, GPRC:$base, GPRC:$index),
                   "lfdx $dst, $base, $index">;
def FMR    : XForm_26<63, 72, 0, 0, 0, (ops FPRC:$frD, FPRC:$frB),
                      "fmr $frD, $frB">;
def FNEG   : XForm_26<63, 80, 0, 0, 0, (ops FPRC:$frD, FPRC:$frB),
                      "fneg $frD, $frB">;
def FRSP   : XForm_26<63, 12, 0, 0, 0, (ops FPRC:$frD, FPRC:$frB),
                      "frsp $frD, $frB">;
def FCTIWZ : XForm_26<63, 15, 0, 0, 0, (ops FPRC:$frD, FPRC:$frB),
                      "fctiwz $frD, $frB">;
def STFSX : XForm_28<31, 663, 0, 0, (ops FPRC:$frS, GPRC:$rA, GPRC:$rB),
                     "stfsx $frS, $rA, $rB">;
def STFDX : XForm_28<31, 727, 0, 0, (ops FPRC:$frS, GPRC:$rA, GPRC:$rB),
                     "stfdx $frS, $rA, $rB">;
def CRAND  : XLForm_1<19, 257, 0, 0, (ops u5imm:$D, u5imm:$A, u5imm:$B),
                      "crand $D, $A, $B">;
def CRANDC : XLForm_1<19, 129, 0, 0, (ops u5imm:$D, u5imm:$A, u5imm:$B),
                      "crandc $D, $A, $B">;
def CRNOR  : XLForm_1<19,  33, 0, 0, (ops u5imm:$D, u5imm:$A, u5imm:$B),
                      "crnor $D, $A, $B">;
def CROR   : XLForm_1<19, 449, 0, 0, (ops u5imm:$D, u5imm:$A, u5imm:$B),
                      "cror $D, $A, $B">;
