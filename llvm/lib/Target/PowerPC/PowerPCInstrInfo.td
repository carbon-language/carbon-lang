//===- PowerPCInstrInfo.td - The PowerPC Instruction Set -----*- tablegen -*-=//
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

include "PowerPCInstrFormats.td"

let isTerminator = 1, isReturn = 1 in
	def BLR : PPC32InstPattern11 <"blr", Imm2, 160768, 32, 0, 0>;

// Pseudo-instructions:
def PHI : PPC32InstPatternPseudo<"PHI", Pseudo>;          // PHI node...
def ADJCALLSTACKDOWN : PPC32InstPatternPseudo<"ADJCALLSTACKDOWN", Pseudo>;
def ADJCALLSTACKUP : PPC32InstPatternPseudo<"ADJCALLSTACKUP", Pseudo>;
let Defs = [LR] in
  def MovePCtoLR : PPC32InstPatternPseudo<"MovePCtoLR", Pseudo>;
def IMPLICIT_DEF : PPC32InstPatternPseudo<"IMPLICIT_DEF", Pseudo>;

def LOADLoIndirect : DForm_2_r0 <"lwz", 14, 0, 0>;
def LOADLoDirect : DForm_2_r0<"la", 14, 0, 0>;
def LOADHiAddr : DForm_2_r0<"addis", 15, 0, 0>;

def ADDI : DForm_2_r0<"addi", 14, 0, 0>;
def SUBI : DForm_2_r0 <"subi", 14, 0, 0>;
def LI : PPC32InstPattern1 <"li", Gpr, Simm16, 14, 0, 0, 0>;
def ADDIS : DForm_2_r0 <"addis", 15, 0, 0>;
def LIS : PPC32InstPattern1 <"lis", Gpr, Simm16, 15, 0, 0, 0>;
def ADDIC  : DForm_2<"addic",  12, 0, 0>;
def ADD : PPC32InstPattern2 <"add", Gpr, Gpr, Gpr, 31, 532, 0, 0>;
def ADDC : PPC32InstPattern2 <"addc", Gpr, Gpr, Gpr, 31, 20, 0, 0>;
def ADDE : PPC32InstPattern2 <"adde", Gpr, Gpr, Gpr, 31, 276, 0, 0>;
def ADDZE : PPC32InstPattern3 <"addze", Gpr, Gpr, 31, 404, 0, 0>;
def ANDIo  : DForm_4<"andi.",  28, 0, 0>;
def AND  : XForm_6<"and",  31, 28, 0, 0, 0>;
def ANDC  : XForm_6<"andc",  31, 60, 0, 0, 0>;

let isBranch = 1, isTerminator = 1 in {
  def COND_BRANCH : PPC32InstPatternPseudo<"COND_BRANCH", Pseudo>;
  def B : PPC32InstPattern6 <"b", PCRelimm24, 18, 0, 0, 0>;
  def BLT : PPC32InstPattern9 <"blt", Crf, PCRelimm14, 524, 0, 0, 0>;
  def BLE : PPC32InstPattern9 <"ble", Crf, PCRelimm14, 516, 0, 0, 0>;
  def BEQ : PPC32InstPattern9 <"beq", Crf, PCRelimm14, 524, 0, 0, 0>;
  def BGE : PPC32InstPattern9 <"bge", Crf, PCRelimm14, 516, 0, 0, 0>;
  def BGT : PPC32InstPattern9 <"bgt", Crf, PCRelimm14, 524, 0, 0, 0>;
  def BNE : PPC32InstPattern9 <"bne", Crf, PCRelimm14, 516, 0, 0, 0>;
}

let isBranch = 1, isTerminator = 1, isCall = 1, 
  // All calls clobber the non-callee saved registers...
  Defs = [R0,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,
          F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,
          LR,XER,CTR,
          CR0,CR1,CR5,CR6,CR7] in {
  // Convenient aliases for call instructions
  def CALLpcrel : PPC32InstPattern6 <"bl", PCRelimm24, 18, 1, 0, 0>;
  def CALLindirect : PPC32InstPattern3 <"bctrl", Imm5, Imm5, 19, 1057, 0, 0>;

  def BL : PPC32InstPattern6 <"bl", PCRelimm24, 18, 1, 0, 0>;
}

def CMPI : PPC32InstPattern13 <"cmpi", Imm3, Imm1, Gpr, Simm16, 11, 0, 0, 0>;
def CMPWI : PPC32InstPattern14 <"cmpwi", Imm3, Gpr, Simm16, 11, 0, 0, 0>;
def CMPW : PPC32InstPattern16 <"cmpw", Imm3, Gpr, Gpr, 31, 0, 0, 0, 0>;
def CMPLI : PPC32InstPattern13 <"cmpli", Imm3, Imm1, Gpr, Zimm16, 10, 0, 0, 0>;
def CMPLWI : PPC32InstPattern14 <"cmplwi", Imm3, Gpr, Zimm16, 10, 0, 0, 0>;
def CMPL : PPC32InstPattern15 <"cmpl", Imm3, Imm1, Gpr, Gpr, 31, 0, 64, 0, 0>;
def CMPLW : PPC32InstPattern16 <"cmplw", Imm3, Gpr, Gpr, 31, 0, 64, 0, 0>;
def CRAND : PPC32InstPattern2 <"crand", Imm5, Imm5, Imm5, 19, 514, 0, 0>;
def CRANDC : PPC32InstPattern2 <"crandc", Imm5, Imm5, Imm5, 19, 258, 0, 0>;
def CRNOR : PPC32InstPattern2 <"crnor", Imm5, Imm5, Imm5, 19, 66, 0, 0>;
def CROR : PPC32InstPattern2 <"cror", Imm5, Imm5, Imm5, 19, 898, 0, 0>;
def DIVW : PPC32InstPattern2 <"divw", Gpr, Gpr, Gpr, 31, 982, 0, 0>;
def DIVWU : PPC32InstPattern2 <"divwu", Gpr, Gpr, Gpr, 31, 918, 0, 0>;
def EXTSB : PPC32InstPattern17 <"extsb", Gpr, Gpr, 31, 1908, 0, 0>;
def EXTSH : PPC32InstPattern17 <"extsh", Gpr, Gpr, 31, 1844, 0, 0>;
def FADD : PPC32InstPattern2 <"fadd", Fpr, Fpr, Fpr, 63, 42, 0, 0>;
def FADDS : PPC32InstPattern2 <"fadds", Fpr, Fpr, Fpr, 59, 42, 0, 0>;
def FSUB : PPC32InstPattern2 <"fsub", Fpr, Fpr, Fpr, 63, 40, 0, 0>;
def FSUBS : PPC32InstPattern2 <"fsubs", Fpr, Fpr, Fpr, 59, 40, 0, 0>;
def FMUL : PPC32InstPattern18 <"fmul", Fpr, Fpr, Fpr, 63, 0, 18, 0, 0>;
def FMULS : PPC32InstPattern18 <"fmuls", Fpr, Fpr, Fpr, 59, 0, 18, 0, 0>;
def FDIV : PPC32InstPattern2 <"fdiv", Fpr, Fpr, Fpr, 63, 36, 0, 0>;
def FDIVS : PPC32InstPattern2 <"fdivs", Fpr, Fpr, Fpr, 59, 36, 0, 0>;
def FMR : PPC32InstPattern20 <"fmr", Fpr, Fpr, 63, 0, 144, 0, 0>;
def FNEG : PPC32InstPattern20 <"fneg", Fpr, Fpr, 63, 0, 80, 0, 0>;
def FRSP : PPC32InstPattern20 <"frsp", Fpr, Fpr, 63, 0, 24, 0, 0>;
def FSEL : PPC32InstPattern19 <"fsel", Fpr, Fpr, Fpr, Fpr, 63, 14, 0, 0>;
def FCTIW : PPC32InstPattern20 <"fctiw", Fpr, Fpr, 63, 0, 28, 0, 0>;
def FCTIWZ : PPC32InstPattern20 <"fctiwz", Fpr, Fpr, 63, 0, 30, 0, 0>;
def FCMPU : PPC32InstPattern16 <"fcmpu", Imm3, Fpr, Fpr, 63, 0, 0, 0, 0>;
def LBZ : PPC32InstPattern25 <"lbz", Gpr, Disimm16, Gpr0, 34, 0, 0>;
def LBZX : PPC32InstPattern2 <"lbzx", Gpr, Gpr0, Gpr, 31, 174, 0, 0>;
def LHZ : PPC32InstPattern25 <"lhz", Gpr, Disimm16, Gpr0, 40, 0, 0>;
def LHZX : PPC32InstPattern2 <"lhzx", Gpr, Gpr0, Gpr, 31, 558, 0, 0>;
def LHA : PPC32InstPattern25 <"lha", Gpr, Disimm16, Gpr0, 42, 0, 0>;
def LHAX : PPC32InstPattern2 <"lhax", Gpr, Gpr0, Gpr, 31, 686, 0, 0>;
def LWZ : PPC32InstPattern25 <"lwz", Gpr, Disimm16, Gpr0, 32, 0, 0>;
def LWZX : PPC32InstPattern2 <"lwzx", Gpr, Gpr0, Gpr, 31, 46, 0, 0>;
def LMW : PPC32InstPattern25 <"lmw", Gpr, Disimm16, Gpr0, 46, 0, 0>;
def STMW : PPC32InstPattern25 <"stmw", Gpr, Disimm16, Gpr0, 47, 0, 0>;
def LFS : PPC32InstPattern25 <"lfs", Fpr, Disimm16, Gpr0, 48, 0, 0>;
def LFSX : PPC32InstPattern2 <"lfsx", Fpr, Gpr0, Gpr, 31, 46, 0, 0>;
def LFD : PPC32InstPattern25 <"lfd", Fpr, Disimm16, Gpr0, 50, 0, 0>;
def LFDX : PPC32InstPattern2 <"lfdx", Fpr, Gpr0, Gpr, 31, 174, 0, 0>;
def MFCR : PPC32InstPattern32 <"mfcr", Gpr, Imm8, 31, 0, 38, 0, 0>;
def MFLR : PPC32InstPattern21 <"mflr", Gpr, 31, 524966, 0, 0>;
def MFCTR : PPC32InstPattern21 <"mfctr", Gpr, 31, 590502, 0, 0>;
def MTLR : PPC32InstPattern21 <"mtlr", Gpr, 31, 525222, 0, 0>;
def MTCTR : PPC32InstPattern21 <"mtctr", Gpr, 31, 590758, 0, 0>;
def MULLW : PPC32InstPattern2 <"mullw", Gpr, Gpr, Gpr, 31, 470, 0, 0>;
def MULHWU : PPC32InstPattern2 <"mulhwu", Gpr, Gpr, Gpr, 31, 22, 0, 0>;
def NAND  : XForm_6<"nand",  31, 476, 0, 0, 0>;
def NEG : PPC32InstPattern3 <"neg", Gpr, Gpr, 31, 208, 0, 0>;
def NOR  : XForm_6<"nor",  31, 124, 0, 0, 0>;
def NOP : PPC32InstPattern33 <"nop", 1610612736, 0, 0>;
def ORI  : DForm_4<"ori",  24, 0, 0>;
def ORIS : DForm_4<"oris", 25, 0, 0>;
def OR  : XForm_6<"or",  31, 444, 0, 0, 0>;
def ORo : XForm_6<"or.", 31, 444, 1, 0, 0>;
def RLWINM : PPC32InstPattern34 <"rlwinm", Gpr, Gpr, Imm5, Imm5, Imm5, 21, 0, 0, 0>;
def RLWNM : PPC32InstPattern34 <"rlwnm", Gpr, Gpr, Gpr, Imm5, Imm5, 23, 0, 0, 0>;
def RLWIMI : PPC32InstPattern34 <"rlwimi", Gpr, Gpr, Imm5, Imm5, Imm5, 20, 0, 0, 0>;
def SLW  : XForm_6<"slw",  31, 24, 0, 0, 0>;
def SRW  : XForm_6<"srw",  31, 24, 0, 0, 0>;
def SRAWI  : XForm_10<"srawi",  31, 824, 0, 0, 0>;
def SRAW  : XForm_6<"sraw",  31, 280, 0, 0, 0>;
def STB : PPC32InstPattern25 <"stb", Gpr, Disimm16, Gpr0, 38, 0, 0>;
def STBU : PPC32InstPattern25 <"stbu", Gpr, Disimm16, Gpr, 39, 0, 0>;
def STBX : PPC32InstPattern2 <"stbx", Gpr, Gpr0, Gpr, 31, 430, 0, 0>;
def STH : PPC32InstPattern25 <"sth", Gpr, Disimm16, Gpr0, 44, 0, 0>;
def STHU : PPC32InstPattern25 <"sthu", Gpr, Disimm16, Gpr, 45, 0, 0>;
def STHX : PPC32InstPattern2 <"sthx", Gpr, Gpr0, Gpr, 31, 814, 0, 0>;
def STW : PPC32InstPattern25 <"stw", Gpr, Disimm16, Gpr0, 36, 0, 0>;
def STWU : PPC32InstPattern25 <"stwu", Gpr, Disimm16, Gpr, 37, 0, 0>;
def STWX : PPC32InstPattern2 <"stwx", Gpr, Gpr0, Gpr, 31, 302, 0, 0>;
def STWUX : PPC32InstPattern2 <"stwux", Gpr, Gpr, Gpr, 31, 366, 0, 0>;
def STDX : PPC32InstPattern2 <"stdx", Gpr, Gpr0, Gpr, 31, 298, 1, 0>;
def STFS : PPC32InstPattern25 <"stfs", Fpr, Disimm16, Gpr0, 52, 0, 0>;
def STFSX : PPC32InstPattern2 <"stfsx", Fpr, Gpr0, Gpr, 31, 302, 0, 0>;
def STFD : PPC32InstPattern25 <"stfd", Fpr, Disimm16, Gpr0, 54, 0, 0>;
def SUBFIC : DForm_2<"subfic", 8, 0, 0>;
def SUB : PPC32InstPattern35 <"sub", Gpr, Gpr, Gpr, 31, 80, 0, 0>;
def SUBF : PPC32InstPattern2 <"subf", Gpr, Gpr, Gpr, 31, 80, 0, 0>;
def SUBC : PPC32InstPattern35 <"subc", Gpr, Gpr, Gpr, 31, 16, 0, 0>;
def SUBFC : PPC32InstPattern2 <"subfc", Gpr, Gpr, Gpr, 31, 16, 0, 0>;
def SUBFE : PPC32InstPattern2 <"subfe", Gpr, Gpr, Gpr, 31, 272, 0, 0>;
def SUBFZE : PPC32InstPattern3 <"subfze", Gpr, Gpr, 31, 400, 0, 0>;
def XORI  : DForm_4<"xori",  26, 0, 0>;
def XORIS : DForm_4<"xoris", 27, 0, 0>;
def XOR  : XForm_6<"xor",  31, 316, 0, 0, 0>;
def MULLI : DForm_2 <"mulli", 7, 0, 0>;
