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

// Pseudo-instructions:
def PHI : Pseudo<"PHI">;          // PHI node...
def ADJCALLSTACKDOWN : Pseudo<"ADJCALLSTACKDOWN">;
def ADJCALLSTACKUP : Pseudo<"ADJCALLSTACKUP">;
let Defs = [LR] in
  def MovePCtoLR : Pseudo<"MovePCtoLR">;
def IMPLICIT_DEF : Pseudo<"IMPLICIT_DEF">;

def LOADLoIndirect : DForm_2_r0 <"lwz", 14, 0, 0>;
def LOADLoDirect : DForm_2_r0<"la", 14, 0, 0>;
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
def ANDIo : DForm_4<"andi.",  28, 0, 0>;
def AND  : XForm_6<"and",  31, 28, 0, 0, 0>;
def ANDC  : XForm_6<"andc",  31, 60, 0, 0, 0>;

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
def CMPW : XForm_16 <"cmpw", 31, 0, 0, 0>;
def CMPLI : DForm_6<"cmpli", 10, 0, 0>;
def CMPLWI : DForm_6_ext<"cmplwi", 10, 0, 0>;
def CMPL : XForm_16<"cmpl", 31, 32, 0, 0>;
def CMPLW : XForm_16_ext<"cmplw", 31, 32, 0, 0>;
def CRAND : XLForm_1<"crand", 19, 257, 0, 0>;
def CRANDC : XLForm_1<"crandc", 19, 129, 0, 0>;
def CRNOR : XLForm_1<"crnor", 19, 33, 0, 0>;
def CROR : XLForm_1<"cror", 19, 449, 0, 0>;
def DIVW : XOForm_1<"divw", 31, 491, 0, 0, 0, 0>;
def DIVWU : XOForm_1<"divwu", 31, 459, 0, 0, 0, 0>;
def EXTSB : XForm_11<"extsb", 31, 954, 0, 0, 0>;
def EXTSH : XForm_11<"extsh", 31, 922, 0, 0, 0>;
def FADD : AForm_2<"fadd", 63, 21, 0, 0, 0>;
def FADDS : AForm_2<"fadds", 59, 21, 0, 0, 0>;
def FSUB : AForm_2<"fsub", 63, 20, 0, 0, 0>;
def FSUBS : AForm_2<"fsubs", 59, 20, 0, 0, 0>;
def FMUL : AForm_3<"fmul", 63, 25, 0, 0, 0>;
def FMULS : AForm_3<"fmuls", 59, 25, 0, 0, 0>;
def FDIV : AForm_2<"fdiv", 63, 18, 0, 0, 0>;
def FDIVS : AForm_2<"fdivs", 59, 18, 0, 0, 0>;
def FMR : XForm_26<"fmr", 63, 72, 0, 0, 0>;
def FNEG : XForm_26<"fneg", 63, 80, 0, 0, 0>;
def FRSP : XForm_26<"frsp", 63, 12, 0, 0, 0>;
def FSEL : AForm_1<"fsel", 63, 23, 0, 0, 0>;
def FCTIW : XForm_26<"fctiw", 63, 14, 0, 0, 0>;
def FCTIWZ : XForm_26<"fctiwz", 63, 15, 0, 0, 0>;
def FCMPU : XForm_17<"fcmpu", 63, 0, 0, 0>;
def LBZ : DForm_1<"lbz", 35, 0, 0>;
def LBZX : XForm_1<"lbzx", 31, 87, 0, 0>;
def LHZ : DForm_1<"lhz", 40, 0, 0>;
def LHZX : XForm_1<"lhzx", 31, 279, 0, 0>;
def LHA : DForm_1<"lha", 42, 0, 0>;
def LHAX : XForm_1<"lhax", 31, 343, 0, 0>;
def LWZ : DForm_1<"lwz", 32, 0, 0>;
def LWZX : XForm_1<"lwzx", 31, 23, 0, 0>;
def LMW : DForm_1<"lmw", 46, 0, 0>;
def STMW : DForm_3<"stmw", 47, 0, 0>;
def LFS : DForm_8<"lfs", 48, 0, 0>;
def LFSX : XForm_25<"lfsx", 31, 535, 0, 0>;
def LFD : DForm_8<"lfd", 50, 0, 0>;
def LFDX : XForm_25<"lfdx", 31, 599, 0, 0>;
def MFCR : XForm_5<"mfcr", 31, 19, 0, 0>;
def MFLR : XFXForm_1_ext<"mflr", 31, 399, 8, 0, 0>;
def MFCTR : XFXForm_1_ext<"mfctr", 31, 399, 9, 0, 0>;
def MTLR : XFXForm_7_ext<"mtlr", 31, 467, 8, 0, 0>;
def MTCTR : XFXForm_7_ext<"mtctr", 31, 467, 9, 0, 0>;
def MULLW : XOForm_1<"mullw", 31, 235, 0, 0, 0, 0>;
def MULHWU : XOForm_2<"mulhwu", 31, 11, 0, 0, 0>;
def NAND  : XForm_6<"nand",  31, 476, 0, 0, 0>;
def NEG : XOForm_3<"neg", 31, 104, 0, 0, 0, 0>;
def NOR  : XForm_6<"nor",  31, 124, 0, 0, 0>;
def NOP : DForm_4_zero<"nop", 24, 0, 0>;
def ORI  : DForm_4<"ori",  24, 0, 0>;
def ORIS : DForm_4<"oris", 25, 0, 0>;
def OR  : XForm_6<"or",  31, 444, 0, 0, 0>;
def ORo : XForm_6<"or.", 31, 444, 1, 0, 0>;
def RLWINM : MForm_2<"rlwinm", 21, 0, 0, 0>;
def RLWNM : MForm_1<"rlwnm", 23, 0, 0, 0>;
def RLWIMI : MForm_2<"rlwimi", 20, 0, 0, 0>;
def SLW  : XForm_6<"slw",  31, 24, 0, 0, 0>;
def SRW  : XForm_6<"srw",  31, 24, 0, 0, 0>;
def SRAWI  : XForm_10<"srawi",  31, 824, 0, 0, 0>;
def SRAW  : XForm_6<"sraw",  31, 280, 0, 0, 0>;
def STB : DForm_3<"stb", 38, 0, 0>;
def STBU : DForm_3<"stbu", 39, 0, 0>;
def STBX : XForm_8<"stbx", 31, 215, 0, 0>;
def STH : DForm_3<"sth", 44, 0, 0>;
def STHU : DForm_3<"sthu", 45, 0, 0>;
def STHX : XForm_8<"sthx", 31, 407, 0, 0>;
def STW : DForm_3<"stw", 36, 0, 0>;
def STWU : DForm_3<"stwu", 37, 0, 0>;
def STWX : XForm_8<"stwx", 31, 151, 0, 0>;
def STWUX : XForm_8<"stwux", 31, 183, 0, 0>;
def STDX : XForm_8<"stdx", 31, 149, 1, 0>;
def STFS : DForm_9<"stfs", 52, 0, 0>;
def STFSX : XForm_28<"stfsx", 31, 302, 0, 0>;
def STFD : DForm_9<"stfd", 54, 0, 0>;
def SUBFIC : DForm_2<"subfic", 8, 0, 0>;
def SUB : XOForm_1_rev<"sub", 31, 40, 0, 0, 0, 0>;
def SUBF : XOForm_1<"subf", 31, 40, 0, 0, 0, 0>;
def SUBC : XOForm_1_rev<"subc", 31, 8, 0, 0, 0, 0>;
def SUBFC : XOForm_1<"subfc", 31, 8, 0, 0, 0, 0>;
def SUBFE : XOForm_1<"subfe", 31, 136, 0, 0, 0, 0>;
def SUBFZE : XOForm_3<"subfze", 31, 200, 0, 0, 0, 0>;
def XORI  : DForm_4<"xori",  26, 0, 0>;
def XORIS : DForm_4<"xoris", 27, 0, 0>;
def XOR  : XForm_6<"xor",  31, 316, 0, 0, 0>;
def MULLI : DForm_2 <"mulli", 7, 0, 0>;
