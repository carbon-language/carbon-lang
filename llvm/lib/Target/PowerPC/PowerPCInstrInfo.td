
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

class isPPC64 { bit PPC64 = 1; }
class isVMX   { bit VMX = 1; }
class isDOT   {
  list<Register> Defs = [CR0];
  bit RC  = 1;
}

let isTerminator = 1 in {
  let isReturn = 1 in
    def BLR : XLForm_2_ext<19, 16, 20, 0, 0, (ops), "blr">;
  def BCTR : XLForm_2_ext<19, 528, 20, 0, 0, (ops), "bctr">;
}

def u5imm   : Operand<i8> {
  let PrintMethod = "printU5ImmOperand";
}
def u6imm   : Operand<i8> {
  let PrintMethod = "printU6ImmOperand";
}
def s16imm  : Operand<i16> {
  let PrintMethod = "printS16ImmOperand";
}
def u16imm  : Operand<i16> {
  let PrintMethod = "printU16ImmOperand";
}
def target : Operand<i32> {
  let PrintMethod = "printBranchOperand";
}
def piclabel: Operand<i32> {
  let PrintMethod = "printPICLabel";
}
def symbolHi: Operand<i32> {
  let PrintMethod = "printSymbolHi";
}
def symbolLo: Operand<i32> {
  let PrintMethod = "printSymbolLo";
}
def crbit: Operand<i8> {
  let PrintMethod = "printcrbit";
}
def crbitm: Operand<i8> {
  let PrintMethod = "printcrbitm";
}

// Pseudo-instructions:
def PHI : Pseudo<(ops), "; PHI">;
let isLoad = 1 in {
def ADJCALLSTACKDOWN : Pseudo<(ops), "; ADJCALLSTACKDOWN">;
def ADJCALLSTACKUP : Pseudo<(ops), "; ADJCALLSTACKUP">;
}
def IMPLICIT_DEF : Pseudo<(ops), "; IMPLICIT_DEF">;

let Defs = [LR] in
  def MovePCtoLR : Pseudo<(ops piclabel:$label), "bl $label">;

let isBranch = 1, isTerminator = 1 in {
  def COND_BRANCH : Pseudo<(ops), "; COND_BRANCH">;
  def B   : IForm<18, 0, 0, (ops target:$func), "b $func">;
//def BA  : IForm<18, 1, 0, (ops target:$func), "ba $func">;
  def BL  : IForm<18, 0, 1, (ops target:$func), "bl $func">;
//def BLA : IForm<18, 1, 1, (ops target:$func), "bla $func">;

  // FIXME: 4*CR# needs to be added to the BI field!
  // This will only work for CR0 as it stands now
  def BLT : BForm_ext<16, 0, 0, 12, 0, (ops CRRC:$crS, target:$block),
                      "blt $block">;
  def BLE : BForm_ext<16, 0, 0, 4,  1, (ops CRRC:$crS, target:$block),
                      "ble $block">;
  def BEQ : BForm_ext<16, 0, 0, 12, 2, (ops CRRC:$crS, target:$block),
                      "beq $block">;
  def BGE : BForm_ext<16, 0, 0, 4,  0, (ops CRRC:$crS, target:$block),
                      "bge $block">;
  def BGT : BForm_ext<16, 0, 0, 12, 1, (ops CRRC:$crS, target:$block),
                      "bgt $block">;
  def BNE : BForm_ext<16, 0, 0, 4,  2, (ops CRRC:$crS, target:$block),
                      "bne $block">;
}

let isCall = 1, 
  // All calls clobber the non-callee saved registers...
  Defs = [R0,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,
          F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,
          LR,XER,CTR,
          CR0,CR1,CR5,CR6,CR7] in {
  // Convenient aliases for call instructions
  def CALLpcrel : IForm<18, 0, 1, (ops target:$func), "bl $func">;
  def CALLindirect : XLForm_2_ext<19, 528, 20, 0, 1, (ops), "bctrl">;
}

// D-Form instructions.  Most instructions that perform an operation on a
// register and an immediate are of this type.
//
let isLoad = 1 in {
def LBZ : DForm_1<34, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lbz $rD, $disp($rA)">;
def LHA : DForm_1<42, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lha $rD, $disp($rA)">;
def LHZ : DForm_1<40, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lhz $rD, $disp($rA)">;
def LMW : DForm_1<46, (ops GPRC:$rD, s16imm:$disp, GPRC:$rA),
                  "lmw $rD, $disp($rA)">;
def LWZ : DForm_1<32, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lwz $rD, $disp($rA)">;
def LWZU : DForm_1<35, (ops GPRC:$rD, s16imm:$disp, GPRC:$rA),
                   "lwzu $rD, $disp($rA)">;
}
def ADDI   : DForm_2<14, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "addi $rD, $rA, $imm">;
def ADDIC  : DForm_2<12, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "addic $rD, $rA, $imm">;
def ADDICo : DForm_2<13, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "addic. $rD, $rA, $imm">;
def ADDIS  : DForm_2<15, (ops GPRC:$rD, GPRC:$rA, symbolHi:$imm),
                     "addis $rD, $rA, $imm">;
def LA     : DForm_2<14, (ops GPRC:$rD, GPRC:$rA, symbolLo:$sym),
                     "la $rD, $sym($rA)">;
def MULLI  : DForm_2< 7, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "mulli $rD, $rA, $imm">;
def SUBFIC : DForm_2< 8, (ops GPRC:$rD, GPRC:$rA, s16imm:$imm),
                     "subfic $rD, $rA, $imm">;
def LI  : DForm_2_r0<14, (ops GPRC:$rD, s16imm:$imm),
                     "li $rD, $imm">;
def LIS : DForm_2_r0<15, (ops GPRC:$rD, symbolHi:$imm),
                     "lis $rD, $imm">;
let isStore = 1 in {
def STMW : DForm_3<47, (ops GPRC:$rS, s16imm:$disp, GPRC:$rA),
                   "stmw $rS, $disp($rA)">;
def STB  : DForm_3<38, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stb $rS, $disp($rA)">;
def STH  : DForm_3<44, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "sth $rS, $disp($rA)">;
def STW  : DForm_3<36, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stw $rS, $disp($rA)">;
def STWU : DForm_3<37, (ops GPRC:$rS, s16imm:$disp, GPRC:$rA),
                   "stwu $rS, $disp($rA)">;
}
def ANDIo : DForm_4<28, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "andi. $dst, $src1, $src2">, isDOT;
def ANDISo : DForm_4<29, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "andis. $dst, $src1, $src2">, isDOT;
def ORI   : DForm_4<24, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "ori $dst, $src1, $src2">;
def ORIS  : DForm_4<25, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "oris $dst, $src1, $src2">;
def XORI  : DForm_4<26, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "xori $dst, $src1, $src2">;
def XORIS : DForm_4<27, (ops GPRC:$dst, GPRC:$src1, u16imm:$src2),
                    "xoris $dst, $src1, $src2">;
def NOP   : DForm_4_zero<24, (ops), "nop">;
def CMPI  : DForm_5<11, (ops CRRC:$crD, i1imm:$L, GPRC:$rA, s16imm:$imm),
                    "cmpi $crD, $L, $rA, $imm">;
def CMPWI : DForm_5_ext<11, (ops CRRC:$crD, GPRC:$rA, s16imm:$imm),
                        "cmpwi $crD, $rA, $imm">;
def CMPDI : DForm_5_ext<11, (ops CRRC:$crD, GPRC:$rA, s16imm:$imm),
                        "cmpdi $crD, $rA, $imm">, isPPC64;
def CMPLI  : DForm_6<10, (ops CRRC:$dst, i1imm:$size, GPRC:$src1, u16imm:$src2),
                     "cmpli $dst, $size, $src1, $src2">;
def CMPLWI : DForm_6_ext<10, (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmplwi $dst, $src1, $src2">;
def CMPLDI : DForm_6_ext<10, (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2">, isPPC64;
let isLoad = 1 in {
def LFS : DForm_8<48, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lfs $rD, $disp($rA)">;
def LFD : DForm_8<50, (ops GPRC:$rD, symbolLo:$disp, GPRC:$rA),
                  "lfd $rD, $disp($rA)">;
}
let isStore = 1 in {
def STFS : DForm_9<52, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stfs $rS, $disp($rA)">;
def STFD : DForm_9<54, (ops GPRC:$rS, symbolLo:$disp, GPRC:$rA),
                   "stfd $rS, $disp($rA)">;
}

// DS-Form instructions.  Load/Store instructions available in PPC-64
//
let isLoad = 1 in {
def LWA  : DSForm_1<58, 2, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "lwa $rT, $DS($rA)">, isPPC64;
def LD   : DSForm_2<58, 0, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "ld $rT, $DS($rA)">, isPPC64;
}
let isStore = 1 in {
def STD  : DSForm_2<62, 0, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "std $rT, $DS($rA)">, isPPC64;
def STDU : DSForm_2<62, 1, (ops GPRC:$rT, s16imm:$DS, GPRC:$rA),
                    "stdu $rT, $DS($rA)">, isPPC64;
}

// X-Form instructions.  Most instructions that perform an operation on a
// register and another register are of this type.
//
let isLoad = 1 in {
def LBZX : XForm_1<31,  87, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lbzx $dst, $base, $index">;
def LHAX : XForm_1<31, 343, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lhax $dst, $base, $index">;
def LHZX : XForm_1<31, 279, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lhzx $dst, $base, $index">;
def LWAX : XForm_1<31, 341, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lwax $dst, $base, $index">, isPPC64;
def LWZX : XForm_1<31,  23, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "lwzx $dst, $base, $index">;
def LDX  : XForm_1<31,  21, (ops GPRC:$dst, GPRC:$base, GPRC:$index),
                   "ldx $dst, $base, $index">, isPPC64;
}
def AND  : XForm_6<31,  28, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "and $rA, $rS, $rB">;
def ANDo : XForm_6<31,  28, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "and. $rA, $rS, $rB">, isDOT;
def ANDC : XForm_6<31,  60, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "andc $rA, $rS, $rB">;
def EQV  : XForm_6<31, 284, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "eqv $rA, $rS, $rB">;
def NAND : XForm_6<31, 476, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "nand $rA, $rS, $rB">;
def NOR  : XForm_6<31, 124, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "nor $rA, $rS, $rB">;
def OR   : XForm_6<31, 444, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "or $rA, $rS, $rB">;
def ORo  : XForm_6<31, 444, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "or. $rA, $rS, $rB">, isDOT;
def ORC  : XForm_6<31, 412, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "orc $rA, $rS, $rB">;
def SLD  : XForm_6<31,  27, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "sld $rA, $rS, $rB">, isPPC64;
def SLW  : XForm_6<31,  24, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "slw $rA, $rS, $rB">;
def SRD  : XForm_6<31, 539, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srd $rA, $rS, $rB">, isPPC64;
def SRW  : XForm_6<31, 536, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srw $rA, $rS, $rB">;
def SRAD : XForm_6<31, 794, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "srad $rA, $rS, $rB">, isPPC64;
def SRAW : XForm_6<31, 792, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "sraw $rA, $rS, $rB">;
def XOR  : XForm_6<31, 316, (ops GPRC:$rA, GPRC:$rS, GPRC:$rB),
                   "xor $rA, $rS, $rB">;
let isStore = 1 in {
def STBX  : XForm_8<31, 215, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stbx $rS, $rA, $rB">;
def STHX  : XForm_8<31, 407, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "sthx $rS, $rA, $rB">;
def STWX  : XForm_8<31, 151, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stwx $rS, $rA, $rB">;
def STWUX : XForm_8<31, 183, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stwux $rS, $rA, $rB">;
def STDX  : XForm_8<31, 149, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stdx $rS, $rA, $rB">, isPPC64;
def STDUX : XForm_8<31, 181, (ops GPRC:$rS, GPRC:$rA, GPRC:$rB),
                   "stdux $rS, $rA, $rB">, isPPC64;
}
def SRAWI : XForm_10<31, 824, (ops GPRC:$rA, GPRC:$rS, u5imm:$SH), 
                     "srawi $rA, $rS, $SH">;
def CNTLZW : XForm_11<31,  26, (ops GPRC:$rA, GPRC:$rS),
                      "cntlzw $rA, $rS">;
def EXTSB  : XForm_11<31, 954, (ops GPRC:$rA, GPRC:$rS),
                      "extsb $rA, $rS">;
def EXTSH  : XForm_11<31, 922, (ops GPRC:$rA, GPRC:$rS),
                      "extsh $rA, $rS">;
def EXTSW  : XForm_11<31, 986, (ops GPRC:$rA, GPRC:$rS),
                      "extsw $rA, $rS">, isPPC64;
def CMP    : XForm_16<31, 0, (ops CRRC:$crD, i1imm:$long, GPRC:$rA, GPRC:$rB),
                      "cmp $crD, $long, $rA, $rB">;
def CMPL   : XForm_16<31, 32, (ops CRRC:$crD, i1imm:$long, GPRC:$rA, GPRC:$rB),
                      "cmpl $crD, $long, $rA, $rB">;
def CMPW   : XForm_16_ext<31, 0, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpw $crD, $rA, $rB">;
def CMPD   : XForm_16_ext<31, 0, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpd $crD, $rA, $rB">, isPPC64;
def CMPLW  : XForm_16_ext<31, 32, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmplw $crD, $rA, $rB">;
def CMPLD  : XForm_16_ext<31, 32, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpld $crD, $rA, $rB">, isPPC64;
def FCMPO  : XForm_17<63, 32, (ops CRRC:$crD, FPRC:$fA, FPRC:$fB),
                      "fcmpo $crD, $fA, $fB">;
def FCMPU  : XForm_17<63, 0, (ops CRRC:$crD, FPRC:$fA, FPRC:$fB),
                      "fcmpu $crD, $fA, $fB">;
let isLoad = 1 in {
def LFSX   : XForm_25<31, 535, (ops FPRC:$dst, GPRC:$base, GPRC:$index),
                      "lfsx $dst, $base, $index">;
def LFDX   : XForm_25<31, 599, (ops FPRC:$dst, GPRC:$base, GPRC:$index),
                      "lfdx $dst, $base, $index">;
}
def FCFID  : XForm_26<63, 846, (ops FPRC:$frD, FPRC:$frB),
                      "fcfid $frD, $frB">, isPPC64;
def FCTIDZ : XForm_26<63, 815, (ops FPRC:$frD, FPRC:$frB),
                      "fctidz $frD, $frB">, isPPC64;
def FCTIWZ : XForm_26<63, 15, (ops FPRC:$frD, FPRC:$frB),
                      "fctiwz $frD, $frB">;
def FABS   : XForm_26<63, 264, (ops FPRC:$frD, FPRC:$frB),
                      "fabs $frD, $frB">;
def FMR    : XForm_26<63, 72, (ops FPRC:$frD, FPRC:$frB),
                      "fmr $frD, $frB">;
def FNABS  : XForm_26<63, 136, (ops FPRC:$frD, FPRC:$frB),
                      "fnabs $frD, $frB">;
def FNEG   : XForm_26<63, 40, (ops FPRC:$frD, FPRC:$frB),
                      "fneg $frD, $frB">;
def FRSP   : XForm_26<63, 12, (ops FPRC:$frD, FPRC:$frB),
                      "frsp $frD, $frB">;
def FSQRT  : XForm_26<63, 22, (ops FPRC:$frD, FPRC:$frB),
                      "fsqrt $frD, $frB">;
def FSQRTS : XForm_26<59, 22, (ops FPRC:$frD, FPRC:$frB),
                      "fsqrts $frD, $frB">;
                      
let isStore = 1 in {
def STFSX : XForm_28<31, 663, (ops FPRC:$frS, GPRC:$rA, GPRC:$rB),
                     "stfsx $frS, $rA, $rB">;
def STFDX : XForm_28<31, 727, (ops FPRC:$frS, GPRC:$rA, GPRC:$rB),
                     "stfdx $frS, $rA, $rB">;
}

// XL-Form instructions.  condition register logical ops.
//
def CRAND  : XLForm_1<19, 257, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "crand $Db, $Ab, $Bb">;
def CRANDC : XLForm_1<19, 129, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "crandc $Db, $Ab, $Bb">;
def CREQV  : XLForm_1<19, 289, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "creqv $Db, $Ab, $Bb">;
def CRNAND : XLForm_1<19, 225, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "crnand $Db, $Ab, $Bb">;
def CRNOR  : XLForm_1<19, 33, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "crnor $Db, $Ab, $Bb">;
def CROR   : XLForm_1<19, 449, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "cror $Db, $Ab, $Bb">;
def CRORC  : XLForm_1<19, 417, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "crorc $Db, $Ab, $Bb">;
def CRXOR  : XLForm_1<19, 193, (ops CRRC:$D, crbit:$Db,
                      CRRC:$A, crbit:$Ab, CRRC:$B, crbit:$Bb),
                      "crxor $Db, $Ab, $Bb">;
def MCRF   : XLForm_3<19, 0, (ops CRRC:$BF, CRRC:$BFA),
                      "mcrf $BF, $BFA">;

// XFX-Form instructions.  Instructions that deal with SPRs
//
// Note that although LR should be listed as `8' and CTR as `9' in the SPR
// field, the manual lists the groups of bits as [5-9] = 0, [0-4] = 8 or 9
// which means the SPR value needs to be multiplied by a factor of 32.
def MFCTR : XFXForm_1_ext<31, 339, 288, (ops GPRC:$rT), "mfctr $rT">;
def MFLR  : XFXForm_1_ext<31, 339, 256, (ops GPRC:$rT), "mflr $rT">;
def MFCR  : XFXForm_3<31, 19, (ops GPRC:$rT), "mfcr $rT">;
def MTCRF : XFXForm_5<31, 144, (ops CRRC:$FXM, GPRC:$rS),
                      "mtcrf $FXM, $rS">;
def MFOCRF : XFXForm_5a<31, 19, (ops GPRC:$rT, crbitm:$FXM),
                        "mfcr $rT, $FXM">;
def MTCTR : XFXForm_7_ext<31, 467, 288, (ops GPRC:$rS), "mtctr $rS">;
def MTLR  : XFXForm_7_ext<31, 467, 256, (ops GPRC:$rS), "mtlr $rS">;

// XS-Form instructions.  Just 'sradi'
//
def SRADI  : XSForm_1<31, 413, (ops GPRC:$rA, GPRC:$rS, u6imm:$SH),
                      "sradi $rA, $rS, $SH">, isPPC64;

// XO-Form instructions.  Arithmetic instructions that can set overflow bit
//
def ADD   : XOForm_1<31, 266, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "add $rT, $rA, $rB">;
def ADDC  : XOForm_1<31, 10, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "addc $rT, $rA, $rB">;
def ADDE  : XOForm_1<31, 138, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "adde $rT, $rA, $rB">;
def DIVD  : XOForm_1<31, 489, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divd $rT, $rA, $rB">, isPPC64;
def DIVDU : XOForm_1<31, 457, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divdu $rT, $rA, $rB">, isPPC64;
def DIVW  : XOForm_1<31, 491, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divw $rT, $rA, $rB">;
def DIVWU : XOForm_1<31, 459, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "divwu $rT, $rA, $rB">;
def MULHW : XOForm_1<31, 75, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mulhw $rT, $rA, $rB">;
def MULHWU : XOForm_1<31, 11, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mulhwu $rT, $rA, $rB">;
def MULLD : XOForm_1<31, 233, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mulld $rT, $rA, $rB">, isPPC64;
def MULLW : XOForm_1<31, 235, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "mullw $rT, $rA, $rB">;
def SUBF  : XOForm_1<31, 40, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "subf $rT, $rA, $rB">;
def SUBFC : XOForm_1<31, 8, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "subfc $rT, $rA, $rB">;
def SUBFE : XOForm_1<31, 136, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "subfe $rT, $rA, $rB">;
def SUB  : XOForm_1r<31, 40, 0, (ops GPRC:$rT, GPRC:$rA, GPRC:$rB),
                     "sub $rT, $rA, $rB">;
def ADDME  : XOForm_3<31, 234, 0, (ops GPRC:$rT, GPRC:$rA),
                      "addme $rT, $rA">;
def ADDZE  : XOForm_3<31, 202, 0, (ops GPRC:$rT, GPRC:$rA),
                      "addze $rT, $rA">;
def NEG    : XOForm_3<31, 104, 0, (ops GPRC:$rT, GPRC:$rA),
                      "neg $rT, $rA">;
def SUBFZE : XOForm_3<31, 200, 0, (ops GPRC:$rT, GPRC:$rA),
                      "subfze $rT, $rA">;

// A-Form instructions.  Most of the instructions executed in the FPU are of
// this type.
//
def FMADD : AForm_1<63, 29, 
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmadd $FRT, $FRA, $FRC, $FRB">;
def FMADDS : AForm_1<59, 29,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmadds $FRT, $FRA, $FRC, $FRB">;
def FMSUB : AForm_1<63, 28,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmsub $FRT, $FRA, $FRC, $FRB">;
def FMSUBS : AForm_1<59, 28,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fmsubs $FRT, $FRA, $FRC, $FRB">;
def FNMADD : AForm_1<63, 31,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmadd $FRT, $FRA, $FRC, $FRB">;
def FNMADDS : AForm_1<59, 31,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmadds $FRT, $FRA, $FRC, $FRB">;
def FNMSUB : AForm_1<63, 30,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmsub $FRT, $FRA, $FRC, $FRB">;
def FNMSUBS : AForm_1<59, 30,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fnmsubs $FRT, $FRA, $FRC, $FRB">;
def FSEL  : AForm_1<63, 23,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRC, FPRC:$FRB),
                    "fsel $FRT, $FRA, $FRC, $FRB">;
def FADD  : AForm_2<63, 21,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fadd $FRT, $FRA, $FRB">;
def FADDS : AForm_2<59, 21,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fadds $FRT, $FRA, $FRB">;
def FDIV  : AForm_2<63, 18,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fdiv $FRT, $FRA, $FRB">;
def FDIVS : AForm_2<59, 18,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fdivs $FRT, $FRA, $FRB">;
def FMUL  : AForm_3<63, 25,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fmul $FRT, $FRA, $FRB">;
def FMULS : AForm_3<59, 25,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fmuls $FRT, $FRA, $FRB">;
def FSUB  : AForm_2<63, 20,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fsub $FRT, $FRA, $FRB">;
def FSUBS : AForm_2<59, 20,
                    (ops FPRC:$FRT, FPRC:$FRA, FPRC:$FRB),
                    "fsubs $FRT, $FRA, $FRB">;

// M-Form instructions.  rotate and mask instructions.
//
let isTwoAddress = 1 in {
def RLWIMI : MForm_2<20,
                     (ops GPRC:$rA, GPRC:$rSi, GPRC:$rS, u5imm:$SH, u5imm:$MB, 
                      u5imm:$ME), "rlwimi $rA, $rS, $SH, $MB, $ME">;
}
def RLWINM : MForm_2<21,
                     (ops GPRC:$rA, GPRC:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                     "rlwinm $rA, $rS, $SH, $MB, $ME">;
def RLWINMo : MForm_2<21,
                     (ops GPRC:$rA, GPRC:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                     "rlwinm. $rA, $rS, $SH, $MB, $ME">, isDOT;
def RLWNM  : MForm_2<23,
                     (ops GPRC:$rA, GPRC:$rS, GPRC:$rB, u5imm:$MB, u5imm:$ME),
                     "rlwnm $rA, $rS, $rB, $MB, $ME">;

// MD-Form instructions.  64 bit rotate instructions.
//
def RLDICL : MDForm_1<30, 0,
                      (ops GPRC:$rA, GPRC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldicl $rA, $rS, $SH, $MB">, isPPC64;
def RLDICR : MDForm_1<30, 1,
                      (ops GPRC:$rA, GPRC:$rS, u6imm:$SH, u6imm:$ME),
                      "rldicr $rA, $rS, $SH, $ME">, isPPC64;

def PowerPCInstrInfo : InstrInfo {
  let PHIInst  = PHI;

  let TSFlagsFields = [ "VMX", "PPC64" ];
  let TSFlagsShifts = [ 0, 1 ];

  let isLittleEndianEncoding = 1;
}

