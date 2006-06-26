//===- PPCInstr64Bit.td - The PowerPC 64-bit Support -------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the PowerPC 64-bit instructions.  These patterns are used
// both when in ppc64 mode and when in "use 64-bit extensions in 32-bit" mode.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 64-bit operands.
//
def s16imm64 : Operand<i64> {
  let PrintMethod = "printS16ImmOperand";
}
def u16imm64 : Operand<i64> {
  let PrintMethod = "printU16ImmOperand";
}
def symbolHi64 : Operand<i64> {
  let PrintMethod = "printSymbolHi";
}
def symbolLo64 : Operand<i64> {
  let PrintMethod = "printSymbolLo";
}

//===----------------------------------------------------------------------===//
// 64-bit transformation functions.
//

def SHL64 : SDNodeXForm<imm, [{
  // Transformation function: 63 - imm
  return getI32Imm(63 - N->getValue());
}]>;

def SRL64 : SDNodeXForm<imm, [{
  // Transformation function: 64 - imm
  return N->getValue() ? getI32Imm(64 - N->getValue()) : getI32Imm(0);
}]>;

def HI32_48 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)(N->getValue() >> 32));
}]>;

def HI48_64 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)(N->getValue() >> 48));
}]>;


//===----------------------------------------------------------------------===//
// Fixed point instructions.
//

let PPC970_Unit = 1 in {  // FXU Operations.

// Copies, extends, truncates.
def OR4To8  : XForm_6<31, 444, (ops G8RC:$rA, GPRC:$rS, GPRC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   []>;
def OR8To4  : XForm_6<31, 444, (ops GPRC:$rA, G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   []>;

def LI8  : DForm_2_r0<14, (ops G8RC:$rD, symbolLo64:$imm),
                      "li $rD, $imm", IntGeneral,
                      [(set G8RC:$rD, immSExt16:$imm)]>;
def LIS8 : DForm_2_r0<15, (ops G8RC:$rD, symbolHi64:$imm),
                      "lis $rD, $imm", IntGeneral,
                      [(set G8RC:$rD, imm16ShiftedSExt:$imm)]>;

// Logical ops.
def NAND8: XForm_6<31, 476, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "nand $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (not (and G8RC:$rS, G8RC:$rB)))]>;
def AND8 : XForm_6<31,  28, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "and $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (and G8RC:$rS, G8RC:$rB))]>;
def ANDC8: XForm_6<31,  60, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "andc $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (and G8RC:$rS, (not G8RC:$rB)))]>;
def OR8  : XForm_6<31, 444, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (or G8RC:$rS, G8RC:$rB))]>;
def NOR8 : XForm_6<31, 124, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "nor $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (not (or G8RC:$rS, G8RC:$rB)))]>;
def ORC8 : XForm_6<31, 412, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "orc $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (or G8RC:$rS, (not G8RC:$rB)))]>;
def EQV8 : XForm_6<31, 284, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "eqv $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (not (xor G8RC:$rS, G8RC:$rB)))]>;
def XOR8 : XForm_6<31, 316, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "xor $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (xor G8RC:$rS, G8RC:$rB))]>;

// Logical ops with immediate.
def ANDIo8  : DForm_4<28, (ops G8RC:$dst, G8RC:$src1, u16imm:$src2),
                      "andi. $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (and G8RC:$src1, immZExt16:$src2))]>,
                      isDOT;
def ANDISo8 : DForm_4<29, (ops G8RC:$dst, G8RC:$src1, u16imm:$src2),
                     "andis. $dst, $src1, $src2", IntGeneral,
                    [(set G8RC:$dst, (and G8RC:$src1,imm16ShiftedZExt:$src2))]>,
                     isDOT;
def ORI8    : DForm_4<24, (ops G8RC:$dst, G8RC:$src1, u16imm:$src2),
                      "ori $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (or G8RC:$src1, immZExt16:$src2))]>;
def ORIS8   : DForm_4<25, (ops G8RC:$dst, G8RC:$src1, u16imm:$src2),
                      "oris $dst, $src1, $src2", IntGeneral,
                    [(set G8RC:$dst, (or G8RC:$src1, imm16ShiftedZExt:$src2))]>;
def XORI8   : DForm_4<26, (ops G8RC:$dst, G8RC:$src1, u16imm:$src2),
                      "xori $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (xor G8RC:$src1, immZExt16:$src2))]>;
def XORIS8  : DForm_4<27, (ops G8RC:$dst, G8RC:$src1, u16imm:$src2),
                      "xoris $dst, $src1, $src2", IntGeneral,
                   [(set G8RC:$dst, (xor G8RC:$src1, imm16ShiftedZExt:$src2))]>;


                   
def ADD8  : XOForm_1<31, 266, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "add $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (add G8RC:$rA, G8RC:$rB))]>;
def ADDI8  : DForm_2<14, (ops G8RC:$rD, G8RC:$rA, s16imm64:$imm),
                     "addi $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (add G8RC:$rA, immSExt16:$imm))]>;
def ADDIS8 : DForm_2<15, (ops G8RC:$rD, G8RC:$rA, symbolHi64:$imm),
                     "addis $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (add G8RC:$rA, imm16ShiftedSExt:$imm))]>;




def MULHD : XOForm_1<31, 73, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "mulhd $rT, $rA, $rB", IntMulHW,
                     [(set G8RC:$rT, (mulhs G8RC:$rA, G8RC:$rB))]>;
def MULHDU : XOForm_1<31, 9, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "mulhdu $rT, $rA, $rB", IntMulHWU,
                     [(set G8RC:$rT, (mulhu G8RC:$rA, G8RC:$rB))]>;

def CMPD   : XForm_16_ext<31, 0, (ops CRRC:$crD, G8RC:$rA, G8RC:$rB),
                          "cmpd $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPLD  : XForm_16_ext<31, 32, (ops CRRC:$crD, G8RC:$rA, G8RC:$rB),
                          "cmpld $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPDI  : DForm_5_ext<11, (ops CRRC:$crD, G8RC:$rA, s16imm:$imm),
                         "cmpdi $crD, $rA, $imm", IntCompare>, isPPC64;
def CMPLDI : DForm_6_ext<10, (ops CRRC:$dst, G8RC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2", IntCompare>, isPPC64;

def SLD  : XForm_6<31,  27, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "sld $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (shl G8RC:$rS, G8RC:$rB))]>, isPPC64;
def SRD  : XForm_6<31, 539, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "srd $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (srl G8RC:$rS, G8RC:$rB))]>, isPPC64;
def SRAD : XForm_6<31, 794, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "srad $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (sra G8RC:$rS, G8RC:$rB))]>, isPPC64;
def EXTSW  : XForm_11<31, 986, (ops G8RC:$rA, G8RC:$rS),
                      "extsw $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i32))]>, isPPC64;
/// EXTSW_32 - Just like EXTSW, but works on '32-bit' registers.
def EXTSW_32 : XForm_11<31, 986, (ops GPRC:$rA, GPRC:$rS),
                      "extsw $rA, $rS", IntGeneral,
                      [(set GPRC:$rA, (PPCextsw_32 GPRC:$rS))]>, isPPC64;
def EXTSW_32_64 : XForm_11<31, 986, (ops G8RC:$rA, GPRC:$rS),
                      "extsw $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (sext GPRC:$rS))]>, isPPC64;

def SRADI  : XSForm_1<31, 413, (ops GPRC:$rA, GPRC:$rS, u6imm:$SH),
                      "sradi $rA, $rS, $SH", IntRotateD>, isPPC64;
def DIVD  : XOForm_1<31, 489, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "divd $rT, $rA, $rB", IntDivD,
                     [(set G8RC:$rT, (sdiv G8RC:$rA, G8RC:$rB))]>, isPPC64,
                     PPC970_DGroup_First, PPC970_DGroup_Cracked;
def DIVDU : XOForm_1<31, 457, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "divdu $rT, $rA, $rB", IntDivD,
                     [(set G8RC:$rT, (udiv G8RC:$rA, G8RC:$rB))]>, isPPC64,
                     PPC970_DGroup_First, PPC970_DGroup_Cracked;
def MULLD : XOForm_1<31, 233, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "mulld $rT, $rA, $rB", IntMulHD,
                     [(set G8RC:$rT, (mul G8RC:$rA, G8RC:$rB))]>, isPPC64;


let isTwoAddress = 1, isCommutable = 1 in {
def RLDIMI : MDForm_1<30, 3,
                      (ops G8RC:$rA, G8RC:$rSi, G8RC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldimi $rA, $rS, $SH, $MB", IntRotateD,
                      []>, isPPC64;
}

// Rotate instructions.
def RLDICL : MDForm_1<30, 0,
                      (ops G8RC:$rA, G8RC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldicl $rA, $rS, $SH, $MB", IntRotateD,
                      []>, isPPC64;
def RLDICR : MDForm_1<30, 1,
                      (ops G8RC:$rA, G8RC:$rS, u6imm:$SH, u6imm:$ME),
                      "rldicr $rA, $rS, $SH, $ME", IntRotateD,
                      []>, isPPC64;
}  // End FXU Operations.


//===----------------------------------------------------------------------===//
// Load/Store instructions.
//


let isLoad = 1, PPC970_Unit = 2 in {
def LWA  : DSForm_1<58, 2, (ops G8RC:$rD, memrix:$src),
                    "lwa $rD, $src", LdStLWA,
                    [(set G8RC:$rD, (sextload ixaddr:$src, i32))]>, isPPC64,
                    PPC970_DGroup_Cracked;
def LD   : DSForm_2<58, 0, (ops G8RC:$rD, memrix:$src),
                    "ld $rD, $src", LdStLD,
                    [(set G8RC:$rD, (load ixaddr:$src))]>, isPPC64;

def LWAX : XForm_1<31, 341, (ops G8RC:$rD, memrr:$src),
                   "lwax $rD, $src", LdStLHA,
                   [(set G8RC:$rD, (sextload xaddr:$src, i32))]>, isPPC64,
                   PPC970_DGroup_Cracked;
def LDX  : XForm_1<31,  21, (ops G8RC:$rD, memrr:$src),
                   "ldx $rD, $src", LdStLD,
                   [(set G8RC:$rD, (load xaddr:$src))]>, isPPC64;
}
let isStore = 1, noResults = 1, PPC970_Unit = 2 in {
def STD  : DSForm_2<62, 0, (ops G8RC:$rS, memrix:$dst),
                    "std $rS, $dst", LdStSTD,
                    [(store G8RC:$rS, ixaddr:$dst)]>, isPPC64;
def STDX  : XForm_8<31, 149, (ops G8RC:$rS, memrr:$dst),
                   "stdx $rS, $dst", LdStSTD,
                   [(store G8RC:$rS, iaddr:$dst)]>, isPPC64,
                   PPC970_DGroup_Cracked;
def STDUX : XForm_8<31, 181, (ops G8RC:$rS, memrr:$dst),
                   "stdux $rS, $dst", LdStSTD,
                   []>, isPPC64;
                   
// STD_32/STDX_32 - Just like STD/STDX, but uses a '32-bit' input register.
def STD_32  : DSForm_2<62, 0, (ops GPRC:$rT, memrix:$dst),
                       "std $rT, $dst", LdStSTD,
                       [(PPCstd_32  GPRC:$rT, ixaddr:$dst)]>, isPPC64;
def STDX_32  : XForm_8<31, 149, (ops GPRC:$rT, memrr:$dst),
                       "stdx $rT, $dst", LdStSTD,
                       [(PPCstd_32  GPRC:$rT, xaddr:$dst)]>, isPPC64,
                       PPC970_DGroup_Cracked;
}



//===----------------------------------------------------------------------===//
// Floating point instructions.
//


let PPC970_Unit = 3 in {  // FPU Operations.
def FCFID  : XForm_26<63, 846, (ops F8RC:$frD, F8RC:$frB),
                      "fcfid $frD, $frB", FPGeneral,
                      [(set F8RC:$frD, (PPCfcfid F8RC:$frB))]>, isPPC64;
def FCTIDZ : XForm_26<63, 815, (ops F8RC:$frD, F8RC:$frB),
                      "fctidz $frD, $frB", FPGeneral,
                      [(set F8RC:$frD, (PPCfctidz F8RC:$frB))]>, isPPC64;
}


//===----------------------------------------------------------------------===//
// Instruction Patterns
//

// Immediate support.
// Handled above:
//   sext(0x0000_0000_0000_FFFF,  i8) -> li imm
//   sext(0x0000_0000_FFFF_0000, i16) -> lis imm>>16

// sext(0x0000_0000_FFFF_FFFF,  i16)  -> lis + ori
def sext_0x0000_0000_FFFF_FFFF_i16 : PatLeaf<(imm), [{
  return N->getValue() == (uint64_t)(int32_t)N->getValue();
}]>;
def : Pat<(i64 sext_0x0000_0000_FFFF_FFFF_i16:$imm),
          (ORI8 (LIS8 (HI16 imm:$imm)), (LO16 imm:$imm))>;

// zext(0x0000_0000_FFFF_7FFF, i16) -> oris (li lo16(imm)), imm>>16
def zext_0x0000_0000_FFFF_7FFF_i16 : PatLeaf<(imm), [{
  return (N->getValue() & 0xFFFFFFFF00008000ULL) == 0;
}]>;
def : Pat<(i64 zext_0x0000_0000_FFFF_7FFF_i16:$imm),
          (ORIS8 (LI8 (LO16 imm:$imm)), (HI16 imm:$imm))>;

// zext(0x0000_0000_FFFF_FFFF, i16) -> oris (ori (li 0), lo16(imm)), imm>>16
def zext_0x0000_0000_FFFF_FFFF_i16 : PatLeaf<(imm), [{
  return (N->getValue() & 0xFFFFFFFF00000000ULL) == 0;
}]>;
def : Pat<(i64 zext_0x0000_0000_FFFF_FFFF_i16:$imm),
          (ORIS8 (ORI8 (LI8 0), (LO16 imm:$imm)), (HI16 imm:$imm))>;

// FIXME: Handle smart forms where the top 32-bits are set.  Right now, stuff
// like 0xABCD0123BCDE0000 hits the case below, which produces ORI R, R, 0's! 

// Fully general (and most expensive: 6 instructions!) immediate pattern.
def : Pat<(i64 imm:$imm),
          (ORI8 
             (ORIS8 
                (RLDICR
                   (ORI8
                      (LIS8 (HI48_64 imm:$imm)),
                      (HI32_48 imm:$imm)),
                   32, 31),
                (HI16 imm:$imm)),
             (LO16 imm:$imm))>;


// Extensions and truncates to/from 32-bit regs.
def : Pat<(i64 (zext GPRC:$in)),
          (RLDICL (OR4To8 GPRC:$in, GPRC:$in), 0, 32)>;
def : Pat<(i64 (anyext GPRC:$in)),
          (OR4To8 GPRC:$in, GPRC:$in)>;
def : Pat<(i32 (trunc G8RC:$in)),
          (OR8To4 G8RC:$in, G8RC:$in)>;

// SHL/SRL
def : Pat<(shl G8RC:$in, (i64 imm:$imm)),
          (RLDICR G8RC:$in, imm:$imm, (SHL64 imm:$imm))>;
def : Pat<(srl G8RC:$in, (i64 imm:$imm)),
          (RLDICL G8RC:$in, (SRL64 imm:$imm), imm:$imm)>;

// Hi and Lo for Darwin Global Addresses.
def : Pat<(PPChi tglobaladdr:$in, 0), (LIS8 tglobaladdr:$in)>;
def : Pat<(PPClo tglobaladdr:$in, 0), (LI8  tglobaladdr:$in)>;
def : Pat<(PPChi tconstpool:$in , 0), (LIS8 tconstpool:$in)>;
def : Pat<(PPClo tconstpool:$in , 0), (LI8  tconstpool:$in)>;
def : Pat<(PPChi tjumptable:$in , 0), (LIS8 tjumptable:$in)>;
def : Pat<(PPClo tjumptable:$in , 0), (LI8  tjumptable:$in)>;
def : Pat<(add G8RC:$in, (PPChi tglobaladdr:$g, 0)),
          (ADDIS8 G8RC:$in, tglobaladdr:$g)>;
def : Pat<(add G8RC:$in, (PPChi tconstpool:$g, 0)),
          (ADDIS8 G8RC:$in, tconstpool:$g)>;
def : Pat<(add G8RC:$in, (PPChi tjumptable:$g, 0)),
          (ADDIS8 G8RC:$in, tjumptable:$g)>;
