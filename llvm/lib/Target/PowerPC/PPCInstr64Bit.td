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
def symbolHi64 : Operand<i64> {
  let PrintMethod = "printSymbolHi";
}
def symbolLo64 : Operand<i64> {
  let PrintMethod = "printSymbolLo";
}




//===----------------------------------------------------------------------===//
// Fixed point instructions.
//

let PPC970_Unit = 1 in {  // FXU Operations.

def LI8  : DForm_2_r0<14, (ops G8RC:$rD, symbolLo64:$imm),
                      "li $rD, $imm", IntGeneral,
                      [(set G8RC:$rD, immSExt16:$imm)]>;
def LIS8 : DForm_2_r0<15, (ops G8RC:$rD, symbolHi64:$imm),
                      "lis $rD, $imm", IntGeneral,
                      [(set G8RC:$rD, imm16Shifted:$imm)]>;

def OR8  : XForm_6<31, 444, (ops G8RC:$rA, G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (or G8RC:$rS, G8RC:$rB))]>;
def OR4To8  : XForm_6<31, 444, (ops G8RC:$rA, GPRC:$rS, GPRC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   []>;
def OR8To4  : XForm_6<31, 444, (ops GPRC:$rA, G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   []>;
                   
def ADD8  : XOForm_1<31, 266, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "add $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (add G8RC:$rA, G8RC:$rB))]>;
def ADDIS8 : DForm_2<15, (ops G8RC:$rD, G8RC:$rA, symbolHi64:$imm),
                     "addis $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (add G8RC:$rA, imm16Shifted:$imm))]>;

def MULHD : XOForm_1<31, 73, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "mulhd $rT, $rA, $rB", IntMulHW,
                     [(set G8RC:$rT, (mulhs G8RC:$rA, G8RC:$rB))]>;
def MULHDU : XOForm_1<31, 9, 0, (ops G8RC:$rT, G8RC:$rA, G8RC:$rB),
                     "mulhdu $rT, $rA, $rB", IntMulHWU,
                     [(set G8RC:$rT, (mulhu G8RC:$rA, G8RC:$rB))]>;

def CMPDI : DForm_5_ext<11, (ops CRRC:$crD, GPRC:$rA, s16imm:$imm),
                        "cmpdi $crD, $rA, $imm", IntCompare>, isPPC64;

def CMPLDI : DForm_6_ext<10, (ops CRRC:$dst, GPRC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2", IntCompare>, isPPC64;
def CMPD   : XForm_16_ext<31, 0, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpd $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPLD  : XForm_16_ext<31, 32, (ops CRRC:$crD, GPRC:$rA, GPRC:$rB),
                          "cmpld $crD, $rA, $rB", IntCompare>, isPPC64;

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
}


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
