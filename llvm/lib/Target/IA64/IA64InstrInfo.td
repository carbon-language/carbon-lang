//===- IA64InstrInfo.td - Describe the IA64 Instruction Set -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the IA64 instruction set, defining the instructions, and
// properties of the instructions which are needed for code generation, machine
// code emission, and analysis.
//
//===----------------------------------------------------------------------===//

include "IA64InstrFormats.td"

//===----------------------------------------------------------------------===//
// IA-64 specific DAG Nodes.
//

def IA64getfd : SDNode<"IA64ISD::GETFD", SDTFPToIntOp, []>;

//===---------

def u2imm : Operand<i8>;
def u6imm : Operand<i8>;
def s8imm : Operand<i8> {
  let PrintMethod = "printS8ImmOperand";
}
def s14imm  : Operand<i64> {
  let PrintMethod = "printS14ImmOperand";
}
def s22imm  : Operand<i64> {
  let PrintMethod = "printS22ImmOperand";
}
def u64imm  : Operand<i64> {
  let PrintMethod = "printU64ImmOperand";
}
def s64imm  : Operand<i64> {
  let PrintMethod = "printS64ImmOperand";
}

let PrintMethod = "printGlobalOperand" in
  def globaladdress : Operand<i64>;

// the asmprinter needs to know about calls
let PrintMethod = "printCallOperand" in
  def calltarget : Operand<i64>;
  
/* new daggy action!!! */

def is32ones : PatLeaf<(i64 imm), [{
  // is32ones predicate - True if the immediate is 0x00000000FFFFFFFF 
  // Used to create ZXT4s appropriately 
  uint64_t v = (uint64_t)N->getValue();
  return (v == 0x00000000FFFFFFFFLL);
}]>;

// isMIXable predicates - True if the immediate is
// 0xFF00FF00FF00FF00, 0x00FF00FF00FF00FF
// etc, through 0x00000000FFFFFFFF
// Used to test for the suitability of mix* 
def isMIX1Lable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getValue()==0xFF00FF00FF00FF00LL);
}]>;
def isMIX1Rable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getValue()==0x00FF00FF00FF00FFLL);
}]>;
def isMIX2Lable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getValue()==0xFFFF0000FFFF0000LL);
}]>;
def isMIX2Rable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getValue()==0x0000FFFF0000FFFFLL);
}]>;
def isMIX4Lable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getValue()==0xFFFFFFFF00000000LL);
}]>;
def isMIX4Rable: PatLeaf<(i64 imm), [{
  return((uint64_t)N->getValue()==0x00000000FFFFFFFFLL);
}]>;

def isSHLADDimm: PatLeaf<(i64 imm), [{
  // isSHLADDimm predicate - True if the immediate is exactly 1, 2, 3 or 4
  // - 0 is *not* okay.
  // Used to create shladd instructions appropriately
  int64_t v = (int64_t)N->getValue();
  return (v >= 1 && v <= 4);
}]>;

def immSExt14  : PatLeaf<(i64 imm), [{
  // immSExt14 predicate - True if the immediate fits in a 14-bit sign extended
  // field.  Used by instructions like 'adds'.
  int64_t v = (int64_t)N->getValue();
  return (v <= 8191 && v >= -8192);
}]>;

def imm64  : PatLeaf<(i64 imm), [{
  // imm64 predicate - True if the immediate fits in a 64-bit 
  // field - i.e., true. used to keep movl happy
  return true;
}]>;

def ADD  : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
           "add $dst = $src1, $src2;;",
	   [(set GR:$dst, (add GR:$src1, GR:$src2))]>;

def ADD1 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
           "add $dst = $src1, $src2, 1;;",
	   [(set GR:$dst, (add (add GR:$src1, GR:$src2), 1))]>;

def ADDS : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, s14imm:$imm),
           "adds $dst = $imm, $src1;;",
	   [(set GR:$dst, (add GR:$src1, immSExt14:$imm))]>;
 
def PADDS: AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, s14imm:$imm, PR:$qp),
           "($qp) adds $dst = $imm, $src1;;",
	   []>;

def MOVL : AForm_DAG<0x03, 0x0b, (ops GR:$dst, s64imm:$imm),
           "movl $dst = $imm;;",
	   [(set GR:$dst, imm64:$imm)]>;

def ADDL_GA : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, globaladdress:$imm),
           "addl $dst = $imm, $src1;;",
	   []>;

// hmm 
def ADDL_EA : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, calltarget:$imm),
           "addl $dst = $imm, $src1;;",
	   []>;
 
def SUB  : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
           "sub $dst = $src1, $src2;;",
	   [(set GR:$dst, (sub GR:$src1, GR:$src2))]>;

def SUB1 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
           "sub $dst = $src1, $src2, 1;;",
	   [(set GR:$dst, (add (sub GR: $src1, GR:$src2), -1))]>;

let isTwoAddress = 1 in {
def TPCADDIMM22 : AForm<0x03, 0x0b,
  (ops GR:$dst, GR:$src1, s22imm:$imm, PR:$qp),
    "($qp) add $dst = $imm, $dst;;">;
def TPCMPIMM8NE : AForm<0x03, 0x0b,
  (ops PR:$dst, PR:$src1, s22imm:$imm, GR:$src2, PR:$qp),
    "($qp) cmp.ne $dst , p0 = $imm, $src2;;">;
}

// zero extend a bool (predicate reg) into an integer reg
def ZXTb : Pat<(zext PR:$src),
          (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src)>;

// normal sign/zero-extends
def SXT1 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src), "sxt1 $dst = $src;;",
           [(set GR:$dst, (sext_inreg GR:$src, i8))]>;
def ZXT1 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src), "zxt1 $dst = $src;;",
           [(set GR:$dst, (and GR:$src, 255))]>;
def SXT2 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src), "sxt2 $dst = $src;;",
           [(set GR:$dst, (sext_inreg GR:$src, i16))]>;
def ZXT2 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src), "zxt2 $dst = $src;;",
           [(set GR:$dst, (and GR:$src, 65535))]>;
def SXT4 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src), "sxt4 $dst = $src;;",
           [(set GR:$dst, (sext_inreg GR:$src, i32))]>;
def ZXT4 : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src), "zxt4 $dst = $src;;",
           [(set GR:$dst, (and GR:$src, is32ones))]>;

// fixme: shrs vs shru?
def MIX1L : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "mix1.l $dst = $src1, $src2;;",
	  [(set GR:$dst, (or (and GR:$src1, isMIX1Lable),
	                     (and (srl GR:$src2, (i64 8)), isMIX1Lable)))]>;

def MIX2L : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "mix2.l $dst = $src1, $src2;;",
	  [(set GR:$dst, (or (and GR:$src1, isMIX2Lable),
	                     (and (srl GR:$src2, (i64 16)), isMIX2Lable)))]>;

def MIX4L : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "mix4.l $dst = $src1, $src2;;",
	  [(set GR:$dst, (or (and GR:$src1, isMIX4Lable),
	                     (and (srl GR:$src2, (i64 32)), isMIX4Lable)))]>;

def MIX1R : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "mix1.r $dst = $src1, $src2;;",
	  [(set GR:$dst, (or (and (shl GR:$src1, (i64 8)), isMIX1Rable),
	                     (and GR:$src2, isMIX1Rable)))]>;

def MIX2R : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "mix2.r $dst = $src1, $src2;;",
	  [(set GR:$dst, (or (and (shl GR:$src1, (i64 16)), isMIX2Rable),
	                     (and GR:$src2, isMIX2Rable)))]>;

def MIX4R : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "mix4.r $dst = $src1, $src2;;",
	  [(set GR:$dst, (or (and (shl GR:$src1, (i64 32)), isMIX4Rable),
	                     (and GR:$src2, isMIX4Rable)))]>;

def GETFSIGD : AForm_DAG<0x03, 0x0b, (ops GR:$dst, FP:$src),
  "getf.sig $dst = $src;;",
  []>;

def SETFSIGD : AForm_DAG<0x03, 0x0b, (ops FP:$dst, GR:$src),
  "setf.sig $dst = $src;;",
  []>;

def XMALD : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "xma.l $dst = $src1, $src2, $src3;;",
  []>;
def XMAHD : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "xma.h $dst = $src1, $src2, $src3;;",
  []>;
def XMAHUD : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "xma.hu $dst = $src1, $src2, $src3;;",
  []>;

// pseudocode for integer multiplication 
def : Pat<(mul GR:$src1, GR:$src2),
           (GETFSIGD (XMALD (SETFSIGD GR:$src1), (SETFSIGD GR:$src2), F0))>;
def : Pat<(mulhs GR:$src1, GR:$src2),
           (GETFSIGD (XMAHD (SETFSIGD GR:$src1), (SETFSIGD GR:$src2), F0))>;
def : Pat<(mulhu GR:$src1, GR:$src2),
           (GETFSIGD (XMAHUD (SETFSIGD GR:$src1), (SETFSIGD GR:$src2), F0))>;

// TODO: addp4 (addp4 dst = src, r0 is a 32-bit add)
// has imm form, too

// def ADDS : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s14imm:$imm),
//   "adds $dst = $imm, $src1;;">;

def AND   : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "and $dst = $src1, $src2;;",
	  [(set GR:$dst, (and GR:$src1, GR:$src2))]>;
def ANDCM : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "andcm $dst = $src1, $src2;;",
	  [(set GR:$dst, (and GR:$src1, (not GR:$src2)))]>;
// TODO: and/andcm/or/xor/add/sub/shift immediate forms
def OR    : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "or $dst = $src1, $src2;;",
	  [(set GR:$dst, (or GR:$src1, GR:$src2))]>;

def pOR   : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2, PR:$qp),
          "($qp) or $dst = $src1, $src2;;">;

// the following are all a bit unfortunate: we throw away the complement
// of the compare!
def CMPEQ : AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.eq $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (seteq GR:$src1, GR:$src2))]>;
def CMPGT : AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.gt $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setgt GR:$src1, GR:$src2))]>;
def CMPGE : AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.ge $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setge GR:$src1, GR:$src2))]>;
def CMPLT : AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.lt $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setlt GR:$src1, GR:$src2))]>;
def CMPLE : AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.le $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setle GR:$src1, GR:$src2))]>;
def CMPNE : AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.ne $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setne GR:$src1, GR:$src2))]>;
def CMPLTU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.ltu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setult GR:$src1, GR:$src2))]>;
def CMPGTU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.gtu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setugt GR:$src1, GR:$src2))]>;
def CMPLEU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.leu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setule GR:$src1, GR:$src2))]>;
def CMPGEU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
          "cmp.geu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setuge GR:$src1, GR:$src2))]>;

// and we do the whole thing again for FP compares!
def FCMPEQ : AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.eq $dst, p0 = $src1, $src2;;",
          [(set PR:$dst, (seteq FP:$src1, FP:$src2))]>;
def FCMPGT : AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.gt $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setgt FP:$src1, FP:$src2))]>;
def FCMPGE : AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.ge $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setge FP:$src1, FP:$src2))]>;
def FCMPLT : AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.lt $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setlt FP:$src1, FP:$src2))]>;
def FCMPLE : AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.le $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setle FP:$src1, FP:$src2))]>;
def FCMPNE : AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.neq $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setne FP:$src1, FP:$src2))]>;
def FCMPLTU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.ltu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setult FP:$src1, FP:$src2))]>;
def FCMPGTU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.gtu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setugt FP:$src1, FP:$src2))]>;
def FCMPLEU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.leu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setule FP:$src1, FP:$src2))]>;
def FCMPGEU: AForm_DAG<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
          "fcmp.geu $dst, p0 = $src1, $src2;;",
	  [(set PR:$dst, (setuge FP:$src1, FP:$src2))]>;

def PCMPEQUNCR0R0 : AForm<0x03, 0x0b, (ops PR:$dst, PR:$qp),
    "($qp) cmp.eq.unc $dst, p0 = r0, r0;;">;

def : Pat<(trunc GR:$src),  // truncate i64 to i1
          (CMPNE GR:$src, r0)>; // $src!=0? If so, PR:$dst=true
	  
let isTwoAddress=1 in {
  def TPCMPEQR0R0 : AForm<0x03, 0x0b, (ops PR:$dst, PR:$bogus, PR:$qp),
    "($qp) cmp.eq $dst, p0 = r0, r0;;">;
  def TPCMPNER0R0 : AForm<0x03, 0x0b, (ops PR:$dst, PR:$bogus, PR:$qp),
    "($qp) cmp.ne $dst, p0 = r0, r0;;">;
}

/* our pseudocode for OR on predicates is:
pC = pA OR pB
-------------
(pA) cmp.eq.unc pC,p0 = r0,r0  // pC = pA
 ;;
(pB) cmp.eq pC,p0 = r0,r0 // if (pB) pC = 1 */

def bOR   : Pat<(or PR:$src1, PR:$src2),
          (TPCMPEQR0R0 (PCMPEQUNCR0R0 PR:$src1), PR:$src2)>;

/* our pseudocode for AND on predicates is:
 *
(pA) cmp.eq.unc pC,p0 = r0,r0   // pC = pA
     cmp.eq pTemp,p0 = r0,r0    // pTemp = NOT pB
     ;;
(pB) cmp.ne pTemp,p0 = r0,r0
     ;;
(pTemp)cmp.ne pC,p0 = r0,r0    // if (NOT pB) pC = 0  */

def bAND  : Pat<(and PR:$src1, PR:$src2),
          ( TPCMPNER0R0 (PCMPEQUNCR0R0 PR:$src1),
	    (TPCMPNER0R0 (CMPEQ r0, r0), PR:$src2) )>;

/* one possible routine for XOR on predicates is:

      // Compute px = py ^ pz
        // using sum of products: px = (py & !pz) | (pz & !py)
        // Uses 5 instructions in 3 cycles.
        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(py)    cmp.eq.unc      pt = r0, r0     // pt = py
        ;;
        // cycle 2
(pt)    cmp.ne.and      px = r0, r0     // px = px & !pt (px = pz & !pt)
(pz)    cmp.ne.and      pt = r0, r0     // pt = pt & !pz
        ;;
        } { .mmi
        // cycle 3
(pt)    cmp.eq.or       px = r0, r0     // px = px | pt

*** Another, which we use here, requires one scratch GR. it is:

        mov             rt = 0          // initialize rt off critical path
        ;;

        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(pz)    mov             rt = 1          // rt = pz
        ;;
        // cycle 2
(py)    cmp.ne          px = 1, rt      // if (py) px = !pz

.. these routines kindly provided by Jim Hull
*/
  
def bXOR  : Pat<(xor PR:$src1, PR:$src2),
          (TPCMPIMM8NE (PCMPEQUNCR0R0 PR:$src2), 1,
	               (PADDS r0, 1, PR:$src2),
                        PR:$src1)>;

def XOR   : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "xor $dst = $src1, $src2;;",
	  [(set GR:$dst, (xor GR:$src1, GR:$src2))]>;

def SHLADD: AForm_DAG<0x03, 0x0b, (ops GR:$dst,GR:$src1,s64imm:$imm,GR:$src2),
          "shladd $dst = $src1, $imm, $src2;;",
          [(set GR:$dst, (add GR:$src2, (shl GR:$src1, isSHLADDimm:$imm)))]>;

def SHL   : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "shl $dst = $src1, $src2;;",
	  [(set GR:$dst, (shl GR:$src1, GR:$src2))]>;

def SHRU  : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "shr.u $dst = $src1, $src2;;",
	  [(set GR:$dst, (srl GR:$src1, GR:$src2))]>;

def SHRS  : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
          "shr $dst = $src1, $src2;;",
	  [(set GR:$dst, (sra GR:$src1, GR:$src2))]>;

def MOV : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "mov $dst = $src;;">;
def FMOV : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "mov $dst = $src;;">; // XXX: there _is_ no fmov
def PMOV : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src, PR:$qp),
  "($qp) mov $dst = $src;;">;

def SPILL_ALL_PREDICATES_TO_GR : AForm<0x03, 0x0b, (ops GR:$dst),
  "mov $dst = pr;;">;
def FILL_ALL_PREDICATES_FROM_GR : AForm<0x03, 0x0b, (ops GR:$src),
  "mov pr = $src;;">;

let isTwoAddress = 1 in {
  def CMOV : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src2, GR:$src, PR:$qp),
    "($qp) mov $dst = $src;;">;
}

def PFMOV : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src, PR:$qp),
  "($qp) mov $dst = $src;;">;

let isTwoAddress = 1 in {
  def CFMOV : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src2, FP:$src, PR:$qp),
    "($qp) mov $dst = $src;;">;
}

def SELECTINT : Pat<(select PR:$which, GR:$src1, GR:$src2),
          (CMOV (MOV GR:$src2), GR:$src1, PR:$which)>; // note order!
def SELECTFP : Pat<(select PR:$which, FP:$src1, FP:$src2),
          (CFMOV (FMOV FP:$src2), FP:$src1, PR:$which)>; // note order!
// TODO: can do this faster, w/o using any integer regs (see pattern isel)
def SELECTBOOL : Pat<(select PR:$which, PR:$src1, PR:$src2), // note order!
          (CMPNE (CMOV
            (MOV (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src2)),
            (TPCADDIMM22 (ADDS r0, 0), 1, PR:$src1), PR:$which), r0)>;

// load constants of various sizes // FIXME: prettyprint -ve constants
def : Pat<(i64 immSExt14:$imm), (ADDS r0, immSExt14:$imm)>;
def : Pat<(i64 imm64:$imm), (MOVL imm64:$imm)>;
def : Pat<(i1 -1), (CMPEQ r0, r0)>; // TODO: this should just be a ref to p0
def : Pat<(i1  0), (CMPNE r0, r0)>; // TODO: any instruction actually *using*
                                    //       this predicate should be killed!

// TODO: support postincrement (reg, imm9) loads+stores - this needs more
// tablegen support

def PHI : PseudoInstIA64<(ops variable_ops), "PHI">;
def IDEF : PseudoInstIA64<(ops variable_ops), "// IDEF">;

def IDEF_GR_D : PseudoInstIA64_DAG<(ops GR:$reg), "// $reg = IDEF",
    [(set GR:$reg, (undef))]>;
def IDEF_FP_D : PseudoInstIA64_DAG<(ops FP:$reg), "// $reg = IDEF",
    [(set FP:$reg, (undef))]>;
def IDEF_PR_D : PseudoInstIA64_DAG<(ops PR:$reg), "// $reg = IDEF",
    [(set PR:$reg, (undef))]>;

def IUSE : PseudoInstIA64<(ops variable_ops), "// IUSE">;
def ADJUSTCALLSTACKUP : PseudoInstIA64<(ops variable_ops),
                                        "// ADJUSTCALLSTACKUP">;
def ADJUSTCALLSTACKDOWN : PseudoInstIA64<(ops variable_ops),
                                         "// ADJUSTCALLSTACKDOWN">;
def PSEUDO_ALLOC : PseudoInstIA64<(ops GR:$foo), "// PSEUDO_ALLOC">;

def ALLOC : AForm<0x03, 0x0b,
  (ops GR:$dst, i8imm:$inputs, i8imm:$locals, i8imm:$outputs, i8imm:$rotating),
    "alloc $dst = ar.pfs,$inputs,$locals,$outputs,$rotating;;">;

let isTwoAddress = 1 in {
  def TCMPNE : AForm<0x03, 0x0b,
  (ops PR:$dst, PR:$src2, GR:$src3, GR:$src4),
    "cmp.ne $dst, p0 = $src3, $src4;;">;
  
  def TPCMPEQOR : AForm<0x03, 0x0b,
  (ops PR:$dst, PR:$src2, GR:$src3, GR:$src4, PR:$qp),
    "($qp) cmp.eq.or $dst, p0 = $src3, $src4;;">;
  
  def TPCMPNE : AForm<0x03, 0x0b,
  (ops PR:$dst, PR:$src2, GR:$src3, GR:$src4, PR:$qp),
    "($qp) cmp.ne $dst, p0 = $src3, $src4;;">;
  
  def TPCMPEQ : AForm<0x03, 0x0b,
  (ops PR:$dst, PR:$src2, GR:$src3, GR:$src4, PR:$qp),
    "($qp) cmp.eq $dst, p0 = $src3, $src4;;">;
}

def MOVSIMM14 : AForm<0x03, 0x0b, (ops GR:$dst, s14imm:$imm),
  "mov $dst = $imm;;">;
def MOVSIMM22 : AForm<0x03, 0x0b, (ops GR:$dst, s22imm:$imm),
  "mov $dst = $imm;;">;
def MOVLIMM64 : AForm<0x03, 0x0b, (ops GR:$dst, s64imm:$imm),
  "movl $dst = $imm;;">;

def SHLI : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, u6imm:$imm), 
  "shl $dst = $src1, $imm;;">;
def SHRUI : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, u6imm:$imm),
  "shr.u $dst = $src1, $imm;;">;
def SHRSI : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, u6imm:$imm),
  "shr $dst = $src1, $imm;;">;

def EXTRU : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, u6imm:$imm1, u6imm:$imm2),
  "extr.u $dst = $src1, $imm1, $imm2;;">;

def DEPZ : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, u6imm:$imm1, u6imm:$imm2),	  "dep.z $dst = $src1, $imm1, $imm2;;">;

def PCMPEQOR : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.eq.or $dst, p0 = $src1, $src2;;">;
def PCMPEQUNC : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.eq.unc $dst, p0 = $src1, $src2;;">;
def PCMPNE : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.ne $dst, p0 = $src1, $src2;;">;

// two destinations! 
def BCMPEQ : AForm<0x03, 0x0b, (ops PR:$dst1, PR:$dst2, GR:$src1, GR:$src2),
  "cmp.eq $dst1, dst2 = $src1, $src2;;">;

def ADDIMM14 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s14imm:$imm),
  "adds $dst = $imm, $src1;;">;

def ADDIMM22 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s22imm:$imm),
  "add $dst = $imm, $src1;;">;
def CADDIMM22 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s22imm:$imm, PR:$qp),
  "($qp) add $dst = $imm, $src1;;">;

def SUBIMM8 : AForm<0x03, 0x0b, (ops GR:$dst, s8imm:$imm, GR:$src2),
  "sub $dst = $imm, $src2;;">;

let isStore = 1, noResults = 1 in {
  def ST1 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
    "st1 [$dstPtr] = $value;;">;
  def ST2 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
    "st2 [$dstPtr] = $value;;">;
  def ST4 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
    "st4 [$dstPtr] = $value;;">;
  def ST8 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
    "st8 [$dstPtr] = $value;;">;
  def STF4 : AForm<0x03, 0x0b, (ops GR:$dstPtr, FP:$value),
    "stfs [$dstPtr] = $value;;">;
  def STF8 : AForm<0x03, 0x0b, (ops GR:$dstPtr, FP:$value),
    "stfd [$dstPtr] = $value;;">;
}

let isLoad = 1 in {
  def LD1 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
    "ld1 $dst = [$srcPtr];;">;
  def LD2 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
    "ld2 $dst = [$srcPtr];;">;
  def LD4 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
    "ld4 $dst = [$srcPtr];;">;
  def LD8 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
    "ld8 $dst = [$srcPtr];;">;
  def LDF4 : AForm<0x03, 0x0b, (ops FP:$dst, GR:$srcPtr),
    "ldfs $dst = [$srcPtr];;">;
  def LDF8 : AForm<0x03, 0x0b, (ops FP:$dst, GR:$srcPtr),
    "ldfd $dst = [$srcPtr];;">;
}

def POPCNT : AForm_DAG<0x03, 0x0b, (ops GR:$dst, GR:$src),
  "popcnt $dst = $src;;",
  [(set GR:$dst, (ctpop GR:$src))]>;

// some FP stuff:  // TODO: single-precision stuff?
def FADD : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fadd $dst = $src1, $src2;;",
  [(set FP:$dst, (fadd FP:$src1, FP:$src2))]>;
def FADDS: AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fadd.s $dst = $src1, $src2;;">;
def FSUB : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fsub $dst = $src1, $src2;;",
  [(set FP:$dst, (fsub FP:$src1, FP:$src2))]>;
def FMPY : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fmpy $dst = $src1, $src2;;",
  [(set FP:$dst, (fmul FP:$src1, FP:$src2))]>;
def FMA : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "fma $dst = $src1, $src2, $src3;;",
  [(set FP:$dst, (fadd (fmul FP:$src1, FP:$src2), FP:$src3))]>;
def FMS : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "fms $dst = $src1, $src2, $src3;;",
  [(set FP:$dst, (fsub (fmul FP:$src1, FP:$src2), FP:$src3))]>;
def FNMA : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "fnma $dst = $src1, $src2, $src3;;",
  [(set FP:$dst, (fneg (fadd (fmul FP:$src1, FP:$src2), FP:$src3)))]>;
def FABS : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fabs $dst = $src;;",
  [(set FP:$dst, (fabs FP:$src))]>;
def FNEG : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fneg $dst = $src;;",
  [(set FP:$dst, (fneg FP:$src))]>;
def FNEGABS : AForm_DAG<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fnegabs $dst = $src;;",
  [(set FP:$dst, (fneg (fabs FP:$src)))]>;

def CFMAS1 : AForm<0x03, 0x0b,
  (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.s1 $dst = $src1, $src2, $src3;;">;
def CFNMAS1 : AForm<0x03, 0x0b,
  (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fnma.s1 $dst = $src1, $src2, $src3;;">;

def FRCPAS1 : AForm<0x03, 0x0b, (ops FP:$dstFR, PR:$dstPR, FP:$src1, FP:$src2),
  "frcpa.s1 $dstFR, $dstPR = $src1, $src2;;">;

def XMAL : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "xma.l $dst = $src1, $src2, $src3;;">;

def FCVTXF : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.xf $dst = $src;;">;
def FCVTXUF : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.xuf $dst = $src;;">;
def FCVTXUFS1 : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.xuf.s1 $dst = $src;;">;
def FCVTFX : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.fx $dst = $src;;">;
def FCVTFXU : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.fxu $dst = $src;;">;

def FCVTFXTRUNC : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.fx.trunc $dst = $src;;">;
def FCVTFXUTRUNC : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.fxu.trunc $dst = $src;;">;

def FCVTFXTRUNCS1 : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.fx.trunc.s1 $dst = $src;;">;
def FCVTFXUTRUNCS1 : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fcvt.fxu.trunc.s1 $dst = $src;;">;

def FNORMD : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fnorm.d $dst = $src;;">;

def GETFD : AForm<0x03, 0x0b, (ops GR:$dst, FP:$src),
  "getf.d $dst = $src;;">;
def SETFD : AForm<0x03, 0x0b, (ops FP:$dst, GR:$src),
  "setf.d $dst = $src;;">;

def GETFSIG : AForm<0x03, 0x0b, (ops GR:$dst, FP:$src),
  "getf.sig $dst = $src;;">;
def SETFSIG : AForm<0x03, 0x0b, (ops FP:$dst, GR:$src),
  "setf.sig $dst = $src;;">;

// these four FP<->int conversion patterns need checking/cleaning
def SINT_TO_FP : Pat<(sint_to_fp GR:$src),
  (FNORMD (FCVTXF (SETFSIG GR:$src)))>;
def UINT_TO_FP : Pat<(uint_to_fp GR:$src),
  (FNORMD (FCVTXUF (SETFSIG GR:$src)))>;
def FP_TO_SINT : Pat<(i64 (fp_to_sint FP:$src)),
  (GETFSIG (FCVTFXTRUNC FP:$src))>;
def FP_TO_UINT : Pat<(i64 (fp_to_uint FP:$src)),
  (GETFSIG (FCVTFXUTRUNC FP:$src))>;


let isTerminator = 1, isBranch = 1, noResults = 1 in {
  def BRL_NOTCALL : RawForm<0x03, 0xb0, (ops i64imm:$dst),
    "(p0) brl.cond.sptk $dst;;">;
  def BRLCOND_NOTCALL : RawForm<0x03, 0xb0, (ops PR:$qp, i64imm:$dst),
    "($qp) brl.cond.sptk $dst;;">;
  def BRCOND_NOTCALL : RawForm<0x03, 0xb0, (ops PR:$qp, GR:$dst),
    "($qp) br.cond.sptk $dst;;">;
}

let isCall = 1, noResults = 1, /* isTerminator = 1, isBranch = 1, */
  Uses = [out0,out1,out2,out3,out4,out5,out6,out7],
// all calls clobber non-callee-saved registers, and for now, they are these:
  Defs = [r2,r3,r8,r9,r10,r11,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,
  r25,r26,r27,r28,r29,r30,r31,
  p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,
  F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,
  F32,F33,F34,F35,F36,F37,F38,F39,F40,F41,F42,F43,F44,F45,F46,F47,F48,F49,
  F50,F51,F52,F53,F54,F55,F56,
  F57,F58,F59,F60,F61,F62,F63,F64,F65,F66,F67,F68,F69,F70,F71,F72,F73,F74,
  F75,F76,F77,F78,F79,F80,F81,
  F82,F83,F84,F85,F86,F87,F88,F89,F90,F91,F92,F93,F94,F95,F96,F97,F98,F99,
  F100,F101,F102,F103,F104,F105,
  F106,F107,F108,F109,F110,F111,F112,F113,F114,F115,F116,F117,F118,F119,
  F120,F121,F122,F123,F124,F125,F126,F127,
  out0,out1,out2,out3,out4,out5,out6,out7] in {
// old pattern call
  def BRCALL: RawForm<0x03, 0xb0, (ops calltarget:$dst),
  "br.call.sptk rp = $dst;;">;       // FIXME: teach llvm about branch regs?
// new daggy stuff!  

// calls a globaladdress
  def BRCALL_IPREL_GA : RawForm<0x03, 0xb0, (ops calltarget:$dst),
  "br.call.sptk rp = $dst;;">;       // FIXME: teach llvm about branch regs?
// calls an externalsymbol
  def BRCALL_IPREL_ES : RawForm<0x03, 0xb0, (ops calltarget:$dst),
  "br.call.sptk rp = $dst;;">;       // FIXME: teach llvm about branch regs?
// calls through a function descriptor
  def BRCALL_INDIRECT : RawForm<0x03, 0xb0, (ops GR:$branchreg),
  "br.call.sptk rp = $branchreg;;">; // FIXME: teach llvm about branch regs?
  def BRLCOND_CALL : RawForm<0x03, 0xb0, (ops PR:$qp, i64imm:$dst),
    "($qp) brl.cond.call.sptk $dst;;">;
  def BRCOND_CALL : RawForm<0x03, 0xb0, (ops PR:$qp, GR:$dst),
    "($qp) br.cond.call.sptk $dst;;">;
}

let isTerminator = 1, isReturn = 1, noResults = 1 in
  def RET : RawForm<0x03, 0xb0, (ops), "br.ret.sptk.many rp;;">; // return


