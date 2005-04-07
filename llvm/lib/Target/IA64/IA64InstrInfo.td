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

def u6imm : Operand<i8>;
def s8imm : Operand<i8> {
  let PrintMethod = "printS8ImmOperand";
}
def s14imm  : Operand<i16> {
  let PrintMethod = "printS14ImmOperand";
}
def s16imm  : Operand<i16>;
def s21imm  : Operand<i32> {
  let PrintMethod = "printS21ImmOperand";
}
def u32imm  : Operand<i32> {
  let PrintMethod = "printU32ImmOperand";
}
def s32imm  : Operand<i32> {
  let PrintMethod = "printS32ImmOperand";
}
def u64imm  : Operand<i64> {
  let PrintMethod = "printU64ImmOperand";
}

// the asmprinter needs to know about calls
let PrintMethod = "printCallOperand" in
  def calltarget : Operand<i64>;
  
def PHI : PseudoInstIA64<(ops), "PHI">;
def IDEF : PseudoInstIA64<(ops), "// IDEF">;
def IUSE : PseudoInstIA64<(ops), "// IUSE">;
def WTF : PseudoInstIA64<(ops), "que??">;
def ADJUSTCALLSTACKUP : PseudoInstIA64<(ops), "// ADJUSTCALLSTACKUP">;
def ADJUSTCALLSTACKDOWN : PseudoInstIA64<(ops), "// ADJUSTCALLSTACKDOWN">;
def PSEUDO_ALLOC : PseudoInstIA64<(ops), "// PSEUDO_ALLOC">;

def ALLOC : AForm<0x03, 0x0b,
  (ops GR:$dst, i8imm:$inputs, i8imm:$locals, i8imm:$outputs, i8imm:$rotating),
    "alloc $dst = ar.pfs,$inputs,$locals,$outputs,$rotating;;">;

def MOV : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "mov $dst = $src;;">;
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

def MOVI32 : AForm<0x03, 0x0b, (ops GR:$dst, u32imm:$imm),
  "mov $dst = $imm;;">;
def MOVLI32 : AForm<0x03, 0x0b, (ops GR:$dst, u32imm:$imm),
  "movl $dst = $imm;;">;
def MOVLSI32 : AForm<0x03, 0x0b, (ops GR:$dst, s32imm:$imm),
  "movl $dst = $imm;;">;
def MOVLI64 : AForm<0x03, 0x0b, (ops GR:$dst, u64imm:$imm),
  "movl $dst = $imm;;">;

def AND : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "and $dst = $src1, $src2;;">;
def OR : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "or $dst = $src1, $src2;;">;
def XOR : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "xor $dst = $src1, $src2;;">;
def SHL : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "shl $dst = $src1, $src2;;">;
def SHLI : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s21imm:$imm), 
  "shl $dst = $src1, $imm;;">; // FIXME: 6 immediate bits, not 21
def SHRU : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "shr.u $dst = $src1, $src2;;">;
def SHRUI : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s21imm:$imm),
  "shr.u $dst = $src1, $imm;;">;
def SHRS : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "shr $dst = $src1, $src2;;">;
def SHRSI : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s21imm:$imm),
  "shr $dst = $src1, $imm;;">;

def DEPZ : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, u6imm:$imm1, u6imm:$imm2),	  "dep.z $dst = $src1, $imm1, $imm2;;">;

def SXT1 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "sxt1 $dst = $src;;">;
def ZXT1 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "zxt1 $dst = $src;;">;
def SXT2 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "sxt2 $dst = $src;;">;
def ZXT2 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "zxt2 $dst = $src;;">;
def SXT4 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "sxt4 $dst = $src;;">;
def ZXT4 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src), "zxt4 $dst = $src;;">;

// the following are all a bit unfortunate: we throw away the complement
// of the compare!
def CMPEQ : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.eq $dst, p0 = $src1, $src2;;">;
def CMPGT : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.gt $dst, p0 = $src1, $src2;;">;
def CMPGE : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.ge $dst, p0 = $src1, $src2;;">;
def CMPLT : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.lt $dst, p0 = $src1, $src2;;">;
def CMPLE : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.le $dst, p0 = $src1, $src2;;">;
def CMPNE : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.ne $dst, p0 = $src1, $src2;;">;
def CMPLTU : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.ltu $dst, p0 = $src1, $src2;;">;
def CMPGTU : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.gtu $dst, p0 = $src1, $src2;;">;
def CMPLEU : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.leu $dst, p0 = $src1, $src2;;">;
def CMPGEU : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2),
  "cmp.geu $dst, p0 = $src1, $src2;;">;

// and we do the whole thing again for FP compares!
def FCMPEQ : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.eq $dst, p0 = $src1, $src2;;">;
def FCMPGT : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.gt $dst, p0 = $src1, $src2;;">;
def FCMPGE : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.ge $dst, p0 = $src1, $src2;;">;
def FCMPLT : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.lt $dst, p0 = $src1, $src2;;">;
def FCMPLE : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.le $dst, p0 = $src1, $src2;;">;
def FCMPNE : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.neq $dst, p0 = $src1, $src2;;">;
def FCMPLTU : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.ltu $dst, p0 = $src1, $src2;;">;
def FCMPGTU : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.gtu $dst, p0 = $src1, $src2;;">;
def FCMPLEU : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.leu $dst, p0 = $src1, $src2;;">;
def FCMPGEU : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "fcmp.geu $dst, p0 = $src1, $src2;;">;

def PCMPEQOR : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.eq.or $dst, p0 = $src1, $src2;;">;
def PCMPEQUNC : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.eq.unc $dst, p0 = $src1, $src2;;">;
def PCMPNE : AForm<0x03, 0x0b, (ops PR:$dst, GR:$src1, GR:$src2, PR:$qp),
  "($qp) cmp.ne $dst, p0 = $src1, $src2;;">;

// two destinations! 
def BCMPEQ : AForm<0x03, 0x0b, (ops PR:$dst1, PR:$dst2, GR:$src1, GR:$src2),
  "cmp.eq $dst1, dst2 = $src1, $src2;;">;

def ADD : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "add $dst = $src1, $src2;;">;
def ADDIMM14 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s14imm:$imm),
  "adds $dst = $imm, $src1;;">;

def ADDIMM22 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s21imm:$imm),
  "add $dst = $imm, $src1;;">;
def CADDIMM22 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, s21imm:$imm, PR:$qp),
  "($qp) add $dst = $imm, $src1;;">;

let isTwoAddress = 1 in {
def TPCADDIMM22 : AForm<0x03, 0x0b,
  (ops GR:$dst, GR:$src1, s21imm:$imm, PR:$qp),
    "($qp) add $dst = $imm, $dst;;">;
def TPCMPIMM8NE : AForm<0x03, 0x0b,
  (ops PR:$dst, PR:$src1, s21imm:$imm, GR:$src2, PR:$qp),
    "($qp) cmp.ne $dst , p0 = $imm, $src2;;">;
}

def SUB : AForm<0x03, 0x0b, (ops GR:$dst, GR:$src1, GR:$src2),
  "sub $dst = $src1, $src2;;">;
def SUBIMM8 : AForm<0x03, 0x0b, (ops GR:$dst, s8imm:$imm, GR:$src2),
  "sub $dst = $imm, $src2;;">;

def ST1 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
  "st1 [$dstPtr] = $value;;">;
def ST2 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
  "st2 [$dstPtr] = $value;;">;
def ST4 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
  "st4 [$dstPtr] = $value;;">;
def ST8 : AForm<0x03, 0x0b, (ops GR:$dstPtr, GR:$value),
  "st8 [$dstPtr] = $value;;">;

def LD1 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
  "ld1 $dst = [$srcPtr];;">;
def LD2 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
  "ld2 $dst = [$srcPtr];;">;
def LD4 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
  "ld4 $dst = [$srcPtr];;">;
def LD8 : AForm<0x03, 0x0b, (ops GR:$dst, GR:$srcPtr),
  "ld8 $dst = [$srcPtr];;">;

// some FP stuff:
def FADD : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fadd $dst = $src1, $src2;;">;
def FADDS: AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fadd.s $dst = $src1, $src2;;">;
def FSUB : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fsub $dst = $src1, $src2;;">;
def FMPY : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "fmpy $dst = $src1, $src2;;">;
def FMOV : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "mov $dst = $src;;">; // XXX: there _is_ no fmov
def FMA : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "fma $dst = $src1, $src2, $src3;;">;
def FMS : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "fms $dst = $src1, $src2, $src3;;">;
def FNMA : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3),
  "fnma $dst = $src1, $src2, $src3;;">;
def FABS : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fabs $dst = $src;;">;
def FNEG : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fneg $dst = $src;;">;
def FNEGABS : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src),
  "fnegabs $dst = $src;;">;

def CFMAS1 : AForm<0x03, 0x0b,
  (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fma.s1 $dst = $src1, $src2, $src3;;">;
def CFNMAS1 : AForm<0x03, 0x0b,
  (ops FP:$dst, FP:$src1, FP:$src2, FP:$src3, PR:$qp),
    "($qp) fnma.s1 $dst = $src1, $src2, $src3;;">;

// FIXME: we 'explode' FRCPA (which should write two registers) into two
// operations that write one each. this is a waste, and is also destroying
// f127. not cool.
def FRCPAS1FLOAT : AForm<0x03, 0x0b, (ops FP:$dst, FP:$src1, FP:$src2),
  "frcpa.s1 $dst , p0 = $src1, $src2;;">;
// XXX: this _will_ break things: (f127)
def FRCPAS1PREDICATE : AForm<0x03, 0x0b, (ops PR:$dst, FP:$src1, FP:$src2),
  "frcpa.s1 f127 , $dst = $src1, $src2;; // XXX FIXME!!!!">;

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

def LDF4 : AForm<0x03, 0x0b, (ops FP:$dst, GR:$srcPtr),
  "ldfs $dst = [$srcPtr];;">;
def LDF8 : AForm<0x03, 0x0b, (ops FP:$dst, GR:$srcPtr),
  "ldfd $dst = [$srcPtr];;">;

def STF4 : AForm<0x03, 0x0b, (ops GR:$dstPtr, FP:$value),
  "stfs [$dstPtr] = $value;;">;
def STF8 : AForm<0x03, 0x0b, (ops GR:$dstPtr, FP:$value),
  "stfd [$dstPtr] = $value;;">;

let isTerminator = 1, isBranch = 1 in {
  def BRLCOND_NOTCALL : RawForm<0x03, 0xb0, (ops PR:$qp, i64imm:$dst),
    "($qp) brl.cond.sptk $dst;;">;
  def BRCOND_NOTCALL : RawForm<0x03, 0xb0, (ops PR:$qp, GR:$dst),
    "($qp) br.cond.sptk $dst;;">;
}

let isCall = 1, isTerminator = 1, isBranch = 1, 
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
  def BRCALL : RawForm<0x03, 0xb0, (ops calltarget:$dst),
  "br.call.sptk rp = $dst;;">;       // FIXME: teach llvm about branch regs?
  def BRLCOND_CALL : RawForm<0x03, 0xb0, (ops PR:$qp, i64imm:$dst),
    "($qp) brl.cond.call.sptk $dst;;">;
  def BRCOND_CALL : RawForm<0x03, 0xb0, (ops PR:$qp, GR:$dst),
    "($qp) br.cond.call.sptk $dst;;">;
}

let isTerminator = 1, isReturn = 1 in
  def RET : RawForm<0x03, 0xb0, (ops), "br.ret.sptk.many rp;;">; // return


