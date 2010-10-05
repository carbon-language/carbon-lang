//===- X86InstrCMovSetCC.td - Conditional Move and SetCC ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 conditional move and set on condition
// instructions.
//
//===----------------------------------------------------------------------===//

// FIXME: Someone please sprinkle some defm's in here!

let Constraints = "$src1 = $dst" in {

// Conditional moves
let Uses = [EFLAGS] in {

let Predicates = [HasCMov] in {
let isCommutable = 1 in {
def CMOVB16rr : I<0x42, MRMSrcReg,       // if <u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovb{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_B, EFLAGS))]>,
                  TB, OpSize;
def CMOVB32rr : I<0x42, MRMSrcReg,       // if <u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovb{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_B, EFLAGS))]>,
                   TB;
def CMOVAE16rr: I<0x43, MRMSrcReg,       // if >=u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovae{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_AE, EFLAGS))]>,
                   TB, OpSize;
def CMOVAE32rr: I<0x43, MRMSrcReg,       // if >=u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovae{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_AE, EFLAGS))]>,
                   TB;
def CMOVE16rr : I<0x44, MRMSrcReg,       // if ==, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmove{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_E, EFLAGS))]>,
                   TB, OpSize;
def CMOVE32rr : I<0x44, MRMSrcReg,       // if ==, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmove{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_E, EFLAGS))]>,
                   TB;
def CMOVNE16rr: I<0x45, MRMSrcReg,       // if !=, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovne{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_NE, EFLAGS))]>,
                   TB, OpSize;
def CMOVNE32rr: I<0x45, MRMSrcReg,       // if !=, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovne{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_NE, EFLAGS))]>,
                   TB;
def CMOVBE16rr: I<0x46, MRMSrcReg,       // if <=u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovbe{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_BE, EFLAGS))]>,
                   TB, OpSize;
def CMOVBE32rr: I<0x46, MRMSrcReg,       // if <=u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovbe{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_BE, EFLAGS))]>,
                   TB;
def CMOVA16rr : I<0x47, MRMSrcReg,       // if >u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmova{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_A, EFLAGS))]>,
                   TB, OpSize;
def CMOVA32rr : I<0x47, MRMSrcReg,       // if >u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmova{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_A, EFLAGS))]>,
                   TB;
def CMOVL16rr : I<0x4C, MRMSrcReg,       // if <s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovl{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_L, EFLAGS))]>,
                   TB, OpSize;
def CMOVL32rr : I<0x4C, MRMSrcReg,       // if <s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovl{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_L, EFLAGS))]>,
                   TB;
def CMOVGE16rr: I<0x4D, MRMSrcReg,       // if >=s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovge{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_GE, EFLAGS))]>,
                   TB, OpSize;
def CMOVGE32rr: I<0x4D, MRMSrcReg,       // if >=s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovge{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_GE, EFLAGS))]>,
                   TB;
def CMOVLE16rr: I<0x4E, MRMSrcReg,       // if <=s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovle{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_LE, EFLAGS))]>,
                   TB, OpSize;
def CMOVLE32rr: I<0x4E, MRMSrcReg,       // if <=s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovle{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_LE, EFLAGS))]>,
                   TB;
def CMOVG16rr : I<0x4F, MRMSrcReg,       // if >s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovg{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_G, EFLAGS))]>,
                   TB, OpSize;
def CMOVG32rr : I<0x4F, MRMSrcReg,       // if >s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovg{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_G, EFLAGS))]>,
                   TB;
def CMOVS16rr : I<0x48, MRMSrcReg,       // if signed, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovs{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_S, EFLAGS))]>,
                  TB, OpSize;
def CMOVS32rr : I<0x48, MRMSrcReg,       // if signed, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovs{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_S, EFLAGS))]>,
                  TB;
def CMOVNS16rr: I<0x49, MRMSrcReg,       // if !signed, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovns{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_NS, EFLAGS))]>,
                  TB, OpSize;
def CMOVNS32rr: I<0x49, MRMSrcReg,       // if !signed, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovns{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_NS, EFLAGS))]>,
                  TB;
def CMOVP16rr : I<0x4A, MRMSrcReg,       // if parity, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovp{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_P, EFLAGS))]>,
                  TB, OpSize;
def CMOVP32rr : I<0x4A, MRMSrcReg,       // if parity, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovp{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_P, EFLAGS))]>,
                  TB;
def CMOVNP16rr : I<0x4B, MRMSrcReg,       // if !parity, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovnp{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                    X86_COND_NP, EFLAGS))]>,
                  TB, OpSize;
def CMOVNP32rr : I<0x4B, MRMSrcReg,       // if !parity, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovnp{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                    X86_COND_NP, EFLAGS))]>,
                  TB;
def CMOVO16rr : I<0x40, MRMSrcReg,       // if overflow, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovo{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_O, EFLAGS))]>,
                  TB, OpSize;
def CMOVO32rr : I<0x40, MRMSrcReg,       // if overflow, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovo{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_O, EFLAGS))]>,
                  TB;
def CMOVNO16rr : I<0x41, MRMSrcReg,       // if !overflow, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovno{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                    X86_COND_NO, EFLAGS))]>,
                  TB, OpSize;
def CMOVNO32rr : I<0x41, MRMSrcReg,       // if !overflow, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovno{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                    X86_COND_NO, EFLAGS))]>,
                  TB;
} // isCommutable = 1

def CMOVB16rm : I<0x42, MRMSrcMem,       // if <u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovb{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_B, EFLAGS))]>,
                  TB, OpSize;
def CMOVB32rm : I<0x42, MRMSrcMem,       // if <u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovb{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_B, EFLAGS))]>,
                   TB;
def CMOVAE16rm: I<0x43, MRMSrcMem,       // if >=u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovae{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_AE, EFLAGS))]>,
                   TB, OpSize;
def CMOVAE32rm: I<0x43, MRMSrcMem,       // if >=u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovae{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_AE, EFLAGS))]>,
                   TB;
def CMOVE16rm : I<0x44, MRMSrcMem,       // if ==, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmove{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_E, EFLAGS))]>,
                   TB, OpSize;
def CMOVE32rm : I<0x44, MRMSrcMem,       // if ==, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmove{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_E, EFLAGS))]>,
                   TB;
def CMOVNE16rm: I<0x45, MRMSrcMem,       // if !=, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovne{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NE, EFLAGS))]>,
                   TB, OpSize;
def CMOVNE32rm: I<0x45, MRMSrcMem,       // if !=, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovne{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NE, EFLAGS))]>,
                   TB;
def CMOVBE16rm: I<0x46, MRMSrcMem,       // if <=u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovbe{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_BE, EFLAGS))]>,
                   TB, OpSize;
def CMOVBE32rm: I<0x46, MRMSrcMem,       // if <=u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovbe{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_BE, EFLAGS))]>,
                   TB;
def CMOVA16rm : I<0x47, MRMSrcMem,       // if >u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmova{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_A, EFLAGS))]>,
                   TB, OpSize;
def CMOVA32rm : I<0x47, MRMSrcMem,       // if >u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmova{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_A, EFLAGS))]>,
                   TB;
def CMOVL16rm : I<0x4C, MRMSrcMem,       // if <s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovl{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_L, EFLAGS))]>,
                   TB, OpSize;
def CMOVL32rm : I<0x4C, MRMSrcMem,       // if <s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovl{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_L, EFLAGS))]>,
                   TB;
def CMOVGE16rm: I<0x4D, MRMSrcMem,       // if >=s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovge{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_GE, EFLAGS))]>,
                   TB, OpSize;
def CMOVGE32rm: I<0x4D, MRMSrcMem,       // if >=s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovge{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_GE, EFLAGS))]>,
                   TB;
def CMOVLE16rm: I<0x4E, MRMSrcMem,       // if <=s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovle{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_LE, EFLAGS))]>,
                   TB, OpSize;
def CMOVLE32rm: I<0x4E, MRMSrcMem,       // if <=s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovle{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_LE, EFLAGS))]>,
                   TB;
def CMOVG16rm : I<0x4F, MRMSrcMem,       // if >s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovg{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_G, EFLAGS))]>,
                   TB, OpSize;
def CMOVG32rm : I<0x4F, MRMSrcMem,       // if >s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovg{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_G, EFLAGS))]>,
                   TB;
def CMOVS16rm : I<0x48, MRMSrcMem,       // if signed, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovs{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_S, EFLAGS))]>,
                  TB, OpSize;
def CMOVS32rm : I<0x48, MRMSrcMem,       // if signed, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovs{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_S, EFLAGS))]>,
                  TB;
def CMOVNS16rm: I<0x49, MRMSrcMem,       // if !signed, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovns{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NS, EFLAGS))]>,
                  TB, OpSize;
def CMOVNS32rm: I<0x49, MRMSrcMem,       // if !signed, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovns{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NS, EFLAGS))]>,
                  TB;
def CMOVP16rm : I<0x4A, MRMSrcMem,       // if parity, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovp{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_P, EFLAGS))]>,
                  TB, OpSize;
def CMOVP32rm : I<0x4A, MRMSrcMem,       // if parity, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovp{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_P, EFLAGS))]>,
                  TB;
def CMOVNP16rm : I<0x4B, MRMSrcMem,       // if !parity, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovnp{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NP, EFLAGS))]>,
                  TB, OpSize;
def CMOVNP32rm : I<0x4B, MRMSrcMem,       // if !parity, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovnp{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NP, EFLAGS))]>,
                  TB;
def CMOVO16rm : I<0x40, MRMSrcMem,       // if overflow, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovo{w}\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_O, EFLAGS))]>,
                  TB, OpSize;
def CMOVO32rm : I<0x40, MRMSrcMem,       // if overflow, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovo{l}\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_O, EFLAGS))]>,
                  TB;
def CMOVNO16rm : I<0x41, MRMSrcMem,       // if !overflow, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovno{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NO, EFLAGS))]>,
                  TB, OpSize;
def CMOVNO32rm : I<0x41, MRMSrcMem,       // if !overflow, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovno{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NO, EFLAGS))]>,
                  TB;
} // Predicates = [HasCMov]

// X86 doesn't have 8-bit conditional moves. Use a customInserter to
// emit control flow. An alternative to this is to mark i8 SELECT as Promote,
// however that requires promoting the operands, and can induce additional
// i8 register pressure. Note that CMOV_GR8 is conservatively considered to
// clobber EFLAGS, because if one of the operands is zero, the expansion
// could involve an xor.
let usesCustomInserter = 1, Constraints = "", Defs = [EFLAGS] in {
def CMOV_GR8 : I<0, Pseudo,
                 (outs GR8:$dst), (ins GR8:$src1, GR8:$src2, i8imm:$cond),
                 "#CMOV_GR8 PSEUDO!",
                 [(set GR8:$dst, (X86cmov GR8:$src1, GR8:$src2,
                                          imm:$cond, EFLAGS))]>;

let Predicates = [NoCMov] in {
def CMOV_GR32 : I<0, Pseudo,
                    (outs GR32:$dst), (ins GR32:$src1, GR32:$src2, i8imm:$cond),
                    "#CMOV_GR32* PSEUDO!",
                    [(set GR32:$dst,
                      (X86cmov GR32:$src1, GR32:$src2, imm:$cond, EFLAGS))]>;
def CMOV_GR16 : I<0, Pseudo,
                    (outs GR16:$dst), (ins GR16:$src1, GR16:$src2, i8imm:$cond),
                    "#CMOV_GR16* PSEUDO!",
                    [(set GR16:$dst,
                      (X86cmov GR16:$src1, GR16:$src2, imm:$cond, EFLAGS))]>;
def CMOV_RFP32 : I<0, Pseudo,
                    (outs RFP32:$dst),
                    (ins RFP32:$src1, RFP32:$src2, i8imm:$cond),
                    "#CMOV_RFP32 PSEUDO!",
                    [(set RFP32:$dst,
                      (X86cmov RFP32:$src1, RFP32:$src2, imm:$cond,
                                                  EFLAGS))]>;
def CMOV_RFP64 : I<0, Pseudo,
                    (outs RFP64:$dst),
                    (ins RFP64:$src1, RFP64:$src2, i8imm:$cond),
                    "#CMOV_RFP64 PSEUDO!",
                    [(set RFP64:$dst,
                      (X86cmov RFP64:$src1, RFP64:$src2, imm:$cond,
                                                  EFLAGS))]>;
def CMOV_RFP80 : I<0, Pseudo,
                    (outs RFP80:$dst),
                    (ins RFP80:$src1, RFP80:$src2, i8imm:$cond),
                    "#CMOV_RFP80 PSEUDO!",
                    [(set RFP80:$dst,
                      (X86cmov RFP80:$src1, RFP80:$src2, imm:$cond,
                                                  EFLAGS))]>;
} // Predicates = [NoCMov]
} // UsesCustomInserter = 1, Constraints = "", Defs = [EFLAGS] 
} // Uses = [EFLAGS]

} // Constraints = "$src1 = $dst" in


// Conditional moves
let Uses = [EFLAGS], Constraints = "$src1 = $dst" in {
let isCommutable = 1 in {
def CMOVB64rr : RI<0x42, MRMSrcReg,       // if <u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovb{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_B, EFLAGS))]>, TB;
def CMOVAE64rr: RI<0x43, MRMSrcReg,       // if >=u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovae{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_AE, EFLAGS))]>, TB;
def CMOVE64rr : RI<0x44, MRMSrcReg,       // if ==, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmove{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_E, EFLAGS))]>, TB;
def CMOVNE64rr: RI<0x45, MRMSrcReg,       // if !=, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovne{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_NE, EFLAGS))]>, TB;
def CMOVBE64rr: RI<0x46, MRMSrcReg,       // if <=u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovbe{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_BE, EFLAGS))]>, TB;
def CMOVA64rr : RI<0x47, MRMSrcReg,       // if >u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmova{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_A, EFLAGS))]>, TB;
def CMOVL64rr : RI<0x4C, MRMSrcReg,       // if <s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovl{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_L, EFLAGS))]>, TB;
def CMOVGE64rr: RI<0x4D, MRMSrcReg,       // if >=s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovge{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_GE, EFLAGS))]>, TB;
def CMOVLE64rr: RI<0x4E, MRMSrcReg,       // if <=s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovle{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_LE, EFLAGS))]>, TB;
def CMOVG64rr : RI<0x4F, MRMSrcReg,       // if >s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovg{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_G, EFLAGS))]>, TB;
def CMOVS64rr : RI<0x48, MRMSrcReg,       // if signed, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovs{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_S, EFLAGS))]>, TB;
def CMOVNS64rr: RI<0x49, MRMSrcReg,       // if !signed, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovns{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_NS, EFLAGS))]>, TB;
def CMOVP64rr : RI<0x4A, MRMSrcReg,       // if parity, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovp{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_P, EFLAGS))]>, TB;
def CMOVNP64rr : RI<0x4B, MRMSrcReg,       // if !parity, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovnp{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_NP, EFLAGS))]>, TB;
def CMOVO64rr : RI<0x40, MRMSrcReg,       // if overflow, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovo{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_O, EFLAGS))]>, TB;
def CMOVNO64rr : RI<0x41, MRMSrcReg,       // if !overflow, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovno{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_NO, EFLAGS))]>, TB;
} // isCommutable = 1

def CMOVB64rm : RI<0x42, MRMSrcMem,       // if <u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovb{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_B, EFLAGS))]>, TB;
def CMOVAE64rm: RI<0x43, MRMSrcMem,       // if >=u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovae{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_AE, EFLAGS))]>, TB;
def CMOVE64rm : RI<0x44, MRMSrcMem,       // if ==, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmove{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_E, EFLAGS))]>, TB;
def CMOVNE64rm: RI<0x45, MRMSrcMem,       // if !=, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovne{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_NE, EFLAGS))]>, TB;
def CMOVBE64rm: RI<0x46, MRMSrcMem,       // if <=u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovbe{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_BE, EFLAGS))]>, TB;
def CMOVA64rm : RI<0x47, MRMSrcMem,       // if >u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmova{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_A, EFLAGS))]>, TB;
def CMOVL64rm : RI<0x4C, MRMSrcMem,       // if <s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovl{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_L, EFLAGS))]>, TB;
def CMOVGE64rm: RI<0x4D, MRMSrcMem,       // if >=s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovge{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_GE, EFLAGS))]>, TB;
def CMOVLE64rm: RI<0x4E, MRMSrcMem,       // if <=s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovle{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_LE, EFLAGS))]>, TB;
def CMOVG64rm : RI<0x4F, MRMSrcMem,       // if >s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovg{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_G, EFLAGS))]>, TB;
def CMOVS64rm : RI<0x48, MRMSrcMem,       // if signed, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovs{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_S, EFLAGS))]>, TB;
def CMOVNS64rm: RI<0x49, MRMSrcMem,       // if !signed, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovns{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_NS, EFLAGS))]>, TB;
def CMOVP64rm : RI<0x4A, MRMSrcMem,       // if parity, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovp{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_P, EFLAGS))]>, TB;
def CMOVNP64rm : RI<0x4B, MRMSrcMem,       // if !parity, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovnp{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_NP, EFLAGS))]>, TB;
def CMOVO64rm : RI<0x40, MRMSrcMem,       // if overflow, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovo{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_O, EFLAGS))]>, TB;
def CMOVNO64rm : RI<0x41, MRMSrcMem,       // if !overflow, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovno{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_NO, EFLAGS))]>, TB;
} // Constraints = "$src1 = $dst"

let Uses = [EFLAGS] in {

def SETEr    : I<0x94, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "sete\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_E, EFLAGS))]>,
               TB;                        // GR8 = ==
def SETEm    : I<0x94, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "sete\t$dst",
                 [(store (X86setcc X86_COND_E, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = ==

def SETNEr   : I<0x95, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setne\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NE, EFLAGS))]>,
               TB;                        // GR8 = !=
def SETNEm   : I<0x95, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setne\t$dst",
                 [(store (X86setcc X86_COND_NE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = !=

def SETLr    : I<0x9C, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setl\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_L, EFLAGS))]>,
               TB;                        // GR8 = <  signed
def SETLm    : I<0x9C, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setl\t$dst",
                 [(store (X86setcc X86_COND_L, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <  signed

def SETGEr   : I<0x9D, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setge\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_GE, EFLAGS))]>,
               TB;                        // GR8 = >= signed
def SETGEm   : I<0x9D, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setge\t$dst",
                 [(store (X86setcc X86_COND_GE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >= signed

def SETLEr   : I<0x9E, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setle\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_LE, EFLAGS))]>,
               TB;                        // GR8 = <= signed
def SETLEm   : I<0x9E, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setle\t$dst",
                 [(store (X86setcc X86_COND_LE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <= signed

def SETGr    : I<0x9F, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setg\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_G, EFLAGS))]>,
               TB;                        // GR8 = >  signed
def SETGm    : I<0x9F, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setg\t$dst",
                 [(store (X86setcc X86_COND_G, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETBr    : I<0x92, MRM0r,
                 (outs GR8   :$dst), (ins),
                 "setb\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_B, EFLAGS))]>,
               TB;                        // GR8 = <  unsign
def SETBm    : I<0x92, MRM0m,
                 (outs), (ins i8mem:$dst),
                 "setb\t$dst",
                 [(store (X86setcc X86_COND_B, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <  unsign

def SETAEr   : I<0x93, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setae\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_AE, EFLAGS))]>,
               TB;                        // GR8 = >= unsign
def SETAEm   : I<0x93, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setae\t$dst",
                 [(store (X86setcc X86_COND_AE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >= unsign

def SETBEr   : I<0x96, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setbe\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_BE, EFLAGS))]>,
               TB;                        // GR8 = <= unsign
def SETBEm   : I<0x96, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setbe\t$dst",
                 [(store (X86setcc X86_COND_BE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <= unsign

def SETAr    : I<0x97, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "seta\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_A, EFLAGS))]>,
               TB;                        // GR8 = >  signed
def SETAm    : I<0x97, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "seta\t$dst",
                 [(store (X86setcc X86_COND_A, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETSr    : I<0x98, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "sets\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_S, EFLAGS))]>,
               TB;                        // GR8 = <sign bit>
def SETSm    : I<0x98, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "sets\t$dst",
                 [(store (X86setcc X86_COND_S, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <sign bit>
def SETNSr   : I<0x99, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setns\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NS, EFLAGS))]>,
               TB;                        // GR8 = !<sign bit>
def SETNSm   : I<0x99, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setns\t$dst",
                 [(store (X86setcc X86_COND_NS, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = !<sign bit>

def SETPr    : I<0x9A, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setp\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_P, EFLAGS))]>,
               TB;                        // GR8 = parity
def SETPm    : I<0x9A, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setp\t$dst",
                 [(store (X86setcc X86_COND_P, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = parity
def SETNPr   : I<0x9B, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setnp\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NP, EFLAGS))]>,
               TB;                        // GR8 = not parity
def SETNPm   : I<0x9B, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setnp\t$dst",
                 [(store (X86setcc X86_COND_NP, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = not parity

def SETOr    : I<0x90, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "seto\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_O, EFLAGS))]>,
               TB;                        // GR8 = overflow
def SETOm    : I<0x90, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "seto\t$dst",
                 [(store (X86setcc X86_COND_O, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = overflow
def SETNOr   : I<0x91, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setno\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NO, EFLAGS))]>,
               TB;                        // GR8 = not overflow
def SETNOm   : I<0x91, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setno\t$dst",
                 [(store (X86setcc X86_COND_NO, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = not overflow
} // Uses = [EFLAGS]

