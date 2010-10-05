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


// SetCC instructions.
multiclass CMOV<bits<8> opc, string Mnemonic, PatLeaf CondNode> {
  let Uses = [EFLAGS], Predicates = [HasCMov], Constraints = "$src1 = $dst",
      isCommutable = 1 in {
    def #NAME#16rr
      : I<opc, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
          !strconcat(Mnemonic, "{w}\t{$src2, $dst|$dst, $src2}"),
          [(set GR16:$dst,
                (X86cmov GR16:$src1, GR16:$src2, CondNode, EFLAGS))]>,TB,OpSize;
    def #NAME#32rr
      : I<opc, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
          !strconcat(Mnemonic, "{l}\t{$src2, $dst|$dst, $src2}"),
          [(set GR32:$dst,
                (X86cmov GR32:$src1, GR32:$src2, CondNode, EFLAGS))]>, TB;
    def #NAME#64rr
      :RI<opc, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
          !strconcat(Mnemonic, "{q}\t{$src2, $dst|$dst, $src2}"),
          [(set GR64:$dst,
                (X86cmov GR64:$src1, GR64:$src2, CondNode, EFLAGS))]>, TB;
  }

  let Uses = [EFLAGS], Predicates = [HasCMov], Constraints = "$src1 = $dst"in {
    def #NAME#16rm
      : I<opc, MRMSrcMem, (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
          !strconcat(Mnemonic, "{w}\t{$src2, $dst|$dst, $src2}"),
          [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    CondNode, EFLAGS))]>, TB, OpSize;
    def #NAME#32rm
      : I<opc, MRMSrcMem, (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
          !strconcat(Mnemonic, "{l}\t{$src2, $dst|$dst, $src2}"),
          [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    CondNode, EFLAGS))]>, TB;
    def #NAME#64rm
      :RI<opc, MRMSrcMem, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
          !strconcat(Mnemonic, "{q}\t{$src2, $dst|$dst, $src2}"),
          [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    CondNode, EFLAGS))]>, TB;
  } // Uses = [EFLAGS], Predicates = [HasCMov], Constraints = "$src1 = $dst"
} // end multiclass


// Conditional Moves.
defm CMOVBE : CMOV<0x46, "cmovbe", X86_COND_BE>;


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


// SetCC instructions.
multiclass SETCC<bits<8> opc, string Mnemonic, PatLeaf OpNode> {
  let Uses = [EFLAGS] in {
    def r    : I<opc, MRM0r,  (outs GR8:$dst), (ins),
                     !strconcat(Mnemonic, "\t$dst"),
                     [(set GR8:$dst, (X86setcc OpNode, EFLAGS))]>, TB;
    def m    : I<opc, MRM0m,  (outs), (ins i8mem:$dst),
                     !strconcat(Mnemonic, "\t$dst"),
                     [(store (X86setcc OpNode, EFLAGS), addr:$dst)]>, TB;
  } // Uses = [EFLAGS]
}

defm SETO  : SETCC<0x90, "seto",  X86_COND_O>;   // is overflow bit set
defm SETNO : SETCC<0x91, "setno", X86_COND_NO>;  // is overflow bit not set
defm SETB  : SETCC<0x92, "setb",  X86_COND_B>;   // unsigned less than
defm SETAE : SETCC<0x93, "setae", X86_COND_AE>;  // unsigned greater or equal
defm SETE  : SETCC<0x94, "sete",  X86_COND_E>;   // equal to
defm SETNE : SETCC<0x95, "setne", X86_COND_NE>;  // not equal to
defm SETBE : SETCC<0x96, "setbe", X86_COND_BE>;  // unsigned less than or equal
defm SETA  : SETCC<0x97, "seta",  X86_COND_A>;   // unsigned greater than
defm SETS  : SETCC<0x98, "sets",  X86_COND_S>;   // is signed bit set
defm SETNS : SETCC<0x99, "setns", X86_COND_NS>;  // is not signed
defm SETP  : SETCC<0x9A, "setp",  X86_COND_P>;   // is parity bit set
defm SETNP : SETCC<0x9B, "setnp", X86_COND_NP>;  // is parity bit not set
defm SETL  : SETCC<0x9C, "setl",  X86_COND_L>;   // signed less than
defm SETGE : SETCC<0x9D, "setge", X86_COND_GE>;  // signed greater or equal
defm SETLE : SETCC<0x9E, "setle", X86_COND_LE>;  // signed less than or equal
defm SETG  : SETCC<0x9F, "setg",  X86_COND_G>;   // signed greater than

