//===-- X86InstrCMovSetCC.td - Conditional Move and SetCC --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 conditional move and set on condition
// instructions.
//
//===----------------------------------------------------------------------===//


// CMOV instructions.
let isCodeGenOnly = 1, ForceDisassemble = 1 in {
let Uses = [EFLAGS], Predicates = [HasCMov], Constraints = "$src1 = $dst",
    isCommutable = 1, SchedRW = [WriteCMOV] in {
  def CMOV16rr
    : I<0x40, MRMSrcRegCC, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2, ccode:$cond),
        "cmov${cond}{w}\t{$src2, $dst|$dst, $src2}",
        [(set GR16:$dst,
              (X86cmov GR16:$src1, GR16:$src2, imm:$cond, EFLAGS))]>,
              TB, OpSize16;
  def CMOV32rr
    : I<0x40, MRMSrcRegCC, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2, ccode:$cond),
        "cmov${cond}{l}\t{$src2, $dst|$dst, $src2}",
        [(set GR32:$dst,
              (X86cmov GR32:$src1, GR32:$src2, imm:$cond, EFLAGS))]>,
              TB, OpSize32;
  def CMOV64rr
    :RI<0x40, MRMSrcRegCC, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2, ccode:$cond),
        "cmov${cond}{q}\t{$src2, $dst|$dst, $src2}",
        [(set GR64:$dst,
              (X86cmov GR64:$src1, GR64:$src2, imm:$cond, EFLAGS))]>, TB;
}

let Uses = [EFLAGS], Predicates = [HasCMov], Constraints = "$src1 = $dst",
    SchedRW = [WriteCMOV.Folded, WriteCMOV.ReadAfterFold] in {
  def CMOV16rm
    : I<0x40, MRMSrcMemCC, (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2, ccode:$cond),
        "cmov${cond}{w}\t{$src2, $dst|$dst, $src2}",
        [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                  imm:$cond, EFLAGS))]>, TB, OpSize16;
  def CMOV32rm
    : I<0x40, MRMSrcMemCC, (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2, ccode:$cond),
        "cmov${cond}{l}\t{$src2, $dst|$dst, $src2}",
        [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                  imm:$cond, EFLAGS))]>, TB, OpSize32;
  def CMOV64rm
    :RI<0x40, MRMSrcMemCC, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2, ccode:$cond),
        "cmov${cond}{q}\t{$src2, $dst|$dst, $src2}",
        [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                  imm:$cond, EFLAGS))]>, TB;
} // Uses = [EFLAGS], Predicates = [HasCMov], Constraints = "$src1 = $dst"
} // isCodeGenOnly = 1, ForceDisassemble = 1

multiclass CMOV_Aliases<string Name, int CC> {
  def : InstAlias<Name#"{w}\t{$src, $dst|$dst, $src}",
                  (CMOV16rr GR16:$dst, GR16:$src, CC), 0>;
  def : InstAlias<Name#"{w}\t{$src, $dst|$dst, $src}",
                  (CMOV16rm GR16:$dst, i16mem:$src, CC), 0>;
  def : InstAlias<Name#"{l}\t{$src, $dst|$dst, $src}",
                  (CMOV32rr GR32:$dst, GR32:$src, CC), 0>;
  def : InstAlias<Name#"{l}\t{$src, $dst|$dst, $src}",
                  (CMOV32rm GR32:$dst, i32mem:$src, CC), 0>;
  def : InstAlias<Name#"{q}\t{$src, $dst|$dst, $src}",
                  (CMOV64rr GR64:$dst, GR64:$src, CC), 0>;
  def : InstAlias<Name#"{q}\t{$src, $dst|$dst, $src}",
                  (CMOV64rm GR64:$dst, i64mem:$src, CC), 0>;
}

defm : CMOV_Aliases<"cmovo" ,  0>;
defm : CMOV_Aliases<"cmovno",  1>;
defm : CMOV_Aliases<"cmovb" ,  2>;
defm : CMOV_Aliases<"cmovae",  3>;
defm : CMOV_Aliases<"cmove" ,  4>;
defm : CMOV_Aliases<"cmovne",  5>;
defm : CMOV_Aliases<"cmovbe",  6>;
defm : CMOV_Aliases<"cmova" ,  7>;
defm : CMOV_Aliases<"cmovs" ,  8>;
defm : CMOV_Aliases<"cmovns",  9>;
defm : CMOV_Aliases<"cmovp" , 10>;
defm : CMOV_Aliases<"cmovnp", 11>;
defm : CMOV_Aliases<"cmovl" , 12>;
defm : CMOV_Aliases<"cmovge", 13>;
defm : CMOV_Aliases<"cmovle", 14>;
defm : CMOV_Aliases<"cmovg" , 15>;


// SetCC instructions.
multiclass SETCC<bits<8> opc, string Mnemonic, PatLeaf OpNode> {
  let Uses = [EFLAGS] in {
    def r    : I<opc, MRMXr,  (outs GR8:$dst), (ins),
                     !strconcat(Mnemonic, "\t$dst"),
                     [(set GR8:$dst, (X86setcc OpNode, EFLAGS))]>,
                     TB, Sched<[WriteSETCC]>;
    def m    : I<opc, MRMXm,  (outs), (ins i8mem:$dst),
                     !strconcat(Mnemonic, "\t$dst"),
                     [(store (X86setcc OpNode, EFLAGS), addr:$dst)]>,
                     TB, Sched<[WriteSETCCStore]>;
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

// SALC is an undocumented instruction. Information for this instruction can be found
// here http://www.rcollins.org/secrets/opcodes/SALC.html
// Set AL if carry. 
let Uses = [EFLAGS], Defs = [AL], SchedRW = [WriteALU] in {
  def SALC : I<0xD6, RawFrm, (outs), (ins), "salc", []>, Requires<[Not64BitMode]>;
}
