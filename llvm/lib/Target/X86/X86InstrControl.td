//===- X86InstrControl.td - Control Flow Instructions ------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 jump, return, call, and related instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

// Return instructions.
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1, FPForm = SpecialFP in {
  def RET    : I   <0xC3, RawFrm, (outs), (ins variable_ops),
                    "ret",
                    [(X86retflag 0)]>;
  def RETI   : Ii16<0xC2, RawFrm, (outs), (ins i16imm:$amt, variable_ops),
                    "ret\t$amt",
                    [(X86retflag timm:$amt)]>;
  def RETIW  : Ii16<0xC2, RawFrm, (outs), (ins i16imm:$amt, variable_ops),
                    "retw\t$amt",
                    [(X86retflag timm:$amt)]>, OpSize;
  def LRET   : I   <0xCB, RawFrm, (outs), (ins),
                    "lret", []>;
  def LRETI  : Ii16<0xCA, RawFrm, (outs), (ins i16imm:$amt),
                    "lret\t$amt", []>;
  def LRETIW : Ii16<0xCA, RawFrm, (outs), (ins i16imm:$amt),
                    "lretw\t$amt", []>, OpSize;
}

// Unconditional branches.
let isBarrier = 1, isBranch = 1, isTerminator = 1 in {
  def JMP_4 : Ii32PCRel<0xE9, RawFrm, (outs), (ins brtarget:$dst),
                        "jmp\t$dst", [(br bb:$dst)]>;
  def JMP_1 : Ii8PCRel<0xEB, RawFrm, (outs), (ins brtarget8:$dst),
                       "jmp\t$dst", []>;
  def JMP64pcrel32 : I<0xE9, RawFrm, (outs), (ins brtarget:$dst), 
                       "jmp{q}\t$dst", []>;
}

// Conditional Branches.
let isBranch = 1, isTerminator = 1, Uses = [EFLAGS] in {
  multiclass ICBr<bits<8> opc1, bits<8> opc4, string asm, PatFrag Cond> {
    def _1 : Ii8PCRel <opc1, RawFrm, (outs), (ins brtarget8:$dst), asm, []>;
    def _4 : Ii32PCRel<opc4, RawFrm, (outs), (ins brtarget:$dst), asm,
                       [(X86brcond bb:$dst, Cond, EFLAGS)]>, TB;
  }
}

defm JO  : ICBr<0x70, 0x80, "jo\t$dst" , X86_COND_O>;
defm JNO : ICBr<0x71, 0x81, "jno\t$dst" , X86_COND_NO>;
defm JB  : ICBr<0x72, 0x82, "jb\t$dst" , X86_COND_B>;
defm JAE : ICBr<0x73, 0x83, "jae\t$dst", X86_COND_AE>;
defm JE  : ICBr<0x74, 0x84, "je\t$dst" , X86_COND_E>;
defm JNE : ICBr<0x75, 0x85, "jne\t$dst", X86_COND_NE>;
defm JBE : ICBr<0x76, 0x86, "jbe\t$dst", X86_COND_BE>;
defm JA  : ICBr<0x77, 0x87, "ja\t$dst" , X86_COND_A>;
defm JS  : ICBr<0x78, 0x88, "js\t$dst" , X86_COND_S>;
defm JNS : ICBr<0x79, 0x89, "jns\t$dst", X86_COND_NS>;
defm JP  : ICBr<0x7A, 0x8A, "jp\t$dst" , X86_COND_P>;
defm JNP : ICBr<0x7B, 0x8B, "jnp\t$dst", X86_COND_NP>;
defm JL  : ICBr<0x7C, 0x8C, "jl\t$dst" , X86_COND_L>;
defm JGE : ICBr<0x7D, 0x8D, "jge\t$dst", X86_COND_GE>;
defm JLE : ICBr<0x7E, 0x8E, "jle\t$dst", X86_COND_LE>;
defm JG  : ICBr<0x7F, 0x8F, "jg\t$dst" , X86_COND_G>;

// jcx/jecx/jrcx instructions.
let isAsmParserOnly = 1, isBranch = 1, isTerminator = 1 in {
  // These are the 32-bit versions of this instruction for the asmparser.  In
  // 32-bit mode, the address size prefix is jcxz and the unprefixed version is
  // jecxz.
  let Uses = [CX] in
    def JCXZ : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                        "jcxz\t$dst", []>, AdSize, Requires<[In32BitMode]>;
  let Uses = [ECX] in
    def JECXZ_32 : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                           "jecxz\t$dst", []>, Requires<[In32BitMode]>;

  // J*CXZ instruction: 64-bit versions of this instruction for the asmparser.
  // In 64-bit mode, the address size prefix is jecxz and the unprefixed version
  // is jrcxz.
  let Uses = [ECX] in
    def JECXZ_64 : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                            "jecxz\t$dst", []>, AdSize, Requires<[In64BitMode]>;
  let Uses = [RCX] in
    def JRCXZ : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                           "jrcxz\t$dst", []>, Requires<[In64BitMode]>;
}

// Indirect branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  def JMP32r     : I<0xFF, MRM4r, (outs), (ins GR32:$dst), "jmp{l}\t{*}$dst",
                     [(brind GR32:$dst)]>, Requires<[In32BitMode]>;
  def JMP32m     : I<0xFF, MRM4m, (outs), (ins i32mem:$dst), "jmp{l}\t{*}$dst",
                     [(brind (loadi32 addr:$dst))]>, Requires<[In32BitMode]>;

  def JMP64r     : I<0xFF, MRM4r, (outs), (ins GR64:$dst), "jmp{q}\t{*}$dst",
                     [(brind GR64:$dst)]>, Requires<[In64BitMode]>;
  def JMP64m     : I<0xFF, MRM4m, (outs), (ins i64mem:$dst), "jmp{q}\t{*}$dst",
                     [(brind (loadi64 addr:$dst))]>, Requires<[In64BitMode]>;

  def FARJMP16i  : Iseg16<0xEA, RawFrmImm16, (outs), 
                          (ins i16imm:$off, i16imm:$seg),
                          "ljmp{w}\t{$seg, $off|$off, $seg}", []>, OpSize;
  def FARJMP32i  : Iseg32<0xEA, RawFrmImm16, (outs),
                          (ins i32imm:$off, i16imm:$seg),
                          "ljmp{l}\t{$seg, $off|$off, $seg}", []>;                     
  def FARJMP64   : RI<0xFF, MRM5m, (outs), (ins opaque80mem:$dst),
                      "ljmp{q}\t{*}$dst", []>;

  def FARJMP16m  : I<0xFF, MRM5m, (outs), (ins opaque32mem:$dst), 
                     "ljmp{w}\t{*}$dst", []>, OpSize;
  def FARJMP32m  : I<0xFF, MRM5m, (outs), (ins opaque48mem:$dst),
                     "ljmp{l}\t{*}$dst", []>;
}


// Loop instructions

def LOOP   : Ii8PCRel<0xE2, RawFrm, (outs), (ins brtarget8:$dst), "loop\t$dst", []>;
def LOOPE  : Ii8PCRel<0xE1, RawFrm, (outs), (ins brtarget8:$dst), "loope\t$dst", []>;
def LOOPNE : Ii8PCRel<0xE0, RawFrm, (outs), (ins brtarget8:$dst), "loopne\t$dst", []>;

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers. ESP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
      Uses = [ESP] in {
    def CALLpcrel32 : Ii32PCRel<0xE8, RawFrm,
                           (outs), (ins i32imm_pcrel:$dst,variable_ops),
                           "call{l}\t$dst", []>, Requires<[In32BitMode]>;
    def CALL32r     : I<0xFF, MRM2r, (outs), (ins GR32:$dst, variable_ops),
                        "call{l}\t{*}$dst", [(X86call GR32:$dst)]>,
                         Requires<[In32BitMode]>;
    def CALL32m     : I<0xFF, MRM2m, (outs), (ins i32mem:$dst, variable_ops),
                        "call{l}\t{*}$dst", [(X86call (loadi32 addr:$dst))]>,
                        Requires<[In32BitMode]>;
  
    def FARCALL16i  : Iseg16<0x9A, RawFrmImm16, (outs), 
                             (ins i16imm:$off, i16imm:$seg),
                             "lcall{w}\t{$seg, $off|$off, $seg}", []>, OpSize;
    def FARCALL32i  : Iseg32<0x9A, RawFrmImm16, (outs),
                             (ins i32imm:$off, i16imm:$seg),
                             "lcall{l}\t{$seg, $off|$off, $seg}", []>;
                             
    def FARCALL16m  : I<0xFF, MRM3m, (outs), (ins opaque32mem:$dst),
                        "lcall{w}\t{*}$dst", []>, OpSize;
    def FARCALL32m  : I<0xFF, MRM3m, (outs), (ins opaque48mem:$dst),
                        "lcall{l}\t{*}$dst", []>;

    // callw for 16 bit code for the assembler.
    let isAsmParserOnly = 1 in
      def CALLpcrel16 : Ii16PCRel<0xE8, RawFrm,
                       (outs), (ins i16imm_pcrel:$dst, variable_ops),
                       "callw\t$dst", []>, OpSize;
  }


// Tail call stuff.

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1,
    isCodeGenOnly = 1 in
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
      Uses = [ESP] in {
  def TCRETURNdi : I<0, Pseudo, (outs), 
                     (ins i32imm_pcrel:$dst, i32imm:$offset, variable_ops),
                   "#TC_RETURN $dst $offset", []>;
  def TCRETURNri : I<0, Pseudo, (outs), 
                     (ins GR32_TC:$dst, i32imm:$offset, variable_ops),
                     "#TC_RETURN $dst $offset", []>;
  let mayLoad = 1 in
  def TCRETURNmi : I<0, Pseudo, (outs), 
                     (ins i32mem_TC:$dst, i32imm:$offset, variable_ops),
                     "#TC_RETURN $dst $offset", []>;

  // FIXME: The should be pseudo instructions that are lowered when going to
  // mcinst.
  def TAILJMPd : Ii32PCRel<0xE9, RawFrm, (outs),
                           (ins i32imm_pcrel:$dst, variable_ops),
                 "jmp\t$dst  # TAILCALL",
                 []>;
  def TAILJMPr : I<0xFF, MRM4r, (outs), (ins GR32_TC:$dst, variable_ops), 
                   "", []>;  // FIXME: Remove encoding when JIT is dead.
  let mayLoad = 1 in
  def TAILJMPm : I<0xFF, MRM4m, (outs), (ins i32mem_TC:$dst, variable_ops),
                   "jmp{l}\t{*}$dst  # TAILCALL", []>;
}


//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers. RSP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
              FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
      Uses = [RSP] in {
      
    // NOTE: this pattern doesn't match "X86call imm", because we do not know
    // that the offset between an arbitrary immediate and the call will fit in
    // the 32-bit pcrel field that we have.
    def CALL64pcrel32 : Ii32PCRel<0xE8, RawFrm,
                          (outs), (ins i64i32imm_pcrel:$dst, variable_ops),
                          "call{q}\t$dst", []>,
                        Requires<[In64BitMode, NotWin64]>;
    def CALL64r       : I<0xFF, MRM2r, (outs), (ins GR64:$dst, variable_ops),
                          "call{q}\t{*}$dst", [(X86call GR64:$dst)]>,
                        Requires<[In64BitMode, NotWin64]>;
    def CALL64m       : I<0xFF, MRM2m, (outs), (ins i64mem:$dst, variable_ops),
                          "call{q}\t{*}$dst", [(X86call (loadi64 addr:$dst))]>,
                        Requires<[In64BitMode, NotWin64]>;
                        
    def FARCALL64   : RI<0xFF, MRM3m, (outs), (ins opaque80mem:$dst),
                         "lcall{q}\t{*}$dst", []>;
  }

  // FIXME: We need to teach codegen about single list of call-clobbered 
  // registers.
let isCall = 1, isCodeGenOnly = 1 in
  // All calls clobber the non-callee saved registers. RSP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [RAX, RCX, RDX, R8, R9, R10, R11,
              FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, EFLAGS],
      Uses = [RSP] in {
    def WINCALL64pcrel32 : Ii32PCRel<0xE8, RawFrm,
                             (outs), (ins i64i32imm_pcrel:$dst, variable_ops),
                             "call{q}\t$dst", []>,
                           Requires<[IsWin64]>;
    def WINCALL64r       : I<0xFF, MRM2r, (outs), (ins GR64:$dst, variable_ops),
                             "call{q}\t{*}$dst",
                             [(X86call GR64:$dst)]>, Requires<[IsWin64]>;
    def WINCALL64m       : I<0xFF, MRM2m, (outs), (ins i64mem:$dst,variable_ops),
                             "call{q}\t{*}$dst",
                             [(X86call (loadi64 addr:$dst))]>, 
                           Requires<[IsWin64]>;
  }


let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1,
    isCodeGenOnly = 1 in
  let Defs = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
              FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
      Uses = [RSP] in {
  def TCRETURNdi64 : I<0, Pseudo, (outs),
                         (ins i64i32imm_pcrel:$dst, i32imm:$offset, variable_ops),
                       "#TC_RETURN $dst $offset", []>;
  def TCRETURNri64 : I<0, Pseudo, (outs), (ins GR64_TC:$dst, i32imm:$offset,
                                           variable_ops),
                       "#TC_RETURN $dst $offset", []>;
  let mayLoad = 1 in
  def TCRETURNmi64 : I<0, Pseudo, (outs), 
                       (ins i64mem_TC:$dst, i32imm:$offset, variable_ops),
                       "#TC_RETURN $dst $offset", []>;

  def TAILJMPd64 : Ii32PCRel<0xE9, RawFrm, (outs),
                                      (ins i64i32imm_pcrel:$dst, variable_ops),
                   "jmp\t$dst  # TAILCALL", []>;
  def TAILJMPr64 : I<0xFF, MRM4r, (outs), (ins GR64_TC:$dst, variable_ops),
                     "jmp{q}\t{*}$dst  # TAILCALL", []>;

  let mayLoad = 1 in
  def TAILJMPm64 : I<0xFF, MRM4m, (outs), (ins i64mem_TC:$dst, variable_ops),
                     "jmp{q}\t{*}$dst  # TAILCALL", []>;
}

