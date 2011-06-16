//===- X86InstrCompiler.td - Compiler Pseudos and Patterns -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the various pseudo instructions used by the compiler,
// as well as Pat patterns used during instruction selection.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pattern Matching Support

def GetLo32XForm : SDNodeXForm<imm, [{
  // Transformation function: get the low 32 bits.
  return getI32Imm((unsigned)N->getZExtValue());
}]>;

def GetLo8XForm : SDNodeXForm<imm, [{
  // Transformation function: get the low 8 bits.
  return getI8Imm((uint8_t)N->getZExtValue());
}]>;


//===----------------------------------------------------------------------===//
// Random Pseudo Instructions.

// PIC base construction.  This expands to code that looks like this:
//     call  $next_inst
//     popl %destreg"
let neverHasSideEffects = 1, isNotDuplicable = 1, Uses = [ESP] in
  def MOVPC32r : Ii32<0xE8, Pseudo, (outs GR32:$reg), (ins i32imm:$label),
                      "", []>;


// ADJCALLSTACKDOWN/UP implicitly use/def ESP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
// Pessimistically assume ADJCALLSTACKDOWN / ADJCALLSTACKUP will become
// sub / add which can clobber EFLAGS.
let Defs = [ESP, EFLAGS], Uses = [ESP] in {
def ADJCALLSTACKDOWN32 : I<0, Pseudo, (outs), (ins i32imm:$amt),
                           "#ADJCALLSTACKDOWN",
                           [(X86callseq_start timm:$amt)]>,
                          Requires<[In32BitMode]>;
def ADJCALLSTACKUP32   : I<0, Pseudo, (outs), (ins i32imm:$amt1, i32imm:$amt2),
                           "#ADJCALLSTACKUP",
                           [(X86callseq_end timm:$amt1, timm:$amt2)]>,
                          Requires<[In32BitMode]>;
}

// ADJCALLSTACKDOWN/UP implicitly use/def RSP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
// Pessimistically assume ADJCALLSTACKDOWN / ADJCALLSTACKUP will become
// sub / add which can clobber EFLAGS.
let Defs = [RSP, EFLAGS], Uses = [RSP] in {
def ADJCALLSTACKDOWN64 : I<0, Pseudo, (outs), (ins i32imm:$amt),
                           "#ADJCALLSTACKDOWN",
                           [(X86callseq_start timm:$amt)]>,
                          Requires<[In64BitMode]>;
def ADJCALLSTACKUP64   : I<0, Pseudo, (outs), (ins i32imm:$amt1, i32imm:$amt2),
                           "#ADJCALLSTACKUP",
                           [(X86callseq_end timm:$amt1, timm:$amt2)]>,
                          Requires<[In64BitMode]>;
}



// x86-64 va_start lowering magic.
let usesCustomInserter = 1 in {
def VASTART_SAVE_XMM_REGS : I<0, Pseudo,
                              (outs),
                              (ins GR8:$al,
                                   i64imm:$regsavefi, i64imm:$offset,
                                   variable_ops),
                              "#VASTART_SAVE_XMM_REGS $al, $regsavefi, $offset",
                              [(X86vastart_save_xmm_regs GR8:$al,
                                                         imm:$regsavefi,
                                                         imm:$offset)]>;

// The VAARG_64 pseudo-instruction takes the address of the va_list,
// and places the address of the next argument into a register.
let Defs = [EFLAGS] in
def VAARG_64 : I<0, Pseudo,
                 (outs GR64:$dst),
                 (ins i8mem:$ap, i32imm:$size, i8imm:$mode, i32imm:$align),
                 "#VAARG_64 $dst, $ap, $size, $mode, $align",
                 [(set GR64:$dst,
                    (X86vaarg64 addr:$ap, imm:$size, imm:$mode, imm:$align)),
                  (implicit EFLAGS)]>;

// Dynamic stack allocation yields a _chkstk or _alloca call for all Windows
// targets.  These calls are needed to probe the stack when allocating more than
// 4k bytes in one go. Touching the stack at 4K increments is necessary to
// ensure that the guard pages used by the OS virtual memory manager are
// allocated in correct sequence.
// The main point of having separate instruction are extra unmodelled effects
// (compared to ordinary calls) like stack pointer change.

let Defs = [EAX, ESP, EFLAGS], Uses = [ESP] in
  def WIN_ALLOCA : I<0, Pseudo, (outs), (ins),
                     "# dynamic stack allocation",
                     [(X86WinAlloca)]>;
}



//===----------------------------------------------------------------------===//
// EH Pseudo Instructions
//
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1, isCodeGenOnly = 1 in {
def EH_RETURN   : I<0xC3, RawFrm, (outs), (ins GR32:$addr),
                    "ret\t#eh_return, addr: $addr",
                    [(X86ehret GR32:$addr)]>;

}

let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1, isCodeGenOnly = 1 in {
def EH_RETURN64   : I<0xC3, RawFrm, (outs), (ins GR64:$addr),
                     "ret\t#eh_return, addr: $addr",
                     [(X86ehret GR64:$addr)]>;

}

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map movr0 to xor.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
// FIXME: Set encoding to pseudo.
let Defs = [EFLAGS], isReMaterializable = 1, isAsCheapAsAMove = 1,
    isCodeGenOnly = 1 in {
def MOV8r0   : I<0x30, MRMInitReg, (outs GR8 :$dst), (ins), "",
                 [(set GR8:$dst, 0)]>;

// We want to rewrite MOV16r0 in terms of MOV32r0, because it's a smaller
// encoding and avoids a partial-register update sometimes, but doing so
// at isel time interferes with rematerialization in the current register
// allocator. For now, this is rewritten when the instruction is lowered
// to an MCInst.
def MOV16r0   : I<0x31, MRMInitReg, (outs GR16:$dst), (ins),
                 "",
                 [(set GR16:$dst, 0)]>, OpSize;

// FIXME: Set encoding to pseudo.
def MOV32r0  : I<0x31, MRMInitReg, (outs GR32:$dst), (ins), "",
                 [(set GR32:$dst, 0)]>;
}

// We want to rewrite MOV64r0 in terms of MOV32r0, because it's sometimes a
// smaller encoding, but doing so at isel time interferes with rematerialization
// in the current register allocator. For now, this is rewritten when the
// instruction is lowered to an MCInst.
// FIXME: AddedComplexity gives this a higher priority than MOV64ri32. Remove
// when we have a better way to specify isel priority.
let Defs = [EFLAGS], isCodeGenOnly=1,
    AddedComplexity = 1, isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOV64r0   : I<0x31, MRMInitReg, (outs GR64:$dst), (ins), "",
                 [(set GR64:$dst, 0)]>;

// Materialize i64 constant where top 32-bits are zero. This could theoretically
// use MOV32ri with a SUBREG_TO_REG to represent the zero-extension, however
// that would make it more difficult to rematerialize.
let AddedComplexity = 1, isReMaterializable = 1, isAsCheapAsAMove = 1,
    isCodeGenOnly = 1 in
def MOV64ri64i32 : Ii32<0xB8, AddRegFrm, (outs GR64:$dst), (ins i64i32imm:$src),
                        "", [(set GR64:$dst, i64immZExt32:$src)]>;

// Use sbb to materialize carry bit.
let Uses = [EFLAGS], Defs = [EFLAGS], isCodeGenOnly = 1 in {
// FIXME: These are pseudo ops that should be replaced with Pat<> patterns.
// However, Pat<> can't replicate the destination reg into the inputs of the
// result.
// FIXME: Change these to have encoding Pseudo when X86MCCodeEmitter replaces
// X86CodeEmitter.
def SETB_C8r : I<0x18, MRMInitReg, (outs GR8:$dst), (ins), "",
                 [(set GR8:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>;
def SETB_C16r : I<0x19, MRMInitReg, (outs GR16:$dst), (ins), "",
                 [(set GR16:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>,
                OpSize;
def SETB_C32r : I<0x19, MRMInitReg, (outs GR32:$dst), (ins), "",
                 [(set GR32:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>;
def SETB_C64r : RI<0x19, MRMInitReg, (outs GR64:$dst), (ins), "",
                 [(set GR64:$dst, (X86setcc_c X86_COND_B, EFLAGS))]>;
} // isCodeGenOnly


def : Pat<(i16 (anyext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C16r)>;
def : Pat<(i32 (anyext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C32r)>;
def : Pat<(i64 (anyext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C64r)>;

def : Pat<(i16 (sext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C16r)>;
def : Pat<(i32 (sext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C32r)>;
def : Pat<(i64 (sext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C64r)>;

// We canonicalize 'setb' to "(and (sbb reg,reg), 1)" on the hope that the and
// will be eliminated and that the sbb can be extended up to a wider type.  When
// this happens, it is great.  However, if we are left with an 8-bit sbb and an
// and, we might as well just match it as a setb.
def : Pat<(and (i8 (X86setcc_c X86_COND_B, EFLAGS)), 1),
          (SETBr)>;

// (add OP, SETB) -> (adc OP, 0)
def : Pat<(add (and (i8 (X86setcc_c X86_COND_B, EFLAGS)), 1), GR8:$op),
          (ADC8ri GR8:$op, 0)>;
def : Pat<(add (and (i32 (X86setcc_c X86_COND_B, EFLAGS)), 1), GR32:$op),
          (ADC32ri8 GR32:$op, 0)>;
def : Pat<(add (and (i64 (X86setcc_c X86_COND_B, EFLAGS)), 1), GR64:$op),
          (ADC64ri8 GR64:$op, 0)>;

// (sub OP, SETB) -> (sbb OP, 0)
def : Pat<(sub GR8:$op, (and (i8 (X86setcc_c X86_COND_B, EFLAGS)), 1)),
          (SBB8ri GR8:$op, 0)>;
def : Pat<(sub GR32:$op, (and (i32 (X86setcc_c X86_COND_B, EFLAGS)), 1)),
          (SBB32ri8 GR32:$op, 0)>;
def : Pat<(sub GR64:$op, (and (i64 (X86setcc_c X86_COND_B, EFLAGS)), 1)),
          (SBB64ri8 GR64:$op, 0)>;

// (sub OP, SETCC_CARRY) -> (adc OP, 0)
def : Pat<(sub GR8:$op, (i8 (X86setcc_c X86_COND_B, EFLAGS))),
          (ADC8ri GR8:$op, 0)>;
def : Pat<(sub GR32:$op, (i32 (X86setcc_c X86_COND_B, EFLAGS))),
          (ADC32ri8 GR32:$op, 0)>;
def : Pat<(sub GR64:$op, (i64 (X86setcc_c X86_COND_B, EFLAGS))),
          (ADC64ri8 GR64:$op, 0)>;

//===----------------------------------------------------------------------===//
// String Pseudo Instructions
//
let Defs = [ECX,EDI,ESI], Uses = [ECX,EDI,ESI], isCodeGenOnly = 1 in {
def REP_MOVSB : I<0xA4, RawFrm, (outs), (ins), "{rep;movsb|rep movsb}",
                  [(X86rep_movs i8)]>, REP;
def REP_MOVSW : I<0xA5, RawFrm, (outs), (ins), "{rep;movsw|rep movsw}",
                  [(X86rep_movs i16)]>, REP, OpSize;
def REP_MOVSD : I<0xA5, RawFrm, (outs), (ins), "{rep;movsl|rep movsd}",
                  [(X86rep_movs i32)]>, REP;
}

let Defs = [RCX,RDI,RSI], Uses = [RCX,RDI,RSI], isCodeGenOnly = 1 in
def REP_MOVSQ : RI<0xA5, RawFrm, (outs), (ins), "{rep;movsq|rep movsq}",
                   [(X86rep_movs i64)]>, REP;


// FIXME: Should use "(X86rep_stos AL)" as the pattern.
let Defs = [ECX,EDI], Uses = [AL,ECX,EDI], isCodeGenOnly = 1 in
def REP_STOSB : I<0xAA, RawFrm, (outs), (ins), "{rep;stosb|rep stosb}",
                  [(X86rep_stos i8)]>, REP;
let Defs = [ECX,EDI], Uses = [AX,ECX,EDI], isCodeGenOnly = 1 in
def REP_STOSW : I<0xAB, RawFrm, (outs), (ins), "{rep;stosw|rep stosw}",
                  [(X86rep_stos i16)]>, REP, OpSize;
let Defs = [ECX,EDI], Uses = [EAX,ECX,EDI], isCodeGenOnly = 1 in
def REP_STOSD : I<0xAB, RawFrm, (outs), (ins), "{rep;stosl|rep stosd}",
                  [(X86rep_stos i32)]>, REP;

let Defs = [RCX,RDI], Uses = [RAX,RCX,RDI], isCodeGenOnly = 1 in
def REP_STOSQ : RI<0xAB, RawFrm, (outs), (ins), "{rep;stosq|rep stosq}",
                   [(X86rep_stos i64)]>, REP;


//===----------------------------------------------------------------------===//
// Thread Local Storage Instructions
//

// ELF TLS Support
// All calls clobber the non-callee saved registers. ESP is marked as
// a use to prevent stack-pointer assignments that appear immediately
// before calls from potentially appearing dead.
let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
            MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
    Uses = [ESP] in
def TLS_addr32 : I<0, Pseudo, (outs), (ins i32mem:$sym),
                  "# TLS_addr32",
                  [(X86tlsaddr tls32addr:$sym)]>,
                  Requires<[In32BitMode]>;

// All calls clobber the non-callee saved registers. RSP is marked as
// a use to prevent stack-pointer assignments that appear immediately
// before calls from potentially appearing dead.
let Defs = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
            FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
            MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
    Uses = [RSP] in
def TLS_addr64 : I<0, Pseudo, (outs), (ins i64mem:$sym),
                   "# TLS_addr64",
                  [(X86tlsaddr tls64addr:$sym)]>,
                  Requires<[In64BitMode]>;

// Darwin TLS Support
// For i386, the address of the thunk is passed on the stack, on return the
// address of the variable is in %eax.  %ecx is trashed during the function
// call.  All other registers are preserved.
let Defs = [EAX, ECX, EFLAGS],
    Uses = [ESP],
    usesCustomInserter = 1 in
def TLSCall_32 : I<0, Pseudo, (outs), (ins i32mem:$sym),
                "# TLSCall_32",
                [(X86TLSCall addr:$sym)]>,
                Requires<[In32BitMode]>;

// For x86_64, the address of the thunk is passed in %rdi, on return
// the address of the variable is in %rax.  All other registers are preserved.
let Defs = [RAX, EFLAGS],
    Uses = [RSP, RDI],
    usesCustomInserter = 1 in
def TLSCall_64 : I<0, Pseudo, (outs), (ins i64mem:$sym),
                  "# TLSCall_64",
                  [(X86TLSCall addr:$sym)]>,
                  Requires<[In64BitMode]>;


//===----------------------------------------------------------------------===//
// Conditional Move Pseudo Instructions

let Constraints = "$src1 = $dst" in {

// Conditional moves
let Uses = [EFLAGS] in {

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


//===----------------------------------------------------------------------===//
// Atomic Instruction Pseudo Instructions
//===----------------------------------------------------------------------===//

// Atomic exchange, and, or, xor
let Constraints = "$val = $dst", Defs = [EFLAGS],
                  usesCustomInserter = 1 in {

def ATOMAND8 : I<0, Pseudo, (outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMAND8 PSEUDO!",
               [(set GR8:$dst, (atomic_load_and_8 addr:$ptr, GR8:$val))]>;
def ATOMOR8 : I<0, Pseudo, (outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMOR8 PSEUDO!",
               [(set GR8:$dst, (atomic_load_or_8 addr:$ptr, GR8:$val))]>;
def ATOMXOR8 : I<0, Pseudo,(outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMXOR8 PSEUDO!",
               [(set GR8:$dst, (atomic_load_xor_8 addr:$ptr, GR8:$val))]>;
def ATOMNAND8 : I<0, Pseudo,(outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMNAND8 PSEUDO!",
               [(set GR8:$dst, (atomic_load_nand_8 addr:$ptr, GR8:$val))]>;

def ATOMAND16 : I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMAND16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_and_16 addr:$ptr, GR16:$val))]>;
def ATOMOR16 : I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMOR16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_or_16 addr:$ptr, GR16:$val))]>;
def ATOMXOR16 : I<0, Pseudo,(outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMXOR16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_xor_16 addr:$ptr, GR16:$val))]>;
def ATOMNAND16 : I<0, Pseudo,(outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMNAND16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_nand_16 addr:$ptr, GR16:$val))]>;
def ATOMMIN16: I<0, Pseudo, (outs GR16:$dst), (ins i16mem:$ptr, GR16:$val),
               "#ATOMMIN16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_min_16 addr:$ptr, GR16:$val))]>;
def ATOMMAX16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMMAX16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_max_16 addr:$ptr, GR16:$val))]>;
def ATOMUMIN16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMUMIN16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_umin_16 addr:$ptr, GR16:$val))]>;
def ATOMUMAX16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMUMAX16 PSEUDO!",
               [(set GR16:$dst, (atomic_load_umax_16 addr:$ptr, GR16:$val))]>;


def ATOMAND32 : I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMAND32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_and_32 addr:$ptr, GR32:$val))]>;
def ATOMOR32 : I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMOR32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_or_32 addr:$ptr, GR32:$val))]>;
def ATOMXOR32 : I<0, Pseudo,(outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMXOR32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_xor_32 addr:$ptr, GR32:$val))]>;
def ATOMNAND32 : I<0, Pseudo,(outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMNAND32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_nand_32 addr:$ptr, GR32:$val))]>;
def ATOMMIN32: I<0, Pseudo, (outs GR32:$dst), (ins i32mem:$ptr, GR32:$val),
               "#ATOMMIN32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_min_32 addr:$ptr, GR32:$val))]>;
def ATOMMAX32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMMAX32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_max_32 addr:$ptr, GR32:$val))]>;
def ATOMUMIN32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMUMIN32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_umin_32 addr:$ptr, GR32:$val))]>;
def ATOMUMAX32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMUMAX32 PSEUDO!",
               [(set GR32:$dst, (atomic_load_umax_32 addr:$ptr, GR32:$val))]>;



def ATOMAND64 : I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMAND64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_and_64 addr:$ptr, GR64:$val))]>;
def ATOMOR64 : I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMOR64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_or_64 addr:$ptr, GR64:$val))]>;
def ATOMXOR64 : I<0, Pseudo,(outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMXOR64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_xor_64 addr:$ptr, GR64:$val))]>;
def ATOMNAND64 : I<0, Pseudo,(outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMNAND64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_nand_64 addr:$ptr, GR64:$val))]>;
def ATOMMIN64: I<0, Pseudo, (outs GR64:$dst), (ins i64mem:$ptr, GR64:$val),
               "#ATOMMIN64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_min_64 addr:$ptr, GR64:$val))]>;
def ATOMMAX64: I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMMAX64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_max_64 addr:$ptr, GR64:$val))]>;
def ATOMUMIN64: I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMUMIN64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_umin_64 addr:$ptr, GR64:$val))]>;
def ATOMUMAX64: I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMUMAX64 PSEUDO!",
               [(set GR64:$dst, (atomic_load_umax_64 addr:$ptr, GR64:$val))]>;
}

let Constraints = "$val1 = $dst1, $val2 = $dst2",
                  Defs = [EFLAGS, EAX, EBX, ECX, EDX],
                  Uses = [EAX, EBX, ECX, EDX],
                  mayLoad = 1, mayStore = 1,
                  usesCustomInserter = 1 in {
def ATOMAND6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMAND6432 PSEUDO!", []>;
def ATOMOR6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMOR6432 PSEUDO!", []>;
def ATOMXOR6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMXOR6432 PSEUDO!", []>;
def ATOMNAND6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMNAND6432 PSEUDO!", []>;
def ATOMADD6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMADD6432 PSEUDO!", []>;
def ATOMSUB6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMSUB6432 PSEUDO!", []>;
def ATOMSWAP6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMSWAP6432 PSEUDO!", []>;
}

//===----------------------------------------------------------------------===//
// Normal-Instructions-With-Lock-Prefix Pseudo Instructions
//===----------------------------------------------------------------------===//

// FIXME: Use normal instructions and add lock prefix dynamically.

// Memory barriers

// TODO: Get this to fold the constant into the instruction.
let isCodeGenOnly = 1 in
def OR32mrLocked  : I<0x09, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$zero),
                      "lock\n\t"
                      "or{l}\t{$zero, $dst|$dst, $zero}",
                      []>, Requires<[In32BitMode]>, LOCK;

let hasSideEffects = 1 in
def Int_MemBarrier : I<0, Pseudo, (outs), (ins),
                     "#MEMBARRIER",
                     [(X86MemBarrier)]>, Requires<[HasSSE2]>;

// TODO: Get this to fold the constant into the instruction.
let hasSideEffects = 1, Defs = [ESP], isCodeGenOnly = 1 in
def Int_MemBarrierNoSSE64  : RI<0x09, MRM1r, (outs), (ins GR64:$zero),
                           "lock\n\t"
                           "or{q}\t{$zero, (%rsp)|(%rsp), $zero}",
                           [(X86MemBarrierNoSSE GR64:$zero)]>,
                           Requires<[In64BitMode]>, LOCK;


// RegOpc corresponds to the mr version of the instruction
// ImmOpc corresponds to the mi version of the instruction
// ImmOpc8 corresponds to the mi8 version of the instruction
// ImmMod corresponds to the instruction format of the mi and mi8 versions
multiclass LOCK_ArithBinOp<bits<8> RegOpc, bits<8> ImmOpc, bits<8> ImmOpc8,
                           Format ImmMod, string mnemonic> {
let Defs = [EFLAGS], mayLoad = 1, mayStore = 1, isCodeGenOnly = 1 in {

def #NAME#8mr : I<{RegOpc{7}, RegOpc{6}, RegOpc{5}, RegOpc{4},
                   RegOpc{3}, RegOpc{2}, RegOpc{1}, 0 },
                   MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src2),
                   !strconcat("lock\n\t", mnemonic, "{b}\t",
                              "{$src2, $dst|$dst, $src2}"),
                   []>, LOCK;
def #NAME#16mr : I<{RegOpc{7}, RegOpc{6}, RegOpc{5}, RegOpc{4},
                    RegOpc{3}, RegOpc{2}, RegOpc{1}, 1 },
                    MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                    !strconcat("lock\n\t", mnemonic, "{w}\t",
                               "{$src2, $dst|$dst, $src2}"),
                    []>, OpSize, LOCK;
def #NAME#32mr : I<{RegOpc{7}, RegOpc{6}, RegOpc{5}, RegOpc{4},
                    RegOpc{3}, RegOpc{2}, RegOpc{1}, 1 },
                    MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                    !strconcat("lock\n\t", mnemonic, "{l}\t",
                               "{$src2, $dst|$dst, $src2}"),
                    []>, LOCK;
def #NAME#64mr : RI<{RegOpc{7}, RegOpc{6}, RegOpc{5}, RegOpc{4},
                     RegOpc{3}, RegOpc{2}, RegOpc{1}, 1 },
                     MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                     !strconcat("lock\n\t", mnemonic, "{q}\t",
                                "{$src2, $dst|$dst, $src2}"),
                     []>, LOCK;

def #NAME#8mi : Ii8<{ImmOpc{7}, ImmOpc{6}, ImmOpc{5}, ImmOpc{4},
                     ImmOpc{3}, ImmOpc{2}, ImmOpc{1}, 0 },
                     ImmMod, (outs), (ins i8mem :$dst, i8imm :$src2),
                     !strconcat("lock\n\t", mnemonic, "{b}\t",
                                "{$src2, $dst|$dst, $src2}"),
                     []>, LOCK;

def #NAME#16mi : Ii16<{ImmOpc{7}, ImmOpc{6}, ImmOpc{5}, ImmOpc{4},
                       ImmOpc{3}, ImmOpc{2}, ImmOpc{1}, 1 },
                       ImmMod, (outs), (ins i16mem :$dst, i16imm :$src2),
                       !strconcat("lock\n\t", mnemonic, "{w}\t",
                                  "{$src2, $dst|$dst, $src2}"),
                       []>, LOCK;

def #NAME#32mi : Ii32<{ImmOpc{7}, ImmOpc{6}, ImmOpc{5}, ImmOpc{4},
                       ImmOpc{3}, ImmOpc{2}, ImmOpc{1}, 1 },
                       ImmMod, (outs), (ins i32mem :$dst, i32imm :$src2),
                       !strconcat("lock\n\t", mnemonic, "{l}\t",
                                  "{$src2, $dst|$dst, $src2}"),
                       []>, LOCK;

def #NAME#64mi32 : RIi32<{ImmOpc{7}, ImmOpc{6}, ImmOpc{5}, ImmOpc{4},
                          ImmOpc{3}, ImmOpc{2}, ImmOpc{1}, 1 },
                          ImmMod, (outs), (ins i64mem :$dst, i64i32imm :$src2),
                          !strconcat("lock\n\t", mnemonic, "{q}\t",
                                     "{$src2, $dst|$dst, $src2}"),
                          []>, LOCK;

def #NAME#16mi8 : Ii8<{ImmOpc8{7}, ImmOpc8{6}, ImmOpc8{5}, ImmOpc8{4},
                       ImmOpc8{3}, ImmOpc8{2}, ImmOpc8{1}, 1 },
                       ImmMod, (outs), (ins i16mem :$dst, i16i8imm :$src2),
                       !strconcat("lock\n\t", mnemonic, "{w}\t",
                                  "{$src2, $dst|$dst, $src2}"),
                       []>, LOCK;
def #NAME#32mi8 : Ii8<{ImmOpc8{7}, ImmOpc8{6}, ImmOpc8{5}, ImmOpc8{4},
                       ImmOpc8{3}, ImmOpc8{2}, ImmOpc8{1}, 1 },
                       ImmMod, (outs), (ins i32mem :$dst, i32i8imm :$src2),
                       !strconcat("lock\n\t", mnemonic, "{l}\t",
                                  "{$src2, $dst|$dst, $src2}"),
                       []>, LOCK;
def #NAME#64mi8 : RIi8<{ImmOpc8{7}, ImmOpc8{6}, ImmOpc8{5}, ImmOpc8{4},
                        ImmOpc8{3}, ImmOpc8{2}, ImmOpc8{1}, 1 },
                        ImmMod, (outs), (ins i64mem :$dst, i64i8imm :$src2),
                        !strconcat("lock\n\t", mnemonic, "{q}\t",
                                   "{$src2, $dst|$dst, $src2}"),
                        []>, LOCK;

}

}

defm LOCK_ADD : LOCK_ArithBinOp<0x00, 0x80, 0x83, MRM0m, "add">;
defm LOCK_SUB : LOCK_ArithBinOp<0x28, 0x80, 0x83, MRM5m, "sub">;
defm LOCK_OR  : LOCK_ArithBinOp<0x08, 0x80, 0x83, MRM1m, "or">;
defm LOCK_AND : LOCK_ArithBinOp<0x08, 0x80, 0x83, MRM4m, "and">;
defm LOCK_XOR : LOCK_ArithBinOp<0x08, 0x80, 0x83, MRM6m, "xor">;

// Optimized codegen when the non-memory output is not used.
let Defs = [EFLAGS], mayLoad = 1, mayStore = 1, isCodeGenOnly = 1 in {

def LOCK_INC8m  : I<0xFE, MRM0m, (outs), (ins i8mem :$dst),
                    "lock\n\t"
                    "inc{b}\t$dst", []>, LOCK;
def LOCK_INC16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst),
                    "lock\n\t"
                    "inc{w}\t$dst", []>, OpSize, LOCK;
def LOCK_INC32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst),
                    "lock\n\t"
                    "inc{l}\t$dst", []>, LOCK;
def LOCK_INC64m : RI<0xFF, MRM0m, (outs), (ins i64mem:$dst),
                     "lock\n\t"
                     "inc{q}\t$dst", []>, LOCK;

def LOCK_DEC8m  : I<0xFE, MRM1m, (outs), (ins i8mem :$dst),
                    "lock\n\t"
                    "dec{b}\t$dst", []>, LOCK;
def LOCK_DEC16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst),
                    "lock\n\t"
                    "dec{w}\t$dst", []>, OpSize, LOCK;
def LOCK_DEC32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst),
                    "lock\n\t"
                    "dec{l}\t$dst", []>, LOCK;
def LOCK_DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst),
                      "lock\n\t"
                      "dec{q}\t$dst", []>, LOCK;
}

// Atomic compare and swap.
let Defs = [EAX, EDX, EFLAGS], Uses = [EAX, EBX, ECX, EDX],
    isCodeGenOnly = 1 in {
def LCMPXCHG8B : I<0xC7, MRM1m, (outs), (ins i64mem:$ptr),
               "lock\n\t"
               "cmpxchg8b\t$ptr",
               [(X86cas8 addr:$ptr)]>, TB, LOCK;
}
let Defs = [AL, EFLAGS], Uses = [AL], isCodeGenOnly = 1 in {
def LCMPXCHG8 : I<0xB0, MRMDestMem, (outs), (ins i8mem:$ptr, GR8:$swap),
               "lock\n\t"
               "cmpxchg{b}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR8:$swap, 1)]>, TB, LOCK;
}

let Defs = [AX, EFLAGS], Uses = [AX], isCodeGenOnly = 1 in {
def LCMPXCHG16 : I<0xB1, MRMDestMem, (outs), (ins i16mem:$ptr, GR16:$swap),
               "lock\n\t"
               "cmpxchg{w}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR16:$swap, 2)]>, TB, OpSize, LOCK;
}

let Defs = [EAX, EFLAGS], Uses = [EAX], isCodeGenOnly = 1 in {
def LCMPXCHG32 : I<0xB1, MRMDestMem, (outs), (ins i32mem:$ptr, GR32:$swap),
               "lock\n\t"
               "cmpxchg{l}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR32:$swap, 4)]>, TB, LOCK;
}

let Defs = [RAX, EFLAGS], Uses = [RAX], isCodeGenOnly = 1 in {
def LCMPXCHG64 : RI<0xB1, MRMDestMem, (outs), (ins i64mem:$ptr, GR64:$swap),
               "lock\n\t"
               "cmpxchgq\t$swap,$ptr",
               [(X86cas addr:$ptr, GR64:$swap, 8)]>, TB, LOCK;
}

// Atomic exchange and add
let Constraints = "$val = $dst", Defs = [EFLAGS], isCodeGenOnly = 1 in {
def LXADD8  : I<0xC0, MRMSrcMem, (outs GR8:$dst), (ins GR8:$val, i8mem:$ptr),
               "lock\n\t"
               "xadd{b}\t{$val, $ptr|$ptr, $val}",
               [(set GR8:$dst, (atomic_load_add_8 addr:$ptr, GR8:$val))]>,
                TB, LOCK;
def LXADD16 : I<0xC1, MRMSrcMem, (outs GR16:$dst), (ins GR16:$val, i16mem:$ptr),
               "lock\n\t"
               "xadd{w}\t{$val, $ptr|$ptr, $val}",
               [(set GR16:$dst, (atomic_load_add_16 addr:$ptr, GR16:$val))]>,
                TB, OpSize, LOCK;
def LXADD32 : I<0xC1, MRMSrcMem, (outs GR32:$dst), (ins GR32:$val, i32mem:$ptr),
               "lock\n\t"
               "xadd{l}\t{$val, $ptr|$ptr, $val}",
               [(set GR32:$dst, (atomic_load_add_32 addr:$ptr, GR32:$val))]>,
                TB, LOCK;
def LXADD64 : RI<0xC1, MRMSrcMem, (outs GR64:$dst), (ins GR64:$val,i64mem:$ptr),
               "lock\n\t"
               "xadd\t$val, $ptr",
               [(set GR64:$dst, (atomic_load_add_64 addr:$ptr, GR64:$val))]>,
                TB, LOCK;
}

//===----------------------------------------------------------------------===//
// Conditional Move Pseudo Instructions.
//===----------------------------------------------------------------------===//


// CMOV* - Used to implement the SSE SELECT DAG operation.  Expanded after
// instruction selection into a branch sequence.
let Uses = [EFLAGS], usesCustomInserter = 1 in {
  def CMOV_FR32 : I<0, Pseudo,
                    (outs FR32:$dst), (ins FR32:$t, FR32:$f, i8imm:$cond),
                    "#CMOV_FR32 PSEUDO!",
                    [(set FR32:$dst, (X86cmov FR32:$t, FR32:$f, imm:$cond,
                                                  EFLAGS))]>;
  def CMOV_FR64 : I<0, Pseudo,
                    (outs FR64:$dst), (ins FR64:$t, FR64:$f, i8imm:$cond),
                    "#CMOV_FR64 PSEUDO!",
                    [(set FR64:$dst, (X86cmov FR64:$t, FR64:$f, imm:$cond,
                                                  EFLAGS))]>;
  def CMOV_V4F32 : I<0, Pseudo,
                    (outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V4F32 PSEUDO!",
                    [(set VR128:$dst,
                      (v4f32 (X86cmov VR128:$t, VR128:$f, imm:$cond,
                                          EFLAGS)))]>;
  def CMOV_V2F64 : I<0, Pseudo,
                    (outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V2F64 PSEUDO!",
                    [(set VR128:$dst,
                      (v2f64 (X86cmov VR128:$t, VR128:$f, imm:$cond,
                                          EFLAGS)))]>;
  def CMOV_V2I64 : I<0, Pseudo,
                    (outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                    "#CMOV_V2I64 PSEUDO!",
                    [(set VR128:$dst,
                      (v2i64 (X86cmov VR128:$t, VR128:$f, imm:$cond,
                                          EFLAGS)))]>;
}


//===----------------------------------------------------------------------===//
// DAG Pattern Matching Rules
//===----------------------------------------------------------------------===//

// ConstantPool GlobalAddress, ExternalSymbol, and JumpTable
def : Pat<(i32 (X86Wrapper tconstpool  :$dst)), (MOV32ri tconstpool  :$dst)>;
def : Pat<(i32 (X86Wrapper tjumptable  :$dst)), (MOV32ri tjumptable  :$dst)>;
def : Pat<(i32 (X86Wrapper tglobaltlsaddr:$dst)),(MOV32ri tglobaltlsaddr:$dst)>;
def : Pat<(i32 (X86Wrapper tglobaladdr :$dst)), (MOV32ri tglobaladdr :$dst)>;
def : Pat<(i32 (X86Wrapper texternalsym:$dst)), (MOV32ri texternalsym:$dst)>;
def : Pat<(i32 (X86Wrapper tblockaddress:$dst)), (MOV32ri tblockaddress:$dst)>;

def : Pat<(add GR32:$src1, (X86Wrapper tconstpool:$src2)),
          (ADD32ri GR32:$src1, tconstpool:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper tjumptable:$src2)),
          (ADD32ri GR32:$src1, tjumptable:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper tglobaladdr :$src2)),
          (ADD32ri GR32:$src1, tglobaladdr:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper texternalsym:$src2)),
          (ADD32ri GR32:$src1, texternalsym:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper tblockaddress:$src2)),
          (ADD32ri GR32:$src1, tblockaddress:$src2)>;

def : Pat<(store (i32 (X86Wrapper tglobaladdr:$src)), addr:$dst),
          (MOV32mi addr:$dst, tglobaladdr:$src)>;
def : Pat<(store (i32 (X86Wrapper texternalsym:$src)), addr:$dst),
          (MOV32mi addr:$dst, texternalsym:$src)>;
def : Pat<(store (i32 (X86Wrapper tblockaddress:$src)), addr:$dst),
          (MOV32mi addr:$dst, tblockaddress:$src)>;



// ConstantPool GlobalAddress, ExternalSymbol, and JumpTable when not in small
// code model mode, should use 'movabs'.  FIXME: This is really a hack, the
//  'movabs' predicate should handle this sort of thing.
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri tconstpool  :$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri tjumptable  :$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri tglobaladdr :$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri texternalsym:$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper tblockaddress:$dst)),
          (MOV64ri tblockaddress:$dst)>, Requires<[FarData]>;

// In static codegen with small code model, we can get the address of a label
// into a register with 'movl'.  FIXME: This is a hack, the 'imm' predicate of
// the MOV64ri64i32 should accept these.
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri64i32 tconstpool  :$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri64i32 tjumptable  :$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri64i32 tglobaladdr :$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri64i32 texternalsym:$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper tblockaddress:$dst)),
          (MOV64ri64i32 tblockaddress:$dst)>, Requires<[SmallCode]>;

// In kernel code model, we can get the address of a label
// into a register with 'movq'.  FIXME: This is a hack, the 'imm' predicate of
// the MOV64ri32 should accept these.
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri32 tconstpool  :$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri32 tjumptable  :$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri32 tglobaladdr :$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri32 texternalsym:$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper tblockaddress:$dst)),
          (MOV64ri32 tblockaddress:$dst)>, Requires<[KernelCode]>;

// If we have small model and -static mode, it is safe to store global addresses
// directly as immediates.  FIXME: This is really a hack, the 'imm' predicate
// for MOV64mi32 should handle this sort of thing.
def : Pat<(store (i64 (X86Wrapper tconstpool:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tconstpool:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tjumptable:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tjumptable:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tglobaladdr:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tglobaladdr:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper texternalsym:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, texternalsym:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tblockaddress:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tblockaddress:$src)>,
          Requires<[NearData, IsStatic]>;



// Calls

// tls has some funny stuff here...
// This corresponds to movabs $foo@tpoff, %rax
def : Pat<(i64 (X86Wrapper tglobaltlsaddr :$dst)),
          (MOV64ri tglobaltlsaddr :$dst)>;
// This corresponds to add $foo@tpoff, %rax
def : Pat<(add GR64:$src1, (X86Wrapper tglobaltlsaddr :$dst)),
          (ADD64ri32 GR64:$src1, tglobaltlsaddr :$dst)>;
// This corresponds to mov foo@tpoff(%rbx), %eax
def : Pat<(load (i64 (X86Wrapper tglobaltlsaddr :$dst))),
          (MOV64rm tglobaltlsaddr :$dst)>;


// Direct PC relative function call for small code model. 32-bit displacement
// sign extended to 64-bit.
def : Pat<(X86call (i64 tglobaladdr:$dst)),
          (CALL64pcrel32 tglobaladdr:$dst)>, Requires<[NotWin64]>;
def : Pat<(X86call (i64 texternalsym:$dst)),
          (CALL64pcrel32 texternalsym:$dst)>, Requires<[NotWin64]>;

def : Pat<(X86call (i64 tglobaladdr:$dst)),
          (WINCALL64pcrel32 tglobaladdr:$dst)>, Requires<[IsWin64]>;
def : Pat<(X86call (i64 texternalsym:$dst)),
          (WINCALL64pcrel32 texternalsym:$dst)>, Requires<[IsWin64]>;

// tailcall stuff
def : Pat<(X86tcret GR32_TC:$dst, imm:$off),
          (TCRETURNri GR32_TC:$dst, imm:$off)>,
          Requires<[In32BitMode]>;

// FIXME: This is disabled for 32-bit PIC mode because the global base
// register which is part of the address mode may be assigned a
// callee-saved register.
def : Pat<(X86tcret (load addr:$dst), imm:$off),
          (TCRETURNmi addr:$dst, imm:$off)>,
          Requires<[In32BitMode, IsNotPIC]>;

def : Pat<(X86tcret (i32 tglobaladdr:$dst), imm:$off),
          (TCRETURNdi texternalsym:$dst, imm:$off)>,
          Requires<[In32BitMode]>;

def : Pat<(X86tcret (i32 texternalsym:$dst), imm:$off),
          (TCRETURNdi texternalsym:$dst, imm:$off)>,
          Requires<[In32BitMode]>;

def : Pat<(X86tcret ptr_rc_tailcall:$dst, imm:$off),
          (TCRETURNri64 ptr_rc_tailcall:$dst, imm:$off)>,
          Requires<[In64BitMode]>;

def : Pat<(X86tcret (load addr:$dst), imm:$off),
          (TCRETURNmi64 addr:$dst, imm:$off)>,
          Requires<[In64BitMode]>;

def : Pat<(X86tcret (i64 tglobaladdr:$dst), imm:$off),
          (TCRETURNdi64 tglobaladdr:$dst, imm:$off)>,
          Requires<[In64BitMode]>;

def : Pat<(X86tcret (i64 texternalsym:$dst), imm:$off),
          (TCRETURNdi64 texternalsym:$dst, imm:$off)>,
          Requires<[In64BitMode]>;

// Normal calls, with various flavors of addresses.
def : Pat<(X86call (i32 tglobaladdr:$dst)),
          (CALLpcrel32 tglobaladdr:$dst)>;
def : Pat<(X86call (i32 texternalsym:$dst)),
          (CALLpcrel32 texternalsym:$dst)>;
def : Pat<(X86call (i32 imm:$dst)),
          (CALLpcrel32 imm:$dst)>, Requires<[CallImmAddr]>;

// Comparisons.

// TEST R,R is smaller than CMP R,0
def : Pat<(X86cmp GR8:$src1, 0),
          (TEST8rr GR8:$src1, GR8:$src1)>;
def : Pat<(X86cmp GR16:$src1, 0),
          (TEST16rr GR16:$src1, GR16:$src1)>;
def : Pat<(X86cmp GR32:$src1, 0),
          (TEST32rr GR32:$src1, GR32:$src1)>;
def : Pat<(X86cmp GR64:$src1, 0),
          (TEST64rr GR64:$src1, GR64:$src1)>;

// Conditional moves with folded loads with operands swapped and conditions
// inverted.
multiclass CMOVmr<PatLeaf InvertedCond, Instruction Inst16, Instruction Inst32,
                  Instruction Inst64> {
  def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, InvertedCond, EFLAGS),
            (Inst16 GR16:$src2, addr:$src1)>;
  def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, InvertedCond, EFLAGS),
            (Inst32 GR32:$src2, addr:$src1)>;
  def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, InvertedCond, EFLAGS),
            (Inst64 GR64:$src2, addr:$src1)>;
}

defm : CMOVmr<X86_COND_B , CMOVAE16rm, CMOVAE32rm, CMOVAE64rm>;
defm : CMOVmr<X86_COND_AE, CMOVB16rm , CMOVB32rm , CMOVB64rm>;
defm : CMOVmr<X86_COND_E , CMOVNE16rm, CMOVNE32rm, CMOVNE64rm>;
defm : CMOVmr<X86_COND_NE, CMOVE16rm , CMOVE32rm , CMOVE64rm>;
defm : CMOVmr<X86_COND_BE, CMOVA16rm , CMOVA32rm , CMOVA64rm>;
defm : CMOVmr<X86_COND_A , CMOVBE16rm, CMOVBE32rm, CMOVBE64rm>;
defm : CMOVmr<X86_COND_L , CMOVGE16rm, CMOVGE32rm, CMOVGE64rm>;
defm : CMOVmr<X86_COND_GE, CMOVL16rm , CMOVL32rm , CMOVL64rm>;
defm : CMOVmr<X86_COND_LE, CMOVG16rm , CMOVG32rm , CMOVG64rm>;
defm : CMOVmr<X86_COND_G , CMOVLE16rm, CMOVLE32rm, CMOVLE64rm>;
defm : CMOVmr<X86_COND_P , CMOVNP16rm, CMOVNP32rm, CMOVNP64rm>;
defm : CMOVmr<X86_COND_NP, CMOVP16rm , CMOVP32rm , CMOVP64rm>;
defm : CMOVmr<X86_COND_S , CMOVNS16rm, CMOVNS32rm, CMOVNS64rm>;
defm : CMOVmr<X86_COND_NS, CMOVS16rm , CMOVS32rm , CMOVS64rm>;
defm : CMOVmr<X86_COND_O , CMOVNO16rm, CMOVNO32rm, CMOVNO64rm>;
defm : CMOVmr<X86_COND_NO, CMOVO16rm , CMOVO32rm , CMOVO64rm>;

// zextload bool -> zextload byte
def : Pat<(zextloadi8i1  addr:$src), (MOV8rm     addr:$src)>;
def : Pat<(zextloadi16i1 addr:$src), (MOVZX16rm8 addr:$src)>;
def : Pat<(zextloadi32i1 addr:$src), (MOVZX32rm8 addr:$src)>;
def : Pat<(zextloadi64i1 addr:$src), (MOVZX64rm8 addr:$src)>;

// extload bool -> extload byte
// When extloading from 16-bit and smaller memory locations into 64-bit
// registers, use zero-extending loads so that the entire 64-bit register is
// defined, avoiding partial-register updates.

def : Pat<(extloadi8i1 addr:$src),   (MOV8rm      addr:$src)>;
def : Pat<(extloadi16i1 addr:$src),  (MOVZX16rm8  addr:$src)>;
def : Pat<(extloadi32i1 addr:$src),  (MOVZX32rm8  addr:$src)>;
def : Pat<(extloadi16i8 addr:$src),  (MOVZX16rm8  addr:$src)>;
def : Pat<(extloadi32i8 addr:$src),  (MOVZX32rm8  addr:$src)>;
def : Pat<(extloadi32i16 addr:$src), (MOVZX32rm16 addr:$src)>;

def : Pat<(extloadi64i1 addr:$src),  (MOVZX64rm8  addr:$src)>;
def : Pat<(extloadi64i8 addr:$src),  (MOVZX64rm8  addr:$src)>;
def : Pat<(extloadi64i16 addr:$src), (MOVZX64rm16 addr:$src)>;
// For other extloads, use subregs, since the high contents of the register are
// defined after an extload.
def : Pat<(extloadi64i32 addr:$src),
          (SUBREG_TO_REG (i64 0), (MOV32rm addr:$src),
                         sub_32bit)>;

// anyext. Define these to do an explicit zero-extend to
// avoid partial-register updates.
def : Pat<(i16 (anyext GR8 :$src)), (EXTRACT_SUBREG
                                     (MOVZX32rr8 GR8 :$src), sub_16bit)>;
def : Pat<(i32 (anyext GR8 :$src)), (MOVZX32rr8  GR8 :$src)>;

// Except for i16 -> i32 since isel expect i16 ops to be promoted to i32.
def : Pat<(i32 (anyext GR16:$src)),
          (INSERT_SUBREG (i32 (IMPLICIT_DEF)), GR16:$src, sub_16bit)>;

def : Pat<(i64 (anyext GR8 :$src)), (MOVZX64rr8  GR8  :$src)>;
def : Pat<(i64 (anyext GR16:$src)), (MOVZX64rr16 GR16 :$src)>;
def : Pat<(i64 (anyext GR32:$src)),
          (SUBREG_TO_REG (i64 0), GR32:$src, sub_32bit)>;


// Any instruction that defines a 32-bit result leaves the high half of the
// register. Truncate can be lowered to EXTRACT_SUBREG. CopyFromReg may
// be copying from a truncate. And x86's cmov doesn't do anything if the
// condition is false. But any other 32-bit operation will zero-extend
// up to 64 bits.
def def32 : PatLeaf<(i32 GR32:$src), [{
  return N->getOpcode() != ISD::TRUNCATE &&
         N->getOpcode() != TargetOpcode::EXTRACT_SUBREG &&
         N->getOpcode() != ISD::CopyFromReg &&
         N->getOpcode() != X86ISD::CMOV;
}]>;

// In the case of a 32-bit def that is known to implicitly zero-extend,
// we can use a SUBREG_TO_REG.
def : Pat<(i64 (zext def32:$src)),
          (SUBREG_TO_REG (i64 0), GR32:$src, sub_32bit)>;

//===----------------------------------------------------------------------===//
// Pattern match OR as ADD
//===----------------------------------------------------------------------===//

// If safe, we prefer to pattern match OR as ADD at isel time. ADD can be
// 3-addressified into an LEA instruction to avoid copies.  However, we also
// want to finally emit these instructions as an or at the end of the code
// generator to make the generated code easier to read.  To do this, we select
// into "disjoint bits" pseudo ops.

// Treat an 'or' node is as an 'add' if the or'ed bits are known to be zero.
def or_is_add : PatFrag<(ops node:$lhs, node:$rhs), (or node:$lhs, node:$rhs),[{
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1)))
    return CurDAG->MaskedValueIsZero(N->getOperand(0), CN->getAPIntValue());

  unsigned BitWidth = N->getValueType(0).getScalarType().getSizeInBits();
  APInt Mask = APInt::getAllOnesValue(BitWidth);
  APInt KnownZero0, KnownOne0;
  CurDAG->ComputeMaskedBits(N->getOperand(0), Mask, KnownZero0, KnownOne0, 0);
  APInt KnownZero1, KnownOne1;
  CurDAG->ComputeMaskedBits(N->getOperand(1), Mask, KnownZero1, KnownOne1, 0);
  return (~KnownZero0 & ~KnownZero1) == 0;
}]>;


// (or x1, x2) -> (add x1, x2) if two operands are known not to share bits.
let AddedComplexity = 5 in { // Try this before the selecting to OR

let isConvertibleToThreeAddress = 1,
    Constraints = "$src1 = $dst", Defs = [EFLAGS] in {
let isCommutable = 1 in {
def ADD16rr_DB  : I<0, Pseudo, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                    "", // orw/addw REG, REG
                    [(set GR16:$dst, (or_is_add GR16:$src1, GR16:$src2))]>;
def ADD32rr_DB  : I<0, Pseudo, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                    "", // orl/addl REG, REG
                    [(set GR32:$dst, (or_is_add GR32:$src1, GR32:$src2))]>;
def ADD64rr_DB  : I<0, Pseudo, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "", // orq/addq REG, REG
                    [(set GR64:$dst, (or_is_add GR64:$src1, GR64:$src2))]>;
} // isCommutable

// NOTE: These are order specific, we want the ri8 forms to be listed
// first so that they are slightly preferred to the ri forms.

def ADD16ri8_DB : I<0, Pseudo,
                    (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                    "", // orw/addw REG, imm8
                    [(set GR16:$dst,(or_is_add GR16:$src1,i16immSExt8:$src2))]>;
def ADD16ri_DB  : I<0, Pseudo, (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                    "", // orw/addw REG, imm
                    [(set GR16:$dst, (or_is_add GR16:$src1, imm:$src2))]>;

def ADD32ri8_DB : I<0, Pseudo,
                    (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                    "", // orl/addl REG, imm8
                    [(set GR32:$dst,(or_is_add GR32:$src1,i32immSExt8:$src2))]>;
def ADD32ri_DB  : I<0, Pseudo, (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "", // orl/addl REG, imm
                    [(set GR32:$dst, (or_is_add GR32:$src1, imm:$src2))]>;


def ADD64ri8_DB : I<0, Pseudo,
                    (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "", // orq/addq REG, imm8
                    [(set GR64:$dst, (or_is_add GR64:$src1,
                                                i64immSExt8:$src2))]>;
def ADD64ri32_DB : I<0, Pseudo,
                     (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                      "", // orq/addq REG, imm
                      [(set GR64:$dst, (or_is_add GR64:$src1,
                                                  i64immSExt32:$src2))]>;
}
} // AddedComplexity


//===----------------------------------------------------------------------===//
// Some peepholes
//===----------------------------------------------------------------------===//

// Odd encoding trick: -128 fits into an 8-bit immediate field while
// +128 doesn't, so in this special case use a sub instead of an add.
def : Pat<(add GR16:$src1, 128),
          (SUB16ri8 GR16:$src1, -128)>;
def : Pat<(store (add (loadi16 addr:$dst), 128), addr:$dst),
          (SUB16mi8 addr:$dst, -128)>;

def : Pat<(add GR32:$src1, 128),
          (SUB32ri8 GR32:$src1, -128)>;
def : Pat<(store (add (loadi32 addr:$dst), 128), addr:$dst),
          (SUB32mi8 addr:$dst, -128)>;

def : Pat<(add GR64:$src1, 128),
          (SUB64ri8 GR64:$src1, -128)>;
def : Pat<(store (add (loadi64 addr:$dst), 128), addr:$dst),
          (SUB64mi8 addr:$dst, -128)>;

// The same trick applies for 32-bit immediate fields in 64-bit
// instructions.
def : Pat<(add GR64:$src1, 0x0000000080000000),
          (SUB64ri32 GR64:$src1, 0xffffffff80000000)>;
def : Pat<(store (add (loadi64 addr:$dst), 0x00000000800000000), addr:$dst),
          (SUB64mi32 addr:$dst, 0xffffffff80000000)>;

// To avoid needing to materialize an immediate in a register, use a 32-bit and
// with implicit zero-extension instead of a 64-bit and if the immediate has at
// least 32 bits of leading zeros. If in addition the last 32 bits can be
// represented with a sign extension of a 8 bit constant, use that.

def : Pat<(and GR64:$src, i64immZExt32SExt8:$imm),
          (SUBREG_TO_REG
            (i64 0),
            (AND32ri8
              (EXTRACT_SUBREG GR64:$src, sub_32bit),
              (i32 (GetLo8XForm imm:$imm))),
            sub_32bit)>;

def : Pat<(and GR64:$src, i64immZExt32:$imm),
          (SUBREG_TO_REG
            (i64 0),
            (AND32ri
              (EXTRACT_SUBREG GR64:$src, sub_32bit),
              (i32 (GetLo32XForm imm:$imm))),
            sub_32bit)>;


// r & (2^16-1) ==> movz
def : Pat<(and GR32:$src1, 0xffff),
          (MOVZX32rr16 (EXTRACT_SUBREG GR32:$src1, sub_16bit))>;
// r & (2^8-1) ==> movz
def : Pat<(and GR32:$src1, 0xff),
          (MOVZX32rr8 (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src1,
                                                             GR32_ABCD)),
                                      sub_8bit))>,
      Requires<[In32BitMode]>;
// r & (2^8-1) ==> movz
def : Pat<(and GR16:$src1, 0xff),
           (EXTRACT_SUBREG (MOVZX32rr8 (EXTRACT_SUBREG
            (i16 (COPY_TO_REGCLASS GR16:$src1, GR16_ABCD)), sub_8bit)),
             sub_16bit)>,
      Requires<[In32BitMode]>;

// r & (2^32-1) ==> movz
def : Pat<(and GR64:$src, 0x00000000FFFFFFFF),
          (MOVZX64rr32 (EXTRACT_SUBREG GR64:$src, sub_32bit))>;
// r & (2^16-1) ==> movz
def : Pat<(and GR64:$src, 0xffff),
          (MOVZX64rr16 (i16 (EXTRACT_SUBREG GR64:$src, sub_16bit)))>;
// r & (2^8-1) ==> movz
def : Pat<(and GR64:$src, 0xff),
          (MOVZX64rr8 (i8 (EXTRACT_SUBREG GR64:$src, sub_8bit)))>;
// r & (2^8-1) ==> movz
def : Pat<(and GR32:$src1, 0xff),
           (MOVZX32rr8 (EXTRACT_SUBREG GR32:$src1, sub_8bit))>,
      Requires<[In64BitMode]>;
// r & (2^8-1) ==> movz
def : Pat<(and GR16:$src1, 0xff),
           (EXTRACT_SUBREG (MOVZX32rr8 (i8
            (EXTRACT_SUBREG GR16:$src1, sub_8bit))), sub_16bit)>,
      Requires<[In64BitMode]>;


// sext_inreg patterns
def : Pat<(sext_inreg GR32:$src, i16),
          (MOVSX32rr16 (EXTRACT_SUBREG GR32:$src, sub_16bit))>;
def : Pat<(sext_inreg GR32:$src, i8),
          (MOVSX32rr8 (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src,
                                                             GR32_ABCD)),
                                      sub_8bit))>,
      Requires<[In32BitMode]>;

def : Pat<(sext_inreg GR16:$src, i8),
           (EXTRACT_SUBREG (i32 (MOVSX32rr8 (EXTRACT_SUBREG
            (i32 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)), sub_8bit))),
             sub_16bit)>,
      Requires<[In32BitMode]>;

def : Pat<(sext_inreg GR64:$src, i32),
          (MOVSX64rr32 (EXTRACT_SUBREG GR64:$src, sub_32bit))>;
def : Pat<(sext_inreg GR64:$src, i16),
          (MOVSX64rr16 (EXTRACT_SUBREG GR64:$src, sub_16bit))>;
def : Pat<(sext_inreg GR64:$src, i8),
          (MOVSX64rr8 (EXTRACT_SUBREG GR64:$src, sub_8bit))>;
def : Pat<(sext_inreg GR32:$src, i8),
          (MOVSX32rr8 (EXTRACT_SUBREG GR32:$src, sub_8bit))>,
      Requires<[In64BitMode]>;
def : Pat<(sext_inreg GR16:$src, i8),
           (EXTRACT_SUBREG (MOVSX32rr8
            (EXTRACT_SUBREG GR16:$src, sub_8bit)), sub_16bit)>,
      Requires<[In64BitMode]>;

// sext, sext_load, zext, zext_load
def: Pat<(i16 (sext GR8:$src)),
          (EXTRACT_SUBREG (MOVSX32rr8 GR8:$src), sub_16bit)>;
def: Pat<(sextloadi16i8 addr:$src),
          (EXTRACT_SUBREG (MOVSX32rm8 addr:$src), sub_16bit)>;
def: Pat<(i16 (zext GR8:$src)),
          (EXTRACT_SUBREG (MOVZX32rr8 GR8:$src), sub_16bit)>;
def: Pat<(zextloadi16i8 addr:$src),
          (EXTRACT_SUBREG (MOVZX32rm8 addr:$src), sub_16bit)>;

// trunc patterns
def : Pat<(i16 (trunc GR32:$src)),
          (EXTRACT_SUBREG GR32:$src, sub_16bit)>;
def : Pat<(i8 (trunc GR32:$src)),
          (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src, GR32_ABCD)),
                          sub_8bit)>,
      Requires<[In32BitMode]>;
def : Pat<(i8 (trunc GR16:$src)),
          (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                          sub_8bit)>,
      Requires<[In32BitMode]>;
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, sub_32bit)>;
def : Pat<(i16 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, sub_16bit)>;
def : Pat<(i8 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, sub_8bit)>;
def : Pat<(i8 (trunc GR32:$src)),
          (EXTRACT_SUBREG GR32:$src, sub_8bit)>,
      Requires<[In64BitMode]>;
def : Pat<(i8 (trunc GR16:$src)),
          (EXTRACT_SUBREG GR16:$src, sub_8bit)>,
      Requires<[In64BitMode]>;

// h-register tricks
def : Pat<(i8 (trunc (srl_su GR16:$src, (i8 8)))),
          (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                          sub_8bit_hi)>,
      Requires<[In32BitMode]>;
def : Pat<(i8 (trunc (srl_su GR32:$src, (i8 8)))),
          (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src, GR32_ABCD)),
                          sub_8bit_hi)>,
      Requires<[In32BitMode]>;
def : Pat<(srl GR16:$src, (i8 8)),
          (EXTRACT_SUBREG
            (MOVZX32rr8
              (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                              sub_8bit_hi)),
            sub_16bit)>,
      Requires<[In32BitMode]>;
def : Pat<(i32 (zext (srl_su GR16:$src, (i8 8)))),
          (MOVZX32rr8 (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src,
                                                             GR16_ABCD)),
                                      sub_8bit_hi))>,
      Requires<[In32BitMode]>;
def : Pat<(i32 (anyext (srl_su GR16:$src, (i8 8)))),
          (MOVZX32rr8 (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src,
                                                             GR16_ABCD)),
                                      sub_8bit_hi))>,
      Requires<[In32BitMode]>;
def : Pat<(and (srl_su GR32:$src, (i8 8)), (i32 255)),
          (MOVZX32rr8 (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src,
                                                             GR32_ABCD)),
                                      sub_8bit_hi))>,
      Requires<[In32BitMode]>;
def : Pat<(srl (and_su GR32:$src, 0xff00), (i8 8)),
          (MOVZX32rr8 (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src,
                                                             GR32_ABCD)),
                                      sub_8bit_hi))>,
      Requires<[In32BitMode]>;

// h-register tricks.
// For now, be conservative on x86-64 and use an h-register extract only if the
// value is immediately zero-extended or stored, which are somewhat common
// cases. This uses a bunch of code to prevent a register requiring a REX prefix
// from being allocated in the same instruction as the h register, as there's
// currently no way to describe this requirement to the register allocator.

// h-register extract and zero-extend.
def : Pat<(and (srl_su GR64:$src, (i8 8)), (i64 255)),
          (SUBREG_TO_REG
            (i64 0),
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (i64 (COPY_TO_REGCLASS GR64:$src, GR64_ABCD)),
                              sub_8bit_hi)),
            sub_32bit)>;
def : Pat<(and (srl_su GR32:$src, (i8 8)), (i32 255)),
          (MOVZX32_NOREXrr8
            (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src, GR32_ABCD)),
                            sub_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(srl (and_su GR32:$src, 0xff00), (i8 8)),
          (MOVZX32_NOREXrr8 (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src,
                                                                   GR32_ABCD)),
                                             sub_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(srl GR16:$src, (i8 8)),
          (EXTRACT_SUBREG
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                              sub_8bit_hi)),
            sub_16bit)>,
      Requires<[In64BitMode]>;
def : Pat<(i32 (zext (srl_su GR16:$src, (i8 8)))),
          (MOVZX32_NOREXrr8
            (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                            sub_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(i32 (anyext (srl_su GR16:$src, (i8 8)))),
          (MOVZX32_NOREXrr8
            (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                            sub_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(i64 (zext (srl_su GR16:$src, (i8 8)))),
          (SUBREG_TO_REG
            (i64 0),
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                              sub_8bit_hi)),
            sub_32bit)>;
def : Pat<(i64 (anyext (srl_su GR16:$src, (i8 8)))),
          (SUBREG_TO_REG
            (i64 0),
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                              sub_8bit_hi)),
            sub_32bit)>;

// h-register extract and store.
def : Pat<(store (i8 (trunc_su (srl_su GR64:$src, (i8 8)))), addr:$dst),
          (MOV8mr_NOREX
            addr:$dst,
            (EXTRACT_SUBREG (i64 (COPY_TO_REGCLASS GR64:$src, GR64_ABCD)),
                            sub_8bit_hi))>;
def : Pat<(store (i8 (trunc_su (srl_su GR32:$src, (i8 8)))), addr:$dst),
          (MOV8mr_NOREX
            addr:$dst,
            (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS GR32:$src, GR32_ABCD)),
                            sub_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(store (i8 (trunc_su (srl_su GR16:$src, (i8 8)))), addr:$dst),
          (MOV8mr_NOREX
            addr:$dst,
            (EXTRACT_SUBREG (i16 (COPY_TO_REGCLASS GR16:$src, GR16_ABCD)),
                            sub_8bit_hi))>,
      Requires<[In64BitMode]>;


// (shl x, 1) ==> (add x, x)
// Note that if x is undef (immediate or otherwise), we could theoretically
// end up with the two uses of x getting different values, producing a result
// where the least significant bit is not 0. However, the probability of this
// happening is considered low enough that this is officially not a
// "real problem".
def : Pat<(shl GR8 :$src1, (i8 1)), (ADD8rr  GR8 :$src1, GR8 :$src1)>;
def : Pat<(shl GR16:$src1, (i8 1)), (ADD16rr GR16:$src1, GR16:$src1)>;
def : Pat<(shl GR32:$src1, (i8 1)), (ADD32rr GR32:$src1, GR32:$src1)>;
def : Pat<(shl GR64:$src1, (i8 1)), (ADD64rr GR64:$src1, GR64:$src1)>;

// (shl x (and y, 31)) ==> (shl x, y)
def : Pat<(shl GR8:$src1, (and CL, 31)),
          (SHL8rCL GR8:$src1)>;
def : Pat<(shl GR16:$src1, (and CL, 31)),
          (SHL16rCL GR16:$src1)>;
def : Pat<(shl GR32:$src1, (and CL, 31)),
          (SHL32rCL GR32:$src1)>;
def : Pat<(store (shl (loadi8 addr:$dst), (and CL, 31)), addr:$dst),
          (SHL8mCL addr:$dst)>;
def : Pat<(store (shl (loadi16 addr:$dst), (and CL, 31)), addr:$dst),
          (SHL16mCL addr:$dst)>;
def : Pat<(store (shl (loadi32 addr:$dst), (and CL, 31)), addr:$dst),
          (SHL32mCL addr:$dst)>;

def : Pat<(srl GR8:$src1, (and CL, 31)),
          (SHR8rCL GR8:$src1)>;
def : Pat<(srl GR16:$src1, (and CL, 31)),
          (SHR16rCL GR16:$src1)>;
def : Pat<(srl GR32:$src1, (and CL, 31)),
          (SHR32rCL GR32:$src1)>;
def : Pat<(store (srl (loadi8 addr:$dst), (and CL, 31)), addr:$dst),
          (SHR8mCL addr:$dst)>;
def : Pat<(store (srl (loadi16 addr:$dst), (and CL, 31)), addr:$dst),
          (SHR16mCL addr:$dst)>;
def : Pat<(store (srl (loadi32 addr:$dst), (and CL, 31)), addr:$dst),
          (SHR32mCL addr:$dst)>;

def : Pat<(sra GR8:$src1, (and CL, 31)),
          (SAR8rCL GR8:$src1)>;
def : Pat<(sra GR16:$src1, (and CL, 31)),
          (SAR16rCL GR16:$src1)>;
def : Pat<(sra GR32:$src1, (and CL, 31)),
          (SAR32rCL GR32:$src1)>;
def : Pat<(store (sra (loadi8 addr:$dst), (and CL, 31)), addr:$dst),
          (SAR8mCL addr:$dst)>;
def : Pat<(store (sra (loadi16 addr:$dst), (and CL, 31)), addr:$dst),
          (SAR16mCL addr:$dst)>;
def : Pat<(store (sra (loadi32 addr:$dst), (and CL, 31)), addr:$dst),
          (SAR32mCL addr:$dst)>;

// (shl x (and y, 63)) ==> (shl x, y)
def : Pat<(shl GR64:$src1, (and CL, 63)),
          (SHL64rCL GR64:$src1)>;
def : Pat<(store (shl (loadi64 addr:$dst), (and CL, 63)), addr:$dst),
          (SHL64mCL addr:$dst)>;

def : Pat<(srl GR64:$src1, (and CL, 63)),
          (SHR64rCL GR64:$src1)>;
def : Pat<(store (srl (loadi64 addr:$dst), (and CL, 63)), addr:$dst),
          (SHR64mCL addr:$dst)>;

def : Pat<(sra GR64:$src1, (and CL, 63)),
          (SAR64rCL GR64:$src1)>;
def : Pat<(store (sra (loadi64 addr:$dst), (and CL, 63)), addr:$dst),
          (SAR64mCL addr:$dst)>;


// (anyext (setcc_carry)) -> (setcc_carry)
def : Pat<(i16 (anyext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C16r)>;
def : Pat<(i32 (anyext (i8 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C32r)>;
def : Pat<(i32 (anyext (i16 (X86setcc_c X86_COND_B, EFLAGS)))),
          (SETB_C32r)>;




//===----------------------------------------------------------------------===//
// EFLAGS-defining Patterns
//===----------------------------------------------------------------------===//

// add reg, reg
def : Pat<(add GR8 :$src1, GR8 :$src2), (ADD8rr  GR8 :$src1, GR8 :$src2)>;
def : Pat<(add GR16:$src1, GR16:$src2), (ADD16rr GR16:$src1, GR16:$src2)>;
def : Pat<(add GR32:$src1, GR32:$src2), (ADD32rr GR32:$src1, GR32:$src2)>;

// add reg, mem
def : Pat<(add GR8:$src1, (loadi8 addr:$src2)),
          (ADD8rm GR8:$src1, addr:$src2)>;
def : Pat<(add GR16:$src1, (loadi16 addr:$src2)),
          (ADD16rm GR16:$src1, addr:$src2)>;
def : Pat<(add GR32:$src1, (loadi32 addr:$src2)),
          (ADD32rm GR32:$src1, addr:$src2)>;

// add reg, imm
def : Pat<(add GR8 :$src1, imm:$src2), (ADD8ri  GR8:$src1 , imm:$src2)>;
def : Pat<(add GR16:$src1, imm:$src2), (ADD16ri GR16:$src1, imm:$src2)>;
def : Pat<(add GR32:$src1, imm:$src2), (ADD32ri GR32:$src1, imm:$src2)>;
def : Pat<(add GR16:$src1, i16immSExt8:$src2),
          (ADD16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(add GR32:$src1, i32immSExt8:$src2),
          (ADD32ri8 GR32:$src1, i32immSExt8:$src2)>;

// sub reg, reg
def : Pat<(sub GR8 :$src1, GR8 :$src2), (SUB8rr  GR8 :$src1, GR8 :$src2)>;
def : Pat<(sub GR16:$src1, GR16:$src2), (SUB16rr GR16:$src1, GR16:$src2)>;
def : Pat<(sub GR32:$src1, GR32:$src2), (SUB32rr GR32:$src1, GR32:$src2)>;

// sub reg, mem
def : Pat<(sub GR8:$src1, (loadi8 addr:$src2)),
          (SUB8rm GR8:$src1, addr:$src2)>;
def : Pat<(sub GR16:$src1, (loadi16 addr:$src2)),
          (SUB16rm GR16:$src1, addr:$src2)>;
def : Pat<(sub GR32:$src1, (loadi32 addr:$src2)),
          (SUB32rm GR32:$src1, addr:$src2)>;

// sub reg, imm
def : Pat<(sub GR8:$src1, imm:$src2),
          (SUB8ri GR8:$src1, imm:$src2)>;
def : Pat<(sub GR16:$src1, imm:$src2),
          (SUB16ri GR16:$src1, imm:$src2)>;
def : Pat<(sub GR32:$src1, imm:$src2),
          (SUB32ri GR32:$src1, imm:$src2)>;
def : Pat<(sub GR16:$src1, i16immSExt8:$src2),
          (SUB16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(sub GR32:$src1, i32immSExt8:$src2),
          (SUB32ri8 GR32:$src1, i32immSExt8:$src2)>;

// mul reg, reg
def : Pat<(mul GR16:$src1, GR16:$src2),
          (IMUL16rr GR16:$src1, GR16:$src2)>;
def : Pat<(mul GR32:$src1, GR32:$src2),
          (IMUL32rr GR32:$src1, GR32:$src2)>;

// mul reg, mem
def : Pat<(mul GR16:$src1, (loadi16 addr:$src2)),
          (IMUL16rm GR16:$src1, addr:$src2)>;
def : Pat<(mul GR32:$src1, (loadi32 addr:$src2)),
          (IMUL32rm GR32:$src1, addr:$src2)>;

// mul reg, imm
def : Pat<(mul GR16:$src1, imm:$src2),
          (IMUL16rri GR16:$src1, imm:$src2)>;
def : Pat<(mul GR32:$src1, imm:$src2),
          (IMUL32rri GR32:$src1, imm:$src2)>;
def : Pat<(mul GR16:$src1, i16immSExt8:$src2),
          (IMUL16rri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(mul GR32:$src1, i32immSExt8:$src2),
          (IMUL32rri8 GR32:$src1, i32immSExt8:$src2)>;

// reg = mul mem, imm
def : Pat<(mul (loadi16 addr:$src1), imm:$src2),
          (IMUL16rmi addr:$src1, imm:$src2)>;
def : Pat<(mul (loadi32 addr:$src1), imm:$src2),
          (IMUL32rmi addr:$src1, imm:$src2)>;
def : Pat<(mul (loadi16 addr:$src1), i16immSExt8:$src2),
          (IMUL16rmi8 addr:$src1, i16immSExt8:$src2)>;
def : Pat<(mul (loadi32 addr:$src1), i32immSExt8:$src2),
          (IMUL32rmi8 addr:$src1, i32immSExt8:$src2)>;

// Patterns for nodes that do not produce flags, for instructions that do.

// addition
def : Pat<(add GR64:$src1, GR64:$src2),
          (ADD64rr GR64:$src1, GR64:$src2)>;
def : Pat<(add GR64:$src1, i64immSExt8:$src2),
          (ADD64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(add GR64:$src1, i64immSExt32:$src2),
          (ADD64ri32 GR64:$src1, i64immSExt32:$src2)>;
def : Pat<(add GR64:$src1, (loadi64 addr:$src2)),
          (ADD64rm GR64:$src1, addr:$src2)>;

// subtraction
def : Pat<(sub GR64:$src1, GR64:$src2),
          (SUB64rr GR64:$src1, GR64:$src2)>;
def : Pat<(sub GR64:$src1, (loadi64 addr:$src2)),
          (SUB64rm GR64:$src1, addr:$src2)>;
def : Pat<(sub GR64:$src1, i64immSExt8:$src2),
          (SUB64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(sub GR64:$src1, i64immSExt32:$src2),
          (SUB64ri32 GR64:$src1, i64immSExt32:$src2)>;

// Multiply
def : Pat<(mul GR64:$src1, GR64:$src2),
          (IMUL64rr GR64:$src1, GR64:$src2)>;
def : Pat<(mul GR64:$src1, (loadi64 addr:$src2)),
          (IMUL64rm GR64:$src1, addr:$src2)>;
def : Pat<(mul GR64:$src1, i64immSExt8:$src2),
          (IMUL64rri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(mul GR64:$src1, i64immSExt32:$src2),
          (IMUL64rri32 GR64:$src1, i64immSExt32:$src2)>;
def : Pat<(mul (loadi64 addr:$src1), i64immSExt8:$src2),
          (IMUL64rmi8 addr:$src1, i64immSExt8:$src2)>;
def : Pat<(mul (loadi64 addr:$src1), i64immSExt32:$src2),
          (IMUL64rmi32 addr:$src1, i64immSExt32:$src2)>;

// Increment reg.
def : Pat<(add GR8 :$src, 1), (INC8r     GR8 :$src)>;
def : Pat<(add GR16:$src, 1), (INC16r    GR16:$src)>, Requires<[In32BitMode]>;
def : Pat<(add GR16:$src, 1), (INC64_16r GR16:$src)>, Requires<[In64BitMode]>;
def : Pat<(add GR32:$src, 1), (INC32r    GR32:$src)>, Requires<[In32BitMode]>;
def : Pat<(add GR32:$src, 1), (INC64_32r GR32:$src)>, Requires<[In64BitMode]>;
def : Pat<(add GR64:$src, 1), (INC64r    GR64:$src)>;

// Decrement reg.
def : Pat<(add GR8 :$src, -1), (DEC8r     GR8 :$src)>;
def : Pat<(add GR16:$src, -1), (DEC16r    GR16:$src)>, Requires<[In32BitMode]>;
def : Pat<(add GR16:$src, -1), (DEC64_16r GR16:$src)>, Requires<[In64BitMode]>;
def : Pat<(add GR32:$src, -1), (DEC32r    GR32:$src)>, Requires<[In32BitMode]>;
def : Pat<(add GR32:$src, -1), (DEC64_32r GR32:$src)>, Requires<[In64BitMode]>;
def : Pat<(add GR64:$src, -1), (DEC64r    GR64:$src)>;

// or reg/reg.
def : Pat<(or GR8 :$src1, GR8 :$src2), (OR8rr  GR8 :$src1, GR8 :$src2)>;
def : Pat<(or GR16:$src1, GR16:$src2), (OR16rr GR16:$src1, GR16:$src2)>;
def : Pat<(or GR32:$src1, GR32:$src2), (OR32rr GR32:$src1, GR32:$src2)>;
def : Pat<(or GR64:$src1, GR64:$src2), (OR64rr GR64:$src1, GR64:$src2)>;

// or reg/mem
def : Pat<(or GR8:$src1, (loadi8 addr:$src2)),
          (OR8rm GR8:$src1, addr:$src2)>;
def : Pat<(or GR16:$src1, (loadi16 addr:$src2)),
          (OR16rm GR16:$src1, addr:$src2)>;
def : Pat<(or GR32:$src1, (loadi32 addr:$src2)),
          (OR32rm GR32:$src1, addr:$src2)>;
def : Pat<(or GR64:$src1, (loadi64 addr:$src2)),
          (OR64rm GR64:$src1, addr:$src2)>;

// or reg/imm
def : Pat<(or GR8:$src1 , imm:$src2), (OR8ri  GR8 :$src1, imm:$src2)>;
def : Pat<(or GR16:$src1, imm:$src2), (OR16ri GR16:$src1, imm:$src2)>;
def : Pat<(or GR32:$src1, imm:$src2), (OR32ri GR32:$src1, imm:$src2)>;
def : Pat<(or GR16:$src1, i16immSExt8:$src2),
          (OR16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(or GR32:$src1, i32immSExt8:$src2),
          (OR32ri8 GR32:$src1, i32immSExt8:$src2)>;
def : Pat<(or GR64:$src1, i64immSExt8:$src2),
          (OR64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(or GR64:$src1, i64immSExt32:$src2),
          (OR64ri32 GR64:$src1, i64immSExt32:$src2)>;

// xor reg/reg
def : Pat<(xor GR8 :$src1, GR8 :$src2), (XOR8rr  GR8 :$src1, GR8 :$src2)>;
def : Pat<(xor GR16:$src1, GR16:$src2), (XOR16rr GR16:$src1, GR16:$src2)>;
def : Pat<(xor GR32:$src1, GR32:$src2), (XOR32rr GR32:$src1, GR32:$src2)>;
def : Pat<(xor GR64:$src1, GR64:$src2), (XOR64rr GR64:$src1, GR64:$src2)>;

// xor reg/mem
def : Pat<(xor GR8:$src1, (loadi8 addr:$src2)),
          (XOR8rm GR8:$src1, addr:$src2)>;
def : Pat<(xor GR16:$src1, (loadi16 addr:$src2)),
          (XOR16rm GR16:$src1, addr:$src2)>;
def : Pat<(xor GR32:$src1, (loadi32 addr:$src2)),
          (XOR32rm GR32:$src1, addr:$src2)>;
def : Pat<(xor GR64:$src1, (loadi64 addr:$src2)),
          (XOR64rm GR64:$src1, addr:$src2)>;

// xor reg/imm
def : Pat<(xor GR8:$src1, imm:$src2),
          (XOR8ri GR8:$src1, imm:$src2)>;
def : Pat<(xor GR16:$src1, imm:$src2),
          (XOR16ri GR16:$src1, imm:$src2)>;
def : Pat<(xor GR32:$src1, imm:$src2),
          (XOR32ri GR32:$src1, imm:$src2)>;
def : Pat<(xor GR16:$src1, i16immSExt8:$src2),
          (XOR16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(xor GR32:$src1, i32immSExt8:$src2),
          (XOR32ri8 GR32:$src1, i32immSExt8:$src2)>;
def : Pat<(xor GR64:$src1, i64immSExt8:$src2),
          (XOR64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(xor GR64:$src1, i64immSExt32:$src2),
          (XOR64ri32 GR64:$src1, i64immSExt32:$src2)>;

// and reg/reg
def : Pat<(and GR8 :$src1, GR8 :$src2), (AND8rr  GR8 :$src1, GR8 :$src2)>;
def : Pat<(and GR16:$src1, GR16:$src2), (AND16rr GR16:$src1, GR16:$src2)>;
def : Pat<(and GR32:$src1, GR32:$src2), (AND32rr GR32:$src1, GR32:$src2)>;
def : Pat<(and GR64:$src1, GR64:$src2), (AND64rr GR64:$src1, GR64:$src2)>;

// and reg/mem
def : Pat<(and GR8:$src1, (loadi8 addr:$src2)),
          (AND8rm GR8:$src1, addr:$src2)>;
def : Pat<(and GR16:$src1, (loadi16 addr:$src2)),
          (AND16rm GR16:$src1, addr:$src2)>;
def : Pat<(and GR32:$src1, (loadi32 addr:$src2)),
          (AND32rm GR32:$src1, addr:$src2)>;
def : Pat<(and GR64:$src1, (loadi64 addr:$src2)),
          (AND64rm GR64:$src1, addr:$src2)>;

// and reg/imm
def : Pat<(and GR8:$src1, imm:$src2),
          (AND8ri GR8:$src1, imm:$src2)>;
def : Pat<(and GR16:$src1, imm:$src2),
          (AND16ri GR16:$src1, imm:$src2)>;
def : Pat<(and GR32:$src1, imm:$src2),
          (AND32ri GR32:$src1, imm:$src2)>;
def : Pat<(and GR16:$src1, i16immSExt8:$src2),
          (AND16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(and GR32:$src1, i32immSExt8:$src2),
          (AND32ri8 GR32:$src1, i32immSExt8:$src2)>;
def : Pat<(and GR64:$src1, i64immSExt8:$src2),
          (AND64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(and GR64:$src1, i64immSExt32:$src2),
          (AND64ri32 GR64:$src1, i64immSExt32:$src2)>;
