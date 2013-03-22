//===-- PPCInstr64Bit.td - The PowerPC 64-bit Support ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  let EncoderMethod = "getHA16Encoding";
}
def symbolLo64 : Operand<i64> {
  let PrintMethod = "printSymbolLo";
  let EncoderMethod = "getLO16Encoding";
}
def tocentry : Operand<iPTR> {
  let MIOperandInfo = (ops i64imm:$imm);
}
def memrs : Operand<iPTR> {   // memri where the immediate is a symbolLo64
  let PrintMethod = "printMemRegImm";
  let EncoderMethod = "getMemRIXEncoding";
  let MIOperandInfo = (ops symbolLo64:$off, ptr_rc_nor0:$reg);
}
def tlsreg : Operand<i64> {
  let EncoderMethod = "getTLSRegEncoding";
}
def tlsgd : Operand<i64> {}

//===----------------------------------------------------------------------===//
// 64-bit transformation functions.
//

def SHL64 : SDNodeXForm<imm, [{
  // Transformation function: 63 - imm
  return getI32Imm(63 - N->getZExtValue());
}]>;

def SRL64 : SDNodeXForm<imm, [{
  // Transformation function: 64 - imm
  return N->getZExtValue() ? getI32Imm(64 - N->getZExtValue()) : getI32Imm(0);
}]>;

def HI32_48 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)(N->getZExtValue() >> 32));
}]>;

def HI48_64 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)(N->getZExtValue() >> 48));
}]>;


//===----------------------------------------------------------------------===//
// Calls.
//

let Defs = [LR8] in
  def MovePCtoLR8 : Pseudo<(outs), (ins), "#MovePCtoLR8", []>,
                    PPC970_Unit_BRU;

// Darwin ABI Calls.
let isCall = 1, PPC970_Unit = 7, Defs = [LR8] in {
  // Convenient aliases for call instructions
  let Uses = [RM] in {
    def BL8_Darwin  : IForm<18, 0, 1,
                            (outs), (ins calltarget:$func),
                            "bl $func", BrB, []>;  // See Pat patterns below.
    def BLA8_Darwin : IForm<18, 1, 1,
                          (outs), (ins aaddr:$func),
                          "bla $func", BrB, [(PPCcall_Darwin (i64 imm:$func))]>;
  }
  let Uses = [CTR8, RM] in {
    def BCTRL8_Darwin : XLForm_2_ext<19, 528, 20, 0, 1, 
                                  (outs), (ins),
                                  "bctrl", BrB,
                                  [(PPCbctrl_Darwin)]>, Requires<[In64BitMode]>;
  }
}

// ELF 64 ABI Calls = Darwin ABI Calls
// Used to define BL8_ELF and BLA8_ELF
let isCall = 1, PPC970_Unit = 7, Defs = [LR8] in {
  // Convenient aliases for call instructions
  let Uses = [RM] in {
    def BL8_ELF  : IForm<18, 0, 1,
                         (outs), (ins calltarget:$func),
                         "bl $func", BrB, []>;  // See Pat patterns below.

    let isCodeGenOnly = 1 in
    def BL8_NOP_ELF  : IForm_and_DForm_4_zero<18, 0, 1, 24,
                             (outs), (ins calltarget:$func),
                             "bl $func\n\tnop", BrB, []>;

    let isCodeGenOnly = 1 in
    def BL8_NOP_ELF_TLSGD : IForm_and_DForm_4_zero<18, 0, 1, 24,
                                  (outs), (ins calltarget:$func, tlsgd:$sym),
                                  "bl $func($sym)\n\tnop", BrB, []>;

    let isCodeGenOnly = 1 in
    def BL8_NOP_ELF_TLSLD : IForm_and_DForm_4_zero<18, 0, 1, 24,
                                  (outs), (ins calltarget:$func, tlsgd:$sym),
                                  "bl $func($sym)\n\tnop", BrB, []>;

    def BLA8_ELF : IForm<18, 1, 1,
                         (outs), (ins aaddr:$func),
                         "bla $func", BrB, [(PPCcall_SVR4 (i64 imm:$func))]>;

    let isCodeGenOnly = 1 in
    def BLA8_NOP_ELF : IForm_and_DForm_4_zero<18, 1, 1, 24,
                             (outs), (ins aaddr:$func),
                             "bla $func\n\tnop", BrB,
                             [(PPCcall_nop_SVR4 (i64 imm:$func))]>;
  }
  let Uses = [X11, CTR8, RM] in {
    def BCTRL8_ELF : XLForm_2_ext<19, 528, 20, 0, 1,
                               (outs), (ins),
                               "bctrl", BrB,
                               [(PPCbctrl_SVR4)]>, Requires<[In64BitMode]>;
  }
}


// Calls
def : Pat<(PPCcall_Darwin (i64 tglobaladdr:$dst)),
          (BL8_Darwin tglobaladdr:$dst)>;
def : Pat<(PPCcall_Darwin (i64 texternalsym:$dst)),
          (BL8_Darwin texternalsym:$dst)>;

def : Pat<(PPCcall_SVR4 (i64 tglobaladdr:$dst)),
          (BL8_ELF tglobaladdr:$dst)>;
def : Pat<(PPCcall_nop_SVR4 (i64 tglobaladdr:$dst)),
          (BL8_NOP_ELF tglobaladdr:$dst)>;

def : Pat<(PPCcall_SVR4 (i64 texternalsym:$dst)),
          (BL8_ELF texternalsym:$dst)>;
def : Pat<(PPCcall_nop_SVR4 (i64 texternalsym:$dst)),
          (BL8_NOP_ELF texternalsym:$dst)>;

def : Pat<(PPCnop),
          (NOP)>;

// Atomic operations
let usesCustomInserter = 1 in {
  let Defs = [CR0] in {
    def ATOMIC_LOAD_ADD_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr), "#ATOMIC_LOAD_ADD_I64",
      [(set G8RC:$dst, (atomic_load_add_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_SUB_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr), "#ATOMIC_LOAD_SUB_I64",
      [(set G8RC:$dst, (atomic_load_sub_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_OR_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr), "#ATOMIC_LOAD_OR_I64",
      [(set G8RC:$dst, (atomic_load_or_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_XOR_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr), "#ATOMIC_LOAD_XOR_I64",
      [(set G8RC:$dst, (atomic_load_xor_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_AND_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr), "#ATOMIC_LOAD_AND_i64",
      [(set G8RC:$dst, (atomic_load_and_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_NAND_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr), "#ATOMIC_LOAD_NAND_I64",
      [(set G8RC:$dst, (atomic_load_nand_64 xoaddr:$ptr, G8RC:$incr))]>;

    def ATOMIC_CMP_SWAP_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$old, G8RC:$new), "#ATOMIC_CMP_SWAP_I64",
      [(set G8RC:$dst, 
                    (atomic_cmp_swap_64 xoaddr:$ptr, G8RC:$old, G8RC:$new))]>;

    def ATOMIC_SWAP_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$new), "#ATOMIC_SWAP_I64",
      [(set G8RC:$dst, (atomic_swap_64 xoaddr:$ptr, G8RC:$new))]>;
  }
}

// Instructions to support atomic operations
def LDARX : XForm_1<31,  84, (outs G8RC:$rD), (ins memrr:$ptr),
                   "ldarx $rD, $ptr", LdStLDARX,
                   [(set G8RC:$rD, (PPClarx xoaddr:$ptr))]>;

let Defs = [CR0] in
def STDCX : XForm_1<31, 214, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stdcx. $rS, $dst", LdStSTDCX,
                   [(PPCstcx G8RC:$rS, xoaddr:$dst)]>,
                   isDOT;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNdi8 :Pseudo< (outs),
                        (ins calltarget:$dst, i32imm:$offset),
                 "#TC_RETURNd8 $dst $offset",
                 []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNai8 :Pseudo<(outs), (ins aaddr:$func, i32imm:$offset),
                 "#TC_RETURNa8 $func $offset",
                 [(PPCtc_return (i64 imm:$func), imm:$offset)]>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNri8 : Pseudo<(outs), (ins CTRRC8:$dst, i32imm:$offset),
                 "#TC_RETURNr8 $dst $offset",
                 []>;


let isTerminator = 1, isBarrier = 1, PPC970_Unit = 7, isBranch = 1,
    isIndirectBranch = 1, isCall = 1, Uses = [CTR8, RM] in {
  let isReturn = 1 in {
    def TAILBCTR8 : XLForm_2_ext<19, 528, 20, 0, 0, (outs), (ins), "bctr", BrB, []>,
        Requires<[In64BitMode]>;
  }

  def BCTR8 : XLForm_2_ext<19, 528, 20, 0, 0, (outs), (ins), "bctr", BrB, []>,
      Requires<[In64BitMode]>;
}


let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7,
    isBarrier = 1, isCall = 1, isReturn = 1, Uses = [RM] in
def TAILB8   : IForm<18, 0, 0, (outs), (ins calltarget:$dst),
                  "b $dst", BrB,
                  []>;


let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7,
    isBarrier = 1, isCall = 1, isReturn = 1, Uses = [RM] in
def TAILBA8   : IForm<18, 0, 0, (outs), (ins aaddr:$dst),
                  "ba $dst", BrB,
                  []>;

def : Pat<(PPCtc_return (i64 tglobaladdr:$dst),  imm:$imm),
          (TCRETURNdi8 tglobaladdr:$dst, imm:$imm)>;

def : Pat<(PPCtc_return (i64 texternalsym:$dst), imm:$imm),
          (TCRETURNdi8 texternalsym:$dst, imm:$imm)>;

def : Pat<(PPCtc_return CTRRC8:$dst, imm:$imm),
          (TCRETURNri8 CTRRC8:$dst, imm:$imm)>;

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7 in {
  let Defs = [CTR8], Uses = [CTR8] in {
    def BDZ8  : BForm_1<16, 18, 0, 0, (outs), (ins condbrtarget:$dst),
                        "bdz $dst">;
    def BDNZ8 : BForm_1<16, 16, 0, 0, (outs), (ins condbrtarget:$dst),
                        "bdnz $dst">;
  }
}

// 64-but CR instructions
def MTCRF8 : XFXForm_5<31, 144, (outs crbitm:$FXM), (ins G8RC:$rS),
                      "mtcrf $FXM, $rS", BrMCRX>,
            PPC970_MicroCode, PPC970_Unit_CRU;

def MFCR8pseud: XFXForm_3<31, 19, (outs G8RC:$rT), (ins crbitm:$FXM),
                       "#MFCR8pseud", SprMFCR>,
            PPC970_MicroCode, PPC970_Unit_CRU;
            
def MFCR8 : XFXForm_3<31, 19, (outs G8RC:$rT), (ins),
                     "mfcr $rT", SprMFCR>,
                     PPC970_MicroCode, PPC970_Unit_CRU;

let hasSideEffects = 1, isBarrier = 1, isCodeGenOnly = 1,
    usesCustomInserter = 1 in {
  def EH_SjLj_SetJmp64  : Pseudo<(outs GPRC:$dst), (ins memr:$buf),
                            "#EH_SJLJ_SETJMP64",
                            [(set GPRC:$dst, (PPCeh_sjlj_setjmp addr:$buf))]>,
                          Requires<[In64BitMode]>;
  let isTerminator = 1 in
  def EH_SjLj_LongJmp64 : Pseudo<(outs), (ins memr:$buf),
                            "#EH_SJLJ_LONGJMP64",
                            [(PPCeh_sjlj_longjmp addr:$buf)]>,
                          Requires<[In64BitMode]>;
}

//===----------------------------------------------------------------------===//
// 64-bit SPR manipulation instrs.

let Uses = [CTR8] in {
def MFCTR8 : XFXForm_1_ext<31, 339, 9, (outs G8RC:$rT), (ins),
                           "mfctr $rT", SprMFSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let Pattern = [(PPCmtctr G8RC:$rS)], Defs = [CTR8] in {
def MTCTR8 : XFXForm_7_ext<31, 467, 9, (outs), (ins G8RC:$rS),
                           "mtctr $rS", SprMTSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}

let Pattern = [(set G8RC:$rT, readcyclecounter)] in
def MFTB8 : XFXForm_1_ext<31, 339, 268, (outs G8RC:$rT), (ins),
                          "mfspr $rT, 268", SprMFTB>,
            PPC970_DGroup_First, PPC970_Unit_FXU;
// Note that encoding mftb using mfspr is now the preferred form,
// and has been since at least ISA v2.03. The mftb instruction has
// now been phased out. Using mfspr, however, is known not to work on
// the POWER3.

let Defs = [X1], Uses = [X1] in
def DYNALLOC8 : Pseudo<(outs G8RC:$result), (ins G8RC:$negsize, memri:$fpsi),"#DYNALLOC8",
                       [(set G8RC:$result,
                             (PPCdynalloc G8RC:$negsize, iaddr:$fpsi))]>;

let Defs = [LR8] in {
def MTLR8  : XFXForm_7_ext<31, 467, 8, (outs), (ins G8RC:$rS),
                           "mtlr $rS", SprMTSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let Uses = [LR8] in {
def MFLR8  : XFXForm_1_ext<31, 339, 8, (outs G8RC:$rT), (ins),
                           "mflr $rT", SprMFSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}

//===----------------------------------------------------------------------===//
// Fixed point instructions.
//

let PPC970_Unit = 1 in {  // FXU Operations.

let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
def LI8  : DForm_2_r0<14, (outs G8RC:$rD), (ins symbolLo64:$imm),
                      "li $rD, $imm", IntSimple,
                      [(set G8RC:$rD, immSExt16:$imm)]>;
def LIS8 : DForm_2_r0<15, (outs G8RC:$rD), (ins symbolHi64:$imm),
                      "lis $rD, $imm", IntSimple,
                      [(set G8RC:$rD, imm16ShiftedSExt:$imm)]>;
}

// Logical ops.
def NAND8: XForm_6<31, 476, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "nand $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (not (and G8RC:$rS, G8RC:$rB)))]>;
def AND8 : XForm_6<31,  28, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "and $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (and G8RC:$rS, G8RC:$rB))]>;
def ANDC8: XForm_6<31,  60, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "andc $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (and G8RC:$rS, (not G8RC:$rB)))]>;
def OR8  : XForm_6<31, 444, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (or G8RC:$rS, G8RC:$rB))]>;
def NOR8 : XForm_6<31, 124, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "nor $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (not (or G8RC:$rS, G8RC:$rB)))]>;
def ORC8 : XForm_6<31, 412, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "orc $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (or G8RC:$rS, (not G8RC:$rB)))]>;
def EQV8 : XForm_6<31, 284, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "eqv $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (not (xor G8RC:$rS, G8RC:$rB)))]>;
def XOR8 : XForm_6<31, 316, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "xor $rA, $rS, $rB", IntSimple,
                   [(set G8RC:$rA, (xor G8RC:$rS, G8RC:$rB))]>;

// Logical ops with immediate.
def ANDIo8  : DForm_4<28, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "andi. $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (and G8RC:$src1, immZExt16:$src2))]>,
                      isDOT;
def ANDISo8 : DForm_4<29, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                     "andis. $dst, $src1, $src2", IntGeneral,
                    [(set G8RC:$dst, (and G8RC:$src1,imm16ShiftedZExt:$src2))]>,
                     isDOT;
def ORI8    : DForm_4<24, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "ori $dst, $src1, $src2", IntSimple,
                      [(set G8RC:$dst, (or G8RC:$src1, immZExt16:$src2))]>;
def ORIS8   : DForm_4<25, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "oris $dst, $src1, $src2", IntSimple,
                    [(set G8RC:$dst, (or G8RC:$src1, imm16ShiftedZExt:$src2))]>;
def XORI8   : DForm_4<26, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "xori $dst, $src1, $src2", IntSimple,
                      [(set G8RC:$dst, (xor G8RC:$src1, immZExt16:$src2))]>;
def XORIS8  : DForm_4<27, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "xoris $dst, $src1, $src2", IntSimple,
                   [(set G8RC:$dst, (xor G8RC:$src1, imm16ShiftedZExt:$src2))]>;

def ADD8  : XOForm_1<31, 266, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "add $rT, $rA, $rB", IntSimple,
                     [(set G8RC:$rT, (add G8RC:$rA, G8RC:$rB))]>;
// ADD8 has a special form: reg = ADD8(reg, sym@tls) for use by the
// initial-exec thread-local storage model.
def ADD8TLS  : XOForm_1<31, 266, 0, (outs G8RC:$rT), (ins G8RC:$rA, tlsreg:$rB),
                        "add $rT, $rA, $rB@tls", IntSimple,
                        [(set G8RC:$rT, (add G8RC:$rA, tglobaltlsaddr:$rB))]>;
                     
let Defs = [CARRY] in {
def ADDC8 : XOForm_1<31, 10, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "addc $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (addc G8RC:$rA, G8RC:$rB))]>,
                     PPC970_DGroup_Cracked;
def ADDIC8 : DForm_2<12, (outs G8RC:$rD), (ins G8RC:$rA, s16imm64:$imm),
                     "addic $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (addc G8RC:$rA, immSExt16:$imm))]>;
}
def ADDI8  : DForm_2<14, (outs G8RC:$rD), (ins G8RC_NOX0:$rA, s16imm64:$imm),
                     "addi $rD, $rA, $imm", IntSimple,
                     [(set G8RC:$rD, (add G8RC_NOX0:$rA, immSExt16:$imm))]>;
def ADDI8L  : DForm_2<14, (outs G8RC:$rD), (ins G8RC_NOX0:$rA, symbolLo64:$imm),
                     "addi $rD, $rA, $imm", IntSimple,
                     [(set G8RC:$rD, (add G8RC_NOX0:$rA, immSExt16:$imm))]>;
def ADDIS8 : DForm_2<15, (outs G8RC:$rD), (ins G8RC_NOX0:$rA, symbolHi64:$imm),
                     "addis $rD, $rA, $imm", IntSimple,
                     [(set G8RC:$rD, (add G8RC_NOX0:$rA,
                                          imm16ShiftedSExt:$imm))]>;

let Defs = [CARRY] in {
def SUBFIC8: DForm_2< 8, (outs G8RC:$rD), (ins G8RC:$rA, s16imm64:$imm),
                     "subfic $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (subc immSExt16:$imm, G8RC:$rA))]>;
def SUBFC8 : XOForm_1<31, 8, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                      "subfc $rT, $rA, $rB", IntGeneral,
                      [(set G8RC:$rT, (subc G8RC:$rB, G8RC:$rA))]>,
                      PPC970_DGroup_Cracked;
}
def SUBF8 : XOForm_1<31, 40, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "subf $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (sub G8RC:$rB, G8RC:$rA))]>;
def NEG8    : XOForm_3<31, 104, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "neg $rT, $rA", IntSimple,
                       [(set G8RC:$rT, (ineg G8RC:$rA))]>;
let Uses = [CARRY], Defs = [CARRY] in {
def ADDE8   : XOForm_1<31, 138, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                       "adde $rT, $rA, $rB", IntGeneral,
                       [(set G8RC:$rT, (adde G8RC:$rA, G8RC:$rB))]>;
def ADDME8  : XOForm_3<31, 234, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "addme $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (adde G8RC:$rA, -1))]>;
def ADDZE8  : XOForm_3<31, 202, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "addze $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (adde G8RC:$rA, 0))]>;
def SUBFE8  : XOForm_1<31, 136, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                       "subfe $rT, $rA, $rB", IntGeneral,
                       [(set G8RC:$rT, (sube G8RC:$rB, G8RC:$rA))]>;
def SUBFME8 : XOForm_3<31, 232, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "subfme $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (sube -1, G8RC:$rA))]>;
def SUBFZE8 : XOForm_3<31, 200, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "subfze $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (sube 0, G8RC:$rA))]>;
}


def MULHD : XOForm_1<31, 73, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "mulhd $rT, $rA, $rB", IntMulHW,
                     [(set G8RC:$rT, (mulhs G8RC:$rA, G8RC:$rB))]>;
def MULHDU : XOForm_1<31, 9, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "mulhdu $rT, $rA, $rB", IntMulHWU,
                     [(set G8RC:$rT, (mulhu G8RC:$rA, G8RC:$rB))]>;

def CMPD   : XForm_16_ext<31, 0, (outs CRRC:$crD), (ins G8RC:$rA, G8RC:$rB),
                          "cmpd $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPLD  : XForm_16_ext<31, 32, (outs CRRC:$crD), (ins G8RC:$rA, G8RC:$rB),
                          "cmpld $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPDI  : DForm_5_ext<11, (outs CRRC:$crD), (ins G8RC:$rA, s16imm:$imm),
                         "cmpdi $crD, $rA, $imm", IntCompare>, isPPC64;
def CMPLDI : DForm_6_ext<10, (outs CRRC:$dst), (ins G8RC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2", IntCompare>, isPPC64;

def SLD  : XForm_6<31,  27, (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB),
                   "sld $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (PPCshl G8RC:$rS, GPRC:$rB))]>, isPPC64;
def SRD  : XForm_6<31, 539, (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB),
                   "srd $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (PPCsrl G8RC:$rS, GPRC:$rB))]>, isPPC64;
let Defs = [CARRY] in {
def SRAD : XForm_6<31, 794, (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB),
                   "srad $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (PPCsra G8RC:$rS, GPRC:$rB))]>, isPPC64;
}
                   
def EXTSB8 : XForm_11<31, 954, (outs G8RC:$rA), (ins G8RC:$rS),
                      "extsb $rA, $rS", IntSimple,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i8))]>;
def EXTSH8 : XForm_11<31, 922, (outs G8RC:$rA), (ins G8RC:$rS),
                      "extsh $rA, $rS", IntSimple,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i16))]>;

def EXTSW  : XForm_11<31, 986, (outs G8RC:$rA), (ins G8RC:$rS),
                      "extsw $rA, $rS", IntSimple,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i32))]>, isPPC64;
/// EXTSW_32 - Just like EXTSW, but works on '32-bit' registers.
def EXTSW_32 : XForm_11<31, 986, (outs GPRC:$rA), (ins GPRC:$rS),
                      "extsw $rA, $rS", IntSimple,
                      [(set GPRC:$rA, (PPCextsw_32 GPRC:$rS))]>, isPPC64;
def EXTSW_32_64 : XForm_11<31, 986, (outs G8RC:$rA), (ins GPRC:$rS),
                      "extsw $rA, $rS", IntSimple,
                      [(set G8RC:$rA, (sext GPRC:$rS))]>, isPPC64;

let Defs = [CARRY] in {
def SRADI  : XSForm_1<31, 413, (outs G8RC:$rA), (ins G8RC:$rS, u6imm:$SH),
                      "sradi $rA, $rS, $SH", IntRotateDI,
                      [(set G8RC:$rA, (sra G8RC:$rS, (i32 imm:$SH)))]>, isPPC64;
}
def CNTLZD : XForm_11<31, 58, (outs G8RC:$rA), (ins G8RC:$rS),
                      "cntlzd $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (ctlz G8RC:$rS))]>;

def DIVD  : XOForm_1<31, 489, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "divd $rT, $rA, $rB", IntDivD,
                     [(set G8RC:$rT, (sdiv G8RC:$rA, G8RC:$rB))]>, isPPC64,
                     PPC970_DGroup_First, PPC970_DGroup_Cracked;
def DIVDU : XOForm_1<31, 457, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "divdu $rT, $rA, $rB", IntDivD,
                     [(set G8RC:$rT, (udiv G8RC:$rA, G8RC:$rB))]>, isPPC64,
                     PPC970_DGroup_First, PPC970_DGroup_Cracked;
def MULLD : XOForm_1<31, 233, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "mulld $rT, $rA, $rB", IntMulHD,
                     [(set G8RC:$rT, (mul G8RC:$rA, G8RC:$rB))]>, isPPC64;


let isCommutable = 1 in {
def RLDIMI : MDForm_1<30, 3,
                      (outs G8RC:$rA), (ins G8RC:$rSi, G8RC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldimi $rA, $rS, $SH, $MB", IntRotateDI,
                      []>, isPPC64, RegConstraint<"$rSi = $rA">,
                      NoEncode<"$rSi">;
}

// Rotate instructions.
def RLDCL  : MDForm_1<30, 0,
                      (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB, u6imm:$MBE),
                      "rldcl $rA, $rS, $rB, $MBE", IntRotateD,
                      []>, isPPC64;
def RLDICL : MDForm_1<30, 0,
                      (outs G8RC:$rA), (ins G8RC:$rS, u6imm:$SH, u6imm:$MBE),
                      "rldicl $rA, $rS, $SH, $MBE", IntRotateDI,
                      []>, isPPC64;
def RLDICR : MDForm_1<30, 1,
                      (outs G8RC:$rA), (ins G8RC:$rS, u6imm:$SH, u6imm:$MBE),
                      "rldicr $rA, $rS, $SH, $MBE", IntRotateDI,
                      []>, isPPC64;

def RLWINM8 : MForm_2<21,
                     (outs G8RC:$rA), (ins G8RC:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                     "rlwinm $rA, $rS, $SH, $MB, $ME", IntGeneral,
                     []>;

def ISEL8   : AForm_4<31, 15,
                     (outs G8RC:$rT), (ins G8RC_NOX0:$rA, G8RC:$rB, pred:$cond),
                     "isel $rT, $rA, $rB, $cond", IntGeneral,
                     []>;
}  // End FXU Operations.


//===----------------------------------------------------------------------===//
// Load/Store instructions.
//


// Sign extending loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LHA8: DForm_1<42, (outs G8RC:$rD), (ins memri:$src),
                  "lha $rD, $src", LdStLHA,
                  [(set G8RC:$rD, (sextloadi16 iaddr:$src))]>,
                  PPC970_DGroup_Cracked;
def LWA  : DSForm_1<58, 2, (outs G8RC:$rD), (ins memrix:$src),
                    "lwa $rD, $src", LdStLWA,
                    [(set G8RC:$rD,
                          (aligned4sextloadi32 ixaddr:$src))]>, isPPC64,
                    PPC970_DGroup_Cracked;
def LHAX8: XForm_1<31, 343, (outs G8RC:$rD), (ins memrr:$src),
                   "lhax $rD, $src", LdStLHA,
                   [(set G8RC:$rD, (sextloadi16 xaddr:$src))]>,
                   PPC970_DGroup_Cracked;
def LWAX : XForm_1<31, 341, (outs G8RC:$rD), (ins memrr:$src),
                   "lwax $rD, $src", LdStLHA,
                   [(set G8RC:$rD, (sextloadi32 xaddr:$src))]>, isPPC64,
                   PPC970_DGroup_Cracked;

// Update forms.
let mayLoad = 1 in {
def LHAU8 : DForm_1<43, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                    (ins memri:$addr),
                    "lhau $rD, $addr", LdStLHAU,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
// NO LWAU!

def LHAUX8 : XForm_1<31, 375, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                    (ins memrr:$addr),
                    "lhaux $rD, $addr", LdStLHAU,
                    []>, RegConstraint<"$addr.offreg = $ea_result">,
                    NoEncode<"$ea_result">;
def LWAUX : XForm_1<31, 373, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                    (ins memrr:$addr),
                    "lwaux $rD, $addr", LdStLHAU,
                    []>, RegConstraint<"$addr.offreg = $ea_result">,
                    NoEncode<"$ea_result">, isPPC64;
}
}

// Zero extending loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LBZ8 : DForm_1<34, (outs G8RC:$rD), (ins memri:$src),
                  "lbz $rD, $src", LdStLoad,
                  [(set G8RC:$rD, (zextloadi8 iaddr:$src))]>;
def LHZ8 : DForm_1<40, (outs G8RC:$rD), (ins memri:$src),
                  "lhz $rD, $src", LdStLoad,
                  [(set G8RC:$rD, (zextloadi16 iaddr:$src))]>;
def LWZ8 : DForm_1<32, (outs G8RC:$rD), (ins memri:$src),
                  "lwz $rD, $src", LdStLoad,
                  [(set G8RC:$rD, (zextloadi32 iaddr:$src))]>, isPPC64;

def LBZX8 : XForm_1<31,  87, (outs G8RC:$rD), (ins memrr:$src),
                   "lbzx $rD, $src", LdStLoad,
                   [(set G8RC:$rD, (zextloadi8 xaddr:$src))]>;
def LHZX8 : XForm_1<31, 279, (outs G8RC:$rD), (ins memrr:$src),
                   "lhzx $rD, $src", LdStLoad,
                   [(set G8RC:$rD, (zextloadi16 xaddr:$src))]>;
def LWZX8 : XForm_1<31,  23, (outs G8RC:$rD), (ins memrr:$src),
                   "lwzx $rD, $src", LdStLoad,
                   [(set G8RC:$rD, (zextloadi32 xaddr:$src))]>;
                   
                   
// Update forms.
let mayLoad = 1 in {
def LBZU8 : DForm_1<35, (outs G8RC:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                    "lbzu $rD, $addr", LdStLoadUpd,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
def LHZU8 : DForm_1<41, (outs G8RC:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                    "lhzu $rD, $addr", LdStLoadUpd,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
def LWZU8 : DForm_1<33, (outs G8RC:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                    "lwzu $rD, $addr", LdStLoadUpd,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;

def LBZUX8 : XForm_1<31, 119, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lbzux $rD, $addr", LdStLoadUpd,
                   []>, RegConstraint<"$addr.offreg = $ea_result">,
                   NoEncode<"$ea_result">;
def LHZUX8 : XForm_1<31, 311, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lhzux $rD, $addr", LdStLoadUpd,
                   []>, RegConstraint<"$addr.offreg = $ea_result">,
                   NoEncode<"$ea_result">;
def LWZUX8 : XForm_1<31, 55, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lwzux $rD, $addr", LdStLoadUpd,
                   []>, RegConstraint<"$addr.offreg = $ea_result">,
                   NoEncode<"$ea_result">;
}
}


// Full 8-byte loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LD   : DSForm_1<58, 0, (outs G8RC:$rD), (ins memrix:$src),
                    "ld $rD, $src", LdStLD,
                    [(set G8RC:$rD, (aligned4load ixaddr:$src))]>, isPPC64;
def LDrs : DSForm_1<58, 0, (outs G8RC:$rD), (ins memrs:$src),
                    "ld $rD, $src", LdStLD,
                    []>, isPPC64;
// The following three definitions are selected for small code model only.
// Otherwise, we need to create two instructions to form a 32-bit offset,
// so we have a custom matcher for TOC_ENTRY in PPCDAGToDAGIsel::Select().
def LDtoc: Pseudo<(outs G8RC:$rD), (ins tocentry:$disp, G8RC:$reg),
                  "#LDtoc",
                  [(set G8RC:$rD,
                     (PPCtoc_entry tglobaladdr:$disp, G8RC:$reg))]>, isPPC64;
def LDtocJTI: Pseudo<(outs G8RC:$rD), (ins tocentry:$disp, G8RC:$reg),
                  "#LDtocJTI",
                  [(set G8RC:$rD,
                     (PPCtoc_entry tjumptable:$disp, G8RC:$reg))]>, isPPC64;
def LDtocCPT: Pseudo<(outs G8RC:$rD), (ins tocentry:$disp, G8RC:$reg),
                  "#LDtocCPT",
                  [(set G8RC:$rD,
                     (PPCtoc_entry tconstpool:$disp, G8RC:$reg))]>, isPPC64;

let hasSideEffects = 1 in { 
let RST = 2, DS = 2 in
def LDinto_toc: DSForm_1a<58, 0, (outs), (ins G8RC:$reg),
                    "ld 2, 8($reg)", LdStLD,
                    [(PPCload_toc G8RC:$reg)]>, isPPC64;
                    
let RST = 2, DS = 10, RA = 1 in
def LDtoc_restore : DSForm_1a<58, 0, (outs), (ins),
                    "ld 2, 40(1)", LdStLD,
                    [(PPCtoc_restore)]>, isPPC64;
}
def LDX  : XForm_1<31,  21, (outs G8RC:$rD), (ins memrr:$src),
                   "ldx $rD, $src", LdStLD,
                   [(set G8RC:$rD, (load xaddr:$src))]>, isPPC64;
                   
let mayLoad = 1 in
def LDU  : DSForm_1<58, 1, (outs G8RC:$rD, ptr_rc_nor0:$ea_result), (ins memrix:$addr),
                    "ldu $rD, $addr", LdStLDU,
                    []>, RegConstraint<"$addr.reg = $ea_result">, isPPC64,
                    NoEncode<"$ea_result">;

def LDUX : XForm_1<31, 53, (outs G8RC:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "ldux $rD, $addr", LdStLDU,
                   []>, RegConstraint<"$addr.offreg = $ea_result">,
                   NoEncode<"$ea_result">, isPPC64;
}

def : Pat<(PPCload ixaddr:$src),
          (LD ixaddr:$src)>;
def : Pat<(PPCload xaddr:$src),
          (LDX xaddr:$src)>;

// Support for medium and large code model.
def ADDIStocHA: Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, tocentry:$disp),
                       "#ADDIStocHA",
                       [(set G8RC:$rD,
                         (PPCaddisTocHA G8RC:$reg, tglobaladdr:$disp))]>,
                       isPPC64;
def LDtocL: Pseudo<(outs G8RC:$rD), (ins tocentry:$disp, G8RC:$reg),
                   "#LDtocL",
                   [(set G8RC:$rD,
                     (PPCldTocL tglobaladdr:$disp, G8RC:$reg))]>, isPPC64;
def ADDItocL: Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, tocentry:$disp),
                     "#ADDItocL",
                     [(set G8RC:$rD,
                       (PPCaddiTocL G8RC:$reg, tglobaladdr:$disp))]>, isPPC64;

// Support for thread-local storage.
def ADDISgotTprelHA: Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolHi64:$disp),
                         "#ADDISgotTprelHA",
                         [(set G8RC:$rD,
                           (PPCaddisGotTprelHA G8RC:$reg,
                                               tglobaltlsaddr:$disp))]>,
                  isPPC64;
def LDgotTprelL: Pseudo<(outs G8RC:$rD), (ins symbolLo64:$disp, G8RC:$reg),
                        "#LDgotTprelL",
                        [(set G8RC:$rD,
                          (PPCldGotTprelL tglobaltlsaddr:$disp, G8RC:$reg))]>,
                 isPPC64;
def : Pat<(PPCaddTls G8RC:$in, tglobaltlsaddr:$g),
          (ADD8TLS G8RC:$in, tglobaltlsaddr:$g)>;
def ADDIStlsgdHA: Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolHi64:$disp),
                         "#ADDIStlsgdHA",
                         [(set G8RC:$rD,
                           (PPCaddisTlsgdHA G8RC:$reg, tglobaltlsaddr:$disp))]>,
                  isPPC64;
def ADDItlsgdL : Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolLo64:$disp),
                       "#ADDItlsgdL",
                       [(set G8RC:$rD,
                         (PPCaddiTlsgdL G8RC:$reg, tglobaltlsaddr:$disp))]>,
                 isPPC64;
def GETtlsADDR : Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, tlsgd:$sym),
                        "#GETtlsADDR",
                        [(set G8RC:$rD,
                          (PPCgetTlsAddr G8RC:$reg, tglobaltlsaddr:$sym))]>,
                 isPPC64;
def ADDIStlsldHA: Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolHi64:$disp),
                         "#ADDIStlsldHA",
                         [(set G8RC:$rD,
                           (PPCaddisTlsldHA G8RC:$reg, tglobaltlsaddr:$disp))]>,
                  isPPC64;
def ADDItlsldL : Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolLo64:$disp),
                       "#ADDItlsldL",
                       [(set G8RC:$rD,
                         (PPCaddiTlsldL G8RC:$reg, tglobaltlsaddr:$disp))]>,
                 isPPC64;
def GETtlsldADDR : Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, tlsgd:$sym),
                          "#GETtlsldADDR",
                          [(set G8RC:$rD,
                            (PPCgetTlsldAddr G8RC:$reg, tglobaltlsaddr:$sym))]>,
                   isPPC64;
def ADDISdtprelHA: Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolHi64:$disp),
                          "#ADDISdtprelHA",
                          [(set G8RC:$rD,
                            (PPCaddisDtprelHA G8RC:$reg,
                                              tglobaltlsaddr:$disp))]>,
                   isPPC64;
def ADDIdtprelL : Pseudo<(outs G8RC:$rD), (ins G8RC:$reg, symbolLo64:$disp),
                         "#ADDIdtprelL",
                         [(set G8RC:$rD,
                           (PPCaddiDtprelL G8RC:$reg, tglobaltlsaddr:$disp))]>,
                  isPPC64;

let PPC970_Unit = 2 in {
// Truncating stores.                       
def STB8 : DForm_1<38, (outs), (ins G8RC:$rS, memri:$src),
                   "stb $rS, $src", LdStStore,
                   [(truncstorei8 G8RC:$rS, iaddr:$src)]>;
def STH8 : DForm_1<44, (outs), (ins G8RC:$rS, memri:$src),
                   "sth $rS, $src", LdStStore,
                   [(truncstorei16 G8RC:$rS, iaddr:$src)]>;
def STW8 : DForm_1<36, (outs), (ins G8RC:$rS, memri:$src),
                   "stw $rS, $src", LdStStore,
                   [(truncstorei32 G8RC:$rS, iaddr:$src)]>;
def STBX8 : XForm_8<31, 215, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stbx $rS, $dst", LdStStore,
                   [(truncstorei8 G8RC:$rS, xaddr:$dst)]>, 
                   PPC970_DGroup_Cracked;
def STHX8 : XForm_8<31, 407, (outs), (ins G8RC:$rS, memrr:$dst),
                   "sthx $rS, $dst", LdStStore,
                   [(truncstorei16 G8RC:$rS, xaddr:$dst)]>, 
                   PPC970_DGroup_Cracked;
def STWX8 : XForm_8<31, 151, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stwx $rS, $dst", LdStStore,
                   [(truncstorei32 G8RC:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
// Normal 8-byte stores.
def STD  : DSForm_1<62, 0, (outs), (ins G8RC:$rS, memrix:$dst),
                    "std $rS, $dst", LdStSTD,
                    [(aligned4store G8RC:$rS, ixaddr:$dst)]>, isPPC64;
def STDX  : XForm_8<31, 149, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stdx $rS, $dst", LdStSTD,
                   [(store G8RC:$rS, xaddr:$dst)]>, isPPC64,
                   PPC970_DGroup_Cracked;
// STD_32/STDX_32 - Just like STD/STDX, but uses a '32-bit' input register.
def STD_32  : DSForm_1<62, 0, (outs), (ins GPRC:$rT, memrix:$dst),
                       "std $rT, $dst", LdStSTD,
                       [(PPCstd_32  GPRC:$rT, ixaddr:$dst)]>, isPPC64;
def STDX_32  : XForm_8<31, 149, (outs), (ins GPRC:$rT, memrr:$dst),
                       "stdx $rT, $dst", LdStSTD,
                       [(PPCstd_32  GPRC:$rT, xaddr:$dst)]>, isPPC64,
                       PPC970_DGroup_Cracked;
}

// Stores with Update (pre-inc).
let PPC970_Unit = 2, mayStore = 1 in {
def STBU8 : DForm_1<39, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memri:$dst),
                   "stbu $rS, $dst", LdStStoreUpd, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STHU8 : DForm_1<45, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memri:$dst),
                   "sthu $rS, $dst", LdStStoreUpd, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STWU8 : DForm_1<37, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memri:$dst),
                   "stwu $rS, $dst", LdStStoreUpd, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STDU : DSForm_1<62, 1, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memrix:$dst),
                   "stdu $rS, $dst", LdStSTDU, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">,
                   isPPC64;

def STBUX8: XForm_8<31, 247, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memrr:$dst),
                    "stbux $rS, $dst", LdStStoreUpd, []>,
                    RegConstraint<"$dst.offreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked;
def STHUX8: XForm_8<31, 439, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memrr:$dst),
                    "sthux $rS, $dst", LdStStoreUpd, []>,
                    RegConstraint<"$dst.offreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked;
def STWUX8: XForm_8<31, 183, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memrr:$dst),
                    "stwux $rS, $dst", LdStStoreUpd, []>,
                    RegConstraint<"$dst.offreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked;
def STDUX : XForm_8<31, 181, (outs ptr_rc_nor0:$ea_res), (ins G8RC:$rS, memrr:$dst),
                    "stdux $rS, $dst", LdStSTDU, []>,
                    RegConstraint<"$dst.offreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked, isPPC64;
}

// Patterns to match the pre-inc stores.  We can't put the patterns on
// the instruction definitions directly as ISel wants the address base
// and offset to be separate operands, not a single complex operand.
def : Pat<(pre_truncsti8 G8RC:$rS, ptr_rc_nor0:$ptrreg, iaddroff:$ptroff),
          (STBU8 G8RC:$rS, iaddroff:$ptroff, ptr_rc_nor0:$ptrreg)>;
def : Pat<(pre_truncsti16 G8RC:$rS, ptr_rc_nor0:$ptrreg, iaddroff:$ptroff),
          (STHU8 G8RC:$rS, iaddroff:$ptroff, ptr_rc_nor0:$ptrreg)>;
def : Pat<(pre_truncsti32 G8RC:$rS, ptr_rc_nor0:$ptrreg, iaddroff:$ptroff),
          (STWU8 G8RC:$rS, iaddroff:$ptroff, ptr_rc_nor0:$ptrreg)>;
def : Pat<(aligned4pre_store G8RC:$rS, ptr_rc_nor0:$ptrreg, iaddroff:$ptroff),
          (STDU G8RC:$rS, iaddroff:$ptroff, ptr_rc_nor0:$ptrreg)>;

def : Pat<(pre_truncsti8 G8RC:$rS, ptr_rc:$ptrreg, ptr_rc_nor0:$ptroff),
          (STBUX8 G8RC:$rS, ptr_rc_nor0:$ptroff, ptr_rc:$ptrreg)>;
def : Pat<(pre_truncsti16 G8RC:$rS, ptr_rc:$ptrreg, ptr_rc_nor0:$ptroff),
          (STHUX8 G8RC:$rS, ptr_rc_nor0:$ptroff, ptr_rc:$ptrreg)>;
def : Pat<(pre_truncsti32 G8RC:$rS, ptr_rc:$ptrreg, ptr_rc_nor0:$ptroff),
          (STWUX8 G8RC:$rS, ptr_rc_nor0:$ptroff, ptr_rc:$ptrreg)>;
def : Pat<(pre_store G8RC:$rS, ptr_rc:$ptrreg, ptr_rc_nor0:$ptroff),
          (STDUX G8RC:$rS, ptr_rc_nor0:$ptroff, ptr_rc:$ptrreg)>;


//===----------------------------------------------------------------------===//
// Floating point instructions.
//


let PPC970_Unit = 3, Uses = [RM] in {  // FPU Operations.
def FCFID  : XForm_26<63, 846, (outs F8RC:$frD), (ins F8RC:$frB),
                      "fcfid $frD, $frB", FPGeneral,
                      [(set F8RC:$frD, (PPCfcfid F8RC:$frB))]>, isPPC64;
def FCTIDZ : XForm_26<63, 815, (outs F8RC:$frD), (ins F8RC:$frB),
                      "fctidz $frD, $frB", FPGeneral,
                      [(set F8RC:$frD, (PPCfctidz F8RC:$frB))]>, isPPC64;
}


//===----------------------------------------------------------------------===//
// Instruction Patterns
//

// Extensions and truncates to/from 32-bit regs.
def : Pat<(i64 (zext GPRC:$in)),
          (RLDICL (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPRC:$in, sub_32),
                  0, 32)>;
def : Pat<(i64 (anyext GPRC:$in)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPRC:$in, sub_32)>;
def : Pat<(i32 (trunc G8RC:$in)),
          (EXTRACT_SUBREG G8RC:$in, sub_32)>;

// Extending loads with i64 targets.
def : Pat<(zextloadi1 iaddr:$src),
          (LBZ8 iaddr:$src)>;
def : Pat<(zextloadi1 xaddr:$src),
          (LBZX8 xaddr:$src)>;
def : Pat<(extloadi1 iaddr:$src),
          (LBZ8 iaddr:$src)>;
def : Pat<(extloadi1 xaddr:$src),
          (LBZX8 xaddr:$src)>;
def : Pat<(extloadi8 iaddr:$src),
          (LBZ8 iaddr:$src)>;
def : Pat<(extloadi8 xaddr:$src),
          (LBZX8 xaddr:$src)>;
def : Pat<(extloadi16 iaddr:$src),
          (LHZ8 iaddr:$src)>;
def : Pat<(extloadi16 xaddr:$src),
          (LHZX8 xaddr:$src)>;
def : Pat<(extloadi32 iaddr:$src),
          (LWZ8 iaddr:$src)>;
def : Pat<(extloadi32 xaddr:$src),
          (LWZX8 xaddr:$src)>;

// Standard shifts.  These are represented separately from the real shifts above
// so that we can distinguish between shifts that allow 6-bit and 7-bit shift
// amounts.
def : Pat<(sra G8RC:$rS, GPRC:$rB),
          (SRAD G8RC:$rS, GPRC:$rB)>;
def : Pat<(srl G8RC:$rS, GPRC:$rB),
          (SRD G8RC:$rS, GPRC:$rB)>;
def : Pat<(shl G8RC:$rS, GPRC:$rB),
          (SLD G8RC:$rS, GPRC:$rB)>;

// SHL/SRL
def : Pat<(shl G8RC:$in, (i32 imm:$imm)),
          (RLDICR G8RC:$in, imm:$imm, (SHL64 imm:$imm))>;
def : Pat<(srl G8RC:$in, (i32 imm:$imm)),
          (RLDICL G8RC:$in, (SRL64 imm:$imm), imm:$imm)>;

// ROTL
def : Pat<(rotl G8RC:$in, GPRC:$sh),
          (RLDCL G8RC:$in, GPRC:$sh, 0)>;
def : Pat<(rotl G8RC:$in, (i32 imm:$imm)),
          (RLDICL G8RC:$in, imm:$imm, 0)>;

// Hi and Lo for Darwin Global Addresses.
def : Pat<(PPChi tglobaladdr:$in, 0), (LIS8 tglobaladdr:$in)>;
def : Pat<(PPClo tglobaladdr:$in, 0), (LI8  tglobaladdr:$in)>;
def : Pat<(PPChi tconstpool:$in , 0), (LIS8 tconstpool:$in)>;
def : Pat<(PPClo tconstpool:$in , 0), (LI8  tconstpool:$in)>;
def : Pat<(PPChi tjumptable:$in , 0), (LIS8 tjumptable:$in)>;
def : Pat<(PPClo tjumptable:$in , 0), (LI8  tjumptable:$in)>;
def : Pat<(PPChi tblockaddress:$in, 0), (LIS8 tblockaddress:$in)>;
def : Pat<(PPClo tblockaddress:$in, 0), (LI8  tblockaddress:$in)>;
def : Pat<(PPChi tglobaltlsaddr:$g, G8RC:$in),
          (ADDIS8 G8RC:$in, tglobaltlsaddr:$g)>;
def : Pat<(PPClo tglobaltlsaddr:$g, G8RC:$in),
          (ADDI8L G8RC:$in, tglobaltlsaddr:$g)>;
def : Pat<(add G8RC:$in, (PPChi tglobaladdr:$g, 0)),
          (ADDIS8 G8RC:$in, tglobaladdr:$g)>;
def : Pat<(add G8RC:$in, (PPChi tconstpool:$g, 0)),
          (ADDIS8 G8RC:$in, tconstpool:$g)>;
def : Pat<(add G8RC:$in, (PPChi tjumptable:$g, 0)),
          (ADDIS8 G8RC:$in, tjumptable:$g)>;
def : Pat<(add G8RC:$in, (PPChi tblockaddress:$g, 0)),
          (ADDIS8 G8RC:$in, tblockaddress:$g)>;

// Patterns to match r+r indexed loads and stores for
// addresses without at least 4-byte alignment.
def : Pat<(i64 (unaligned4sextloadi32 xoaddr:$src)),
          (LWAX xoaddr:$src)>;
def : Pat<(i64 (unaligned4load xoaddr:$src)),
          (LDX xoaddr:$src)>;
def : Pat<(unaligned4store G8RC:$rS, xoaddr:$dst),
          (STDX G8RC:$rS, xoaddr:$dst)>;

