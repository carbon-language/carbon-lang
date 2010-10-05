//====- X86Instr64bit.td - Describe X86-64 Instructions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86-64 instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// 64-bits but only 32 bits are significant.
def i64i32imm  : Operand<i64> {
  let ParserMatchClass = ImmSExti64i32AsmOperand;
}

// 64-bits but only 32 bits are significant, and those bits are treated as being
// pc relative.
def i64i32imm_pcrel : Operand<i64> {
  let PrintMethod = "print_pcrel_imm";
  let ParserMatchClass = X86AbsMemAsmOperand;
}


// 64-bits but only 8 bits are significant.
def i64i8imm   : Operand<i64> {
  let ParserMatchClass = ImmSExti64i8AsmOperand;
}

def lea64_32mem : Operand<i32> {
  let PrintMethod = "printi32mem";
  let AsmOperandLowerMethod = "lower_lea64_32mem";
  let MIOperandInfo = (ops GR32, i8imm, GR32_NOSP, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}


// Special i64mem for addresses of load folding tail calls. These are not
// allowed to use callee-saved registers since they must be scheduled
// after callee-saved register are popped.
def i64mem_TC : Operand<i64> {
  let PrintMethod = "printi64mem";
  let MIOperandInfo = (ops GR64_TC, i8imm, GR64_TC, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}

//===----------------------------------------------------------------------===//
// Complex Pattern Definitions.
//
def lea64addr : ComplexPattern<i64, 5, "SelectLEAAddr",
                        [add, sub, mul, X86mul_imm, shl, or, frameindex,
                         X86WrapperRIP], []>;

def tls64addr : ComplexPattern<i64, 5, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;
                               
//===----------------------------------------------------------------------===//
// Pattern fragments.
//

def i64immSExt8  : PatLeaf<(i64 immSext8)>;

def GetLo32XForm : SDNodeXForm<imm, [{
  // Transformation function: get the low 32 bits.
  return getI32Imm((unsigned)N->getZExtValue());
}]>;

def i64immSExt32  : PatLeaf<(i64 imm), [{ return i64immSExt32(N); }]>;


def i64immZExt32  : PatLeaf<(i64 imm), [{
  // i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // unsignedsign extended field.
  return (uint64_t)N->getZExtValue() == (uint32_t)N->getZExtValue();
}]>;

def sextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (sextloadi8 node:$ptr))>;
def sextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi64i1  : PatFrag<(ops node:$ptr), (i64 (zextloadi1 node:$ptr))>;
def zextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (zextloadi8 node:$ptr))>;
def zextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;

def extloadi64i1   : PatFrag<(ops node:$ptr), (i64 (extloadi1 node:$ptr))>;
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8 node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;

//===----------------------------------------------------------------------===//
// Instruction list...
//


//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//

def POPCNT64rr : RI<0xB8, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                    "popcnt{q}\t{$src, $dst|$dst, $src}", []>, XS;
let mayLoad = 1 in
def POPCNT64rm : RI<0xB8, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                    "popcnt{q}\t{$src, $dst|$dst, $src}", []>, XS;

let Defs = [RBP,RSP], Uses = [RBP,RSP], mayLoad = 1, neverHasSideEffects = 1 in
def LEAVE64  : I<0xC9, RawFrm,
                 (outs), (ins), "leave", []>, Requires<[In64BitMode]>;
let Defs = [RSP], Uses = [RSP], neverHasSideEffects=1 in {
let mayLoad = 1 in {
def POP64r   : I<0x58, AddRegFrm,
                 (outs GR64:$reg), (ins), "pop{q}\t$reg", []>;
def POP64rmr: I<0x8F, MRM0r, (outs GR64:$reg), (ins), "pop{q}\t$reg", []>;
def POP64rmm: I<0x8F, MRM0m, (outs i64mem:$dst), (ins), "pop{q}\t$dst", []>;
}
let mayStore = 1 in {
def PUSH64r  : I<0x50, AddRegFrm,
                 (outs), (ins GR64:$reg), "push{q}\t$reg", []>;
def PUSH64rmr: I<0xFF, MRM6r, (outs), (ins GR64:$reg), "push{q}\t$reg", []>;
def PUSH64rmm: I<0xFF, MRM6m, (outs), (ins i64mem:$src), "push{q}\t$src", []>;
}
}

let Defs = [RSP], Uses = [RSP], neverHasSideEffects = 1, mayStore = 1 in {
def PUSH64i8   : Ii8<0x6a, RawFrm, (outs), (ins i8imm:$imm), 
                     "push{q}\t$imm", []>;
def PUSH64i16  : Ii16<0x68, RawFrm, (outs), (ins i16imm:$imm), 
                      "push{q}\t$imm", []>;
def PUSH64i32  : Ii32<0x68, RawFrm, (outs), (ins i64i32imm:$imm),
                      "push{q}\t$imm", []>;
}

let Defs = [RSP, EFLAGS], Uses = [RSP], mayLoad = 1, neverHasSideEffects=1 in
def POPF64   : I<0x9D, RawFrm, (outs), (ins), "popfq", []>,
               Requires<[In64BitMode]>;
let Defs = [RSP], Uses = [RSP, EFLAGS], mayStore = 1, neverHasSideEffects=1 in
def PUSHF64    : I<0x9C, RawFrm, (outs), (ins), "pushfq", []>,
                 Requires<[In64BitMode]>;

def LEA64_32r : I<0x8D, MRMSrcMem,
                  (outs GR32:$dst), (ins lea64_32mem:$src),
                  "lea{l}\t{$src|$dst}, {$dst|$src}",
                  [(set GR32:$dst, lea32addr:$src)]>, Requires<[In64BitMode]>;

let isReMaterializable = 1 in
def LEA64r   : RI<0x8D, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "lea{q}\t{$src|$dst}, {$dst|$src}",
                  [(set GR64:$dst, lea64addr:$src)]>;

let Constraints = "$src = $dst" in
def BSWAP64r : RI<0xC8, AddRegFrm, (outs GR64:$dst), (ins GR64:$src),
                  "bswap{q}\t$dst", 
                  [(set GR64:$dst, (bswap GR64:$src))]>, TB;

// Bit scan instructions.
let Defs = [EFLAGS] in {
def BSF64rr  : RI<0xBC, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                  "bsf{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsf GR64:$src))]>, TB;
def BSF64rm  : RI<0xBC, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "bsf{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsf (loadi64 addr:$src)))]>, TB;

def BSR64rr  : RI<0xBD, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                  "bsr{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsr GR64:$src))]>, TB;
def BSR64rm  : RI<0xBD, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "bsr{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsr (loadi64 addr:$src)))]>, TB;
} // Defs = [EFLAGS]



//===----------------------------------------------------------------------===//
//  Move Instructions...
//

let neverHasSideEffects = 1 in
def MOV64rr : RI<0x89, MRMDestReg, (outs GR64:$dst), (ins GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", []>;

let isReMaterializable = 1, isAsCheapAsAMove = 1  in {
def MOV64ri : RIi64<0xB8, AddRegFrm, (outs GR64:$dst), (ins i64imm:$src),
                    "movabs{q}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, imm:$src)]>;
def MOV64ri32 : RIi32<0xC7, MRM0r, (outs GR64:$dst), (ins i64i32imm:$src),
                      "mov{q}\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, i64immSExt32:$src)]>;
}

// The assembler accepts movq of a 64-bit immediate as an alternate spelling of
// movabsq.
let isAsmParserOnly = 1 in {
def MOV64ri_alt : RIi64<0xB8, AddRegFrm, (outs GR64:$dst), (ins i64imm:$src),
                    "mov{q}\t{$src, $dst|$dst, $src}", []>;
}

let isCodeGenOnly = 1 in {
def MOV64rr_REV : RI<0x8B, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                     "mov{q}\t{$src, $dst|$dst, $src}", []>;
}

let canFoldAsLoad = 1, isReMaterializable = 1 in
def MOV64rm : RI<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}",
                 [(set GR64:$dst, (load addr:$src))]>;

def MOV64mr : RI<0x89, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}",
                 [(store GR64:$src, addr:$dst)]>;
def MOV64mi32 : RIi32<0xC7, MRM0m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                      "mov{q}\t{$src, $dst|$dst, $src}",
                      [(store i64immSExt32:$src, addr:$dst)]>;

/// Versions of MOV64rr, MOV64rm, and MOV64mr for i64mem_TC and GR64_TC.
let isCodeGenOnly = 1 in {
let neverHasSideEffects = 1 in
def MOV64rr_TC : RI<0x89, MRMDestReg, (outs GR64_TC:$dst), (ins GR64_TC:$src),
                "mov{q}\t{$src, $dst|$dst, $src}", []>;

let mayLoad = 1,
    canFoldAsLoad = 1, isReMaterializable = 1 in
def MOV64rm_TC : RI<0x8B, MRMSrcMem, (outs GR64_TC:$dst), (ins i64mem_TC:$src),
                "mov{q}\t{$src, $dst|$dst, $src}",
                []>;

let mayStore = 1 in
def MOV64mr_TC : RI<0x89, MRMDestMem, (outs), (ins i64mem_TC:$dst, GR64_TC:$src),
                "mov{q}\t{$src, $dst|$dst, $src}",
                []>;
}

// FIXME: These definitions are utterly broken
// Just leave them commented out for now because they're useless outside
// of the large code model, and most compilers won't generate the instructions
// in question.
/*
def MOV64o8a : RIi8<0xA0, RawFrm, (outs), (ins offset8:$src),
                      "mov{q}\t{$src, %rax|%rax, $src}", []>;
def MOV64o64a : RIi32<0xA1, RawFrm, (outs), (ins offset64:$src),
                       "mov{q}\t{$src, %rax|%rax, $src}", []>;
def MOV64ao8 : RIi8<0xA2, RawFrm, (outs offset8:$dst), (ins),
                       "mov{q}\t{%rax, $dst|$dst, %rax}", []>;
def MOV64ao64 : RIi32<0xA3, RawFrm, (outs offset64:$dst), (ins),
                       "mov{q}\t{%rax, $dst|$dst, %rax}", []>;
*/


// Sign/Zero extenders

// MOVSX64rr8 always has a REX prefix and it has an 8-bit register
// operand, which makes it a rare instruction with an 8-bit register
// operand that can never access an h register. If support for h registers
// were generalized, this would require a special register class.
def MOVSX64rr8 : RI<0xBE, MRMSrcReg, (outs GR64:$dst), (ins GR8 :$src),
                    "movs{bq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR8:$src))]>, TB;
def MOVSX64rm8 : RI<0xBE, MRMSrcMem, (outs GR64:$dst), (ins i8mem :$src),
                    "movs{bq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i8 addr:$src))]>, TB;
def MOVSX64rr16: RI<0xBF, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                    "movs{wq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR16:$src))]>, TB;
def MOVSX64rm16: RI<0xBF, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                    "movs{wq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i16 addr:$src))]>, TB;
def MOVSX64rr32: RI<0x63, MRMSrcReg, (outs GR64:$dst), (ins GR32:$src),
                    "movs{lq|xd}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR32:$src))]>;
def MOVSX64rm32: RI<0x63, MRMSrcMem, (outs GR64:$dst), (ins i32mem:$src),
                    "movs{lq|xd}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i32 addr:$src))]>;

// movzbq and movzwq encodings for the disassembler
def MOVZX64rr8_Q : RI<0xB6, MRMSrcReg, (outs GR64:$dst), (ins GR8:$src),
                       "movz{bq|x}\t{$src, $dst|$dst, $src}", []>, TB;
def MOVZX64rm8_Q : RI<0xB6, MRMSrcMem, (outs GR64:$dst), (ins i8mem:$src),
                       "movz{bq|x}\t{$src, $dst|$dst, $src}", []>, TB;
def MOVZX64rr16_Q : RI<0xB7, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                       "movz{wq|x}\t{$src, $dst|$dst, $src}", []>, TB;
def MOVZX64rm16_Q : RI<0xB7, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                       "movz{wq|x}\t{$src, $dst|$dst, $src}", []>, TB;

// Use movzbl instead of movzbq when the destination is a register; it's
// equivalent due to implicit zero-extending, and it has a smaller encoding.
def MOVZX64rr8 : I<0xB6, MRMSrcReg, (outs GR64:$dst), (ins GR8 :$src),
                   "", [(set GR64:$dst, (zext GR8:$src))]>, TB;
def MOVZX64rm8 : I<0xB6, MRMSrcMem, (outs GR64:$dst), (ins i8mem :$src),
                   "", [(set GR64:$dst, (zextloadi64i8 addr:$src))]>, TB;
// Use movzwl instead of movzwq when the destination is a register; it's
// equivalent due to implicit zero-extending, and it has a smaller encoding.
def MOVZX64rr16: I<0xB7, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                   "", [(set GR64:$dst, (zext GR16:$src))]>, TB;
def MOVZX64rm16: I<0xB7, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                   "", [(set GR64:$dst, (zextloadi64i16 addr:$src))]>, TB;

// There's no movzlq instruction, but movl can be used for this purpose, using
// implicit zero-extension. The preferred way to do 32-bit-to-64-bit zero
// extension on x86-64 is to use a SUBREG_TO_REG to utilize implicit
// zero-extension, however this isn't possible when the 32-bit value is
// defined by a truncate or is copied from something where the high bits aren't
// necessarily all zero. In such cases, we fall back to these explicit zext
// instructions.
def MOVZX64rr32 : I<0x89, MRMDestReg, (outs GR64:$dst), (ins GR32:$src),
                    "", [(set GR64:$dst, (zext GR32:$src))]>;
def MOVZX64rm32 : I<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i32mem:$src),
                    "", [(set GR64:$dst, (zextloadi64i32 addr:$src))]>;

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

let neverHasSideEffects = 1 in {
  let Defs = [RAX], Uses = [EAX] in
  def CDQE : RI<0x98, RawFrm, (outs), (ins),
               "{cltq|cdqe}", []>;     // RAX = signext(EAX)

  let Defs = [RAX,RDX], Uses = [RAX] in
  def CQO  : RI<0x99, RawFrm, (outs), (ins),
                "{cqto|cqo}", []>; // RDX:RAX = signext(RAX)
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions...
//

let Defs = [EFLAGS] in {

def ADD64i32 : RIi32<0x05, RawFrm, (outs), (ins i64i32imm:$src),
                     "add{q}\t{$src, %rax|%rax, $src}", []>;

let Constraints = "$src1 = $dst" in {
let isConvertibleToThreeAddress = 1 in {
let isCommutable = 1 in
// Register-Register Addition
def ADD64rr    : RI<0x01, MRMDestReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "add{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86add_flag GR64:$src1, GR64:$src2))]>;

// These are alternate spellings for use by the disassembler, we mark them as
// code gen only to ensure they aren't matched by the assembler.
let isCodeGenOnly = 1 in {
  def ADD64rr_alt  : RI<0x03, MRMSrcReg, (outs GR64:$dst), 
                       (ins GR64:$src1, GR64:$src2),
                       "add{l}\t{$src2, $dst|$dst, $src2}", []>;
}

// Register-Integer Addition
def ADD64ri8  : RIi8<0x83, MRM0r, (outs GR64:$dst), 
                     (ins GR64:$src1, i64i8imm:$src2),
                     "add{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, EFLAGS,
                           (X86add_flag GR64:$src1, i64immSExt8:$src2))]>;
def ADD64ri32 : RIi32<0x81, MRM0r, (outs GR64:$dst), 
                      (ins GR64:$src1, i64i32imm:$src2),
                      "add{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86add_flag GR64:$src1, i64immSExt32:$src2))]>;
} // isConvertibleToThreeAddress

// Register-Memory Addition
def ADD64rm     : RI<0x03, MRMSrcMem, (outs GR64:$dst), 
                     (ins GR64:$src1, i64mem:$src2),
                     "add{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, EFLAGS,
                           (X86add_flag GR64:$src1, (load addr:$src2)))]>;

} // Constraints = "$src1 = $dst"

// Memory-Register Addition
def ADD64mr  : RI<0x01, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                  "add{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (load addr:$dst), GR64:$src2), addr:$dst),
                   (implicit EFLAGS)]>;
def ADD64mi8 : RIi8<0x83, MRM0m, (outs), (ins i64mem:$dst, i64i8imm :$src2),
                    "add{q}\t{$src2, $dst|$dst, $src2}",
                [(store (add (load addr:$dst), i64immSExt8:$src2), addr:$dst),
                 (implicit EFLAGS)]>;
def ADD64mi32 : RIi32<0x81, MRM0m, (outs), (ins i64mem:$dst, i64i32imm :$src2),
                      "add{q}\t{$src2, $dst|$dst, $src2}",
               [(store (add (load addr:$dst), i64immSExt32:$src2), addr:$dst),
                (implicit EFLAGS)]>;

let Uses = [EFLAGS] in {

def ADC64i32 : RIi32<0x15, RawFrm, (outs), (ins i64i32imm:$src),
                     "adc{q}\t{$src, %rax|%rax, $src}", []>;

let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in
def ADC64rr  : RI<0x11, MRMDestReg, (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, GR64:$src2))]>;

let isCodeGenOnly = 1 in {
def ADC64rr_REV : RI<0x13, MRMSrcReg , (outs GR32:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}", []>;
}

def ADC64rm  : RI<0x13, MRMSrcMem , (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, (load addr:$src2)))]>;

def ADC64ri8 : RIi8<0x83, MRM2r, (outs GR64:$dst), 
                    (ins GR64:$src1, i64i8imm:$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (adde GR64:$src1, i64immSExt8:$src2))]>;
def ADC64ri32 : RIi32<0x81, MRM2r, (outs GR64:$dst), 
                      (ins GR64:$src1, i64i32imm:$src2),
                      "adc{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (adde GR64:$src1, i64immSExt32:$src2))]>;
} // Constraints = "$src1 = $dst"

def ADC64mr  : RI<0x11, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (load addr:$dst), GR64:$src2), addr:$dst)]>;
def ADC64mi8 : RIi8<0x83, MRM2m, (outs), (ins i64mem:$dst, i64i8imm :$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}",
                 [(store (adde (load addr:$dst), i64immSExt8:$src2), 
                  addr:$dst)]>;
def ADC64mi32 : RIi32<0x81, MRM2m, (outs), (ins i64mem:$dst, i64i32imm:$src2),
                      "adc{q}\t{$src2, $dst|$dst, $src2}",
                 [(store (adde (load addr:$dst), i64immSExt32:$src2), 
                  addr:$dst)]>;
} // Uses = [EFLAGS]

let Constraints = "$src1 = $dst" in {
// Register-Register Subtraction
def SUB64rr  : RI<0x29, MRMDestReg, (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2),
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86sub_flag GR64:$src1, GR64:$src2))]>;

let isCodeGenOnly = 1 in {
def SUB64rr_REV : RI<0x2B, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                     "sub{q}\t{$src2, $dst|$dst, $src2}", []>;
}

// Register-Memory Subtraction
def SUB64rm  : RI<0x2B, MRMSrcMem, (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2),
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS, 
                        (X86sub_flag GR64:$src1, (load addr:$src2)))]>;

// Register-Integer Subtraction
def SUB64ri8 : RIi8<0x83, MRM5r, (outs GR64:$dst),
                                 (ins GR64:$src1, i64i8imm:$src2),
                    "sub{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86sub_flag GR64:$src1, i64immSExt8:$src2))]>;
def SUB64ri32 : RIi32<0x81, MRM5r, (outs GR64:$dst),
                                   (ins GR64:$src1, i64i32imm:$src2),
                      "sub{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86sub_flag GR64:$src1, i64immSExt32:$src2))]>;
} // Constraints = "$src1 = $dst"

def SUB64i32 : RIi32<0x2D, RawFrm, (outs), (ins i64i32imm:$src),
                     "sub{q}\t{$src, %rax|%rax, $src}", []>;

// Memory-Register Subtraction
def SUB64mr  : RI<0x29, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (sub (load addr:$dst), GR64:$src2), addr:$dst),
                   (implicit EFLAGS)]>;

// Memory-Integer Subtraction
def SUB64mi8 : RIi8<0x83, MRM5m, (outs), (ins i64mem:$dst, i64i8imm :$src2), 
                    "sub{q}\t{$src2, $dst|$dst, $src2}",
                    [(store (sub (load addr:$dst), i64immSExt8:$src2),
                            addr:$dst),
                     (implicit EFLAGS)]>;
def SUB64mi32 : RIi32<0x81, MRM5m, (outs), (ins i64mem:$dst, i64i32imm:$src2),
                      "sub{q}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (load addr:$dst), i64immSExt32:$src2),
                              addr:$dst),
                       (implicit EFLAGS)]>;

let Uses = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
def SBB64rr    : RI<0x19, MRMDestReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, GR64:$src2))]>;

let isCodeGenOnly = 1 in {
def SBB64rr_REV : RI<0x1B, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                     "sbb{q}\t{$src2, $dst|$dst, $src2}", []>;
}
                     
def SBB64rm  : RI<0x1B, MRMSrcMem, (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2),
                  "sbb{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sube GR64:$src1, (load addr:$src2)))]>;

def SBB64ri8 : RIi8<0x83, MRM3r, (outs GR64:$dst), 
                    (ins GR64:$src1, i64i8imm:$src2),
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, i64immSExt8:$src2))]>;
def SBB64ri32 : RIi32<0x81, MRM3r, (outs GR64:$dst), 
                      (ins GR64:$src1, i64i32imm:$src2),
                      "sbb{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (sube GR64:$src1, i64immSExt32:$src2))]>;
} // Constraints = "$src1 = $dst"

def SBB64i32 : RIi32<0x1D, RawFrm, (outs), (ins i64i32imm:$src),
                     "sbb{q}\t{$src, %rax|%rax, $src}", []>;

def SBB64mr  : RI<0x19, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                  "sbb{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (sube (load addr:$dst), GR64:$src2), addr:$dst)]>;
def SBB64mi8 : RIi8<0x83, MRM3m, (outs), (ins i64mem:$dst, i64i8imm :$src2), 
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;
def SBB64mi32 : RIi32<0x81, MRM3m, (outs), (ins i64mem:$dst, i64i32imm:$src2), 
                      "sbb{q}\t{$src2, $dst|$dst, $src2}",
              [(store (sube (load addr:$dst), i64immSExt32:$src2), addr:$dst)]>;
} // Uses = [EFLAGS]
} // Defs = [EFLAGS]

// Unsigned multiplication
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX], neverHasSideEffects = 1 in {
def MUL64r : RI<0xF7, MRM4r, (outs), (ins GR64:$src),
                "mul{q}\t$src", []>;         // RAX,RDX = RAX*GR64
let mayLoad = 1 in
def MUL64m : RI<0xF7, MRM4m, (outs), (ins i64mem:$src),
                "mul{q}\t$src", []>;         // RAX,RDX = RAX*[mem64]

// Signed multiplication
def IMUL64r : RI<0xF7, MRM5r, (outs), (ins GR64:$src),
                 "imul{q}\t$src", []>;         // RAX,RDX = RAX*GR64
let mayLoad = 1 in
def IMUL64m : RI<0xF7, MRM5m, (outs), (ins i64mem:$src),
                 "imul{q}\t$src", []>;         // RAX,RDX = RAX*[mem64]
}

let Defs = [EFLAGS] in {
let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in
// Register-Register Signed Integer Multiplication
def IMUL64rr : RI<0xAF, MRMSrcReg, (outs GR64:$dst),
                                   (ins GR64:$src1, GR64:$src2),
                  "imul{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86smul_flag GR64:$src1, GR64:$src2))]>, TB;

// Register-Memory Signed Integer Multiplication
def IMUL64rm : RI<0xAF, MRMSrcMem, (outs GR64:$dst),
                                   (ins GR64:$src1, i64mem:$src2),
                  "imul{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86smul_flag GR64:$src1, (load addr:$src2)))]>, TB;
} // Constraints = "$src1 = $dst"

// Suprisingly enough, these are not two address instructions!

// Register-Integer Signed Integer Multiplication
def IMUL64rri8 : RIi8<0x6B, MRMSrcReg,                      // GR64 = GR64*I8
                      (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                      "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86smul_flag GR64:$src1, i64immSExt8:$src2))]>;
def IMUL64rri32 : RIi32<0x69, MRMSrcReg,                    // GR64 = GR64*I32
                        (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                        "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set GR64:$dst, EFLAGS,
                             (X86smul_flag GR64:$src1, i64immSExt32:$src2))]>;

// Memory-Integer Signed Integer Multiplication
def IMUL64rmi8 : RIi8<0x6B, MRMSrcMem,                      // GR64 = [mem64]*I8
                      (outs GR64:$dst), (ins i64mem:$src1, i64i8imm: $src2),
                      "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86smul_flag (load addr:$src1),
                                          i64immSExt8:$src2))]>;
def IMUL64rmi32 : RIi32<0x69, MRMSrcMem,                   // GR64 = [mem64]*I32
                        (outs GR64:$dst), (ins i64mem:$src1, i64i32imm:$src2),
                        "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set GR64:$dst, EFLAGS,
                              (X86smul_flag (load addr:$src1),
                                            i64immSExt32:$src2))]>;
} // Defs = [EFLAGS]

// Unsigned division / remainder
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in {
// RDX:RAX/r64 = RAX,RDX
def DIV64r : RI<0xF7, MRM6r, (outs), (ins GR64:$src),
                "div{q}\t$src", []>;
// Signed division / remainder
// RDX:RAX/r64 = RAX,RDX
def IDIV64r: RI<0xF7, MRM7r, (outs), (ins GR64:$src),
                "idiv{q}\t$src", []>;
let mayLoad = 1 in {
// RDX:RAX/[mem64] = RAX,RDX
def DIV64m : RI<0xF7, MRM6m, (outs), (ins i64mem:$src),
                "div{q}\t$src", []>;
// RDX:RAX/[mem64] = RAX,RDX
def IDIV64m: RI<0xF7, MRM7m, (outs), (ins i64mem:$src),
                "idiv{q}\t$src", []>;
}
}

// Unary instructions
let Defs = [EFLAGS], CodeSize = 2 in {
let Constraints = "$src = $dst" in
def NEG64r : RI<0xF7, MRM3r, (outs GR64:$dst), (ins GR64:$src), "neg{q}\t$dst",
                [(set GR64:$dst, (ineg GR64:$src)),
                 (implicit EFLAGS)]>;
def NEG64m : RI<0xF7, MRM3m, (outs), (ins i64mem:$dst), "neg{q}\t$dst",
                [(store (ineg (loadi64 addr:$dst)), addr:$dst),
                 (implicit EFLAGS)]>;

let Constraints = "$src = $dst", isConvertibleToThreeAddress = 1 in
def INC64r : RI<0xFF, MRM0r, (outs GR64:$dst), (ins GR64:$src), "inc{q}\t$dst",
                [(set GR64:$dst, EFLAGS, (X86inc_flag GR64:$src))]>;
def INC64m : RI<0xFF, MRM0m, (outs), (ins i64mem:$dst), "inc{q}\t$dst",
                [(store (add (loadi64 addr:$dst), 1), addr:$dst),
                 (implicit EFLAGS)]>;

let Constraints = "$src = $dst", isConvertibleToThreeAddress = 1 in
def DEC64r : RI<0xFF, MRM1r, (outs GR64:$dst), (ins GR64:$src), "dec{q}\t$dst",
                [(set GR64:$dst, EFLAGS, (X86dec_flag GR64:$src))]>;
def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
                [(store (add (loadi64 addr:$dst), -1), addr:$dst),
                 (implicit EFLAGS)]>;

// In 64-bit mode, single byte INC and DEC cannot be encoded.
let Constraints = "$src = $dst", isConvertibleToThreeAddress = 1 in {
// Can transform into LEA.
def INC64_16r : I<0xFF, MRM0r, (outs GR16:$dst), (ins GR16:$src), 
                  "inc{w}\t$dst",
                  [(set GR16:$dst, EFLAGS, (X86inc_flag GR16:$src))]>,
                OpSize, Requires<[In64BitMode]>;
def INC64_32r : I<0xFF, MRM0r, (outs GR32:$dst), (ins GR32:$src), 
                  "inc{l}\t$dst",
                  [(set GR32:$dst, EFLAGS, (X86inc_flag GR32:$src))]>,
                Requires<[In64BitMode]>;
def DEC64_16r : I<0xFF, MRM1r, (outs GR16:$dst), (ins GR16:$src), 
                  "dec{w}\t$dst",
                  [(set GR16:$dst, EFLAGS, (X86dec_flag GR16:$src))]>,
                OpSize, Requires<[In64BitMode]>;
def DEC64_32r : I<0xFF, MRM1r, (outs GR32:$dst), (ins GR32:$src), 
                  "dec{l}\t$dst",
                  [(set GR32:$dst, EFLAGS, (X86dec_flag GR32:$src))]>,
                Requires<[In64BitMode]>;
} // Constraints = "$src = $dst", isConvertibleToThreeAddress

// These are duplicates of their 32-bit counterparts. Only needed so X86 knows
// how to unfold them.
def INC64_16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst), "inc{w}\t$dst",
                  [(store (add (loadi16 addr:$dst), 1), addr:$dst),
                    (implicit EFLAGS)]>,
                OpSize, Requires<[In64BitMode]>;
def INC64_32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst), "inc{l}\t$dst",
                  [(store (add (loadi32 addr:$dst), 1), addr:$dst),
                    (implicit EFLAGS)]>,
                Requires<[In64BitMode]>;
def DEC64_16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst), "dec{w}\t$dst",
                  [(store (add (loadi16 addr:$dst), -1), addr:$dst),
                    (implicit EFLAGS)]>,
                OpSize, Requires<[In64BitMode]>;
def DEC64_32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst), "dec{l}\t$dst",
                  [(store (add (loadi32 addr:$dst), -1), addr:$dst),
                    (implicit EFLAGS)]>,
                Requires<[In64BitMode]>;
} // Defs = [EFLAGS], CodeSize


let Defs = [EFLAGS] in {
// Shift instructions
let Constraints = "$src1 = $dst" in {
let Uses = [CL] in
def SHL64rCL : RI<0xD3, MRM4r, (outs GR64:$dst), (ins GR64:$src1),
                  "shl{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (shl GR64:$src1, CL))]>;
let isConvertibleToThreeAddress = 1 in   // Can transform into LEA.
def SHL64ri  : RIi8<0xC1, MRM4r, (outs GR64:$dst), 
                    (ins GR64:$src1, i8imm:$src2),
                    "shl{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (shl GR64:$src1, (i8 imm:$src2)))]>;
// NOTE: We don't include patterns for shifts of a register by one, because
// 'add reg,reg' is cheaper.
def SHL64r1  : RI<0xD1, MRM4r, (outs GR64:$dst), (ins GR64:$src1),
                 "shl{q}\t$dst", []>;
} // Constraints = "$src1 = $dst"

let Uses = [CL] in
def SHL64mCL : RI<0xD3, MRM4m, (outs), (ins i64mem:$dst),
                  "shl{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (shl (loadi64 addr:$dst), CL), addr:$dst)]>;
def SHL64mi : RIi8<0xC1, MRM4m, (outs), (ins i64mem:$dst, i8imm:$src),
                  "shl{q}\t{$src, $dst|$dst, $src}",
                 [(store (shl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SHL64m1 : RI<0xD1, MRM4m, (outs), (ins i64mem:$dst),
                  "shl{q}\t$dst",
                 [(store (shl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let Constraints = "$src1 = $dst" in {
let Uses = [CL] in
def SHR64rCL : RI<0xD3, MRM5r, (outs GR64:$dst), (ins GR64:$src1),
                  "shr{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (srl GR64:$src1, CL))]>;
def SHR64ri : RIi8<0xC1, MRM5r, (outs GR64:$dst), (ins GR64:$src1, i8imm:$src2),
                  "shr{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (srl GR64:$src1, (i8 imm:$src2)))]>;
def SHR64r1  : RI<0xD1, MRM5r, (outs GR64:$dst), (ins GR64:$src1),
                 "shr{q}\t$dst",
                 [(set GR64:$dst, (srl GR64:$src1, (i8 1)))]>;
} // Constraints = "$src1 = $dst"

let Uses = [CL] in
def SHR64mCL : RI<0xD3, MRM5m, (outs), (ins i64mem:$dst),
                  "shr{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (srl (loadi64 addr:$dst), CL), addr:$dst)]>;
def SHR64mi : RIi8<0xC1, MRM5m, (outs), (ins i64mem:$dst, i8imm:$src),
                  "shr{q}\t{$src, $dst|$dst, $src}",
                 [(store (srl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SHR64m1 : RI<0xD1, MRM5m, (outs), (ins i64mem:$dst),
                  "shr{q}\t$dst",
                 [(store (srl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let Constraints = "$src1 = $dst" in {
let Uses = [CL] in
def SAR64rCL : RI<0xD3, MRM7r, (outs GR64:$dst), (ins GR64:$src1),
                 "sar{q}\t{%cl, $dst|$dst, %CL}",
                 [(set GR64:$dst, (sra GR64:$src1, CL))]>;
def SAR64ri  : RIi8<0xC1, MRM7r, (outs GR64:$dst),
                    (ins GR64:$src1, i8imm:$src2),
                    "sar{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sra GR64:$src1, (i8 imm:$src2)))]>;
def SAR64r1  : RI<0xD1, MRM7r, (outs GR64:$dst), (ins GR64:$src1),
                 "sar{q}\t$dst",
                 [(set GR64:$dst, (sra GR64:$src1, (i8 1)))]>;
} // Constraints = "$src = $dst"

let Uses = [CL] in
def SAR64mCL : RI<0xD3, MRM7m, (outs), (ins i64mem:$dst), 
                 "sar{q}\t{%cl, $dst|$dst, %CL}",
                 [(store (sra (loadi64 addr:$dst), CL), addr:$dst)]>;
def SAR64mi  : RIi8<0xC1, MRM7m, (outs), (ins i64mem:$dst, i8imm:$src),
                    "sar{q}\t{$src, $dst|$dst, $src}",
                 [(store (sra (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SAR64m1 : RI<0xD1, MRM7m, (outs), (ins i64mem:$dst),
                  "sar{q}\t$dst",
                 [(store (sra (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

// Rotate instructions

let Constraints = "$src = $dst" in {
def RCL64r1 : RI<0xD1, MRM2r, (outs GR64:$dst), (ins GR64:$src),
                 "rcl{q}\t{1, $dst|$dst, 1}", []>;
def RCL64ri : RIi8<0xC1, MRM2r, (outs GR64:$dst), (ins GR64:$src, i8imm:$cnt),
                   "rcl{q}\t{$cnt, $dst|$dst, $cnt}", []>;

def RCR64r1 : RI<0xD1, MRM3r, (outs GR64:$dst), (ins GR64:$src),
                 "rcr{q}\t{1, $dst|$dst, 1}", []>;
def RCR64ri : RIi8<0xC1, MRM3r, (outs GR64:$dst), (ins GR64:$src, i8imm:$cnt),
                   "rcr{q}\t{$cnt, $dst|$dst, $cnt}", []>;

let Uses = [CL] in {
def RCL64rCL : RI<0xD3, MRM2r, (outs GR64:$dst), (ins GR64:$src),
                  "rcl{q}\t{%cl, $dst|$dst, CL}", []>;
def RCR64rCL : RI<0xD3, MRM3r, (outs GR64:$dst), (ins GR64:$src),
                  "rcr{q}\t{%cl, $dst|$dst, CL}", []>;
}
} // Constraints = "$src = $dst"

def RCL64m1 : RI<0xD1, MRM2m, (outs), (ins i64mem:$dst),
                 "rcl{q}\t{1, $dst|$dst, 1}", []>;
def RCL64mi : RIi8<0xC1, MRM2m, (outs), (ins i64mem:$dst, i8imm:$cnt),
                   "rcl{q}\t{$cnt, $dst|$dst, $cnt}", []>;
def RCR64m1 : RI<0xD1, MRM3m, (outs), (ins i64mem:$dst),
                 "rcr{q}\t{1, $dst|$dst, 1}", []>;
def RCR64mi : RIi8<0xC1, MRM3m, (outs), (ins i64mem:$dst, i8imm:$cnt),
                   "rcr{q}\t{$cnt, $dst|$dst, $cnt}", []>;

let Uses = [CL] in {
def RCL64mCL : RI<0xD3, MRM2m, (outs), (ins i64mem:$dst),
                  "rcl{q}\t{%cl, $dst|$dst, CL}", []>;
def RCR64mCL : RI<0xD3, MRM3m, (outs), (ins i64mem:$dst),
                  "rcr{q}\t{%cl, $dst|$dst, CL}", []>;
}

let Constraints = "$src1 = $dst" in {
let Uses = [CL] in
def ROL64rCL : RI<0xD3, MRM0r, (outs GR64:$dst), (ins GR64:$src1),
                  "rol{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (rotl GR64:$src1, CL))]>;
def ROL64ri  : RIi8<0xC1, MRM0r, (outs GR64:$dst), 
                    (ins GR64:$src1, i8imm:$src2),
                    "rol{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotl GR64:$src1, (i8 imm:$src2)))]>;
def ROL64r1  : RI<0xD1, MRM0r, (outs GR64:$dst), (ins GR64:$src1),
                  "rol{q}\t$dst",
                  [(set GR64:$dst, (rotl GR64:$src1, (i8 1)))]>;
} // Constraints = "$src1 = $dst"

let Uses = [CL] in
def ROL64mCL :  RI<0xD3, MRM0m, (outs), (ins i64mem:$dst),
                   "rol{q}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi64 addr:$dst), CL), addr:$dst)]>;
def ROL64mi  : RIi8<0xC1, MRM0m, (outs), (ins i64mem:$dst, i8imm:$src),
                    "rol{q}\t{$src, $dst|$dst, $src}",
                [(store (rotl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROL64m1  : RI<0xD1, MRM0m, (outs), (ins i64mem:$dst),
                 "rol{q}\t$dst",
               [(store (rotl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let Constraints = "$src1 = $dst" in {
let Uses = [CL] in
def ROR64rCL : RI<0xD3, MRM1r, (outs GR64:$dst), (ins GR64:$src1),
                  "ror{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (rotr GR64:$src1, CL))]>;
def ROR64ri  : RIi8<0xC1, MRM1r, (outs GR64:$dst), 
                    (ins GR64:$src1, i8imm:$src2),
                    "ror{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotr GR64:$src1, (i8 imm:$src2)))]>;
def ROR64r1  : RI<0xD1, MRM1r, (outs GR64:$dst), (ins GR64:$src1),
                  "ror{q}\t$dst",
                  [(set GR64:$dst, (rotr GR64:$src1, (i8 1)))]>;
} // Constraints = "$src1 = $dst"

let Uses = [CL] in
def ROR64mCL : RI<0xD3, MRM1m, (outs), (ins i64mem:$dst), 
                  "ror{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (rotr (loadi64 addr:$dst), CL), addr:$dst)]>;
def ROR64mi  : RIi8<0xC1, MRM1m, (outs), (ins i64mem:$dst, i8imm:$src),
                    "ror{q}\t{$src, $dst|$dst, $src}",
                [(store (rotr (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROR64m1  : RI<0xD1, MRM1m, (outs), (ins i64mem:$dst),
                 "ror{q}\t$dst",
               [(store (rotr (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

// Double shift instructions (generalizations of rotate)
let Constraints = "$src1 = $dst" in {
let Uses = [CL] in {
def SHLD64rrCL : RI<0xA5, MRMDestReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "shld{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(set GR64:$dst, (X86shld GR64:$src1, GR64:$src2, CL))]>, 
                    TB;
def SHRD64rrCL : RI<0xAD, MRMDestReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "shrd{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(set GR64:$dst, (X86shrd GR64:$src1, GR64:$src2, CL))]>, 
                    TB;
}

let isCommutable = 1 in {  // FIXME: Update X86InstrInfo::commuteInstruction
def SHLD64rri8 : RIi8<0xA4, MRMDestReg,
                      (outs GR64:$dst), 
                      (ins GR64:$src1, GR64:$src2, i8imm:$src3),
                      "shld{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set GR64:$dst, (X86shld GR64:$src1, GR64:$src2,
                                       (i8 imm:$src3)))]>,
                 TB;
def SHRD64rri8 : RIi8<0xAC, MRMDestReg,
                      (outs GR64:$dst), 
                      (ins GR64:$src1, GR64:$src2, i8imm:$src3),
                      "shrd{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set GR64:$dst, (X86shrd GR64:$src1, GR64:$src2,
                                       (i8 imm:$src3)))]>,
                 TB;
} // isCommutable
} // Constraints = "$src1 = $dst"

let Uses = [CL] in {
def SHLD64mrCL : RI<0xA5, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                    "shld{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shld (loadi64 addr:$dst), GR64:$src2, CL),
                      addr:$dst)]>, TB;
def SHRD64mrCL : RI<0xAD, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                    "shrd{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shrd (loadi64 addr:$dst), GR64:$src2, CL),
                      addr:$dst)]>, TB;
}
def SHLD64mri8 : RIi8<0xA4, MRMDestMem,
                      (outs), (ins i64mem:$dst, GR64:$src2, i8imm:$src3),
                      "shld{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi64 addr:$dst), GR64:$src2,
                                       (i8 imm:$src3)), addr:$dst)]>,
                 TB;
def SHRD64mri8 : RIi8<0xAC, MRMDestMem, 
                      (outs), (ins i64mem:$dst, GR64:$src2, i8imm:$src3),
                      "shrd{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shrd (loadi64 addr:$dst), GR64:$src2,
                                       (i8 imm:$src3)), addr:$dst)]>,
                 TB;
} // Defs = [EFLAGS]

//===----------------------------------------------------------------------===//
//  Logical Instructions...
//

let Constraints = "$src = $dst" , AddedComplexity = 15 in
def NOT64r : RI<0xF7, MRM2r, (outs GR64:$dst), (ins GR64:$src), "not{q}\t$dst",
                [(set GR64:$dst, (not GR64:$src))]>;
def NOT64m : RI<0xF7, MRM2m, (outs), (ins i64mem:$dst), "not{q}\t$dst",
                [(store (not (loadi64 addr:$dst)), addr:$dst)]>;

let Defs = [EFLAGS] in {
def AND64i32 : RIi32<0x25, RawFrm, (outs), (ins i64i32imm:$src),
                     "and{q}\t{$src, %rax|%rax, $src}", []>;

let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in
def AND64rr  : RI<0x21, MRMDestReg, 
                  (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                  "and{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86and_flag GR64:$src1, GR64:$src2))]>;
let isCodeGenOnly = 1 in {
def AND64rr_REV : RI<0x23, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                     "and{q}\t{$src2, $dst|$dst, $src2}", []>;
}
def AND64rm  : RI<0x23, MRMSrcMem,
                  (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                  "and{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86and_flag GR64:$src1, (load addr:$src2)))]>;
def AND64ri8 : RIi8<0x83, MRM4r, 
                    (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "and{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86and_flag GR64:$src1, i64immSExt8:$src2))]>;
def AND64ri32  : RIi32<0x81, MRM4r, 
                       (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                       "and{q}\t{$src2, $dst|$dst, $src2}",
                       [(set GR64:$dst, EFLAGS,
                             (X86and_flag GR64:$src1, i64immSExt32:$src2))]>;
} // Constraints = "$src1 = $dst"

def AND64mr  : RI<0x21, MRMDestMem,
                  (outs), (ins i64mem:$dst, GR64:$src),
                  "and{q}\t{$src, $dst|$dst, $src}",
                  [(store (and (load addr:$dst), GR64:$src), addr:$dst),
                   (implicit EFLAGS)]>;
def AND64mi8 : RIi8<0x83, MRM4m,
                    (outs), (ins i64mem:$dst, i64i8imm :$src),
                    "and{q}\t{$src, $dst|$dst, $src}",
                 [(store (and (load addr:$dst), i64immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
def AND64mi32  : RIi32<0x81, MRM4m,
                       (outs), (ins i64mem:$dst, i64i32imm:$src),
                       "and{q}\t{$src, $dst|$dst, $src}",
             [(store (and (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
              (implicit EFLAGS)]>;

let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in
def OR64rr   : RI<0x09, MRMDestReg, (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2),
                  "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86or_flag GR64:$src1, GR64:$src2))]>;
let isCodeGenOnly = 1 in {
def OR64rr_REV : RI<0x0B, MRMSrcReg, (outs GR64:$dst), 
                    (ins GR64:$src1, GR64:$src2),
                    "or{q}\t{$src2, $dst|$dst, $src2}", []>;
}
def OR64rm   : RI<0x0B, MRMSrcMem , (outs GR64:$dst),
                  (ins GR64:$src1, i64mem:$src2),
                  "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86or_flag GR64:$src1, (load addr:$src2)))]>;
def OR64ri8  : RIi8<0x83, MRM1r, (outs GR64:$dst),
                    (ins GR64:$src1, i64i8imm:$src2),
                    "or{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, EFLAGS,
                         (X86or_flag GR64:$src1, i64immSExt8:$src2))]>;
def OR64ri32 : RIi32<0x81, MRM1r, (outs GR64:$dst),
                     (ins GR64:$src1, i64i32imm:$src2),
                     "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86or_flag GR64:$src1, i64immSExt32:$src2))]>;
} // Constraints = "$src1 = $dst"

def OR64mr : RI<0x09, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                "or{q}\t{$src, $dst|$dst, $src}",
                [(store (or (load addr:$dst), GR64:$src), addr:$dst),
                 (implicit EFLAGS)]>;
def OR64mi8  : RIi8<0x83, MRM1m, (outs), (ins i64mem:$dst, i64i8imm:$src),
                    "or{q}\t{$src, $dst|$dst, $src}",
                  [(store (or (load addr:$dst), i64immSExt8:$src), addr:$dst),
                   (implicit EFLAGS)]>;
def OR64mi32 : RIi32<0x81, MRM1m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                     "or{q}\t{$src, $dst|$dst, $src}",
              [(store (or (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
               (implicit EFLAGS)]>;

def OR64i32 : RIi32<0x0D, RawFrm, (outs), (ins i64i32imm:$src),
                    "or{q}\t{$src, %rax|%rax, $src}", []>;

let Constraints = "$src1 = $dst" in {
let isCommutable = 1 in
def XOR64rr  : RI<0x31, MRMDestReg,  (outs GR64:$dst), 
                  (ins GR64:$src1, GR64:$src2), 
                  "xor{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86xor_flag GR64:$src1, GR64:$src2))]>;
let isCodeGenOnly = 1 in {
def XOR64rr_REV : RI<0x33, MRMSrcReg, (outs GR64:$dst), 
                     (ins GR64:$src1, GR64:$src2),
                    "xor{q}\t{$src2, $dst|$dst, $src2}", []>;
}
def XOR64rm  : RI<0x33, MRMSrcMem, (outs GR64:$dst), 
                  (ins GR64:$src1, i64mem:$src2), 
                  "xor{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, EFLAGS,
                        (X86xor_flag GR64:$src1, (load addr:$src2)))]>;
def XOR64ri8 : RIi8<0x83, MRM6r,  (outs GR64:$dst), 
                    (ins GR64:$src1, i64i8imm:$src2),
                    "xor{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, EFLAGS,
                          (X86xor_flag GR64:$src1, i64immSExt8:$src2))]>;
def XOR64ri32 : RIi32<0x81, MRM6r, 
                      (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2), 
                      "xor{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, EFLAGS,
                            (X86xor_flag GR64:$src1, i64immSExt32:$src2))]>;
} // Constraints = "$src1 = $dst"

def XOR64mr  : RI<0x31, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                  "xor{q}\t{$src, $dst|$dst, $src}",
                  [(store (xor (load addr:$dst), GR64:$src), addr:$dst),
                   (implicit EFLAGS)]>;
def XOR64mi8 : RIi8<0x83, MRM6m, (outs), (ins i64mem:$dst, i64i8imm :$src),
                    "xor{q}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i64immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
def XOR64mi32 : RIi32<0x81, MRM6m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                      "xor{q}\t{$src, $dst|$dst, $src}",
             [(store (xor (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
              (implicit EFLAGS)]>;
              
def XOR64i32 : RIi32<0x35, RawFrm, (outs), (ins i64i32imm:$src),
                     "xor{q}\t{$src, %rax|%rax, $src}", []>;

} // Defs = [EFLAGS]

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

// Integer comparison
let Defs = [EFLAGS] in {
def TEST64i32 : RIi32<0xa9, RawFrm, (outs), (ins i64i32imm:$src),
                      "test{q}\t{$src, %rax|%rax, $src}", []>;
let isCommutable = 1 in
def TEST64rr : RI<0x85, MRMSrcReg, (outs), (ins GR64:$src1, GR64:$src2),
                  "test{q}\t{$src2, $src1|$src1, $src2}",
                  [(set EFLAGS, (X86cmp (and GR64:$src1, GR64:$src2), 0))]>;
def TEST64rm : RI<0x85, MRMSrcMem, (outs), (ins GR64:$src1, i64mem:$src2),
                  "test{q}\t{$src2, $src1|$src1, $src2}",
                  [(set EFLAGS, (X86cmp (and GR64:$src1, (loadi64 addr:$src2)),
                    0))]>;
def TEST64ri32 : RIi32<0xF7, MRM0r, (outs),
                                        (ins GR64:$src1, i64i32imm:$src2),
                       "test{q}\t{$src2, $src1|$src1, $src2}",
                     [(set EFLAGS, (X86cmp (and GR64:$src1, i64immSExt32:$src2),
                      0))]>;
def TEST64mi32 : RIi32<0xF7, MRM0m, (outs),
                                        (ins i64mem:$src1, i64i32imm:$src2),
                       "test{q}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86cmp (and (loadi64 addr:$src1),
                                           i64immSExt32:$src2), 0))]>;


def CMP64i32 : RIi32<0x3D, RawFrm, (outs), (ins i64i32imm:$src),
                     "cmp{q}\t{$src, %rax|%rax, $src}", []>;
def CMP64rr : RI<0x39, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(set EFLAGS, (X86cmp GR64:$src1, GR64:$src2))]>;

// These are alternate spellings for use by the disassembler, we mark them as
// code gen only to ensure they aren't matched by the assembler.
let isCodeGenOnly = 1 in {
  def CMP64mrmrr : RI<0x3B, MRMSrcReg, (outs), (ins GR64:$src1, GR64:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}", []>;
}

def CMP64mr : RI<0x39, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(set EFLAGS, (X86cmp (loadi64 addr:$src1), GR64:$src2))]>;
def CMP64rm : RI<0x3B, MRMSrcMem, (outs), (ins GR64:$src1, i64mem:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(set EFLAGS, (X86cmp GR64:$src1, (loadi64 addr:$src2)))]>;
def CMP64ri8 : RIi8<0x83, MRM7r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "cmp{q}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp GR64:$src1, i64immSExt8:$src2))]>;
def CMP64ri32 : RIi32<0x81, MRM7r, (outs), (ins GR64:$src1, i64i32imm:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}",
                      [(set EFLAGS, (X86cmp GR64:$src1, i64immSExt32:$src2))]>;
def CMP64mi8 : RIi8<0x83, MRM7m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "cmp{q}\t{$src2, $src1|$src1, $src2}",
                    [(set EFLAGS, (X86cmp (loadi64 addr:$src1),
                                          i64immSExt8:$src2))]>;
def CMP64mi32 : RIi32<0x81, MRM7m, (outs),
                                       (ins i64mem:$src1, i64i32imm:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}",
                      [(set EFLAGS, (X86cmp (loadi64 addr:$src1),
                                            i64immSExt32:$src2))]>;
} // Defs = [EFLAGS]

// Bit tests.
// TODO: BTC, BTR, and BTS
let Defs = [EFLAGS] in {
def BT64rr : RI<0xA3, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
               "bt{q}\t{$src2, $src1|$src1, $src2}",
               [(set EFLAGS, (X86bt GR64:$src1, GR64:$src2))]>, TB;

// Unlike with the register+register form, the memory+register form of the
// bt instruction does not ignore the high bits of the index. From ISel's
// perspective, this is pretty bizarre. Disable these instructions for now.
def BT64mr : RI<0xA3, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
               "bt{q}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi64 addr:$src1), GR64:$src2),
//                (implicit EFLAGS)]
                []
                >, TB;

def BT64ri8 : RIi8<0xBA, MRM4r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                "bt{q}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt GR64:$src1, i64immSExt8:$src2))]>, TB;
// Note that these instructions don't need FastBTMem because that
// only applies when the other operand is in a register. When it's
// an immediate, bt is still fast.
def BT64mi8 : RIi8<0xBA, MRM4m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                "bt{q}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt (loadi64 addr:$src1),
                                     i64immSExt8:$src2))]>, TB;

def BTC64rr : RI<0xBB, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "btc{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTC64mr : RI<0xBB, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "btc{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTC64ri8 : RIi8<0xBA, MRM7r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "btc{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTC64mi8 : RIi8<0xBA, MRM7m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "btc{q}\t{$src2, $src1|$src1, $src2}", []>, TB;

def BTR64rr : RI<0xB3, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "btr{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR64mr : RI<0xB3, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "btr{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR64ri8 : RIi8<0xBA, MRM6r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "btr{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR64mi8 : RIi8<0xBA, MRM6m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "btr{q}\t{$src2, $src1|$src1, $src2}", []>, TB;

def BTS64rr : RI<0xAB, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "bts{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTS64mr : RI<0xAB, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "bts{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTS64ri8 : RIi8<0xBA, MRM5r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "bts{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTS64mi8 : RIi8<0xBA, MRM5m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "bts{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
} // Defs = [EFLAGS]



//===----------------------------------------------------------------------===//
// X86-64 SSE Instructions
//===----------------------------------------------------------------------===//

// Move instructions...

def MOV64toPQIrr : RPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))]>;
def MOVPQIto64rr  : RPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                         "mov{d|q}\t{$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (vector_extract (v2i64 VR128:$src),
                                           (iPTR 0)))]>;

def MOV64toSDrr : RPDI<0x6E, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))]>;
def MOV64toSDrm : S3SI<0x7E, MRMSrcMem, (outs FR64:$dst), (ins i64mem:$src),
                       "movq\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert (loadi64 addr:$src)))]>;

def MOVSDto64rr  : RPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (bitconvert FR64:$src))]>;
def MOVSDto64mr  : RPDI<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                        [(store (i64 (bitconvert FR64:$src)), addr:$dst)]>;

