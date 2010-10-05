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


//===----------------------------------------------------------------------===//
//  Arithmetic Instructions...
//

let CodeSize = 2, Defs = [EFLAGS] in {

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



//===----------------------------------------------------------------------===//
//  Logical Instructions...
//

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

