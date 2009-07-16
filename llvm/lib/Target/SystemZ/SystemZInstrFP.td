//===- SystemZInstrFP.td - SystemZ FP Instruction defs --------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the SystemZ (binary) floating point instructions in 
// TableGen format.
//
//===----------------------------------------------------------------------===//

// FIXME: multiclassify!

//===----------------------------------------------------------------------===//
// Move Instructions

let neverHasSideEffects = 1 in {
def FMOV32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src),
                      "ler\t{$dst, $src}",
                      []>;
def FMOV64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src),
                      "ldr\t{$dst, $src}",
                      []>;
}

let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def FMOV32rm  : Pseudo<(outs FP32:$dst), (ins rriaddr12:$src),
                      "le\t{$dst, $src}",
                      [(set FP32:$dst, (load rriaddr12:$src))]>;
def FMOV32rmy : Pseudo<(outs FP32:$dst), (ins rriaddr:$src),
                      "ley\t{$dst, $src}",
                      [(set FP32:$dst, (load rriaddr:$src))]>;
def FMOV64rm  : Pseudo<(outs FP64:$dst), (ins rriaddr12:$src),
                      "ld\t{$dst, $src}",
                      [(set FP64:$dst, (load rriaddr12:$src))]>;
def FMOV64rmy : Pseudo<(outs FP64:$dst), (ins rriaddr:$src),
                      "ldy\t{$dst, $src}",
                      [(set FP64:$dst, (load rriaddr:$src))]>;
}

def FMOV32mr  : Pseudo<(outs), (ins rriaddr12:$dst, FP32:$src),
                       "ste\t{$src, $dst}",
                       [(store FP32:$src, rriaddr12:$dst)]>;
def FMOV32mry : Pseudo<(outs), (ins rriaddr:$dst, FP32:$src),
                       "stey\t{$src, $dst}",
                       [(store FP32:$src, rriaddr:$dst)]>;
def FMOV64mr  : Pseudo<(outs), (ins rriaddr12:$dst, FP64:$src),
                       "std\t{$src, $dst}",
                       [(store FP64:$src, rriaddr12:$dst)]>;
def FMOV64mry : Pseudo<(outs), (ins rriaddr:$dst, FP64:$src),
                       "stdy\t{$src, $dst}",
                       [(store FP64:$src, rriaddr:$dst)]>;

//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let isTwoAddress = 1 in {

let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
def FADD32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "aebr\t{$dst, $src2}",
                       [(set FP32:$dst, (fadd FP32:$src1, FP32:$src2))]>;
def FADD64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "adbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fadd FP64:$src1, FP64:$src2))]>;
}

def FADD32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr:$src2),
                       "aeb\t{$dst, $src2}",
                       [(set FP32:$dst, (fadd FP32:$src1, (load rriaddr:$src2)))]>;
def FADD64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr:$src2),
                       "adb\t{$dst, $src2}",
                       [(set FP64:$dst, (fadd FP64:$src1, (load rriaddr:$src2)))]>;

def FSUB32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "sebr\t{$dst, $src2}",
                       [(set FP32:$dst, (fsub FP32:$src1, FP32:$src2))]>;
def FSUB64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "sdbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fsub FP64:$src1, FP64:$src2))]>;

def FSUB32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr:$src2),
                       "seb\t{$dst, $src2}",
                       [(set FP32:$dst, (fsub FP32:$src1, (load rriaddr:$src2)))]>;
def FSUB64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr:$src2),
                       "sdb\t{$dst, $src2}",
                       [(set FP64:$dst, (fsub FP64:$src1, (load rriaddr:$src2)))]>;

let isCommutable = 1 in { // X = MUL Y, Z  == X = MUL Z, Y
def FMUL32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "meebr\t{$dst, $src2}",
                       [(set FP32:$dst, (fmul FP32:$src1, FP32:$src2))]>;
def FMUL64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "mdbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fmul FP64:$src1, FP64:$src2))]>;
}

def FMUL32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr:$src2),
                       "meeb\t{$dst, $src2}",
                       [(set FP32:$dst, (fmul FP32:$src1, (load rriaddr:$src2)))]>;
def FMUL64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr:$src2),
                       "mdb\t{$dst, $src2}",
                       [(set FP64:$dst, (fmul FP64:$src1, (load rriaddr:$src2)))]>;

def FDIV32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "debr\t{$dst, $src2}",
                       [(set FP32:$dst, (fdiv FP32:$src1, FP32:$src2))]>;
def FDIV64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "ddbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fdiv FP64:$src1, FP64:$src2))]>;

def FDIV32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr:$src2),
                       "deb\t{$dst, $src2}",
                       [(set FP32:$dst, (fdiv FP32:$src1, (load rriaddr:$src2)))]>;
def FDIV64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr:$src2),
                       "ddb\t{$dst, $src2}",
                       [(set FP64:$dst, (fdiv FP64:$src1, (load rriaddr:$src2)))]>;

} // isTwoAddress = 1

def FROUND64r32 : Pseudo<(outs FP32:$dst), (ins FP64:$src),
                         "ledbr\t{$dst, $src}",
                         [(set FP32:$dst, (fround FP64:$src))]>;

def FCONVFP32   : Pseudo<(outs FP32:$dst), (ins GR32:$src),
                         "cefbr\t{$dst, $src}",
                         [(set FP32:$dst, (sint_to_fp GR32:$src))]>;
def FCONVFP32r64: Pseudo<(outs FP32:$dst), (ins GR64:$src),
                         "cegbr\t{$dst, $src}",
                         [(set FP32:$dst, (sint_to_fp GR64:$src))]>;

def FCONVFP64r32: Pseudo<(outs FP64:$dst), (ins GR32:$src),
                         "cdfbr\t{$dst, $src}",
                         [(set FP64:$dst, (sint_to_fp GR32:$src))]>;

def FCONVFP64   : Pseudo<(outs FP64:$dst), (ins GR64:$src),
                         "cdgbr\t{$dst, $src}",
                         [(set FP64:$dst, (sint_to_fp GR64:$src))]>;

//===----------------------------------------------------------------------===//
// Test instructions (like AND but do not produce any result)

// Integer comparisons
let Defs = [PSW] in {
def FCMP32rr : Pseudo<(outs), (ins FP32:$src1, FP32:$src2),
                      "cebr\t$src1, $src2",
                      [(SystemZcmp FP32:$src1, FP32:$src2), (implicit PSW)]>;
def FCMP64rr : Pseudo<(outs), (ins FP64:$src1, FP64:$src2),
                      "cdbr\t$src1, $src2",
                      [(SystemZcmp FP64:$src1, FP64:$src2), (implicit PSW)]>;

def FCMP32rm : Pseudo<(outs), (ins FP32:$src1, rriaddr:$src2),
                      "ceb\t$src1, $src2",
                      [(SystemZcmp FP32:$src1, (load rriaddr:$src2)),
                       (implicit PSW)]>;
def FCMP64rm : Pseudo<(outs), (ins FP64:$src1, rriaddr:$src2),
                      "cdb\t$src1, $src2",
                      [(SystemZcmp FP64:$src1, (load rriaddr:$src2)),
                       (implicit PSW)]>;
} // Defs = [PSW]
