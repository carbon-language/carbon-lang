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
// FP Pattern fragments

def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fpimmneg0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

let Uses = [PSW], usesCustomInserter = 1 in {
  def SelectF32 : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2, i8imm:$cc),
                        "# SelectF32 PSEUDO",
                        [(set FP32:$dst,
                              (SystemZselect FP32:$src1, FP32:$src2, imm:$cc, PSW))]>;
  def SelectF64 : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2, i8imm:$cc),
                        "# SelectF64 PSEUDO",
                        [(set FP64:$dst,
                              (SystemZselect FP64:$src1, FP64:$src2, imm:$cc, PSW))]>;
}

//===----------------------------------------------------------------------===//
// Move Instructions

// Floating point constant loads.
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def LD_Fp032 : Pseudo<(outs FP32:$dst), (ins),
                      "lzer\t{$dst}",
                      [(set FP32:$dst, fpimm0)]>;
def LD_Fp064 : Pseudo<(outs FP64:$dst), (ins),
                      "lzdr\t{$dst}",
                      [(set FP64:$dst, fpimm0)]>;
}

let neverHasSideEffects = 1 in {
def FMOV32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src),
                      "ler\t{$dst, $src}",
                      []>;
def FMOV64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src),
                      "ldr\t{$dst, $src}",
                      []>;
}

let canFoldAsLoad = 1, isReMaterializable = 1 in {
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

def FCOPYSIGN32 : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                         "cpsdr\t{$dst, $src2, $src1}",
                         [(set FP32:$dst, (fcopysign FP32:$src1, FP32:$src2))]>;
def FCOPYSIGN64 : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                         "cpsdr\t{$dst, $src2, $src1}",
                         [(set FP64:$dst, (fcopysign FP64:$src1, FP64:$src2))]>;

//===----------------------------------------------------------------------===//
// Arithmetic Instructions


let Defs = [PSW] in {
def FNEG32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src),
                       "lcebr\t{$dst, $src}",
                       [(set FP32:$dst, (fneg FP32:$src)),
                        (implicit PSW)]>;
def FNEG64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src),
                       "lcdbr\t{$dst, $src}",
                       [(set FP64:$dst, (fneg FP64:$src)),
                        (implicit PSW)]>;

def FABS32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src),
                       "lpebr\t{$dst, $src}",
                       [(set FP32:$dst, (fabs FP32:$src)),
                        (implicit PSW)]>;
def FABS64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src),
                       "lpdbr\t{$dst, $src}",
                       [(set FP64:$dst, (fabs FP64:$src)),
                        (implicit PSW)]>;

def FNABS32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src),
                       "lnebr\t{$dst, $src}",
                       [(set FP32:$dst, (fneg(fabs FP32:$src))),
                        (implicit PSW)]>;
def FNABS64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src),
                       "lndbr\t{$dst, $src}",
                       [(set FP64:$dst, (fneg(fabs FP64:$src))),
                        (implicit PSW)]>;
}

let isTwoAddress = 1 in {
let Defs = [PSW] in {
let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
def FADD32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "aebr\t{$dst, $src2}",
                       [(set FP32:$dst, (fadd FP32:$src1, FP32:$src2)),
                        (implicit PSW)]>;
def FADD64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "adbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fadd FP64:$src1, FP64:$src2)),
                        (implicit PSW)]>;
}

def FADD32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr12:$src2),
                       "aeb\t{$dst, $src2}",
                       [(set FP32:$dst, (fadd FP32:$src1, (load rriaddr12:$src2))),
                        (implicit PSW)]>;
def FADD64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr12:$src2),
                       "adb\t{$dst, $src2}",
                       [(set FP64:$dst, (fadd FP64:$src1, (load rriaddr12:$src2))),
                        (implicit PSW)]>;

def FSUB32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "sebr\t{$dst, $src2}",
                       [(set FP32:$dst, (fsub FP32:$src1, FP32:$src2)),
                        (implicit PSW)]>;
def FSUB64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "sdbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fsub FP64:$src1, FP64:$src2)),
                        (implicit PSW)]>;

def FSUB32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr12:$src2),
                       "seb\t{$dst, $src2}",
                       [(set FP32:$dst, (fsub FP32:$src1, (load rriaddr12:$src2))),
                        (implicit PSW)]>;
def FSUB64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr12:$src2),
                       "sdb\t{$dst, $src2}",
                       [(set FP64:$dst, (fsub FP64:$src1, (load rriaddr12:$src2))),
                        (implicit PSW)]>;
} // Defs = [PSW]

let isCommutable = 1 in { // X = MUL Y, Z  == X = MUL Z, Y
def FMUL32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "meebr\t{$dst, $src2}",
                       [(set FP32:$dst, (fmul FP32:$src1, FP32:$src2))]>;
def FMUL64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "mdbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fmul FP64:$src1, FP64:$src2))]>;
}

def FMUL32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr12:$src2),
                       "meeb\t{$dst, $src2}",
                       [(set FP32:$dst, (fmul FP32:$src1, (load rriaddr12:$src2)))]>;
def FMUL64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr12:$src2),
                       "mdb\t{$dst, $src2}",
                       [(set FP64:$dst, (fmul FP64:$src1, (load rriaddr12:$src2)))]>;

def FMADD32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2, FP32:$src3),
                       "maebr\t{$dst, $src3, $src2}",
                       [(set FP32:$dst, (fadd (fmul FP32:$src2, FP32:$src3),
                                              FP32:$src1))]>;
def FMADD32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr12:$src2, FP32:$src3),
                       "maeb\t{$dst, $src3, $src2}",
                       [(set FP32:$dst, (fadd (fmul (load rriaddr12:$src2),
                                                     FP32:$src3),
                                              FP32:$src1))]>;

def FMADD64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2, FP64:$src3),
                       "madbr\t{$dst, $src3, $src2}",
                       [(set FP64:$dst, (fadd (fmul FP64:$src2, FP64:$src3),
                                              FP64:$src1))]>;
def FMADD64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr12:$src2, FP64:$src3),
                       "madb\t{$dst, $src3, $src2}",
                       [(set FP64:$dst, (fadd (fmul (load rriaddr12:$src2),
                                                     FP64:$src3),
                                              FP64:$src1))]>;

def FMSUB32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2, FP32:$src3),
                       "msebr\t{$dst, $src3, $src2}",
                       [(set FP32:$dst, (fsub (fmul FP32:$src2, FP32:$src3),
                                              FP32:$src1))]>;
def FMSUB32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr12:$src2, FP32:$src3),
                       "mseb\t{$dst, $src3, $src2}",
                       [(set FP32:$dst, (fsub (fmul (load rriaddr12:$src2),
                                                     FP32:$src3),
                                              FP32:$src1))]>;

def FMSUB64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2, FP64:$src3),
                       "msdbr\t{$dst, $src3, $src2}",
                       [(set FP64:$dst, (fsub (fmul FP64:$src2, FP64:$src3),
                                              FP64:$src1))]>;
def FMSUB64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr12:$src2, FP64:$src3),
                       "msdb\t{$dst, $src3, $src2}",
                       [(set FP64:$dst, (fsub (fmul (load rriaddr12:$src2),
                                                     FP64:$src3),
                                              FP64:$src1))]>;

def FDIV32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src1, FP32:$src2),
                       "debr\t{$dst, $src2}",
                       [(set FP32:$dst, (fdiv FP32:$src1, FP32:$src2))]>;
def FDIV64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src1, FP64:$src2),
                       "ddbr\t{$dst, $src2}",
                       [(set FP64:$dst, (fdiv FP64:$src1, FP64:$src2))]>;

def FDIV32rm : Pseudo<(outs FP32:$dst), (ins FP32:$src1, rriaddr12:$src2),
                       "deb\t{$dst, $src2}",
                       [(set FP32:$dst, (fdiv FP32:$src1, (load rriaddr12:$src2)))]>;
def FDIV64rm : Pseudo<(outs FP64:$dst), (ins FP64:$src1, rriaddr12:$src2),
                       "ddb\t{$dst, $src2}",
                       [(set FP64:$dst, (fdiv FP64:$src1, (load rriaddr12:$src2)))]>;

} // isTwoAddress = 1

def FSQRT32rr : Pseudo<(outs FP32:$dst), (ins FP32:$src),
                       "sqebr\t{$dst, $src}",
                       [(set FP32:$dst, (fsqrt FP32:$src))]>;
def FSQRT64rr : Pseudo<(outs FP64:$dst), (ins FP64:$src),
                       "sqdbr\t{$dst, $src}",
                       [(set FP64:$dst, (fsqrt FP64:$src))]>;

def FSQRT32rm : Pseudo<(outs FP32:$dst), (ins rriaddr12:$src),
                       "sqeb\t{$dst, $src}",
                       [(set FP32:$dst, (fsqrt (load rriaddr12:$src)))]>;
def FSQRT64rm : Pseudo<(outs FP64:$dst), (ins rriaddr12:$src),
                       "sqdb\t{$dst, $src}",
                       [(set FP64:$dst, (fsqrt (load rriaddr12:$src)))]>;

def FROUND64r32 : Pseudo<(outs FP32:$dst), (ins FP64:$src),
                         "ledbr\t{$dst, $src}",
                         [(set FP32:$dst, (fround FP64:$src))]>;

def FEXT32r64   : Pseudo<(outs FP64:$dst), (ins FP32:$src),
                         "ldebr\t{$dst, $src}",
                         [(set FP64:$dst, (fextend FP32:$src))]>;
def FEXT32m64   : Pseudo<(outs FP64:$dst), (ins rriaddr12:$src),
                         "ldeb\t{$dst, $src}",
                         [(set FP64:$dst, (fextend (load rriaddr12:$src)))]>;

let Defs = [PSW] in {
def FCONVFP32   : Pseudo<(outs FP32:$dst), (ins GR32:$src),
                         "cefbr\t{$dst, $src}",
                         [(set FP32:$dst, (sint_to_fp GR32:$src)),
                          (implicit PSW)]>;
def FCONVFP32r64: Pseudo<(outs FP32:$dst), (ins GR64:$src),
                         "cegbr\t{$dst, $src}",
                         [(set FP32:$dst, (sint_to_fp GR64:$src)),
                          (implicit PSW)]>;

def FCONVFP64r32: Pseudo<(outs FP64:$dst), (ins GR32:$src),
                         "cdfbr\t{$dst, $src}",
                         [(set FP64:$dst, (sint_to_fp GR32:$src)),
                          (implicit PSW)]>;
def FCONVFP64   : Pseudo<(outs FP64:$dst), (ins GR64:$src),
                         "cdgbr\t{$dst, $src}",
                         [(set FP64:$dst, (sint_to_fp GR64:$src)),
                          (implicit PSW)]>;

def FCONVGR32   : Pseudo<(outs GR32:$dst), (ins FP32:$src),
                         "cfebr\t{$dst, 5, $src}",
                         [(set GR32:$dst, (fp_to_sint FP32:$src)),
                          (implicit PSW)]>;
def FCONVGR32r64: Pseudo<(outs GR32:$dst), (ins FP64:$src),
                         "cfdbr\t{$dst, 5, $src}",
                         [(set GR32:$dst, (fp_to_sint FP64:$src)),
                          (implicit PSW)]>;

def FCONVGR64r32: Pseudo<(outs GR64:$dst), (ins FP32:$src),
                         "cgebr\t{$dst, 5, $src}",
                         [(set GR64:$dst, (fp_to_sint FP32:$src)),
                          (implicit PSW)]>;
def FCONVGR64   : Pseudo<(outs GR64:$dst), (ins FP64:$src),
                         "cgdbr\t{$dst, 5, $src}",
                         [(set GR64:$dst, (fp_to_sint FP64:$src)),
                          (implicit PSW)]>;
} // Defs = [PSW]

def FBCONVG64   : Pseudo<(outs GR64:$dst), (ins FP64:$src),
                         "lgdr\t{$dst, $src}",
                         [(set GR64:$dst, (bitconvert FP64:$src))]>;
def FBCONVF64   : Pseudo<(outs FP64:$dst), (ins GR64:$src),
                         "ldgr\t{$dst, $src}",
                         [(set FP64:$dst, (bitconvert GR64:$src))]>;

//===----------------------------------------------------------------------===//
// Test instructions (like AND but do not produce any result)

// Integer comparisons
let Defs = [PSW] in {
def FCMP32rr : Pseudo<(outs), (ins FP32:$src1, FP32:$src2),
                      "cebr\t$src1, $src2",
                      [(set PSW, (SystemZcmp FP32:$src1, FP32:$src2))]>;
def FCMP64rr : Pseudo<(outs), (ins FP64:$src1, FP64:$src2),
                      "cdbr\t$src1, $src2",
                      [(set PSW, (SystemZcmp FP64:$src1, FP64:$src2))]>;

def FCMP32rm : Pseudo<(outs), (ins FP32:$src1, rriaddr12:$src2),
                      "ceb\t$src1, $src2",
                      [(set PSW, (SystemZcmp FP32:$src1,
                                             (load rriaddr12:$src2)))]>;
def FCMP64rm : Pseudo<(outs), (ins FP64:$src1, rriaddr12:$src2),
                      "cdb\t$src1, $src2",
                      [(set PSW, (SystemZcmp FP64:$src1,
                                             (load rriaddr12:$src2)))]>;
} // Defs = [PSW]

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Floating point constant -0.0
def : Pat<(f32 fpimmneg0), (FNEG32rr (LD_Fp032))>;
def : Pat<(f64 fpimmneg0), (FNEG64rr (LD_Fp064))>;
