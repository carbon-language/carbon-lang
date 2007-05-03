//===- ARMInstrVFP.td - VFP support for ARM -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM VP instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM VFP Instruction templates.
//

// ARM Float Instruction
class ASI<dag ops, string asm, list<dag> pattern> : AI<ops, asm, pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class ASI5<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode5, Size4Bytes, IndexModeNone, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

// ARM Double Instruction
class ADI<dag ops, string asm, list<dag> pattern> : AI<ops, asm, pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class ADI5<dag ops, string asm, list<dag> pattern>
  : I<ops, AddrMode5, Size4Bytes, IndexModeNone, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

def SDT_FTOI :
SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisFP<1>]>;
def SDT_ITOF :
SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f32>]>;
def SDT_CMPFP0 :
SDTypeProfile<0, 1, [SDTCisFP<0>]>;
def SDT_FMDRR :
SDTypeProfile<1, 2, [SDTCisVT<0, f64>, SDTCisVT<1, i32>,
                     SDTCisSameAs<1, 2>]>;

def arm_ftoui  : SDNode<"ARMISD::FTOUI", SDT_FTOI>;
def arm_ftosi  : SDNode<"ARMISD::FTOSI", SDT_FTOI>;
def arm_sitof  : SDNode<"ARMISD::SITOF", SDT_ITOF>;
def arm_uitof  : SDNode<"ARMISD::UITOF", SDT_ITOF>;
def arm_fmstat : SDNode<"ARMISD::FMSTAT", SDTRet, [SDNPInFlag,SDNPOutFlag]>;
def arm_cmpfp  : SDNode<"ARMISD::CMPFP", SDT_ARMCmp, [SDNPOutFlag]>;
def arm_cmpfp0 : SDNode<"ARMISD::CMPFPw0", SDT_CMPFP0, [SDNPOutFlag]>;
def arm_fmdrr  : SDNode<"ARMISD::FMDRR", SDT_FMDRR>;

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

let isLoad = 1 in {
def FLDD  : ADI5<(ops DPR:$dst, addrmode5:$addr),
                 "fldd $dst, $addr",
                 [(set DPR:$dst, (load addrmode5:$addr))]>;

def FLDS  : ASI5<(ops SPR:$dst, addrmode5:$addr),
                 "flds $dst, $addr",
                 [(set SPR:$dst, (load addrmode5:$addr))]>;
} // isLoad

let isStore = 1 in {
def FSTD  : ADI5<(ops DPR:$src, addrmode5:$addr),
                 "fstd $src, $addr",
                 [(store DPR:$src, addrmode5:$addr)]>;

def FSTS  : ASI5<(ops SPR:$src, addrmode5:$addr),
                 "fsts $src, $addr",
                 [(store SPR:$src, addrmode5:$addr)]>;
} // isStore

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let isLoad = 1 in {
def FLDMD : ADI5<(ops addrmode5:$addr, reglist:$dst1, variable_ops),
                 "fldm${addr:submode}d ${addr:base}, $dst1",
                 []>;

def FLDMS : ASI5<(ops addrmode5:$addr, reglist:$dst1, variable_ops),
                 "fldm${addr:submode}s ${addr:base}, $dst1",
                 []>;
} // isLoad

let isStore = 1 in {
def FSTMD : ADI5<(ops addrmode5:$addr, reglist:$src1, variable_ops),
                 "fstm${addr:submode}d ${addr:base}, $src1",
                 []>;

def FSTMS : ASI5<(ops addrmode5:$addr, reglist:$src1, variable_ops),
                 "fstm${addr:submode}s ${addr:base}, $src1",
                 []>;
} // isStore

// FLDMX, FSTMX - mixing S/D registers for pre-armv6 cores

//===----------------------------------------------------------------------===//
// FP Binary Operations.
//

def FADDD  : ADI<(ops DPR:$dst, DPR:$a, DPR:$b),
                 "faddd $dst, $a, $b",
                 [(set DPR:$dst, (fadd DPR:$a, DPR:$b))]>;

def FADDS  : ASI<(ops SPR:$dst, SPR:$a, SPR:$b),
                 "fadds $dst, $a, $b",
                 [(set SPR:$dst, (fadd SPR:$a, SPR:$b))]>;

def FCMPED : ADI<(ops DPR:$a, DPR:$b),
                 "fcmped $a, $b",
                 [(arm_cmpfp DPR:$a, DPR:$b)]>;

def FCMPES : ASI<(ops SPR:$a, SPR:$b),
                 "fcmpes $a, $b",
                 [(arm_cmpfp SPR:$a, SPR:$b)]>;

def FDIVD  : ADI<(ops DPR:$dst, DPR:$a, DPR:$b),
                 "fdivd $dst, $a, $b",
                 [(set DPR:$dst, (fdiv DPR:$a, DPR:$b))]>;

def FDIVS  : ASI<(ops SPR:$dst, SPR:$a, SPR:$b),
                 "fdivs $dst, $a, $b",
                 [(set SPR:$dst, (fdiv SPR:$a, SPR:$b))]>;

def FMULD  : ADI<(ops DPR:$dst, DPR:$a, DPR:$b),
                 "fmuld $dst, $a, $b",
                 [(set DPR:$dst, (fmul DPR:$a, DPR:$b))]>;

def FMULS  : ASI<(ops SPR:$dst, SPR:$a, SPR:$b),
                 "fmuls $dst, $a, $b",
                 [(set SPR:$dst, (fmul SPR:$a, SPR:$b))]>;
                 
def FNMULD  : ADI<(ops DPR:$dst, DPR:$a, DPR:$b),
                  "fnmuld $dst, $a, $b",
                  [(set DPR:$dst, (fneg (fmul DPR:$a, DPR:$b)))]>;

def FNMULS  : ASI<(ops SPR:$dst, SPR:$a, SPR:$b),
                  "fnmuls $dst, $a, $b",
                  [(set SPR:$dst, (fneg (fmul SPR:$a, SPR:$b)))]>;

// Match reassociated forms only if not sign dependent rounding.
def : Pat<(fmul (fneg DPR:$a), DPR:$b),
          (FNMULD DPR:$a, DPR:$b)>, Requires<[NoHonorSignDependentRounding]>;
def : Pat<(fmul (fneg SPR:$a), SPR:$b),
          (FNMULS SPR:$a, SPR:$b)>, Requires<[NoHonorSignDependentRounding]>;


def FSUBD  : ADI<(ops DPR:$dst, DPR:$a, DPR:$b),
                 "fsubd $dst, $a, $b",
                 [(set DPR:$dst, (fsub DPR:$a, DPR:$b))]>;

def FSUBS  : ASI<(ops SPR:$dst, SPR:$a, SPR:$b),
                 "fsubs $dst, $a, $b",
                 [(set SPR:$dst, (fsub SPR:$a, SPR:$b))]>;

//===----------------------------------------------------------------------===//
// FP Unary Operations.
//

def FABSD  : ADI<(ops DPR:$dst, DPR:$a),
                 "fabsd $dst, $a",
                 [(set DPR:$dst, (fabs DPR:$a))]>;

def FABSS  : ASI<(ops SPR:$dst, SPR:$a),
                 "fabss $dst, $a",
                 [(set SPR:$dst, (fabs SPR:$a))]>;

def FCMPEZD : ADI<(ops DPR:$a),
                  "fcmpezd $a",
                  [(arm_cmpfp0 DPR:$a)]>;

def FCMPEZS : ASI<(ops SPR:$a),
                  "fcmpezs $a",
                  [(arm_cmpfp0 SPR:$a)]>;

def FCVTDS : ADI<(ops DPR:$dst, SPR:$a),
                 "fcvtds $dst, $a",
                 [(set DPR:$dst, (fextend SPR:$a))]>;

def FCVTSD : ADI<(ops SPR:$dst, DPR:$a),
                 "fcvtsd $dst, $a",
                 [(set SPR:$dst, (fround DPR:$a))]>;

def FCPYD  : ADI<(ops DPR:$dst, DPR:$a),
                 "fcpyd $dst, $a",
                 [/*(set DPR:$dst, DPR:$a)*/]>;

def FCPYS  : ASI<(ops SPR:$dst, SPR:$a),
                 "fcpys $dst, $a",
                 [/*(set SPR:$dst, SPR:$a)*/]>;

def FNEGD  : ADI<(ops DPR:$dst, DPR:$a),
                 "fnegd $dst, $a",
                 [(set DPR:$dst, (fneg DPR:$a))]>;

def FNEGS  : ASI<(ops SPR:$dst, SPR:$a),
                 "fnegs $dst, $a",
                 [(set SPR:$dst, (fneg SPR:$a))]>;

def FSQRTD  : ADI<(ops DPR:$dst, DPR:$a),
                 "fsqrtd $dst, $a",
                 [(set DPR:$dst, (fsqrt DPR:$a))]>;

def FSQRTS  : ASI<(ops SPR:$dst, SPR:$a),
                 "fsqrts $dst, $a",
                 [(set SPR:$dst, (fsqrt SPR:$a))]>;

//===----------------------------------------------------------------------===//
// FP <-> GPR Copies.  Int <-> FP Conversions.
//

def IMPLICIT_DEF_SPR : PseudoInst<(ops SPR:$rD),
                                  "@ IMPLICIT_DEF_SPR $rD",
                                  [(set SPR:$rD, (undef))]>;
def IMPLICIT_DEF_DPR : PseudoInst<(ops DPR:$rD),
                                  "@ IMPLICIT_DEF_DPR $rD",
                                  [(set DPR:$rD, (undef))]>;

def FMRS   : ASI<(ops GPR:$dst, SPR:$src),
                 "fmrs $dst, $src",
                 [(set GPR:$dst, (bitconvert SPR:$src))]>;

def FMSR   : ASI<(ops SPR:$dst, GPR:$src),
                 "fmsr $dst, $src",
                 [(set SPR:$dst, (bitconvert GPR:$src))]>;


def FMRRD  : ADI<(ops GPR:$dst1, GPR:$dst2, DPR:$src),
                 "fmrrd $dst1, $dst2, $src",
                 [/* FIXME: Can't write pattern for multiple result instr*/]>;

// FMDHR: GPR -> SPR
// FMDLR: GPR -> SPR

def FMDRR : ADI<(ops DPR:$dst, GPR:$src1, GPR:$src2),
                "fmdrr $dst, $src1, $src2",
                [(set DPR:$dst, (arm_fmdrr GPR:$src1, GPR:$src2))]>;

// FMRDH: SPR -> GPR
// FMRDL: SPR -> GPR
// FMRRS: SPR -> GPR
// FMRX : SPR system reg -> GPR

// FMSRR: GPR -> SPR


def FMSTAT : ASI<(ops), "fmstat", [(arm_fmstat)]>;

// FMXR: GPR -> VFP Sstem reg


// Int to FP:

def FSITOD : ADI<(ops DPR:$dst, SPR:$a),
                 "fsitod $dst, $a",
                 [(set DPR:$dst, (arm_sitof SPR:$a))]>;

def FSITOS : ASI<(ops SPR:$dst, SPR:$a),
                 "fsitos $dst, $a",
                 [(set SPR:$dst, (arm_sitof SPR:$a))]>;

def FUITOD : ADI<(ops DPR:$dst, SPR:$a),
                 "fuitod $dst, $a",
                 [(set DPR:$dst, (arm_uitof SPR:$a))]>;

def FUITOS : ASI<(ops SPR:$dst, SPR:$a),
                 "fuitos $dst, $a",
                 [(set SPR:$dst, (arm_uitof SPR:$a))]>;

// FP to Int:
// Always set Z bit in the instruction, i.e. "round towards zero" variants.

def FTOSIZD : ADI<(ops SPR:$dst, DPR:$a),
                 "ftosizd $dst, $a",
                 [(set SPR:$dst, (arm_ftosi DPR:$a))]>;

def FTOSIZS : ASI<(ops SPR:$dst, SPR:$a),
                 "ftosizs $dst, $a",
                 [(set SPR:$dst, (arm_ftosi SPR:$a))]>;

def FTOUIZD : ADI<(ops SPR:$dst, DPR:$a),
                 "ftouizd $dst, $a",
                 [(set SPR:$dst, (arm_ftoui DPR:$a))]>;

def FTOUIZS : ASI<(ops SPR:$dst, SPR:$a),
                 "ftouizs $dst, $a",
                 [(set SPR:$dst, (arm_ftoui SPR:$a))]>;

//===----------------------------------------------------------------------===//
// FP FMA Operations.
//

def FMACD : ADI<(ops DPR:$dst, DPR:$dstin, DPR:$a, DPR:$b),
                "fmacd $dst, $a, $b",
                [(set DPR:$dst, (fadd (fmul DPR:$a, DPR:$b), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMACS : ASI<(ops SPR:$dst, SPR:$dstin, SPR:$a, SPR:$b),
                "fmacs $dst, $a, $b",
                [(set SPR:$dst, (fadd (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMSCD : ADI<(ops DPR:$dst, DPR:$dstin, DPR:$a, DPR:$b),
                "fmscd $dst, $a, $b",
                [(set DPR:$dst, (fsub (fmul DPR:$a, DPR:$b), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMSCS : ASI<(ops SPR:$dst, SPR:$dstin, SPR:$a, SPR:$b),
                "fmscs $dst, $a, $b",
                [(set SPR:$dst, (fsub (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMACD : ADI<(ops DPR:$dst, DPR:$dstin, DPR:$a, DPR:$b),
                 "fnmacd $dst, $a, $b",
             [(set DPR:$dst, (fadd (fneg (fmul DPR:$a, DPR:$b)), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMACS : ASI<(ops SPR:$dst, SPR:$dstin, SPR:$a, SPR:$b),
                "fnmacs $dst, $a, $b",
             [(set SPR:$dst, (fadd (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMSCD : ADI<(ops DPR:$dst, DPR:$dstin, DPR:$a, DPR:$b),
                 "fnmscd $dst, $a, $b",
             [(set DPR:$dst, (fsub (fneg (fmul DPR:$a, DPR:$b)), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMSCS : ASI<(ops SPR:$dst, SPR:$dstin, SPR:$a, SPR:$b),
                "fnmscs $dst, $a, $b",
             [(set SPR:$dst, (fsub (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

//===----------------------------------------------------------------------===//
// FP Conditional moves.
//

def FCPYDcc  : ADI<(ops DPR:$dst, DPR:$false, DPR:$true, CCOp:$cc),
                   "fcpyd$cc $dst, $true",
                   [(set DPR:$dst, (ARMcmov DPR:$false, DPR:$true, imm:$cc))]>,
                   RegConstraint<"$false = $dst">;

def FCPYScc  : ASI<(ops SPR:$dst, SPR:$false, SPR:$true, CCOp:$cc),
                   "fcpys$cc $dst, $true",
                   [(set SPR:$dst, (ARMcmov SPR:$false, SPR:$true, imm:$cc))]>,
                   RegConstraint<"$false = $dst">;

def FNEGDcc  : ADI<(ops DPR:$dst, DPR:$false, DPR:$true, CCOp:$cc),
                   "fnegd$cc $dst, $true",
                   [(set DPR:$dst, (ARMcneg DPR:$false, DPR:$true, imm:$cc))]>,
                   RegConstraint<"$false = $dst">;

def FNEGScc  : ASI<(ops SPR:$dst, SPR:$false, SPR:$true, CCOp:$cc),
                   "fnegs$cc $dst, $true",
                   [(set SPR:$dst, (ARMcneg SPR:$false, SPR:$true, imm:$cc))]>,
                   RegConstraint<"$false = $dst">;
