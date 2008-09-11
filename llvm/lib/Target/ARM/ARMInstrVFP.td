//===- ARMInstrVFP.td - VFP support for ARM -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM VFP instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM VFP Instruction templates.
//

// ARM Float Instruction
class ASI<dag outs, dag ins, string opc, string asm, list<dag> pattern>
  : AI<0x0, outs, ins, VFPFrm, opc, asm, pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class ASI5<dag outs, dag ins, string opc, string asm, list<dag> pattern>
  : I<0x0, outs, ins, AddrMode5, Size4Bytes, IndexModeNone,
      VFPFrm, opc, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

// ARM Double Instruction
class ADI<dag outs, dag ins, string opc, string asm, list<dag> pattern>
  : AI<0x0, outs, ins, VFPFrm, opc, asm, pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class ADI5<dag outs, dag ins, string opc, string asm, list<dag> pattern>
  : I<0x0, outs, ins, AddrMode5, Size4Bytes, IndexModeNone,
      VFPFrm, opc, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

// Special cases.
class AXSI<dag outs, dag ins, string asm, list<dag> pattern>
  : XI<0x0, outs, ins, AddrModeNone, Size4Bytes, IndexModeNone,
       VFPFrm, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class AXSI5<dag outs, dag ins, string asm, list<dag> pattern>
  : XI<0x0, outs, ins, AddrMode5, Size4Bytes, IndexModeNone,
       VFPFrm, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class AXDI<dag outs, dag ins, string asm, list<dag> pattern>
  : XI<0x0, outs, ins, AddrModeNone, Size4Bytes, IndexModeNone,
       VFPFrm, asm, "", pattern> {
  // TODO: Mark the instructions with the appropriate subtarget info.
}

class AXDI5<dag outs, dag ins, string asm, list<dag> pattern>
  : XI<0x0, outs, ins, AddrMode5, Size4Bytes, IndexModeNone,
       VFPFrm, asm, "", pattern> {
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
def arm_fmstat : SDNode<"ARMISD::FMSTAT", SDTNone, [SDNPInFlag,SDNPOutFlag]>;
def arm_cmpfp  : SDNode<"ARMISD::CMPFP", SDT_ARMCmp, [SDNPOutFlag]>;
def arm_cmpfp0 : SDNode<"ARMISD::CMPFPw0", SDT_CMPFP0, [SDNPOutFlag]>;
def arm_fmdrr  : SDNode<"ARMISD::FMDRR", SDT_FMDRR>;

//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

let isSimpleLoad = 1 in {
def FLDD  : ADI5<(outs DPR:$dst), (ins addrmode5:$addr),
                 "fldd", " $dst, $addr",
                 [(set DPR:$dst, (load addrmode5:$addr))]>;

def FLDS  : ASI5<(outs SPR:$dst), (ins addrmode5:$addr),
                 "flds", " $dst, $addr",
                 [(set SPR:$dst, (load addrmode5:$addr))]>;
} // isSimpleLoad

def FSTD  : ADI5<(outs), (ins DPR:$src, addrmode5:$addr),
                 "fstd", " $src, $addr",
                 [(store DPR:$src, addrmode5:$addr)]>;

def FSTS  : ASI5<(outs), (ins SPR:$src, addrmode5:$addr),
                 "fsts", " $src, $addr",
                 [(store SPR:$src, addrmode5:$addr)]>;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let mayLoad = 1 in {
def FLDMD : AXDI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$dst1,
                           variable_ops),
                  "fldm${addr:submode}d${p} ${addr:base}, $dst1",
                  []>;

def FLDMS : AXSI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$dst1,
                           variable_ops),
                  "fldm${addr:submode}s${p} ${addr:base}, $dst1",
                  []>;
}

let mayStore = 1 in {
def FSTMD : AXDI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$src1,
                           variable_ops),
                 "fstm${addr:submode}d${p} ${addr:base}, $src1",
                 []>;

def FSTMS : AXSI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$src1,
                           variable_ops),
                 "fstm${addr:submode}s${p} ${addr:base}, $src1",
                 []>;
} // mayStore

// FLDMX, FSTMX - mixing S/D registers for pre-armv6 cores

//===----------------------------------------------------------------------===//
// FP Binary Operations.
//

def FADDD  : ADI<(outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 "faddd", " $dst, $a, $b",
                 [(set DPR:$dst, (fadd DPR:$a, DPR:$b))]>;

def FADDS  : ASI<(outs SPR:$dst), (ins SPR:$a, SPR:$b),
                 "fadds", " $dst, $a, $b",
                 [(set SPR:$dst, (fadd SPR:$a, SPR:$b))]>;

def FCMPED : ADI<(outs), (ins DPR:$a, DPR:$b),
                 "fcmped", " $a, $b",
                 [(arm_cmpfp DPR:$a, DPR:$b)]>;

def FCMPES : ASI<(outs), (ins SPR:$a, SPR:$b),
                 "fcmpes", " $a, $b",
                 [(arm_cmpfp SPR:$a, SPR:$b)]>;

def FDIVD  : ADI<(outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 "fdivd", " $dst, $a, $b",
                 [(set DPR:$dst, (fdiv DPR:$a, DPR:$b))]>;

def FDIVS  : ASI<(outs SPR:$dst), (ins SPR:$a, SPR:$b),
                 "fdivs", " $dst, $a, $b",
                 [(set SPR:$dst, (fdiv SPR:$a, SPR:$b))]>;

def FMULD  : ADI<(outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 "fmuld", " $dst, $a, $b",
                 [(set DPR:$dst, (fmul DPR:$a, DPR:$b))]>;

def FMULS  : ASI<(outs SPR:$dst), (ins SPR:$a, SPR:$b),
                 "fmuls", " $dst, $a, $b",
                 [(set SPR:$dst, (fmul SPR:$a, SPR:$b))]>;
                 
def FNMULD  : ADI<(outs DPR:$dst), (ins DPR:$a, DPR:$b),
                  "fnmuld", " $dst, $a, $b",
                  [(set DPR:$dst, (fneg (fmul DPR:$a, DPR:$b)))]>;

def FNMULS  : ASI<(outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  "fnmuls", " $dst, $a, $b",
                  [(set SPR:$dst, (fneg (fmul SPR:$a, SPR:$b)))]>;

// Match reassociated forms only if not sign dependent rounding.
def : Pat<(fmul (fneg DPR:$a), DPR:$b),
          (FNMULD DPR:$a, DPR:$b)>, Requires<[NoHonorSignDependentRounding]>;
def : Pat<(fmul (fneg SPR:$a), SPR:$b),
          (FNMULS SPR:$a, SPR:$b)>, Requires<[NoHonorSignDependentRounding]>;


def FSUBD  : ADI<(outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 "fsubd", " $dst, $a, $b",
                 [(set DPR:$dst, (fsub DPR:$a, DPR:$b))]>;

def FSUBS  : ASI<(outs SPR:$dst), (ins SPR:$a, SPR:$b),
                 "fsubs", " $dst, $a, $b",
                 [(set SPR:$dst, (fsub SPR:$a, SPR:$b))]>;

//===----------------------------------------------------------------------===//
// FP Unary Operations.
//

def FABSD  : ADI<(outs DPR:$dst), (ins DPR:$a),
                 "fabsd", " $dst, $a",
                 [(set DPR:$dst, (fabs DPR:$a))]>;

def FABSS  : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "fabss", " $dst, $a",
                 [(set SPR:$dst, (fabs SPR:$a))]>;

def FCMPEZD : ADI<(outs), (ins DPR:$a),
                  "fcmpezd", " $a",
                  [(arm_cmpfp0 DPR:$a)]>;

def FCMPEZS : ASI<(outs), (ins SPR:$a),
                  "fcmpezs", " $a",
                  [(arm_cmpfp0 SPR:$a)]>;

def FCVTDS : ADI<(outs DPR:$dst), (ins SPR:$a),
                 "fcvtds", " $dst, $a",
                 [(set DPR:$dst, (fextend SPR:$a))]>;

def FCVTSD : ADI<(outs SPR:$dst), (ins DPR:$a),
                 "fcvtsd", " $dst, $a",
                 [(set SPR:$dst, (fround DPR:$a))]>;

def FCPYD  : ADI<(outs DPR:$dst), (ins DPR:$a),
                 "fcpyd", " $dst, $a", []>;

def FCPYS  : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "fcpys", " $dst, $a", []>;

def FNEGD  : ADI<(outs DPR:$dst), (ins DPR:$a),
                 "fnegd", " $dst, $a",
                 [(set DPR:$dst, (fneg DPR:$a))]>;

def FNEGS  : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "fnegs", " $dst, $a",
                 [(set SPR:$dst, (fneg SPR:$a))]>;

def FSQRTD  : ADI<(outs DPR:$dst), (ins DPR:$a),
                 "fsqrtd", " $dst, $a",
                 [(set DPR:$dst, (fsqrt DPR:$a))]>;

def FSQRTS  : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "fsqrts", " $dst, $a",
                 [(set SPR:$dst, (fsqrt SPR:$a))]>;

//===----------------------------------------------------------------------===//
// FP <-> GPR Copies.  Int <-> FP Conversions.
//

def FMRS   : ASI<(outs GPR:$dst), (ins SPR:$src),
                 "fmrs", " $dst, $src",
                 [(set GPR:$dst, (bitconvert SPR:$src))]>;

def FMSR   : ASI<(outs SPR:$dst), (ins GPR:$src),
                 "fmsr", " $dst, $src",
                 [(set SPR:$dst, (bitconvert GPR:$src))]>;


def FMRRD  : ADI<(outs GPR:$dst1, GPR:$dst2), (ins DPR:$src),
                 "fmrrd", " $dst1, $dst2, $src",
                 [/* FIXME: Can't write pattern for multiple result instr*/]>;

// FMDHR: GPR -> SPR
// FMDLR: GPR -> SPR

def FMDRR : ADI<(outs DPR:$dst), (ins GPR:$src1, GPR:$src2),
                "fmdrr", " $dst, $src1, $src2",
                [(set DPR:$dst, (arm_fmdrr GPR:$src1, GPR:$src2))]>;

// FMRDH: SPR -> GPR
// FMRDL: SPR -> GPR
// FMRRS: SPR -> GPR
// FMRX : SPR system reg -> GPR

// FMSRR: GPR -> SPR

let Defs = [CPSR] in
def FMSTAT : ASI<(outs), (ins), "fmstat", "", [(arm_fmstat)]>;

// FMXR: GPR -> VFP Sstem reg


// Int to FP:

def FSITOD : ADI<(outs DPR:$dst), (ins SPR:$a),
                 "fsitod", " $dst, $a",
                 [(set DPR:$dst, (arm_sitof SPR:$a))]>;

def FSITOS : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "fsitos", " $dst, $a",
                 [(set SPR:$dst, (arm_sitof SPR:$a))]>;

def FUITOD : ADI<(outs DPR:$dst), (ins SPR:$a),
                 "fuitod", " $dst, $a",
                 [(set DPR:$dst, (arm_uitof SPR:$a))]>;

def FUITOS : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "fuitos", " $dst, $a",
                 [(set SPR:$dst, (arm_uitof SPR:$a))]>;

// FP to Int:
// Always set Z bit in the instruction, i.e. "round towards zero" variants.

def FTOSIZD : ADI<(outs SPR:$dst), (ins DPR:$a),
                 "ftosizd", " $dst, $a",
                 [(set SPR:$dst, (arm_ftosi DPR:$a))]>;

def FTOSIZS : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "ftosizs", " $dst, $a",
                 [(set SPR:$dst, (arm_ftosi SPR:$a))]>;

def FTOUIZD : ADI<(outs SPR:$dst), (ins DPR:$a),
                 "ftouizd", " $dst, $a",
                 [(set SPR:$dst, (arm_ftoui DPR:$a))]>;

def FTOUIZS : ASI<(outs SPR:$dst), (ins SPR:$a),
                 "ftouizs", " $dst, $a",
                 [(set SPR:$dst, (arm_ftoui SPR:$a))]>;

//===----------------------------------------------------------------------===//
// FP FMA Operations.
//

def FMACD : ADI<(outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                "fmacd", " $dst, $a, $b",
                [(set DPR:$dst, (fadd (fmul DPR:$a, DPR:$b), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMACS : ASI<(outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                "fmacs", " $dst, $a, $b",
                [(set SPR:$dst, (fadd (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMSCD : ADI<(outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                "fmscd", " $dst, $a, $b",
                [(set DPR:$dst, (fsub (fmul DPR:$a, DPR:$b), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMSCS : ASI<(outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                "fmscs", " $dst, $a, $b",
                [(set SPR:$dst, (fsub (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMACD : ADI<(outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                 "fnmacd", " $dst, $a, $b",
             [(set DPR:$dst, (fadd (fneg (fmul DPR:$a, DPR:$b)), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMACS : ASI<(outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                "fnmacs", " $dst, $a, $b",
             [(set SPR:$dst, (fadd (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMSCD : ADI<(outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                 "fnmscd", " $dst, $a, $b",
             [(set DPR:$dst, (fsub (fneg (fmul DPR:$a, DPR:$b)), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMSCS : ASI<(outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                "fnmscs", " $dst, $a, $b",
             [(set SPR:$dst, (fsub (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

//===----------------------------------------------------------------------===//
// FP Conditional moves.
//

def FCPYDcc  : ADI<(outs DPR:$dst), (ins DPR:$false, DPR:$true),
                    "fcpyd", " $dst, $true",
                [/*(set DPR:$dst, (ARMcmov DPR:$false, DPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def FCPYScc  : ASI<(outs SPR:$dst), (ins SPR:$false, SPR:$true),
                    "fcpys", " $dst, $true",
                [/*(set SPR:$dst, (ARMcmov SPR:$false, SPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def FNEGDcc  : ADI<(outs DPR:$dst), (ins DPR:$false, DPR:$true),
                    "fnegd", " $dst, $true",
                [/*(set DPR:$dst, (ARMcneg DPR:$false, DPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def FNEGScc  : ASI<(outs SPR:$dst), (ins SPR:$false, SPR:$true),
                    "fnegs", " $dst, $true",
                [/*(set SPR:$dst, (ARMcneg SPR:$false, SPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;
