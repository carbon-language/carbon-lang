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

def SDT_FTOI :
SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisFP<1>]>;
def SDT_ITOF :
SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f32>]>;
def SDT_CMPFP0 :
SDTypeProfile<0, 1, [SDTCisFP<0>]>;
def SDT_FMDRR :
SDTypeProfile<1, 2, [SDTCisVT<0, f64>, SDTCisVT<1, i32>,
                     SDTCisSameAs<1, 2>]>;

def arm_ftoui  : SDNode<"ARMISD::FTOUI",  SDT_FTOI>;
def arm_ftosi  : SDNode<"ARMISD::FTOSI",  SDT_FTOI>;
def arm_sitof  : SDNode<"ARMISD::SITOF",  SDT_ITOF>;
def arm_uitof  : SDNode<"ARMISD::UITOF",  SDT_ITOF>;
def arm_fmstat : SDNode<"ARMISD::FMSTAT", SDTNone, [SDNPInFlag,SDNPOutFlag]>;
def arm_cmpfp  : SDNode<"ARMISD::CMPFP",  SDT_ARMCmp, [SDNPOutFlag]>;
def arm_cmpfp0 : SDNode<"ARMISD::CMPFPw0",SDT_CMPFP0, [SDNPOutFlag]>;
def arm_fmdrr  : SDNode<"ARMISD::FMDRR",  SDT_FMDRR>;

//===----------------------------------------------------------------------===//
// Operand Definitions.
//


def vfp_f32imm : Operand<f32>,
                 PatLeaf<(f32 fpimm), [{
      return ARM::getVFPf32Imm(N->getValueAPF()) != -1;
    }]> {
  let PrintMethod = "printVFPf32ImmOperand";
}

def vfp_f64imm : Operand<f64>,
                 PatLeaf<(f64 fpimm), [{
      return ARM::getVFPf64Imm(N->getValueAPF()) != -1;
    }]> {
  let PrintMethod = "printVFPf64ImmOperand";
}


//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

let canFoldAsLoad = 1 in {
def FLDD  : ADI5<0b1101, 0b01, (outs DPR:$dst), (ins addrmode5:$addr),
                 IIC_fpLoad64, "fldd", "\t$dst, $addr",
                 [(set DPR:$dst, (load addrmode5:$addr))]>;

def FLDS  : ASI5<0b1101, 0b01, (outs SPR:$dst), (ins addrmode5:$addr),
                 IIC_fpLoad32, "flds", "\t$dst, $addr",
                 [(set SPR:$dst, (load addrmode5:$addr))]>;
} // canFoldAsLoad

def FSTD  : ADI5<0b1101, 0b00, (outs), (ins DPR:$src, addrmode5:$addr),
                 IIC_fpStore64, "fstd", "\t$src, $addr",
                 [(store DPR:$src, addrmode5:$addr)]>;

def FSTS  : ASI5<0b1101, 0b00, (outs), (ins SPR:$src, addrmode5:$addr),
                 IIC_fpStore32, "fsts", "\t$src, $addr",
                 [(store SPR:$src, addrmode5:$addr)]>;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let mayLoad = 1, hasExtraDefRegAllocReq = 1 in {
def FLDMD : AXDI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$wb,
                           variable_ops), IIC_fpLoadm,
                  "fldm${addr:submode}d${p}\t${addr:base}, $wb",
                  []> {
  let Inst{20} = 1;
}

def FLDMS : AXSI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$wb,
                           variable_ops), IIC_fpLoadm, 
                  "fldm${addr:submode}s${p}\t${addr:base}, $wb",
                  []> {
  let Inst{20} = 1;
}
} // mayLoad, hasExtraDefRegAllocReq

let mayStore = 1, hasExtraSrcRegAllocReq = 1 in {
def FSTMD : AXDI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$wb,
                           variable_ops), IIC_fpStorem,
                 "fstm${addr:submode}d${p}\t${addr:base}, $wb",
                 []> {
  let Inst{20} = 0;
}

def FSTMS : AXSI5<(outs), (ins addrmode5:$addr, pred:$p, reglist:$wb,
                           variable_ops), IIC_fpStorem,
                 "fstm${addr:submode}s${p}\t${addr:base}, $wb",
                 []> {
  let Inst{20} = 0;
}
} // mayStore, hasExtraSrcRegAllocReq

// FLDMX, FSTMX - mixing S/D registers for pre-armv6 cores

//===----------------------------------------------------------------------===//
// FP Binary Operations.
//

def FADDD  : ADbI<0b11100011, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpALU64, "faddd", "\t$dst, $a, $b",
                 [(set DPR:$dst, (fadd DPR:$a, DPR:$b))]>;

def FADDS  : ASbIn<0b11100011, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpALU32, "fadds", "\t$dst, $a, $b",
                  [(set SPR:$dst, (fadd SPR:$a, SPR:$b))]>;

// These are encoded as unary instructions.
let Defs = [FPSCR] in {
def FCMPED : ADuI<0b11101011, 0b0100, 0b1100, (outs), (ins DPR:$a, DPR:$b),
                 IIC_fpCMP64, "fcmped", "\t$a, $b",
                 [(arm_cmpfp DPR:$a, DPR:$b)]>;

def FCMPES : ASuI<0b11101011, 0b0100, 0b1100, (outs), (ins SPR:$a, SPR:$b),
                 IIC_fpCMP32, "fcmpes", "\t$a, $b",
                 [(arm_cmpfp SPR:$a, SPR:$b)]>;
}

def FDIVD  : ADbI<0b11101000, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpDIV64, "fdivd", "\t$dst, $a, $b",
                 [(set DPR:$dst, (fdiv DPR:$a, DPR:$b))]>;

def FDIVS  : ASbI<0b11101000, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                 IIC_fpDIV32, "fdivs", "\t$dst, $a, $b",
                 [(set SPR:$dst, (fdiv SPR:$a, SPR:$b))]>;

def FMULD  : ADbI<0b11100010, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpMUL64, "fmuld", "\t$dst, $a, $b",
                 [(set DPR:$dst, (fmul DPR:$a, DPR:$b))]>;

def FMULS  : ASbIn<0b11100010, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpMUL32, "fmuls", "\t$dst, $a, $b",
                  [(set SPR:$dst, (fmul SPR:$a, SPR:$b))]>;
                 
def FNMULD  : ADbI<0b11100010, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                  IIC_fpMUL64, "fnmuld", "\t$dst, $a, $b",
                  [(set DPR:$dst, (fneg (fmul DPR:$a, DPR:$b)))]> {
  let Inst{6} = 1;
}

def FNMULS  : ASbI<0b11100010, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpMUL32, "fnmuls", "\t$dst, $a, $b",
                  [(set SPR:$dst, (fneg (fmul SPR:$a, SPR:$b)))]> {
  let Inst{6} = 1;
}

// Match reassociated forms only if not sign dependent rounding.
def : Pat<(fmul (fneg DPR:$a), DPR:$b),
          (FNMULD DPR:$a, DPR:$b)>, Requires<[NoHonorSignDependentRounding]>;
def : Pat<(fmul (fneg SPR:$a), SPR:$b),
          (FNMULS SPR:$a, SPR:$b)>, Requires<[NoHonorSignDependentRounding]>;


def FSUBD  : ADbI<0b11100011, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpALU64, "fsubd", "\t$dst, $a, $b",
                 [(set DPR:$dst, (fsub DPR:$a, DPR:$b))]> {
  let Inst{6} = 1;
}

def FSUBS  : ASbIn<0b11100011, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpALU32, "fsubs", "\t$dst, $a, $b",
                  [(set SPR:$dst, (fsub SPR:$a, SPR:$b))]> {
  let Inst{6} = 1;
}

//===----------------------------------------------------------------------===//
// FP Unary Operations.
//

def FABSD  : ADuI<0b11101011, 0b0000, 0b1100, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpUNA64, "fabsd", "\t$dst, $a",
                 [(set DPR:$dst, (fabs DPR:$a))]>;

def FABSS  : ASuIn<0b11101011, 0b0000, 0b1100, (outs SPR:$dst), (ins SPR:$a),
                  IIC_fpUNA32, "fabss", "\t$dst, $a",
                  [(set SPR:$dst, (fabs SPR:$a))]>;

let Defs = [FPSCR] in {
def FCMPEZD : ADuI<0b11101011, 0b0101, 0b1100, (outs), (ins DPR:$a),
                  IIC_fpCMP64, "fcmpezd", "\t$a",
                  [(arm_cmpfp0 DPR:$a)]>;

def FCMPEZS : ASuI<0b11101011, 0b0101, 0b1100, (outs), (ins SPR:$a),
                  IIC_fpCMP32, "fcmpezs", "\t$a",
                  [(arm_cmpfp0 SPR:$a)]>;
}

def FCVTDS : ASuI<0b11101011, 0b0111, 0b1100, (outs DPR:$dst), (ins SPR:$a),
                 IIC_fpCVTDS, "fcvtds", "\t$dst, $a",
                 [(set DPR:$dst, (fextend SPR:$a))]>;

// Special case encoding: bits 11-8 is 0b1011.
def FCVTSD : VFPAI<(outs SPR:$dst), (ins DPR:$a), VFPUnaryFrm,
                   IIC_fpCVTSD, "fcvtsd", "\t$dst, $a",
                   [(set SPR:$dst, (fround DPR:$a))]> {
  let Inst{27-23} = 0b11101;
  let Inst{21-16} = 0b110111;
  let Inst{11-8}  = 0b1011;
  let Inst{7-4}   = 0b1100;
}

let neverHasSideEffects = 1 in {
def FCPYD  : ADuI<0b11101011, 0b0000, 0b0100, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpUNA64, "fcpyd", "\t$dst, $a", []>;

def FCPYS  : ASuI<0b11101011, 0b0000, 0b0100, (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpUNA32, "fcpys", "\t$dst, $a", []>;
} // neverHasSideEffects

def FNEGD  : ADuI<0b11101011, 0b0001, 0b0100, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpUNA64, "fnegd", "\t$dst, $a",
                 [(set DPR:$dst, (fneg DPR:$a))]>;

def FNEGS  : ASuIn<0b11101011, 0b0001, 0b0100, (outs SPR:$dst), (ins SPR:$a),
                  IIC_fpUNA32, "fnegs", "\t$dst, $a",
                  [(set SPR:$dst, (fneg SPR:$a))]>;

def FSQRTD  : ADuI<0b11101011, 0b0001, 0b1100, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpSQRT64, "fsqrtd", "\t$dst, $a",
                 [(set DPR:$dst, (fsqrt DPR:$a))]>;

def FSQRTS  : ASuI<0b11101011, 0b0001, 0b1100, (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpSQRT32, "fsqrts", "\t$dst, $a",
                 [(set SPR:$dst, (fsqrt SPR:$a))]>;

//===----------------------------------------------------------------------===//
// FP <-> GPR Copies.  Int <-> FP Conversions.
//

def FMRS   : AVConv2I<0b11100001, 0b1010, (outs GPR:$dst), (ins SPR:$src),
                 IIC_VMOVSI, "fmrs", "\t$dst, $src",
                 [(set GPR:$dst, (bitconvert SPR:$src))]>;

def FMSR   : AVConv4I<0b11100000, 0b1010, (outs SPR:$dst), (ins GPR:$src),
                 IIC_VMOVIS, "fmsr", "\t$dst, $src",
                 [(set SPR:$dst, (bitconvert GPR:$src))]>;

def FMRRD  : AVConv3I<0b11000101, 0b1011,
                      (outs GPR:$wb, GPR:$dst2), (ins DPR:$src),
                 IIC_VMOVDI, "fmrrd", "\t$wb, $dst2, $src",
                 [/* FIXME: Can't write pattern for multiple result instr*/]>;

// FMDHR: GPR -> SPR
// FMDLR: GPR -> SPR

def FMDRR : AVConv5I<0b11000100, 0b1011,
                     (outs DPR:$dst), (ins GPR:$src1, GPR:$src2),
                IIC_VMOVID, "fmdrr", "\t$dst, $src1, $src2",
                [(set DPR:$dst, (arm_fmdrr GPR:$src1, GPR:$src2))]>;

// FMRDH: SPR -> GPR
// FMRDL: SPR -> GPR
// FMRRS: SPR -> GPR
// FMRX : SPR system reg -> GPR

// FMSRR: GPR -> SPR

// FMXR: GPR -> VFP Sstem reg


// Int to FP:

def FSITOD : AVConv1I<0b11101011, 0b1000, 0b1011, (outs DPR:$dst), (ins SPR:$a),
                 IIC_fpCVTID, "fsitod", "\t$dst, $a",
                 [(set DPR:$dst, (arm_sitof SPR:$a))]> {
  let Inst{7} = 1;
}

def FSITOS : AVConv1In<0b11101011, 0b1000, 0b1010, (outs SPR:$dst),(ins SPR:$a),
                 IIC_fpCVTIS, "fsitos", "\t$dst, $a",
                 [(set SPR:$dst, (arm_sitof SPR:$a))]> {
  let Inst{7} = 1;
}

def FUITOD : AVConv1I<0b11101011, 0b1000, 0b1011, (outs DPR:$dst), (ins SPR:$a),
                 IIC_fpCVTID, "fuitod", "\t$dst, $a",
                 [(set DPR:$dst, (arm_uitof SPR:$a))]>;

def FUITOS : AVConv1In<0b11101011, 0b1000, 0b1010, (outs SPR:$dst),(ins SPR:$a),
                 IIC_fpCVTIS, "fuitos", "\t$dst, $a",
                 [(set SPR:$dst, (arm_uitof SPR:$a))]>;

// FP to Int:
// Always set Z bit in the instruction, i.e. "round towards zero" variants.

def FTOSIZD : AVConv1I<0b11101011, 0b1101, 0b1011,
                       (outs SPR:$dst), (ins DPR:$a),
                 IIC_fpCVTDI, "ftosizd", "\t$dst, $a",
                 [(set SPR:$dst, (arm_ftosi DPR:$a))]> {
  let Inst{7} = 1; // Z bit
}

def FTOSIZS : AVConv1In<0b11101011, 0b1101, 0b1010,
                        (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTSI, "ftosizs", "\t$dst, $a",
                 [(set SPR:$dst, (arm_ftosi SPR:$a))]> {
  let Inst{7} = 1; // Z bit
}

def FTOUIZD : AVConv1I<0b11101011, 0b1100, 0b1011,
                       (outs SPR:$dst), (ins DPR:$a),
                 IIC_fpCVTDI, "ftouizd", "\t$dst, $a",
                 [(set SPR:$dst, (arm_ftoui DPR:$a))]> {
  let Inst{7} = 1; // Z bit
}

def FTOUIZS : AVConv1In<0b11101011, 0b1100, 0b1010,
                        (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTSI, "ftouizs", "\t$dst, $a",
                 [(set SPR:$dst, (arm_ftoui SPR:$a))]> {
  let Inst{7} = 1; // Z bit
}

//===----------------------------------------------------------------------===//
// FP FMA Operations.
//

def FMACD : ADbI<0b11100000, (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                IIC_fpMAC64, "fmacd", "\t$dst, $a, $b",
                [(set DPR:$dst, (fadd (fmul DPR:$a, DPR:$b), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMACS : ASbIn<0b11100000, (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                 IIC_fpMAC32, "fmacs", "\t$dst, $a, $b",
                 [(set SPR:$dst, (fadd (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                 RegConstraint<"$dstin = $dst">;

def FMSCD : ADbI<0b11100001, (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                IIC_fpMAC64, "fmscd", "\t$dst, $a, $b",
                [(set DPR:$dst, (fsub (fmul DPR:$a, DPR:$b), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FMSCS : ASbI<0b11100001, (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                IIC_fpMAC32, "fmscs", "\t$dst, $a, $b",
                [(set SPR:$dst, (fsub (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def FNMACD : ADbI<0b11100000, (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                 IIC_fpMAC64, "fnmacd", "\t$dst, $a, $b",
             [(set DPR:$dst, (fadd (fneg (fmul DPR:$a, DPR:$b)), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst"> {
  let Inst{6} = 1;
}

def FNMACS : ASbIn<0b11100000, (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                  IIC_fpMAC32, "fnmacs", "\t$dst, $a, $b",
             [(set SPR:$dst, (fadd (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst"> {
  let Inst{6} = 1;
}

def : Pat<(fsub DPR:$dstin, (fmul DPR:$a, DPR:$b)),
          (FNMACD DPR:$dstin, DPR:$a, DPR:$b)>, Requires<[DontUseNEONForFP]>;
def : Pat<(fsub SPR:$dstin, (fmul SPR:$a, SPR:$b)),
          (FNMACS SPR:$dstin, SPR:$a, SPR:$b)>, Requires<[DontUseNEONForFP]>;

def FNMSCD : ADbI<0b11100001, (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                 IIC_fpMAC64, "fnmscd", "\t$dst, $a, $b",
             [(set DPR:$dst, (fsub (fneg (fmul DPR:$a, DPR:$b)), DPR:$dstin))]>,
                RegConstraint<"$dstin = $dst"> {
  let Inst{6} = 1;
}

def FNMSCS : ASbI<0b11100001, (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                IIC_fpMAC32, "fnmscs", "\t$dst, $a, $b",
             [(set SPR:$dst, (fsub (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst"> {
  let Inst{6} = 1;
}

//===----------------------------------------------------------------------===//
// FP Conditional moves.
//

def FCPYDcc  : ADuI<0b11101011, 0b0000, 0b0100,
                    (outs DPR:$dst), (ins DPR:$false, DPR:$true),
                    IIC_fpUNA64, "fcpyd", "\t$dst, $true",
                [/*(set DPR:$dst, (ARMcmov DPR:$false, DPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def FCPYScc  : ASuI<0b11101011, 0b0000, 0b0100,
                    (outs SPR:$dst), (ins SPR:$false, SPR:$true),
                    IIC_fpUNA32, "fcpys", "\t$dst, $true",
                [/*(set SPR:$dst, (ARMcmov SPR:$false, SPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def FNEGDcc  : ADuI<0b11101011, 0b0001, 0b0100,
                    (outs DPR:$dst), (ins DPR:$false, DPR:$true),
                    IIC_fpUNA64, "fnegd", "\t$dst, $true",
                [/*(set DPR:$dst, (ARMcneg DPR:$false, DPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def FNEGScc  : ASuI<0b11101011, 0b0001, 0b0100,
                    (outs SPR:$dst), (ins SPR:$false, SPR:$true),
                    IIC_fpUNA32, "fnegs", "\t$dst, $true",
                [/*(set SPR:$dst, (ARMcneg SPR:$false, SPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;


//===----------------------------------------------------------------------===//
// Misc.
//

let Defs = [CPSR], Uses = [FPSCR] in
def FMSTAT : VFPAI<(outs), (ins), VFPMiscFrm, IIC_fpSTAT, "fmstat", "",
             [(arm_fmstat)]> {
  let Inst{27-20} = 0b11101111;
  let Inst{19-16} = 0b0001;
  let Inst{15-12} = 0b1111;
  let Inst{11-8}  = 0b1010;
  let Inst{7}     = 0;
  let Inst{4}     = 1;
}


// Materialize FP immediates. VFP3 only.
let isReMaterializable = 1 in 
def FCONSTS : VFPAI<(outs SPR:$dst), (ins vfp_f32imm:$imm),
                    VFPMiscFrm, IIC_VMOVImm,
                    "fconsts", "\t$dst, $imm",
                    [(set SPR:$dst, vfp_f32imm:$imm)]>, Requires<[HasVFP3]> {
  let Inst{27-23} = 0b11101;
  let Inst{21-20} = 0b11;
  let Inst{11-9}  = 0b101;
  let Inst{8}     = 0;
  let Inst{7-4}   = 0b0000;
}

let isReMaterializable = 1 in 
def FCONSTD : VFPAI<(outs DPR:$dst), (ins vfp_f64imm:$imm),
                    VFPMiscFrm, IIC_VMOVImm,
                    "fconstd", "\t$dst, $imm",
                    [(set DPR:$dst, vfp_f64imm:$imm)]>, Requires<[HasVFP3]> {
  let Inst{27-23} = 0b11101;
  let Inst{21-20} = 0b11;
  let Inst{11-9}  = 0b101;
  let Inst{8}     = 1;
  let Inst{7-4}   = 0b0000;
}
