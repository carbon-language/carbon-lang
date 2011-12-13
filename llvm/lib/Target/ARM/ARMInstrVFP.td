//===- ARMInstrVFP.td - VFP support for ARM ----------------*- tablegen -*-===//
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

def SDT_FTOI    : SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisFP<1>]>;
def SDT_ITOF    : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f32>]>;
def SDT_CMPFP0  : SDTypeProfile<0, 1, [SDTCisFP<0>]>;
def SDT_VMOVDRR : SDTypeProfile<1, 2, [SDTCisVT<0, f64>, SDTCisVT<1, i32>,
                                       SDTCisSameAs<1, 2>]>;

def arm_ftoui  : SDNode<"ARMISD::FTOUI",   SDT_FTOI>;
def arm_ftosi  : SDNode<"ARMISD::FTOSI",   SDT_FTOI>;
def arm_sitof  : SDNode<"ARMISD::SITOF",   SDT_ITOF>;
def arm_uitof  : SDNode<"ARMISD::UITOF",   SDT_ITOF>;
def arm_fmstat : SDNode<"ARMISD::FMSTAT",  SDTNone, [SDNPInGlue, SDNPOutGlue]>;
def arm_cmpfp  : SDNode<"ARMISD::CMPFP",   SDT_ARMCmp, [SDNPOutGlue]>;
def arm_cmpfp0 : SDNode<"ARMISD::CMPFPw0", SDT_CMPFP0, [SDNPOutGlue]>;
def arm_fmdrr  : SDNode<"ARMISD::VMOVDRR", SDT_VMOVDRR>;


//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// 8-bit floating-point immediate encodings.
def FPImmOperand : AsmOperandClass {
  let Name = "FPImm";
  let ParserMethod = "parseFPImm";
}

def vfp_f32imm : Operand<f32>,
                 PatLeaf<(f32 fpimm), [{
      return ARM_AM::getFP32Imm(N->getValueAPF()) != -1;
    }], SDNodeXForm<fpimm, [{
      APFloat InVal = N->getValueAPF();
      uint32_t enc = ARM_AM::getFP32Imm(InVal);
      return CurDAG->getTargetConstant(enc, MVT::i32);
    }]>> {
  let PrintMethod = "printFPImmOperand";
  let ParserMatchClass = FPImmOperand;
}

def vfp_f64imm : Operand<f64>,
                 PatLeaf<(f64 fpimm), [{
      return ARM_AM::getFP64Imm(N->getValueAPF()) != -1;
    }], SDNodeXForm<fpimm, [{
      APFloat InVal = N->getValueAPF();
      uint32_t enc = ARM_AM::getFP64Imm(InVal);
      return CurDAG->getTargetConstant(enc, MVT::i32);
    }]>> {
  let PrintMethod = "printFPImmOperand";
  let ParserMatchClass = FPImmOperand;
}


//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

let canFoldAsLoad = 1, isReMaterializable = 1 in {

def VLDRD : ADI5<0b1101, 0b01, (outs DPR:$Dd), (ins addrmode5:$addr),
                 IIC_fpLoad64, "vldr", "\t$Dd, $addr",
                 [(set DPR:$Dd, (f64 (load addrmode5:$addr)))]>;

def VLDRS : ASI5<0b1101, 0b01, (outs SPR:$Sd), (ins addrmode5:$addr),
                 IIC_fpLoad32, "vldr", "\t$Sd, $addr",
                 [(set SPR:$Sd, (load addrmode5:$addr))]> {
  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
}

} // End of 'let canFoldAsLoad = 1, isReMaterializable = 1 in'

def VSTRD : ADI5<0b1101, 0b00, (outs), (ins DPR:$Dd, addrmode5:$addr),
                 IIC_fpStore64, "vstr", "\t$Dd, $addr",
                 [(store (f64 DPR:$Dd), addrmode5:$addr)]>;

def VSTRS : ASI5<0b1101, 0b00, (outs), (ins SPR:$Sd, addrmode5:$addr),
                 IIC_fpStore32, "vstr", "\t$Sd, $addr",
                 [(store SPR:$Sd, addrmode5:$addr)]> {
  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
}

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

multiclass vfp_ldst_mult<string asm, bit L_bit,
                         InstrItinClass itin, InstrItinClass itin_upd> {
  // Double Precision
  def DIA :
    AXDI4<(outs), (ins GPR:$Rn, pred:$p, dpr_reglist:$regs, variable_ops),
          IndexModeNone, itin,
          !strconcat(asm, "ia${p}\t$Rn, $regs"), "", []> {
    let Inst{24-23} = 0b01;       // Increment After
    let Inst{21}    = 0;          // No writeback
    let Inst{20}    = L_bit;
  }
  def DIA_UPD :
    AXDI4<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, dpr_reglist:$regs,
                               variable_ops),
          IndexModeUpd, itin_upd,
          !strconcat(asm, "ia${p}\t$Rn!, $regs"), "$Rn = $wb", []> {
    let Inst{24-23} = 0b01;       // Increment After
    let Inst{21}    = 1;          // Writeback
    let Inst{20}    = L_bit;
  }
  def DDB_UPD :
    AXDI4<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, dpr_reglist:$regs,
                               variable_ops),
          IndexModeUpd, itin_upd,
          !strconcat(asm, "db${p}\t$Rn!, $regs"), "$Rn = $wb", []> {
    let Inst{24-23} = 0b10;       // Decrement Before
    let Inst{21}    = 1;          // Writeback
    let Inst{20}    = L_bit;
  }

  // Single Precision
  def SIA :
    AXSI4<(outs), (ins GPR:$Rn, pred:$p, spr_reglist:$regs, variable_ops),
          IndexModeNone, itin,
          !strconcat(asm, "ia${p}\t$Rn, $regs"), "", []> {
    let Inst{24-23} = 0b01;       // Increment After
    let Inst{21}    = 0;          // No writeback
    let Inst{20}    = L_bit;

    // Some single precision VFP instructions may be executed on both NEON and
    // VFP pipelines.
    let D = VFPNeonDomain;
  }
  def SIA_UPD :
    AXSI4<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, spr_reglist:$regs,
                               variable_ops),
          IndexModeUpd, itin_upd,
          !strconcat(asm, "ia${p}\t$Rn!, $regs"), "$Rn = $wb", []> {
    let Inst{24-23} = 0b01;       // Increment After
    let Inst{21}    = 1;          // Writeback
    let Inst{20}    = L_bit;

    // Some single precision VFP instructions may be executed on both NEON and
    // VFP pipelines.
    let D = VFPNeonDomain;
  }
  def SDB_UPD :
    AXSI4<(outs GPR:$wb), (ins GPR:$Rn, pred:$p, spr_reglist:$regs,
                               variable_ops),
          IndexModeUpd, itin_upd,
          !strconcat(asm, "db${p}\t$Rn!, $regs"), "$Rn = $wb", []> {
    let Inst{24-23} = 0b10;       // Decrement Before
    let Inst{21}    = 1;          // Writeback
    let Inst{20}    = L_bit;

    // Some single precision VFP instructions may be executed on both NEON and
    // VFP pipelines.
    let D = VFPNeonDomain;
  }
}

let neverHasSideEffects = 1 in {

let mayLoad = 1, hasExtraDefRegAllocReq = 1 in
defm VLDM : vfp_ldst_mult<"vldm", 1, IIC_fpLoad_m, IIC_fpLoad_mu>;

let mayStore = 1, hasExtraSrcRegAllocReq = 1 in
defm VSTM : vfp_ldst_mult<"vstm", 0, IIC_fpLoad_m, IIC_fpLoad_mu>;

} // neverHasSideEffects

def : MnemonicAlias<"vldm", "vldmia">;
def : MnemonicAlias<"vstm", "vstmia">;

def : InstAlias<"vpush${p} $r", (VSTMDDB_UPD SP, pred:$p, dpr_reglist:$r)>,
                Requires<[HasVFP2]>;
def : InstAlias<"vpush${p} $r", (VSTMSDB_UPD SP, pred:$p, spr_reglist:$r)>,
                Requires<[HasVFP2]>;
def : InstAlias<"vpop${p} $r",  (VLDMDIA_UPD SP, pred:$p, dpr_reglist:$r)>,
                Requires<[HasVFP2]>;
def : InstAlias<"vpop${p} $r",  (VLDMSIA_UPD SP, pred:$p, spr_reglist:$r)>,
                Requires<[HasVFP2]>;

// FLDMX, FSTMX - mixing S/D registers for pre-armv6 cores

//===----------------------------------------------------------------------===//
// FP Binary Operations.
//

def VADDD  : ADbI<0b11100, 0b11, 0, 0,
                  (outs DPR:$Dd), (ins DPR:$Dn, DPR:$Dm),
                  IIC_fpALU64, "vadd", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd, (fadd DPR:$Dn, (f64 DPR:$Dm)))]>;

def VADDS  : ASbIn<0b11100, 0b11, 0, 0,
                   (outs SPR:$Sd), (ins SPR:$Sn, SPR:$Sm),
                   IIC_fpALU32, "vadd", ".f32\t$Sd, $Sn, $Sm",
                   [(set SPR:$Sd, (fadd SPR:$Sn, SPR:$Sm))]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VSUBD  : ADbI<0b11100, 0b11, 1, 0,
                  (outs DPR:$Dd), (ins DPR:$Dn, DPR:$Dm),
                  IIC_fpALU64, "vsub", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd, (fsub DPR:$Dn, (f64 DPR:$Dm)))]>;

def VSUBS  : ASbIn<0b11100, 0b11, 1, 0,
                   (outs SPR:$Sd), (ins SPR:$Sn, SPR:$Sm),
                   IIC_fpALU32, "vsub", ".f32\t$Sd, $Sn, $Sm",
                   [(set SPR:$Sd, (fsub SPR:$Sn, SPR:$Sm))]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VDIVD  : ADbI<0b11101, 0b00, 0, 0,
                  (outs DPR:$Dd), (ins DPR:$Dn, DPR:$Dm),
                  IIC_fpDIV64, "vdiv", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd, (fdiv DPR:$Dn, (f64 DPR:$Dm)))]>;

def VDIVS  : ASbI<0b11101, 0b00, 0, 0,
                  (outs SPR:$Sd), (ins SPR:$Sn, SPR:$Sm),
                  IIC_fpDIV32, "vdiv", ".f32\t$Sd, $Sn, $Sm",
                  [(set SPR:$Sd, (fdiv SPR:$Sn, SPR:$Sm))]>;

def VMULD  : ADbI<0b11100, 0b10, 0, 0,
                  (outs DPR:$Dd), (ins DPR:$Dn, DPR:$Dm),
                  IIC_fpMUL64, "vmul", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd, (fmul DPR:$Dn, (f64 DPR:$Dm)))]>;

def VMULS  : ASbIn<0b11100, 0b10, 0, 0,
                   (outs SPR:$Sd), (ins SPR:$Sn, SPR:$Sm),
                   IIC_fpMUL32, "vmul", ".f32\t$Sd, $Sn, $Sm",
                   [(set SPR:$Sd, (fmul SPR:$Sn, SPR:$Sm))]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VNMULD : ADbI<0b11100, 0b10, 1, 0,
                  (outs DPR:$Dd), (ins DPR:$Dn, DPR:$Dm),
                  IIC_fpMUL64, "vnmul", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd, (fneg (fmul DPR:$Dn, (f64 DPR:$Dm))))]>;

def VNMULS : ASbI<0b11100, 0b10, 1, 0,
                  (outs SPR:$Sd), (ins SPR:$Sn, SPR:$Sm),
                  IIC_fpMUL32, "vnmul", ".f32\t$Sd, $Sn, $Sm",
                  [(set SPR:$Sd, (fneg (fmul SPR:$Sn, SPR:$Sm)))]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

// Match reassociated forms only if not sign dependent rounding.
def : Pat<(fmul (fneg DPR:$a), (f64 DPR:$b)),
          (VNMULD DPR:$a, DPR:$b)>, Requires<[NoHonorSignDependentRounding]>;
def : Pat<(fmul (fneg SPR:$a), SPR:$b),
          (VNMULS SPR:$a, SPR:$b)>, Requires<[NoHonorSignDependentRounding]>;

// These are encoded as unary instructions.
let Defs = [FPSCR] in {
def VCMPED : ADuI<0b11101, 0b11, 0b0100, 0b11, 0,
                  (outs), (ins DPR:$Dd, DPR:$Dm),
                  IIC_fpCMP64, "vcmpe", ".f64\t$Dd, $Dm",
                  [(arm_cmpfp DPR:$Dd, (f64 DPR:$Dm))]>;

def VCMPES : ASuI<0b11101, 0b11, 0b0100, 0b11, 0,
                  (outs), (ins SPR:$Sd, SPR:$Sm),
                  IIC_fpCMP32, "vcmpe", ".f32\t$Sd, $Sm",
                  [(arm_cmpfp SPR:$Sd, SPR:$Sm)]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

// FIXME: Verify encoding after integrated assembler is working.
def VCMPD  : ADuI<0b11101, 0b11, 0b0100, 0b01, 0,
                  (outs), (ins DPR:$Dd, DPR:$Dm),
                  IIC_fpCMP64, "vcmp", ".f64\t$Dd, $Dm",
                  [/* For disassembly only; pattern left blank */]>;

def VCMPS  : ASuI<0b11101, 0b11, 0b0100, 0b01, 0,
                  (outs), (ins SPR:$Sd, SPR:$Sm),
                  IIC_fpCMP32, "vcmp", ".f32\t$Sd, $Sm",
                  [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}
} // Defs = [FPSCR]

//===----------------------------------------------------------------------===//
// FP Unary Operations.
//

def VABSD  : ADuI<0b11101, 0b11, 0b0000, 0b11, 0,
                  (outs DPR:$Dd), (ins DPR:$Dm),
                  IIC_fpUNA64, "vabs", ".f64\t$Dd, $Dm",
                  [(set DPR:$Dd, (fabs (f64 DPR:$Dm)))]>;

def VABSS  : ASuIn<0b11101, 0b11, 0b0000, 0b11, 0,
                   (outs SPR:$Sd), (ins SPR:$Sm),
                   IIC_fpUNA32, "vabs", ".f32\t$Sd, $Sm",
                   [(set SPR:$Sd, (fabs SPR:$Sm))]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

let Defs = [FPSCR] in {
def VCMPEZD : ADuI<0b11101, 0b11, 0b0101, 0b11, 0,
                   (outs), (ins DPR:$Dd),
                   IIC_fpCMP64, "vcmpe", ".f64\t$Dd, #0",
                   [(arm_cmpfp0 (f64 DPR:$Dd))]> {
  let Inst{3-0} = 0b0000;
  let Inst{5}   = 0;
}

def VCMPEZS : ASuI<0b11101, 0b11, 0b0101, 0b11, 0,
                   (outs), (ins SPR:$Sd),
                   IIC_fpCMP32, "vcmpe", ".f32\t$Sd, #0",
                   [(arm_cmpfp0 SPR:$Sd)]> {
  let Inst{3-0} = 0b0000;
  let Inst{5}   = 0;

  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

// FIXME: Verify encoding after integrated assembler is working.
def VCMPZD  : ADuI<0b11101, 0b11, 0b0101, 0b01, 0,
                   (outs), (ins DPR:$Dd),
                   IIC_fpCMP64, "vcmp", ".f64\t$Dd, #0",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{3-0} = 0b0000;
  let Inst{5}   = 0;
}

def VCMPZS  : ASuI<0b11101, 0b11, 0b0101, 0b01, 0,
                   (outs), (ins SPR:$Sd),
                   IIC_fpCMP32, "vcmp", ".f32\t$Sd, #0",
                   [/* For disassembly only; pattern left blank */]> {
  let Inst{3-0} = 0b0000;
  let Inst{5}   = 0;

  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}
} // Defs = [FPSCR]

def VCVTDS  : ASuI<0b11101, 0b11, 0b0111, 0b11, 0,
                   (outs DPR:$Dd), (ins SPR:$Sm),
                   IIC_fpCVTDS, "vcvt", ".f64.f32\t$Dd, $Sm",
                   [(set DPR:$Dd, (fextend SPR:$Sm))]> {
  // Instruction operands.
  bits<5> Dd;
  bits<5> Sm;

  // Encode instruction operands.
  let Inst{3-0}   = Sm{4-1};
  let Inst{5}     = Sm{0};
  let Inst{15-12} = Dd{3-0};
  let Inst{22}    = Dd{4};
}

// Special case encoding: bits 11-8 is 0b1011.
def VCVTSD  : VFPAI<(outs SPR:$Sd), (ins DPR:$Dm), VFPUnaryFrm,
                    IIC_fpCVTSD, "vcvt", ".f32.f64\t$Sd, $Dm",
                    [(set SPR:$Sd, (fround DPR:$Dm))]> {
  // Instruction operands.
  bits<5> Sd;
  bits<5> Dm;

  // Encode instruction operands.
  let Inst{3-0}   = Dm{3-0};
  let Inst{5}     = Dm{4};
  let Inst{15-12} = Sd{4-1};
  let Inst{22}    = Sd{0};

  let Inst{27-23} = 0b11101;
  let Inst{21-16} = 0b110111;
  let Inst{11-8}  = 0b1011;
  let Inst{7-6}   = 0b11;
  let Inst{4}     = 0;
}

// Between half-precision and single-precision.  For disassembly only.

// FIXME: Verify encoding after integrated assembler is working.
def VCVTBSH: ASuI<0b11101, 0b11, 0b0010, 0b01, 0, (outs SPR:$Sd), (ins SPR:$Sm),
                 /* FIXME */ IIC_fpCVTSH, "vcvtb", ".f32.f16\t$Sd, $Sm",
                 [/* For disassembly only; pattern left blank */]>;

def : ARMPat<(f32_to_f16 SPR:$a),
             (i32 (COPY_TO_REGCLASS (VCVTBSH SPR:$a), GPR))>;

def VCVTBHS: ASuI<0b11101, 0b11, 0b0011, 0b01, 0, (outs SPR:$Sd), (ins SPR:$Sm),
                 /* FIXME */ IIC_fpCVTHS, "vcvtb", ".f16.f32\t$Sd, $Sm",
                 [/* For disassembly only; pattern left blank */]>;

def : ARMPat<(f16_to_f32 GPR:$a),
             (VCVTBHS (COPY_TO_REGCLASS GPR:$a, SPR))>;

def VCVTTSH: ASuI<0b11101, 0b11, 0b0010, 0b11, 0, (outs SPR:$Sd), (ins SPR:$Sm),
                 /* FIXME */ IIC_fpCVTSH, "vcvtt", ".f32.f16\t$Sd, $Sm",
                 [/* For disassembly only; pattern left blank */]>;

def VCVTTHS: ASuI<0b11101, 0b11, 0b0011, 0b11, 0, (outs SPR:$Sd), (ins SPR:$Sm),
                 /* FIXME */ IIC_fpCVTHS, "vcvtt", ".f16.f32\t$Sd, $Sm",
                 [/* For disassembly only; pattern left blank */]>;

def VNEGD  : ADuI<0b11101, 0b11, 0b0001, 0b01, 0,
                  (outs DPR:$Dd), (ins DPR:$Dm),
                  IIC_fpUNA64, "vneg", ".f64\t$Dd, $Dm",
                  [(set DPR:$Dd, (fneg (f64 DPR:$Dm)))]>;

def VNEGS  : ASuIn<0b11101, 0b11, 0b0001, 0b01, 0,
                   (outs SPR:$Sd), (ins SPR:$Sm),
                   IIC_fpUNA32, "vneg", ".f32\t$Sd, $Sm",
                   [(set SPR:$Sd, (fneg SPR:$Sm))]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VSQRTD : ADuI<0b11101, 0b11, 0b0001, 0b11, 0,
                  (outs DPR:$Dd), (ins DPR:$Dm),
                  IIC_fpSQRT64, "vsqrt", ".f64\t$Dd, $Dm",
                  [(set DPR:$Dd, (fsqrt (f64 DPR:$Dm)))]>;

def VSQRTS : ASuI<0b11101, 0b11, 0b0001, 0b11, 0,
                  (outs SPR:$Sd), (ins SPR:$Sm),
                  IIC_fpSQRT32, "vsqrt", ".f32\t$Sd, $Sm",
                  [(set SPR:$Sd, (fsqrt SPR:$Sm))]>;

let neverHasSideEffects = 1 in {
def VMOVD  : ADuI<0b11101, 0b11, 0b0000, 0b01, 0,
                  (outs DPR:$Dd), (ins DPR:$Dm),
                  IIC_fpUNA64, "vmov", ".f64\t$Dd, $Dm", []>;

def VMOVS  : ASuI<0b11101, 0b11, 0b0000, 0b01, 0,
                  (outs SPR:$Sd), (ins SPR:$Sm),
                  IIC_fpUNA32, "vmov", ".f32\t$Sd, $Sm", []>;
} // neverHasSideEffects

//===----------------------------------------------------------------------===//
// FP <-> GPR Copies.  Int <-> FP Conversions.
//

def VMOVRS : AVConv2I<0b11100001, 0b1010,
                      (outs GPR:$Rt), (ins SPR:$Sn),
                      IIC_fpMOVSI, "vmov", "\t$Rt, $Sn",
                      [(set GPR:$Rt, (bitconvert SPR:$Sn))]> {
  // Instruction operands.
  bits<4> Rt;
  bits<5> Sn;

  // Encode instruction operands.
  let Inst{19-16} = Sn{4-1};
  let Inst{7}     = Sn{0};
  let Inst{15-12} = Rt;

  let Inst{6-5}   = 0b00;
  let Inst{3-0}   = 0b0000;

  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
}

def VMOVSR : AVConv4I<0b11100000, 0b1010,
                      (outs SPR:$Sn), (ins GPR:$Rt),
                      IIC_fpMOVIS, "vmov", "\t$Sn, $Rt",
                      [(set SPR:$Sn, (bitconvert GPR:$Rt))]> {
  // Instruction operands.
  bits<5> Sn;
  bits<4> Rt;

  // Encode instruction operands.
  let Inst{19-16} = Sn{4-1};
  let Inst{7}     = Sn{0};
  let Inst{15-12} = Rt;

  let Inst{6-5}   = 0b00;
  let Inst{3-0}   = 0b0000;

  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
}

let neverHasSideEffects = 1 in {
def VMOVRRD  : AVConv3I<0b11000101, 0b1011,
                        (outs GPR:$Rt, GPR:$Rt2), (ins DPR:$Dm),
                        IIC_fpMOVDI, "vmov", "\t$Rt, $Rt2, $Dm",
                 [/* FIXME: Can't write pattern for multiple result instr*/]> {
  // Instruction operands.
  bits<5> Dm;
  bits<4> Rt;
  bits<4> Rt2;

  // Encode instruction operands.
  let Inst{3-0}   = Dm{3-0};
  let Inst{5}     = Dm{4};
  let Inst{15-12} = Rt;
  let Inst{19-16} = Rt2;

  let Inst{7-6} = 0b00;

  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
}

def VMOVRRS  : AVConv3I<0b11000101, 0b1010,
                      (outs GPR:$Rt, GPR:$Rt2), (ins SPR:$src1, SPR:$src2),
                 IIC_fpMOVDI, "vmov", "\t$Rt, $Rt2, $src1, $src2",
                 [/* For disassembly only; pattern left blank */]> {
  bits<5> src1;
  bits<4> Rt;
  bits<4> Rt2;

  // Encode instruction operands.
  let Inst{3-0}   = src1{3-0};
  let Inst{5}     = src1{4};
  let Inst{15-12} = Rt;
  let Inst{19-16} = Rt2;

  let Inst{7-6} = 0b00;

  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
  let DecoderMethod = "DecodeVMOVRRS";
}
} // neverHasSideEffects

// FMDHR: GPR -> SPR
// FMDLR: GPR -> SPR

def VMOVDRR : AVConv5I<0b11000100, 0b1011,
                      (outs DPR:$Dm), (ins GPR:$Rt, GPR:$Rt2),
                      IIC_fpMOVID, "vmov", "\t$Dm, $Rt, $Rt2",
                      [(set DPR:$Dm, (arm_fmdrr GPR:$Rt, GPR:$Rt2))]> {
  // Instruction operands.
  bits<5> Dm;
  bits<4> Rt;
  bits<4> Rt2;

  // Encode instruction operands.
  let Inst{3-0}   = Dm{3-0};
  let Inst{5}     = Dm{4};
  let Inst{15-12} = Rt;
  let Inst{19-16} = Rt2;

  let Inst{7-6}   = 0b00;

  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;
}

let neverHasSideEffects = 1 in
def VMOVSRR : AVConv5I<0b11000100, 0b1010,
                     (outs SPR:$dst1, SPR:$dst2), (ins GPR:$src1, GPR:$src2),
                IIC_fpMOVID, "vmov", "\t$dst1, $dst2, $src1, $src2",
                [/* For disassembly only; pattern left blank */]> {
  // Instruction operands.
  bits<5> dst1;
  bits<4> src1;
  bits<4> src2;

  // Encode instruction operands.
  let Inst{3-0}   = dst1{3-0};
  let Inst{5}     = dst1{4};
  let Inst{15-12} = src1;
  let Inst{19-16} = src2;

  let Inst{7-6} = 0b00;

  // Some single precision VFP instructions may be executed on both NEON and VFP
  // pipelines.
  let D = VFPNeonDomain;

  let DecoderMethod = "DecodeVMOVSRR";
}

// FMRDH: SPR -> GPR
// FMRDL: SPR -> GPR
// FMRRS: SPR -> GPR
// FMRX:  SPR system reg -> GPR
// FMSRR: GPR -> SPR
// FMXR:  GPR -> VFP system reg


// Int -> FP:

class AVConv1IDs_Encode<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3,
                        bits<4> opcod4, dag oops, dag iops,
                        InstrItinClass itin, string opc, string asm,
                        list<dag> pattern>
  : AVConv1I<opcod1, opcod2, opcod3, opcod4, oops, iops, itin, opc, asm,
             pattern> {
  // Instruction operands.
  bits<5> Dd;
  bits<5> Sm;

  // Encode instruction operands.
  let Inst{3-0}   = Sm{4-1};
  let Inst{5}     = Sm{0};
  let Inst{15-12} = Dd{3-0};
  let Inst{22}    = Dd{4};
}

class AVConv1InSs_Encode<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3,
                         bits<4> opcod4, dag oops, dag iops,InstrItinClass itin,
                         string opc, string asm, list<dag> pattern>
  : AVConv1In<opcod1, opcod2, opcod3, opcod4, oops, iops, itin, opc, asm,
              pattern> {
  // Instruction operands.
  bits<5> Sd;
  bits<5> Sm;

  // Encode instruction operands.
  let Inst{3-0}   = Sm{4-1};
  let Inst{5}     = Sm{0};
  let Inst{15-12} = Sd{4-1};
  let Inst{22}    = Sd{0};
}

def VSITOD : AVConv1IDs_Encode<0b11101, 0b11, 0b1000, 0b1011,
                               (outs DPR:$Dd), (ins SPR:$Sm),
                               IIC_fpCVTID, "vcvt", ".f64.s32\t$Dd, $Sm",
                               [(set DPR:$Dd, (f64 (arm_sitof SPR:$Sm)))]> {
  let Inst{7} = 1; // s32
}

def VSITOS : AVConv1InSs_Encode<0b11101, 0b11, 0b1000, 0b1010,
                                (outs SPR:$Sd),(ins SPR:$Sm),
                                IIC_fpCVTIS, "vcvt", ".f32.s32\t$Sd, $Sm",
                                [(set SPR:$Sd, (arm_sitof SPR:$Sm))]> {
  let Inst{7} = 1; // s32

  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VUITOD : AVConv1IDs_Encode<0b11101, 0b11, 0b1000, 0b1011,
                               (outs DPR:$Dd), (ins SPR:$Sm),
                               IIC_fpCVTID, "vcvt", ".f64.u32\t$Dd, $Sm",
                               [(set DPR:$Dd, (f64 (arm_uitof SPR:$Sm)))]> {
  let Inst{7} = 0; // u32
}

def VUITOS : AVConv1InSs_Encode<0b11101, 0b11, 0b1000, 0b1010,
                                (outs SPR:$Sd), (ins SPR:$Sm),
                                IIC_fpCVTIS, "vcvt", ".f32.u32\t$Sd, $Sm",
                                [(set SPR:$Sd, (arm_uitof SPR:$Sm))]> {
  let Inst{7} = 0; // u32

  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

// FP -> Int:

class AVConv1IsD_Encode<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3,
                        bits<4> opcod4, dag oops, dag iops,
                        InstrItinClass itin, string opc, string asm,
                        list<dag> pattern>
  : AVConv1I<opcod1, opcod2, opcod3, opcod4, oops, iops, itin, opc, asm,
             pattern> {
  // Instruction operands.
  bits<5> Sd;
  bits<5> Dm;

  // Encode instruction operands.
  let Inst{3-0}   = Dm{3-0};
  let Inst{5}     = Dm{4};
  let Inst{15-12} = Sd{4-1};
  let Inst{22}    = Sd{0};
}

class AVConv1InsS_Encode<bits<5> opcod1, bits<2> opcod2, bits<4> opcod3,
                         bits<4> opcod4, dag oops, dag iops,
                         InstrItinClass itin, string opc, string asm,
                         list<dag> pattern>
  : AVConv1In<opcod1, opcod2, opcod3, opcod4, oops, iops, itin, opc, asm,
              pattern> {
  // Instruction operands.
  bits<5> Sd;
  bits<5> Sm;

  // Encode instruction operands.
  let Inst{3-0}   = Sm{4-1};
  let Inst{5}     = Sm{0};
  let Inst{15-12} = Sd{4-1};
  let Inst{22}    = Sd{0};
}

// Always set Z bit in the instruction, i.e. "round towards zero" variants.
def VTOSIZD : AVConv1IsD_Encode<0b11101, 0b11, 0b1101, 0b1011,
                                (outs SPR:$Sd), (ins DPR:$Dm),
                                IIC_fpCVTDI, "vcvt", ".s32.f64\t$Sd, $Dm",
                                [(set SPR:$Sd, (arm_ftosi (f64 DPR:$Dm)))]> {
  let Inst{7} = 1; // Z bit
}

def VTOSIZS : AVConv1InsS_Encode<0b11101, 0b11, 0b1101, 0b1010,
                                 (outs SPR:$Sd), (ins SPR:$Sm),
                                 IIC_fpCVTSI, "vcvt", ".s32.f32\t$Sd, $Sm",
                                 [(set SPR:$Sd, (arm_ftosi SPR:$Sm))]> {
  let Inst{7} = 1; // Z bit

  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VTOUIZD : AVConv1IsD_Encode<0b11101, 0b11, 0b1100, 0b1011,
                               (outs SPR:$Sd), (ins DPR:$Dm),
                               IIC_fpCVTDI, "vcvt", ".u32.f64\t$Sd, $Dm",
                               [(set SPR:$Sd, (arm_ftoui (f64 DPR:$Dm)))]> {
  let Inst{7} = 1; // Z bit
}

def VTOUIZS : AVConv1InsS_Encode<0b11101, 0b11, 0b1100, 0b1010,
                                 (outs SPR:$Sd), (ins SPR:$Sm),
                                 IIC_fpCVTSI, "vcvt", ".u32.f32\t$Sd, $Sm",
                                 [(set SPR:$Sd, (arm_ftoui SPR:$Sm))]> {
  let Inst{7} = 1; // Z bit

  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

// And the Z bit '0' variants, i.e. use the rounding mode specified by FPSCR.
let Uses = [FPSCR] in {
// FIXME: Verify encoding after integrated assembler is working.
def VTOSIRD : AVConv1IsD_Encode<0b11101, 0b11, 0b1101, 0b1011,
                                (outs SPR:$Sd), (ins DPR:$Dm),
                                IIC_fpCVTDI, "vcvtr", ".s32.f64\t$Sd, $Dm",
                                [(set SPR:$Sd, (int_arm_vcvtr (f64 DPR:$Dm)))]>{
  let Inst{7} = 0; // Z bit
}

def VTOSIRS : AVConv1InsS_Encode<0b11101, 0b11, 0b1101, 0b1010,
                                 (outs SPR:$Sd), (ins SPR:$Sm),
                                 IIC_fpCVTSI, "vcvtr", ".s32.f32\t$Sd, $Sm",
                                 [(set SPR:$Sd, (int_arm_vcvtr SPR:$Sm))]> {
  let Inst{7} = 0; // Z bit
}

def VTOUIRD : AVConv1IsD_Encode<0b11101, 0b11, 0b1100, 0b1011,
                                (outs SPR:$Sd), (ins DPR:$Dm),
                                IIC_fpCVTDI, "vcvtr", ".u32.f64\t$Sd, $Dm",
                                [(set SPR:$Sd, (int_arm_vcvtru(f64 DPR:$Dm)))]>{
  let Inst{7} = 0; // Z bit
}

def VTOUIRS : AVConv1InsS_Encode<0b11101, 0b11, 0b1100, 0b1010,
                                 (outs SPR:$Sd), (ins SPR:$Sm),
                                 IIC_fpCVTSI, "vcvtr", ".u32.f32\t$Sd, $Sm",
                                 [(set SPR:$Sd, (int_arm_vcvtru SPR:$Sm))]> {
  let Inst{7} = 0; // Z bit
}
}

// Convert between floating-point and fixed-point
// Data type for fixed-point naming convention:
//   S16 (U=0, sx=0) -> SH
//   U16 (U=1, sx=0) -> UH
//   S32 (U=0, sx=1) -> SL
//   U32 (U=1, sx=1) -> UL

// FIXME: Marking these as codegen only seems wrong. They are real
//        instructions(?)
let Constraints = "$a = $dst", isCodeGenOnly = 1 in {

// FP to Fixed-Point:

def VTOSHS : AVConv1XI<0b11101, 0b11, 0b1110, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".s16.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VTOUHS : AVConv1XI<0b11101, 0b11, 0b1111, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".u16.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VTOSLS : AVConv1XI<0b11101, 0b11, 0b1110, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".s32.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VTOULS : AVConv1XI<0b11101, 0b11, 0b1111, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".u32.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VTOSHD : AVConv1XI<0b11101, 0b11, 0b1110, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".s16.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOUHD : AVConv1XI<0b11101, 0b11, 0b1111, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".u16.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOSLD : AVConv1XI<0b11101, 0b11, 0b1110, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".s32.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOULD : AVConv1XI<0b11101, 0b11, 0b1111, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".u32.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

// Fixed-Point to FP:

def VSHTOS : AVConv1XI<0b11101, 0b11, 0b1010, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.s16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VUHTOS : AVConv1XI<0b11101, 0b11, 0b1011, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.u16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VSLTOS : AVConv1XI<0b11101, 0b11, 0b1010, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.s32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VULTOS : AVConv1XI<0b11101, 0b11, 0b1011, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.u32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def VSHTOD : AVConv1XI<0b11101, 0b11, 0b1010, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.s16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VUHTOD : AVConv1XI<0b11101, 0b11, 0b1011, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.u16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VSLTOD : AVConv1XI<0b11101, 0b11, 0b1010, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.s32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VULTOD : AVConv1XI<0b11101, 0b11, 0b1011, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.u32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

} // End of 'let Constraints = "$a = $dst", isCodeGenOnly = 1 in'

//===----------------------------------------------------------------------===//
// FP Multiply-Accumulate Operations.
//

def VMLAD : ADbI<0b11100, 0b00, 0, 0,
                 (outs DPR:$Dd), (ins DPR:$Ddin, DPR:$Dn, DPR:$Dm),
                 IIC_fpMAC64, "vmla", ".f64\t$Dd, $Dn, $Dm",
                 [(set DPR:$Dd, (fadd_mlx (fmul_su DPR:$Dn, DPR:$Dm),
                                          (f64 DPR:$Ddin)))]>,
              RegConstraint<"$Ddin = $Dd">,
              Requires<[HasVFP2,UseFPVMLx]>;

def VMLAS : ASbIn<0b11100, 0b00, 0, 0,
                  (outs SPR:$Sd), (ins SPR:$Sdin, SPR:$Sn, SPR:$Sm),
                  IIC_fpMAC32, "vmla", ".f32\t$Sd, $Sn, $Sm",
                  [(set SPR:$Sd, (fadd_mlx (fmul_su SPR:$Sn, SPR:$Sm),
                                           SPR:$Sdin))]>,
              RegConstraint<"$Sdin = $Sd">,
              Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def : Pat<(fadd_mlx DPR:$dstin, (fmul_su DPR:$a, (f64 DPR:$b))),
          (VMLAD DPR:$dstin, DPR:$a, DPR:$b)>,
          Requires<[HasVFP2,UseFPVMLx]>;
def : Pat<(fadd_mlx SPR:$dstin, (fmul_su SPR:$a, SPR:$b)),
          (VMLAS SPR:$dstin, SPR:$a, SPR:$b)>,
          Requires<[HasVFP2,DontUseNEONForFP, UseFPVMLx]>;

def VMLSD : ADbI<0b11100, 0b00, 1, 0,
                 (outs DPR:$Dd), (ins DPR:$Ddin, DPR:$Dn, DPR:$Dm),
                 IIC_fpMAC64, "vmls", ".f64\t$Dd, $Dn, $Dm",
                 [(set DPR:$Dd, (fadd_mlx (fneg (fmul_su DPR:$Dn,DPR:$Dm)),
                                          (f64 DPR:$Ddin)))]>,
              RegConstraint<"$Ddin = $Dd">,
              Requires<[HasVFP2,UseFPVMLx]>;

def VMLSS : ASbIn<0b11100, 0b00, 1, 0,
                  (outs SPR:$Sd), (ins SPR:$Sdin, SPR:$Sn, SPR:$Sm),
                  IIC_fpMAC32, "vmls", ".f32\t$Sd, $Sn, $Sm",
                  [(set SPR:$Sd, (fadd_mlx (fneg (fmul_su SPR:$Sn, SPR:$Sm)),
                                           SPR:$Sdin))]>,
              RegConstraint<"$Sdin = $Sd">,
              Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def : Pat<(fsub_mlx DPR:$dstin, (fmul_su DPR:$a, (f64 DPR:$b))),
          (VMLSD DPR:$dstin, DPR:$a, DPR:$b)>,
          Requires<[HasVFP2,UseFPVMLx]>;
def : Pat<(fsub_mlx SPR:$dstin, (fmul_su SPR:$a, SPR:$b)),
          (VMLSS SPR:$dstin, SPR:$a, SPR:$b)>,
          Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]>;

def VNMLAD : ADbI<0b11100, 0b01, 1, 0,
                  (outs DPR:$Dd), (ins DPR:$Ddin, DPR:$Dn, DPR:$Dm),
                  IIC_fpMAC64, "vnmla", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd,(fsub_mlx (fneg (fmul_su DPR:$Dn,DPR:$Dm)),
                                          (f64 DPR:$Ddin)))]>,
                RegConstraint<"$Ddin = $Dd">,
                Requires<[HasVFP2,UseFPVMLx]>;

def VNMLAS : ASbI<0b11100, 0b01, 1, 0,
                  (outs SPR:$Sd), (ins SPR:$Sdin, SPR:$Sn, SPR:$Sm),
                  IIC_fpMAC32, "vnmla", ".f32\t$Sd, $Sn, $Sm",
                  [(set SPR:$Sd, (fsub_mlx (fneg (fmul_su SPR:$Sn, SPR:$Sm)),
                                           SPR:$Sdin))]>,
                RegConstraint<"$Sdin = $Sd">,
                Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def : Pat<(fsub_mlx (fneg (fmul_su DPR:$a, (f64 DPR:$b))), DPR:$dstin),
          (VNMLAD DPR:$dstin, DPR:$a, DPR:$b)>,
          Requires<[HasVFP2,UseFPVMLx]>;
def : Pat<(fsub_mlx (fneg (fmul_su SPR:$a, SPR:$b)), SPR:$dstin),
          (VNMLAS SPR:$dstin, SPR:$a, SPR:$b)>,
          Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]>;

def VNMLSD : ADbI<0b11100, 0b01, 0, 0,
                  (outs DPR:$Dd), (ins DPR:$Ddin, DPR:$Dn, DPR:$Dm),
                  IIC_fpMAC64, "vnmls", ".f64\t$Dd, $Dn, $Dm",
                  [(set DPR:$Dd, (fsub_mlx (fmul_su DPR:$Dn, DPR:$Dm),
                                           (f64 DPR:$Ddin)))]>,
               RegConstraint<"$Ddin = $Dd">,
               Requires<[HasVFP2,UseFPVMLx]>;

def VNMLSS : ASbI<0b11100, 0b01, 0, 0,
                  (outs SPR:$Sd), (ins SPR:$Sdin, SPR:$Sn, SPR:$Sm),
                  IIC_fpMAC32, "vnmls", ".f32\t$Sd, $Sn, $Sm",
             [(set SPR:$Sd, (fsub_mlx (fmul_su SPR:$Sn, SPR:$Sm), SPR:$Sdin))]>,
                         RegConstraint<"$Sdin = $Sd">,
                  Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]> {
  // Some single precision VFP instructions may be executed on both NEON and
  // VFP pipelines on A8.
  let D = VFPNeonA8Domain;
}

def : Pat<(fsub_mlx (fmul_su DPR:$a, (f64 DPR:$b)), DPR:$dstin),
          (VNMLSD DPR:$dstin, DPR:$a, DPR:$b)>,
          Requires<[HasVFP2,UseFPVMLx]>;
def : Pat<(fsub_mlx (fmul_su SPR:$a, SPR:$b), SPR:$dstin),
          (VNMLSS SPR:$dstin, SPR:$a, SPR:$b)>,
          Requires<[HasVFP2,DontUseNEONForFP,UseFPVMLx]>;


//===----------------------------------------------------------------------===//
// FP Conditional moves.
//

let neverHasSideEffects = 1 in {
def VMOVDcc  : ARMPseudoInst<(outs DPR:$Dd), (ins DPR:$Dn, DPR:$Dm, pred:$p),
                    4, IIC_fpUNA64,
                    [/*(set DPR:$Dd, (ARMcmov DPR:$Dn, DPR:$Dm, imm:$cc))*/]>,
                 RegConstraint<"$Dn = $Dd">;

def VMOVScc  : ARMPseudoInst<(outs SPR:$Sd), (ins SPR:$Sn, SPR:$Sm, pred:$p),
                    4, IIC_fpUNA32,
                    [/*(set SPR:$Sd, (ARMcmov SPR:$Sn, SPR:$Sm, imm:$cc))*/]>,
                 RegConstraint<"$Sn = $Sd">;
} // neverHasSideEffects

//===----------------------------------------------------------------------===//
// Move from VFP System Register to ARM core register.
//

class MovFromVFP<bits<4> opc19_16, dag oops, dag iops, string opc, string asm,
                 list<dag> pattern>:
  VFPAI<oops, iops, VFPMiscFrm, IIC_fpSTAT, opc, asm, pattern> {

  // Instruction operand.
  bits<4> Rt;

  let Inst{27-20} = 0b11101111;
  let Inst{19-16} = opc19_16;
  let Inst{15-12} = Rt;
  let Inst{11-8}  = 0b1010;
  let Inst{7}     = 0;
  let Inst{6-5}   = 0b00;
  let Inst{4}     = 1;
  let Inst{3-0}   = 0b0000;
}

// APSR is the application level alias of CPSR. This FPSCR N, Z, C, V flags
// to APSR.
let Defs = [CPSR], Uses = [FPSCR], Rt = 0b1111 /* apsr_nzcv */ in
def FMSTAT : MovFromVFP<0b0001 /* fpscr */, (outs), (ins),
                        "vmrs", "\tapsr_nzcv, fpscr", [(arm_fmstat)]>;

// Application level FPSCR -> GPR
let hasSideEffects = 1, Uses = [FPSCR] in
def VMRS : MovFromVFP<0b0001 /* fpscr */, (outs GPR:$Rt), (ins),
                      "vmrs", "\t$Rt, fpscr",
                      [(set GPR:$Rt, (int_arm_get_fpscr))]>;

// System level FPEXC, FPSID -> GPR
let Uses = [FPSCR] in {
  def VMRS_FPEXC : MovFromVFP<0b1000 /* fpexc */, (outs GPR:$Rt), (ins),
                              "vmrs", "\t$Rt, fpexc", []>;
  def VMRS_FPSID : MovFromVFP<0b0000 /* fpsid */, (outs GPR:$Rt), (ins),
                              "vmrs", "\t$Rt, fpsid", []>;
}

//===----------------------------------------------------------------------===//
// Move from ARM core register to VFP System Register.
//

class MovToVFP<bits<4> opc19_16, dag oops, dag iops, string opc, string asm,
               list<dag> pattern>:
  VFPAI<oops, iops, VFPMiscFrm, IIC_fpSTAT, opc, asm, pattern> {

  // Instruction operand.
  bits<4> src;

  // Encode instruction operand.
  let Inst{15-12} = src;

  let Inst{27-20} = 0b11101110;
  let Inst{19-16} = opc19_16;
  let Inst{11-8}  = 0b1010;
  let Inst{7}     = 0;
  let Inst{4}     = 1;
}

let Defs = [FPSCR] in {
  // Application level GPR -> FPSCR
  def VMSR : MovToVFP<0b0001 /* fpscr */, (outs), (ins GPR:$src),
                      "vmsr", "\tfpscr, $src", [(int_arm_set_fpscr GPR:$src)]>;
  // System level GPR -> FPEXC
  def VMSR_FPEXC : MovToVFP<0b1000 /* fpexc */, (outs), (ins GPR:$src),
                      "vmsr", "\tfpexc, $src", []>;
  // System level GPR -> FPSID
  def VMSR_FPSID : MovToVFP<0b0000 /* fpsid */, (outs), (ins GPR:$src),
                      "vmsr", "\tfpsid, $src", []>;
}

//===----------------------------------------------------------------------===//
// Misc.
//

// Materialize FP immediates. VFP3 only.
let isReMaterializable = 1 in {
def FCONSTD : VFPAI<(outs DPR:$Dd), (ins vfp_f64imm:$imm),
                    VFPMiscFrm, IIC_fpUNA64,
                    "vmov", ".f64\t$Dd, $imm",
                    [(set DPR:$Dd, vfp_f64imm:$imm)]>, Requires<[HasVFP3]> {
  bits<5> Dd;
  bits<8> imm;

  let Inst{27-23} = 0b11101;
  let Inst{22}    = Dd{4};
  let Inst{21-20} = 0b11;
  let Inst{19-16} = imm{7-4};
  let Inst{15-12} = Dd{3-0};
  let Inst{11-9}  = 0b101;
  let Inst{8}     = 1;          // Double precision.
  let Inst{7-4}   = 0b0000;
  let Inst{3-0}   = imm{3-0};
}

def FCONSTS : VFPAI<(outs SPR:$Sd), (ins vfp_f32imm:$imm),
                     VFPMiscFrm, IIC_fpUNA32,
                     "vmov", ".f32\t$Sd, $imm",
                     [(set SPR:$Sd, vfp_f32imm:$imm)]>, Requires<[HasVFP3]> {
  bits<5> Sd;
  bits<8> imm;

  let Inst{27-23} = 0b11101;
  let Inst{22}    = Sd{0};
  let Inst{21-20} = 0b11;
  let Inst{19-16} = imm{7-4};
  let Inst{15-12} = Sd{4-1};
  let Inst{11-9}  = 0b101;
  let Inst{8}     = 0;          // Single precision.
  let Inst{7-4}   = 0b0000;
  let Inst{3-0}   = imm{3-0};
}
}

//===----------------------------------------------------------------------===//
// Assembler aliases.
//
// A few mnemnoic aliases for pre-unifixed syntax. We don't guarantee to
// support them all, but supporting at least some of the basics is
// good to be friendly.
def : VFP2MnemonicAlias<"flds", "vldr">;
def : VFP2MnemonicAlias<"fldd", "vldr">;
def : VFP2MnemonicAlias<"fmrs", "vmov">;
def : VFP2MnemonicAlias<"fmsr", "vmov">;
def : VFP2MnemonicAlias<"fsqrts", "vsqrt">;
def : VFP2MnemonicAlias<"fsqrtd", "vsqrt">;
def : VFP2MnemonicAlias<"fadds", "vadd.f32">;
def : VFP2MnemonicAlias<"faddd", "vadd.f64">;
def : VFP2MnemonicAlias<"fmrdd", "vmov">;
def : VFP2MnemonicAlias<"fmrds", "vmov">;
def : VFP2MnemonicAlias<"fmrrd", "vmov">;
def : VFP2MnemonicAlias<"fmdrr", "vmov">;
def : VFP2MnemonicAlias<"fmuld", "vmul.f64">;
def : VFP2MnemonicAlias<"fnegs", "vneg.f32">;
def : VFP2MnemonicAlias<"fnegd", "vneg.f64">;
def : VFP2MnemonicAlias<"ftosizd", "vcvt.s32.f64">;
def : VFP2MnemonicAlias<"ftosid", "vcvtr.s32.f64">;
def : VFP2MnemonicAlias<"ftosizs", "vcvt.s32.f32">;
def : VFP2MnemonicAlias<"ftosis", "vcvtr.s32.f32">;
def : VFP2MnemonicAlias<"ftouizd", "vcvt.u32.f64">;
def : VFP2MnemonicAlias<"ftouid", "vcvtr.u32.f64">;
def : VFP2MnemonicAlias<"ftouizs", "vcvt.u32.f32">;
def : VFP2MnemonicAlias<"ftouis", "vcvtr.u32.f32">;
def : VFP2MnemonicAlias<"fsitod", "vcvt.f64.s32">;
def : VFP2MnemonicAlias<"fsitos", "vcvt.f32.s32">;
def : VFP2MnemonicAlias<"fuitod", "vcvt.f64.u32">;
def : VFP2MnemonicAlias<"fuitos", "vcvt.f32.u32">;
def : VFP2MnemonicAlias<"fsts", "vstr">;
def : VFP2MnemonicAlias<"fstd", "vstr">;
def : VFP2MnemonicAlias<"fmacd", "vmla.f64">;
def : VFP2MnemonicAlias<"fmacs", "vmla.f32">;

def : VFP2InstAlias<"fmstat${p}", (FMSTAT pred:$p)>;
def : VFP2InstAlias<"fadds${p} $Sd, $Sn, $Sm",
                    (VADDS SPR:$Sd, SPR:$Sn, SPR:$Sm, pred:$p)>;
def : VFP2InstAlias<"faddd${p} $Dd, $Dn, $Dm",
                    (VADDD DPR:$Dd, DPR:$Dn, DPR:$Dm, pred:$p)>;
def : VFP2InstAlias<"fsubs${p} $Sd, $Sn, $Sm",
                    (VSUBS SPR:$Sd, SPR:$Sn, SPR:$Sm, pred:$p)>;
def : VFP2InstAlias<"fsubd${p} $Dd, $Dn, $Dm",
                    (VSUBD DPR:$Dd, DPR:$Dn, DPR:$Dm, pred:$p)>;

// No need for the size suffix on VSQRT. It's implied by the register classes.
def : VFP2InstAlias<"vsqrt${p} $Sd, $Sm", (VSQRTS SPR:$Sd, SPR:$Sm, pred:$p)>;
def : VFP2InstAlias<"vsqrt${p} $Dd, $Dm", (VSQRTD DPR:$Dd, DPR:$Dm, pred:$p)>;

// VLDR/VSTR accept an optional type suffix.
def : VFP2InstAlias<"vldr${p}.32 $Sd, $addr",
                    (VLDRS SPR:$Sd, addrmode5:$addr, pred:$p)>;
def : VFP2InstAlias<"vstr${p}.32 $Sd, $addr",
                    (VSTRS SPR:$Sd, addrmode5:$addr, pred:$p)>;
def : VFP2InstAlias<"vldr${p}.64 $Dd, $addr",
                    (VLDRD DPR:$Dd, addrmode5:$addr, pred:$p)>;
def : VFP2InstAlias<"vstr${p}.64 $Dd, $addr",
                    (VSTRD DPR:$Dd, addrmode5:$addr, pred:$p)>;

// VMUL has a two-operand form (implied destination operand)
def : VFP2InstAlias<"vmul${p}.f64 $Dn, $Dm",
                    (VMULD DPR:$Dn, DPR:$Dn, DPR:$Dm, pred:$p)>;
def : VFP2InstAlias<"vmul${p}.f32 $Sn, $Sm",
                    (VMULS SPR:$Sn, SPR:$Sn, SPR:$Sm, pred:$p)>;
// VADD has a two-operand form (implied destination operand)
def : VFP2InstAlias<"vadd${p}.f64 $Dn, $Dm",
                    (VADDD DPR:$Dn, DPR:$Dn, DPR:$Dm, pred:$p)>;
def : VFP2InstAlias<"vadd${p}.f32 $Sn, $Sm",
                    (VADDS SPR:$Sn, SPR:$Sn, SPR:$Sm, pred:$p)>;
// VSUB has a two-operand form (implied destination operand)
def : VFP2InstAlias<"vsub${p}.f64 $Dn, $Dm",
                    (VSUBD DPR:$Dn, DPR:$Dn, DPR:$Dm, pred:$p)>;
def : VFP2InstAlias<"vsub${p}.f32 $Sn, $Sm",
                    (VSUBS SPR:$Sn, SPR:$Sn, SPR:$Sm, pred:$p)>;

// VMOV can accept optional .f32/.f64 suffix.
def : VFP2InstAlias<"vmov${p}.f32 $Rt, $Sn",
                    (VMOVRS GPR:$Rt, SPR:$Sn, pred:$p)>;
def : VFP2InstAlias<"vmov${p}.f32 $Sn, $Rt",
                    (VMOVSR SPR:$Sn, GPR:$Rt, pred:$p)>;

def : VFP2InstAlias<"vmov${p}.f64 $Rt, $Rt2, $Dn",
                    (VMOVRRD GPR:$Rt, GPR:$Rt2, DPR:$Dn, pred:$p)>;
def : VFP2InstAlias<"vmov${p}.f64 $Dn, $Rt, $Rt2",
                    (VMOVDRR DPR:$Dn, GPR:$Rt, GPR:$Rt2, pred:$p)>;

// VMOVS doesn't need the .f32 to disambiguate from the NEON encoding the way
// VMOVD does.
def : VFP2InstAlias<"vmov${p} $Sd, $Sm",
                    (VMOVS SPR:$Sd, SPR:$Sm, pred:$p)>;
