//===- ARM64InstrInfo.td - Describe the ARM64 Instructions -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ARM64 Instruction definitions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ARM64-specific DAG Nodes.
//

// SDTBinaryArithWithFlagsOut - RES1, FLAGS = op LHS, RHS
def SDTBinaryArithWithFlagsOut : SDTypeProfile<2, 2,
                                              [SDTCisSameAs<0, 2>,
                                               SDTCisSameAs<0, 3>,
                                               SDTCisInt<0>, SDTCisVT<1, i32>]>;

// SDTBinaryArithWithFlagsIn - RES1, FLAGS = op LHS, RHS, FLAGS
def SDTBinaryArithWithFlagsIn : SDTypeProfile<1, 3,
                                            [SDTCisSameAs<0, 1>,
                                             SDTCisSameAs<0, 2>,
                                             SDTCisInt<0>,
                                             SDTCisVT<3, i32>]>;

// SDTBinaryArithWithFlagsInOut - RES1, FLAGS = op LHS, RHS, FLAGS
def SDTBinaryArithWithFlagsInOut : SDTypeProfile<2, 3,
                                            [SDTCisSameAs<0, 2>,
                                             SDTCisSameAs<0, 3>,
                                             SDTCisInt<0>,
                                             SDTCisVT<1, i32>,
                                             SDTCisVT<4, i32>]>;

def SDT_ARM64Brcond  : SDTypeProfile<0, 3,
                                     [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>,
                                      SDTCisVT<2, i32>]>;
def SDT_ARM64cbz : SDTypeProfile<0, 2, [SDTCisInt<0>, SDTCisVT<1, OtherVT>]>;
def SDT_ARM64tbz : SDTypeProfile<0, 3, [SDTCisVT<0, i64>, SDTCisVT<1, i64>,
                                        SDTCisVT<2, OtherVT>]>;


def SDT_ARM64CSel  : SDTypeProfile<1, 4,
                                   [SDTCisSameAs<0, 1>,
                                    SDTCisSameAs<0, 2>,
                                    SDTCisInt<3>,
                                    SDTCisVT<4, i32>]>;
def SDT_ARM64FCmp   : SDTypeProfile<0, 2,
                                   [SDTCisFP<0>,
                                    SDTCisSameAs<0, 1>]>;
def SDT_ARM64Dup   : SDTypeProfile<1, 1, [SDTCisVec<0>]>;
def SDT_ARM64DupLane   : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisInt<2>]>;
def SDT_ARM64Zip   : SDTypeProfile<1, 2, [SDTCisVec<0>,
                                          SDTCisSameAs<0, 1>,
                                          SDTCisSameAs<0, 2>]>;
def SDT_ARM64MOVIedit : SDTypeProfile<1, 1, [SDTCisInt<1>]>;
def SDT_ARM64MOVIshift : SDTypeProfile<1, 2, [SDTCisInt<1>, SDTCisInt<2>]>;
def SDT_ARM64vecimm : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                           SDTCisInt<2>, SDTCisInt<3>]>;
def SDT_ARM64UnaryVec: SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisSameAs<0,1>]>;
def SDT_ARM64ExtVec: SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                          SDTCisSameAs<0,2>, SDTCisInt<3>]>;
def SDT_ARM64vshift : SDTypeProfile<1, 2, [SDTCisSameAs<0,1>, SDTCisInt<2>]>;

def SDT_ARM64unvec : SDTypeProfile<1, 1, [SDTCisVec<0>, SDTCisSameAs<0,1>]>;
def SDT_ARM64fcmpz : SDTypeProfile<1, 1, []>;
def SDT_ARM64fcmp  : SDTypeProfile<1, 2, [SDTCisSameAs<1,2>]>;
def SDT_ARM64binvec : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                           SDTCisSameAs<0,2>]>;
def SDT_ARM64trivec : SDTypeProfile<1, 3, [SDTCisVec<0>, SDTCisSameAs<0,1>,
                                           SDTCisSameAs<0,2>,
                                           SDTCisSameAs<0,3>]>;
def SDT_ARM64TCRET : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;
def SDT_ARM64PREFETCH : SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisPtrTy<1>]>;

def SDT_ARM64ITOF  : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisSameAs<0,1>]>;

def SDT_ARM64TLSDescCall : SDTypeProfile<0, -2, [SDTCisPtrTy<0>,
                                                 SDTCisPtrTy<1>]>;
def SDT_ARM64WrapperLarge : SDTypeProfile<1, 4,
                                        [SDTCisVT<0, i64>, SDTCisVT<1, i32>,
                                         SDTCisSameAs<1, 2>, SDTCisSameAs<1, 3>,
                                         SDTCisSameAs<1, 4>]>;


// Node definitions.
def ARM64adrp          : SDNode<"ARM64ISD::ADRP", SDTIntUnaryOp, []>;
def ARM64addlow        : SDNode<"ARM64ISD::ADDlow", SDTIntBinOp, []>;
def ARM64LOADgot       : SDNode<"ARM64ISD::LOADgot", SDTIntUnaryOp>;
def ARM64callseq_start : SDNode<"ISD::CALLSEQ_START",
                                SDCallSeqStart<[ SDTCisVT<0, i32> ]>,
                                [SDNPHasChain, SDNPOutGlue]>;
def ARM64callseq_end   : SDNode<"ISD::CALLSEQ_END",
                                SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                               SDTCisVT<1, i32> ]>,
                                [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def ARM64call          : SDNode<"ARM64ISD::CALL",
                                SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
                                [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                                 SDNPVariadic]>;
def ARM64brcond        : SDNode<"ARM64ISD::BRCOND", SDT_ARM64Brcond,
                                [SDNPHasChain]>;
def ARM64cbz           : SDNode<"ARM64ISD::CBZ", SDT_ARM64cbz,
                                [SDNPHasChain]>;
def ARM64cbnz           : SDNode<"ARM64ISD::CBNZ", SDT_ARM64cbz,
                                [SDNPHasChain]>;
def ARM64tbz           : SDNode<"ARM64ISD::TBZ", SDT_ARM64tbz,
                                [SDNPHasChain]>;
def ARM64tbnz           : SDNode<"ARM64ISD::TBNZ", SDT_ARM64tbz,
                                [SDNPHasChain]>;


def ARM64csel          : SDNode<"ARM64ISD::CSEL", SDT_ARM64CSel>;
def ARM64csinv         : SDNode<"ARM64ISD::CSINV", SDT_ARM64CSel>;
def ARM64csneg         : SDNode<"ARM64ISD::CSNEG", SDT_ARM64CSel>;
def ARM64csinc         : SDNode<"ARM64ISD::CSINC", SDT_ARM64CSel>;
def ARM64retflag       : SDNode<"ARM64ISD::RET_FLAG", SDTNone,
                                [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def ARM64adc       : SDNode<"ARM64ISD::ADC",  SDTBinaryArithWithFlagsIn >;
def ARM64sbc       : SDNode<"ARM64ISD::SBC",  SDTBinaryArithWithFlagsIn>;
def ARM64add_flag  : SDNode<"ARM64ISD::ADDS",  SDTBinaryArithWithFlagsOut,
                            [SDNPCommutative]>;
def ARM64sub_flag  : SDNode<"ARM64ISD::SUBS",  SDTBinaryArithWithFlagsOut>;
def ARM64and_flag  : SDNode<"ARM64ISD::ANDS",  SDTBinaryArithWithFlagsOut>;
def ARM64adc_flag  : SDNode<"ARM64ISD::ADCS",  SDTBinaryArithWithFlagsInOut>;
def ARM64sbc_flag  : SDNode<"ARM64ISD::SBCS",  SDTBinaryArithWithFlagsInOut>;

def ARM64threadpointer : SDNode<"ARM64ISD::THREAD_POINTER", SDTPtrLeaf>;

def ARM64fcmp      : SDNode<"ARM64ISD::FCMP", SDT_ARM64FCmp>;

def ARM64fmax      : SDNode<"ARM64ISD::FMAX", SDTFPBinOp>;
def ARM64fmin      : SDNode<"ARM64ISD::FMIN", SDTFPBinOp>;

def ARM64dup       : SDNode<"ARM64ISD::DUP", SDT_ARM64Dup>;
def ARM64duplane8  : SDNode<"ARM64ISD::DUPLANE8", SDT_ARM64DupLane>;
def ARM64duplane16 : SDNode<"ARM64ISD::DUPLANE16", SDT_ARM64DupLane>;
def ARM64duplane32 : SDNode<"ARM64ISD::DUPLANE32", SDT_ARM64DupLane>;
def ARM64duplane64 : SDNode<"ARM64ISD::DUPLANE64", SDT_ARM64DupLane>;

def ARM64zip1      : SDNode<"ARM64ISD::ZIP1", SDT_ARM64Zip>;
def ARM64zip2      : SDNode<"ARM64ISD::ZIP2", SDT_ARM64Zip>;
def ARM64uzp1      : SDNode<"ARM64ISD::UZP1", SDT_ARM64Zip>;
def ARM64uzp2      : SDNode<"ARM64ISD::UZP2", SDT_ARM64Zip>;
def ARM64trn1      : SDNode<"ARM64ISD::TRN1", SDT_ARM64Zip>;
def ARM64trn2      : SDNode<"ARM64ISD::TRN2", SDT_ARM64Zip>;

def ARM64movi_edit : SDNode<"ARM64ISD::MOVIedit", SDT_ARM64MOVIedit>;
def ARM64movi_shift : SDNode<"ARM64ISD::MOVIshift", SDT_ARM64MOVIshift>;
def ARM64movi_msl : SDNode<"ARM64ISD::MOVImsl", SDT_ARM64MOVIshift>;
def ARM64mvni_shift : SDNode<"ARM64ISD::MVNIshift", SDT_ARM64MOVIshift>;
def ARM64mvni_msl : SDNode<"ARM64ISD::MVNImsl", SDT_ARM64MOVIshift>;
def ARM64movi : SDNode<"ARM64ISD::MOVI", SDT_ARM64MOVIedit>;
def ARM64fmov : SDNode<"ARM64ISD::FMOV", SDT_ARM64MOVIedit>;

def ARM64rev16 : SDNode<"ARM64ISD::REV16", SDT_ARM64UnaryVec>;
def ARM64rev32 : SDNode<"ARM64ISD::REV32", SDT_ARM64UnaryVec>;
def ARM64rev64 : SDNode<"ARM64ISD::REV64", SDT_ARM64UnaryVec>;
def ARM64ext : SDNode<"ARM64ISD::EXT", SDT_ARM64ExtVec>;

def ARM64vashr : SDNode<"ARM64ISD::VASHR", SDT_ARM64vshift>;
def ARM64vlshr : SDNode<"ARM64ISD::VLSHR", SDT_ARM64vshift>;
def ARM64vshl : SDNode<"ARM64ISD::VSHL", SDT_ARM64vshift>;
def ARM64sqshli : SDNode<"ARM64ISD::SQSHL_I", SDT_ARM64vshift>;
def ARM64uqshli : SDNode<"ARM64ISD::UQSHL_I", SDT_ARM64vshift>;
def ARM64sqshlui : SDNode<"ARM64ISD::SQSHLU_I", SDT_ARM64vshift>;
def ARM64srshri : SDNode<"ARM64ISD::SRSHR_I", SDT_ARM64vshift>;
def ARM64urshri : SDNode<"ARM64ISD::URSHR_I", SDT_ARM64vshift>;

def ARM64not: SDNode<"ARM64ISD::NOT", SDT_ARM64unvec>;
def ARM64bit: SDNode<"ARM64ISD::BIT", SDT_ARM64trivec>;

def ARM64cmeq: SDNode<"ARM64ISD::CMEQ", SDT_ARM64binvec>;
def ARM64cmge: SDNode<"ARM64ISD::CMGE", SDT_ARM64binvec>;
def ARM64cmgt: SDNode<"ARM64ISD::CMGT", SDT_ARM64binvec>;
def ARM64cmhi: SDNode<"ARM64ISD::CMHI", SDT_ARM64binvec>;
def ARM64cmhs: SDNode<"ARM64ISD::CMHS", SDT_ARM64binvec>;

def ARM64fcmeq: SDNode<"ARM64ISD::FCMEQ", SDT_ARM64fcmp>;
def ARM64fcmge: SDNode<"ARM64ISD::FCMGE", SDT_ARM64fcmp>;
def ARM64fcmgt: SDNode<"ARM64ISD::FCMGT", SDT_ARM64fcmp>;

def ARM64cmeqz: SDNode<"ARM64ISD::CMEQz", SDT_ARM64unvec>;
def ARM64cmgez: SDNode<"ARM64ISD::CMGEz", SDT_ARM64unvec>;
def ARM64cmgtz: SDNode<"ARM64ISD::CMGTz", SDT_ARM64unvec>;
def ARM64cmlez: SDNode<"ARM64ISD::CMLEz", SDT_ARM64unvec>;
def ARM64cmltz: SDNode<"ARM64ISD::CMLTz", SDT_ARM64unvec>;
def ARM64cmtst : PatFrag<(ops node:$LHS, node:$RHS),
                         (ARM64not (ARM64cmeqz (and node:$LHS, node:$RHS)))>;

def ARM64fcmeqz: SDNode<"ARM64ISD::FCMEQz", SDT_ARM64fcmpz>;
def ARM64fcmgez: SDNode<"ARM64ISD::FCMGEz", SDT_ARM64fcmpz>;
def ARM64fcmgtz: SDNode<"ARM64ISD::FCMGTz", SDT_ARM64fcmpz>;
def ARM64fcmlez: SDNode<"ARM64ISD::FCMLEz", SDT_ARM64fcmpz>;
def ARM64fcmltz: SDNode<"ARM64ISD::FCMLTz", SDT_ARM64fcmpz>;

def ARM64bici: SDNode<"ARM64ISD::BICi", SDT_ARM64vecimm>;
def ARM64orri: SDNode<"ARM64ISD::ORRi", SDT_ARM64vecimm>;

def ARM64neg : SDNode<"ARM64ISD::NEG", SDT_ARM64unvec>;

def ARM64tcret: SDNode<"ARM64ISD::TC_RETURN", SDT_ARM64TCRET,
                  [SDNPHasChain,  SDNPOptInGlue, SDNPVariadic]>;

def ARM64Prefetch        : SDNode<"ARM64ISD::PREFETCH", SDT_ARM64PREFETCH,
                               [SDNPHasChain, SDNPSideEffect]>;

def ARM64sitof: SDNode<"ARM64ISD::SITOF", SDT_ARM64ITOF>;
def ARM64uitof: SDNode<"ARM64ISD::UITOF", SDT_ARM64ITOF>;

def ARM64tlsdesc_call : SDNode<"ARM64ISD::TLSDESC_CALL", SDT_ARM64TLSDescCall,
                               [SDNPInGlue, SDNPOutGlue, SDNPHasChain,
                                SDNPVariadic]>;

def ARM64WrapperLarge : SDNode<"ARM64ISD::WrapperLarge", SDT_ARM64WrapperLarge>;


//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// ARM64 Instruction Predicate Definitions.
//
def HasZCZ    : Predicate<"Subtarget->hasZeroCycleZeroing()">;
def NoZCZ     : Predicate<"!Subtarget->hasZeroCycleZeroing()">;
def IsDarwin  : Predicate<"Subtarget->isTargetDarwin()">;
def IsNotDarwin: Predicate<"!Subtarget->isTargetDarwin()">;
def ForCodeSize   : Predicate<"ForCodeSize">;
def NotForCodeSize   : Predicate<"!ForCodeSize">;

include "ARM64InstrFormats.td"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Miscellaneous instructions.
//===----------------------------------------------------------------------===//

let Defs = [SP], Uses = [SP], hasSideEffects = 1, isCodeGenOnly = 1 in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt),
                              [(ARM64callseq_start timm:$amt)]>;
def ADJCALLSTACKUP : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            [(ARM64callseq_end timm:$amt1, timm:$amt2)]>;
} // Defs = [SP], Uses = [SP], hasSideEffects = 1, isCodeGenOnly = 1

let isReMaterializable = 1, isCodeGenOnly = 1 in {
// FIXME: The following pseudo instructions are only needed because remat
// cannot handle multiple instructions.  When that changes, they can be
// removed, along with the ARM64Wrapper node.

let AddedComplexity = 10 in
def LOADgot : Pseudo<(outs GPR64:$dst), (ins i64imm:$addr),
                     [(set GPR64:$dst, (ARM64LOADgot tglobaladdr:$addr))]>,
              Sched<[WriteLDAdr]>;

// The MOVaddr instruction should match only when the add is not folded
// into a load or store address.
def MOVaddr
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$hi, i64imm:$low),
             [(set GPR64:$dst, (ARM64addlow (ARM64adrp tglobaladdr:$hi),
                                            tglobaladdr:$low))]>,
      Sched<[WriteAdrAdr]>;
def MOVaddrJT
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$hi, i64imm:$low),
             [(set GPR64:$dst, (ARM64addlow (ARM64adrp tjumptable:$hi),
                                             tjumptable:$low))]>,
      Sched<[WriteAdrAdr]>;
def MOVaddrCP
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$hi, i64imm:$low),
             [(set GPR64:$dst, (ARM64addlow (ARM64adrp tconstpool:$hi),
                                             tconstpool:$low))]>,
      Sched<[WriteAdrAdr]>;
def MOVaddrBA
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$hi, i64imm:$low),
             [(set GPR64:$dst, (ARM64addlow (ARM64adrp tblockaddress:$hi),
                                             tblockaddress:$low))]>,
      Sched<[WriteAdrAdr]>;
def MOVaddrTLS
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$hi, i64imm:$low),
             [(set GPR64:$dst, (ARM64addlow (ARM64adrp tglobaltlsaddr:$hi),
                                            tglobaltlsaddr:$low))]>,
      Sched<[WriteAdrAdr]>;
def MOVaddrEXT
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$hi, i64imm:$low),
             [(set GPR64:$dst, (ARM64addlow (ARM64adrp texternalsym:$hi),
                                            texternalsym:$low))]>,
      Sched<[WriteAdrAdr]>;

} // isReMaterializable, isCodeGenOnly

def : Pat<(ARM64LOADgot tglobaltlsaddr:$addr),
          (LOADgot tglobaltlsaddr:$addr)>;

def : Pat<(ARM64LOADgot texternalsym:$addr),
          (LOADgot texternalsym:$addr)>;

def : Pat<(ARM64LOADgot tconstpool:$addr),
          (LOADgot tconstpool:$addr)>;

//===----------------------------------------------------------------------===//
// System instructions.
//===----------------------------------------------------------------------===//

def HINT  : HintI<"hint">;
def : InstAlias<"nop",  (HINT 0b000)>;
def : InstAlias<"yield",(HINT 0b001)>;
def : InstAlias<"wfe",  (HINT 0b010)>;
def : InstAlias<"wfi",  (HINT 0b011)>;
def : InstAlias<"sev",  (HINT 0b100)>;
def : InstAlias<"sevl", (HINT 0b101)>;

  // As far as LLVM is concerned this writes to the system's exclusive monitors.
let mayLoad = 1, mayStore = 1 in
def CLREX : CRmSystemI<imm0_15, 0b010, "clrex">;

def DMB   : CRmSystemI<barrier_op, 0b101, "dmb">;
def DSB   : CRmSystemI<barrier_op, 0b100, "dsb">;
def ISB   : CRmSystemI<barrier_op, 0b110, "isb">;
def : InstAlias<"clrex", (CLREX 0xf)>;
def : InstAlias<"isb", (ISB 0xf)>;

def MRS    : MRSI;
def MSR    : MSRI;
def MSRcpsr: MSRcpsrI;

// The thread pointer (on Linux, at least, where this has been implemented) is
// TPIDR_EL0.
def : Pat<(ARM64threadpointer), (MRS 0xde82)>;

// Generic system instructions
def SYSxt  : SystemXtI<0, "sys">;
def SYSLxt : SystemLXtI<1, "sysl">;

def : InstAlias<"sys $op1, $Cn, $Cm, $op2",
                (SYSxt imm0_7:$op1, sys_cr_op:$Cn,
                 sys_cr_op:$Cm, imm0_7:$op2, XZR)>;

//===----------------------------------------------------------------------===//
// Move immediate instructions.
//===----------------------------------------------------------------------===//

defm MOVK : InsertImmediate<0b11, "movk">;
defm MOVN : MoveImmediate<0b00, "movn">;

let PostEncoderMethod = "fixMOVZ" in
defm MOVZ : MoveImmediate<0b10, "movz">;

def : InstAlias<"movk $dst, $imm", (MOVKWi GPR32:$dst, imm0_65535:$imm, 0)>;
def : InstAlias<"movk $dst, $imm", (MOVKXi GPR64:$dst, imm0_65535:$imm, 0)>;
def : InstAlias<"movn $dst, $imm", (MOVNWi GPR32:$dst, imm0_65535:$imm, 0)>;
def : InstAlias<"movn $dst, $imm", (MOVNXi GPR64:$dst, imm0_65535:$imm, 0)>;
def : InstAlias<"movz $dst, $imm", (MOVZWi GPR32:$dst, imm0_65535:$imm, 0)>;
def : InstAlias<"movz $dst, $imm", (MOVZXi GPR64:$dst, imm0_65535:$imm, 0)>;

def : InstAlias<"movz $Rd, $sym", (MOVZXi GPR64:$Rd, movz_symbol_g3:$sym, 48)>;
def : InstAlias<"movz $Rd, $sym", (MOVZXi GPR64:$Rd, movz_symbol_g2:$sym, 32)>;
def : InstAlias<"movz $Rd, $sym", (MOVZXi GPR64:$Rd, movz_symbol_g1:$sym, 16)>;
def : InstAlias<"movz $Rd, $sym", (MOVZXi GPR64:$Rd, movz_symbol_g0:$sym, 0)>;

def : InstAlias<"movn $Rd, $sym", (MOVNXi GPR64:$Rd, movz_symbol_g3:$sym, 48)>;
def : InstAlias<"movn $Rd, $sym", (MOVNXi GPR64:$Rd, movz_symbol_g2:$sym, 32)>;
def : InstAlias<"movn $Rd, $sym", (MOVNXi GPR64:$Rd, movz_symbol_g1:$sym, 16)>;
def : InstAlias<"movn $Rd, $sym", (MOVNXi GPR64:$Rd, movz_symbol_g0:$sym, 0)>;

def : InstAlias<"movz $Rd, $sym", (MOVZWi GPR32:$Rd, movz_symbol_g3:$sym, 48)>;
def : InstAlias<"movz $Rd, $sym", (MOVZWi GPR32:$Rd, movz_symbol_g2:$sym, 32)>;
def : InstAlias<"movz $Rd, $sym", (MOVZWi GPR32:$Rd, movz_symbol_g1:$sym, 16)>;
def : InstAlias<"movz $Rd, $sym", (MOVZWi GPR32:$Rd, movz_symbol_g0:$sym, 0)>;

def : InstAlias<"movk $Rd, $sym", (MOVKXi GPR64:$Rd, movk_symbol_g2:$sym, 32)>;
def : InstAlias<"movk $Rd, $sym", (MOVKXi GPR64:$Rd, movk_symbol_g1:$sym, 16)>;
def : InstAlias<"movk $Rd, $sym", (MOVKXi GPR64:$Rd, movk_symbol_g0:$sym, 0)>;

def : InstAlias<"movk $Rd, $sym", (MOVKWi GPR32:$Rd, movk_symbol_g2:$sym, 32)>;
def : InstAlias<"movk $Rd, $sym", (MOVKWi GPR32:$Rd, movk_symbol_g1:$sym, 16)>;
def : InstAlias<"movk $Rd, $sym", (MOVKWi GPR32:$Rd, movk_symbol_g0:$sym, 0)>;

let isReMaterializable = 1, isCodeGenOnly = 1, isMoveImm = 1,
    isAsCheapAsAMove = 1 in {
// FIXME: The following pseudo instructions are only needed because remat
// cannot handle multiple instructions.  When that changes, we can select
// directly to the real instructions and get rid of these pseudos.

def MOVi32imm
    : Pseudo<(outs GPR32:$dst), (ins i32imm:$src),
             [(set GPR32:$dst, imm:$src)]>,
      Sched<[WriteImm]>;
def MOVi64imm
    : Pseudo<(outs GPR64:$dst), (ins i64imm:$src),
             [(set GPR64:$dst, imm:$src)]>,
      Sched<[WriteImm]>;
} // isReMaterializable, isCodeGenOnly

def : Pat<(ARM64WrapperLarge tglobaladdr:$g3, tglobaladdr:$g2,
                             tglobaladdr:$g1, tglobaladdr:$g0),
          (MOVKXi (MOVKXi (MOVKXi (MOVZXi tglobaladdr:$g3, 48),
                                  tglobaladdr:$g2, 32),
                          tglobaladdr:$g1, 16),
                  tglobaladdr:$g0, 0)>;

def : Pat<(ARM64WrapperLarge tblockaddress:$g3, tblockaddress:$g2,
                             tblockaddress:$g1, tblockaddress:$g0),
          (MOVKXi (MOVKXi (MOVKXi (MOVZXi tblockaddress:$g3, 48),
                                  tblockaddress:$g2, 32),
                          tblockaddress:$g1, 16),
                  tblockaddress:$g0, 0)>;

def : Pat<(ARM64WrapperLarge tconstpool:$g3, tconstpool:$g2,
                             tconstpool:$g1, tconstpool:$g0),
          (MOVKXi (MOVKXi (MOVKXi (MOVZXi tconstpool:$g3, 48),
                                  tconstpool:$g2, 32),
                          tconstpool:$g1, 16),
                  tconstpool:$g0, 0)>;


//===----------------------------------------------------------------------===//
// Arithmetic instructions.
//===----------------------------------------------------------------------===//

// Add/subtract with carry.
defm ADC : AddSubCarry<0, "adc", "adcs", ARM64adc, ARM64adc_flag>;
defm SBC : AddSubCarry<1, "sbc", "sbcs", ARM64sbc, ARM64sbc_flag>;

def : InstAlias<"ngc $dst, $src",  (SBCWr  GPR32:$dst, WZR, GPR32:$src)>;
def : InstAlias<"ngc $dst, $src",  (SBCXr  GPR64:$dst, XZR, GPR64:$src)>;
def : InstAlias<"ngcs $dst, $src", (SBCSWr GPR32:$dst, WZR, GPR32:$src)>;
def : InstAlias<"ngcs $dst, $src", (SBCSXr GPR64:$dst, XZR, GPR64:$src)>;

// Add/subtract
defm ADD : AddSub<0, "add", add>;
defm SUB : AddSub<1, "sub">;

defm ADDS : AddSubS<0, "adds", ARM64add_flag>;
defm SUBS : AddSubS<1, "subs", ARM64sub_flag>;

// Use SUBS instead of SUB to enable CSE between SUBS and SUB.
def : Pat<(sub GPR32sp:$Rn, addsub_shifted_imm32:$imm),
          (SUBSWri GPR32sp:$Rn, addsub_shifted_imm32:$imm)>;
def : Pat<(sub GPR64sp:$Rn, addsub_shifted_imm64:$imm),
          (SUBSXri GPR64sp:$Rn, addsub_shifted_imm64:$imm)>;
def : Pat<(sub GPR32:$Rn, GPR32:$Rm),
          (SUBSWrr GPR32:$Rn, GPR32:$Rm)>;
def : Pat<(sub GPR64:$Rn, GPR64:$Rm),
          (SUBSXrr GPR64:$Rn, GPR64:$Rm)>;
def : Pat<(sub GPR32:$Rn, arith_shifted_reg32:$Rm),
          (SUBSWrs GPR32:$Rn, arith_shifted_reg32:$Rm)>;
def : Pat<(sub GPR64:$Rn, arith_shifted_reg64:$Rm),
          (SUBSXrs GPR64:$Rn, arith_shifted_reg64:$Rm)>;
def : Pat<(sub GPR32sp:$R2, arith_extended_reg32<i32>:$R3),
          (SUBSWrx GPR32sp:$R2, arith_extended_reg32<i32>:$R3)>;
def : Pat<(sub GPR64sp:$R2, arith_extended_reg32to64<i64>:$R3),
          (SUBSXrx GPR64sp:$R2, arith_extended_reg32to64<i64>:$R3)>;

// Because of the immediate format for add/sub-imm instructions, the
// expression (add x, -1) must be transformed to (SUB{W,X}ri x, 1).
//  These patterns capture that transformation.
let AddedComplexity = 1 in {
def : Pat<(add GPR32:$Rn, neg_addsub_shifted_imm32:$imm),
          (SUBSWri GPR32:$Rn, neg_addsub_shifted_imm32:$imm)>;
def : Pat<(add GPR64:$Rn, neg_addsub_shifted_imm64:$imm),
          (SUBSXri GPR64:$Rn, neg_addsub_shifted_imm64:$imm)>;
def : Pat<(sub GPR32:$Rn, neg_addsub_shifted_imm32:$imm),
          (ADDWri GPR32:$Rn, neg_addsub_shifted_imm32:$imm)>;
def : Pat<(sub GPR64:$Rn, neg_addsub_shifted_imm64:$imm),
          (ADDXri GPR64:$Rn, neg_addsub_shifted_imm64:$imm)>;
}

def : InstAlias<"neg $dst, $src", (SUBWrs GPR32:$dst, WZR, GPR32:$src, 0)>;
def : InstAlias<"neg $dst, $src", (SUBXrs GPR64:$dst, XZR, GPR64:$src, 0)>;
def : InstAlias<"neg $dst, $src, $shift",
                (SUBWrs GPR32:$dst, WZR, GPR32:$src, arith_shift:$shift)>;
def : InstAlias<"neg $dst, $src, $shift",
                (SUBXrs GPR64:$dst, XZR, GPR64:$src, arith_shift:$shift)>;

// Because of the immediate format for add/sub-imm instructions, the
// expression (add x, -1) must be transformed to (SUB{W,X}ri x, 1).
//  These patterns capture that transformation.
let AddedComplexity = 1 in {
def : Pat<(ARM64add_flag GPR32:$Rn, neg_addsub_shifted_imm32:$imm),
          (SUBSWri GPR32:$Rn, neg_addsub_shifted_imm32:$imm)>;
def : Pat<(ARM64add_flag GPR64:$Rn, neg_addsub_shifted_imm64:$imm),
          (SUBSXri GPR64:$Rn, neg_addsub_shifted_imm64:$imm)>;
def : Pat<(ARM64sub_flag GPR32:$Rn, neg_addsub_shifted_imm32:$imm),
          (ADDSWri GPR32:$Rn, neg_addsub_shifted_imm32:$imm)>;
def : Pat<(ARM64sub_flag GPR64:$Rn, neg_addsub_shifted_imm64:$imm),
          (ADDSXri GPR64:$Rn, neg_addsub_shifted_imm64:$imm)>;
}

def : InstAlias<"negs $dst, $src", (SUBSWrs GPR32:$dst, WZR, GPR32:$src, 0)>;
def : InstAlias<"negs $dst, $src", (SUBSXrs GPR64:$dst, XZR, GPR64:$src, 0)>;
def : InstAlias<"negs $dst, $src, $shift",
                (SUBSWrs GPR32:$dst, WZR, GPR32:$src, arith_shift:$shift)>;
def : InstAlias<"negs $dst, $src, $shift",
                (SUBSXrs GPR64:$dst, XZR, GPR64:$src, arith_shift:$shift)>;

// Unsigned/Signed divide
defm UDIV : Div<0, "udiv", udiv>;
defm SDIV : Div<1, "sdiv", sdiv>;
let isCodeGenOnly = 1 in {
defm UDIV_Int : Div<0, "udiv", int_arm64_udiv>;
defm SDIV_Int : Div<1, "sdiv", int_arm64_sdiv>;
}

// Variable shift
defm ASRV : Shift<0b10, "asrv", sra>;
defm LSLV : Shift<0b00, "lslv", shl>;
defm LSRV : Shift<0b01, "lsrv", srl>;
defm RORV : Shift<0b11, "rorv", rotr>;

def : ShiftAlias<"asr", ASRVWr, GPR32>;
def : ShiftAlias<"asr", ASRVXr, GPR64>;
def : ShiftAlias<"lsl", LSLVWr, GPR32>;
def : ShiftAlias<"lsl", LSLVXr, GPR64>;
def : ShiftAlias<"lsr", LSRVWr, GPR32>;
def : ShiftAlias<"lsr", LSRVXr, GPR64>;
def : ShiftAlias<"ror", RORVWr, GPR32>;
def : ShiftAlias<"ror", RORVXr, GPR64>;

// Multiply-add
let AddedComplexity = 7 in {
defm MADD : MulAccum<0, "madd", add>;
defm MSUB : MulAccum<1, "msub", sub>;

def : Pat<(i32 (mul GPR32:$Rn, GPR32:$Rm)),
          (MADDWrrr GPR32:$Rn, GPR32:$Rm, WZR)>;
def : Pat<(i64 (mul GPR64:$Rn, GPR64:$Rm)),
          (MADDXrrr GPR64:$Rn, GPR64:$Rm, XZR)>;

def : Pat<(i32 (ineg (mul GPR32:$Rn, GPR32:$Rm))),
          (MSUBWrrr GPR32:$Rn, GPR32:$Rm, WZR)>;
def : Pat<(i64 (ineg (mul GPR64:$Rn, GPR64:$Rm))),
          (MSUBXrrr GPR64:$Rn, GPR64:$Rm, XZR)>;
} // AddedComplexity = 7

let AddedComplexity = 5 in {
def SMADDLrrr : WideMulAccum<0, 0b001, "smaddl", add, sext>;
def SMSUBLrrr : WideMulAccum<1, 0b001, "smsubl", sub, sext>;
def UMADDLrrr : WideMulAccum<0, 0b101, "umaddl", add, zext>;
def UMSUBLrrr : WideMulAccum<1, 0b101, "umsubl", sub, zext>;

def : Pat<(i64 (mul (sext GPR32:$Rn), (sext GPR32:$Rm))),
          (SMADDLrrr GPR32:$Rn, GPR32:$Rm, XZR)>;
def : Pat<(i64 (mul (zext GPR32:$Rn), (zext GPR32:$Rm))),
          (UMADDLrrr GPR32:$Rn, GPR32:$Rm, XZR)>;

def : Pat<(i64 (ineg (mul (sext GPR32:$Rn), (sext GPR32:$Rm)))),
          (SMSUBLrrr GPR32:$Rn, GPR32:$Rm, XZR)>;
def : Pat<(i64 (ineg (mul (zext GPR32:$Rn), (zext GPR32:$Rm)))),
          (UMSUBLrrr GPR32:$Rn, GPR32:$Rm, XZR)>;
} // AddedComplexity = 5

def : MulAccumWAlias<"mul", MADDWrrr>;
def : MulAccumXAlias<"mul", MADDXrrr>;
def : MulAccumWAlias<"mneg", MSUBWrrr>;
def : MulAccumXAlias<"mneg", MSUBXrrr>;
def : WideMulAccumAlias<"smull", SMADDLrrr>;
def : WideMulAccumAlias<"smnegl", SMSUBLrrr>;
def : WideMulAccumAlias<"umull", UMADDLrrr>;
def : WideMulAccumAlias<"umnegl", UMSUBLrrr>;

// Multiply-high
def SMULHrr : MulHi<0b010, "smulh", mulhs>;
def UMULHrr : MulHi<0b110, "umulh", mulhu>;

// CRC32
def CRC32Brr : BaseCRC32<0, 0b00, 0, GPR32, int_arm64_crc32b, "crc32b">;
def CRC32Hrr : BaseCRC32<0, 0b01, 0, GPR32, int_arm64_crc32h, "crc32h">;
def CRC32Wrr : BaseCRC32<0, 0b10, 0, GPR32, int_arm64_crc32w, "crc32w">;
def CRC32Xrr : BaseCRC32<1, 0b11, 0, GPR64, int_arm64_crc32x, "crc32x">;

def CRC32CBrr : BaseCRC32<0, 0b00, 1, GPR32, int_arm64_crc32cb, "crc32cb">;
def CRC32CHrr : BaseCRC32<0, 0b01, 1, GPR32, int_arm64_crc32ch, "crc32ch">;
def CRC32CWrr : BaseCRC32<0, 0b10, 1, GPR32, int_arm64_crc32cw, "crc32cw">;
def CRC32CXrr : BaseCRC32<1, 0b11, 1, GPR64, int_arm64_crc32cx, "crc32cx">;


//===----------------------------------------------------------------------===//
// Logical instructions.
//===----------------------------------------------------------------------===//

// (immediate)
defm ANDS : LogicalImmS<0b11, "ands", ARM64and_flag>;
defm AND  : LogicalImm<0b00, "and", and>;
defm EOR  : LogicalImm<0b10, "eor", xor>;
defm ORR  : LogicalImm<0b01, "orr", or>;

def : InstAlias<"mov $dst, $imm", (ORRWri GPR32sp:$dst, WZR,
                                          logical_imm32:$imm)>;
def : InstAlias<"mov $dst, $imm", (ORRXri GPR64sp:$dst, XZR,
                                          logical_imm64:$imm)>;


// (register)
defm ANDS : LogicalRegS<0b11, 0, "ands">;
defm BICS : LogicalRegS<0b11, 1, "bics">;
defm AND  : LogicalReg<0b00, 0, "and", and>;
defm BIC  : LogicalReg<0b00, 1, "bic",
                       BinOpFrag<(and node:$LHS, (not node:$RHS))>>;
defm EON  : LogicalReg<0b10, 1, "eon",
                       BinOpFrag<(xor node:$LHS, (not node:$RHS))>>;
defm EOR  : LogicalReg<0b10, 0, "eor", xor>;
defm ORN  : LogicalReg<0b01, 1, "orn",
                       BinOpFrag<(or node:$LHS, (not node:$RHS))>>;
defm ORR  : LogicalReg<0b01, 0, "orr", or>;

def : InstAlias<"tst $src1, $src2",
                (ANDSWri WZR, GPR32:$src1, logical_imm32:$src2)>;
def : InstAlias<"tst $src1, $src2",
                (ANDSXri XZR, GPR64:$src1, logical_imm64:$src2)>;

def : InstAlias<"tst $src1, $src2",
                (ANDSWrs WZR, GPR32:$src1, GPR32:$src2, 0)>;
def : InstAlias<"tst $src1, $src2",
                (ANDSXrs XZR, GPR64:$src1, GPR64:$src2, 0)>;

def : InstAlias<"tst $src1, $src2, $sh",
                (ANDSWrs WZR, GPR32:$src1, GPR32:$src2, logical_shift:$sh)>;
def : InstAlias<"tst $src1, $src2, $sh",
                (ANDSXrs XZR, GPR64:$src1, GPR64:$src2, logical_shift:$sh)>;

def : InstAlias<"mvn $Wd, $Wm",
                (ORNWrs GPR32:$Wd, WZR, GPR32:$Wm, 0)>;
def : InstAlias<"mvn $Xd, $Xm",
                (ORNXrs GPR64:$Xd, XZR, GPR64:$Xm, 0)>;

def : InstAlias<"mvn $Wd, $Wm, $sh",
                (ORNWrs GPR32:$Wd, WZR, GPR32:$Wm, logical_shift:$sh)>;
def : InstAlias<"mvn $Xd, $Xm, $sh",
                (ORNXrs GPR64:$Xd, XZR, GPR64:$Xm, logical_shift:$sh)>;

def : Pat<(not GPR32:$Wm), (ORNWrr WZR, GPR32:$Wm)>;
def : Pat<(not GPR64:$Xm), (ORNXrr XZR, GPR64:$Xm)>;


//===----------------------------------------------------------------------===//
// One operand data processing instructions.
//===----------------------------------------------------------------------===//

defm CLS    : OneOperandData<0b101, "cls">;
defm CLZ    : OneOperandData<0b100, "clz", ctlz>;
defm RBIT   : OneOperandData<0b000, "rbit">;
def  REV16Wr : OneWRegData<0b001, "rev16",
                                  UnOpFrag<(rotr (bswap node:$LHS), (i64 16))>>;
def  REV16Xr : OneXRegData<0b001, "rev16", null_frag>;

def : Pat<(cttz GPR32:$Rn),
          (CLZWr (RBITWr GPR32:$Rn))>;
def : Pat<(cttz GPR64:$Rn),
          (CLZXr (RBITXr GPR64:$Rn))>;
def : Pat<(ctlz (or (shl (xor (sra GPR32:$Rn, (i64 31)), GPR32:$Rn), (i64 1)),
                (i32 1))),
          (CLSWr GPR32:$Rn)>;
def : Pat<(ctlz (or (shl (xor (sra GPR64:$Rn, (i64 63)), GPR64:$Rn), (i64 1)),
                (i64 1))),
          (CLSXr GPR64:$Rn)>;

// Unlike the other one operand instructions, the instructions with the "rev"
// mnemonic do *not* just different in the size bit, but actually use different
// opcode bits for the different sizes.
def REVWr   : OneWRegData<0b010, "rev", bswap>;
def REVXr   : OneXRegData<0b011, "rev", bswap>;
def REV32Xr : OneXRegData<0b010, "rev32",
                                 UnOpFrag<(rotr (bswap node:$LHS), (i64 32))>>;

// The bswap commutes with the rotr so we want a pattern for both possible
// orders.
def : Pat<(bswap (rotr GPR32:$Rn, (i64 16))), (REV16Wr GPR32:$Rn)>;
def : Pat<(bswap (rotr GPR64:$Rn, (i64 32))), (REV32Xr GPR64:$Rn)>;

//===----------------------------------------------------------------------===//
// Bitfield immediate extraction instruction.
//===----------------------------------------------------------------------===//
let neverHasSideEffects = 1 in
defm EXTR : ExtractImm<"extr">;
def : InstAlias<"ror $dst, $src, $shift",
            (EXTRWrri GPR32:$dst, GPR32:$src, GPR32:$src, imm0_31:$shift)>;
def : InstAlias<"ror $dst, $src, $shift",
            (EXTRXrri GPR64:$dst, GPR64:$src, GPR64:$src, imm0_63:$shift)>;

def : Pat<(rotr GPR32:$Rn, (i64 imm0_31:$imm)),
          (EXTRWrri GPR32:$Rn, GPR32:$Rn, imm0_31:$imm)>;
def : Pat<(rotr GPR64:$Rn, (i64 imm0_63:$imm)),
          (EXTRXrri GPR64:$Rn, GPR64:$Rn, imm0_63:$imm)>;

//===----------------------------------------------------------------------===//
// Other bitfield immediate instructions.
//===----------------------------------------------------------------------===//
let neverHasSideEffects = 1 in {
defm BFM  : BitfieldImmWith2RegArgs<0b01, "bfm">;
defm SBFM : BitfieldImm<0b00, "sbfm">;
defm UBFM : BitfieldImm<0b10, "ubfm">;
}

def i32shift_a : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = (32 - N->getZExtValue()) & 0x1f;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

def i32shift_b : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 31 - N->getZExtValue();
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

// min(7, 31 - shift_amt)
def i32shift_sext_i8 : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 31 - N->getZExtValue();
  enc = enc > 7 ? 7 : enc;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

// min(15, 31 - shift_amt)
def i32shift_sext_i16 : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 31 - N->getZExtValue();
  enc = enc > 15 ? 15 : enc;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

def i64shift_a : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = (64 - N->getZExtValue()) & 0x3f;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

def i64shift_b : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 63 - N->getZExtValue();
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

// min(7, 63 - shift_amt)
def i64shift_sext_i8 : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 63 - N->getZExtValue();
  enc = enc > 7 ? 7 : enc;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

// min(15, 63 - shift_amt)
def i64shift_sext_i16 : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 63 - N->getZExtValue();
  enc = enc > 15 ? 15 : enc;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

// min(31, 63 - shift_amt)
def i64shift_sext_i32 : Operand<i64>, SDNodeXForm<imm, [{
  uint64_t enc = 63 - N->getZExtValue();
  enc = enc > 31 ? 31 : enc;
  return CurDAG->getTargetConstant(enc, MVT::i64);
}]>;

def : Pat<(shl GPR32:$Rn, (i64 imm0_31:$imm)),
          (UBFMWri GPR32:$Rn, (i64 (i32shift_a imm0_31:$imm)),
                              (i64 (i32shift_b imm0_31:$imm)))>;
def : Pat<(shl GPR64:$Rn, (i64 imm0_63:$imm)),
          (UBFMXri GPR64:$Rn, (i64 (i64shift_a imm0_63:$imm)),
                              (i64 (i64shift_b imm0_63:$imm)))>;

let AddedComplexity = 10 in {
def : Pat<(sra GPR32:$Rn, (i64 imm0_31:$imm)),
          (SBFMWri GPR32:$Rn, imm0_31:$imm, 31)>;
def : Pat<(sra GPR64:$Rn, (i64 imm0_63:$imm)),
          (SBFMXri GPR64:$Rn, imm0_63:$imm, 63)>;
}

def : InstAlias<"asr $dst, $src, $shift",
                (SBFMWri GPR32:$dst, GPR32:$src, imm0_31:$shift, 31)>;
def : InstAlias<"asr $dst, $src, $shift",
                (SBFMXri GPR64:$dst, GPR64:$src, imm0_63:$shift, 63)>;
def : InstAlias<"sxtb $dst, $src", (SBFMWri GPR32:$dst, GPR32:$src, 0, 7)>;
def : InstAlias<"sxtb $dst, $src", (SBFMXri GPR64:$dst, GPR64:$src, 0, 7)>;
def : InstAlias<"sxth $dst, $src", (SBFMWri GPR32:$dst, GPR32:$src, 0, 15)>;
def : InstAlias<"sxth $dst, $src", (SBFMXri GPR64:$dst, GPR64:$src, 0, 15)>;
def : InstAlias<"sxtw $dst, $src", (SBFMXri GPR64:$dst, GPR64:$src, 0, 31)>;

def : Pat<(srl GPR32:$Rn, (i64 imm0_31:$imm)),
          (UBFMWri GPR32:$Rn, imm0_31:$imm, 31)>;
def : Pat<(srl GPR64:$Rn, (i64 imm0_63:$imm)),
          (UBFMXri GPR64:$Rn, imm0_63:$imm, 63)>;

def : InstAlias<"lsr $dst, $src, $shift",
                (UBFMWri GPR32:$dst, GPR32:$src, imm0_31:$shift, 31)>;
def : InstAlias<"lsr $dst, $src, $shift",
                (UBFMXri GPR64:$dst, GPR64:$src, imm0_63:$shift, 63)>;
def : InstAlias<"uxtb $dst, $src", (UBFMWri GPR32:$dst, GPR32:$src, 0, 7)>;
def : InstAlias<"uxtb $dst, $src", (UBFMXri GPR64:$dst, GPR64:$src, 0, 7)>;
def : InstAlias<"uxth $dst, $src", (UBFMWri GPR32:$dst, GPR32:$src, 0, 15)>;
def : InstAlias<"uxth $dst, $src", (UBFMXri GPR64:$dst, GPR64:$src, 0, 15)>;
def : InstAlias<"uxtw $dst, $src", (UBFMXri GPR64:$dst, GPR64:$src, 0, 31)>;

//===----------------------------------------------------------------------===//
// Conditionally set flags instructions.
//===----------------------------------------------------------------------===//
defm CCMN : CondSetFlagsImm<0, "ccmn">;
defm CCMP : CondSetFlagsImm<1, "ccmp">;

defm CCMN : CondSetFlagsReg<0, "ccmn">;
defm CCMP : CondSetFlagsReg<1, "ccmp">;

//===----------------------------------------------------------------------===//
// Conditional select instructions.
//===----------------------------------------------------------------------===//
defm CSEL  : CondSelect<0, 0b00, "csel">;

def inc : PatFrag<(ops node:$in), (add node:$in, 1)>;
defm CSINC : CondSelectOp<0, 0b01, "csinc", inc>;
defm CSINV : CondSelectOp<1, 0b00, "csinv", not>;
defm CSNEG : CondSelectOp<1, 0b01, "csneg", ineg>;

def : Pat<(ARM64csinv GPR32:$tval, GPR32:$fval, (i32 imm:$cc), CPSR),
          (CSINVWr GPR32:$tval, GPR32:$fval, (i32 imm:$cc))>;
def : Pat<(ARM64csinv GPR64:$tval, GPR64:$fval, (i32 imm:$cc), CPSR),
          (CSINVXr GPR64:$tval, GPR64:$fval, (i32 imm:$cc))>;
def : Pat<(ARM64csneg GPR32:$tval, GPR32:$fval, (i32 imm:$cc), CPSR),
          (CSNEGWr GPR32:$tval, GPR32:$fval, (i32 imm:$cc))>;
def : Pat<(ARM64csneg GPR64:$tval, GPR64:$fval, (i32 imm:$cc), CPSR),
          (CSNEGXr GPR64:$tval, GPR64:$fval, (i32 imm:$cc))>;
def : Pat<(ARM64csinc GPR32:$tval, GPR32:$fval, (i32 imm:$cc), CPSR),
          (CSINCWr GPR32:$tval, GPR32:$fval, (i32 imm:$cc))>;
def : Pat<(ARM64csinc GPR64:$tval, GPR64:$fval, (i32 imm:$cc), CPSR),
          (CSINCXr GPR64:$tval, GPR64:$fval, (i32 imm:$cc))>;

def : Pat<(ARM64csel (i32 0), (i32 1), (i32 imm:$cc), CPSR),
          (CSINCWr WZR, WZR, (i32 imm:$cc))>;
def : Pat<(ARM64csel (i64 0), (i64 1), (i32 imm:$cc), CPSR),
          (CSINCXr XZR, XZR, (i32 imm:$cc))>;
def : Pat<(ARM64csel (i32 0), (i32 -1), (i32 imm:$cc), CPSR),
          (CSINVWr WZR, WZR, (i32 imm:$cc))>;
def : Pat<(ARM64csel (i64 0), (i64 -1), (i32 imm:$cc), CPSR),
          (CSINVXr XZR, XZR, (i32 imm:$cc))>;

// The inverse of the condition code from the alias instruction is what is used
// in the aliased instruction. The parser all ready inverts the condition code
// for these aliases.
// FIXME: Is this the correct way to handle these aliases?
def : InstAlias<"cset $dst, $cc", (CSINCWr GPR32:$dst, WZR, WZR, ccode:$cc)>;
def : InstAlias<"cset $dst, $cc", (CSINCXr GPR64:$dst, XZR, XZR, ccode:$cc)>;

def : InstAlias<"csetm $dst, $cc", (CSINVWr GPR32:$dst, WZR, WZR, ccode:$cc)>;
def : InstAlias<"csetm $dst, $cc", (CSINVXr GPR64:$dst, XZR, XZR, ccode:$cc)>;

def : InstAlias<"cinc $dst, $src, $cc",
                (CSINCWr GPR32:$dst, GPR32:$src, GPR32:$src, ccode:$cc)>;
def : InstAlias<"cinc $dst, $src, $cc",
                (CSINCXr GPR64:$dst, GPR64:$src, GPR64:$src, ccode:$cc)>;

def : InstAlias<"cinv $dst, $src, $cc",
                (CSINVWr GPR32:$dst, GPR32:$src, GPR32:$src, ccode:$cc)>;
def : InstAlias<"cinv $dst, $src, $cc",
                (CSINVXr GPR64:$dst, GPR64:$src, GPR64:$src, ccode:$cc)>;

def : InstAlias<"cneg $dst, $src, $cc",
                (CSNEGWr GPR32:$dst, GPR32:$src, GPR32:$src, ccode:$cc)>;
def : InstAlias<"cneg $dst, $src, $cc",
                (CSNEGXr GPR64:$dst, GPR64:$src, GPR64:$src, ccode:$cc)>;

//===----------------------------------------------------------------------===//
// PC-relative instructions.
//===----------------------------------------------------------------------===//
let isReMaterializable = 1 in {
let neverHasSideEffects = 1, mayStore = 0, mayLoad = 0 in {
def ADR  : ADRI<0, "adr", adrlabel, []>;
} // neverHasSideEffects = 1

def ADRP : ADRI<1, "adrp", adrplabel,
                [(set GPR64:$Xd, (ARM64adrp tglobaladdr:$label))]>;
} // isReMaterializable = 1

// page address of a constant pool entry, block address
def : Pat<(ARM64adrp tconstpool:$cp), (ADRP tconstpool:$cp)>;
def : Pat<(ARM64adrp tblockaddress:$cp), (ADRP tblockaddress:$cp)>;

//===----------------------------------------------------------------------===//
// Unconditional branch (register) instructions.
//===----------------------------------------------------------------------===//

let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
def RET  : BranchReg<0b0010, "ret", []>;
def DRPS : SpecialReturn<0b0101, "drps">;
def ERET : SpecialReturn<0b0100, "eret">;
} // isReturn = 1, isTerminator = 1, isBarrier = 1

// Default to the LR register.
def : InstAlias<"ret", (RET LR)>;

let isCall = 1, Defs = [LR], Uses = [SP] in {
def BLR : BranchReg<0b0001, "blr", [(ARM64call GPR64:$Rn)]>;
} // isCall

let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
def BR  : BranchReg<0b0000, "br", [(brind GPR64:$Rn)]>;
} // isBranch, isTerminator, isBarrier, isIndirectBranch

// Create a separate pseudo-instruction for codegen to use so that we don't
// flag lr as used in every function. It'll be restored before the RET by the
// epilogue if it's legitimately used.
def RET_ReallyLR : Pseudo<(outs), (ins), [(ARM64retflag)]> {
  let isTerminator = 1;
  let isBarrier = 1;
  let isReturn = 1;
}

// This is a directive-like pseudo-instruction. The purpose is to insert an
// R_AARCH64_TLSDESC_CALL relocation at the offset of the following instruction
// (which in the usual case is a BLR).
let hasSideEffects = 1 in
def TLSDESCCALL : Pseudo<(outs), (ins i64imm:$sym), []> {
  let AsmString = ".tlsdesccall $sym";
}

// Pseudo-instruction representing a BLR with attached TLSDESC relocation. It
// gets expanded to two MCInsts during lowering.
let isCall = 1, Defs = [LR] in
def TLSDESC_BLR
    : Pseudo<(outs), (ins GPR64:$dest, i64imm:$sym),
             [(ARM64tlsdesc_call GPR64:$dest, tglobaltlsaddr:$sym)]>;

def : Pat<(ARM64tlsdesc_call GPR64:$dest, texternalsym:$sym),
          (TLSDESC_BLR GPR64:$dest, texternalsym:$sym)>;
//===----------------------------------------------------------------------===//
// Conditional branch (immediate) instruction.
//===----------------------------------------------------------------------===//
def Bcc : BranchCond;

//===----------------------------------------------------------------------===//
// Compare-and-branch instructions.
//===----------------------------------------------------------------------===//
defm CBZ  : CmpBranch<0, "cbz", ARM64cbz>;
defm CBNZ : CmpBranch<1, "cbnz", ARM64cbnz>;

//===----------------------------------------------------------------------===//
// Test-bit-and-branch instructions.
//===----------------------------------------------------------------------===//
def TBZ  : TestBranch<0, "tbz", ARM64tbz>;
def TBNZ : TestBranch<1, "tbnz", ARM64tbnz>;

//===----------------------------------------------------------------------===//
// Unconditional branch (immediate) instructions.
//===----------------------------------------------------------------------===//
let isBranch = 1, isTerminator = 1, isBarrier = 1 in {
def B  : BranchImm<0, "b", [(br bb:$addr)]>;
} // isBranch, isTerminator, isBarrier

let isCall = 1, Defs = [LR], Uses = [SP] in {
def BL : CallImm<1, "bl", [(ARM64call tglobaladdr:$addr)]>;
} // isCall
def : Pat<(ARM64call texternalsym:$func), (BL texternalsym:$func)>;

//===----------------------------------------------------------------------===//
// Exception generation instructions.
//===----------------------------------------------------------------------===//
def BRK   : ExceptionGeneration<0b001, 0b00, "brk">;
def DCPS1 : ExceptionGeneration<0b101, 0b01, "dcps1">;
def DCPS2 : ExceptionGeneration<0b101, 0b10, "dcps2">;
def DCPS3 : ExceptionGeneration<0b101, 0b11, "dcps3">;
def HLT   : ExceptionGeneration<0b010, 0b00, "hlt">;
def HVC   : ExceptionGeneration<0b000, 0b10, "hvc">;
def SMC   : ExceptionGeneration<0b000, 0b11, "smc">;
def SVC   : ExceptionGeneration<0b000, 0b01, "svc">;

// DCPSn defaults to an immediate operand of zero if unspecified.
def : InstAlias<"dcps1", (DCPS1 0)>;
def : InstAlias<"dcps2", (DCPS2 0)>;
def : InstAlias<"dcps3", (DCPS3 0)>;

//===----------------------------------------------------------------------===//
// Load instructions.
//===----------------------------------------------------------------------===//

// Pair (indexed, offset)
def LDPWi : LoadPairOffset<0b00, 0, GPR32, am_indexed32simm7, "ldp">;
def LDPXi : LoadPairOffset<0b10, 0, GPR64, am_indexed64simm7, "ldp">;
def LDPSi : LoadPairOffset<0b00, 1, FPR32, am_indexed32simm7, "ldp">;
def LDPDi : LoadPairOffset<0b01, 1, FPR64, am_indexed64simm7, "ldp">;
def LDPQi : LoadPairOffset<0b10, 1, FPR128, am_indexed128simm7, "ldp">;

def LDPSWi : LoadPairOffset<0b01, 0, GPR64, am_indexed32simm7, "ldpsw">;

// Pair (pre-indexed)
def LDPWpre : LoadPairPreIdx<0b00, 0, GPR32, am_indexed32simm7_wb, "ldp">;
def LDPXpre : LoadPairPreIdx<0b10, 0, GPR64, am_indexed64simm7_wb, "ldp">;
def LDPSpre : LoadPairPreIdx<0b00, 1, FPR32, am_indexed32simm7_wb, "ldp">;
def LDPDpre : LoadPairPreIdx<0b01, 1, FPR64, am_indexed64simm7_wb, "ldp">;
def LDPQpre : LoadPairPreIdx<0b10, 1, FPR128, am_indexed128simm7_wb, "ldp">;

def LDPSWpre : LoadPairPreIdx<0b01, 0, GPR64, am_indexed32simm7_wb, "ldpsw">;

// Pair (post-indexed)
def LDPWpost : LoadPairPostIdx<0b00, 0, GPR32, simm7s4, "ldp">;
def LDPXpost : LoadPairPostIdx<0b10, 0, GPR64, simm7s8, "ldp">;
def LDPSpost : LoadPairPostIdx<0b00, 1, FPR32, simm7s4, "ldp">;
def LDPDpost : LoadPairPostIdx<0b01, 1, FPR64, simm7s8, "ldp">;
def LDPQpost : LoadPairPostIdx<0b10, 1, FPR128, simm7s16, "ldp">;

def LDPSWpost : LoadPairPostIdx<0b01, 0, GPR64, simm7s4, "ldpsw">;


// Pair (no allocate)
def LDNPWi : LoadPairNoAlloc<0b00, 0, GPR32, am_indexed32simm7, "ldnp">;
def LDNPXi : LoadPairNoAlloc<0b10, 0, GPR64, am_indexed64simm7, "ldnp">;
def LDNPSi : LoadPairNoAlloc<0b00, 1, FPR32, am_indexed32simm7, "ldnp">;
def LDNPDi : LoadPairNoAlloc<0b01, 1, FPR64, am_indexed64simm7, "ldnp">;
def LDNPQi : LoadPairNoAlloc<0b10, 1, FPR128, am_indexed128simm7, "ldnp">;

//---
// (register offset)
//---

let AddedComplexity = 10 in {
// Integer
def LDRBBro : Load8RO<0b00,  0, 0b01, GPR32, "ldrb",
                      [(set GPR32:$Rt, (zextloadi8 ro_indexed8:$addr))]>;
def LDRHHro : Load16RO<0b01, 0, 0b01, GPR32, "ldrh",
                      [(set GPR32:$Rt, (zextloadi16 ro_indexed16:$addr))]>;
def LDRWro  : Load32RO<0b10,   0, 0b01, GPR32, "ldr",
                      [(set GPR32:$Rt, (load ro_indexed32:$addr))]>;
def LDRXro  : Load64RO<0b11,   0, 0b01, GPR64, "ldr",
                      [(set GPR64:$Rt, (load ro_indexed64:$addr))]>;

// Floating-point
def LDRBro : Load8RO<0b00,   1, 0b01, FPR8,   "ldr",
                      [(set FPR8:$Rt, (load ro_indexed8:$addr))]>;
def LDRHro : Load16RO<0b01,  1, 0b01, FPR16,  "ldr",
                      [(set (f16 FPR16:$Rt), (load ro_indexed16:$addr))]>;
def LDRSro : Load32RO<0b10,    1, 0b01, FPR32,  "ldr",
                      [(set (f32 FPR32:$Rt), (load ro_indexed32:$addr))]>;
def LDRDro : Load64RO<0b11,    1, 0b01, FPR64,  "ldr",
                      [(set (f64 FPR64:$Rt), (load ro_indexed64:$addr))]>;
def LDRQro : Load128RO<0b00,    1, 0b11, FPR128, "ldr", []> {
  let mayLoad = 1;
}

// For regular load, we do not have any alignment requirement.
// Thus, it is safe to directly map the vector loads with interesting
// addressing modes.
// FIXME: We could do the same for bitconvert to floating point vectors.
def : Pat <(v8i8 (scalar_to_vector (i32 (extloadi8 ro_indexed8:$addr)))),
           (INSERT_SUBREG (v8i8 (IMPLICIT_DEF)),
                          (LDRBro ro_indexed8:$addr), bsub)>;
def : Pat <(v16i8 (scalar_to_vector (i32 (extloadi8 ro_indexed8:$addr)))),
           (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
                          (LDRBro ro_indexed8:$addr), bsub)>;
def : Pat <(v4i16 (scalar_to_vector (i32 (extloadi16 ro_indexed16:$addr)))),
           (INSERT_SUBREG (v4i16 (IMPLICIT_DEF)),
                          (LDRHro ro_indexed16:$addr), hsub)>;
def : Pat <(v8i16 (scalar_to_vector (i32 (extloadi16 ro_indexed16:$addr)))),
           (INSERT_SUBREG (v8i16 (IMPLICIT_DEF)),
                          (LDRHro ro_indexed16:$addr), hsub)>;
def : Pat <(v2i32 (scalar_to_vector (i32 (load ro_indexed32:$addr)))),
           (INSERT_SUBREG (v2i32 (IMPLICIT_DEF)),
                          (LDRSro ro_indexed32:$addr), ssub)>;
def : Pat <(v4i32 (scalar_to_vector (i32 (load ro_indexed32:$addr)))),
           (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)),
                          (LDRSro ro_indexed32:$addr), ssub)>;
def : Pat <(v1i64 (scalar_to_vector (i64 (load ro_indexed64:$addr)))),
           (LDRDro ro_indexed64:$addr)>;
def : Pat <(v2i64 (scalar_to_vector (i64 (load ro_indexed64:$addr)))),
           (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)),
                          (LDRDro ro_indexed64:$addr), dsub)>;

// Match all load 64 bits width whose type is compatible with FPR64
def : Pat<(v2f32 (load ro_indexed64:$addr)), (LDRDro ro_indexed64:$addr)>;
def : Pat<(v1f64 (load ro_indexed64:$addr)), (LDRDro ro_indexed64:$addr)>;
def : Pat<(v8i8 (load ro_indexed64:$addr)), (LDRDro ro_indexed64:$addr)>;
def : Pat<(v4i16 (load ro_indexed64:$addr)), (LDRDro ro_indexed64:$addr)>;
def : Pat<(v2i32 (load ro_indexed64:$addr)), (LDRDro ro_indexed64:$addr)>;
def : Pat<(v1i64 (load ro_indexed64:$addr)), (LDRDro ro_indexed64:$addr)>;

// Match all load 128 bits width whose type is compatible with FPR128
def : Pat<(v4f32 (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;
def : Pat<(v2f64 (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;
def : Pat<(v16i8 (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;
def : Pat<(v8i16 (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;
def : Pat<(v4i32 (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;
def : Pat<(v2i64 (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;
def : Pat<(f128  (load ro_indexed128:$addr)), (LDRQro ro_indexed128:$addr)>;

// Load sign-extended half-word
def LDRSHWro : Load16RO<0b01, 0, 0b11, GPR32, "ldrsh",
                      [(set GPR32:$Rt, (sextloadi16 ro_indexed16:$addr))]>;
def LDRSHXro : Load16RO<0b01, 0, 0b10, GPR64, "ldrsh",
                      [(set GPR64:$Rt, (sextloadi16 ro_indexed16:$addr))]>;

// Load sign-extended byte
def LDRSBWro : Load8RO<0b00, 0, 0b11, GPR32, "ldrsb",
                      [(set GPR32:$Rt, (sextloadi8 ro_indexed8:$addr))]>;
def LDRSBXro : Load8RO<0b00, 0, 0b10, GPR64, "ldrsb",
                      [(set GPR64:$Rt, (sextloadi8 ro_indexed8:$addr))]>;

// Load sign-extended word
def LDRSWro  : Load32RO<0b10, 0, 0b10, GPR64, "ldrsw",
                      [(set GPR64:$Rt, (sextloadi32 ro_indexed32:$addr))]>;

// Pre-fetch.
def PRFMro : PrefetchRO<0b11, 0, 0b10, "prfm",
                        [(ARM64Prefetch imm:$Rt, ro_indexed64:$addr)]>;

// zextload -> i64
def : Pat<(i64 (zextloadi8 ro_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBro ro_indexed8:$addr), sub_32)>;
def : Pat<(i64 (zextloadi16 ro_indexed16:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRHHro ro_indexed16:$addr), sub_32)>;

// zextloadi1 -> zextloadi8
def : Pat<(i32 (zextloadi1 ro_indexed8:$addr)), (LDRBBro ro_indexed8:$addr)>;
def : Pat<(i64 (zextloadi1 ro_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBro ro_indexed8:$addr), sub_32)>;

// extload -> zextload
def : Pat<(i32 (extloadi16 ro_indexed16:$addr)), (LDRHHro ro_indexed16:$addr)>;
def : Pat<(i32 (extloadi8 ro_indexed8:$addr)), (LDRBBro ro_indexed8:$addr)>;
def : Pat<(i32 (extloadi1 ro_indexed8:$addr)), (LDRBBro ro_indexed8:$addr)>;
def : Pat<(i64 (extloadi32 ro_indexed32:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRWro ro_indexed32:$addr), sub_32)>;
def : Pat<(i64 (extloadi16 ro_indexed16:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRHHro ro_indexed16:$addr), sub_32)>;
def : Pat<(i64 (extloadi8 ro_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBro ro_indexed8:$addr), sub_32)>;
def : Pat<(i64 (extloadi1 ro_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBro ro_indexed8:$addr), sub_32)>;

} // AddedComplexity = 10

//---
// (unsigned immediate)
//---
def LDRXui : LoadUI<0b11, 0, 0b01, GPR64, am_indexed64, "ldr",
                    [(set GPR64:$Rt, (load am_indexed64:$addr))]>;
def LDRWui : LoadUI<0b10, 0, 0b01, GPR32, am_indexed32, "ldr",
                    [(set GPR32:$Rt, (load am_indexed32:$addr))]>;
def LDRBui : LoadUI<0b00, 1, 0b01, FPR8, am_indexed8, "ldr",
                    [(set FPR8:$Rt, (load am_indexed8:$addr))]>;
def LDRHui : LoadUI<0b01, 1, 0b01, FPR16, am_indexed16, "ldr",
                    [(set (f16 FPR16:$Rt), (load am_indexed16:$addr))]>;
def LDRSui : LoadUI<0b10, 1, 0b01, FPR32, am_indexed32, "ldr",
                    [(set (f32 FPR32:$Rt), (load am_indexed32:$addr))]>;
def LDRDui : LoadUI<0b11, 1, 0b01, FPR64, am_indexed64, "ldr",
                    [(set (f64 FPR64:$Rt), (load am_indexed64:$addr))]>;
def LDRQui : LoadUI<0b00, 1, 0b11, FPR128, am_indexed128, "ldr",
                    [(set (f128 FPR128:$Rt), (load am_indexed128:$addr))]>;

// For regular load, we do not have any alignment requirement.
// Thus, it is safe to directly map the vector loads with interesting
// addressing modes.
// FIXME: We could do the same for bitconvert to floating point vectors.
def : Pat <(v8i8 (scalar_to_vector (i32 (extloadi8 am_indexed8:$addr)))),
           (INSERT_SUBREG (v8i8 (IMPLICIT_DEF)),
                          (LDRBui am_indexed8:$addr), bsub)>;
def : Pat <(v16i8 (scalar_to_vector (i32 (extloadi8 am_indexed8:$addr)))),
           (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
                          (LDRBui am_indexed8:$addr), bsub)>;
def : Pat <(v4i16 (scalar_to_vector (i32 (extloadi16 am_indexed16:$addr)))),
           (INSERT_SUBREG (v4i16 (IMPLICIT_DEF)),
                          (LDRHui am_indexed16:$addr), hsub)>;
def : Pat <(v8i16 (scalar_to_vector (i32 (extloadi16 am_indexed16:$addr)))),
           (INSERT_SUBREG (v8i16 (IMPLICIT_DEF)),
                          (LDRHui am_indexed16:$addr), hsub)>;
def : Pat <(v2i32 (scalar_to_vector (i32 (load am_indexed32:$addr)))),
           (INSERT_SUBREG (v2i32 (IMPLICIT_DEF)),
                          (LDRSui am_indexed32:$addr), ssub)>;
def : Pat <(v4i32 (scalar_to_vector (i32 (load am_indexed32:$addr)))),
           (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)),
                          (LDRSui am_indexed32:$addr), ssub)>;
def : Pat <(v1i64 (scalar_to_vector (i64 (load am_indexed64:$addr)))),
           (LDRDui am_indexed64:$addr)>;
def : Pat <(v2i64 (scalar_to_vector (i64 (load am_indexed64:$addr)))),
           (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)),
                          (LDRDui am_indexed64:$addr), dsub)>;

// Match all load 64 bits width whose type is compatible with FPR64
def : Pat<(v2f32 (load am_indexed64:$addr)), (LDRDui am_indexed64:$addr)>;
def : Pat<(v1f64 (load am_indexed64:$addr)), (LDRDui am_indexed64:$addr)>;
def : Pat<(v8i8 (load am_indexed64:$addr)), (LDRDui am_indexed64:$addr)>;
def : Pat<(v4i16 (load am_indexed64:$addr)), (LDRDui am_indexed64:$addr)>;
def : Pat<(v2i32 (load am_indexed64:$addr)), (LDRDui am_indexed64:$addr)>;
def : Pat<(v1i64 (load am_indexed64:$addr)), (LDRDui am_indexed64:$addr)>;

// Match all load 128 bits width whose type is compatible with FPR128
def : Pat<(v4f32 (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;
def : Pat<(v2f64 (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;
def : Pat<(v16i8 (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;
def : Pat<(v8i16 (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;
def : Pat<(v4i32 (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;
def : Pat<(v2i64 (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;
def : Pat<(f128  (load am_indexed128:$addr)), (LDRQui am_indexed128:$addr)>;

def LDRHHui : LoadUI<0b01, 0, 0b01, GPR32, am_indexed16, "ldrh",
                     [(set GPR32:$Rt, (zextloadi16 am_indexed16:$addr))]>;
def LDRBBui : LoadUI<0b00, 0, 0b01, GPR32, am_indexed8, "ldrb",
                     [(set GPR32:$Rt, (zextloadi8 am_indexed8:$addr))]>;
// zextload -> i64
def : Pat<(i64 (zextloadi8 am_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBui am_indexed8:$addr), sub_32)>;
def : Pat<(i64 (zextloadi16 am_indexed16:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRHHui am_indexed16:$addr), sub_32)>;

// zextloadi1 -> zextloadi8
def : Pat<(i32 (zextloadi1 am_indexed8:$addr)), (LDRBBui am_indexed8:$addr)>;
def : Pat<(i64 (zextloadi1 am_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBui am_indexed8:$addr), sub_32)>;

// extload -> zextload
def : Pat<(i32 (extloadi16 am_indexed16:$addr)), (LDRHHui am_indexed16:$addr)>;
def : Pat<(i32 (extloadi8 am_indexed8:$addr)), (LDRBBui am_indexed8:$addr)>;
def : Pat<(i32 (extloadi1 am_indexed8:$addr)), (LDRBBui am_indexed8:$addr)>;
def : Pat<(i64 (extloadi32 am_indexed32:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRWui am_indexed32:$addr), sub_32)>;
def : Pat<(i64 (extloadi16 am_indexed16:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRHHui am_indexed16:$addr), sub_32)>;
def : Pat<(i64 (extloadi8 am_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBui am_indexed8:$addr), sub_32)>;
def : Pat<(i64 (extloadi1 am_indexed8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDRBBui am_indexed8:$addr), sub_32)>;

// load sign-extended half-word
def LDRSHWui : LoadUI<0b01, 0, 0b11, GPR32, am_indexed16, "ldrsh",
                      [(set GPR32:$Rt, (sextloadi16 am_indexed16:$addr))]>;
def LDRSHXui : LoadUI<0b01, 0, 0b10, GPR64, am_indexed16, "ldrsh",
                      [(set GPR64:$Rt, (sextloadi16 am_indexed16:$addr))]>;

// load sign-extended byte
def LDRSBWui : LoadUI<0b00, 0, 0b11, GPR32, am_indexed8, "ldrsb",
                      [(set GPR32:$Rt, (sextloadi8 am_indexed8:$addr))]>;
def LDRSBXui : LoadUI<0b00, 0, 0b10, GPR64, am_indexed8, "ldrsb",
                      [(set GPR64:$Rt, (sextloadi8 am_indexed8:$addr))]>;

// load sign-extended word
def LDRSWui  : LoadUI<0b10, 0, 0b10, GPR64, am_indexed32, "ldrsw",
                      [(set GPR64:$Rt, (sextloadi32 am_indexed32:$addr))]>;

// load zero-extended word
def : Pat<(i64 (zextloadi32 am_indexed32:$addr)),
 (SUBREG_TO_REG (i64 0), (LDRWui am_indexed32:$addr), sub_32)>;

// Pre-fetch.
def PRFMui : PrefetchUI<0b11, 0, 0b10, "prfm",
                        [(ARM64Prefetch imm:$Rt, am_indexed64:$addr)]>;

//---
// (literal)
def LDRWl : LoadLiteral<0b00, 0, GPR32, "ldr">;
def LDRXl : LoadLiteral<0b01, 0, GPR64, "ldr">;
def LDRSl : LoadLiteral<0b00, 1, FPR32, "ldr">;
def LDRDl : LoadLiteral<0b01, 1, FPR64, "ldr">;
def LDRQl : LoadLiteral<0b10, 1, FPR128, "ldr">;

// load sign-extended word
def LDRSWl : LoadLiteral<0b10, 0, GPR64, "ldrsw">;

// prefetch
def PRFMl : PrefetchLiteral<0b11, 0, "prfm", []>;
//                   [(ARM64Prefetch imm:$Rt, tglobaladdr:$label)]>;

//---
// (unscaled immediate)
def LDURXi : LoadUnscaled<0b11, 0, 0b01, GPR64, am_unscaled64, "ldur",
                          [(set GPR64:$Rt, (load am_unscaled64:$addr))]>;
def LDURWi : LoadUnscaled<0b10, 0, 0b01, GPR32, am_unscaled32, "ldur",
                          [(set GPR32:$Rt, (load am_unscaled32:$addr))]>;
def LDURBi : LoadUnscaled<0b00, 1, 0b01, FPR8,  am_unscaled8, "ldur",
                          [(set FPR8:$Rt, (load am_unscaled8:$addr))]>;
def LDURHi : LoadUnscaled<0b01, 1, 0b01, FPR16, am_unscaled16, "ldur",
                          [(set (f16 FPR16:$Rt), (load am_unscaled16:$addr))]>;
def LDURSi : LoadUnscaled<0b10, 1, 0b01, FPR32, am_unscaled32, "ldur",
                          [(set (f32 FPR32:$Rt), (load am_unscaled32:$addr))]>;
def LDURDi : LoadUnscaled<0b11, 1, 0b01, FPR64, am_unscaled64, "ldur",
                          [(set (f64 FPR64:$Rt), (load am_unscaled64:$addr))]>;
def LDURQi : LoadUnscaled<0b00, 1, 0b11, FPR128, am_unscaled128, "ldur",
                        [(set (v2f64 FPR128:$Rt), (load am_unscaled128:$addr))]>;

def LDURHHi
    : LoadUnscaled<0b01, 0, 0b01, GPR32, am_unscaled16, "ldurh",
                   [(set GPR32:$Rt, (zextloadi16 am_unscaled16:$addr))]>;
def LDURBBi
    : LoadUnscaled<0b00, 0, 0b01, GPR32, am_unscaled8, "ldurb",
                   [(set GPR32:$Rt, (zextloadi8 am_unscaled8:$addr))]>;

// Match all load 64 bits width whose type is compatible with FPR64
def : Pat<(v2f32 (load am_unscaled64:$addr)), (LDURDi am_unscaled64:$addr)>;
def : Pat<(v1f64 (load am_unscaled64:$addr)), (LDURDi am_unscaled64:$addr)>;
def : Pat<(v8i8 (load am_unscaled64:$addr)), (LDURDi am_unscaled64:$addr)>;
def : Pat<(v4i16 (load am_unscaled64:$addr)), (LDURDi am_unscaled64:$addr)>;
def : Pat<(v2i32 (load am_unscaled64:$addr)), (LDURDi am_unscaled64:$addr)>;
def : Pat<(v1i64 (load am_unscaled64:$addr)), (LDURDi am_unscaled64:$addr)>;

// Match all load 128 bits width whose type is compatible with FPR128
def : Pat<(v4f32 (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;
def : Pat<(v2f64 (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;
def : Pat<(v16i8 (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;
def : Pat<(v8i16 (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;
def : Pat<(v4i32 (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;
def : Pat<(v2i64 (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;
def : Pat<(f128  (load am_unscaled128:$addr)), (LDURQi am_unscaled128:$addr)>;

//  anyext -> zext
def : Pat<(i32 (extloadi16 am_unscaled16:$addr)), (LDURHHi am_unscaled16:$addr)>;
def : Pat<(i32 (extloadi8 am_unscaled8:$addr)), (LDURBBi am_unscaled8:$addr)>;
def : Pat<(i32 (extloadi1 am_unscaled8:$addr)), (LDURBBi am_unscaled8:$addr)>;
def : Pat<(i64 (extloadi32 am_unscaled32:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURWi am_unscaled32:$addr), sub_32)>;
def : Pat<(i64 (extloadi16 am_unscaled16:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURHHi am_unscaled16:$addr), sub_32)>;
def : Pat<(i64 (extloadi8 am_unscaled8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURBBi am_unscaled8:$addr), sub_32)>;
def : Pat<(i64 (extloadi1 am_unscaled8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURBBi am_unscaled8:$addr), sub_32)>;
// unscaled zext
def : Pat<(i32 (zextloadi16 am_unscaled16:$addr)),
    (LDURHHi am_unscaled16:$addr)>;
def : Pat<(i32 (zextloadi8 am_unscaled8:$addr)),
    (LDURBBi am_unscaled8:$addr)>;
def : Pat<(i32 (zextloadi1 am_unscaled8:$addr)),
    (LDURBBi am_unscaled8:$addr)>;
def : Pat<(i64 (zextloadi32 am_unscaled32:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURWi am_unscaled32:$addr), sub_32)>;
def : Pat<(i64 (zextloadi16 am_unscaled16:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURHHi am_unscaled16:$addr), sub_32)>;
def : Pat<(i64 (zextloadi8 am_unscaled8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURBBi am_unscaled8:$addr), sub_32)>;
def : Pat<(i64 (zextloadi1 am_unscaled8:$addr)),
    (SUBREG_TO_REG (i64 0), (LDURBBi am_unscaled8:$addr), sub_32)>;


//---
// LDR mnemonics fall back to LDUR for negative or unaligned offsets.

// Define new assembler match classes as we want to only match these when
// the don't otherwise match the scaled addressing mode for LDR/STR. Don't
// associate a DiagnosticType either, as we want the diagnostic for the
// canonical form (the scaled operand) to take precedence.
def MemoryUnscaledFB8Operand : AsmOperandClass {
  let Name = "MemoryUnscaledFB8";
  let RenderMethod = "addMemoryUnscaledOperands";
}
def MemoryUnscaledFB16Operand : AsmOperandClass {
  let Name = "MemoryUnscaledFB16";
  let RenderMethod = "addMemoryUnscaledOperands";
}
def MemoryUnscaledFB32Operand : AsmOperandClass {
  let Name = "MemoryUnscaledFB32";
  let RenderMethod = "addMemoryUnscaledOperands";
}
def MemoryUnscaledFB64Operand : AsmOperandClass {
  let Name = "MemoryUnscaledFB64";
  let RenderMethod = "addMemoryUnscaledOperands";
}
def MemoryUnscaledFB128Operand : AsmOperandClass {
  let Name = "MemoryUnscaledFB128";
  let RenderMethod = "addMemoryUnscaledOperands";
}
def am_unscaled_fb8 : Operand<i64> {
  let ParserMatchClass = MemoryUnscaledFB8Operand;
  let MIOperandInfo = (ops GPR64sp:$base, i64imm:$offset);
}
def am_unscaled_fb16 : Operand<i64> {
  let ParserMatchClass = MemoryUnscaledFB16Operand;
  let MIOperandInfo = (ops GPR64sp:$base, i64imm:$offset);
}
def am_unscaled_fb32 : Operand<i64> {
  let ParserMatchClass = MemoryUnscaledFB32Operand;
  let MIOperandInfo = (ops GPR64sp:$base, i64imm:$offset);
}
def am_unscaled_fb64 : Operand<i64> {
  let ParserMatchClass = MemoryUnscaledFB64Operand;
  let MIOperandInfo = (ops GPR64sp:$base, i64imm:$offset);
}
def am_unscaled_fb128 : Operand<i64> {
  let ParserMatchClass = MemoryUnscaledFB128Operand;
  let MIOperandInfo = (ops GPR64sp:$base, i64imm:$offset);
}
def : InstAlias<"ldr $Rt, $addr", (LDURXi GPR64:$Rt, am_unscaled_fb64:$addr)>;
def : InstAlias<"ldr $Rt, $addr", (LDURWi GPR32:$Rt, am_unscaled_fb32:$addr)>;
def : InstAlias<"ldr $Rt, $addr", (LDURBi FPR8:$Rt, am_unscaled_fb8:$addr)>;
def : InstAlias<"ldr $Rt, $addr", (LDURHi FPR16:$Rt, am_unscaled_fb16:$addr)>;
def : InstAlias<"ldr $Rt, $addr", (LDURSi FPR32:$Rt, am_unscaled_fb32:$addr)>;
def : InstAlias<"ldr $Rt, $addr", (LDURDi FPR64:$Rt, am_unscaled_fb64:$addr)>;
def : InstAlias<"ldr $Rt, $addr", (LDURQi FPR128:$Rt, am_unscaled_fb128:$addr)>;

// zextload -> i64
def : Pat<(i64 (zextloadi8 am_unscaled8:$addr)),
  (SUBREG_TO_REG (i64 0), (LDURBBi am_unscaled8:$addr), sub_32)>;
def : Pat<(i64 (zextloadi16 am_unscaled16:$addr)),
  (SUBREG_TO_REG (i64 0), (LDURHHi am_unscaled16:$addr), sub_32)>;

// load sign-extended half-word
def LDURSHWi
    : LoadUnscaled<0b01, 0, 0b11, GPR32, am_unscaled16, "ldursh",
                   [(set GPR32:$Rt, (sextloadi16 am_unscaled16:$addr))]>;
def LDURSHXi
    : LoadUnscaled<0b01, 0, 0b10, GPR64, am_unscaled16, "ldursh",
                   [(set GPR64:$Rt, (sextloadi16 am_unscaled16:$addr))]>;

// load sign-extended byte
def LDURSBWi
    : LoadUnscaled<0b00, 0, 0b11, GPR32, am_unscaled8, "ldursb",
                   [(set GPR32:$Rt, (sextloadi8 am_unscaled8:$addr))]>;
def LDURSBXi
    : LoadUnscaled<0b00, 0, 0b10, GPR64, am_unscaled8, "ldursb",
                   [(set GPR64:$Rt, (sextloadi8 am_unscaled8:$addr))]>;

// load sign-extended word
def LDURSWi
    : LoadUnscaled<0b10, 0, 0b10, GPR64, am_unscaled32, "ldursw",
                   [(set GPR64:$Rt, (sextloadi32 am_unscaled32:$addr))]>;

// zero and sign extending aliases from generic LDR* mnemonics to LDUR*.
def : InstAlias<"ldrb $Rt, $addr", (LDURBBi GPR32:$Rt, am_unscaled_fb8:$addr)>;
def : InstAlias<"ldrh $Rt, $addr", (LDURHHi GPR32:$Rt, am_unscaled_fb16:$addr)>;
def : InstAlias<"ldrsb $Rt, $addr", (LDURSBWi GPR32:$Rt, am_unscaled_fb8:$addr)>;
def : InstAlias<"ldrsb $Rt, $addr", (LDURSBXi GPR64:$Rt, am_unscaled_fb8:$addr)>;
def : InstAlias<"ldrsh $Rt, $addr", (LDURSHWi GPR32:$Rt, am_unscaled_fb16:$addr)>;
def : InstAlias<"ldrsh $Rt, $addr", (LDURSHXi GPR64:$Rt, am_unscaled_fb16:$addr)>;
def : InstAlias<"ldrsw $Rt, $addr", (LDURSWi GPR64:$Rt, am_unscaled_fb32:$addr)>;

// Pre-fetch.
def PRFUMi : PrefetchUnscaled<0b11, 0, 0b10, "prfum",
                               [(ARM64Prefetch imm:$Rt, am_unscaled64:$addr)]>;

//---
// (unscaled immediate, unprivileged)
def LDTRXi : LoadUnprivileged<0b11, 0, 0b01, GPR64, "ldtr">;
def LDTRWi : LoadUnprivileged<0b10, 0, 0b01, GPR32, "ldtr">;

def LDTRHi : LoadUnprivileged<0b01, 0, 0b01, GPR32, "ldtrh">;
def LDTRBi : LoadUnprivileged<0b00, 0, 0b01, GPR32, "ldtrb">;

// load sign-extended half-word
def LDTRSHWi : LoadUnprivileged<0b01, 0, 0b11, GPR32, "ldtrsh">;
def LDTRSHXi : LoadUnprivileged<0b01, 0, 0b10, GPR64, "ldtrsh">;

// load sign-extended byte
def LDTRSBWi : LoadUnprivileged<0b00, 0, 0b11, GPR32, "ldtrsb">;
def LDTRSBXi : LoadUnprivileged<0b00, 0, 0b10, GPR64, "ldtrsb">;

// load sign-extended word
def LDTRSWi  : LoadUnprivileged<0b10, 0, 0b10, GPR64, "ldtrsw">;

//---
// (immediate pre-indexed)
def LDRWpre : LoadPreIdx<0b10, 0, 0b01, GPR32, "ldr">;
def LDRXpre : LoadPreIdx<0b11, 0, 0b01, GPR64, "ldr">;
def LDRBpre : LoadPreIdx<0b00, 1, 0b01, FPR8,  "ldr">;
def LDRHpre : LoadPreIdx<0b01, 1, 0b01, FPR16, "ldr">;
def LDRSpre : LoadPreIdx<0b10, 1, 0b01, FPR32, "ldr">;
def LDRDpre : LoadPreIdx<0b11, 1, 0b01, FPR64, "ldr">;
def LDRQpre : LoadPreIdx<0b00, 1, 0b11, FPR128, "ldr">;

// load sign-extended half-word
def LDRSHWpre : LoadPreIdx<0b01, 0, 0b11, GPR32, "ldrsh">;
def LDRSHXpre : LoadPreIdx<0b01, 0, 0b10, GPR64, "ldrsh">;

// load sign-extended byte
def LDRSBWpre : LoadPreIdx<0b00, 0, 0b11, GPR32, "ldrsb">;
def LDRSBXpre : LoadPreIdx<0b00, 0, 0b10, GPR64, "ldrsb">;

// load zero-extended byte
def LDRBBpre : LoadPreIdx<0b00, 0, 0b01, GPR32, "ldrb">;
def LDRHHpre : LoadPreIdx<0b01, 0, 0b01, GPR32, "ldrh">;

// load sign-extended word
def LDRSWpre : LoadPreIdx<0b10, 0, 0b10, GPR64, "ldrsw">;

// ISel pseudos and patterns. See expanded comment on LoadPreIdxPseudo.
def LDRDpre_isel  : LoadPreIdxPseudo<FPR64>;
def LDRSpre_isel  : LoadPreIdxPseudo<FPR32>;
def LDRXpre_isel  : LoadPreIdxPseudo<GPR64>;
def LDRWpre_isel  : LoadPreIdxPseudo<GPR32>;
def LDRHHpre_isel : LoadPreIdxPseudo<GPR32>;
def LDRBBpre_isel : LoadPreIdxPseudo<GPR32>;

def LDRSWpre_isel : LoadPreIdxPseudo<GPR64>;
def LDRSHWpre_isel : LoadPreIdxPseudo<GPR32>;
def LDRSHXpre_isel : LoadPreIdxPseudo<GPR64>;
def LDRSBWpre_isel : LoadPreIdxPseudo<GPR32>;
def LDRSBXpre_isel : LoadPreIdxPseudo<GPR64>;

//---
// (immediate post-indexed)
def LDRWpost : LoadPostIdx<0b10, 0, 0b01, GPR32, "ldr">;
def LDRXpost : LoadPostIdx<0b11, 0, 0b01, GPR64, "ldr">;
def LDRBpost : LoadPostIdx<0b00, 1, 0b01, FPR8,  "ldr">;
def LDRHpost : LoadPostIdx<0b01, 1, 0b01, FPR16, "ldr">;
def LDRSpost : LoadPostIdx<0b10, 1, 0b01, FPR32, "ldr">;
def LDRDpost : LoadPostIdx<0b11, 1, 0b01, FPR64, "ldr">;
def LDRQpost : LoadPostIdx<0b00, 1, 0b11, FPR128, "ldr">;

// load sign-extended half-word
def LDRSHWpost : LoadPostIdx<0b01, 0, 0b11, GPR32, "ldrsh">;
def LDRSHXpost : LoadPostIdx<0b01, 0, 0b10, GPR64, "ldrsh">;

// load sign-extended byte
def LDRSBWpost : LoadPostIdx<0b00, 0, 0b11, GPR32, "ldrsb">;
def LDRSBXpost : LoadPostIdx<0b00, 0, 0b10, GPR64, "ldrsb">;

// load zero-extended byte
def LDRBBpost : LoadPostIdx<0b00, 0, 0b01, GPR32, "ldrb">;
def LDRHHpost : LoadPostIdx<0b01, 0, 0b01, GPR32, "ldrh">;

// load sign-extended word
def LDRSWpost : LoadPostIdx<0b10, 0, 0b10, GPR64, "ldrsw">;

// ISel pseudos and patterns. See expanded comment on LoadPostIdxPseudo.
def LDRDpost_isel  : LoadPostIdxPseudo<FPR64>;
def LDRSpost_isel  : LoadPostIdxPseudo<FPR32>;
def LDRXpost_isel  : LoadPostIdxPseudo<GPR64>;
def LDRWpost_isel  : LoadPostIdxPseudo<GPR32>;
def LDRHHpost_isel : LoadPostIdxPseudo<GPR32>;
def LDRBBpost_isel : LoadPostIdxPseudo<GPR32>;

def LDRSWpost_isel : LoadPostIdxPseudo<GPR64>;
def LDRSHWpost_isel : LoadPostIdxPseudo<GPR32>;
def LDRSHXpost_isel : LoadPostIdxPseudo<GPR64>;
def LDRSBWpost_isel : LoadPostIdxPseudo<GPR32>;
def LDRSBXpost_isel : LoadPostIdxPseudo<GPR64>;

//===----------------------------------------------------------------------===//
// Store instructions.
//===----------------------------------------------------------------------===//

// Pair (indexed, offset)
// FIXME: Use dedicated range-checked addressing mode operand here.
def STPWi : StorePairOffset<0b00, 0, GPR32, am_indexed32simm7, "stp">;
def STPXi : StorePairOffset<0b10, 0, GPR64, am_indexed64simm7, "stp">;
def STPSi : StorePairOffset<0b00, 1, FPR32, am_indexed32simm7, "stp">;
def STPDi : StorePairOffset<0b01, 1, FPR64, am_indexed64simm7, "stp">;
def STPQi : StorePairOffset<0b10, 1, FPR128, am_indexed128simm7, "stp">;

// Pair (pre-indexed)
def STPWpre : StorePairPreIdx<0b00, 0, GPR32, am_indexed32simm7_wb, "stp">;
def STPXpre : StorePairPreIdx<0b10, 0, GPR64, am_indexed64simm7_wb, "stp">;
def STPSpre : StorePairPreIdx<0b00, 1, FPR32, am_indexed32simm7_wb, "stp">;
def STPDpre : StorePairPreIdx<0b01, 1, FPR64, am_indexed64simm7_wb, "stp">;
def STPQpre : StorePairPreIdx<0b10, 1, FPR128, am_indexed128simm7_wb, "stp">;

// Pair (pre-indexed)
def STPWpost : StorePairPostIdx<0b00, 0, GPR32, simm7s4, "stp">;
def STPXpost : StorePairPostIdx<0b10, 0, GPR64, simm7s8, "stp">;
def STPSpost : StorePairPostIdx<0b00, 1, FPR32, simm7s4, "stp">;
def STPDpost : StorePairPostIdx<0b01, 1, FPR64, simm7s8, "stp">;
def STPQpost : StorePairPostIdx<0b10, 1, FPR128, simm7s16, "stp">;

// Pair (no allocate)
def STNPWi : StorePairNoAlloc<0b00, 0, GPR32, am_indexed32simm7, "stnp">;
def STNPXi : StorePairNoAlloc<0b10, 0, GPR64, am_indexed64simm7, "stnp">;
def STNPSi : StorePairNoAlloc<0b00, 1, FPR32, am_indexed32simm7, "stnp">;
def STNPDi : StorePairNoAlloc<0b01, 1, FPR64, am_indexed64simm7, "stnp">;
def STNPQi : StorePairNoAlloc<0b10, 1, FPR128, am_indexed128simm7, "stnp">;

//---
// (Register offset)

let AddedComplexity = 10 in {

// Integer
def STRHHro : Store16RO<0b01, 0, 0b00, GPR32, "strh",
                            [(truncstorei16 GPR32:$Rt, ro_indexed16:$addr)]>;
def STRBBro : Store8RO<0b00,  0, 0b00, GPR32, "strb",
                            [(truncstorei8 GPR32:$Rt, ro_indexed8:$addr)]>;
def STRWro  : Store32RO<0b10,   0, 0b00, GPR32, "str",
                            [(store GPR32:$Rt, ro_indexed32:$addr)]>;
def STRXro  : Store64RO<0b11,   0, 0b00, GPR64, "str",
                            [(store GPR64:$Rt, ro_indexed64:$addr)]>;

// truncstore i64
def : Pat<(truncstorei8 GPR64:$Rt, ro_indexed8:$addr),
           (STRBBro (EXTRACT_SUBREG GPR64:$Rt, sub_32), ro_indexed8:$addr)>;
def : Pat<(truncstorei16 GPR64:$Rt, ro_indexed16:$addr),
           (STRHHro (EXTRACT_SUBREG GPR64:$Rt, sub_32), ro_indexed16:$addr)>;
def : Pat<(truncstorei32 GPR64:$Rt, ro_indexed32:$addr),
           (STRWro (EXTRACT_SUBREG GPR64:$Rt, sub_32), ro_indexed32:$addr)>;


// Floating-point
def STRBro : Store8RO<0b00,  1, 0b00, FPR8,  "str",
                            [(store FPR8:$Rt, ro_indexed8:$addr)]>;
def STRHro : Store16RO<0b01, 1, 0b00, FPR16, "str",
                            [(store (f16 FPR16:$Rt), ro_indexed16:$addr)]>;
def STRSro : Store32RO<0b10,   1, 0b00, FPR32, "str",
                            [(store (f32 FPR32:$Rt), ro_indexed32:$addr)]>;
def STRDro : Store64RO<0b11,   1, 0b00, FPR64, "str",
                            [(store (f64 FPR64:$Rt), ro_indexed64:$addr)]>;
def STRQro : Store128RO<0b00,   1, 0b10, FPR128, "str", []> {
  let mayStore = 1;
}

// Match all store 64 bits width whose type is compatible with FPR64
def : Pat<(store (v2f32 FPR64:$Rn), ro_indexed64:$addr),
          (STRDro FPR64:$Rn, ro_indexed64:$addr)>;
def : Pat<(store (v1f64 FPR64:$Rn), ro_indexed64:$addr),
          (STRDro FPR64:$Rn, ro_indexed64:$addr)>;
def : Pat<(store (v8i8 FPR64:$Rn), ro_indexed64:$addr),
          (STRDro FPR64:$Rn, ro_indexed64:$addr)>;
def : Pat<(store (v4i16 FPR64:$Rn), ro_indexed64:$addr),
          (STRDro FPR64:$Rn, ro_indexed64:$addr)>;
def : Pat<(store (v2i32 FPR64:$Rn), ro_indexed64:$addr),
          (STRDro FPR64:$Rn, ro_indexed64:$addr)>;
def : Pat<(store (v1i64 FPR64:$Rn), ro_indexed64:$addr),
          (STRDro FPR64:$Rn, ro_indexed64:$addr)>;

// Match all store 128 bits width whose type is compatible with FPR128
def : Pat<(store (v4f32 FPR128:$Rn), ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;
def : Pat<(store (v2f64 FPR128:$Rn), ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;
def : Pat<(store (v16i8 FPR128:$Rn), ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;
def : Pat<(store (v8i16 FPR128:$Rn), ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;
def : Pat<(store (v4i32 FPR128:$Rn), ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;
def : Pat<(store (v2i64 FPR128:$Rn), ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;
def : Pat<(store (f128 FPR128:$Rn),  ro_indexed128:$addr),
          (STRQro FPR128:$Rn, ro_indexed128:$addr)>;

//---
// (unsigned immediate)
def STRXui : StoreUI<0b11, 0, 0b00, GPR64, am_indexed64, "str",
                     [(store GPR64:$Rt, am_indexed64:$addr)]>;
def STRWui : StoreUI<0b10, 0, 0b00, GPR32, am_indexed32, "str",
                     [(store GPR32:$Rt, am_indexed32:$addr)]>;
def STRBui : StoreUI<0b00, 1, 0b00, FPR8, am_indexed8, "str",
                     [(store FPR8:$Rt, am_indexed8:$addr)]>;
def STRHui : StoreUI<0b01, 1, 0b00, FPR16, am_indexed16, "str",
                     [(store (f16 FPR16:$Rt), am_indexed16:$addr)]>;
def STRSui : StoreUI<0b10, 1, 0b00, FPR32, am_indexed32, "str",
                     [(store (f32 FPR32:$Rt), am_indexed32:$addr)]>;
def STRDui : StoreUI<0b11, 1, 0b00, FPR64, am_indexed64, "str",
                     [(store (f64 FPR64:$Rt), am_indexed64:$addr)]>;
def STRQui : StoreUI<0b00, 1, 0b10, FPR128, am_indexed128, "str", []> {
  let mayStore = 1;
}

// Match all store 64 bits width whose type is compatible with FPR64
def : Pat<(store (v2f32 FPR64:$Rn), am_indexed64:$addr),
          (STRDui FPR64:$Rn, am_indexed64:$addr)>;
def : Pat<(store (v1f64 FPR64:$Rn), am_indexed64:$addr),
          (STRDui FPR64:$Rn, am_indexed64:$addr)>;
def : Pat<(store (v8i8 FPR64:$Rn), am_indexed64:$addr),
          (STRDui FPR64:$Rn, am_indexed64:$addr)>;
def : Pat<(store (v4i16 FPR64:$Rn), am_indexed64:$addr),
          (STRDui FPR64:$Rn, am_indexed64:$addr)>;
def : Pat<(store (v2i32 FPR64:$Rn), am_indexed64:$addr),
          (STRDui FPR64:$Rn, am_indexed64:$addr)>;
def : Pat<(store (v1i64 FPR64:$Rn), am_indexed64:$addr),
          (STRDui FPR64:$Rn, am_indexed64:$addr)>;

// Match all store 128 bits width whose type is compatible with FPR128
def : Pat<(store (v4f32 FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;
def : Pat<(store (v2f64 FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;
def : Pat<(store (v16i8 FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;
def : Pat<(store (v8i16 FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;
def : Pat<(store (v4i32 FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;
def : Pat<(store (v2i64 FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;
def : Pat<(store (f128  FPR128:$Rn), am_indexed128:$addr),
          (STRQui FPR128:$Rn, am_indexed128:$addr)>;

def STRHHui : StoreUI<0b01, 0, 0b00, GPR32, am_indexed16, "strh",
                      [(truncstorei16 GPR32:$Rt, am_indexed16:$addr)]>;
def STRBBui : StoreUI<0b00, 0, 0b00, GPR32, am_indexed8,  "strb",
                      [(truncstorei8 GPR32:$Rt, am_indexed8:$addr)]>;

// truncstore i64
def : Pat<(truncstorei32 GPR64:$Rt, am_indexed32:$addr),
  (STRWui (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_indexed32:$addr)>;
def : Pat<(truncstorei16 GPR64:$Rt, am_indexed16:$addr),
  (STRHHui (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_indexed16:$addr)>;
def : Pat<(truncstorei8 GPR64:$Rt, am_indexed8:$addr),
  (STRBBui (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_indexed8:$addr)>;

} // AddedComplexity = 10

//---
// (unscaled immediate)
def STURXi : StoreUnscaled<0b11, 0, 0b00, GPR64, am_unscaled64, "stur",
                           [(store GPR64:$Rt, am_unscaled64:$addr)]>;
def STURWi : StoreUnscaled<0b10, 0, 0b00, GPR32, am_unscaled32, "stur",
                           [(store GPR32:$Rt, am_unscaled32:$addr)]>;
def STURBi : StoreUnscaled<0b00, 1, 0b00, FPR8,  am_unscaled8, "stur",
                           [(store FPR8:$Rt, am_unscaled8:$addr)]>;
def STURHi : StoreUnscaled<0b01, 1, 0b00, FPR16, am_unscaled16, "stur",
                           [(store (f16 FPR16:$Rt), am_unscaled16:$addr)]>;
def STURSi : StoreUnscaled<0b10, 1, 0b00, FPR32, am_unscaled32, "stur",
                           [(store (f32 FPR32:$Rt), am_unscaled32:$addr)]>;
def STURDi : StoreUnscaled<0b11, 1, 0b00, FPR64, am_unscaled64, "stur",
                           [(store (f64 FPR64:$Rt), am_unscaled64:$addr)]>;
def STURQi : StoreUnscaled<0b00, 1, 0b10, FPR128, am_unscaled128, "stur",
                           [(store (v2f64 FPR128:$Rt), am_unscaled128:$addr)]>;
def STURHHi : StoreUnscaled<0b01, 0, 0b00, GPR32, am_unscaled16, "sturh",
                            [(truncstorei16 GPR32:$Rt, am_unscaled16:$addr)]>;
def STURBBi : StoreUnscaled<0b00, 0, 0b00, GPR32, am_unscaled8, "sturb",
                            [(truncstorei8 GPR32:$Rt, am_unscaled8:$addr)]>;

// Match all store 64 bits width whose type is compatible with FPR64
def : Pat<(store (v2f32 FPR64:$Rn), am_unscaled64:$addr),
          (STURDi FPR64:$Rn, am_unscaled64:$addr)>;
def : Pat<(store (v1f64 FPR64:$Rn), am_unscaled64:$addr),
          (STURDi FPR64:$Rn, am_unscaled64:$addr)>;
def : Pat<(store (v8i8 FPR64:$Rn), am_unscaled64:$addr),
          (STURDi FPR64:$Rn, am_unscaled64:$addr)>;
def : Pat<(store (v4i16 FPR64:$Rn), am_unscaled64:$addr),
          (STURDi FPR64:$Rn, am_unscaled64:$addr)>;
def : Pat<(store (v2i32 FPR64:$Rn), am_unscaled64:$addr),
          (STURDi FPR64:$Rn, am_unscaled64:$addr)>;
def : Pat<(store (v1i64 FPR64:$Rn), am_unscaled64:$addr),
          (STURDi FPR64:$Rn, am_unscaled64:$addr)>;

// Match all store 128 bits width whose type is compatible with FPR128
def : Pat<(store (v4f32 FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;
def : Pat<(store (v2f64 FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;
def : Pat<(store (v16i8 FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;
def : Pat<(store (v8i16 FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;
def : Pat<(store (v4i32 FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;
def : Pat<(store (v2i64 FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;
def : Pat<(store (f128  FPR128:$Rn), am_unscaled128:$addr),
          (STURQi FPR128:$Rn, am_unscaled128:$addr)>;

// unscaled i64 truncating stores
def : Pat<(truncstorei32 GPR64:$Rt, am_unscaled32:$addr),
  (STURWi (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_unscaled32:$addr)>;
def : Pat<(truncstorei16 GPR64:$Rt, am_unscaled16:$addr),
  (STURHHi (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_unscaled16:$addr)>;
def : Pat<(truncstorei8 GPR64:$Rt, am_unscaled8:$addr),
  (STURBBi (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_unscaled8:$addr)>;

//---
// STR mnemonics fall back to STUR for negative or unaligned offsets.
def : InstAlias<"str $Rt, $addr", (STURXi GPR64:$Rt, am_unscaled_fb64:$addr)>;
def : InstAlias<"str $Rt, $addr", (STURWi GPR32:$Rt, am_unscaled_fb32:$addr)>;
def : InstAlias<"str $Rt, $addr", (STURBi FPR8:$Rt, am_unscaled_fb8:$addr)>;
def : InstAlias<"str $Rt, $addr", (STURHi FPR16:$Rt, am_unscaled_fb16:$addr)>;
def : InstAlias<"str $Rt, $addr", (STURSi FPR32:$Rt, am_unscaled_fb32:$addr)>;
def : InstAlias<"str $Rt, $addr", (STURDi FPR64:$Rt, am_unscaled_fb64:$addr)>;
def : InstAlias<"str $Rt, $addr", (STURQi FPR128:$Rt, am_unscaled_fb128:$addr)>;

def : InstAlias<"strb $Rt, $addr", (STURBBi GPR32:$Rt, am_unscaled_fb8:$addr)>;
def : InstAlias<"strh $Rt, $addr", (STURHHi GPR32:$Rt, am_unscaled_fb16:$addr)>;

//---
// (unscaled immediate, unprivileged)
def STTRWi : StoreUnprivileged<0b10, 0, 0b00, GPR32, "sttr">;
def STTRXi : StoreUnprivileged<0b11, 0, 0b00, GPR64, "sttr">;

def STTRHi : StoreUnprivileged<0b01, 0, 0b00, GPR32, "sttrh">;
def STTRBi : StoreUnprivileged<0b00, 0, 0b00, GPR32, "sttrb">;

//---
// (immediate pre-indexed)
def STRWpre : StorePreIdx<0b10, 0, 0b00, GPR32, "str">;
def STRXpre : StorePreIdx<0b11, 0, 0b00, GPR64, "str">;
def STRBpre : StorePreIdx<0b00, 1, 0b00, FPR8,  "str">;
def STRHpre : StorePreIdx<0b01, 1, 0b00, FPR16, "str">;
def STRSpre : StorePreIdx<0b10, 1, 0b00, FPR32, "str">;
def STRDpre : StorePreIdx<0b11, 1, 0b00, FPR64, "str">;
def STRQpre : StorePreIdx<0b00, 1, 0b10, FPR128, "str">;

def STRBBpre : StorePreIdx<0b00, 0, 0b00, GPR32, "strb">;
def STRHHpre : StorePreIdx<0b01, 0, 0b00, GPR32, "strh">;

// ISel pseudos and patterns. See expanded comment on StorePreIdxPseudo.
defm STRDpre : StorePreIdxPseudo<FPR64, f64, pre_store>;
defm STRSpre : StorePreIdxPseudo<FPR32, f32, pre_store>;
defm STRXpre : StorePreIdxPseudo<GPR64, i64, pre_store>;
defm STRWpre : StorePreIdxPseudo<GPR32, i32, pre_store>;
defm STRHHpre : StorePreIdxPseudo<GPR32, i32, pre_truncsti16>;
defm STRBBpre : StorePreIdxPseudo<GPR32, i32, pre_truncsti8>;
// truncstore i64
def : Pat<(pre_truncsti32 GPR64:$Rt, am_noindex:$addr, simm9:$off),
  (STRWpre_isel (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_noindex:$addr,
                  simm9:$off)>;
def : Pat<(pre_truncsti16 GPR64:$Rt, am_noindex:$addr, simm9:$off),
  (STRHHpre_isel (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_noindex:$addr,
                  simm9:$off)>;
def : Pat<(pre_truncsti8 GPR64:$Rt, am_noindex:$addr, simm9:$off),
  (STRBBpre_isel (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_noindex:$addr,
                  simm9:$off)>;

//---
// (immediate post-indexed)
def STRWpost : StorePostIdx<0b10, 0, 0b00, GPR32, "str">;
def STRXpost : StorePostIdx<0b11, 0, 0b00, GPR64, "str">;
def STRBpost : StorePostIdx<0b00, 1, 0b00, FPR8,  "str">;
def STRHpost : StorePostIdx<0b01, 1, 0b00, FPR16, "str">;
def STRSpost : StorePostIdx<0b10, 1, 0b00, FPR32, "str">;
def STRDpost : StorePostIdx<0b11, 1, 0b00, FPR64, "str">;
def STRQpost : StorePostIdx<0b00, 1, 0b10, FPR128, "str">;

def STRBBpost : StorePostIdx<0b00, 0, 0b00, GPR32, "strb">;
def STRHHpost : StorePostIdx<0b01, 0, 0b00, GPR32, "strh">;

// ISel pseudos and patterns. See expanded comment on StorePostIdxPseudo.
defm STRDpost : StorePostIdxPseudo<FPR64, f64, post_store, STRDpost>;
defm STRSpost : StorePostIdxPseudo<FPR32, f32, post_store, STRSpost>;
defm STRXpost : StorePostIdxPseudo<GPR64, i64, post_store, STRXpost>;
defm STRWpost : StorePostIdxPseudo<GPR32, i32, post_store, STRWpost>;
defm STRHHpost : StorePostIdxPseudo<GPR32, i32, post_truncsti16, STRHHpost>;
defm STRBBpost : StorePostIdxPseudo<GPR32, i32, post_truncsti8, STRBBpost>;
// truncstore i64
def : Pat<(post_truncsti32 GPR64:$Rt, am_noindex:$addr, simm9:$off),
  (STRWpost_isel (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_noindex:$addr,
                  simm9:$off)>;
def : Pat<(post_truncsti16 GPR64:$Rt, am_noindex:$addr, simm9:$off),
  (STRHHpost_isel (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_noindex:$addr,
                  simm9:$off)>;
def : Pat<(post_truncsti8 GPR64:$Rt, am_noindex:$addr, simm9:$off),
  (STRBBpost_isel (EXTRACT_SUBREG GPR64:$Rt, sub_32), am_noindex:$addr,
                  simm9:$off)>;


//===----------------------------------------------------------------------===//
// Load/store exclusive instructions.
//===----------------------------------------------------------------------===//

def LDARW  : LoadAcquire   <0b10, 1, 1, 0, 1, GPR32, "ldar">;
def LDARX  : LoadAcquire   <0b11, 1, 1, 0, 1, GPR64, "ldar">;
def LDARB  : LoadAcquire   <0b00, 1, 1, 0, 1, GPR32, "ldarb">;
def LDARH  : LoadAcquire   <0b01, 1, 1, 0, 1, GPR32, "ldarh">;

def LDAXRW : LoadExclusive <0b10, 0, 1, 0, 1, GPR32, "ldaxr">;
def LDAXRX : LoadExclusive <0b11, 0, 1, 0, 1, GPR64, "ldaxr">;
def LDAXRB : LoadExclusive <0b00, 0, 1, 0, 1, GPR32, "ldaxrb">;
def LDAXRH : LoadExclusive <0b01, 0, 1, 0, 1, GPR32, "ldaxrh">;

def LDXRW  : LoadExclusive <0b10, 0, 1, 0, 0, GPR32, "ldxr">;
def LDXRX  : LoadExclusive <0b11, 0, 1, 0, 0, GPR64, "ldxr">;
def LDXRB  : LoadExclusive <0b00, 0, 1, 0, 0, GPR32, "ldxrb">;
def LDXRH  : LoadExclusive <0b01, 0, 1, 0, 0, GPR32, "ldxrh">;

def STLRW  : StoreRelease  <0b10, 1, 0, 0, 1, GPR32, "stlr">;
def STLRX  : StoreRelease  <0b11, 1, 0, 0, 1, GPR64, "stlr">;
def STLRB  : StoreRelease  <0b00, 1, 0, 0, 1, GPR32, "stlrb">;
def STLRH  : StoreRelease  <0b01, 1, 0, 0, 1, GPR32, "stlrh">;

def STLXRW : StoreExclusive<0b10, 0, 0, 0, 1, GPR32, "stlxr">;
def STLXRX : StoreExclusive<0b11, 0, 0, 0, 1, GPR64, "stlxr">;
def STLXRB : StoreExclusive<0b00, 0, 0, 0, 1, GPR32, "stlxrb">;
def STLXRH : StoreExclusive<0b01, 0, 0, 0, 1, GPR32, "stlxrh">;

def STXRW  : StoreExclusive<0b10, 0, 0, 0, 0, GPR32, "stxr">;
def STXRX  : StoreExclusive<0b11, 0, 0, 0, 0, GPR64, "stxr">;
def STXRB  : StoreExclusive<0b00, 0, 0, 0, 0, GPR32, "stxrb">;
def STXRH  : StoreExclusive<0b01, 0, 0, 0, 0, GPR32, "stxrh">;

def LDAXPW : LoadExclusivePair<0b10, 0, 1, 1, 1, GPR32, "ldaxp">;
def LDAXPX : LoadExclusivePair<0b11, 0, 1, 1, 1, GPR64, "ldaxp">;

def LDXPW  : LoadExclusivePair<0b10, 0, 1, 1, 0, GPR32, "ldxp">;
def LDXPX  : LoadExclusivePair<0b11, 0, 1, 1, 0, GPR64, "ldxp">;

def STLXPW : StoreExclusivePair<0b10, 0, 0, 1, 1, GPR32, "stlxp">;
def STLXPX : StoreExclusivePair<0b11, 0, 0, 1, 1, GPR64, "stlxp">;

def STXPW  : StoreExclusivePair<0b10, 0, 0, 1, 0, GPR32, "stxp">;
def STXPX  : StoreExclusivePair<0b11, 0, 0, 1, 0, GPR64, "stxp">;

//===----------------------------------------------------------------------===//
// Scaled floating point to integer conversion instructions.
//===----------------------------------------------------------------------===//

defm FCVTAS : FPToIntegerUnscaled<0b00, 0b100, "fcvtas", int_arm64_neon_fcvtas>;
defm FCVTAU : FPToIntegerUnscaled<0b00, 0b101, "fcvtau", int_arm64_neon_fcvtau>;
defm FCVTMS : FPToIntegerUnscaled<0b10, 0b000, "fcvtms", int_arm64_neon_fcvtms>;
defm FCVTMU : FPToIntegerUnscaled<0b10, 0b001, "fcvtmu", int_arm64_neon_fcvtmu>;
defm FCVTNS : FPToIntegerUnscaled<0b00, 0b000, "fcvtns", int_arm64_neon_fcvtns>;
defm FCVTNU : FPToIntegerUnscaled<0b00, 0b001, "fcvtnu", int_arm64_neon_fcvtnu>;
defm FCVTPS : FPToIntegerUnscaled<0b01, 0b000, "fcvtps", int_arm64_neon_fcvtps>;
defm FCVTPU : FPToIntegerUnscaled<0b01, 0b001, "fcvtpu", int_arm64_neon_fcvtpu>;
defm FCVTZS : FPToIntegerUnscaled<0b11, 0b000, "fcvtzs", fp_to_sint>;
defm FCVTZU : FPToIntegerUnscaled<0b11, 0b001, "fcvtzu", fp_to_uint>;
defm FCVTZS : FPToIntegerScaled<0b11, 0b000, "fcvtzs", fp_to_sint>;
defm FCVTZU : FPToIntegerScaled<0b11, 0b001, "fcvtzu", fp_to_uint>;
let isCodeGenOnly = 1 in {
defm FCVTZS_Int : FPToIntegerUnscaled<0b11, 0b000, "fcvtzs", int_arm64_neon_fcvtzs>;
defm FCVTZU_Int : FPToIntegerUnscaled<0b11, 0b001, "fcvtzu", int_arm64_neon_fcvtzu>;
defm FCVTZS_Int : FPToIntegerScaled<0b11, 0b000, "fcvtzs", int_arm64_neon_fcvtzs>;
defm FCVTZU_Int : FPToIntegerScaled<0b11, 0b001, "fcvtzu", int_arm64_neon_fcvtzu>;
}

//===----------------------------------------------------------------------===//
// Scaled integer to floating point conversion instructions.
//===----------------------------------------------------------------------===//

defm SCVTF : IntegerToFP<0, "scvtf", sint_to_fp>;
defm UCVTF : IntegerToFP<1, "ucvtf", uint_to_fp>;

//===----------------------------------------------------------------------===//
// Unscaled integer to floating point conversion instruction.
//===----------------------------------------------------------------------===//

defm FMOV : UnscaledConversion<"fmov">;

def : Pat<(f32 (fpimm0)), (FMOVWSr WZR)>, Requires<[NoZCZ]>;
def : Pat<(f64 (fpimm0)), (FMOVXDr XZR)>, Requires<[NoZCZ]>;

def : Pat<(v8i8  (bitconvert GPR64:$Xn)), (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v4i16 (bitconvert GPR64:$Xn)), (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v2i32 (bitconvert GPR64:$Xn)), (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v1i64 (bitconvert GPR64:$Xn)), (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v2f32 (bitconvert GPR64:$Xn)), (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v1f64 (bitconvert GPR64:$Xn)), (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v1i64 (scalar_to_vector GPR64:$Xn)),
          (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v1f64 (scalar_to_vector GPR64:$Xn)),
          (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(v1f64 (scalar_to_vector (f64 FPR64:$Xn))), (v1f64 FPR64:$Xn)>;

def : Pat<(i64 (bitconvert (v8i8  V64:$Vn))),
          (COPY_TO_REGCLASS V64:$Vn, GPR64)>;
def : Pat<(i64 (bitconvert (v4i16 V64:$Vn))),
          (COPY_TO_REGCLASS V64:$Vn, GPR64)>;
def : Pat<(i64 (bitconvert (v2i32 V64:$Vn))),
          (COPY_TO_REGCLASS V64:$Vn, GPR64)>;
def : Pat<(i64 (bitconvert (v1i64 V64:$Vn))),
          (COPY_TO_REGCLASS V64:$Vn, GPR64)>;
def : Pat<(i64 (bitconvert (v2f32 V64:$Vn))),
          (COPY_TO_REGCLASS V64:$Vn, GPR64)>;
def : Pat<(i64 (bitconvert (v1f64 V64:$Vn))),
          (COPY_TO_REGCLASS V64:$Vn, GPR64)>;

def : Pat<(f32 (bitconvert (i32 GPR32:$Xn))),
          (COPY_TO_REGCLASS GPR32:$Xn, FPR32)>;
def : Pat<(i32 (bitconvert (f32 FPR32:$Xn))),
          (COPY_TO_REGCLASS FPR32:$Xn, GPR32)>;
def : Pat<(f64 (bitconvert (i64 GPR64:$Xn))),
          (COPY_TO_REGCLASS GPR64:$Xn, FPR64)>;
def : Pat<(i64 (bitconvert (f64 FPR64:$Xn))),
          (COPY_TO_REGCLASS FPR64:$Xn, GPR64)>;

//===----------------------------------------------------------------------===//
// Floating point conversion instruction.
//===----------------------------------------------------------------------===//

defm FCVT : FPConversion<"fcvt">;

def : Pat<(f32_to_f16 FPR32:$Rn),
          (i32 (COPY_TO_REGCLASS
                   (f32 (SUBREG_TO_REG (i32 0), (FCVTHSr FPR32:$Rn), hsub)),
                   GPR32))>;

def FCVTSHpseudo : Pseudo<(outs FPR32:$Rd), (ins FPR32:$Rn),
                          [(set (f32 FPR32:$Rd), (f16_to_f32 i32:$Rn))]>;

//===----------------------------------------------------------------------===//
// Floating point single operand instructions.
//===----------------------------------------------------------------------===//

defm FABS   : SingleOperandFPData<0b0001, "fabs", fabs>;
defm FMOV   : SingleOperandFPData<0b0000, "fmov">;
defm FNEG   : SingleOperandFPData<0b0010, "fneg", fneg>;
defm FRINTA : SingleOperandFPData<0b1100, "frinta", frnd>;
defm FRINTI : SingleOperandFPData<0b1111, "frinti", fnearbyint>;
defm FRINTM : SingleOperandFPData<0b1010, "frintm", ffloor>;
defm FRINTN : SingleOperandFPData<0b1000, "frintn", int_arm64_neon_frintn>;
defm FRINTP : SingleOperandFPData<0b1001, "frintp", fceil>;

def : Pat<(v1f64 (int_arm64_neon_frintn (v1f64 FPR64:$Rn))),
          (FRINTNDr FPR64:$Rn)>;

// FRINTX is inserted to set the flags as required by FENV_ACCESS ON behavior
// in the C spec. Setting hasSideEffects ensures it is not DCE'd.
// <rdar://problem/13715968>
// TODO: We should really model the FPSR flags correctly. This is really ugly.
let hasSideEffects = 1 in {
defm FRINTX : SingleOperandFPData<0b1110, "frintx", frint>;
}

defm FRINTZ : SingleOperandFPData<0b1011, "frintz", ftrunc>;

let SchedRW = [WriteFDiv] in {
defm FSQRT  : SingleOperandFPData<0b0011, "fsqrt", fsqrt>;
}

//===----------------------------------------------------------------------===//
// Floating point two operand instructions.
//===----------------------------------------------------------------------===//

defm FADD   : TwoOperandFPData<0b0010, "fadd", fadd>;
let SchedRW = [WriteFDiv] in {
defm FDIV   : TwoOperandFPData<0b0001, "fdiv", fdiv>;
}
defm FMAXNM : TwoOperandFPData<0b0110, "fmaxnm", int_arm64_neon_fmaxnm>;
defm FMAX   : TwoOperandFPData<0b0100, "fmax", ARM64fmax>;
defm FMINNM : TwoOperandFPData<0b0111, "fminnm", int_arm64_neon_fminnm>;
defm FMIN   : TwoOperandFPData<0b0101, "fmin", ARM64fmin>;
let SchedRW = [WriteFMul] in {
defm FMUL   : TwoOperandFPData<0b0000, "fmul", fmul>;
defm FNMUL  : TwoOperandFPDataNeg<0b1000, "fnmul", fmul>;
}
defm FSUB   : TwoOperandFPData<0b0011, "fsub", fsub>;

def : Pat<(v1f64 (ARM64fmax (v1f64 FPR64:$Rn), (v1f64 FPR64:$Rm))),
          (FMAXDrr FPR64:$Rn, FPR64:$Rm)>;
def : Pat<(v1f64 (ARM64fmin (v1f64 FPR64:$Rn), (v1f64 FPR64:$Rm))),
          (FMINDrr FPR64:$Rn, FPR64:$Rm)>;
def : Pat<(v1f64 (int_arm64_neon_fmaxnm (v1f64 FPR64:$Rn), (v1f64 FPR64:$Rm))),
          (FMAXNMDrr FPR64:$Rn, FPR64:$Rm)>;
def : Pat<(v1f64 (int_arm64_neon_fminnm (v1f64 FPR64:$Rn), (v1f64 FPR64:$Rm))),
          (FMINNMDrr FPR64:$Rn, FPR64:$Rm)>;

//===----------------------------------------------------------------------===//
// Floating point three operand instructions.
//===----------------------------------------------------------------------===//

defm FMADD  : ThreeOperandFPData<0, 0, "fmadd", fma>;
defm FMSUB  : ThreeOperandFPData<0, 1, "fmsub",
     TriOpFrag<(fma node:$LHS, (fneg node:$MHS), node:$RHS)> >;
defm FNMADD : ThreeOperandFPData<1, 0, "fnmadd",
     TriOpFrag<(fneg (fma node:$LHS, node:$MHS, node:$RHS))> >;
defm FNMSUB : ThreeOperandFPData<1, 1, "fnmsub",
     TriOpFrag<(fma node:$LHS, node:$MHS, (fneg node:$RHS))> >;

// The following def pats catch the case where the LHS of an FMA is negated.
// The TriOpFrag above catches the case where the middle operand is negated.

// N.b. FMSUB etc have the accumulator at the *end* of (outs), unlike
// the NEON variant.
def : Pat<(f32 (fma (fneg FPR32:$Rn), FPR32:$Rm, FPR32:$Ra)),
          (FMSUBSrrr FPR32:$Rn, FPR32:$Rm, FPR32:$Ra)>;

def : Pat<(f64 (fma (fneg FPR64:$Rn), FPR64:$Rm, FPR64:$Ra)),
          (FMSUBDrrr FPR64:$Rn, FPR64:$Rm, FPR64:$Ra)>;

//===----------------------------------------------------------------------===//
// Floating point comparison instructions.
//===----------------------------------------------------------------------===//

defm FCMPE : FPComparison<1, "fcmpe">;
defm FCMP  : FPComparison<0, "fcmp", ARM64fcmp>;

//===----------------------------------------------------------------------===//
// Floating point conditional comparison instructions.
//===----------------------------------------------------------------------===//

defm FCCMPE : FPCondComparison<1, "fccmpe">;
defm FCCMP  : FPCondComparison<0, "fccmp">;

//===----------------------------------------------------------------------===//
// Floating point conditional select instruction.
//===----------------------------------------------------------------------===//

defm FCSEL : FPCondSelect<"fcsel">;

// CSEL instructions providing f128 types need to be handled by a
// pseudo-instruction since the eventual code will need to introduce basic
// blocks and control flow.
def F128CSEL : Pseudo<(outs FPR128:$Rd),
                      (ins FPR128:$Rn, FPR128:$Rm, ccode:$cond),
                      [(set (f128 FPR128:$Rd),
                            (ARM64csel FPR128:$Rn, FPR128:$Rm,
                                       (i32 imm:$cond), CPSR))]> {
  let Uses = [CPSR];
  let usesCustomInserter = 1;
}


//===----------------------------------------------------------------------===//
// Floating point immediate move.
//===----------------------------------------------------------------------===//

let isReMaterializable = 1 in {
defm FMOV : FPMoveImmediate<"fmov">;
}

//===----------------------------------------------------------------------===//
// Advanced SIMD two vector instructions.
//===----------------------------------------------------------------------===//

defm ABS    : SIMDTwoVectorBHSD<0, 0b01011, "abs", int_arm64_neon_abs>;
defm CLS    : SIMDTwoVectorBHS<0, 0b00100, "cls", int_arm64_neon_cls>;
defm CLZ    : SIMDTwoVectorBHS<1, 0b00100, "clz", ctlz>;
defm CMEQ   : SIMDCmpTwoVector<0, 0b01001, "cmeq", ARM64cmeqz>;
defm CMGE   : SIMDCmpTwoVector<1, 0b01000, "cmge", ARM64cmgez>;
defm CMGT   : SIMDCmpTwoVector<0, 0b01000, "cmgt", ARM64cmgtz>;
defm CMLE   : SIMDCmpTwoVector<1, 0b01001, "cmle", ARM64cmlez>;
defm CMLT   : SIMDCmpTwoVector<0, 0b01010, "cmlt", ARM64cmltz>;
defm CNT    : SIMDTwoVectorB<0, 0b00, 0b00101, "cnt", ctpop>;
defm FABS   : SIMDTwoVectorFP<0, 1, 0b01111, "fabs", fabs>;

defm FCMEQ  : SIMDFPCmpTwoVector<0, 1, 0b01101, "fcmeq", ARM64fcmeqz>;
defm FCMGE  : SIMDFPCmpTwoVector<1, 1, 0b01100, "fcmge", ARM64fcmgez>;
defm FCMGT  : SIMDFPCmpTwoVector<0, 1, 0b01100, "fcmgt", ARM64fcmgtz>;
defm FCMLE  : SIMDFPCmpTwoVector<1, 1, 0b01101, "fcmle", ARM64fcmlez>;
defm FCMLT  : SIMDFPCmpTwoVector<0, 1, 0b01110, "fcmlt", ARM64fcmltz>;
defm FCVTAS : SIMDTwoVectorFPToInt<0,0,0b11100, "fcvtas",int_arm64_neon_fcvtas>;
defm FCVTAU : SIMDTwoVectorFPToInt<1,0,0b11100, "fcvtau",int_arm64_neon_fcvtau>;
defm FCVTL  : SIMDFPWidenTwoVector<0, 0, 0b10111, "fcvtl">;
def : Pat<(v4f32 (int_arm64_neon_vcvthf2fp (v4i16 V64:$Rn))),
          (FCVTLv4i16 V64:$Rn)>;
def : Pat<(v4f32 (int_arm64_neon_vcvthf2fp (extract_subvector (v8i16 V128:$Rn),
                                                              (i64 4)))),
          (FCVTLv8i16 V128:$Rn)>;
def : Pat<(v2f64 (fextend (v2f32 V64:$Rn))), (FCVTLv2i32 V64:$Rn)>;
def : Pat<(v2f64 (fextend (v2f32 (extract_subvector (v4f32 V128:$Rn),
                                                    (i64 2))))),
          (FCVTLv4i32 V128:$Rn)>;

defm FCVTMS : SIMDTwoVectorFPToInt<0,0,0b11011, "fcvtms",int_arm64_neon_fcvtms>;
defm FCVTMU : SIMDTwoVectorFPToInt<1,0,0b11011, "fcvtmu",int_arm64_neon_fcvtmu>;
defm FCVTNS : SIMDTwoVectorFPToInt<0,0,0b11010, "fcvtns",int_arm64_neon_fcvtns>;
defm FCVTNU : SIMDTwoVectorFPToInt<1,0,0b11010, "fcvtnu",int_arm64_neon_fcvtnu>;
defm FCVTN  : SIMDFPNarrowTwoVector<0, 0, 0b10110, "fcvtn">;
def : Pat<(v4i16 (int_arm64_neon_vcvtfp2hf (v4f32 V128:$Rn))),
          (FCVTNv4i16 V128:$Rn)>;
def : Pat<(concat_vectors V64:$Rd,
                          (v4i16 (int_arm64_neon_vcvtfp2hf (v4f32 V128:$Rn)))),
          (FCVTNv8i16 (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rd, dsub), V128:$Rn)>;
def : Pat<(v2f32 (fround (v2f64 V128:$Rn))), (FCVTNv2i32 V128:$Rn)>;
def : Pat<(concat_vectors V64:$Rd, (v2f32 (fround (v2f64 V128:$Rn)))),
          (FCVTNv4i32 (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rd, dsub), V128:$Rn)>;
defm FCVTPS : SIMDTwoVectorFPToInt<0,1,0b11010, "fcvtps",int_arm64_neon_fcvtps>;
defm FCVTPU : SIMDTwoVectorFPToInt<1,1,0b11010, "fcvtpu",int_arm64_neon_fcvtpu>;
defm FCVTXN : SIMDFPInexactCvtTwoVector<1, 0, 0b10110, "fcvtxn",
                                        int_arm64_neon_fcvtxn>;
defm FCVTZS : SIMDTwoVectorFPToInt<0, 1, 0b11011, "fcvtzs", fp_to_sint>;
defm FCVTZU : SIMDTwoVectorFPToInt<1, 1, 0b11011, "fcvtzu", fp_to_uint>;
let isCodeGenOnly = 1 in {
defm FCVTZS_Int : SIMDTwoVectorFPToInt<0, 1, 0b11011, "fcvtzs",
                                       int_arm64_neon_fcvtzs>;
defm FCVTZU_Int : SIMDTwoVectorFPToInt<1, 1, 0b11011, "fcvtzu",
                                       int_arm64_neon_fcvtzu>;
}
defm FNEG   : SIMDTwoVectorFP<1, 1, 0b01111, "fneg", fneg>;
defm FRECPE : SIMDTwoVectorFP<0, 1, 0b11101, "frecpe", int_arm64_neon_frecpe>;
defm FRINTA : SIMDTwoVectorFP<1, 0, 0b11000, "frinta", frnd>;
defm FRINTI : SIMDTwoVectorFP<1, 1, 0b11001, "frinti", fnearbyint>;
defm FRINTM : SIMDTwoVectorFP<0, 0, 0b11001, "frintm", ffloor>;
defm FRINTN : SIMDTwoVectorFP<0, 0, 0b11000, "frintn", int_arm64_neon_frintn>;
defm FRINTP : SIMDTwoVectorFP<0, 1, 0b11000, "frintp", fceil>;
defm FRINTX : SIMDTwoVectorFP<1, 0, 0b11001, "frintx", frint>;
defm FRINTZ : SIMDTwoVectorFP<0, 1, 0b11001, "frintz", ftrunc>;
defm FRSQRTE: SIMDTwoVectorFP<1, 1, 0b11101, "frsqrte", int_arm64_neon_frsqrte>;
defm FSQRT  : SIMDTwoVectorFP<1, 1, 0b11111, "fsqrt", fsqrt>;
defm NEG    : SIMDTwoVectorBHSD<1, 0b01011, "neg",
                               UnOpFrag<(sub immAllZerosV, node:$LHS)> >;
defm NOT    : SIMDTwoVectorB<1, 0b00, 0b00101, "not", vnot>;
// Aliases for MVN -> NOT.
def : InstAlias<"mvn.8b $Vd, $Vn", (NOTv8i8 V64:$Vd, V64:$Vn)>;
def : InstAlias<"mvn.16b $Vd, $Vn", (NOTv16i8 V128:$Vd, V128:$Vn)>;
def : InstAlias<"mvn $Vd.8b, $Vn.8b", (NOTv8i8 V64:$Vd, V64:$Vn)>;
def : InstAlias<"mvn $Vd.16b, $Vn.16b", (NOTv16i8 V128:$Vd, V128:$Vn)>;

def : Pat<(ARM64neg (v8i8  V64:$Rn)),  (NEGv8i8  V64:$Rn)>;
def : Pat<(ARM64neg (v16i8 V128:$Rn)), (NEGv16i8 V128:$Rn)>;
def : Pat<(ARM64neg (v4i16 V64:$Rn)),  (NEGv4i16 V64:$Rn)>;
def : Pat<(ARM64neg (v8i16 V128:$Rn)), (NEGv8i16 V128:$Rn)>;
def : Pat<(ARM64neg (v2i32 V64:$Rn)),  (NEGv2i32 V64:$Rn)>;
def : Pat<(ARM64neg (v4i32 V128:$Rn)), (NEGv4i32 V128:$Rn)>;
def : Pat<(ARM64neg (v2i64 V128:$Rn)), (NEGv2i64 V128:$Rn)>;

def : Pat<(ARM64not (v8i8 V64:$Rn)),   (NOTv8i8  V64:$Rn)>;
def : Pat<(ARM64not (v16i8 V128:$Rn)), (NOTv16i8 V128:$Rn)>;
def : Pat<(ARM64not (v4i16 V64:$Rn)),  (NOTv8i8  V64:$Rn)>;
def : Pat<(ARM64not (v8i16 V128:$Rn)), (NOTv16i8 V128:$Rn)>;
def : Pat<(ARM64not (v2i32 V64:$Rn)),  (NOTv8i8  V64:$Rn)>;
def : Pat<(ARM64not (v1i64 V64:$Rn)),  (NOTv8i8  V64:$Rn)>;
def : Pat<(ARM64not (v4i32 V128:$Rn)), (NOTv16i8 V128:$Rn)>;
def : Pat<(ARM64not (v2i64 V128:$Rn)), (NOTv16i8 V128:$Rn)>;

def : Pat<(vnot (v4i16 V64:$Rn)),  (NOTv8i8  V64:$Rn)>;
def : Pat<(vnot (v8i16 V128:$Rn)), (NOTv16i8 V128:$Rn)>;
def : Pat<(vnot (v2i32 V64:$Rn)),  (NOTv8i8  V64:$Rn)>;
def : Pat<(vnot (v4i32 V128:$Rn)), (NOTv16i8 V128:$Rn)>;
def : Pat<(vnot (v2i64 V128:$Rn)), (NOTv16i8 V128:$Rn)>;

defm RBIT   : SIMDTwoVectorB<1, 0b01, 0b00101, "rbit", int_arm64_neon_rbit>;
defm REV16  : SIMDTwoVectorB<0, 0b00, 0b00001, "rev16", ARM64rev16>;
defm REV32  : SIMDTwoVectorBH<1, 0b00000, "rev32", ARM64rev32>;
defm REV64  : SIMDTwoVectorBHS<0, 0b00000, "rev64", ARM64rev64>;
defm SADALP : SIMDLongTwoVectorTied<0, 0b00110, "sadalp",
       BinOpFrag<(add node:$LHS, (int_arm64_neon_saddlp node:$RHS))> >;
defm SADDLP : SIMDLongTwoVector<0, 0b00010, "saddlp", int_arm64_neon_saddlp>;
defm SCVTF  : SIMDTwoVectorIntToFP<0, 0, 0b11101, "scvtf", sint_to_fp>;
defm SHLL   : SIMDVectorLShiftLongBySizeBHS;
defm SQABS  : SIMDTwoVectorBHSD<0, 0b00111, "sqabs", int_arm64_neon_sqabs>;
defm SQNEG  : SIMDTwoVectorBHSD<1, 0b00111, "sqneg", int_arm64_neon_sqneg>;
defm SQXTN  : SIMDMixedTwoVector<0, 0b10100, "sqxtn", int_arm64_neon_sqxtn>;
defm SQXTUN : SIMDMixedTwoVector<1, 0b10010, "sqxtun", int_arm64_neon_sqxtun>;
defm SUQADD : SIMDTwoVectorBHSDTied<0, 0b00011, "suqadd",int_arm64_neon_suqadd>;
defm UADALP : SIMDLongTwoVectorTied<1, 0b00110, "uadalp",
       BinOpFrag<(add node:$LHS, (int_arm64_neon_uaddlp node:$RHS))> >;
defm UADDLP : SIMDLongTwoVector<1, 0b00010, "uaddlp",
                    int_arm64_neon_uaddlp>;
defm UCVTF  : SIMDTwoVectorIntToFP<1, 0, 0b11101, "ucvtf", uint_to_fp>;
defm UQXTN  : SIMDMixedTwoVector<1, 0b10100, "uqxtn", int_arm64_neon_uqxtn>;
defm URECPE : SIMDTwoVectorS<0, 1, 0b11100, "urecpe", int_arm64_neon_urecpe>;
defm URSQRTE: SIMDTwoVectorS<1, 1, 0b11100, "ursqrte", int_arm64_neon_ursqrte>;
defm USQADD : SIMDTwoVectorBHSDTied<1, 0b00011, "usqadd",int_arm64_neon_usqadd>;
defm XTN    : SIMDMixedTwoVector<0, 0b10010, "xtn", trunc>;

def : Pat<(v2f32 (ARM64rev64 V64:$Rn)), (REV64v2i32 V64:$Rn)>;
def : Pat<(v4f32 (ARM64rev64 V128:$Rn)), (REV64v4i32 V128:$Rn)>;

// Patterns for vector long shift (by element width). These need to match all
// three of zext, sext and anyext so it's easier to pull the patterns out of the
// definition.
multiclass SIMDVectorLShiftLongBySizeBHSPats<SDPatternOperator ext> {
  def : Pat<(ARM64vshl (v8i16 (ext (v8i8 V64:$Rn))), (i32 8)),
            (SHLLv8i8 V64:$Rn)>;
  def : Pat<(ARM64vshl (v8i16 (ext (extract_high_v16i8 V128:$Rn))), (i32 8)),
            (SHLLv16i8 V128:$Rn)>;
  def : Pat<(ARM64vshl (v4i32 (ext (v4i16 V64:$Rn))), (i32 16)),
            (SHLLv4i16 V64:$Rn)>;
  def : Pat<(ARM64vshl (v4i32 (ext (extract_high_v8i16 V128:$Rn))), (i32 16)),
            (SHLLv8i16 V128:$Rn)>;
  def : Pat<(ARM64vshl (v2i64 (ext (v2i32 V64:$Rn))), (i32 32)),
            (SHLLv2i32 V64:$Rn)>;
  def : Pat<(ARM64vshl (v2i64 (ext (extract_high_v4i32 V128:$Rn))), (i32 32)),
            (SHLLv4i32 V128:$Rn)>;
}

defm : SIMDVectorLShiftLongBySizeBHSPats<anyext>;
defm : SIMDVectorLShiftLongBySizeBHSPats<zext>;
defm : SIMDVectorLShiftLongBySizeBHSPats<sext>;

//===----------------------------------------------------------------------===//
// Advanced SIMD three vector instructions.
//===----------------------------------------------------------------------===//

defm ADD     : SIMDThreeSameVector<0, 0b10000, "add", add>;
defm ADDP    : SIMDThreeSameVector<0, 0b10111, "addp", int_arm64_neon_addp>;
defm CMEQ    : SIMDThreeSameVector<1, 0b10001, "cmeq", ARM64cmeq>;
defm CMGE    : SIMDThreeSameVector<0, 0b00111, "cmge", ARM64cmge>;
defm CMGT    : SIMDThreeSameVector<0, 0b00110, "cmgt", ARM64cmgt>;
defm CMHI    : SIMDThreeSameVector<1, 0b00110, "cmhi", ARM64cmhi>;
defm CMHS    : SIMDThreeSameVector<1, 0b00111, "cmhs", ARM64cmhs>;
defm CMTST   : SIMDThreeSameVector<0, 0b10001, "cmtst", ARM64cmtst>;
defm FABD    : SIMDThreeSameVectorFP<1,1,0b11010,"fabd", int_arm64_neon_fabd>;
defm FACGE   : SIMDThreeSameVectorFPCmp<1,0,0b11101,"facge",int_arm64_neon_facge>;
defm FACGT   : SIMDThreeSameVectorFPCmp<1,1,0b11101,"facgt",int_arm64_neon_facgt>;
defm FADDP   : SIMDThreeSameVectorFP<1,0,0b11010,"faddp",int_arm64_neon_addp>;
defm FADD    : SIMDThreeSameVectorFP<0,0,0b11010,"fadd", fadd>;
defm FCMEQ   : SIMDThreeSameVectorFPCmp<0, 0, 0b11100, "fcmeq", ARM64fcmeq>;
defm FCMGE   : SIMDThreeSameVectorFPCmp<1, 0, 0b11100, "fcmge", ARM64fcmge>;
defm FCMGT   : SIMDThreeSameVectorFPCmp<1, 1, 0b11100, "fcmgt", ARM64fcmgt>;
defm FDIV    : SIMDThreeSameVectorFP<1,0,0b11111,"fdiv", fdiv>;
defm FMAXNMP : SIMDThreeSameVectorFP<1,0,0b11000,"fmaxnmp", int_arm64_neon_fmaxnmp>;
defm FMAXNM  : SIMDThreeSameVectorFP<0,0,0b11000,"fmaxnm", int_arm64_neon_fmaxnm>;
defm FMAXP   : SIMDThreeSameVectorFP<1,0,0b11110,"fmaxp", int_arm64_neon_fmaxp>;
defm FMAX    : SIMDThreeSameVectorFP<0,0,0b11110,"fmax", ARM64fmax>;
defm FMINNMP : SIMDThreeSameVectorFP<1,1,0b11000,"fminnmp", int_arm64_neon_fminnmp>;
defm FMINNM  : SIMDThreeSameVectorFP<0,1,0b11000,"fminnm", int_arm64_neon_fminnm>;
defm FMINP   : SIMDThreeSameVectorFP<1,1,0b11110,"fminp", int_arm64_neon_fminp>;
defm FMIN    : SIMDThreeSameVectorFP<0,1,0b11110,"fmin", ARM64fmin>;

// NOTE: The operands of the PatFrag are reordered on FMLA/FMLS because the
// instruction expects the addend first, while the fma intrinsic puts it last.
defm FMLA     : SIMDThreeSameVectorFPTied<0, 0, 0b11001, "fmla",
            TriOpFrag<(fma node:$RHS, node:$MHS, node:$LHS)> >;
defm FMLS     : SIMDThreeSameVectorFPTied<0, 1, 0b11001, "fmls",
            TriOpFrag<(fma node:$MHS, (fneg node:$RHS), node:$LHS)> >;

// The following def pats catch the case where the LHS of an FMA is negated.
// The TriOpFrag above catches the case where the middle operand is negated.
def : Pat<(v2f32 (fma (fneg V64:$Rn), V64:$Rm, V64:$Rd)),
          (FMLSv2f32 V64:$Rd, V64:$Rn, V64:$Rm)>;

def : Pat<(v4f32 (fma (fneg V128:$Rn), V128:$Rm, V128:$Rd)),
          (FMLSv4f32 V128:$Rd, V128:$Rn, V128:$Rm)>;

def : Pat<(v2f64 (fma (fneg V128:$Rn), V128:$Rm, V128:$Rd)),
          (FMLSv2f64 V128:$Rd, V128:$Rn, V128:$Rm)>;

defm FMULX    : SIMDThreeSameVectorFP<0,0,0b11011,"fmulx", int_arm64_neon_fmulx>;
defm FMUL     : SIMDThreeSameVectorFP<1,0,0b11011,"fmul", fmul>;
defm FRECPS   : SIMDThreeSameVectorFP<0,0,0b11111,"frecps", int_arm64_neon_frecps>;
defm FRSQRTS  : SIMDThreeSameVectorFP<0,1,0b11111,"frsqrts", int_arm64_neon_frsqrts>;
defm FSUB     : SIMDThreeSameVectorFP<0,1,0b11010,"fsub", fsub>;
defm MLA      : SIMDThreeSameVectorBHSTied<0, 0b10010, "mla",
                      TriOpFrag<(add node:$LHS, (mul node:$MHS, node:$RHS))> >;
defm MLS      : SIMDThreeSameVectorBHSTied<1, 0b10010, "mls",
                      TriOpFrag<(sub node:$LHS, (mul node:$MHS, node:$RHS))> >;
defm MUL      : SIMDThreeSameVectorBHS<0, 0b10011, "mul", mul>;
defm PMUL     : SIMDThreeSameVectorB<1, 0b10011, "pmul", int_arm64_neon_pmul>;
defm SABA     : SIMDThreeSameVectorBHSTied<0, 0b01111, "saba",
      TriOpFrag<(add node:$LHS, (int_arm64_neon_sabd node:$MHS, node:$RHS))> >;
defm SABD     : SIMDThreeSameVectorBHS<0,0b01110,"sabd", int_arm64_neon_sabd>;
defm SHADD    : SIMDThreeSameVectorBHS<0,0b00000,"shadd", int_arm64_neon_shadd>;
defm SHSUB    : SIMDThreeSameVectorBHS<0,0b00100,"shsub", int_arm64_neon_shsub>;
defm SMAXP    : SIMDThreeSameVectorBHS<0,0b10100,"smaxp", int_arm64_neon_smaxp>;
defm SMAX     : SIMDThreeSameVectorBHS<0,0b01100,"smax", int_arm64_neon_smax>;
defm SMINP    : SIMDThreeSameVectorBHS<0,0b10101,"sminp", int_arm64_neon_sminp>;
defm SMIN     : SIMDThreeSameVectorBHS<0,0b01101,"smin", int_arm64_neon_smin>;
defm SQADD    : SIMDThreeSameVector<0,0b00001,"sqadd", int_arm64_neon_sqadd>;
defm SQDMULH  : SIMDThreeSameVectorHS<0,0b10110,"sqdmulh",int_arm64_neon_sqdmulh>;
defm SQRDMULH : SIMDThreeSameVectorHS<1,0b10110,"sqrdmulh",int_arm64_neon_sqrdmulh>;
defm SQRSHL   : SIMDThreeSameVector<0,0b01011,"sqrshl", int_arm64_neon_sqrshl>;
defm SQSHL    : SIMDThreeSameVector<0,0b01001,"sqshl", int_arm64_neon_sqshl>;
defm SQSUB    : SIMDThreeSameVector<0,0b00101,"sqsub", int_arm64_neon_sqsub>;
defm SRHADD   : SIMDThreeSameVectorBHS<0,0b00010,"srhadd",int_arm64_neon_srhadd>;
defm SRSHL    : SIMDThreeSameVector<0,0b01010,"srshl", int_arm64_neon_srshl>;
defm SSHL     : SIMDThreeSameVector<0,0b01000,"sshl", int_arm64_neon_sshl>;
defm SUB      : SIMDThreeSameVector<1,0b10000,"sub", sub>;
defm UABA     : SIMDThreeSameVectorBHSTied<1, 0b01111, "uaba",
      TriOpFrag<(add node:$LHS, (int_arm64_neon_uabd node:$MHS, node:$RHS))> >;
defm UABD     : SIMDThreeSameVectorBHS<1,0b01110,"uabd", int_arm64_neon_uabd>;
defm UHADD    : SIMDThreeSameVectorBHS<1,0b00000,"uhadd", int_arm64_neon_uhadd>;
defm UHSUB    : SIMDThreeSameVectorBHS<1,0b00100,"uhsub", int_arm64_neon_uhsub>;
defm UMAXP    : SIMDThreeSameVectorBHS<1,0b10100,"umaxp", int_arm64_neon_umaxp>;
defm UMAX     : SIMDThreeSameVectorBHS<1,0b01100,"umax", int_arm64_neon_umax>;
defm UMINP    : SIMDThreeSameVectorBHS<1,0b10101,"uminp", int_arm64_neon_uminp>;
defm UMIN     : SIMDThreeSameVectorBHS<1,0b01101,"umin", int_arm64_neon_umin>;
defm UQADD    : SIMDThreeSameVector<1,0b00001,"uqadd", int_arm64_neon_uqadd>;
defm UQRSHL   : SIMDThreeSameVector<1,0b01011,"uqrshl", int_arm64_neon_uqrshl>;
defm UQSHL    : SIMDThreeSameVector<1,0b01001,"uqshl", int_arm64_neon_uqshl>;
defm UQSUB    : SIMDThreeSameVector<1,0b00101,"uqsub", int_arm64_neon_uqsub>;
defm URHADD   : SIMDThreeSameVectorBHS<1,0b00010,"urhadd", int_arm64_neon_urhadd>;
defm URSHL    : SIMDThreeSameVector<1,0b01010,"urshl", int_arm64_neon_urshl>;
defm USHL     : SIMDThreeSameVector<1,0b01000,"ushl", int_arm64_neon_ushl>;

defm AND : SIMDLogicalThreeVector<0, 0b00, "and", and>;
defm BIC : SIMDLogicalThreeVector<0, 0b01, "bic",
                                  BinOpFrag<(and node:$LHS, (vnot node:$RHS))> >;
defm BIF : SIMDLogicalThreeVector<1, 0b11, "bif">;
defm BIT : SIMDLogicalThreeVectorTied<1, 0b10, "bit", ARM64bit>;
defm BSL : SIMDLogicalThreeVectorTied<1, 0b01, "bsl",
    TriOpFrag<(or (and node:$LHS, node:$MHS), (and (vnot node:$LHS), node:$RHS))>>;
defm EOR : SIMDLogicalThreeVector<1, 0b00, "eor", xor>;
defm ORN : SIMDLogicalThreeVector<0, 0b11, "orn",
                                  BinOpFrag<(or node:$LHS, (vnot node:$RHS))> >;
defm ORR : SIMDLogicalThreeVector<0, 0b10, "orr", or>;

// FIXME: the .16b and .8b variantes should be emitted by the
// AsmWriter. TableGen's AsmWriter-generator doesn't deal with variant syntaxes
// in aliases yet though.
def : InstAlias<"mov{\t$dst.16b, $src.16b|.16b\t$dst, $src}",
                (ORRv16i8 V128:$dst, V128:$src, V128:$src), 0>;
def : InstAlias<"{mov\t$dst.8h, $src.8h|mov.8h\t$dst, $src}",
                (ORRv16i8 V128:$dst, V128:$src, V128:$src), 0>;
def : InstAlias<"{mov\t$dst.4s, $src.4s|mov.4s\t$dst, $src}",
                (ORRv16i8 V128:$dst, V128:$src, V128:$src), 0>;
def : InstAlias<"{mov\t$dst.2d, $src.2d|mov.2d\t$dst, $src}",
                (ORRv16i8 V128:$dst, V128:$src, V128:$src), 0>;

def : InstAlias<"{mov\t$dst.8b, $src.8b|mov.8b\t$dst, $src}",
                (ORRv8i8 V64:$dst, V64:$src, V64:$src), 0>;
def : InstAlias<"{mov\t$dst.4h, $src.4h|mov.4h\t$dst, $src}",
                (ORRv8i8 V64:$dst, V64:$src, V64:$src), 0>;
def : InstAlias<"{mov\t$dst.2s, $src.2s|mov.2s\t$dst, $src}",
                (ORRv8i8 V64:$dst, V64:$src, V64:$src), 0>;
def : InstAlias<"{mov\t$dst.1d, $src.1d|mov.1d\t$dst, $src}",
                (ORRv8i8 V64:$dst, V64:$src, V64:$src), 0>;

def : InstAlias<"{cmls\t$dst.8b, $src1.8b, $src2.8b" #
                "|cmls.8b\t$dst, $src1, $src2}",
                (CMHSv8i8 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmls\t$dst.16b, $src1.16b, $src2.16b" #
                "|cmls.16b\t$dst, $src1, $src2}",
                (CMHSv16i8 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmls\t$dst.4h, $src1.4h, $src2.4h" #
                "|cmls.4h\t$dst, $src1, $src2}",
                (CMHSv4i16 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmls\t$dst.8h, $src1.8h, $src2.8h" #
                "|cmls.8h\t$dst, $src1, $src2}",
                (CMHSv8i16 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmls\t$dst.2s, $src1.2s, $src2.2s" #
                "|cmls.2s\t$dst, $src1, $src2}",
                (CMHSv2i32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmls\t$dst.4s, $src1.4s, $src2.4s" #
                "|cmls.4s\t$dst, $src1, $src2}",
                (CMHSv4i32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmls\t$dst.2d, $src1.2d, $src2.2d" #
                "|cmls.2d\t$dst, $src1, $src2}",
                (CMHSv2i64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{cmlo\t$dst.8b, $src1.8b, $src2.8b" #
                "|cmlo.8b\t$dst, $src1, $src2}",
                (CMHIv8i8 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmlo\t$dst.16b, $src1.16b, $src2.16b" #
                "|cmlo.16b\t$dst, $src1, $src2}",
                (CMHIv16i8 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmlo\t$dst.4h, $src1.4h, $src2.4h" #
                "|cmlo.4h\t$dst, $src1, $src2}",
                (CMHIv4i16 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmlo\t$dst.8h, $src1.8h, $src2.8h" #
                "|cmlo.8h\t$dst, $src1, $src2}",
                (CMHIv8i16 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmlo\t$dst.2s, $src1.2s, $src2.2s" #
                "|cmlo.2s\t$dst, $src1, $src2}",
                (CMHIv2i32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmlo\t$dst.4s, $src1.4s, $src2.4s" #
                "|cmlo.4s\t$dst, $src1, $src2}",
                (CMHIv4i32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmlo\t$dst.2d, $src1.2d, $src2.2d" #
                "|cmlo.2d\t$dst, $src1, $src2}",
                (CMHIv2i64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{cmle\t$dst.8b, $src1.8b, $src2.8b" #
                "|cmle.8b\t$dst, $src1, $src2}",
                (CMGEv8i8 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmle\t$dst.16b, $src1.16b, $src2.16b" #
                "|cmle.16b\t$dst, $src1, $src2}",
                (CMGEv16i8 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmle\t$dst.4h, $src1.4h, $src2.4h" #
                "|cmle.4h\t$dst, $src1, $src2}",
                (CMGEv4i16 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmle\t$dst.8h, $src1.8h, $src2.8h" #
                "|cmle.8h\t$dst, $src1, $src2}",
                (CMGEv8i16 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmle\t$dst.2s, $src1.2s, $src2.2s" #
                "|cmle.2s\t$dst, $src1, $src2}",
                (CMGEv2i32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmle\t$dst.4s, $src1.4s, $src2.4s" #
                "|cmle.4s\t$dst, $src1, $src2}",
                (CMGEv4i32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmle\t$dst.2d, $src1.2d, $src2.2d" #
                "|cmle.2d\t$dst, $src1, $src2}",
                (CMGEv2i64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{cmlt\t$dst.8b, $src1.8b, $src2.8b" #
                "|cmlt.8b\t$dst, $src1, $src2}",
                (CMGTv8i8 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmlt\t$dst.16b, $src1.16b, $src2.16b" #
                "|cmlt.16b\t$dst, $src1, $src2}",
                (CMGTv16i8 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmlt\t$dst.4h, $src1.4h, $src2.4h" #
                "|cmlt.4h\t$dst, $src1, $src2}",
                (CMGTv4i16 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmlt\t$dst.8h, $src1.8h, $src2.8h" #
                "|cmlt.8h\t$dst, $src1, $src2}",
                (CMGTv8i16 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmlt\t$dst.2s, $src1.2s, $src2.2s" #
                "|cmlt.2s\t$dst, $src1, $src2}",
                (CMGTv2i32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{cmlt\t$dst.4s, $src1.4s, $src2.4s" #
                "|cmlt.4s\t$dst, $src1, $src2}",
                (CMGTv4i32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{cmlt\t$dst.2d, $src1.2d, $src2.2d" #
                "|cmlt.2d\t$dst, $src1, $src2}",
                (CMGTv2i64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{fcmle\t$dst.2s, $src1.2s, $src2.2s" #
                "|fcmle.2s\t$dst, $src1, $src2}",
                (FCMGEv2f32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{fcmle\t$dst.4s, $src1.4s, $src2.4s" #
                "|fcmle.4s\t$dst, $src1, $src2}",
                (FCMGEv4f32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{fcmle\t$dst.2d, $src1.2d, $src2.2d" #
                "|fcmle.2d\t$dst, $src1, $src2}",
                (FCMGEv2f64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{fcmlt\t$dst.2s, $src1.2s, $src2.2s" #
                "|fcmlt.2s\t$dst, $src1, $src2}",
                (FCMGTv2f32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{fcmlt\t$dst.4s, $src1.4s, $src2.4s" #
                "|fcmlt.4s\t$dst, $src1, $src2}",
                (FCMGTv4f32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{fcmlt\t$dst.2d, $src1.2d, $src2.2d" #
                "|fcmlt.2d\t$dst, $src1, $src2}",
                (FCMGTv2f64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{facle\t$dst.2s, $src1.2s, $src2.2s" #
                "|facle.2s\t$dst, $src1, $src2}",
                (FACGEv2f32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{facle\t$dst.4s, $src1.4s, $src2.4s" #
                "|facle.4s\t$dst, $src1, $src2}",
                (FACGEv4f32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{facle\t$dst.2d, $src1.2d, $src2.2d" #
                "|facle.2d\t$dst, $src1, $src2}",
                (FACGEv2f64 V128:$dst, V128:$src2, V128:$src1), 0>;

def : InstAlias<"{faclt\t$dst.2s, $src1.2s, $src2.2s" #
                "|faclt.2s\t$dst, $src1, $src2}",
                (FACGTv2f32 V64:$dst, V64:$src2, V64:$src1), 0>;
def : InstAlias<"{faclt\t$dst.4s, $src1.4s, $src2.4s" #
                "|faclt.4s\t$dst, $src1, $src2}",
                (FACGTv4f32 V128:$dst, V128:$src2, V128:$src1), 0>;
def : InstAlias<"{faclt\t$dst.2d, $src1.2d, $src2.2d" #
                "|faclt.2d\t$dst, $src1, $src2}",
                (FACGTv2f64 V128:$dst, V128:$src2, V128:$src1), 0>;

//===----------------------------------------------------------------------===//
// Advanced SIMD three scalar instructions.
//===----------------------------------------------------------------------===//

defm ADD      : SIMDThreeScalarD<0, 0b10000, "add", add>;
defm CMEQ     : SIMDThreeScalarD<1, 0b10001, "cmeq", ARM64cmeq>;
defm CMGE     : SIMDThreeScalarD<0, 0b00111, "cmge", ARM64cmge>;
defm CMGT     : SIMDThreeScalarD<0, 0b00110, "cmgt", ARM64cmgt>;
defm CMHI     : SIMDThreeScalarD<1, 0b00110, "cmhi", ARM64cmhi>;
defm CMHS     : SIMDThreeScalarD<1, 0b00111, "cmhs", ARM64cmhs>;
defm CMTST    : SIMDThreeScalarD<0, 0b10001, "cmtst", ARM64cmtst>;
defm FABD     : SIMDThreeScalarSD<1, 1, 0b11010, "fabd", int_arm64_sisd_fabd>;
def : Pat<(v1f64 (int_arm64_neon_fabd (v1f64 FPR64:$Rn), (v1f64 FPR64:$Rm))),
          (FABD64 FPR64:$Rn, FPR64:$Rm)>;
defm FACGE    : SIMDThreeScalarFPCmp<1, 0, 0b11101, "facge",
                                     int_arm64_neon_facge>;
defm FACGT    : SIMDThreeScalarFPCmp<1, 1, 0b11101, "facgt",
                                     int_arm64_neon_facgt>;
defm FCMEQ    : SIMDThreeScalarFPCmp<0, 0, 0b11100, "fcmeq", ARM64fcmeq>;
defm FCMGE    : SIMDThreeScalarFPCmp<1, 0, 0b11100, "fcmge", ARM64fcmge>;
defm FCMGT    : SIMDThreeScalarFPCmp<1, 1, 0b11100, "fcmgt", ARM64fcmgt>;
defm FMULX    : SIMDThreeScalarSD<0, 0, 0b11011, "fmulx", int_arm64_neon_fmulx>;
defm FRECPS   : SIMDThreeScalarSD<0, 0, 0b11111, "frecps", int_arm64_neon_frecps>;
defm FRSQRTS  : SIMDThreeScalarSD<0, 1, 0b11111, "frsqrts", int_arm64_neon_frsqrts>;
defm SQADD    : SIMDThreeScalarBHSD<0, 0b00001, "sqadd", int_arm64_neon_sqadd>;
defm SQDMULH  : SIMDThreeScalarHS<  0, 0b10110, "sqdmulh", int_arm64_neon_sqdmulh>;
defm SQRDMULH : SIMDThreeScalarHS<  1, 0b10110, "sqrdmulh", int_arm64_neon_sqrdmulh>;
defm SQRSHL   : SIMDThreeScalarBHSD<0, 0b01011, "sqrshl",int_arm64_neon_sqrshl>;
defm SQSHL    : SIMDThreeScalarBHSD<0, 0b01001, "sqshl", int_arm64_neon_sqshl>;
defm SQSUB    : SIMDThreeScalarBHSD<0, 0b00101, "sqsub", int_arm64_neon_sqsub>;
defm SRSHL    : SIMDThreeScalarD<   0, 0b01010, "srshl", int_arm64_neon_srshl>;
defm SSHL     : SIMDThreeScalarD<   0, 0b01000, "sshl", int_arm64_neon_sshl>;
defm SUB      : SIMDThreeScalarD<   1, 0b10000, "sub", sub>;
defm UQADD    : SIMDThreeScalarBHSD<1, 0b00001, "uqadd", int_arm64_neon_uqadd>;
defm UQRSHL   : SIMDThreeScalarBHSD<1, 0b01011, "uqrshl",int_arm64_neon_uqrshl>;
defm UQSHL    : SIMDThreeScalarBHSD<1, 0b01001, "uqshl", int_arm64_neon_uqshl>;
defm UQSUB    : SIMDThreeScalarBHSD<1, 0b00101, "uqsub", int_arm64_neon_uqsub>;
defm URSHL    : SIMDThreeScalarD<   1, 0b01010, "urshl", int_arm64_neon_urshl>;
defm USHL     : SIMDThreeScalarD<   1, 0b01000, "ushl", int_arm64_neon_ushl>;

def : InstAlias<"cmls $dst, $src1, $src2",
                (CMHSv1i64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"cmle $dst, $src1, $src2",
                (CMGEv1i64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"cmlo $dst, $src1, $src2",
                (CMHIv1i64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"cmlt $dst, $src1, $src2",
                (CMGTv1i64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"fcmle $dst, $src1, $src2",
                (FCMGE32 FPR32:$dst, FPR32:$src2, FPR32:$src1)>;
def : InstAlias<"fcmle $dst, $src1, $src2",
                (FCMGE64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"fcmlt $dst, $src1, $src2",
                (FCMGT32 FPR32:$dst, FPR32:$src2, FPR32:$src1)>;
def : InstAlias<"fcmlt $dst, $src1, $src2",
                (FCMGT64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"facle $dst, $src1, $src2",
                (FACGE32 FPR32:$dst, FPR32:$src2, FPR32:$src1)>;
def : InstAlias<"facle $dst, $src1, $src2",
                (FACGE64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;
def : InstAlias<"faclt $dst, $src1, $src2",
                (FACGT32 FPR32:$dst, FPR32:$src2, FPR32:$src1)>;
def : InstAlias<"faclt $dst, $src1, $src2",
                (FACGT64 FPR64:$dst, FPR64:$src2, FPR64:$src1)>;

//===----------------------------------------------------------------------===//
// Advanced SIMD three scalar instructions (mixed operands).
//===----------------------------------------------------------------------===//
defm SQDMULL  : SIMDThreeScalarMixedHS<0, 0b11010, "sqdmull",
                                       int_arm64_neon_sqdmulls_scalar>;
defm SQDMLAL  : SIMDThreeScalarMixedTiedHS<0, 0b10010, "sqdmlal">;
defm SQDMLSL  : SIMDThreeScalarMixedTiedHS<0, 0b10110, "sqdmlsl">;

def : Pat<(i64 (int_arm64_neon_sqadd (i64 FPR64:$Rd),
                   (i64 (int_arm64_neon_sqdmulls_scalar (i32 FPR32:$Rn),
                                                        (i32 FPR32:$Rm))))),
          (SQDMLALi32 FPR64:$Rd, FPR32:$Rn, FPR32:$Rm)>;
def : Pat<(i64 (int_arm64_neon_sqsub (i64 FPR64:$Rd),
                   (i64 (int_arm64_neon_sqdmulls_scalar (i32 FPR32:$Rn),
                                                        (i32 FPR32:$Rm))))),
          (SQDMLSLi32 FPR64:$Rd, FPR32:$Rn, FPR32:$Rm)>;

//===----------------------------------------------------------------------===//
// Advanced SIMD two scalar instructions.
//===----------------------------------------------------------------------===//

defm ABS    : SIMDTwoScalarD<    0, 0b01011, "abs", int_arm64_neon_abs>;
defm CMEQ   : SIMDCmpTwoScalarD< 0, 0b01001, "cmeq", ARM64cmeqz>;
defm CMGE   : SIMDCmpTwoScalarD< 1, 0b01000, "cmge", ARM64cmgez>;
defm CMGT   : SIMDCmpTwoScalarD< 0, 0b01000, "cmgt", ARM64cmgtz>;
defm CMLE   : SIMDCmpTwoScalarD< 1, 0b01001, "cmle", ARM64cmlez>;
defm CMLT   : SIMDCmpTwoScalarD< 0, 0b01010, "cmlt", ARM64cmltz>;
defm FCMEQ  : SIMDCmpTwoScalarSD<0, 1, 0b01101, "fcmeq", ARM64fcmeqz>;
defm FCMGE  : SIMDCmpTwoScalarSD<1, 1, 0b01100, "fcmge", ARM64fcmgez>;
defm FCMGT  : SIMDCmpTwoScalarSD<0, 1, 0b01100, "fcmgt", ARM64fcmgtz>;
defm FCMLE  : SIMDCmpTwoScalarSD<1, 1, 0b01101, "fcmle", ARM64fcmlez>;
defm FCMLT  : SIMDCmpTwoScalarSD<0, 1, 0b01110, "fcmlt", ARM64fcmltz>;
defm FCVTAS : SIMDTwoScalarSD<   0, 0, 0b11100, "fcvtas">;
defm FCVTAU : SIMDTwoScalarSD<   1, 0, 0b11100, "fcvtau">;
defm FCVTMS : SIMDTwoScalarSD<   0, 0, 0b11011, "fcvtms">;
defm FCVTMU : SIMDTwoScalarSD<   1, 0, 0b11011, "fcvtmu">;
defm FCVTNS : SIMDTwoScalarSD<   0, 0, 0b11010, "fcvtns">;
defm FCVTNU : SIMDTwoScalarSD<   1, 0, 0b11010, "fcvtnu">;
defm FCVTPS : SIMDTwoScalarSD<   0, 1, 0b11010, "fcvtps">;
defm FCVTPU : SIMDTwoScalarSD<   1, 1, 0b11010, "fcvtpu">;
def  FCVTXNv1i64 : SIMDInexactCvtTwoScalar<0b10110, "fcvtxn">;
defm FCVTZS : SIMDTwoScalarSD<   0, 1, 0b11011, "fcvtzs">;
defm FCVTZU : SIMDTwoScalarSD<   1, 1, 0b11011, "fcvtzu">;
defm FRECPE : SIMDTwoScalarSD<   0, 1, 0b11101, "frecpe">;
defm FRECPX : SIMDTwoScalarSD<   0, 1, 0b11111, "frecpx">;
defm FRSQRTE : SIMDTwoScalarSD<  1, 1, 0b11101, "frsqrte">;
defm NEG    : SIMDTwoScalarD<    1, 0b01011, "neg",
                                 UnOpFrag<(sub immAllZerosV, node:$LHS)> >;
defm SCVTF  : SIMDTwoScalarCVTSD<   0, 0, 0b11101, "scvtf", ARM64sitof>;
defm SQABS  : SIMDTwoScalarBHSD< 0, 0b00111, "sqabs", int_arm64_neon_sqabs>;
defm SQNEG  : SIMDTwoScalarBHSD< 1, 0b00111, "sqneg", int_arm64_neon_sqneg>;
defm SQXTN  : SIMDTwoScalarMixedBHS< 0, 0b10100, "sqxtn", int_arm64_neon_scalar_sqxtn>;
defm SQXTUN : SIMDTwoScalarMixedBHS< 1, 0b10010, "sqxtun", int_arm64_neon_scalar_sqxtun>;
defm SUQADD : SIMDTwoScalarBHSDTied< 0, 0b00011, "suqadd",
                                     int_arm64_neon_suqadd>;
defm UCVTF  : SIMDTwoScalarCVTSD<   1, 0, 0b11101, "ucvtf", ARM64uitof>;
defm UQXTN  : SIMDTwoScalarMixedBHS<1, 0b10100, "uqxtn", int_arm64_neon_scalar_uqxtn>;
defm USQADD : SIMDTwoScalarBHSDTied< 1, 0b00011, "usqadd",
                                    int_arm64_neon_usqadd>;

def : Pat<(v1i64 (int_arm64_neon_fcvtas (v1f64 FPR64:$Rn))),
          (FCVTASv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtau (v1f64 FPR64:$Rn))),
          (FCVTAUv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtms (v1f64 FPR64:$Rn))),
          (FCVTMSv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtmu (v1f64 FPR64:$Rn))),
          (FCVTMUv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtns (v1f64 FPR64:$Rn))),
          (FCVTNSv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtnu (v1f64 FPR64:$Rn))),
          (FCVTNUv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtps (v1f64 FPR64:$Rn))),
          (FCVTPSv1i64 FPR64:$Rn)>;
def : Pat<(v1i64 (int_arm64_neon_fcvtpu (v1f64 FPR64:$Rn))),
          (FCVTPUv1i64 FPR64:$Rn)>;

def : Pat<(f32 (int_arm64_neon_frecpe (f32 FPR32:$Rn))),
          (FRECPEv1i32 FPR32:$Rn)>;
def : Pat<(f64 (int_arm64_neon_frecpe (f64 FPR64:$Rn))),
          (FRECPEv1i64 FPR64:$Rn)>;
def : Pat<(v1f64 (int_arm64_neon_frecpe (v1f64 FPR64:$Rn))),
          (FRECPEv1i64 FPR64:$Rn)>;

def : Pat<(f32 (int_arm64_neon_frecpx (f32 FPR32:$Rn))),
          (FRECPXv1i32 FPR32:$Rn)>;
def : Pat<(f64 (int_arm64_neon_frecpx (f64 FPR64:$Rn))),
          (FRECPXv1i64 FPR64:$Rn)>;

def : Pat<(f32 (int_arm64_neon_frsqrte (f32 FPR32:$Rn))),
          (FRSQRTEv1i32 FPR32:$Rn)>;
def : Pat<(f64 (int_arm64_neon_frsqrte (f64 FPR64:$Rn))),
          (FRSQRTEv1i64 FPR64:$Rn)>;
def : Pat<(v1f64 (int_arm64_neon_frsqrte (v1f64 FPR64:$Rn))),
          (FRSQRTEv1i64 FPR64:$Rn)>;

// If an integer is about to be converted to a floating point value,
// just load it on the floating point unit.
// Here are the patterns for 8 and 16-bits to float.
// 8-bits -> float.
def : Pat <(f32 (uint_to_fp (i32 (zextloadi8 ro_indexed8:$addr)))),
           (UCVTFv1i32 (INSERT_SUBREG (f32 (IMPLICIT_DEF)),
                          (LDRBro ro_indexed8:$addr), bsub))>;
def : Pat <(f32 (uint_to_fp (i32 (zextloadi8 am_indexed8:$addr)))),
           (UCVTFv1i32 (INSERT_SUBREG (f32 (IMPLICIT_DEF)),
                          (LDRBui am_indexed8:$addr), bsub))>;
def : Pat <(f32 (uint_to_fp (i32 (zextloadi8 am_unscaled8:$addr)))),
           (UCVTFv1i32 (INSERT_SUBREG (f32 (IMPLICIT_DEF)),
                          (LDURBi am_unscaled8:$addr), bsub))>;
// 16-bits -> float.
def : Pat <(f32 (uint_to_fp (i32 (zextloadi16 ro_indexed16:$addr)))),
           (UCVTFv1i32 (INSERT_SUBREG (f32 (IMPLICIT_DEF)),
                          (LDRHro ro_indexed16:$addr), hsub))>;
def : Pat <(f32 (uint_to_fp (i32 (zextloadi16 am_indexed16:$addr)))),
           (UCVTFv1i32 (INSERT_SUBREG (f32 (IMPLICIT_DEF)),
                          (LDRHui am_indexed16:$addr), hsub))>;
def : Pat <(f32 (uint_to_fp (i32 (zextloadi16 am_unscaled16:$addr)))),
           (UCVTFv1i32 (INSERT_SUBREG (f32 (IMPLICIT_DEF)),
                          (LDURHi am_unscaled16:$addr), hsub))>;
// 32-bits are handled in target specific dag combine:
// performIntToFpCombine.
// 64-bits integer to 32-bits floating point, not possible with
// UCVTF on floating point registers (both source and destination
// must have the same size).

// Here are the patterns for 8, 16, 32, and 64-bits to double.
// 8-bits -> double.
def : Pat <(f64 (uint_to_fp (i32 (zextloadi8 ro_indexed8:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDRBro ro_indexed8:$addr), bsub))>;
def : Pat <(f64 (uint_to_fp (i32 (zextloadi8 am_indexed8:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDRBui am_indexed8:$addr), bsub))>;
def : Pat <(f64 (uint_to_fp (i32 (zextloadi8 am_unscaled8:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDURBi am_unscaled8:$addr), bsub))>;
// 16-bits -> double.
def : Pat <(f64 (uint_to_fp (i32 (zextloadi16 ro_indexed16:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDRHro ro_indexed16:$addr), hsub))>;
def : Pat <(f64 (uint_to_fp (i32 (zextloadi16 am_indexed16:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDRHui am_indexed16:$addr), hsub))>;
def : Pat <(f64 (uint_to_fp (i32 (zextloadi16 am_unscaled16:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDURHi am_unscaled16:$addr), hsub))>;
// 32-bits -> double.
def : Pat <(f64 (uint_to_fp (i32 (load ro_indexed32:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDRSro ro_indexed32:$addr), ssub))>;
def : Pat <(f64 (uint_to_fp (i32 (load am_indexed32:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDRSui am_indexed32:$addr), ssub))>;
def : Pat <(f64 (uint_to_fp (i32 (load am_unscaled32:$addr)))),
           (UCVTFv1i64 (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                          (LDURSi am_unscaled32:$addr), ssub))>;
// 64-bits -> double are handled in target specific dag combine:
// performIntToFpCombine.

//===----------------------------------------------------------------------===//
// Advanced SIMD three different-sized vector instructions.
//===----------------------------------------------------------------------===//

defm ADDHN  : SIMDNarrowThreeVectorBHS<0,0b0100,"addhn", int_arm64_neon_addhn>;
defm SUBHN  : SIMDNarrowThreeVectorBHS<0,0b0110,"subhn", int_arm64_neon_subhn>;
defm RADDHN : SIMDNarrowThreeVectorBHS<1,0b0100,"raddhn",int_arm64_neon_raddhn>;
defm RSUBHN : SIMDNarrowThreeVectorBHS<1,0b0110,"rsubhn",int_arm64_neon_rsubhn>;
defm PMULL  : SIMDDifferentThreeVectorBD<0,0b1110,"pmull",int_arm64_neon_pmull>;
defm SABAL  : SIMDLongThreeVectorTiedBHSabal<0,0b0101,"sabal",
                                             int_arm64_neon_sabd>;
defm SABDL   : SIMDLongThreeVectorBHSabdl<0, 0b0111, "sabdl",
                                          int_arm64_neon_sabd>;
defm SADDL   : SIMDLongThreeVectorBHS<   0, 0b0000, "saddl",
            BinOpFrag<(add (sext node:$LHS), (sext node:$RHS))>>;
defm SADDW   : SIMDWideThreeVectorBHS<   0, 0b0001, "saddw",
                 BinOpFrag<(add node:$LHS, (sext node:$RHS))>>;
defm SMLAL   : SIMDLongThreeVectorTiedBHS<0, 0b1000, "smlal",
    TriOpFrag<(add node:$LHS, (int_arm64_neon_smull node:$MHS, node:$RHS))>>;
defm SMLSL   : SIMDLongThreeVectorTiedBHS<0, 0b1010, "smlsl",
    TriOpFrag<(sub node:$LHS, (int_arm64_neon_smull node:$MHS, node:$RHS))>>;
defm SMULL   : SIMDLongThreeVectorBHS<0, 0b1100, "smull", int_arm64_neon_smull>;
defm SQDMLAL : SIMDLongThreeVectorSQDMLXTiedHS<0, 0b1001, "sqdmlal",
                                               int_arm64_neon_sqadd>;
defm SQDMLSL : SIMDLongThreeVectorSQDMLXTiedHS<0, 0b1011, "sqdmlsl",
                                               int_arm64_neon_sqsub>;
defm SQDMULL : SIMDLongThreeVectorHS<0, 0b1101, "sqdmull",
                                     int_arm64_neon_sqdmull>;
defm SSUBL   : SIMDLongThreeVectorBHS<0, 0b0010, "ssubl",
                 BinOpFrag<(sub (sext node:$LHS), (sext node:$RHS))>>;
defm SSUBW   : SIMDWideThreeVectorBHS<0, 0b0011, "ssubw",
                 BinOpFrag<(sub node:$LHS, (sext node:$RHS))>>;
defm UABAL   : SIMDLongThreeVectorTiedBHSabal<1, 0b0101, "uabal",
                                              int_arm64_neon_uabd>;
defm UABDL   : SIMDLongThreeVectorBHSabdl<1, 0b0111, "uabdl",
                                          int_arm64_neon_uabd>;
defm UADDL   : SIMDLongThreeVectorBHS<1, 0b0000, "uaddl",
                 BinOpFrag<(add (zext node:$LHS), (zext node:$RHS))>>;
defm UADDW   : SIMDWideThreeVectorBHS<1, 0b0001, "uaddw",
                 BinOpFrag<(add node:$LHS, (zext node:$RHS))>>;
defm UMLAL   : SIMDLongThreeVectorTiedBHS<1, 0b1000, "umlal",
    TriOpFrag<(add node:$LHS, (int_arm64_neon_umull node:$MHS, node:$RHS))>>;
defm UMLSL   : SIMDLongThreeVectorTiedBHS<1, 0b1010, "umlsl",
    TriOpFrag<(sub node:$LHS, (int_arm64_neon_umull node:$MHS, node:$RHS))>>;
defm UMULL   : SIMDLongThreeVectorBHS<1, 0b1100, "umull", int_arm64_neon_umull>;
defm USUBL   : SIMDLongThreeVectorBHS<1, 0b0010, "usubl",
                 BinOpFrag<(sub (zext node:$LHS), (zext node:$RHS))>>;
defm USUBW   : SIMDWideThreeVectorBHS<   1, 0b0011, "usubw",
                 BinOpFrag<(sub node:$LHS, (zext node:$RHS))>>;

// Patterns for 64-bit pmull
def : Pat<(int_arm64_neon_pmull64 V64:$Rn, V64:$Rm),
          (PMULLv1i64 V64:$Rn, V64:$Rm)>;
def : Pat<(int_arm64_neon_pmull64 (vector_extract (v2i64 V128:$Rn), (i64 1)),
                                  (vector_extract (v2i64 V128:$Rm), (i64 1))),
          (PMULLv2i64 V128:$Rn, V128:$Rm)>;

// CodeGen patterns for addhn and subhn instructions, which can actually be
// written in LLVM IR without too much difficulty.

// ADDHN
def : Pat<(v8i8 (trunc (v8i16 (ARM64vlshr (add V128:$Rn, V128:$Rm), (i32 8))))),
          (ADDHNv8i16_v8i8 V128:$Rn, V128:$Rm)>;
def : Pat<(v4i16 (trunc (v4i32 (ARM64vlshr (add V128:$Rn, V128:$Rm),
                                           (i32 16))))),
          (ADDHNv4i32_v4i16 V128:$Rn, V128:$Rm)>;
def : Pat<(v2i32 (trunc (v2i64 (ARM64vlshr (add V128:$Rn, V128:$Rm),
                                           (i32 32))))),
          (ADDHNv2i64_v2i32 V128:$Rn, V128:$Rm)>;
def : Pat<(concat_vectors (v8i8 V64:$Rd),
                          (trunc (v8i16 (ARM64vlshr (add V128:$Rn, V128:$Rm),
                                                    (i32 8))))),
          (ADDHNv8i16_v16i8 (SUBREG_TO_REG (i32 0), V64:$Rd, dsub),
                            V128:$Rn, V128:$Rm)>;
def : Pat<(concat_vectors (v4i16 V64:$Rd),
                          (trunc (v4i32 (ARM64vlshr (add V128:$Rn, V128:$Rm),
                                                    (i32 16))))),
          (ADDHNv4i32_v8i16 (SUBREG_TO_REG (i32 0), V64:$Rd, dsub),
                            V128:$Rn, V128:$Rm)>;
def : Pat<(concat_vectors (v2i32 V64:$Rd),
                          (trunc (v2i64 (ARM64vlshr (add V128:$Rn, V128:$Rm),
                                                    (i32 32))))),
          (ADDHNv2i64_v4i32 (SUBREG_TO_REG (i32 0), V64:$Rd, dsub),
                            V128:$Rn, V128:$Rm)>;

// SUBHN
def : Pat<(v8i8 (trunc (v8i16 (ARM64vlshr (sub V128:$Rn, V128:$Rm), (i32 8))))),
          (SUBHNv8i16_v8i8 V128:$Rn, V128:$Rm)>;
def : Pat<(v4i16 (trunc (v4i32 (ARM64vlshr (sub V128:$Rn, V128:$Rm),
                                           (i32 16))))),
          (SUBHNv4i32_v4i16 V128:$Rn, V128:$Rm)>;
def : Pat<(v2i32 (trunc (v2i64 (ARM64vlshr (sub V128:$Rn, V128:$Rm),
                                           (i32 32))))),
          (SUBHNv2i64_v2i32 V128:$Rn, V128:$Rm)>;
def : Pat<(concat_vectors (v8i8 V64:$Rd),
                          (trunc (v8i16 (ARM64vlshr (sub V128:$Rn, V128:$Rm),
                                                    (i32 8))))),
          (SUBHNv8i16_v16i8 (SUBREG_TO_REG (i32 0), V64:$Rd, dsub),
                            V128:$Rn, V128:$Rm)>;
def : Pat<(concat_vectors (v4i16 V64:$Rd),
                          (trunc (v4i32 (ARM64vlshr (sub V128:$Rn, V128:$Rm),
                                                    (i32 16))))),
          (SUBHNv4i32_v8i16 (SUBREG_TO_REG (i32 0), V64:$Rd, dsub),
                            V128:$Rn, V128:$Rm)>;
def : Pat<(concat_vectors (v2i32 V64:$Rd),
                          (trunc (v2i64 (ARM64vlshr (sub V128:$Rn, V128:$Rm),
                                                    (i32 32))))),
          (SUBHNv2i64_v4i32 (SUBREG_TO_REG (i32 0), V64:$Rd, dsub),
                            V128:$Rn, V128:$Rm)>;

//----------------------------------------------------------------------------
// AdvSIMD bitwise extract from vector instruction.
//----------------------------------------------------------------------------

defm EXT : SIMDBitwiseExtract<"ext">;

def : Pat<(v4i16 (ARM64ext V64:$Rn, V64:$Rm, (i32 imm:$imm))),
          (EXTv8i8 V64:$Rn, V64:$Rm, imm:$imm)>;
def : Pat<(v8i16 (ARM64ext V128:$Rn, V128:$Rm, (i32 imm:$imm))),
          (EXTv16i8 V128:$Rn, V128:$Rm, imm:$imm)>;
def : Pat<(v2i32 (ARM64ext V64:$Rn, V64:$Rm, (i32 imm:$imm))),
          (EXTv8i8 V64:$Rn, V64:$Rm, imm:$imm)>;
def : Pat<(v2f32 (ARM64ext V64:$Rn, V64:$Rm, (i32 imm:$imm))),
          (EXTv8i8 V64:$Rn, V64:$Rm, imm:$imm)>;
def : Pat<(v4i32 (ARM64ext V128:$Rn, V128:$Rm, (i32 imm:$imm))),
          (EXTv16i8 V128:$Rn, V128:$Rm, imm:$imm)>;
def : Pat<(v4f32 (ARM64ext V128:$Rn, V128:$Rm, (i32 imm:$imm))),
          (EXTv16i8 V128:$Rn, V128:$Rm, imm:$imm)>;
def : Pat<(v2i64 (ARM64ext V128:$Rn, V128:$Rm, (i32 imm:$imm))),
          (EXTv16i8 V128:$Rn, V128:$Rm, imm:$imm)>;
def : Pat<(v2f64 (ARM64ext V128:$Rn, V128:$Rm, (i32 imm:$imm))),
          (EXTv16i8 V128:$Rn, V128:$Rm, imm:$imm)>;

// We use EXT to handle extract_subvector to copy the upper 64-bits of a
// 128-bit vector.
def : Pat<(v8i8  (extract_subvector V128:$Rn, (i64 8))),
          (EXTRACT_SUBREG (EXTv16i8 V128:$Rn, V128:$Rn, 8), dsub)>;
def : Pat<(v4i16 (extract_subvector V128:$Rn, (i64 4))),
          (EXTRACT_SUBREG (EXTv16i8 V128:$Rn, V128:$Rn, 8), dsub)>;
def : Pat<(v2i32 (extract_subvector V128:$Rn, (i64 2))),
          (EXTRACT_SUBREG (EXTv16i8 V128:$Rn, V128:$Rn, 8), dsub)>;
def : Pat<(v1i64 (extract_subvector V128:$Rn, (i64 1))),
          (EXTRACT_SUBREG (EXTv16i8 V128:$Rn, V128:$Rn, 8), dsub)>;
def : Pat<(v2f32 (extract_subvector V128:$Rn, (i64 2))),
          (EXTRACT_SUBREG (EXTv16i8 V128:$Rn, V128:$Rn, 8), dsub)>;
def : Pat<(v1f64 (extract_subvector V128:$Rn, (i64 1))),
          (EXTRACT_SUBREG (EXTv16i8 V128:$Rn, V128:$Rn, 8), dsub)>;


//----------------------------------------------------------------------------
// AdvSIMD zip vector
//----------------------------------------------------------------------------

defm TRN1 : SIMDZipVector<0b010, "trn1", ARM64trn1>;
defm TRN2 : SIMDZipVector<0b110, "trn2", ARM64trn2>;
defm UZP1 : SIMDZipVector<0b001, "uzp1", ARM64uzp1>;
defm UZP2 : SIMDZipVector<0b101, "uzp2", ARM64uzp2>;
defm ZIP1 : SIMDZipVector<0b011, "zip1", ARM64zip1>;
defm ZIP2 : SIMDZipVector<0b111, "zip2", ARM64zip2>;

//----------------------------------------------------------------------------
// AdvSIMD TBL/TBX instructions
//----------------------------------------------------------------------------

defm TBL : SIMDTableLookup<    0, "tbl">;
defm TBX : SIMDTableLookupTied<1, "tbx">;

def : Pat<(v8i8 (int_arm64_neon_tbl1 (v16i8 VecListOne128:$Rn), (v8i8 V64:$Ri))),
          (TBLv8i8One VecListOne128:$Rn, V64:$Ri)>;
def : Pat<(v16i8 (int_arm64_neon_tbl1 (v16i8 V128:$Ri), (v16i8 V128:$Rn))),
          (TBLv16i8One V128:$Ri, V128:$Rn)>;

def : Pat<(v8i8 (int_arm64_neon_tbx1 (v8i8 V64:$Rd),
                  (v16i8 VecListOne128:$Rn), (v8i8 V64:$Ri))),
          (TBXv8i8One V64:$Rd, VecListOne128:$Rn, V64:$Ri)>;
def : Pat<(v16i8 (int_arm64_neon_tbx1 (v16i8 V128:$Rd),
                   (v16i8 V128:$Ri), (v16i8 V128:$Rn))),
          (TBXv16i8One V128:$Rd, V128:$Ri, V128:$Rn)>;


//----------------------------------------------------------------------------
// AdvSIMD scalar CPY instruction
//----------------------------------------------------------------------------

defm CPY : SIMDScalarCPY<"cpy">;

//----------------------------------------------------------------------------
// AdvSIMD scalar pairwise instructions
//----------------------------------------------------------------------------

defm ADDP    : SIMDPairwiseScalarD<0, 0b11011, "addp">;
defm FADDP   : SIMDPairwiseScalarSD<1, 0, 0b01101, "faddp">;
defm FMAXNMP : SIMDPairwiseScalarSD<1, 0, 0b01100, "fmaxnmp">;
defm FMAXP   : SIMDPairwiseScalarSD<1, 0, 0b01111, "fmaxp">;
defm FMINNMP : SIMDPairwiseScalarSD<1, 1, 0b01100, "fminnmp">;
defm FMINP   : SIMDPairwiseScalarSD<1, 1, 0b01111, "fminp">;
def : Pat<(i64 (int_arm64_neon_saddv (v2i64 V128:$Rn))),
          (ADDPv2i64p V128:$Rn)>;
def : Pat<(i64 (int_arm64_neon_uaddv (v2i64 V128:$Rn))),
          (ADDPv2i64p V128:$Rn)>;
def : Pat<(f32 (int_arm64_neon_faddv (v2f32 V64:$Rn))),
          (FADDPv2i32p V64:$Rn)>;
def : Pat<(f32 (int_arm64_neon_faddv (v4f32 V128:$Rn))),
          (FADDPv2i32p (EXTRACT_SUBREG (FADDPv4f32 V128:$Rn, V128:$Rn), dsub))>;
def : Pat<(f64 (int_arm64_neon_faddv (v2f64 V128:$Rn))),
          (FADDPv2i64p V128:$Rn)>;
def : Pat<(f32 (int_arm64_neon_fmaxnmv (v2f32 V64:$Rn))),
          (FMAXNMPv2i32p V64:$Rn)>;
def : Pat<(f64 (int_arm64_neon_fmaxnmv (v2f64 V128:$Rn))),
          (FMAXNMPv2i64p V128:$Rn)>;
def : Pat<(f32 (int_arm64_neon_fmaxv (v2f32 V64:$Rn))),
          (FMAXPv2i32p V64:$Rn)>;
def : Pat<(f64 (int_arm64_neon_fmaxv (v2f64 V128:$Rn))),
          (FMAXPv2i64p V128:$Rn)>;
def : Pat<(f32 (int_arm64_neon_fminnmv (v2f32 V64:$Rn))),
          (FMINNMPv2i32p V64:$Rn)>;
def : Pat<(f64 (int_arm64_neon_fminnmv (v2f64 V128:$Rn))),
          (FMINNMPv2i64p V128:$Rn)>;
def : Pat<(f32 (int_arm64_neon_fminv (v2f32 V64:$Rn))),
          (FMINPv2i32p V64:$Rn)>;
def : Pat<(f64 (int_arm64_neon_fminv (v2f64 V128:$Rn))),
          (FMINPv2i64p V128:$Rn)>;

//----------------------------------------------------------------------------
// AdvSIMD INS/DUP instructions
//----------------------------------------------------------------------------

def DUPv8i8gpr  : SIMDDupFromMain<0, 0b00001, ".8b", v8i8, V64, GPR32>;
def DUPv16i8gpr : SIMDDupFromMain<1, 0b00001, ".16b", v16i8, V128, GPR32>;
def DUPv4i16gpr : SIMDDupFromMain<0, 0b00010, ".4h", v4i16, V64, GPR32>;
def DUPv8i16gpr : SIMDDupFromMain<1, 0b00010, ".8h", v8i16, V128, GPR32>;
def DUPv2i32gpr : SIMDDupFromMain<0, 0b00100, ".2s", v2i32, V64, GPR32>;
def DUPv4i32gpr : SIMDDupFromMain<1, 0b00100, ".4s", v4i32, V128, GPR32>;
def DUPv2i64gpr : SIMDDupFromMain<1, 0b01000, ".2d", v2i64, V128, GPR64>;

def DUPv2i64lane : SIMDDup64FromElement;
def DUPv2i32lane : SIMDDup32FromElement<0, ".2s", v2i32, V64>;
def DUPv4i32lane : SIMDDup32FromElement<1, ".4s", v4i32, V128>;
def DUPv4i16lane : SIMDDup16FromElement<0, ".4h", v4i16, V64>;
def DUPv8i16lane : SIMDDup16FromElement<1, ".8h", v8i16, V128>;
def DUPv8i8lane  : SIMDDup8FromElement <0, ".8b", v8i8, V64>;
def DUPv16i8lane : SIMDDup8FromElement <1, ".16b", v16i8, V128>;

def : Pat<(v2f32 (ARM64dup (f32 FPR32:$Rn))),
          (v2f32 (DUPv2i32lane
            (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR32:$Rn, ssub),
            (i64 0)))>;
def : Pat<(v4f32 (ARM64dup (f32 FPR32:$Rn))),
          (v4f32 (DUPv4i32lane
            (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR32:$Rn, ssub),
            (i64 0)))>;
def : Pat<(v2f64 (ARM64dup (f64 FPR64:$Rn))),
          (v2f64 (DUPv2i64lane
            (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR64:$Rn, dsub),
            (i64 0)))>;

def : Pat<(v2f32 (ARM64duplane32 (v4f32 V128:$Rn), VectorIndexS:$imm)),
          (DUPv2i32lane V128:$Rn, VectorIndexS:$imm)>;
def : Pat<(v4f32 (ARM64duplane32 (v4f32 V128:$Rn), VectorIndexS:$imm)),
         (DUPv4i32lane V128:$Rn, VectorIndexS:$imm)>;
def : Pat<(v2f64 (ARM64duplane64 (v2f64 V128:$Rn), VectorIndexD:$imm)),
          (DUPv2i64lane V128:$Rn, VectorIndexD:$imm)>;

defm SMOV : SMov;
defm UMOV : UMov;

def : Pat<(sext_inreg (vector_extract (v16i8 V128:$Rn), VectorIndexB:$idx), i8),
          (i32 (SMOVvi8to32 V128:$Rn, VectorIndexB:$idx))>;
def : Pat<(sext_inreg (vector_extract (v16i8 V128:$Rn), VectorIndexB:$idx), i8),
          (i64 (SMOVvi8to64 V128:$Rn, VectorIndexB:$idx))>;
def : Pat<(sext_inreg (vector_extract (v8i16 V128:$Rn), VectorIndexH:$idx),i16),
          (i32 (SMOVvi16to32 V128:$Rn, VectorIndexH:$idx))>;
def : Pat<(sext_inreg (vector_extract (v8i16 V128:$Rn), VectorIndexH:$idx),i16),
          (i64 (SMOVvi16to64 V128:$Rn, VectorIndexH:$idx))>;
def : Pat<(sext_inreg (vector_extract (v8i16 V128:$Rn), VectorIndexH:$idx),i16),
          (i32 (SMOVvi16to32 V128:$Rn, VectorIndexH:$idx))>;
def : Pat<(sext (i32 (vector_extract (v4i32 V128:$Rn), VectorIndexS:$idx))),
          (i64 (SMOVvi32to64 V128:$Rn, VectorIndexS:$idx))>;

// Extracting i8 or i16 elements will have the zero-extend transformed to
// an 'and' mask by type legalization since neither i8 nor i16 are legal types
// for ARM64. Match these patterns here since UMOV already zeroes out the high
// bits of the destination register.
def : Pat<(and (vector_extract (v16i8 V128:$Rn), VectorIndexB:$idx),
               (i32 0xff)),
          (i32 (UMOVvi8 V128:$Rn, VectorIndexB:$idx))>;
def : Pat<(and (vector_extract (v8i16 V128:$Rn), VectorIndexH:$idx),
               (i32 0xffff)),
          (i32 (UMOVvi16 V128:$Rn, VectorIndexH:$idx))>;

defm INS : SIMDIns;

def : Pat<(v16i8 (scalar_to_vector GPR32:$Rn)),
          (SUBREG_TO_REG (i32 0),
                         (f32 (COPY_TO_REGCLASS GPR32:$Rn, FPR32)), ssub)>;
def : Pat<(v8i8 (scalar_to_vector GPR32:$Rn)),
          (SUBREG_TO_REG (i32 0),
                         (f32 (COPY_TO_REGCLASS GPR32:$Rn, FPR32)), ssub)>;

def : Pat<(v8i16 (scalar_to_vector GPR32:$Rn)),
          (SUBREG_TO_REG (i32 0),
                         (f32 (COPY_TO_REGCLASS GPR32:$Rn, FPR32)), ssub)>;
def : Pat<(v4i16 (scalar_to_vector GPR32:$Rn)),
          (SUBREG_TO_REG (i32 0),
                         (f32 (COPY_TO_REGCLASS GPR32:$Rn, FPR32)), ssub)>;

def : Pat<(v2i32 (scalar_to_vector (i32 FPR32:$Rn))),
            (v2i32 (INSERT_SUBREG (v2i32 (IMPLICIT_DEF)),
                                  (i32 FPR32:$Rn), ssub))>;
def : Pat<(v4i32 (scalar_to_vector (i32 FPR32:$Rn))),
            (v4i32 (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)),
                                  (i32 FPR32:$Rn), ssub))>;
def : Pat<(v2i64 (scalar_to_vector (i64 FPR64:$Rn))),
            (v2i64 (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)),
                                  (i64 FPR64:$Rn), dsub))>;

def : Pat<(v4f32 (scalar_to_vector (f32 FPR32:$Rn))),
          (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), FPR32:$Rn, ssub)>;
def : Pat<(v2f32 (scalar_to_vector (f32 FPR32:$Rn))),
          (INSERT_SUBREG (v2f32 (IMPLICIT_DEF)), FPR32:$Rn, ssub)>;
def : Pat<(v2f64 (scalar_to_vector (f64 FPR64:$Rn))),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), FPR64:$Rn, dsub)>;

def : Pat<(v2f32 (vector_insert (v2f32 V64:$Rn),
            (f32 FPR32:$Rm), (i64 VectorIndexS:$imm))),
          (EXTRACT_SUBREG
            (INSvi32lane
              (v4f32 (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), V64:$Rn, dsub)),
              VectorIndexS:$imm,
              (v4f32 (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), FPR32:$Rm, ssub)),
              (i64 0)),
            dsub)>;
def : Pat<(v4f32 (vector_insert (v4f32 V128:$Rn),
            (f32 FPR32:$Rm), (i64 VectorIndexS:$imm))),
          (INSvi32lane
            V128:$Rn, VectorIndexS:$imm,
            (v4f32 (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), FPR32:$Rm, ssub)),
            (i64 0))>;
def : Pat<(v2f64 (vector_insert (v2f64 V128:$Rn),
            (f64 FPR64:$Rm), (i64 VectorIndexD:$imm))),
          (INSvi64lane
            V128:$Rn, VectorIndexD:$imm,
            (v2f64 (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), FPR64:$Rm, dsub)),
            (i64 0))>;

// Copy an element at a constant index in one vector into a constant indexed
// element of another.
// FIXME refactor to a shared class/dev parameterized on vector type, vector
// index type and INS extension
def : Pat<(v16i8 (int_arm64_neon_vcopy_lane
                   (v16i8 V128:$Vd), VectorIndexB:$idx, (v16i8 V128:$Vs),
                   VectorIndexB:$idx2)),
          (v16i8 (INSvi8lane
                   V128:$Vd, VectorIndexB:$idx, V128:$Vs, VectorIndexB:$idx2)
          )>;
def : Pat<(v8i16 (int_arm64_neon_vcopy_lane
                   (v8i16 V128:$Vd), VectorIndexH:$idx, (v8i16 V128:$Vs),
                   VectorIndexH:$idx2)),
          (v8i16 (INSvi16lane
                   V128:$Vd, VectorIndexH:$idx, V128:$Vs, VectorIndexH:$idx2)
          )>;
def : Pat<(v4i32 (int_arm64_neon_vcopy_lane
                   (v4i32 V128:$Vd), VectorIndexS:$idx, (v4i32 V128:$Vs),
                   VectorIndexS:$idx2)),
          (v4i32 (INSvi32lane
                   V128:$Vd, VectorIndexS:$idx, V128:$Vs, VectorIndexS:$idx2)
          )>;
def : Pat<(v2i64 (int_arm64_neon_vcopy_lane
                   (v2i64 V128:$Vd), VectorIndexD:$idx, (v2i64 V128:$Vs),
                   VectorIndexD:$idx2)),
          (v2i64 (INSvi64lane
                   V128:$Vd, VectorIndexD:$idx, V128:$Vs, VectorIndexD:$idx2)
          )>;

// Floating point vector extractions are codegen'd as either a sequence of
// subregister extractions, possibly fed by an INS if the lane number is
// anything other than zero.
def : Pat<(vector_extract (v2f64 V128:$Rn), 0),
          (f64 (EXTRACT_SUBREG V128:$Rn, dsub))>;
def : Pat<(vector_extract (v4f32 V128:$Rn), 0),
          (f32 (EXTRACT_SUBREG V128:$Rn, ssub))>;
def : Pat<(vector_extract (v2f64 V128:$Rn), VectorIndexD:$idx),
          (f64 (EXTRACT_SUBREG
            (INSvi64lane (v2f64 (IMPLICIT_DEF)), 0,
                         V128:$Rn, VectorIndexD:$idx),
            dsub))>;
def : Pat<(vector_extract (v4f32 V128:$Rn), VectorIndexS:$idx),
          (f32 (EXTRACT_SUBREG
            (INSvi32lane (v4f32 (IMPLICIT_DEF)), 0,
                         V128:$Rn, VectorIndexS:$idx),
            ssub))>;

// All concat_vectors operations are canonicalised to act on i64 vectors for
// ARM64. In the general case we need an instruction, which had just as well be
// INS.
class ConcatPat<ValueType DstTy, ValueType SrcTy>
  : Pat<(DstTy (concat_vectors (SrcTy V64:$Rd), V64:$Rn)),
        (INSvi64lane (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rd, dsub), 1,
                     (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rn, dsub), 0)>;

def : ConcatPat<v2i64, v1i64>;
def : ConcatPat<v2f64, v1f64>;
def : ConcatPat<v4i32, v2i32>;
def : ConcatPat<v4f32, v2f32>;
def : ConcatPat<v8i16, v4i16>;
def : ConcatPat<v16i8, v8i8>;

// If the high lanes are undef, though, we can just ignore them:
class ConcatUndefPat<ValueType DstTy, ValueType SrcTy>
  : Pat<(DstTy (concat_vectors (SrcTy V64:$Rn), undef)),
        (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rn, dsub)>;

def : ConcatUndefPat<v2i64, v1i64>;
def : ConcatUndefPat<v2f64, v1f64>;
def : ConcatUndefPat<v4i32, v2i32>;
def : ConcatUndefPat<v4f32, v2f32>;
def : ConcatUndefPat<v8i16, v4i16>;
def : ConcatUndefPat<v16i8, v8i8>;

//----------------------------------------------------------------------------
// AdvSIMD across lanes instructions
//----------------------------------------------------------------------------

defm ADDV    : SIMDAcrossLanesBHS<0, 0b11011, "addv">;
defm SMAXV   : SIMDAcrossLanesBHS<0, 0b01010, "smaxv">;
defm SMINV   : SIMDAcrossLanesBHS<0, 0b11010, "sminv">;
defm UMAXV   : SIMDAcrossLanesBHS<1, 0b01010, "umaxv">;
defm UMINV   : SIMDAcrossLanesBHS<1, 0b11010, "uminv">;
defm SADDLV  : SIMDAcrossLanesHSD<0, 0b00011, "saddlv">;
defm UADDLV  : SIMDAcrossLanesHSD<1, 0b00011, "uaddlv">;
defm FMAXNMV : SIMDAcrossLanesS<0b01100, 0, "fmaxnmv", int_arm64_neon_fmaxnmv>;
defm FMAXV   : SIMDAcrossLanesS<0b01111, 0, "fmaxv", int_arm64_neon_fmaxv>;
defm FMINNMV : SIMDAcrossLanesS<0b01100, 1, "fminnmv", int_arm64_neon_fminnmv>;
defm FMINV   : SIMDAcrossLanesS<0b01111, 1, "fminv", int_arm64_neon_fminv>;

multiclass SIMDAcrossLanesSignedIntrinsic<string baseOpc, Intrinsic intOp> {
// If there is a sign extension after this intrinsic, consume it as smov already
// performed it
  def : Pat<(i32 (sext_inreg (i32 (intOp (v8i8 V64:$Rn))), i8)),
        (i32 (SMOVvi8to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i8v")) V64:$Rn), bsub),
          (i64 0)))>;
  def : Pat<(i32 (intOp (v8i8 V64:$Rn))),
        (i32 (SMOVvi8to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i8v")) V64:$Rn), bsub),
          (i64 0)))>;
// If there is a sign extension after this intrinsic, consume it as smov already
// performed it
def : Pat<(i32 (sext_inreg (i32 (intOp (v16i8 V128:$Rn))), i8)),
        (i32 (SMOVvi8to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v16i8v")) V128:$Rn), bsub),
          (i64 0)))>;
def : Pat<(i32 (intOp (v16i8 V128:$Rn))),
        (i32 (SMOVvi8to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v16i8v")) V128:$Rn), bsub),
          (i64 0)))>;
// If there is a sign extension after this intrinsic, consume it as smov already
// performed it
def : Pat<(i32 (sext_inreg (i32 (intOp (v4i16 V64:$Rn))), i16)),
          (i32 (SMOVvi16to32
           (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v4i16v")) V64:$Rn), hsub),
           (i64 0)))>;
def : Pat<(i32 (intOp (v4i16 V64:$Rn))),
          (i32 (SMOVvi16to32
           (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v4i16v")) V64:$Rn), hsub),
           (i64 0)))>;
// If there is a sign extension after this intrinsic, consume it as smov already
// performed it
def : Pat<(i32 (sext_inreg (i32 (intOp (v8i16 V128:$Rn))), i16)),
        (i32 (SMOVvi16to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v8i16v")) V128:$Rn), hsub),
          (i64 0)))>;
def : Pat<(i32 (intOp (v8i16 V128:$Rn))),
        (i32 (SMOVvi16to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v8i16v")) V128:$Rn), hsub),
          (i64 0)))>;

def : Pat<(i32 (intOp (v4i32 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v4i32v")) V128:$Rn), ssub),
          ssub))>;
}

multiclass SIMDAcrossLanesUnsignedIntrinsic<string baseOpc, Intrinsic intOp> {
// If there is a masking operation keeping only what has been actually
// generated, consume it.
  def : Pat<(i32 (and (i32 (intOp (v8i8 V64:$Rn))), maski8_or_more)),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i8v")) V64:$Rn), bsub),
          ssub))>;
  def : Pat<(i32 (intOp (v8i8 V64:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i8v")) V64:$Rn), bsub),
          ssub))>;
// If there is a masking operation keeping only what has been actually
// generated, consume it.
def : Pat<(i32 (and (i32 (intOp (v16i8 V128:$Rn))), maski8_or_more)),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v16i8v")) V128:$Rn), bsub),
          ssub))>;
def : Pat<(i32 (intOp (v16i8 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v16i8v")) V128:$Rn), bsub),
          ssub))>;

// If there is a masking operation keeping only what has been actually
// generated, consume it.
def : Pat<(i32 (and (i32 (intOp (v4i16 V64:$Rn))), maski16_or_more)),
          (i32 (EXTRACT_SUBREG
            (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
              (!cast<Instruction>(!strconcat(baseOpc, "v4i16v")) V64:$Rn), hsub),
            ssub))>;
def : Pat<(i32 (intOp (v4i16 V64:$Rn))),
          (i32 (EXTRACT_SUBREG
            (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
              (!cast<Instruction>(!strconcat(baseOpc, "v4i16v")) V64:$Rn), hsub),
            ssub))>;
// If there is a masking operation keeping only what has been actually
// generated, consume it.
def : Pat<(i32 (and (i32 (intOp (v8i16 V128:$Rn))), maski16_or_more)),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i16v")) V128:$Rn), hsub),
          ssub))>;
def : Pat<(i32 (intOp (v8i16 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i16v")) V128:$Rn), hsub),
          ssub))>;

def : Pat<(i32 (intOp (v4i32 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v4i32v")) V128:$Rn), ssub),
          ssub))>;

}

multiclass SIMDAcrossLanesSignedLongIntrinsic<string baseOpc, Intrinsic intOp> {
  def : Pat<(i32 (intOp (v8i8 V64:$Rn))),
        (i32 (SMOVvi16to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i8v")) V64:$Rn), hsub),
          (i64 0)))>;
def : Pat<(i32 (intOp (v16i8 V128:$Rn))),
        (i32 (SMOVvi16to32
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v16i8v")) V128:$Rn), hsub),
          (i64 0)))>;

def : Pat<(i32 (intOp (v4i16 V64:$Rn))),
          (i32 (EXTRACT_SUBREG
           (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v4i16v")) V64:$Rn), ssub),
           ssub))>;
def : Pat<(i32 (intOp (v8i16 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v8i16v")) V128:$Rn), ssub),
          ssub))>;

def : Pat<(i64 (intOp (v4i32 V128:$Rn))),
        (i64 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
           (!cast<Instruction>(!strconcat(baseOpc, "v4i32v")) V128:$Rn), dsub),
          dsub))>;
}

multiclass SIMDAcrossLanesUnsignedLongIntrinsic<string baseOpc,
                                                Intrinsic intOp> {
  def : Pat<(i32 (intOp (v8i8 V64:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i8v")) V64:$Rn), hsub),
          ssub))>;
def : Pat<(i32 (intOp (v16i8 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v16i8v")) V128:$Rn), hsub),
          ssub))>;

def : Pat<(i32 (intOp (v4i16 V64:$Rn))),
          (i32 (EXTRACT_SUBREG
            (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
              (!cast<Instruction>(!strconcat(baseOpc, "v4i16v")) V64:$Rn), ssub),
            ssub))>;
def : Pat<(i32 (intOp (v8i16 V128:$Rn))),
        (i32 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v8i16v")) V128:$Rn), ssub),
          ssub))>;

def : Pat<(i64 (intOp (v4i32 V128:$Rn))),
        (i64 (EXTRACT_SUBREG
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
            (!cast<Instruction>(!strconcat(baseOpc, "v4i32v")) V128:$Rn), dsub),
          dsub))>;
}

defm : SIMDAcrossLanesSignedIntrinsic<"ADDV",  int_arm64_neon_saddv>;
// vaddv_[su]32 is special; -> ADDP Vd.2S,Vn.2S,Vm.2S; return Vd.s[0];Vn==Vm
def : Pat<(i32 (int_arm64_neon_saddv (v2i32 V64:$Rn))),
          (EXTRACT_SUBREG (ADDPv2i32 V64:$Rn, V64:$Rn), ssub)>;

defm : SIMDAcrossLanesUnsignedIntrinsic<"ADDV",  int_arm64_neon_uaddv>;
// vaddv_[su]32 is special; -> ADDP Vd.2S,Vn.2S,Vm.2S; return Vd.s[0];Vn==Vm
def : Pat<(i32 (int_arm64_neon_uaddv (v2i32 V64:$Rn))),
          (EXTRACT_SUBREG (ADDPv2i32 V64:$Rn, V64:$Rn), ssub)>;

defm : SIMDAcrossLanesSignedIntrinsic<"SMAXV", int_arm64_neon_smaxv>;
def : Pat<(i32 (int_arm64_neon_smaxv (v2i32 V64:$Rn))),
           (EXTRACT_SUBREG (SMAXPv2i32 V64:$Rn, V64:$Rn), ssub)>;

defm : SIMDAcrossLanesSignedIntrinsic<"SMINV", int_arm64_neon_sminv>;
def : Pat<(i32 (int_arm64_neon_sminv (v2i32 V64:$Rn))),
           (EXTRACT_SUBREG (SMINPv2i32 V64:$Rn, V64:$Rn), ssub)>;

defm : SIMDAcrossLanesUnsignedIntrinsic<"UMAXV", int_arm64_neon_umaxv>;
def : Pat<(i32 (int_arm64_neon_umaxv (v2i32 V64:$Rn))),
           (EXTRACT_SUBREG (UMAXPv2i32 V64:$Rn, V64:$Rn), ssub)>;

defm : SIMDAcrossLanesUnsignedIntrinsic<"UMINV", int_arm64_neon_uminv>;
def : Pat<(i32 (int_arm64_neon_uminv (v2i32 V64:$Rn))),
           (EXTRACT_SUBREG (UMINPv2i32 V64:$Rn, V64:$Rn), ssub)>;

defm : SIMDAcrossLanesSignedLongIntrinsic<"SADDLV", int_arm64_neon_saddlv>;
defm : SIMDAcrossLanesUnsignedLongIntrinsic<"UADDLV", int_arm64_neon_uaddlv>;

// The vaddlv_s32 intrinsic gets mapped to SADDLP.
def : Pat<(i64 (int_arm64_neon_saddlv (v2i32 V64:$Rn))),
          (i64 (EXTRACT_SUBREG
            (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
              (SADDLPv2i32_v1i64 V64:$Rn), dsub),
            dsub))>;
// The vaddlv_u32 intrinsic gets mapped to UADDLP.
def : Pat<(i64 (int_arm64_neon_uaddlv (v2i32 V64:$Rn))),
          (i64 (EXTRACT_SUBREG
            (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)),
              (UADDLPv2i32_v1i64 V64:$Rn), dsub),
            dsub))>;

//------------------------------------------------------------------------------
// AdvSIMD modified immediate instructions
//------------------------------------------------------------------------------

// AdvSIMD BIC
defm BIC : SIMDModifiedImmVectorShiftTied<1, 0b11, 0b01, "bic", ARM64bici>;
// AdvSIMD ORR
defm ORR : SIMDModifiedImmVectorShiftTied<0, 0b11, 0b01, "orr", ARM64orri>;


// AdvSIMD FMOV
def FMOVv2f64_ns : SIMDModifiedImmVectorNoShift<1, 1, 0b1111, V128, fpimm8,
                                              "fmov", ".2d",
                       [(set (v2f64 V128:$Rd), (ARM64fmov imm0_255:$imm8))]>;
def FMOVv2f32_ns : SIMDModifiedImmVectorNoShift<0, 0, 0b1111, V64,  fpimm8,
                                              "fmov", ".2s",
                       [(set (v2f32 V64:$Rd), (ARM64fmov imm0_255:$imm8))]>;
def FMOVv4f32_ns : SIMDModifiedImmVectorNoShift<1, 0, 0b1111, V128, fpimm8,
                                              "fmov", ".4s",
                       [(set (v4f32 V128:$Rd), (ARM64fmov imm0_255:$imm8))]>;

// AdvSIMD MOVI

// EDIT byte mask: scalar
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOVID      : SIMDModifiedImmScalarNoShift<0, 1, 0b1110, "movi",
                    [(set FPR64:$Rd, simdimmtype10:$imm8)]>;
// The movi_edit node has the immediate value already encoded, so we use
// a plain imm0_255 here.
def : Pat<(f64 (ARM64movi_edit imm0_255:$shift)),
          (MOVID imm0_255:$shift)>;

def : Pat<(v1i64 immAllZerosV), (MOVID (i32 0))>;
def : Pat<(v2i32 immAllZerosV), (MOVID (i32 0))>;
def : Pat<(v4i16 immAllZerosV), (MOVID (i32 0))>;
def : Pat<(v8i8  immAllZerosV), (MOVID (i32 0))>;

def : Pat<(v1i64 immAllOnesV), (MOVID (i32 255))>;
def : Pat<(v2i32 immAllOnesV), (MOVID (i32 255))>;
def : Pat<(v4i16 immAllOnesV), (MOVID (i32 255))>;
def : Pat<(v8i8  immAllOnesV), (MOVID (i32 255))>;

// EDIT byte mask: 2d

// The movi_edit node has the immediate value already encoded, so we use
// a plain imm0_255 in the pattern
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOVIv2d_ns   : SIMDModifiedImmVectorNoShift<1, 1, 0b1110, V128,
                                                simdimmtype10,
                                                "movi", ".2d",
                   [(set (v2i64 V128:$Rd), (ARM64movi_edit imm0_255:$imm8))]>;


// Use movi.2d to materialize 0.0 if the HW does zero-cycle zeroing.
// Complexity is added to break a tie with a plain MOVI.
let AddedComplexity = 1 in {
def : Pat<(f32   fpimm0),
          (f32 (EXTRACT_SUBREG (v2i64 (MOVIv2d_ns (i32 0))), ssub))>,
      Requires<[HasZCZ]>;
def : Pat<(f64   fpimm0),
          (f64 (EXTRACT_SUBREG (v2i64 (MOVIv2d_ns (i32 0))), dsub))>,
      Requires<[HasZCZ]>;
}

def : Pat<(v2i64 immAllZerosV), (MOVIv2d_ns (i32 0))>;
def : Pat<(v4i32 immAllZerosV), (MOVIv2d_ns (i32 0))>;
def : Pat<(v8i16 immAllZerosV), (MOVIv2d_ns (i32 0))>;
def : Pat<(v16i8 immAllZerosV), (MOVIv2d_ns (i32 0))>;

def : Pat<(v2i64 immAllOnesV), (MOVIv2d_ns (i32 255))>;
def : Pat<(v4i32 immAllOnesV), (MOVIv2d_ns (i32 255))>;
def : Pat<(v8i16 immAllOnesV), (MOVIv2d_ns (i32 255))>;
def : Pat<(v16i8 immAllOnesV), (MOVIv2d_ns (i32 255))>;

// EDIT per word & halfword: 2s, 4h, 4s, & 8h
defm MOVI      : SIMDModifiedImmVectorShift<0, 0b10, 0b00, "movi">;
def : Pat<(v2i32 (ARM64movi_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MOVIv2i32 imm0_255:$imm8, imm:$shift)>;
def : Pat<(v4i32 (ARM64movi_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MOVIv4i32 imm0_255:$imm8, imm:$shift)>;
def : Pat<(v4i16 (ARM64movi_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MOVIv4i16 imm0_255:$imm8, imm:$shift)>;
def : Pat<(v8i16 (ARM64movi_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MOVIv8i16 imm0_255:$imm8, imm:$shift)>;

// EDIT per word: 2s & 4s with MSL shifter
def MOVIv2s_msl  : SIMDModifiedImmMoveMSL<0, 0, {1,1,0,?}, V64, "movi", ".2s",
                      [(set (v2i32 V64:$Rd),
                            (ARM64movi_msl imm0_255:$imm8, (i32 imm:$shift)))]>;
def MOVIv4s_msl  : SIMDModifiedImmMoveMSL<1, 0, {1,1,0,?}, V128, "movi", ".4s",
                      [(set (v4i32 V128:$Rd),
                            (ARM64movi_msl imm0_255:$imm8, (i32 imm:$shift)))]>;

// Per byte: 8b & 16b
def MOVIv8b_ns   : SIMDModifiedImmVectorNoShift<0, 0, 0b1110, V64,  imm0_255,
                                                 "movi", ".8b",
                       [(set (v8i8 V64:$Rd), (ARM64movi imm0_255:$imm8))]>;
def MOVIv16b_ns  : SIMDModifiedImmVectorNoShift<1, 0, 0b1110, V128, imm0_255,
                                                 "movi", ".16b",
                       [(set (v16i8 V128:$Rd), (ARM64movi imm0_255:$imm8))]>;

// AdvSIMD MVNI

// EDIT per word & halfword: 2s, 4h, 4s, & 8h
defm MVNI      : SIMDModifiedImmVectorShift<1, 0b10, 0b00, "mvni">;
def : Pat<(v2i32 (ARM64mvni_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MVNIv2i32 imm0_255:$imm8, imm:$shift)>;
def : Pat<(v4i32 (ARM64mvni_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MVNIv4i32 imm0_255:$imm8, imm:$shift)>;
def : Pat<(v4i16 (ARM64mvni_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MVNIv4i16 imm0_255:$imm8, imm:$shift)>;
def : Pat<(v8i16 (ARM64mvni_shift imm0_255:$imm8, (i32 imm:$shift))),
          (MVNIv8i16 imm0_255:$imm8, imm:$shift)>;

// EDIT per word: 2s & 4s with MSL shifter
def MVNIv2s_msl   : SIMDModifiedImmMoveMSL<0, 1, {1,1,0,?}, V64, "mvni", ".2s",
                      [(set (v2i32 V64:$Rd),
                            (ARM64mvni_msl imm0_255:$imm8, (i32 imm:$shift)))]>;
def MVNIv4s_msl   : SIMDModifiedImmMoveMSL<1, 1, {1,1,0,?}, V128, "mvni", ".4s",
                      [(set (v4i32 V128:$Rd),
                            (ARM64mvni_msl imm0_255:$imm8, (i32 imm:$shift)))]>;

//----------------------------------------------------------------------------
// AdvSIMD indexed element
//----------------------------------------------------------------------------

let neverHasSideEffects = 1 in {
  defm FMLA  : SIMDFPIndexedSDTied<0, 0b0001, "fmla">;
  defm FMLS  : SIMDFPIndexedSDTied<0, 0b0101, "fmls">;
}

// NOTE: Operands are reordered in the FMLA/FMLS PatFrags because the
// instruction expects the addend first, while the intrinsic expects it last.

// On the other hand, there are quite a few valid combinatorial options due to
// the commutativity of multiplication and the fact that (-x) * y = x * (-y).
defm : SIMDFPIndexedSDTiedPatterns<"FMLA",
           TriOpFrag<(fma node:$RHS, node:$MHS, node:$LHS)>>;
defm : SIMDFPIndexedSDTiedPatterns<"FMLA",
           TriOpFrag<(fma node:$MHS, node:$RHS, node:$LHS)>>;

defm : SIMDFPIndexedSDTiedPatterns<"FMLS",
           TriOpFrag<(fma node:$MHS, (fneg node:$RHS), node:$LHS)> >;
defm : SIMDFPIndexedSDTiedPatterns<"FMLS",
           TriOpFrag<(fma node:$RHS, (fneg node:$MHS), node:$LHS)> >;
defm : SIMDFPIndexedSDTiedPatterns<"FMLS",
           TriOpFrag<(fma (fneg node:$RHS), node:$MHS, node:$LHS)> >;
defm : SIMDFPIndexedSDTiedPatterns<"FMLS",
           TriOpFrag<(fma (fneg node:$MHS), node:$RHS, node:$LHS)> >;

multiclass FMLSIndexedAfterNegPatterns<SDPatternOperator OpNode> {
  // 3 variants for the .2s version: DUPLANE from 128-bit, DUPLANE from 64-bit
  // and DUP scalar.
  def : Pat<(v2f32 (OpNode (v2f32 V64:$Rd), (v2f32 V64:$Rn),
                           (ARM64duplane32 (v4f32 (fneg V128:$Rm)),
                                           VectorIndexS:$idx))),
            (FMLSv2i32_indexed V64:$Rd, V64:$Rn, V128:$Rm, VectorIndexS:$idx)>;
  def : Pat<(v2f32 (OpNode (v2f32 V64:$Rd), (v2f32 V64:$Rn),
                           (v2f32 (ARM64duplane32
                                      (v4f32 (insert_subvector undef,
                                                 (v2f32 (fneg V64:$Rm)),
                                                 (i32 0))),
                                      VectorIndexS:$idx)))),
            (FMLSv2i32_indexed V64:$Rd, V64:$Rn,
                               (SUBREG_TO_REG (i32 0), V64:$Rm, dsub),
                               VectorIndexS:$idx)>;
  def : Pat<(v2f32 (OpNode (v2f32 V64:$Rd), (v2f32 V64:$Rn),
                           (ARM64dup (f32 (fneg FPR32Op:$Rm))))),
            (FMLSv2i32_indexed V64:$Rd, V64:$Rn,
                (SUBREG_TO_REG (i32 0), FPR32Op:$Rm, ssub), (i64 0))>;

  // 3 variants for the .4s version: DUPLANE from 128-bit, DUPLANE from 64-bit
  // and DUP scalar.
  def : Pat<(v4f32 (OpNode (v4f32 V128:$Rd), (v4f32 V128:$Rn),
                           (ARM64duplane32 (v4f32 (fneg V128:$Rm)),
                                           VectorIndexS:$idx))),
            (FMLSv4i32_indexed V128:$Rd, V128:$Rn, V128:$Rm,
                               VectorIndexS:$idx)>;
  def : Pat<(v4f32 (OpNode (v4f32 V128:$Rd), (v4f32 V128:$Rn),
                           (v4f32 (ARM64duplane32
                                      (v4f32 (insert_subvector undef,
                                                 (v2f32 (fneg V64:$Rm)),
                                                 (i32 0))),
                                      VectorIndexS:$idx)))),
            (FMLSv4i32_indexed V128:$Rd, V128:$Rn,
                               (SUBREG_TO_REG (i32 0), V64:$Rm, dsub),
                               VectorIndexS:$idx)>;
  def : Pat<(v4f32 (OpNode (v4f32 V128:$Rd), (v4f32 V128:$Rn),
                           (ARM64dup (f32 (fneg FPR32Op:$Rm))))),
            (FMLSv4i32_indexed V128:$Rd, V128:$Rn,
                (SUBREG_TO_REG (i32 0), FPR32Op:$Rm, ssub), (i64 0))>;

  // 2 variants for the .2d version: DUPLANE from 128-bit, and DUP scalar
  // (DUPLANE from 64-bit would be trivial).
  def : Pat<(v2f64 (OpNode (v2f64 V128:$Rd), (v2f64 V128:$Rn),
                           (ARM64duplane64 (v2f64 (fneg V128:$Rm)),
                                           VectorIndexD:$idx))),
            (FMLSv2i64_indexed
                V128:$Rd, V128:$Rn, V128:$Rm, VectorIndexS:$idx)>;
  def : Pat<(v2f64 (OpNode (v2f64 V128:$Rd), (v2f64 V128:$Rn),
                           (ARM64dup (f64 (fneg FPR64Op:$Rm))))),
            (FMLSv2i64_indexed V128:$Rd, V128:$Rn,
                (SUBREG_TO_REG (i32 0), FPR64Op:$Rm, dsub), (i64 0))>;

  // 2 variants for 32-bit scalar version: extract from .2s or from .4s
  def : Pat<(f32 (OpNode (f32 FPR32:$Rd), (f32 FPR32:$Rn),
                         (vector_extract (v4f32 (fneg V128:$Rm)),
                                         VectorIndexS:$idx))),
            (FMLSv1i32_indexed FPR32:$Rd, FPR32:$Rn,
                V128:$Rm, VectorIndexS:$idx)>;
  def : Pat<(f32 (OpNode (f32 FPR32:$Rd), (f32 FPR32:$Rn),
                         (vector_extract (v2f32 (fneg V64:$Rm)),
                                         VectorIndexS:$idx))),
            (FMLSv1i32_indexed FPR32:$Rd, FPR32:$Rn,
                (SUBREG_TO_REG (i32 0), V64:$Rm, dsub), VectorIndexS:$idx)>;

  // 1 variant for 64-bit scalar version: extract from .1d or from .2d
  def : Pat<(f64 (OpNode (f64 FPR64:$Rd), (f64 FPR64:$Rn),
                         (vector_extract (v2f64 (fneg V128:$Rm)),
                                         VectorIndexS:$idx))),
            (FMLSv1i64_indexed FPR64:$Rd, FPR64:$Rn,
                V128:$Rm, VectorIndexS:$idx)>;
}

defm : FMLSIndexedAfterNegPatterns<
           TriOpFrag<(fma node:$RHS, node:$MHS, node:$LHS)> >;
defm : FMLSIndexedAfterNegPatterns<
           TriOpFrag<(fma node:$MHS, node:$RHS, node:$LHS)> >;

defm FMULX : SIMDFPIndexedSD<1, 0b1001, "fmulx", int_arm64_neon_fmulx>;
defm FMUL  : SIMDFPIndexedSD<0, 0b1001, "fmul", fmul>;

def : Pat<(v2f32 (fmul V64:$Rn, (ARM64dup (f32 FPR32:$Rm)))),
          (FMULv2i32_indexed V64:$Rn,
            (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR32:$Rm, ssub),
            (i64 0))>;
def : Pat<(v4f32 (fmul V128:$Rn, (ARM64dup (f32 FPR32:$Rm)))),
          (FMULv4i32_indexed V128:$Rn,
            (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR32:$Rm, ssub),
            (i64 0))>;
def : Pat<(v2f64 (fmul V128:$Rn, (ARM64dup (f64 FPR64:$Rm)))),
          (FMULv2i64_indexed V128:$Rn,
            (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR64:$Rm, dsub),
            (i64 0))>;

defm SQDMULH : SIMDIndexedHS<0, 0b1100, "sqdmulh", int_arm64_neon_sqdmulh>;
defm SQRDMULH : SIMDIndexedHS<0, 0b1101, "sqrdmulh", int_arm64_neon_sqrdmulh>;
defm MLA   : SIMDVectorIndexedHSTied<1, 0b0000, "mla",
              TriOpFrag<(add node:$LHS, (mul node:$MHS, node:$RHS))>>;
defm MLS   : SIMDVectorIndexedHSTied<1, 0b0100, "mls",
              TriOpFrag<(sub node:$LHS, (mul node:$MHS, node:$RHS))>>;
defm MUL   : SIMDVectorIndexedHS<0, 0b1000, "mul", mul>;
defm SMLAL : SIMDVectorIndexedLongSDTied<0, 0b0010, "smlal",
    TriOpFrag<(add node:$LHS, (int_arm64_neon_smull node:$MHS, node:$RHS))>>;
defm SMLSL : SIMDVectorIndexedLongSDTied<0, 0b0110, "smlsl",
    TriOpFrag<(sub node:$LHS, (int_arm64_neon_smull node:$MHS, node:$RHS))>>;
defm SMULL : SIMDVectorIndexedLongSD<0, 0b1010, "smull",
                int_arm64_neon_smull>;
defm SQDMLAL : SIMDIndexedLongSQDMLXSDTied<0, 0b0011, "sqdmlal",
                                           int_arm64_neon_sqadd>;
defm SQDMLSL : SIMDIndexedLongSQDMLXSDTied<0, 0b0111, "sqdmlsl",
                                           int_arm64_neon_sqsub>;
defm SQDMULL : SIMDIndexedLongSD<0, 0b1011, "sqdmull", int_arm64_neon_sqdmull>;
defm UMLAL   : SIMDVectorIndexedLongSDTied<1, 0b0010, "umlal",
    TriOpFrag<(add node:$LHS, (int_arm64_neon_umull node:$MHS, node:$RHS))>>;
defm UMLSL   : SIMDVectorIndexedLongSDTied<1, 0b0110, "umlsl",
    TriOpFrag<(sub node:$LHS, (int_arm64_neon_umull node:$MHS, node:$RHS))>>;
defm UMULL   : SIMDVectorIndexedLongSD<1, 0b1010, "umull",
                int_arm64_neon_umull>;

// A scalar sqdmull with the second operand being a vector lane can be
// handled directly with the indexed instruction encoding.
def : Pat<(int_arm64_neon_sqdmulls_scalar (i32 FPR32:$Rn),
                                          (vector_extract (v4i32 V128:$Vm),
                                                           VectorIndexS:$idx)),
          (SQDMULLv1i64_indexed FPR32:$Rn, V128:$Vm, VectorIndexS:$idx)>;

//----------------------------------------------------------------------------
// AdvSIMD scalar shift instructions
//----------------------------------------------------------------------------
defm FCVTZS : SIMDScalarRShiftSD<0, 0b11111, "fcvtzs">;
defm FCVTZU : SIMDScalarRShiftSD<1, 0b11111, "fcvtzu">;
defm SCVTF  : SIMDScalarRShiftSD<0, 0b11100, "scvtf">;
defm UCVTF  : SIMDScalarRShiftSD<1, 0b11100, "ucvtf">;
// Codegen patterns for the above. We don't put these directly on the
// instructions because TableGen's type inference can't handle the truth.
// Having the same base pattern for fp <--> int totally freaks it out.
def : Pat<(int_arm64_neon_vcvtfp2fxs FPR32:$Rn, vecshiftR32:$imm),
          (FCVTZSs FPR32:$Rn, vecshiftR32:$imm)>;
def : Pat<(int_arm64_neon_vcvtfp2fxu FPR32:$Rn, vecshiftR32:$imm),
          (FCVTZUs FPR32:$Rn, vecshiftR32:$imm)>;
def : Pat<(i64 (int_arm64_neon_vcvtfp2fxs (f64 FPR64:$Rn), vecshiftR64:$imm)),
          (FCVTZSd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(i64 (int_arm64_neon_vcvtfp2fxu (f64 FPR64:$Rn), vecshiftR64:$imm)),
          (FCVTZUd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(v1i64 (int_arm64_neon_vcvtfp2fxs (v1f64 FPR64:$Rn),
                                            vecshiftR64:$imm)),
          (FCVTZSd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(v1i64 (int_arm64_neon_vcvtfp2fxu (v1f64 FPR64:$Rn),
                                            vecshiftR64:$imm)),
          (FCVTZUd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(int_arm64_neon_vcvtfxs2fp FPR32:$Rn, vecshiftR32:$imm),
          (SCVTFs FPR32:$Rn, vecshiftR32:$imm)>;
def : Pat<(int_arm64_neon_vcvtfxu2fp FPR32:$Rn, vecshiftR32:$imm),
          (UCVTFs FPR32:$Rn, vecshiftR32:$imm)>;
def : Pat<(f64 (int_arm64_neon_vcvtfxs2fp (i64 FPR64:$Rn), vecshiftR64:$imm)),
          (SCVTFd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(f64 (int_arm64_neon_vcvtfxu2fp (i64 FPR64:$Rn), vecshiftR64:$imm)),
          (UCVTFd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(v1f64 (int_arm64_neon_vcvtfxs2fp (v1i64 FPR64:$Rn),
                                            vecshiftR64:$imm)),
          (SCVTFd FPR64:$Rn, vecshiftR64:$imm)>;
def : Pat<(v1f64 (int_arm64_neon_vcvtfxu2fp (v1i64 FPR64:$Rn),
                                            vecshiftR64:$imm)),
          (UCVTFd FPR64:$Rn, vecshiftR64:$imm)>;

defm SHL      : SIMDScalarLShiftD<   0, 0b01010, "shl", ARM64vshl>;
defm SLI      : SIMDScalarLShiftDTied<1, 0b01010, "sli">;
defm SQRSHRN  : SIMDScalarRShiftBHS< 0, 0b10011, "sqrshrn",
                                     int_arm64_neon_sqrshrn>;
defm SQRSHRUN : SIMDScalarRShiftBHS< 1, 0b10001, "sqrshrun",
                                     int_arm64_neon_sqrshrun>;
defm SQSHLU   : SIMDScalarLShiftBHSD<1, 0b01100, "sqshlu", ARM64sqshlui>;
defm SQSHL    : SIMDScalarLShiftBHSD<0, 0b01110, "sqshl", ARM64sqshli>;
defm SQSHRN   : SIMDScalarRShiftBHS< 0, 0b10010, "sqshrn",
                                     int_arm64_neon_sqshrn>;
defm SQSHRUN  : SIMDScalarRShiftBHS< 1, 0b10000, "sqshrun",
                                     int_arm64_neon_sqshrun>;
defm SRI      : SIMDScalarRShiftDTied<   1, 0b01000, "sri">;
defm SRSHR    : SIMDScalarRShiftD<   0, 0b00100, "srshr", ARM64srshri>;
defm SRSRA    : SIMDScalarRShiftDTied<   0, 0b00110, "srsra",
    TriOpFrag<(add node:$LHS,
                   (ARM64srshri node:$MHS, node:$RHS))>>;
defm SSHR     : SIMDScalarRShiftD<   0, 0b00000, "sshr", ARM64vashr>;
defm SSRA     : SIMDScalarRShiftDTied<   0, 0b00010, "ssra",
    TriOpFrag<(add node:$LHS,
                   (ARM64vashr node:$MHS, node:$RHS))>>;
defm UQRSHRN  : SIMDScalarRShiftBHS< 1, 0b10011, "uqrshrn",
                                     int_arm64_neon_uqrshrn>;
defm UQSHL    : SIMDScalarLShiftBHSD<1, 0b01110, "uqshl", ARM64uqshli>;
defm UQSHRN   : SIMDScalarRShiftBHS< 1, 0b10010, "uqshrn",
                                     int_arm64_neon_uqshrn>;
defm URSHR    : SIMDScalarRShiftD<   1, 0b00100, "urshr", ARM64urshri>;
defm URSRA    : SIMDScalarRShiftDTied<   1, 0b00110, "ursra",
    TriOpFrag<(add node:$LHS,
                   (ARM64urshri node:$MHS, node:$RHS))>>;
defm USHR     : SIMDScalarRShiftD<   1, 0b00000, "ushr", ARM64vlshr>;
defm USRA     : SIMDScalarRShiftDTied<   1, 0b00010, "usra",
    TriOpFrag<(add node:$LHS,
                   (ARM64vlshr node:$MHS, node:$RHS))>>;

//----------------------------------------------------------------------------
// AdvSIMD vector shift instructions
//----------------------------------------------------------------------------
defm FCVTZS:SIMDVectorRShiftSD<0, 0b11111, "fcvtzs", int_arm64_neon_vcvtfp2fxs>;
defm FCVTZU:SIMDVectorRShiftSD<1, 0b11111, "fcvtzu", int_arm64_neon_vcvtfp2fxu>;
defm SCVTF: SIMDVectorRShiftSDToFP<0, 0b11100, "scvtf",
                                   int_arm64_neon_vcvtfxs2fp>;
defm RSHRN   : SIMDVectorRShiftNarrowBHS<0, 0b10001, "rshrn",
                                         int_arm64_neon_rshrn>;
defm SHL     : SIMDVectorLShiftBHSD<0, 0b01010, "shl", ARM64vshl>;
defm SHRN    : SIMDVectorRShiftNarrowBHS<0, 0b10000, "shrn",
                          BinOpFrag<(trunc (ARM64vashr node:$LHS, node:$RHS))>>;
defm SLI     : SIMDVectorLShiftBHSDTied<1, 0b01010, "sli", int_arm64_neon_vsli>;
def : Pat<(v1i64 (int_arm64_neon_vsli (v1i64 FPR64:$Rd), (v1i64 FPR64:$Rn),
                                      (i32 vecshiftL64:$imm))),
          (SLId FPR64:$Rd, FPR64:$Rn, vecshiftL64:$imm)>;
defm SQRSHRN : SIMDVectorRShiftNarrowBHS<0, 0b10011, "sqrshrn",
                                         int_arm64_neon_sqrshrn>;
defm SQRSHRUN: SIMDVectorRShiftNarrowBHS<1, 0b10001, "sqrshrun",
                                         int_arm64_neon_sqrshrun>;
defm SQSHLU : SIMDVectorLShiftBHSD<1, 0b01100, "sqshlu", ARM64sqshlui>;
defm SQSHL  : SIMDVectorLShiftBHSD<0, 0b01110, "sqshl", ARM64sqshli>;
defm SQSHRN  : SIMDVectorRShiftNarrowBHS<0, 0b10010, "sqshrn",
                                         int_arm64_neon_sqshrn>;
defm SQSHRUN : SIMDVectorRShiftNarrowBHS<1, 0b10000, "sqshrun",
                                         int_arm64_neon_sqshrun>;
defm SRI     : SIMDVectorRShiftBHSDTied<1, 0b01000, "sri", int_arm64_neon_vsri>;
def : Pat<(v1i64 (int_arm64_neon_vsri (v1i64 FPR64:$Rd), (v1i64 FPR64:$Rn),
                                      (i32 vecshiftR64:$imm))),
          (SRId FPR64:$Rd, FPR64:$Rn, vecshiftR64:$imm)>;
defm SRSHR   : SIMDVectorRShiftBHSD<0, 0b00100, "srshr", ARM64srshri>;
defm SRSRA   : SIMDVectorRShiftBHSDTied<0, 0b00110, "srsra",
                 TriOpFrag<(add node:$LHS,
                                (ARM64srshri node:$MHS, node:$RHS))> >;
defm SSHLL   : SIMDVectorLShiftLongBHSD<0, 0b10100, "sshll",
                BinOpFrag<(ARM64vshl (sext node:$LHS), node:$RHS)>>;

defm SSHR    : SIMDVectorRShiftBHSD<0, 0b00000, "sshr", ARM64vashr>;
defm SSRA    : SIMDVectorRShiftBHSDTied<0, 0b00010, "ssra",
                TriOpFrag<(add node:$LHS, (ARM64vashr node:$MHS, node:$RHS))>>;
defm UCVTF   : SIMDVectorRShiftSDToFP<1, 0b11100, "ucvtf",
                        int_arm64_neon_vcvtfxu2fp>;
defm UQRSHRN : SIMDVectorRShiftNarrowBHS<1, 0b10011, "uqrshrn",
                                         int_arm64_neon_uqrshrn>;
defm UQSHL   : SIMDVectorLShiftBHSD<1, 0b01110, "uqshl", ARM64uqshli>;
defm UQSHRN  : SIMDVectorRShiftNarrowBHS<1, 0b10010, "uqshrn",
                                         int_arm64_neon_uqshrn>;
defm URSHR   : SIMDVectorRShiftBHSD<1, 0b00100, "urshr", ARM64urshri>;
defm URSRA   : SIMDVectorRShiftBHSDTied<1, 0b00110, "ursra",
                TriOpFrag<(add node:$LHS,
                               (ARM64urshri node:$MHS, node:$RHS))> >;
defm USHLL   : SIMDVectorLShiftLongBHSD<1, 0b10100, "ushll",
                BinOpFrag<(ARM64vshl (zext node:$LHS), node:$RHS)>>;
defm USHR    : SIMDVectorRShiftBHSD<1, 0b00000, "ushr", ARM64vlshr>;
defm USRA    : SIMDVectorRShiftBHSDTied<1, 0b00010, "usra",
                TriOpFrag<(add node:$LHS, (ARM64vlshr node:$MHS, node:$RHS))> >;

// SHRN patterns for when a logical right shift was used instead of arithmetic
// (the immediate guarantees no sign bits actually end up in the result so it
// doesn't matter).
def : Pat<(v8i8 (trunc (ARM64vlshr (v8i16 V128:$Rn), vecshiftR16Narrow:$imm))),
          (SHRNv8i8_shift V128:$Rn, vecshiftR16Narrow:$imm)>;
def : Pat<(v4i16 (trunc (ARM64vlshr (v4i32 V128:$Rn), vecshiftR32Narrow:$imm))),
          (SHRNv4i16_shift V128:$Rn, vecshiftR32Narrow:$imm)>;
def : Pat<(v2i32 (trunc (ARM64vlshr (v2i64 V128:$Rn), vecshiftR64Narrow:$imm))),
          (SHRNv2i32_shift V128:$Rn, vecshiftR64Narrow:$imm)>;

def : Pat<(v16i8 (concat_vectors (v8i8 V64:$Rd),
                                 (trunc (ARM64vlshr (v8i16 V128:$Rn),
                                                    vecshiftR16Narrow:$imm)))),
          (SHRNv16i8_shift (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rd, dsub),
                           V128:$Rn, vecshiftR16Narrow:$imm)>;
def : Pat<(v8i16 (concat_vectors (v4i16 V64:$Rd),
                                 (trunc (ARM64vlshr (v4i32 V128:$Rn),
                                                    vecshiftR32Narrow:$imm)))),
          (SHRNv8i16_shift (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rd, dsub),
                           V128:$Rn, vecshiftR32Narrow:$imm)>;
def : Pat<(v4i32 (concat_vectors (v2i32 V64:$Rd),
                                 (trunc (ARM64vlshr (v2i64 V128:$Rn),
                                                    vecshiftR64Narrow:$imm)))),
          (SHRNv4i32_shift (INSERT_SUBREG (IMPLICIT_DEF), V64:$Rd, dsub),
                           V128:$Rn, vecshiftR32Narrow:$imm)>;

// Vector sign and zero extensions are implemented with SSHLL and USSHLL.
// Anyexts are implemented as zexts.
def : Pat<(v8i16 (sext   (v8i8 V64:$Rn))),  (SSHLLv8i8_shift  V64:$Rn, (i32 0))>;
def : Pat<(v8i16 (zext   (v8i8 V64:$Rn))),  (USHLLv8i8_shift  V64:$Rn, (i32 0))>;
def : Pat<(v8i16 (anyext (v8i8 V64:$Rn))),  (USHLLv8i8_shift  V64:$Rn, (i32 0))>;
def : Pat<(v4i32 (sext   (v4i16 V64:$Rn))), (SSHLLv4i16_shift V64:$Rn, (i32 0))>;
def : Pat<(v4i32 (zext   (v4i16 V64:$Rn))), (USHLLv4i16_shift V64:$Rn, (i32 0))>;
def : Pat<(v4i32 (anyext (v4i16 V64:$Rn))), (USHLLv4i16_shift V64:$Rn, (i32 0))>;
def : Pat<(v2i64 (sext   (v2i32 V64:$Rn))), (SSHLLv2i32_shift V64:$Rn, (i32 0))>;
def : Pat<(v2i64 (zext   (v2i32 V64:$Rn))), (USHLLv2i32_shift V64:$Rn, (i32 0))>;
def : Pat<(v2i64 (anyext (v2i32 V64:$Rn))), (USHLLv2i32_shift V64:$Rn, (i32 0))>;
// Also match an extend from the upper half of a 128 bit source register.
def : Pat<(v8i16 (anyext (v8i8 (extract_subvector V128:$Rn, (i64 8)) ))),
          (USHLLv16i8_shift V128:$Rn, (i32 0))>;
def : Pat<(v8i16 (zext   (v8i8 (extract_subvector V128:$Rn, (i64 8)) ))),
          (USHLLv16i8_shift V128:$Rn, (i32 0))>;
def : Pat<(v8i16 (sext   (v8i8 (extract_subvector V128:$Rn, (i64 8)) ))),
          (SSHLLv16i8_shift V128:$Rn, (i32 0))>;
def : Pat<(v4i32 (anyext (v4i16 (extract_subvector V128:$Rn, (i64 4)) ))),
          (USHLLv8i16_shift V128:$Rn, (i32 0))>;
def : Pat<(v4i32 (zext   (v4i16 (extract_subvector V128:$Rn, (i64 4)) ))),
          (USHLLv8i16_shift V128:$Rn, (i32 0))>;
def : Pat<(v4i32 (sext   (v4i16 (extract_subvector V128:$Rn, (i64 4)) ))),
          (SSHLLv8i16_shift V128:$Rn, (i32 0))>;
def : Pat<(v2i64 (anyext (v2i32 (extract_subvector V128:$Rn, (i64 2)) ))),
          (USHLLv4i32_shift V128:$Rn, (i32 0))>;
def : Pat<(v2i64 (zext   (v2i32 (extract_subvector V128:$Rn, (i64 2)) ))),
          (USHLLv4i32_shift V128:$Rn, (i32 0))>;
def : Pat<(v2i64 (sext   (v2i32 (extract_subvector V128:$Rn, (i64 2)) ))),
          (SSHLLv4i32_shift V128:$Rn, (i32 0))>;

// Vector shift sxtl aliases
def : InstAlias<"sxtl.8h $dst, $src1",
                (SSHLLv8i8_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"sxtl $dst.8h, $src1.8b",
                (SSHLLv8i8_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"sxtl.4s $dst, $src1",
                (SSHLLv4i16_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"sxtl $dst.4s, $src1.4h",
                (SSHLLv4i16_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"sxtl.2d $dst, $src1",
                (SSHLLv2i32_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"sxtl $dst.2d, $src1.2s",
                (SSHLLv2i32_shift V128:$dst, V64:$src1, 0)>;

// Vector shift sxtl2 aliases
def : InstAlias<"sxtl2.8h $dst, $src1",
                (SSHLLv16i8_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"sxtl2 $dst.8h, $src1.16b",
                (SSHLLv16i8_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"sxtl2.4s $dst, $src1",
                (SSHLLv8i16_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"sxtl2 $dst.4s, $src1.8h",
                (SSHLLv8i16_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"sxtl2.2d $dst, $src1",
                (SSHLLv4i32_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"sxtl2 $dst.2d, $src1.4s",
                (SSHLLv4i32_shift V128:$dst, V128:$src1, 0)>;

// Vector shift uxtl aliases
def : InstAlias<"uxtl.8h $dst, $src1",
                (USHLLv8i8_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"uxtl $dst.8h, $src1.8b",
                (USHLLv8i8_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"uxtl.4s $dst, $src1",
                (USHLLv4i16_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"uxtl $dst.4s, $src1.4h",
                (USHLLv4i16_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"uxtl.2d $dst, $src1",
                (USHLLv2i32_shift V128:$dst, V64:$src1, 0)>;
def : InstAlias<"uxtl $dst.2d, $src1.2s",
                (USHLLv2i32_shift V128:$dst, V64:$src1, 0)>;

// Vector shift uxtl2 aliases
def : InstAlias<"uxtl2.8h $dst, $src1",
                (USHLLv16i8_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"uxtl2 $dst.8h, $src1.16b",
                (USHLLv16i8_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"uxtl2.4s $dst, $src1",
                (USHLLv8i16_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"uxtl2 $dst.4s, $src1.8h",
                (USHLLv8i16_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"uxtl2.2d $dst, $src1",
                (USHLLv4i32_shift V128:$dst, V128:$src1, 0)>;
def : InstAlias<"uxtl2 $dst.2d, $src1.4s",
                (USHLLv4i32_shift V128:$dst, V128:$src1, 0)>;

// If an integer is about to be converted to a floating point value,
// just load it on the floating point unit.
// These patterns are more complex because floating point loads do not
// support sign extension.
// The sign extension has to be explicitly added and is only supported for
// one step: byte-to-half, half-to-word, word-to-doubleword.
// SCVTF GPR -> FPR is 9 cycles.
// SCVTF FPR -> FPR is 4 cyclces.
// (sign extension with lengthen) SXTL FPR -> FPR is 2 cycles.
// Therefore, we can do 2 sign extensions and one SCVTF FPR -> FPR
// and still being faster.
// However, this is not good for code size.
// 8-bits -> float. 2 sizes step-up.
def : Pat <(f32 (sint_to_fp (i32 (sextloadi8 ro_indexed8:$addr)))),
           (SCVTFv1i32 (f32 (EXTRACT_SUBREG
                              (SSHLLv4i16_shift
                                (f64
                                  (EXTRACT_SUBREG
                                    (SSHLLv8i8_shift
                                      (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                                  (LDRBro ro_indexed8:$addr),
                                                  bsub),
                                     0),
                                   dsub)),
                               0),
                           ssub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f32 (sint_to_fp (i32 (sextloadi8 am_indexed8:$addr)))),
           (SCVTFv1i32 (f32 (EXTRACT_SUBREG
                              (SSHLLv4i16_shift
                                (f64
                                  (EXTRACT_SUBREG
                                    (SSHLLv8i8_shift
                                      (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                                  (LDRBui am_indexed8:$addr),
                                                  bsub),
                                     0),
                                   dsub)),
                               0),
                           ssub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f32 (sint_to_fp (i32 (sextloadi8 am_unscaled8:$addr)))),
           (SCVTFv1i32 (f32 (EXTRACT_SUBREG
                              (SSHLLv4i16_shift
                                (f64
                                  (EXTRACT_SUBREG
                                    (SSHLLv8i8_shift
                                      (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                                  (LDURBi am_unscaled8:$addr),
                                                  bsub),
                                     0),
                                   dsub)),
                               0),
                           ssub)))>, Requires<[NotForCodeSize]>;
// 16-bits -> float. 1 size step-up.
def : Pat <(f32 (sint_to_fp (i32 (sextloadi16 ro_indexed16:$addr)))),
           (SCVTFv1i32 (f32 (EXTRACT_SUBREG
                              (SSHLLv4i16_shift
                                (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                               (LDRHro ro_indexed16:$addr),
                                               hsub),
                               0),
                           ssub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f32 (sint_to_fp (i32 (sextloadi16 am_indexed16:$addr)))),
           (SCVTFv1i32 (f32 (EXTRACT_SUBREG
                              (SSHLLv4i16_shift
                                (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                               (LDRHui am_indexed16:$addr),
                                               hsub),
                               0),
                           ssub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f32 (sint_to_fp (i32 (sextloadi16 am_unscaled16:$addr)))),
           (SCVTFv1i32 (f32 (EXTRACT_SUBREG
                              (SSHLLv4i16_shift
                                (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                               (LDURHi am_unscaled16:$addr),
                                               hsub),
                               0),
                           ssub)))>, Requires<[NotForCodeSize]>;
// 32-bits to 32-bits are handled in target specific dag combine:
// performIntToFpCombine.
// 64-bits integer to 32-bits floating point, not possible with
// SCVTF on floating point registers (both source and destination
// must have the same size).

// Here are the patterns for 8, 16, 32, and 64-bits to double.
// 8-bits -> double. 3 size step-up: give up.
// 16-bits -> double. 2 size step.
def : Pat <(f64 (sint_to_fp (i32 (sextloadi16 ro_indexed16:$addr)))),
           (SCVTFv1i64 (f64 (EXTRACT_SUBREG
                              (SSHLLv2i32_shift
                                 (f64
                                  (EXTRACT_SUBREG
                                    (SSHLLv4i16_shift
                                      (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                                  (LDRHro ro_indexed16:$addr),
                                                  hsub),
                                     0),
                                   dsub)),
                               0),
                             dsub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f64 (sint_to_fp (i32 (sextloadi16 am_indexed16:$addr)))),
           (SCVTFv1i64  (f64 (EXTRACT_SUBREG
                               (SSHLLv2i32_shift
                                 (f64
                                   (EXTRACT_SUBREG
                                     (SSHLLv4i16_shift
                                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                                  (LDRHui am_indexed16:$addr),
                                                  hsub),
                                      0),
                                    dsub)),
                                 0),
                              dsub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f64 (sint_to_fp (i32 (sextloadi16 am_unscaled16:$addr)))),
           (SCVTFv1i64 (f64 (EXTRACT_SUBREG
                              (SSHLLv2i32_shift
                                (f64
                                  (EXTRACT_SUBREG
                                    (SSHLLv4i16_shift
                                     (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                                  (LDURHi am_unscaled16:$addr),
                                                  hsub),
                                      0),
                                   dsub)),
                               0),
                             dsub)))>, Requires<[NotForCodeSize]>;
// 32-bits -> double. 1 size step-up.
def : Pat <(f64 (sint_to_fp (i32 (load ro_indexed32:$addr)))),
           (SCVTFv1i64 (f64 (EXTRACT_SUBREG
                              (SSHLLv2i32_shift
                                (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                               (LDRSro ro_indexed32:$addr),
                                               ssub),
                               0),
                             dsub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f64 (sint_to_fp (i32 (load am_indexed32:$addr)))),
           (SCVTFv1i64 (f64 (EXTRACT_SUBREG
                              (SSHLLv2i32_shift
                                (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                               (LDRSui am_indexed32:$addr),
                                               ssub),
                               0),
                             dsub)))>, Requires<[NotForCodeSize]>;
def : Pat <(f64 (sint_to_fp (i32 (load am_unscaled32:$addr)))),
           (SCVTFv1i64 (f64 (EXTRACT_SUBREG
                              (SSHLLv2i32_shift
                                (INSERT_SUBREG (f64 (IMPLICIT_DEF)),
                                               (LDURSi am_unscaled32:$addr),
                                               ssub),
                               0),
                             dsub)))>, Requires<[NotForCodeSize]>;
// 64-bits -> double are handled in target specific dag combine:
// performIntToFpCombine.


//----------------------------------------------------------------------------
// AdvSIMD Load-Store Structure
//----------------------------------------------------------------------------
defm LD1 : SIMDLd1Multiple<"ld1">;
defm LD2 : SIMDLd2Multiple<"ld2">;
defm LD3 : SIMDLd3Multiple<"ld3">;
defm LD4 : SIMDLd4Multiple<"ld4">;

defm ST1 : SIMDSt1Multiple<"st1">;
defm ST2 : SIMDSt2Multiple<"st2">;
defm ST3 : SIMDSt3Multiple<"st3">;
defm ST4 : SIMDSt4Multiple<"st4">;

class Ld1Pat<ValueType ty, Instruction INST>
  : Pat<(ty (load am_simdnoindex:$vaddr)), (INST am_simdnoindex:$vaddr)>;

def : Ld1Pat<v16i8, LD1Onev16b>;
def : Ld1Pat<v8i16, LD1Onev8h>;
def : Ld1Pat<v4i32, LD1Onev4s>;
def : Ld1Pat<v2i64, LD1Onev2d>;
def : Ld1Pat<v8i8,  LD1Onev8b>;
def : Ld1Pat<v4i16, LD1Onev4h>;
def : Ld1Pat<v2i32, LD1Onev2s>;
def : Ld1Pat<v1i64, LD1Onev1d>;

class St1Pat<ValueType ty, Instruction INST>
  : Pat<(store ty:$Vt, am_simdnoindex:$vaddr),
        (INST ty:$Vt, am_simdnoindex:$vaddr)>;

def : St1Pat<v16i8, ST1Onev16b>;
def : St1Pat<v8i16, ST1Onev8h>;
def : St1Pat<v4i32, ST1Onev4s>;
def : St1Pat<v2i64, ST1Onev2d>;
def : St1Pat<v8i8,  ST1Onev8b>;
def : St1Pat<v4i16, ST1Onev4h>;
def : St1Pat<v2i32, ST1Onev2s>;
def : St1Pat<v1i64, ST1Onev1d>;

//---
// Single-element
//---

defm LD1R          : SIMDLdR<0, 0b110, 0, "ld1r", "One", 1, 2, 4, 8>;
defm LD2R          : SIMDLdR<1, 0b110, 0, "ld2r", "Two", 2, 4, 8, 16>;
defm LD3R          : SIMDLdR<0, 0b111, 0, "ld3r", "Three", 3, 6, 12, 24>;
defm LD4R          : SIMDLdR<1, 0b111, 0, "ld4r", "Four", 4, 8, 16, 32>;
let mayLoad = 1, neverHasSideEffects = 1 in {
defm LD1 : SIMDLdSingleBTied<0, 0b000,       "ld1", VecListOneb,   GPR64pi1>;
defm LD1 : SIMDLdSingleHTied<0, 0b010, 0,    "ld1", VecListOneh,   GPR64pi2>;
defm LD1 : SIMDLdSingleSTied<0, 0b100, 0b00, "ld1", VecListOnes,   GPR64pi4>;
defm LD1 : SIMDLdSingleDTied<0, 0b100, 0b01, "ld1", VecListOned,   GPR64pi8>;
defm LD2 : SIMDLdSingleBTied<1, 0b000,       "ld2", VecListTwob,   GPR64pi2>;
defm LD2 : SIMDLdSingleHTied<1, 0b010, 0,    "ld2", VecListTwoh,   GPR64pi4>;
defm LD2 : SIMDLdSingleSTied<1, 0b100, 0b00, "ld2", VecListTwos,   GPR64pi8>;
defm LD2 : SIMDLdSingleDTied<1, 0b100, 0b01, "ld2", VecListTwod,   GPR64pi16>;
defm LD3 : SIMDLdSingleBTied<0, 0b001,       "ld3", VecListThreeb, GPR64pi3>;
defm LD3 : SIMDLdSingleHTied<0, 0b011, 0,    "ld3", VecListThreeh, GPR64pi6>;
defm LD3 : SIMDLdSingleSTied<0, 0b101, 0b00, "ld3", VecListThrees, GPR64pi12>;
defm LD3 : SIMDLdSingleDTied<0, 0b101, 0b01, "ld3", VecListThreed, GPR64pi24>;
defm LD4 : SIMDLdSingleBTied<1, 0b001,       "ld4", VecListFourb,  GPR64pi4>;
defm LD4 : SIMDLdSingleHTied<1, 0b011, 0,    "ld4", VecListFourh,  GPR64pi8>;
defm LD4 : SIMDLdSingleSTied<1, 0b101, 0b00, "ld4", VecListFours,  GPR64pi16>;
defm LD4 : SIMDLdSingleDTied<1, 0b101, 0b01, "ld4", VecListFourd,  GPR64pi32>;
}

def : Pat<(v8i8 (ARM64dup (i32 (extloadi8 am_simdnoindex:$vaddr)))),
          (LD1Rv8b am_simdnoindex:$vaddr)>;
def : Pat<(v16i8 (ARM64dup (i32 (extloadi8 am_simdnoindex:$vaddr)))),
          (LD1Rv16b am_simdnoindex:$vaddr)>;
def : Pat<(v4i16 (ARM64dup (i32 (extloadi16 am_simdnoindex:$vaddr)))),
          (LD1Rv4h am_simdnoindex:$vaddr)>;
def : Pat<(v8i16 (ARM64dup (i32 (extloadi16 am_simdnoindex:$vaddr)))),
          (LD1Rv8h am_simdnoindex:$vaddr)>;
def : Pat<(v2i32 (ARM64dup (i32 (load am_simdnoindex:$vaddr)))),
          (LD1Rv2s am_simdnoindex:$vaddr)>;
def : Pat<(v4i32 (ARM64dup (i32 (load am_simdnoindex:$vaddr)))),
          (LD1Rv4s am_simdnoindex:$vaddr)>;
def : Pat<(v2i64 (ARM64dup (i64 (load am_simdnoindex:$vaddr)))),
          (LD1Rv2d am_simdnoindex:$vaddr)>;
def : Pat<(v1i64 (ARM64dup (i64 (load am_simdnoindex:$vaddr)))),
          (LD1Rv1d am_simdnoindex:$vaddr)>;
// Grab the floating point version too
def : Pat<(v2f32 (ARM64dup (f32 (load am_simdnoindex:$vaddr)))),
          (LD1Rv2s am_simdnoindex:$vaddr)>;
def : Pat<(v4f32 (ARM64dup (f32 (load am_simdnoindex:$vaddr)))),
          (LD1Rv4s am_simdnoindex:$vaddr)>;
def : Pat<(v2f64 (ARM64dup (f64 (load am_simdnoindex:$vaddr)))),
          (LD1Rv2d am_simdnoindex:$vaddr)>;
def : Pat<(v1f64 (ARM64dup (f64 (load am_simdnoindex:$vaddr)))),
          (LD1Rv1d am_simdnoindex:$vaddr)>;

class Ld1Lane128Pat<SDPatternOperator scalar_load, Operand VecIndex,
                    ValueType VTy, ValueType STy, Instruction LD1>
  : Pat<(vector_insert (VTy VecListOne128:$Rd),
           (STy (scalar_load am_simdnoindex:$vaddr)), VecIndex:$idx),
        (LD1 VecListOne128:$Rd, VecIndex:$idx, am_simdnoindex:$vaddr)>;

def : Ld1Lane128Pat<extloadi8,  VectorIndexB, v16i8, i32, LD1i8>;
def : Ld1Lane128Pat<extloadi16, VectorIndexH, v8i16, i32, LD1i16>;
def : Ld1Lane128Pat<load,       VectorIndexS, v4i32, i32, LD1i32>;
def : Ld1Lane128Pat<load,       VectorIndexS, v4f32, f32, LD1i32>;
def : Ld1Lane128Pat<load,       VectorIndexD, v2i64, i64, LD1i64>;
def : Ld1Lane128Pat<load,       VectorIndexD, v2f64, f64, LD1i64>;

class Ld1Lane64Pat<SDPatternOperator scalar_load, Operand VecIndex,
                   ValueType VTy, ValueType STy, Instruction LD1>
  : Pat<(vector_insert (VTy VecListOne64:$Rd),
           (STy (scalar_load am_simdnoindex:$vaddr)), VecIndex:$idx),
        (EXTRACT_SUBREG
            (LD1 (SUBREG_TO_REG (i32 0), VecListOne64:$Rd, dsub),
                          VecIndex:$idx, am_simdnoindex:$vaddr),
            dsub)>;

def : Ld1Lane64Pat<extloadi8,  VectorIndexB, v8i8,  i32, LD1i8>;
def : Ld1Lane64Pat<extloadi16, VectorIndexH, v4i16, i32, LD1i16>;
def : Ld1Lane64Pat<load,       VectorIndexS, v2i32, i32, LD1i32>;
def : Ld1Lane64Pat<load,       VectorIndexS, v2f32, f32, LD1i32>;


defm LD1 : SIMDLdSt1SingleAliases<"ld1">;
defm LD2 : SIMDLdSt2SingleAliases<"ld2">;
defm LD3 : SIMDLdSt3SingleAliases<"ld3">;
defm LD4 : SIMDLdSt4SingleAliases<"ld4">;

// Stores
defm ST1 : SIMDStSingleB<0, 0b000,       "st1", VecListOneb, GPR64pi1>;
defm ST1 : SIMDStSingleH<0, 0b010, 0,    "st1", VecListOneh, GPR64pi2>;
defm ST1 : SIMDStSingleS<0, 0b100, 0b00, "st1", VecListOnes, GPR64pi4>;
defm ST1 : SIMDStSingleD<0, 0b100, 0b01, "st1", VecListOned, GPR64pi8>;

let AddedComplexity = 8 in
class St1Lane128Pat<SDPatternOperator scalar_store, Operand VecIndex,
                    ValueType VTy, ValueType STy, Instruction ST1>
  : Pat<(scalar_store
             (STy (vector_extract (VTy VecListOne128:$Vt), VecIndex:$idx)),
             am_simdnoindex:$vaddr),
        (ST1 VecListOne128:$Vt, VecIndex:$idx, am_simdnoindex:$vaddr)>;

def : St1Lane128Pat<truncstorei8,  VectorIndexB, v16i8, i32, ST1i8>;
def : St1Lane128Pat<truncstorei16, VectorIndexH, v8i16, i32, ST1i16>;
def : St1Lane128Pat<store,         VectorIndexS, v4i32, i32, ST1i32>;
def : St1Lane128Pat<store,         VectorIndexS, v4f32, f32, ST1i32>;
def : St1Lane128Pat<store,         VectorIndexD, v2i64, i64, ST1i64>;
def : St1Lane128Pat<store,         VectorIndexD, v2f64, f64, ST1i64>;

let AddedComplexity = 8 in
class St1Lane64Pat<SDPatternOperator scalar_store, Operand VecIndex,
                   ValueType VTy, ValueType STy, Instruction ST1>
  : Pat<(scalar_store
             (STy (vector_extract (VTy VecListOne64:$Vt), VecIndex:$idx)),
             am_simdnoindex:$vaddr),
        (ST1 (SUBREG_TO_REG (i32 0), VecListOne64:$Vt, dsub),
             VecIndex:$idx, am_simdnoindex:$vaddr)>;

def : St1Lane64Pat<truncstorei8,  VectorIndexB, v8i8, i32, ST1i8>;
def : St1Lane64Pat<truncstorei16, VectorIndexH, v4i16, i32, ST1i16>;
def : St1Lane64Pat<store,         VectorIndexS, v2i32, i32, ST1i32>;
def : St1Lane64Pat<store,         VectorIndexS, v2f32, f32, ST1i32>;

let mayStore = 1, neverHasSideEffects = 1 in {
defm ST2 : SIMDStSingleB<1, 0b000,       "st2", VecListTwob,   GPR64pi2>;
defm ST2 : SIMDStSingleH<1, 0b010, 0,    "st2", VecListTwoh,   GPR64pi4>;
defm ST2 : SIMDStSingleS<1, 0b100, 0b00, "st2", VecListTwos,   GPR64pi8>;
defm ST2 : SIMDStSingleD<1, 0b100, 0b01, "st2", VecListTwod,   GPR64pi16>;
defm ST3 : SIMDStSingleB<0, 0b001,       "st3", VecListThreeb, GPR64pi3>;
defm ST3 : SIMDStSingleH<0, 0b011, 0,    "st3", VecListThreeh, GPR64pi6>;
defm ST3 : SIMDStSingleS<0, 0b101, 0b00, "st3", VecListThrees, GPR64pi12>;
defm ST3 : SIMDStSingleD<0, 0b101, 0b01, "st3", VecListThreed, GPR64pi24>;
defm ST4 : SIMDStSingleB<1, 0b001,       "st4", VecListFourb,  GPR64pi4>;
defm ST4 : SIMDStSingleH<1, 0b011, 0,    "st4", VecListFourh,  GPR64pi8>;
defm ST4 : SIMDStSingleS<1, 0b101, 0b00, "st4", VecListFours,  GPR64pi16>;
defm ST4 : SIMDStSingleD<1, 0b101, 0b01, "st4", VecListFourd,  GPR64pi32>;
}

defm ST1 : SIMDLdSt1SingleAliases<"st1">;
defm ST2 : SIMDLdSt2SingleAliases<"st2">;
defm ST3 : SIMDLdSt3SingleAliases<"st3">;
defm ST4 : SIMDLdSt4SingleAliases<"st4">;

//----------------------------------------------------------------------------
// Crypto extensions
//----------------------------------------------------------------------------

def AESErr   : AESTiedInst<0b0100, "aese",   int_arm64_crypto_aese>;
def AESDrr   : AESTiedInst<0b0101, "aesd",   int_arm64_crypto_aesd>;
def AESMCrr  : AESInst<    0b0110, "aesmc",  int_arm64_crypto_aesmc>;
def AESIMCrr : AESInst<    0b0111, "aesimc", int_arm64_crypto_aesimc>;

def SHA1Crrr     : SHATiedInstQSV<0b000, "sha1c",   int_arm64_crypto_sha1c>;
def SHA1Prrr     : SHATiedInstQSV<0b001, "sha1p",   int_arm64_crypto_sha1p>;
def SHA1Mrrr     : SHATiedInstQSV<0b010, "sha1m",   int_arm64_crypto_sha1m>;
def SHA1SU0rrr   : SHATiedInstVVV<0b011, "sha1su0", int_arm64_crypto_sha1su0>;
def SHA256Hrrr   : SHATiedInstQQV<0b100, "sha256h", int_arm64_crypto_sha256h>;
def SHA256H2rrr  : SHATiedInstQQV<0b101, "sha256h2",int_arm64_crypto_sha256h2>;
def SHA256SU1rrr :SHATiedInstVVV<0b110, "sha256su1",int_arm64_crypto_sha256su1>;

def SHA1Hrr     : SHAInstSS<    0b0000, "sha1h",    int_arm64_crypto_sha1h>;
def SHA1SU1rr   : SHATiedInstVV<0b0001, "sha1su1",  int_arm64_crypto_sha1su1>;
def SHA256SU0rr : SHATiedInstVV<0b0010, "sha256su0",int_arm64_crypto_sha256su0>;

//----------------------------------------------------------------------------
// Compiler-pseudos
//----------------------------------------------------------------------------
// FIXME: Like for X86, these should go in their own separate .td file.

// Any instruction that defines a 32-bit result leaves the high half of the
// register. Truncate can be lowered to EXTRACT_SUBREG. CopyFromReg may
// be copying from a truncate. But any other 32-bit operation will zero-extend
// up to 64 bits.
// FIXME: X86 also checks for CMOV here. Do we need something similar?
def def32 : PatLeaf<(i32 GPR32:$src), [{
  return N->getOpcode() != ISD::TRUNCATE &&
         N->getOpcode() != TargetOpcode::EXTRACT_SUBREG &&
         N->getOpcode() != ISD::CopyFromReg;
}]>;

// In the case of a 32-bit def that is known to implicitly zero-extend,
// we can use a SUBREG_TO_REG.
def : Pat<(i64 (zext def32:$src)), (SUBREG_TO_REG (i64 0), GPR32:$src, sub_32)>;

// For an anyext, we don't care what the high bits are, so we can perform an
// INSERT_SUBREF into an IMPLICIT_DEF.
def : Pat<(i64 (anyext GPR32:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPR32:$src, sub_32)>;

// When we need to explicitly zero-extend, we use an unsigned bitfield move
// instruction (UBFM) on the enclosing super-reg.
def : Pat<(i64 (zext GPR32:$src)),
 (UBFMXri (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPR32:$src, sub_32), 0, 31)>;

// To sign extend, we use a signed bitfield move instruction (SBFM) on the
// containing super-reg.
def : Pat<(i64 (sext GPR32:$src)),
   (SBFMXri (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPR32:$src, sub_32), 0, 31)>;
def : Pat<(i64 (sext_inreg GPR64:$src, i32)), (SBFMXri GPR64:$src, 0, 31)>;
def : Pat<(i64 (sext_inreg GPR64:$src, i16)), (SBFMXri GPR64:$src, 0, 15)>;
def : Pat<(i64 (sext_inreg GPR64:$src, i8)),  (SBFMXri GPR64:$src, 0, 7)>;
def : Pat<(i64 (sext_inreg GPR64:$src, i1)),  (SBFMXri GPR64:$src, 0, 0)>;
def : Pat<(i32 (sext_inreg GPR32:$src, i16)), (SBFMWri GPR32:$src, 0, 15)>;
def : Pat<(i32 (sext_inreg GPR32:$src, i8)),  (SBFMWri GPR32:$src, 0, 7)>;
def : Pat<(i32 (sext_inreg GPR32:$src, i1)),  (SBFMWri GPR32:$src, 0, 0)>;

def : Pat<(shl (sext_inreg GPR32:$Rn, i8), (i64 imm0_31:$imm)),
          (SBFMWri GPR32:$Rn, (i64 (i32shift_a       imm0_31:$imm)),
                              (i64 (i32shift_sext_i8 imm0_31:$imm)))>;
def : Pat<(shl (sext_inreg GPR64:$Rn, i8), (i64 imm0_63:$imm)),
          (SBFMXri GPR64:$Rn, (i64 (i64shift_a imm0_63:$imm)),
                              (i64 (i64shift_sext_i8 imm0_63:$imm)))>;

def : Pat<(shl (sext_inreg GPR32:$Rn, i16), (i64 imm0_31:$imm)),
          (SBFMWri GPR32:$Rn, (i64 (i32shift_a        imm0_31:$imm)),
                              (i64 (i32shift_sext_i16 imm0_31:$imm)))>;
def : Pat<(shl (sext_inreg GPR64:$Rn, i16), (i64 imm0_63:$imm)),
          (SBFMXri GPR64:$Rn, (i64 (i64shift_a        imm0_63:$imm)),
                              (i64 (i64shift_sext_i16 imm0_63:$imm)))>;

def : Pat<(shl (i64 (sext GPR32:$Rn)), (i64 imm0_63:$imm)),
          (SBFMXri (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPR32:$Rn, sub_32),
                   (i64 (i64shift_a        imm0_63:$imm)),
                   (i64 (i64shift_sext_i32 imm0_63:$imm)))>;

// sra patterns have an AddedComplexity of 10, so make sure we have a higher
// AddedComplexity for the following patterns since we want to match sext + sra
// patterns before we attempt to match a single sra node.
let AddedComplexity = 20 in {
// We support all sext + sra combinations which preserve at least one bit of the
// original value which is to be sign extended. E.g. we support shifts up to
// bitwidth-1 bits.
def : Pat<(sra (sext_inreg GPR32:$Rn, i8), (i64 imm0_7:$imm)),
          (SBFMWri GPR32:$Rn, (i64 imm0_7:$imm), 7)>;
def : Pat<(sra (sext_inreg GPR64:$Rn, i8), (i64 imm0_7:$imm)),
          (SBFMXri GPR64:$Rn, (i64 imm0_7:$imm), 7)>;

def : Pat<(sra (sext_inreg GPR32:$Rn, i16), (i64 imm0_15:$imm)),
          (SBFMWri GPR32:$Rn, (i64 imm0_15:$imm), 15)>;
def : Pat<(sra (sext_inreg GPR64:$Rn, i16), (i64 imm0_15:$imm)),
          (SBFMXri GPR64:$Rn, (i64 imm0_15:$imm), 15)>;

def : Pat<(sra (i64 (sext GPR32:$Rn)), (i64 imm0_31:$imm)),
          (SBFMXri (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GPR32:$Rn, sub_32),
                   (i64 imm0_31:$imm), 31)>;
} // AddedComplexity = 20

// To truncate, we can simply extract from a subregister.
def : Pat<(i32 (trunc GPR64sp:$src)),
          (i32 (EXTRACT_SUBREG GPR64sp:$src, sub_32))>;

// __builtin_trap() uses the BRK instruction on ARM64.
def : Pat<(trap), (BRK 1)>;

// Conversions within AdvSIMD types in the same register size are free.

def : Pat<(v1i64 (bitconvert (v2i32 FPR64:$src))), (v1i64 FPR64:$src)>;
def : Pat<(v1i64 (bitconvert (v4i16 FPR64:$src))), (v1i64 FPR64:$src)>;
def : Pat<(v1i64 (bitconvert (v8i8  FPR64:$src))), (v1i64 FPR64:$src)>;
def : Pat<(v1i64 (bitconvert (f64   FPR64:$src))), (v1i64 FPR64:$src)>;
def : Pat<(v1i64 (bitconvert (v2f32 FPR64:$src))), (v1i64 FPR64:$src)>;
def : Pat<(v1i64 (bitconvert (v1f64 FPR64:$src))), (v1i64 FPR64:$src)>;

def : Pat<(v2i32 (bitconvert (v1i64 FPR64:$src))), (v2i32 FPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v4i16 FPR64:$src))), (v2i32 FPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v8i8  FPR64:$src))), (v2i32 FPR64:$src)>;
def : Pat<(v2i32 (bitconvert (f64   FPR64:$src))), (v2i32 FPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v2f32 FPR64:$src))), (v2i32 FPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v1f64 FPR64:$src))), (v2i32 FPR64:$src)>;

def : Pat<(v4i16 (bitconvert (v1i64 FPR64:$src))), (v4i16 FPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v2i32 FPR64:$src))), (v4i16 FPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v8i8  FPR64:$src))), (v4i16 FPR64:$src)>;
def : Pat<(v4i16 (bitconvert (f64   FPR64:$src))), (v4i16 FPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v2f32 FPR64:$src))), (v4i16 FPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v1f64 FPR64:$src))), (v4i16 FPR64:$src)>;

def : Pat<(v8i8  (bitconvert (v1i64 FPR64:$src))), (v8i8  FPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v2i32 FPR64:$src))), (v8i8  FPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v4i16 FPR64:$src))), (v8i8  FPR64:$src)>;
def : Pat<(v8i8  (bitconvert (f64   FPR64:$src))), (v8i8  FPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v2f32 FPR64:$src))), (v8i8  FPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v1f64 FPR64:$src))), (v8i8  FPR64:$src)>;

def : Pat<(f64   (bitconvert (v1i64 FPR64:$src))), (f64   FPR64:$src)>;
def : Pat<(f64   (bitconvert (v2i32 FPR64:$src))), (f64   FPR64:$src)>;
def : Pat<(f64   (bitconvert (v4i16 FPR64:$src))), (f64   FPR64:$src)>;
def : Pat<(f64   (bitconvert (v8i8  FPR64:$src))), (f64   FPR64:$src)>;
def : Pat<(f64   (bitconvert (v2f32 FPR64:$src))), (f64   FPR64:$src)>;
def : Pat<(f64   (bitconvert (v1f64 FPR64:$src))), (f64   FPR64:$src)>;

def : Pat<(v1f64 (bitconvert (v1i64 FPR64:$src))), (v1f64 FPR64:$src)>;
def : Pat<(v1f64 (bitconvert (v2i32 FPR64:$src))), (v1f64 FPR64:$src)>;
def : Pat<(v1f64 (bitconvert (v4i16 FPR64:$src))), (v1f64 FPR64:$src)>;
def : Pat<(v1f64 (bitconvert (v8i8  FPR64:$src))), (v1f64 FPR64:$src)>;
def : Pat<(v1f64 (bitconvert (f64   FPR64:$src))), (v1f64 FPR64:$src)>;
def : Pat<(v1f64 (bitconvert (v2f32 FPR64:$src))), (v1f64 FPR64:$src)>;

def : Pat<(v2f32 (bitconvert (f64   FPR64:$src))), (v2f32 FPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v1i64 FPR64:$src))), (v2f32 FPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v2i32 FPR64:$src))), (v2f32 FPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v4i16 FPR64:$src))), (v2f32 FPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v8i8  FPR64:$src))), (v2f32 FPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v1f64 FPR64:$src))), (v2f32 FPR64:$src)>;


def : Pat<(f128 (bitconvert (v2i64 FPR128:$src))), (f128 FPR128:$src)>;
def : Pat<(f128 (bitconvert (v4i32 FPR128:$src))), (f128 FPR128:$src)>;
def : Pat<(f128 (bitconvert (v8i16 FPR128:$src))), (f128 FPR128:$src)>;
def : Pat<(f128 (bitconvert (v2f64 FPR128:$src))), (f128 FPR128:$src)>;
def : Pat<(f128 (bitconvert (v4f32 FPR128:$src))), (f128 FPR128:$src)>;

def : Pat<(v2f64 (bitconvert (f128  FPR128:$src))), (v2f64 FPR128:$src)>;
def : Pat<(v2f64 (bitconvert (v4i32 FPR128:$src))), (v2f64 FPR128:$src)>;
def : Pat<(v2f64 (bitconvert (v8i16 FPR128:$src))), (v2f64 FPR128:$src)>;
def : Pat<(v2f64 (bitconvert (v16i8 FPR128:$src))), (v2f64 FPR128:$src)>;
def : Pat<(v2f64 (bitconvert (v2i64 FPR128:$src))), (v2f64 FPR128:$src)>;
def : Pat<(v2f64 (bitconvert (v4f32 FPR128:$src))), (v2f64 FPR128:$src)>;

def : Pat<(v4f32 (bitconvert (f128  FPR128:$src))), (v4f32 FPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 FPR128:$src))), (v4f32 FPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 FPR128:$src))), (v4f32 FPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v16i8 FPR128:$src))), (v4f32 FPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v2i64 FPR128:$src))), (v4f32 FPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v2f64 FPR128:$src))), (v4f32 FPR128:$src)>;

def : Pat<(v2i64 (bitconvert (f128  FPR128:$src))), (v2i64 FPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v4i32 FPR128:$src))), (v2i64 FPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v8i16 FPR128:$src))), (v2i64 FPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v16i8 FPR128:$src))), (v2i64 FPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v2f64 FPR128:$src))), (v2i64 FPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v4f32 FPR128:$src))), (v2i64 FPR128:$src)>;

def : Pat<(v4i32 (bitconvert (f128  FPR128:$src))), (v4i32 FPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v2i64 FPR128:$src))), (v4i32 FPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 FPR128:$src))), (v4i32 FPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v16i8 FPR128:$src))), (v4i32 FPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v2f64 FPR128:$src))), (v4i32 FPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 FPR128:$src))), (v4i32 FPR128:$src)>;

def : Pat<(v8i16 (bitconvert (f128  FPR128:$src))), (v8i16 FPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v2i64 FPR128:$src))), (v8i16 FPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 FPR128:$src))), (v8i16 FPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v16i8 FPR128:$src))), (v8i16 FPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v2f64 FPR128:$src))), (v8i16 FPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 FPR128:$src))), (v8i16 FPR128:$src)>;

def : Pat<(v16i8 (bitconvert (f128  FPR128:$src))), (v16i8 FPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v2i64 FPR128:$src))), (v16i8 FPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 FPR128:$src))), (v16i8 FPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v8i16 FPR128:$src))), (v16i8 FPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v2f64 FPR128:$src))), (v16i8 FPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 FPR128:$src))), (v16i8 FPR128:$src)>;

def : Pat<(v8i8 (extract_subvector (v16i8 FPR128:$Rn), (i64 1))),
          (EXTRACT_SUBREG (DUPv2i64lane FPR128:$Rn, 1), dsub)>;
def : Pat<(v4i16 (extract_subvector (v8i16 FPR128:$Rn), (i64 1))),
          (EXTRACT_SUBREG (DUPv2i64lane FPR128:$Rn, 1), dsub)>;
def : Pat<(v2i32 (extract_subvector (v4i32 FPR128:$Rn), (i64 1))),
          (EXTRACT_SUBREG (DUPv2i64lane FPR128:$Rn, 1), dsub)>;
def : Pat<(v1i64 (extract_subvector (v2i64 FPR128:$Rn), (i64 1))),
          (EXTRACT_SUBREG (DUPv2i64lane FPR128:$Rn, 1), dsub)>;

// A 64-bit subvector insert to the first 128-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(insert_subvector undef, (v1i64 FPR64:$src), (i32 0)),
          (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)), FPR64:$src, dsub)>;
def : Pat<(insert_subvector undef, (v1f64 FPR64:$src), (i32 0)),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)), FPR64:$src, dsub)>;
def : Pat<(insert_subvector undef, (v2i32 FPR64:$src), (i32 0)),
          (INSERT_SUBREG (v4i32 (IMPLICIT_DEF)), FPR64:$src, dsub)>;
def : Pat<(insert_subvector undef, (v2f32 FPR64:$src), (i32 0)),
          (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), FPR64:$src, dsub)>;
def : Pat<(insert_subvector undef, (v4i16 FPR64:$src), (i32 0)),
          (INSERT_SUBREG (v8i16 (IMPLICIT_DEF)), FPR64:$src, dsub)>;
def : Pat<(insert_subvector undef, (v8i8 FPR64:$src), (i32 0)),
          (INSERT_SUBREG (v16i8 (IMPLICIT_DEF)), FPR64:$src, dsub)>;

// Use pair-wise add instructions when summing up the lanes for v2f64, v2i64
// or v2f32.
def : Pat<(i64 (add (vector_extract (v2i64 FPR128:$Rn), (i64 0)),
                    (vector_extract (v2i64 FPR128:$Rn), (i64 1)))),
           (i64 (ADDPv2i64p (v2i64 FPR128:$Rn)))>;
def : Pat<(f64 (fadd (vector_extract (v2f64 FPR128:$Rn), (i64 0)),
                     (vector_extract (v2f64 FPR128:$Rn), (i64 1)))),
           (f64 (FADDPv2i64p (v2f64 FPR128:$Rn)))>;
    // vector_extract on 64-bit vectors gets promoted to a 128 bit vector,
    // so we match on v4f32 here, not v2f32. This will also catch adding
    // the low two lanes of a true v4f32 vector.
def : Pat<(fadd (vector_extract (v4f32 FPR128:$Rn), (i64 0)),
                (vector_extract (v4f32 FPR128:$Rn), (i64 1))),
          (f32 (FADDPv2i32p (EXTRACT_SUBREG FPR128:$Rn, dsub)))>;

// Scalar 64-bit shifts in FPR64 registers.
def : Pat<(i64 (int_arm64_neon_sshl (i64 FPR64:$Rn), (i64 FPR64:$Rm))),
          (SSHLv1i64 FPR64:$Rn, FPR64:$Rm)>;
def : Pat<(i64 (int_arm64_neon_ushl (i64 FPR64:$Rn), (i64 FPR64:$Rm))),
          (USHLv1i64 FPR64:$Rn, FPR64:$Rm)>;
def : Pat<(i64 (int_arm64_neon_srshl (i64 FPR64:$Rn), (i64 FPR64:$Rm))),
          (SRSHLv1i64 FPR64:$Rn, FPR64:$Rm)>;
def : Pat<(i64 (int_arm64_neon_urshl (i64 FPR64:$Rn), (i64 FPR64:$Rm))),
          (URSHLv1i64 FPR64:$Rn, FPR64:$Rm)>;

// Tail call return handling. These are all compiler pseudo-instructions,
// so no encoding information or anything like that.
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [SP] in {
  def TCRETURNdi : Pseudo<(outs), (ins i64imm:$dst), []>;
  def TCRETURNri : Pseudo<(outs), (ins tcGPR64:$dst), []>;
}

def : Pat<(ARM64tcret tcGPR64:$dst), (TCRETURNri tcGPR64:$dst)>;
def : Pat<(ARM64tcret (i64 tglobaladdr:$dst)), (TCRETURNdi texternalsym:$dst)>;
def : Pat<(ARM64tcret (i64 texternalsym:$dst)), (TCRETURNdi texternalsym:$dst)>;

include "ARM64InstrAtomics.td"
