//=- HexagonInstrInfoV4.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V4 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

let neverHasSideEffects = 1 in
class T_Immext<dag ins> :
  EXTENDERInst<(outs), ins, "immext(#$imm)", []>,
  Requires<[HasV4T]>;

def IMMEXT_b : T_Immext<(ins brtarget:$imm)>;
def IMMEXT_c : T_Immext<(ins calltarget:$imm)>;
def IMMEXT_g : T_Immext<(ins globaladdress:$imm)>;
def IMMEXT_i : T_Immext<(ins u26_6Imm:$imm)>;

// Fold (add (CONST32 tglobaladdr:$addr) <offset>) into a global address.
def FoldGlobalAddr : ComplexPattern<i32, 1, "foldGlobalAddress", [], []>;

// Fold (add (CONST32_GP tglobaladdr:$addr) <offset>) into a global address.
def FoldGlobalAddrGP : ComplexPattern<i32, 1, "foldGlobalAddressGP", [], []>;

def NumUsesBelowThresCONST32 : PatFrag<(ops node:$addr),
                                       (HexagonCONST32 node:$addr), [{
  return hasNumUsesBelowThresGA(N->getOperand(0).getNode());
}]>;

// Hexagon V4 Architecture spec defines 8 instruction classes:
// LD ST ALU32 XTYPE J JR MEMOP NV CR SYSTEM(system is not implemented in the
// compiler)

// LD Instructions:
// ========================================
// Loads (8/16/32/64 bit)
// Deallocframe

// ST Instructions:
// ========================================
// Stores (8/16/32/64 bit)
// Allocframe

// ALU32 Instructions:
// ========================================
// Arithmetic / Logical (32 bit)
// Vector Halfword

// XTYPE Instructions (32/64 bit):
// ========================================
// Arithmetic, Logical, Bit Manipulation
// Multiply (Integer, Fractional, Complex)
// Permute / Vector Permute Operations
// Predicate Operations
// Shift / Shift with Add/Sub/Logical
// Vector Byte ALU
// Vector Halfword (ALU, Shift, Multiply)
// Vector Word (ALU, Shift)

// J Instructions:
// ========================================
// Jump/Call PC-relative

// JR Instructions:
// ========================================
// Jump/Call Register

// MEMOP Instructions:
// ========================================
// Operation on memory (8/16/32 bit)

// NV Instructions:
// ========================================
// New-value Jumps
// New-value Stores

// CR Instructions:
// ========================================
// Control-Register Transfers
// Hardware Loop Setup
// Predicate Logicals & Reductions

// SYSTEM Instructions (not implemented in the compiler):
// ========================================
// Prefetch
// Cache Maintenance
// Bus Operations


//===----------------------------------------------------------------------===//
// ALU32 +
//===----------------------------------------------------------------------===//

// Shift halfword.

let isPredicated = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in {
def ASLH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASLH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASLH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASLH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = aslh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASRH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASRH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASRH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;

def ASRH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = asrh($src2)",
            []>,
            Requires<[HasV4T]>;
}

// Sign extend.

let isPredicated = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in {
def SXTB_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def SXTB_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def SXTB_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def SXTB_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = sxtb($src2)",
            []>,
            Requires<[HasV4T]>;


def SXTH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

def SXTH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

def SXTH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;

def SXTH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = sxth($src2)",
            []>,
            Requires<[HasV4T]>;
}

// Zero exten.

let neverHasSideEffects = 1, isPredicated = 1, validSubTargets = HasV4SubT in {
def ZXTB_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTB_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTB_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTB_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

def ZXTH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;
}

// Generate frame index addresses.
let neverHasSideEffects = 1, isReMaterializable = 1,
isExtended = 1, opExtendable = 2, validSubTargets = HasV4SubT in
def TFR_FI_immext_V4 : ALU32_ri<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s32Imm:$offset),
            "$dst = add($src1, ##$offset)",
            []>,
            Requires<[HasV4T]>;

// Rd=cmp.eq(Rs,#s8)
let validSubTargets = HasV4SubT, isExtendable = 1, opExtendable = 2,
isExtentSigned = 1, opExtentBits = 8 in
def V4_A4_rcmpeqi : ALU32_ri<(outs IntRegs:$Rd),
                    (ins IntRegs:$Rs, s8Ext:$s8),
                    "$Rd = cmp.eq($Rs, #$s8)",
                    [(set (i32 IntRegs:$Rd),
                          (i32 (zext (i1 (seteq (i32 IntRegs:$Rs),
                                                s8ExtPred:$s8)))))]>,
                    Requires<[HasV4T]>;

// Preserve the TSTBIT generation
def : Pat <(i32 (zext (i1 (setne (i32 (and (i32 (shl 1, (i32 IntRegs:$src2))),
                                           (i32 IntRegs:$src1))), 0)))),
      (i32 (MUX_ii (i1 (TSTBIT_rr (i32 IntRegs:$src1), (i32 IntRegs:$src2))),
                   1, 0))>;

// Interfered with tstbit generation, above pattern preserves, see : tstbit.ll
// Rd=cmp.ne(Rs,#s8)
let validSubTargets = HasV4SubT, isExtendable = 1, opExtendable = 2,
isExtentSigned = 1, opExtentBits = 8 in
def V4_A4_rcmpneqi : ALU32_ri<(outs IntRegs:$Rd),
                     (ins IntRegs:$Rs, s8Ext:$s8),
                     "$Rd = !cmp.eq($Rs, #$s8)",
                     [(set (i32 IntRegs:$Rd),
                           (i32 (zext (i1 (setne (i32 IntRegs:$Rs),
                                                 s8ExtPred:$s8)))))]>,
                     Requires<[HasV4T]>;

// Rd=cmp.eq(Rs,Rt)
let validSubTargets = HasV4SubT in
def V4_A4_rcmpeq : ALU32_ri<(outs IntRegs:$Rd),
                   (ins IntRegs:$Rs, IntRegs:$Rt),
                   "$Rd = cmp.eq($Rs, $Rt)",
                   [(set (i32 IntRegs:$Rd),
                         (i32 (zext (i1 (seteq (i32 IntRegs:$Rs),
                                               IntRegs:$Rt)))))]>,
                   Requires<[HasV4T]>;

// Rd=cmp.ne(Rs,Rt)
let validSubTargets = HasV4SubT in
def V4_A4_rcmpneq : ALU32_ri<(outs IntRegs:$Rd),
                    (ins IntRegs:$Rs, IntRegs:$Rt),
                    "$Rd = !cmp.eq($Rs, $Rt)",
                    [(set (i32 IntRegs:$Rd),
                          (i32 (zext (i1 (setne (i32 IntRegs:$Rs),
                                               IntRegs:$Rt)))))]>,
                    Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// ALU32 -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//

// Combine
// Rdd=combine(Rs, #s8)
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 8,
    neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def COMBINE_rI_V4 : ALU32_ri<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, s8Ext:$src2),
            "$dst = combine($src1, #$src2)",
            []>,
            Requires<[HasV4T]>;

// Rdd=combine(#s8, Rs)
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 8,
    neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def COMBINE_Ir_V4 : ALU32_ir<(outs DoubleRegs:$dst),
            (ins s8Ext:$src1, IntRegs:$src2),
            "$dst = combine(#$src1, $src2)",
            []>,
            Requires<[HasV4T]>;

def HexagonWrapperCombineRI_V4 :
  SDNode<"HexagonISD::WrapperCombineRI_V4", SDTHexagonI64I32I32>;
def HexagonWrapperCombineIR_V4 :
  SDNode<"HexagonISD::WrapperCombineIR_V4", SDTHexagonI64I32I32>;

def : Pat <(HexagonWrapperCombineRI_V4 IntRegs:$r, s8ExtPred:$i),
           (COMBINE_rI_V4 IntRegs:$r, s8ExtPred:$i)>,
          Requires<[HasV4T]>;

def : Pat <(HexagonWrapperCombineIR_V4 s8ExtPred:$i, IntRegs:$r),
           (COMBINE_Ir_V4 s8ExtPred:$i, IntRegs:$r)>,
          Requires<[HasV4T]>;

let isExtendable = 1, opExtendable = 2, isExtentSigned = 0, opExtentBits = 6,
    neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def COMBINE_iI_V4 : ALU32_ii<(outs DoubleRegs:$dst),
            (ins s8Imm:$src1, u6Ext:$src2),
            "$dst = combine(#$src1, #$src2)",
            []>,
            Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LD +
//===----------------------------------------------------------------------===//
//
// These absolute set addressing mode instructions accept immediate as
// an operand. We have duplicated these patterns to take global address.

let isExtended = 1, opExtendable = 2, neverHasSideEffects = 1,
validSubTargets = HasV4SubT in {
def LDrid_abs_setimm_V4 : LDInst2<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = memd($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memb(Re=#U6)
def LDrib_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = memb($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memh(Re=#U6)
def LDrih_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = memh($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memub(Re=#U6)
def LDriub_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = memub($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memuh(Re=#U6)
def LDriuh_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = memuh($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memw(Re=#U6)
def LDriw_abs_setimm_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = memw($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;
}

// Following patterns are defined for absolute set addressing mode
// instruction which take global address as operand.
let isExtended = 1, opExtendable = 2, neverHasSideEffects = 1,
validSubTargets = HasV4SubT in {
def LDrid_abs_set_V4 : LDInst2<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins globaladdressExt:$addr),
            "$dst1 = memd($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memb(Re=#U6)
def LDrib_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdressExt:$addr),
            "$dst1 = memb($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memh(Re=#U6)
def LDrih_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdressExt:$addr),
            "$dst1 = memh($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memub(Re=#U6)
def LDriub_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdressExt:$addr),
            "$dst1 = memub($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memuh(Re=#U6)
def LDriuh_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdressExt:$addr),
            "$dst1 = memuh($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

// Rd=memw(Re=#U6)
def LDriw_abs_set_V4 : LDInst2<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins globaladdressExt:$addr),
            "$dst1 = memw($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;
}

// multiclass for load instructions with base + register offset
// addressing mode
multiclass ld_idxd_shl_pbase<string mnemonic, RegisterClass RC, bit isNot,
                             bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME : LDInst2<(outs RC:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$offset),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#"$dst = "#mnemonic#"($src2+$src3<<#$offset)",
            []>, Requires<[HasV4T]>;
}

multiclass ld_idxd_shl_pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ld_idxd_shl_pbase<mnemonic, RC, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ld_idxd_shl_pbase<mnemonic, RC, PredNot, 1>;
  }
}

let neverHasSideEffects  = 1 in
multiclass ld_idxd_shl<string mnemonic, string CextOp, RegisterClass RC> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed_shl in {
    let isPredicable = 1 in
    def NAME#_V4 : LDInst2<(outs RC:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
            "$dst = "#mnemonic#"($src1+$src2<<#$offset)",
            []>, Requires<[HasV4T]>;

    let isPredicated = 1 in {
      defm Pt_V4 : ld_idxd_shl_pred<mnemonic, RC, 0 >;
      defm NotPt_V4 : ld_idxd_shl_pred<mnemonic, RC, 1>;
    }
  }
}

let addrMode = BaseRegOffset in {
  defm LDrib_indexed_shl: ld_idxd_shl<"memb", "LDrib", IntRegs>, AddrModeRel;
  defm LDriub_indexed_shl: ld_idxd_shl<"memub", "LDriub", IntRegs>, AddrModeRel;
  defm LDrih_indexed_shl: ld_idxd_shl<"memh", "LDrih", IntRegs>, AddrModeRel;
  defm LDriuh_indexed_shl: ld_idxd_shl<"memuh", "LDriuh", IntRegs>, AddrModeRel;
  defm LDriw_indexed_shl: ld_idxd_shl<"memw", "LDriw", IntRegs>, AddrModeRel;
  defm LDrid_indexed_shl: ld_idxd_shl<"memd", "LDrid", DoubleRegs>, AddrModeRel;
}

// 'def pats' for load instructions with base + register offset and non-zero
// immediate value. Immediate value is used to left-shift the second
// register operand.
let AddedComplexity = 40 in {
def : Pat <(i32 (sextloadi8 (add IntRegs:$src1,
                                 (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDrib_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (zextloadi8 (add IntRegs:$src1,
                                 (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDriub_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (extloadi8 (add IntRegs:$src1,
                                (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDriub_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (sextloadi16 (add IntRegs:$src1,
                                  (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDrih_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (zextloadi16 (add IntRegs:$src1,
                                  (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDriuh_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (extloadi16 (add IntRegs:$src1,
                                 (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDriuh_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (load (add IntRegs:$src1,
                           (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDriw_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i64 (load (add IntRegs:$src1,
                           (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (LDrid_indexed_shl_V4 IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;
}


// 'def pats' for load instruction base + register offset and
// zero immediate value.
let AddedComplexity = 10 in {
def : Pat <(i64 (load (add IntRegs:$src1, IntRegs:$src2))),
           (LDrid_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (sextloadi8 (add IntRegs:$src1, IntRegs:$src2))),
           (LDrib_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (zextloadi8 (add IntRegs:$src1, IntRegs:$src2))),
           (LDriub_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (extloadi8 (add IntRegs:$src1, IntRegs:$src2))),
           (LDriub_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (sextloadi16 (add IntRegs:$src1, IntRegs:$src2))),
           (LDrih_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (zextloadi16 (add IntRegs:$src1, IntRegs:$src2))),
           (LDriuh_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (extloadi16 (add IntRegs:$src1, IntRegs:$src2))),
           (LDriuh_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (load (add IntRegs:$src1, IntRegs:$src2))),
           (LDriw_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2, 0)>,
            Requires<[HasV4T]>;
}

let isPredicable = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def LDd_GP_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memd(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rtt=memd(##global)
let neverHasSideEffects = 1, isPredicated = 1, isExtended = 1, opExtendable = 2,
validSubTargets = HasV4SubT in {
def LDd_GP_cPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rtt=memd(##global)
def LDd_GP_cNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rtt=memd(##global)
def LDd_GP_cdnPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rtt=memd(##global)
def LDd_GP_cdnNotPt_V4 : LDInst2<(outs DoubleRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memd(##$global)",
            []>,
            Requires<[HasV4T]>;
}

let isPredicable = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def LDb_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memb(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memb(##global)
let neverHasSideEffects = 1, isPredicated = 1, isExtended = 1, opExtendable = 2,
validSubTargets = HasV4SubT in {
def LDb_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memb(##global)
def LDb_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memb(##global)
def LDb_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memb(##global)
def LDb_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memb(##$global)",
            []>,
            Requires<[HasV4T]>;
}

let isPredicable = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def LDub_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memub(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memub(##global)
let neverHasSideEffects = 1, isPredicated = 1, isExtended = 1, opExtendable = 2,
validSubTargets = HasV4SubT in {
def LDub_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memub(##global)
def LDub_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memub(##global)
def LDub_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memub(##global)
def LDub_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memub(##$global)",
            []>,
            Requires<[HasV4T]>;
}

let isPredicable = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def LDh_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memh(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memh(##global)
let neverHasSideEffects = 1, isPredicated = 1, isExtended = 1, opExtendable = 2,
validSubTargets = HasV4SubT in {
def LDh_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memh(##global)
def LDh_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memh(##global)
def LDh_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memh(##global)
def LDh_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memh(##$global)",
            []>,
            Requires<[HasV4T]>;
}

let isPredicable = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def LDuh_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memuh(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memuh(##global)
let neverHasSideEffects = 1, isPredicated = 1, isExtended = 1, opExtendable = 2,
validSubTargets = HasV4SubT in {
def LDuh_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memuh(##global)
def LDuh_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memuh(##global)
def LDuh_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) Rt=memuh(##global)
def LDuh_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memuh(##$global)",
            []>,
            Requires<[HasV4T]>;
}

let isPredicable = 1, neverHasSideEffects = 1, validSubTargets = HasV4SubT in
def LDw_GP_V4 : LDInst2<(outs IntRegs:$dst),
            (ins globaladdress:$global),
            "$dst=memw(#$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memw(##global)
let neverHasSideEffects = 1, isPredicated = 1, isExtended = 1, opExtendable = 2,
validSubTargets = HasV4SubT in {
def LDw_GP_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memw(##global)
def LDw_GP_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;

// if (Pv) Rt=memw(##global)
def LDw_GP_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if ($src1.new) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;


// if (!Pv) Rt=memw(##global)
def LDw_GP_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, globaladdress:$global),
            "if (!$src1.new) $dst=memw(##$global)",
            []>,
            Requires<[HasV4T]>;
}


def : Pat <(atomic_load_64 (HexagonCONST32_GP tglobaladdr:$global)),
           (i64 (LDd_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_32 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDw_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_16 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDuh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

def : Pat <(atomic_load_8 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memw(#foo + 0)
let AddedComplexity = 100 in
def : Pat <(i64 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i64 (LDd_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from Pd = load(globaladdress) -> Rd = memb(globaladdress), Pd = Rd
let AddedComplexity = 100 in
def : Pat <(i1 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i1 (TFR_PdRs (i32 (LDb_GP_V4 tglobaladdr:$global))))>,
           Requires<[HasV4T]>;

// When the Interprocedural Global Variable optimizer realizes that a certain
// global variable takes only two constant values, it shrinks the global to
// a boolean. Catch those loads here in the following 3 patterns.
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memub(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memuh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDuh_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDw_GP_V4 tglobaladdr:$global))>,
            Requires<[HasV4T]>;

// zext i1->i64
def : Pat <(i64 (zext (i1 PredRegs:$src1))),
      (i64 (COMBINE_Ir_V4 0, (MUX_ii (i1 PredRegs:$src1), 1, 0)))>,
      Requires<[HasV4T]>;

// zext i32->i64
def : Pat <(i64 (zext (i32 IntRegs:$src1))),
      (i64 (COMBINE_Ir_V4 0, (i32 IntRegs:$src1)))>,
      Requires<[HasV4T]>;
// zext i8->i64
def:  Pat <(i64 (zextloadi8 ADDRriS11_0:$src1)),
      (i64 (COMBINE_Ir_V4 0, (LDriub ADDRriS11_0:$src1)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (zextloadi8 (add (i32 IntRegs:$src1),
                                s11_0ExtPred:$offset))),
      (i64 (COMBINE_Ir_V4 0, (LDriub_indexed IntRegs:$src1,
                                  s11_0ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// zext i16->i64
def:  Pat <(i64 (zextloadi16 ADDRriS11_1:$src1)),
      (i64 (COMBINE_Ir_V4 0, (LDriuh ADDRriS11_1:$src1)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (zextloadi16 (add (i32 IntRegs:$src1),
                                  s11_1ExtPred:$offset))),
      (i64 (COMBINE_Ir_V4 0, (LDriuh_indexed IntRegs:$src1,
                                  s11_1ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// anyext i16->i64
def:  Pat <(i64 (extloadi16 ADDRriS11_2:$src1)),
      (i64 (COMBINE_Ir_V4 0, (LDrih ADDRriS11_2:$src1)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (extloadi16 (add (i32 IntRegs:$src1),
                                  s11_1ExtPred:$offset))),
      (i64 (COMBINE_Ir_V4 0, (LDrih_indexed IntRegs:$src1,
                                  s11_1ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// zext i32->i64
def:  Pat <(i64 (zextloadi32 ADDRriS11_2:$src1)),
      (i64 (COMBINE_Ir_V4 0, (LDriw ADDRriS11_2:$src1)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 100 in
def:  Pat <(i64 (zextloadi32 (i32 (add IntRegs:$src1, s11_2ExtPred:$offset)))),
      (i64 (COMBINE_Ir_V4 0, (LDriw_indexed IntRegs:$src1,
                                  s11_2ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// anyext i32->i64
def:  Pat <(i64 (extloadi32 ADDRriS11_2:$src1)),
      (i64 (COMBINE_Ir_V4 0, (LDriw ADDRriS11_2:$src1)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 100 in
def:  Pat <(i64 (extloadi32 (i32 (add IntRegs:$src1, s11_2ExtPred:$offset)))),
      (i64 (COMBINE_Ir_V4 0, (LDriw_indexed IntRegs:$src1,
                                  s11_2ExtPred:$offset)))>,
      Requires<[HasV4T]>;



//===----------------------------------------------------------------------===//
// LD -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ST +
//===----------------------------------------------------------------------===//
///
/// Assumptions::: ****** DO NOT IGNORE ********
/// 1. Make sure that in post increment store, the zero'th operand is always the
///    post increment operand.
/// 2. Make sure that the store value operand(Rt/Rtt) in a store is always the
///    last operand.
///

// memd(Re=#U)=Rtt
let isExtended = 1, opExtendable = 2, validSubTargets = HasV4SubT in {
def STrid_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins DoubleRegs:$src1, u0AlwaysExt:$src2),
            "memd($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memb(Re=#U)=Rs
def STrib_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, u0AlwaysExt:$src2),
            "memb($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memh(Re=#U)=Rs
def STrih_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, u0AlwaysExt:$src2),
            "memh($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memw(Re=#U)=Rs
def STriw_abs_setimm_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, u0AlwaysExt:$src2),
            "memw($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;
}

// memd(Re=#U)=Rtt
let isExtended = 1, opExtendable = 2, validSubTargets = HasV4SubT in {
def STrid_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins DoubleRegs:$src1, globaladdressExt:$src2),
            "memd($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memb(Re=#U)=Rs
def STrib_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, globaladdressExt:$src2),
            "memb($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memh(Re=#U)=Rs
def STrih_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, globaladdressExt:$src2),
            "memh($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

// memw(Re=#U)=Rs
def STriw_abs_set_V4 : STInst2<(outs IntRegs:$dst1),
            (ins IntRegs:$src1, globaladdressExt:$src2),
            "memw($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;
}

// multiclass for store instructions with base + register offset addressing
// mode
multiclass ST_Idxd_shl_Pbase<string mnemonic, RegisterClass RC, bit isNot,
                             bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 RC:$src5),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Idxd_shl_Pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_Idxd_shl_Pbase<mnemonic, RC, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ST_Idxd_shl_Pbase<mnemonic, RC, PredNot, 1>;
  }
}

let isNVStorable = 1 in
multiclass ST_Idxd_shl<string mnemonic, string CextOp, RegisterClass RC> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed_shl in {
    let isPredicable = 1 in
    def NAME#_V4 : STInst2<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, RC:$src4),
            mnemonic#"($src1+$src2<<#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

    let isPredicated = 1 in {
      defm Pt_V4 : ST_Idxd_shl_Pred<mnemonic, RC, 0 >;
      defm NotPt_V4 : ST_Idxd_shl_Pred<mnemonic, RC, 1>;
    }
  }
}

// multiclass for new-value store instructions with base + register offset
// addressing mode.
multiclass ST_Idxd_shl_Pbase_nv<string mnemonic, RegisterClass RC, bit isNot,
                             bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 RC:$src5),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Idxd_shl_Pred_nv<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_Idxd_shl_Pbase_nv<mnemonic, RC, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ST_Idxd_shl_Pbase_nv<mnemonic, RC, PredNot, 1>;
  }
}

let mayStore = 1, isNVStore = 1 in
multiclass ST_Idxd_shl_nv<string mnemonic, string CextOp, RegisterClass RC> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed_shl in {
    let isPredicable = 1 in
    def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, RC:$src4),
            mnemonic#"($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

    let isPredicated = 1 in {
      defm Pt : ST_Idxd_shl_Pred_nv<mnemonic, RC, 0 >;
      defm NotPt : ST_Idxd_shl_Pred_nv<mnemonic, RC, 1>;
    }
  }
}

let addrMode = BaseRegOffset, neverHasSideEffects = 1,
validSubTargets = HasV4SubT in {
  defm STrib_indexed_shl: ST_Idxd_shl<"memb", "STrib", IntRegs>,
                          ST_Idxd_shl_nv<"memb", "STrib", IntRegs>, AddrModeRel;

  defm STrih_indexed_shl: ST_Idxd_shl<"memh", "STrih", IntRegs>,
                          ST_Idxd_shl_nv<"memh", "STrih", IntRegs>, AddrModeRel;

  defm STriw_indexed_shl: ST_Idxd_shl<"memw", "STriw", IntRegs>,
                          ST_Idxd_shl_nv<"memw", "STriw", IntRegs>, AddrModeRel;

  let isNVStorable = 0 in
  defm STrid_indexed_shl: ST_Idxd_shl<"memd", "STrid", DoubleRegs>, AddrModeRel;
}

let Predicates = [HasV4T], AddedComplexity = 10 in {
def : Pat<(truncstorei8 (i32 IntRegs:$src4),
                       (add IntRegs:$src1, (shl IntRegs:$src2,
                                                u2ImmPred:$src3))),
          (STrib_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, IntRegs:$src4)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src4),
                        (add IntRegs:$src1, (shl IntRegs:$src2,
                                                 u2ImmPred:$src3))),
          (STrih_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, IntRegs:$src4)>;

def : Pat<(store (i32 IntRegs:$src4),
                 (add IntRegs:$src1, (shl IntRegs:$src2, u2ImmPred:$src3))),
          (STriw_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, IntRegs:$src4)>;

def : Pat<(store (i64 DoubleRegs:$src4),
                (add IntRegs:$src1, (shl IntRegs:$src2, u2ImmPred:$src3))),
          (STrid_indexed_shl_V4 IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, DoubleRegs:$src4)>;
}

// memd(Ru<<#u2+#U6)=Rtt
let isExtended = 1, opExtendable = 2, AddedComplexity = 10,
validSubTargets = HasV4SubT in
def STrid_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, DoubleRegs:$src4),
            "memd($src1<<#$src2+#$src3) = $src4",
            [(store (i64 DoubleRegs:$src4),
                    (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                         u0AlwaysExtPred:$src3))]>,
            Requires<[HasV4T]>;

// memd(Rx++#s4:3)=Rtt
// memd(Rx++#s4:3:circ(Mu))=Rtt
// memd(Rx++I:circ(Mu))=Rtt
// memd(Rx++Mu)=Rtt
// memd(Rx++Mu:brev)=Rtt
// memd(gp+#u16:3)=Rtt

// Store doubleword conditionally.
// if ([!]Pv[.new]) memd(#u6)=Rtt
// TODO: needs to be implemented.

//===----------------------------------------------------------------------===//
// multiclass for store instructions with base + immediate offset
// addressing mode and immediate stored value.
// mem[bhw](Rx++#s4:3)=#s8
// if ([!]Pv[.new]) mem[bhw](Rx++#s4:3)=#s6
//===----------------------------------------------------------------------===//
multiclass ST_Imm_Pbase<string mnemonic, Operand OffsetOp, bit isNot,
                        bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, OffsetOp:$src3, s6Ext:$src4),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Imm_Pred<string mnemonic, Operand OffsetOp, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_Imm_Pbase<mnemonic, OffsetOp, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ST_Imm_Pbase<mnemonic, OffsetOp, PredNot, 1>;
  }
}

let isExtendable = 1, isExtentSigned = 1, neverHasSideEffects = 1 in
multiclass ST_Imm<string mnemonic, string CextOp, Operand OffsetOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_imm in {
    let opExtendable = 2, opExtentBits = 8, isPredicable = 1 in
    def NAME#_V4 : STInst2<(outs),
            (ins IntRegs:$src1, OffsetOp:$src2, s8Ext:$src3),
            mnemonic#"($src1+#$src2) = #$src3",
            []>,
            Requires<[HasV4T]>;

    let opExtendable = 3, opExtentBits = 6, isPredicated = 1 in {
      defm Pt_V4 : ST_Imm_Pred<mnemonic, OffsetOp, 0>;
      defm NotPt_V4 : ST_Imm_Pred<mnemonic, OffsetOp, 1 >;
    }
  }
}

let addrMode = BaseImmOffset, InputType = "imm",
    validSubTargets = HasV4SubT in {
  defm STrib_imm : ST_Imm<"memb", "STrib", u6_0Imm>, ImmRegRel, PredNewRel;
  defm STrih_imm : ST_Imm<"memh", "STrih", u6_1Imm>, ImmRegRel, PredNewRel;
  defm STriw_imm : ST_Imm<"memw", "STriw", u6_2Imm>, ImmRegRel, PredNewRel;
}

let Predicates = [HasV4T], AddedComplexity = 10 in {
def: Pat<(truncstorei8 s8ExtPred:$src3, (add IntRegs:$src1, u6_0ImmPred:$src2)),
            (STrib_imm_V4 IntRegs:$src1, u6_0ImmPred:$src2, s8ExtPred:$src3)>;

def: Pat<(truncstorei16 s8ExtPred:$src3, (add IntRegs:$src1,
                                              u6_1ImmPred:$src2)),
            (STrih_imm_V4 IntRegs:$src1, u6_1ImmPred:$src2, s8ExtPred:$src3)>;

def: Pat<(store s8ExtPred:$src3, (add IntRegs:$src1, u6_2ImmPred:$src2)),
            (STriw_imm_V4 IntRegs:$src1, u6_2ImmPred:$src2, s8ExtPred:$src3)>;
}

let AddedComplexity = 6 in
def : Pat <(truncstorei8 s8ExtPred:$src2, (i32 IntRegs:$src1)),
           (STrib_imm_V4 IntRegs:$src1, 0, s8ExtPred:$src2)>,
           Requires<[HasV4T]>;

// memb(Ru<<#u2+#U6)=Rt
let isExtended = 1, opExtendable = 2, AddedComplexity = 10, isNVStorable = 1,
validSubTargets = HasV4SubT in
def STrib_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            "memb($src1<<#$src2+#$src3) = $src4",
            [(truncstorei8 (i32 IntRegs:$src4),
                           (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                                u0AlwaysExtPred:$src3))]>,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0:circ(Mu))=Rt
// memb(Rx++I:circ(Mu))=Rt
// memb(Rx++Mu)=Rt
// memb(Rx++Mu:brev)=Rt
// memb(gp+#u16:0)=Rt


// Store halfword.
// TODO: needs to be implemented
// memh(Re=#U6)=Rt.H
// memh(Rs+#s11:1)=Rt.H
let AddedComplexity = 6 in
def : Pat <(truncstorei16 s8ExtPred:$src2, (i32 IntRegs:$src1)),
           (STrih_imm_V4 IntRegs:$src1, 0, s8ExtPred:$src2)>,
           Requires<[HasV4T]>;

// memh(Rs+Ru<<#u2)=Rt.H
// TODO: needs to be implemented.

// memh(Ru<<#u2+#U6)=Rt.H
// memh(Ru<<#u2+#U6)=Rt
let isExtended = 1, opExtendable = 2, AddedComplexity = 10, isNVStorable = 1,
validSubTargets = HasV4SubT in
def STrih_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            "memh($src1<<#$src2+#$src3) = $src4",
            [(truncstorei16 (i32 IntRegs:$src4),
                            (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                                 u0AlwaysExtPred:$src3))]>,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1:circ(Mu))=Rt.H
// memh(Rx++#s4:1:circ(Mu))=Rt
// memh(Rx++I:circ(Mu))=Rt.H
// memh(Rx++I:circ(Mu))=Rt
// memh(Rx++Mu)=Rt.H
// memh(Rx++Mu)=Rt
// memh(Rx++Mu:brev)=Rt.H
// memh(Rx++Mu:brev)=Rt
// memh(gp+#u16:1)=Rt
// if ([!]Pv[.new]) memh(#u6)=Rt.H
// if ([!]Pv[.new]) memh(#u6)=Rt


// if ([!]Pv[.new]) memh(Rs+#u6:1)=Rt.H
// TODO: needs to be implemented.

// if ([!]Pv[.new]) memh(Rx++#s4:1)=Rt.H
// TODO: Needs to be implemented.

// Store word.
// memw(Re=#U6)=Rt
// TODO: Needs to be implemented.

// Store predicate:
let neverHasSideEffects = 1 in
def STriw_pred_V4 : STInst2<(outs),
            (ins MEMri:$addr, PredRegs:$src1),
            "Error; should not emit",
            []>,
            Requires<[HasV4T]>;

let AddedComplexity = 6 in
def : Pat <(store s8ExtPred:$src2, (i32 IntRegs:$src1)),
           (STriw_imm_V4 IntRegs:$src1, 0, s8ExtPred:$src2)>,
           Requires<[HasV4T]>;

// memw(Ru<<#u2+#U6)=Rt
let isExtended = 1, opExtendable = 2, AddedComplexity = 10, isNVStorable = 1,
validSubTargets = HasV4SubT in
def STriw_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            "memw($src1<<#$src2+#$src3) = $src4",
            [(store (i32 IntRegs:$src4),
                    (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                              u0AlwaysExtPred:$src3))]>,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2)=Rt
// memw(Rx++#s4:2:circ(Mu))=Rt
// memw(Rx++I:circ(Mu))=Rt
// memw(Rx++Mu)=Rt
// memw(Rx++Mu:brev)=Rt
// memw(gp+#u16:2)=Rt


// memd(#global)=Rtt
let isPredicable = 1, mayStore = 1, neverHasSideEffects = 1,
validSubTargets = HasV4SubT in
def STd_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, DoubleRegs:$src),
            "memd(#$global) = $src",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memd(##global) = Rtt
let mayStore = 1, neverHasSideEffects = 1, isPredicated = 1,
isExtended = 1, opExtendable = 1, validSubTargets = HasV4SubT in {
def STd_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if ($src1) memd(##$global) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memd(##global) = Rtt
def STd_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if (!$src1) memd(##$global) = $src2",
            []>,
              Requires<[HasV4T]>;

// if (Pv) memd(##global) = Rtt
def STd_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if ($src1.new) memd(##$global) = $src2",
            []>,
              Requires<[HasV4T]>;

// if (!Pv) memd(##global) = Rtt
def STd_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, DoubleRegs:$src2),
            "if (!$src1.new) memd(##$global) = $src2",
            []>,
            Requires<[HasV4T]>;
}

// memb(#global)=Rt
let isPredicable = 1, neverHasSideEffects = 1, isNVStorable = 1,
validSubTargets = HasV4SubT in
def STb_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memb(#$global) = $src",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1, isNVStorable = 1,
isExtended = 1, opExtendable = 1, validSubTargets = HasV4SubT in {
def STb_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
def STb_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
def STb_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
def STb_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memb(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;
}

// memh(#global)=Rt
let isPredicable = 1, neverHasSideEffects = 1, isNVStorable = 1,
validSubTargets = HasV4SubT in
def STh_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memh(#$global) = $src",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1, isNVStorable = 1,
isExtended = 1, opExtendable = 1, validSubTargets = HasV4SubT in {
def STh_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
def STh_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
def STh_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
def STh_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memh(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;
}

// memw(#global)=Rt
let isPredicable = 1, neverHasSideEffects = 1, isNVStorable = 1,
validSubTargets = HasV4SubT in
def STw_GP_V4 : STInst2<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memw(#$global) = $src",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
let neverHasSideEffects = 1, isPredicated = 1, isNVStorable = 1,
isExtended = 1, opExtendable = 1, validSubTargets = HasV4SubT in {
def STw_GP_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memw(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
def STw_GP_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memw(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
def STw_GP_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memw(##$global) = $src2",
              []>,
              Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
def STw_GP_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memw(##$global) = $src2",
            []>,
              Requires<[HasV4T]>;
}

// 64 bit atomic store
def : Pat <(atomic_store_64 (HexagonCONST32_GP tglobaladdr:$global),
                            (i64 DoubleRegs:$src1)),
           (STd_GP_V4 tglobaladdr:$global, (i64 DoubleRegs:$src1))>,
           Requires<[HasV4T]>;

// Map from store(globaladdress) -> memd(#foo)
let AddedComplexity = 100 in
def : Pat <(store (i64 DoubleRegs:$src1),
                  (HexagonCONST32_GP tglobaladdr:$global)),
           (STd_GP_V4 tglobaladdr:$global, (i64 DoubleRegs:$src1))>,
           Requires<[HasV4T]>;

// 8 bit atomic store
def : Pat < (atomic_store_8 (HexagonCONST32_GP tglobaladdr:$global),
                            (i32 IntRegs:$src1)),
            (STb_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
              Requires<[HasV4T]>;

// Map from store(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
          (HexagonCONST32_GP tglobaladdr:$global)),
          (STb_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from "i1 = constant<-1>; memw(CONST32(#foo)) = i1"
//       to "r0 = 1; memw(#foo) = r0"
let AddedComplexity = 100 in
def : Pat<(store (i1 -1), (HexagonCONST32_GP tglobaladdr:$global)),
          (STb_GP_V4 tglobaladdr:$global, (TFRI 1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_16 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (STh_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                         (HexagonCONST32_GP tglobaladdr:$global)),
          (STh_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// 32 bit atomic store
def : Pat<(atomic_store_32 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (STw_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32_GP tglobaladdr:$global)),
          (STw_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>,
          Requires<[HasV4T]>;

//===----------------------------------------------------------------------===
// ST -
//===----------------------------------------------------------------------===


//===----------------------------------------------------------------------===//
// NV/ST +
//===----------------------------------------------------------------------===//

// multiclass for new-value store instructions with base + immediate offset.
//
multiclass ST_Idxd_Pbase_nv<string mnemonic, RegisterClass RC,
                            Operand predImmOp, bit isNot, bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, predImmOp:$src3, RC: $src4),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Idxd_Pred_nv<string mnemonic, RegisterClass RC, Operand predImmOp,
                           bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_Idxd_Pbase_nv<mnemonic, RC, predImmOp, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ST_Idxd_Pbase_nv<mnemonic, RC, predImmOp, PredNot, 1>;
  }
}

let mayStore = 1, isNVStore = 1, neverHasSideEffects = 1, isExtendable = 1 in
multiclass ST_Idxd_nv<string mnemonic, string CextOp, RegisterClass RC,
                   Operand ImmOp, Operand predImmOp, bits<5> ImmBits,
                   bits<5> PredImmBits> {

  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed in {
    let opExtendable = 1, isExtentSigned = 1, opExtentBits = ImmBits,
    isPredicable = 1 in
    def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
            mnemonic#"($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

    let opExtendable = 2, isExtentSigned = 0, opExtentBits = PredImmBits,
    isPredicated = 1 in {
      defm Pt : ST_Idxd_Pred_nv<mnemonic, RC, predImmOp, 0>;
      defm NotPt : ST_Idxd_Pred_nv<mnemonic, RC, predImmOp, 1>;
    }
  }
}

let addrMode = BaseImmOffset, validSubTargets = HasV4SubT in {
  defm STrib_indexed: ST_Idxd_nv<"memb", "STrib", IntRegs, s11_0Ext,
                                 u6_0Ext, 11, 6>, AddrModeRel;
  defm STrih_indexed: ST_Idxd_nv<"memh", "STrih", IntRegs, s11_1Ext,
                                 u6_1Ext, 12, 7>, AddrModeRel;
  defm STriw_indexed: ST_Idxd_nv<"memw", "STriw", IntRegs, s11_2Ext,
                                 u6_2Ext, 13, 8>, AddrModeRel;
}

// multiclass for new-value store instructions with base + immediate offset.
// and MEMri operand.
multiclass ST_MEMri_Pbase_nv<string mnemonic, RegisterClass RC, bit isNot,
                          bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, RC: $src2),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_MEMri_Pred_nv<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_MEMri_Pbase_nv<mnemonic, RC, PredNot, 0>;

    // Predicate new
    defm _cdn#NAME : ST_MEMri_Pbase_nv<mnemonic, RC, PredNot, 1>;
  }
}

let mayStore = 1, isNVStore = 1, isExtendable = 1, neverHasSideEffects = 1 in
multiclass ST_MEMri_nv<string mnemonic, string CextOp, RegisterClass RC,
                    bits<5> ImmBits, bits<5> PredImmBits> {

  let CextOpcode = CextOp, BaseOpcode = CextOp in {
    let opExtendable = 1, isExtentSigned = 1, opExtentBits = ImmBits,
         isPredicable = 1 in
    def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins MEMri:$addr, RC:$src),
            mnemonic#"($addr) = $src.new",
            []>,
            Requires<[HasV4T]>;

    let opExtendable = 2, isExtentSigned = 0, opExtentBits = PredImmBits,
        neverHasSideEffects = 1, isPredicated = 1 in {
      defm Pt : ST_MEMri_Pred_nv<mnemonic, RC, 0>;
      defm NotPt : ST_MEMri_Pred_nv<mnemonic, RC, 1>;
    }
  }
}

let addrMode = BaseImmOffset, isMEMri = "true", validSubTargets = HasV4SubT,
mayStore = 1 in {
  defm STrib: ST_MEMri_nv<"memb", "STrib", IntRegs, 11, 6>, AddrModeRel;
  defm STrih: ST_MEMri_nv<"memh", "STrih", IntRegs, 12, 7>, AddrModeRel;
  defm STriw: ST_MEMri_nv<"memw", "STriw", IntRegs, 13, 8>, AddrModeRel;
}

// memb(Ru<<#u2+#U6)=Nt.new
let isExtended = 1, opExtendable = 2, mayStore = 1, AddedComplexity = 10,
isNVStore = 1, validSubTargets = HasV4SubT in
def STrib_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            "memb($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// Post increment store
// mem[bhwd](Rx++#s4:[0123])=Nt.new
//===----------------------------------------------------------------------===//

multiclass ST_PostInc_Pbase_nv<string mnemonic, RegisterClass RC, Operand ImmOp,
                            bit isNot, bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME#_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$offset, RC:$src3),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2++#$offset) = $src3.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;
}

multiclass ST_PostInc_Pred_nv<string mnemonic, RegisterClass RC,
                           Operand ImmOp, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_PostInc_Pbase_nv<mnemonic, RC, ImmOp, PredNot, 0>;
    // Predicate new
    let Predicates = [HasV4T], validSubTargets = HasV4SubT in
    defm _cdn#NAME : ST_PostInc_Pbase_nv<mnemonic, RC, ImmOp, PredNot, 1>;
  }
}

let hasCtrlDep = 1, isNVStore = 1, neverHasSideEffects = 1 in
multiclass ST_PostInc_nv<string mnemonic, string BaseOp, RegisterClass RC,
                      Operand ImmOp> {

  let BaseOpcode = "POST_"#BaseOp in {
    let isPredicable = 1 in
    def NAME#_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
                (ins IntRegs:$src1, ImmOp:$offset, RC:$src2),
                mnemonic#"($src1++#$offset) = $src2.new",
                [],
                "$src1 = $dst">,
                Requires<[HasV4T]>;

    let isPredicated = 1 in {
      defm Pt : ST_PostInc_Pred_nv<mnemonic, RC, ImmOp, 0 >;
      defm NotPt : ST_PostInc_Pred_nv<mnemonic, RC, ImmOp, 1 >;
    }
  }
}

let validSubTargets = HasV4SubT in {
defm POST_STbri: ST_PostInc_nv <"memb", "STrib", IntRegs, s4_0Imm>, AddrModeRel;
defm POST_SThri: ST_PostInc_nv <"memh", "STrih", IntRegs, s4_1Imm>, AddrModeRel;
defm POST_STwri: ST_PostInc_nv <"memw", "STriw", IntRegs, s4_2Imm>, AddrModeRel;
}

// memb(Rx++#s4:0:circ(Mu))=Nt.new
// memb(Rx++I:circ(Mu))=Nt.new
// memb(Rx++Mu)=Nt.new
// memb(Rx++Mu:brev)=Nt.new

// memb(#global)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STb_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memb(#$global) = $src.new",
            []>,
            Requires<[HasV4T]>;

// memh(Ru<<#u2+#U6)=Nt.new
let isExtended = 1, opExtendable = 2, mayStore = 1, AddedComplexity = 10,
isNVStore = 1, validSubTargets = HasV4SubT in
def STrih_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            "memh($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1:circ(Mu))=Nt.new
// memh(Rx++I:circ(Mu))=Nt.new
// memh(Rx++Mu)=Nt.new
// memh(Rx++Mu:brev)=Nt.new

// memh(#global)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STh_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memh(#$global) = $src.new",
            []>,
            Requires<[HasV4T]>;

// memw(Ru<<#u2+#U6)=Nt.new
let isExtended = 1, opExtendable = 2, mayStore = 1, AddedComplexity = 10,
isNVStore = 1, validSubTargets = HasV4SubT in
def STriw_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            "memw($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2:circ(Mu))=Nt.new
// memw(Rx++I:circ(Mu))=Nt.new
// memw(Rx++Mu)=Nt.new
// memw(Rx++Mu:brev)=Nt.new
// memw(gp+#u16:2)=Nt.new

let mayStore = 1, neverHasSideEffects = 1, isNVStore = 1,
validSubTargets = HasV4SubT in
def STw_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, IntRegs:$src),
            "memw(#$global) = $src.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
let mayStore = 1, neverHasSideEffects = 1, isNVStore = 1,
isExtended = 1, opExtendable = 1, validSubTargets = HasV4SubT in {
def STb_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
def STb_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(##global) = Rt
def STb_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(##global) = Rt
def STb_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memb(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
def STh_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
def STh_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(##global) = Rt
def STh_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(##global) = Rt
def STh_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memh(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
def STw_GP_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
def STw_GP_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(##global) = Rt
def STw_GP_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if ($src1.new) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(##global) = Rt
def STw_GP_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdress:$global, IntRegs:$src2),
            "if (!$src1.new) memw(##$global) = $src2.new",
            []>,
            Requires<[HasV4T]>;
}

//===----------------------------------------------------------------------===//
// NV/ST -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NV/J +
//===----------------------------------------------------------------------===//

multiclass NVJ_type_basic_reg<string NotStr, string OpcStr, string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, $src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, $src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_2ndDotNew<string NotStr, string OpcStr,
                                                   string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1, $src2.new)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1, $src2.new)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_imm<string NotStr, string OpcStr, string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_neg<string NotStr, string OpcStr, string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, nOneImm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, nOneImm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

multiclass NVJ_type_basic_tstbit<string NotStr, string OpcStr,
                                                string TakenStr> {
  def _ie_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u1Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;

  def _nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u1Imm:$src2, brtarget:$offset),
            !strconcat("if (", !strconcat(NotStr, !strconcat(OpcStr,
            !strconcat("($src1.new, #$src2)) jump:",
            !strconcat(TakenStr, " $offset"))))),
            []>,
            Requires<[HasV4T]>;
}

// Multiclass for regular dot new of Ist operand register.
multiclass NVJ_type_br_pred_reg<string NotStr, string OpcStr> {
  defm Pt  : NVJ_type_basic_reg<NotStr, OpcStr, "t">;
  defm Pnt : NVJ_type_basic_reg<NotStr, OpcStr, "nt">;
}

// Multiclass for dot new of 2nd operand register.
multiclass NVJ_type_br_pred_2ndDotNew<string NotStr, string OpcStr> {
  defm Pt  : NVJ_type_basic_2ndDotNew<NotStr, OpcStr, "t">;
  defm Pnt : NVJ_type_basic_2ndDotNew<NotStr, OpcStr, "nt">;
}

// Multiclass for 2nd operand immediate, including -1.
multiclass NVJ_type_br_pred_imm<string NotStr, string OpcStr> {
  defm Pt     : NVJ_type_basic_imm<NotStr, OpcStr, "t">;
  defm Pnt    : NVJ_type_basic_imm<NotStr, OpcStr, "nt">;
  defm Ptneg  : NVJ_type_basic_neg<NotStr, OpcStr, "t">;
  defm Pntneg : NVJ_type_basic_neg<NotStr, OpcStr, "nt">;
}

// Multiclass for 2nd operand immediate, excluding -1.
multiclass NVJ_type_br_pred_imm_only<string NotStr, string OpcStr> {
  defm Pt     : NVJ_type_basic_imm<NotStr, OpcStr, "t">;
  defm Pnt    : NVJ_type_basic_imm<NotStr, OpcStr, "nt">;
}

// Multiclass for tstbit, where 2nd operand is always #0.
multiclass NVJ_type_br_pred_tstbit<string NotStr, string OpcStr> {
  defm Pt     : NVJ_type_basic_tstbit<NotStr, OpcStr, "t">;
  defm Pnt    : NVJ_type_basic_tstbit<NotStr, OpcStr, "nt">;
}

// Multiclass for GT.
multiclass NVJ_type_rr_ri<string OpcStr> {
  defm rrNot   : NVJ_type_br_pred_reg<"!", OpcStr>;
  defm rr      : NVJ_type_br_pred_reg<"",  OpcStr>;
  defm rrdnNot : NVJ_type_br_pred_2ndDotNew<"!", OpcStr>;
  defm rrdn    : NVJ_type_br_pred_2ndDotNew<"",  OpcStr>;
  defm riNot   : NVJ_type_br_pred_imm<"!", OpcStr>;
  defm ri      : NVJ_type_br_pred_imm<"",  OpcStr>;
}

// Multiclass for EQ.
multiclass NVJ_type_rr_ri_no_2ndDotNew<string OpcStr> {
  defm rrNot   : NVJ_type_br_pred_reg<"!", OpcStr>;
  defm rr      : NVJ_type_br_pred_reg<"",  OpcStr>;
  defm riNot   : NVJ_type_br_pred_imm<"!", OpcStr>;
  defm ri      : NVJ_type_br_pred_imm<"",  OpcStr>;
}

// Multiclass for GTU.
multiclass NVJ_type_rr_ri_no_nOne<string OpcStr> {
  defm rrNot   : NVJ_type_br_pred_reg<"!", OpcStr>;
  defm rr      : NVJ_type_br_pred_reg<"",  OpcStr>;
  defm rrdnNot : NVJ_type_br_pred_2ndDotNew<"!", OpcStr>;
  defm rrdn    : NVJ_type_br_pred_2ndDotNew<"",  OpcStr>;
  defm riNot   : NVJ_type_br_pred_imm_only<"!", OpcStr>;
  defm ri      : NVJ_type_br_pred_imm_only<"",  OpcStr>;
}

// Multiclass for tstbit.
multiclass NVJ_type_r0<string OpcStr> {
  defm r0Not : NVJ_type_br_pred_tstbit<"!", OpcStr>;
  defm r0    : NVJ_type_br_pred_tstbit<"",  OpcStr>;
 }

// Base Multiclass for New Value Jump.
multiclass NVJ_type {
  defm GT     : NVJ_type_rr_ri<"cmp.gt">;
  defm EQ     : NVJ_type_rr_ri_no_2ndDotNew<"cmp.eq">;
  defm GTU    : NVJ_type_rr_ri_no_nOne<"cmp.gtu">;
  defm TSTBIT : NVJ_type_r0<"tstbit">;
}

let isBranch = 1, isTerminator=1, neverHasSideEffects = 1, Defs = [PC] in {
  defm JMP_ : NVJ_type;
}

//===----------------------------------------------------------------------===//
// NV/J -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/ALU +
//===----------------------------------------------------------------------===//

//  Add and accumulate.
//  Rd=add(Rs,add(Ru,#s6))
let isExtendable = 1, opExtendable = 3, isExtentSigned = 1, opExtentBits = 6,
validSubTargets = HasV4SubT in
def ADDr_ADDri_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, IntRegs:$src2, s6Ext:$src3),
          "$dst = add($src1, add($src2, #$src3))",
          [(set (i32 IntRegs:$dst),
           (add (i32 IntRegs:$src1), (add (i32 IntRegs:$src2),
                                          s6_16ExtPred:$src3)))]>,
          Requires<[HasV4T]>;

//  Rd=add(Rs,sub(#s6,Ru))
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 6,
validSubTargets = HasV4SubT in
def ADDr_SUBri_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, s6Ext:$src2, IntRegs:$src3),
          "$dst = add($src1, sub(#$src2, $src3))",
          [(set (i32 IntRegs:$dst),
           (add (i32 IntRegs:$src1), (sub s6_10ExtPred:$src2,
                                          (i32 IntRegs:$src3))))]>,
          Requires<[HasV4T]>;

// Generates the same instruction as ADDr_SUBri_V4 but matches different
// pattern.
//  Rd=add(Rs,sub(#s6,Ru))
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 6,
validSubTargets = HasV4SubT in
def ADDri_SUBr_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, s6Ext:$src2, IntRegs:$src3),
          "$dst = add($src1, sub(#$src2, $src3))",
          [(set (i32 IntRegs:$dst),
                (sub (add (i32 IntRegs:$src1), s6_10ExtPred:$src2),
                     (i32 IntRegs:$src3)))]>,
          Requires<[HasV4T]>;


//  Add or subtract doublewords with carry.
//TODO:
//  Rdd=add(Rss,Rtt,Px):carry
//TODO:
//  Rdd=sub(Rss,Rtt,Px):carry


//  Logical doublewords.
//  Rdd=and(Rtt,~Rss)
let validSubTargets = HasV4SubT in
def ANDd_NOTd_V4 : MInst<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2),
          "$dst = and($src1, ~$src2)",
          [(set (i64 DoubleRegs:$dst), (and (i64 DoubleRegs:$src1),
                                      (not (i64 DoubleRegs:$src2))))]>,
          Requires<[HasV4T]>;

//  Rdd=or(Rtt,~Rss)
let validSubTargets = HasV4SubT in
def ORd_NOTd_V4 : MInst<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2),
          "$dst = or($src1, ~$src2)",
          [(set (i64 DoubleRegs:$dst),
           (or (i64 DoubleRegs:$src1), (not (i64 DoubleRegs:$src2))))]>,
          Requires<[HasV4T]>;


//  Logical-logical doublewords.
//  Rxx^=xor(Rss,Rtt)
let validSubTargets = HasV4SubT in
def XORd_XORdd: MInst_acc<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2, DoubleRegs:$src3),
          "$dst ^= xor($src2, $src3)",
          [(set (i64 DoubleRegs:$dst),
           (xor (i64 DoubleRegs:$src1), (xor (i64 DoubleRegs:$src2),
                                             (i64 DoubleRegs:$src3))))],
          "$src1 = $dst">,
          Requires<[HasV4T]>;


// Logical-logical words.
// Rx=or(Ru,and(Rx,#s10))
let isExtendable = 1, opExtendable = 3, isExtentSigned = 1, opExtentBits = 10,
validSubTargets = HasV4SubT in
def ORr_ANDri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Ext:$src3),
            "$dst = or($src1, and($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                s10ExtPred:$src3)))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=and(Rs,Rt)
// Rx&=and(Rs,Rt)
let validSubTargets = HasV4SubT in
def ANDr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= and($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                 (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,Rt)
let validSubTargets = HasV4SubT, CextOpcode = "ORr_ANDr", InputType = "reg" in
def ORr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= and($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>, ImmRegRel;

// Rx^=and(Rs,Rt)
let validSubTargets = HasV4SubT in
def XORr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= and($src2, $src3)",
            [(set (i32 IntRegs:$dst),
             (xor (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                            (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=and(Rs,~Rt)
// Rx&=and(Rs,~Rt)
let validSubTargets = HasV4SubT in
def ANDr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= and($src2, ~$src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                 (not (i32 IntRegs:$src3)))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,~Rt)
let validSubTargets = HasV4SubT in
def ORr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= and($src2, ~$src3)",
            [(set (i32 IntRegs:$dst),
             (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                           (not (i32 IntRegs:$src3)))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=and(Rs,~Rt)
let validSubTargets = HasV4SubT in
def XORr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= and($src2, ~$src3)",
            [(set (i32 IntRegs:$dst),
             (xor (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                            (not (i32 IntRegs:$src3)))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=or(Rs,Rt)
// Rx&=or(Rs,Rt)
let validSubTargets = HasV4SubT in
def ANDr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= or($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (or (i32 IntRegs:$src2),
                                                (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=or(Rs,Rt)
let validSubTargets = HasV4SubT, CextOpcode = "ORr_ORr", InputType = "reg" in
def ORr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= or($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (or (i32 IntRegs:$src2),
                                               (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>, ImmRegRel;

// Rx^=or(Rs,Rt)
let validSubTargets = HasV4SubT in
def XORr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= or($src2, $src3)",
            [(set (i32 IntRegs:$dst),
             (xor (i32 IntRegs:$src1), (or (i32 IntRegs:$src2),
                                           (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=xor(Rs,Rt)
// Rx&=xor(Rs,Rt)
let validSubTargets = HasV4SubT in
def ANDr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= xor($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (xor (i32 IntRegs:$src2),
                                                 (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=xor(Rs,Rt)
let validSubTargets = HasV4SubT in
def ORr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= xor($src2, $src3)",
            [(set (i32 IntRegs:$dst),
                  (and (i32 IntRegs:$src1), (xor (i32 IntRegs:$src2),
                                                 (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=xor(Rs,Rt)
let validSubTargets = HasV4SubT in
def XORr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= xor($src2, $src3)",
            [(set (i32 IntRegs:$dst),
             (and (i32 IntRegs:$src1), (xor (i32 IntRegs:$src2),
                                            (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,#s10)
let isExtendable = 1, opExtendable = 3, isExtentSigned = 1, opExtentBits = 10,
validSubTargets = HasV4SubT, CextOpcode = "ORr_ANDr", InputType = "imm" in
def ORr_ANDri2_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Ext:$src3),
            "$dst |= and($src2, #$src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                s10ExtPred:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>, ImmRegRel;

// Rx|=or(Rs,#s10)
let isExtendable = 1, opExtendable = 3, isExtentSigned = 1, opExtentBits = 10,
validSubTargets = HasV4SubT, CextOpcode = "ORr_ORr", InputType = "imm" in
def ORr_ORri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Ext:$src3),
            "$dst |= or($src2, #$src3)",
            [(set (i32 IntRegs:$dst),
                  (or (i32 IntRegs:$src1), (and (i32 IntRegs:$src2),
                                                s10ExtPred:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>, ImmRegRel;


//    Modulo wrap
//        Rd=modwrap(Rs,Rt)
//    Round
//        Rd=cround(Rs,#u5)
//        Rd=cround(Rs,Rt)
//        Rd=round(Rs,#u5)[:sat]
//        Rd=round(Rs,Rt)[:sat]
//    Vector reduce add unsigned halfwords
//        Rd=vraddh(Rss,Rtt)
//    Vector add bytes
//        Rdd=vaddb(Rss,Rtt)
//    Vector conditional negate
//        Rdd=vcnegh(Rss,Rt)
//        Rxx+=vrcnegh(Rss,Rt)
//    Vector maximum bytes
//        Rdd=vmaxb(Rtt,Rss)
//    Vector reduce maximum halfwords
//        Rxx=vrmaxh(Rss,Ru)
//        Rxx=vrmaxuh(Rss,Ru)
//    Vector reduce maximum words
//        Rxx=vrmaxuw(Rss,Ru)
//        Rxx=vrmaxw(Rss,Ru)
//    Vector minimum bytes
//        Rdd=vminb(Rtt,Rss)
//    Vector reduce minimum halfwords
//        Rxx=vrminh(Rss,Ru)
//        Rxx=vrminuh(Rss,Ru)
//    Vector reduce minimum words
//        Rxx=vrminuw(Rss,Ru)
//        Rxx=vrminw(Rss,Ru)
//    Vector subtract bytes
//        Rdd=vsubb(Rss,Rtt)

//===----------------------------------------------------------------------===//
// XTYPE/ALU -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// XTYPE/MPY +
//===----------------------------------------------------------------------===//

// Multiply and user lower result.
// Rd=add(#u6,mpyi(Rs,#U6))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 6,
validSubTargets = HasV4SubT in
def ADDi_MPYri_V4 : MInst<(outs IntRegs:$dst),
            (ins u6Ext:$src1, IntRegs:$src2, u6Imm:$src3),
            "$dst = add(#$src1, mpyi($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (mul (i32 IntRegs:$src2), u6ImmPred:$src3),
                       u6ExtPred:$src1))]>,
            Requires<[HasV4T]>;

// Rd=add(##,mpyi(Rs,#U6))
def : Pat <(add (mul (i32 IntRegs:$src2), u6ImmPred:$src3),
                     (HexagonCONST32 tglobaladdr:$src1)),
           (i32 (ADDi_MPYri_V4 tglobaladdr:$src1, IntRegs:$src2,
                               u6ImmPred:$src3))>;

// Rd=add(#u6,mpyi(Rs,Rt))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 6,
validSubTargets = HasV4SubT, InputType = "imm", CextOpcode = "ADD_MPY" in
def ADDi_MPYrr_V4 : MInst<(outs IntRegs:$dst),
            (ins u6Ext:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst = add(#$src1, mpyi($src2, $src3))",
            [(set (i32 IntRegs:$dst),
                  (add (mul (i32 IntRegs:$src2), (i32 IntRegs:$src3)),
                       u6ExtPred:$src1))]>,
            Requires<[HasV4T]>, ImmRegRel;

// Rd=add(##,mpyi(Rs,Rt))
def : Pat <(add (mul (i32 IntRegs:$src2), (i32 IntRegs:$src3)),
                     (HexagonCONST32 tglobaladdr:$src1)),
           (i32 (ADDi_MPYrr_V4 tglobaladdr:$src1, IntRegs:$src2,
                               IntRegs:$src3))>;

// Rd=add(Ru,mpyi(#u6:2,Rs))
let validSubTargets = HasV4SubT in
def ADDr_MPYir_V4 : MInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            "$dst = add($src1, mpyi(#$src2, $src3))",
            [(set (i32 IntRegs:$dst),
             (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src3),
                                            u6_2ImmPred:$src2)))]>,
            Requires<[HasV4T]>;

// Rd=add(Ru,mpyi(Rs,#u6))
let isExtendable = 1, opExtendable = 3, isExtentSigned = 0, opExtentBits = 6,
validSubTargets = HasV4SubT, InputType = "imm", CextOpcode = "ADD_MPY" in
def ADDr_MPYri_V4 : MInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, u6Ext:$src3),
            "$dst = add($src1, mpyi($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src2),
                                                 u6ExtPred:$src3)))]>,
            Requires<[HasV4T]>, ImmRegRel;

// Rx=add(Ru,mpyi(Rx,Rs))
let validSubTargets = HasV4SubT, InputType = "reg", CextOpcode = "ADD_MPY" in
def ADDr_MPYrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst = add($src1, mpyi($src2, $src3))",
            [(set (i32 IntRegs:$dst),
             (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src2),
                                            (i32 IntRegs:$src3))))],
            "$src2 = $dst">,
            Requires<[HasV4T]>, ImmRegRel;


// Polynomial multiply words
// Rdd=pmpyw(Rs,Rt)
// Rxx^=pmpyw(Rs,Rt)

// Vector reduce multiply word by signed half (32x16)
// Rdd=vrmpyweh(Rss,Rtt)[:<<1]
// Rdd=vrmpywoh(Rss,Rtt)[:<<1]
// Rxx+=vrmpyweh(Rss,Rtt)[:<<1]
// Rxx+=vrmpywoh(Rss,Rtt)[:<<1]

// Multiply and use upper result
// Rd=mpy(Rs,Rt.H):<<1:sat
// Rd=mpy(Rs,Rt.L):<<1:sat
// Rd=mpy(Rs,Rt):<<1
// Rd=mpy(Rs,Rt):<<1:sat
// Rd=mpysu(Rs,Rt)
// Rx+=mpy(Rs,Rt):<<1:sat
// Rx-=mpy(Rs,Rt):<<1:sat

// Vector multiply bytes
// Rdd=vmpybsu(Rs,Rt)
// Rdd=vmpybu(Rs,Rt)
// Rxx+=vmpybsu(Rs,Rt)
// Rxx+=vmpybu(Rs,Rt)

// Vector polynomial multiply halfwords
// Rdd=vpmpyh(Rs,Rt)
// Rxx^=vpmpyh(Rs,Rt)

//===----------------------------------------------------------------------===//
// XTYPE/MPY -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// XTYPE/SHIFT +
//===----------------------------------------------------------------------===//

// Shift by immediate and accumulate.
// Rx=add(#u8,asl(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
validSubTargets = HasV4SubT in
def ADDi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = add(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=add(#u8,lsr(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
validSubTargets = HasV4SubT in
def ADDi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = add(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (add (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=sub(#u8,asl(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
validSubTargets = HasV4SubT in
def SUBi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = sub(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (sub (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=sub(#u8,lsr(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
validSubTargets = HasV4SubT in
def SUBi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = sub(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (sub (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


//Shift by immediate and logical.
//Rx=and(#u8,asl(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
validSubTargets = HasV4SubT in
def ANDi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = and(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (and (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=and(#u8,lsr(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
validSubTargets = HasV4SubT in
def ANDi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = and(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (and (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                       u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=or(#u8,asl(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
AddedComplexity = 30, validSubTargets = HasV4SubT in
def ORi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = or(#$src1, asl($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (or (shl (i32 IntRegs:$src2), u5ImmPred:$src3),
                      u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=or(#u8,lsr(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
AddedComplexity = 30, validSubTargets = HasV4SubT in
def ORi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Ext:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = or(#$src1, lsr($src2, #$src3))",
            [(set (i32 IntRegs:$dst),
                  (or (srl (i32 IntRegs:$src2), u5ImmPred:$src3),
                      u8ExtPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


//Shift by register.
//Rd=lsl(#s6,Rt)
let validSubTargets = HasV4SubT in {
def LSLi_V4 : MInst<(outs IntRegs:$dst), (ins s6Imm:$src1, IntRegs:$src2),
            "$dst = lsl(#$src1, $src2)",
            [(set (i32 IntRegs:$dst), (shl s6ImmPred:$src1,
                                           (i32 IntRegs:$src2)))]>,
            Requires<[HasV4T]>;


//Shift by register and logical.
//Rxx^=asl(Rss,Rt)
def ASLd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= asl($src2, $src3)",
            [(set (i64 DoubleRegs:$dst),
                  (xor (i64 DoubleRegs:$src1), (shl (i64 DoubleRegs:$src2),
                                                    (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=asr(Rss,Rt)
def ASRd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= asr($src2, $src3)",
            [(set (i64 DoubleRegs:$dst),
                  (xor (i64 DoubleRegs:$src1), (sra (i64 DoubleRegs:$src2),
                                                    (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=lsl(Rss,Rt)
def LSLd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= lsl($src2, $src3)",
            [(set (i64 DoubleRegs:$dst), (xor (i64 DoubleRegs:$src1),
                                              (shl (i64 DoubleRegs:$src2),
                                                   (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=lsr(Rss,Rt)
def LSRd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= lsr($src2, $src3)",
            [(set (i64 DoubleRegs:$dst),
                  (xor (i64 DoubleRegs:$src1), (srl (i64 DoubleRegs:$src2),
                                                    (i32 IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;
}

//===----------------------------------------------------------------------===//
// XTYPE/SHIFT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MEMOP: Word, Half, Byte
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MEMOP: Word
//
//  Implemented:
//     MEMw_ADDi_indexed_V4  : memw(Rs+#u6:2)+=#U5
//     MEMw_SUBi_indexed_V4  : memw(Rs+#u6:2)-=#U5
//     MEMw_ADDr_indexed_V4  : memw(Rs+#u6:2)+=Rt
//     MEMw_SUBr_indexed_V4  : memw(Rs+#u6:2)-=Rt
//     MEMw_CLRr_indexed_V4  : memw(Rs+#u6:2)&=Rt
//     MEMw_SETr_indexed_V4  : memw(Rs+#u6:2)|=Rt
//     MEMw_ADDi_V4          : memw(Rs+#u6:2)+=#U5
//     MEMw_SUBi_V4          : memw(Rs+#u6:2)-=#U5
//     MEMw_ADDr_V4          : memw(Rs+#u6:2)+=Rt
//     MEMw_SUBr_V4          : memw(Rs+#u6:2)-=Rt
//     MEMw_CLRr_V4          : memw(Rs+#u6:2)&=Rt
//     MEMw_SETr_V4          : memw(Rs+#u6:2)|=Rt
//
//   Not implemented:
//     MEMw_CLRi_indexed_V4  : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMw_SETi_indexed_V4  : memw(Rs+#u6:2)=setbit(#U5)
//     MEMw_CLRi_V4          : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMw_SETi_V4          : memw(Rs+#u6:2)=setbit(#U5)
//===----------------------------------------------------------------------===//



// memw(Rs+#u6:2) += #U5
let AddedComplexity = 30 in
def MEMw_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, u5Imm:$addend),
            "memw($base+#$offset) += #$addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= #U5
let AddedComplexity = 30 in
def MEMw_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, u5Imm:$subend),
            "memw($base+#$offset) -= #$subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += Rt
let AddedComplexity = 30 in
def MEMw_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$addend),
            "memw($base+#$offset) += $addend",
            [(store (add (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         (i32 IntRegs:$addend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= Rt
let AddedComplexity = 30 in
def MEMw_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$subend),
            "memw($base+#$offset) -= $subend",
            [(store (sub (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         (i32 IntRegs:$subend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) &= Rt
let AddedComplexity = 30 in
def MEMw_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$andend),
            "memw($base+#$offset) &= $andend",
            [(store (and (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                         (i32 IntRegs:$andend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) |= Rt
let AddedComplexity = 30 in
def MEMw_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$orend),
            "memw($base+#$offset) |= $orend",
            [(store (or (load (add (i32 IntRegs:$base), u6_2ImmPred:$offset)),
                        (i32 IntRegs:$orend)),
                    (add (i32 IntRegs:$base), u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += #U5
let AddedComplexity = 30 in
def MEMw_ADDi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$addend),
            "memw($addr) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= #U5
let AddedComplexity = 30 in
def MEMw_SUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$subend),
            "memw($addr) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += Rt
let AddedComplexity = 30 in
def MEMw_ADDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$addend),
            "memw($addr) += $addend",
            [(store (add (load ADDRriU6_2:$addr), (i32 IntRegs:$addend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= Rt
let AddedComplexity = 30 in
def MEMw_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memw($addr) -= $subend",
            [(store (sub (load ADDRriU6_2:$addr), (i32 IntRegs:$subend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) &= Rt
let AddedComplexity = 30 in
def MEMw_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memw($addr) &= $andend",
            [(store (and (load ADDRriU6_2:$addr), (i32 IntRegs:$andend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) |= Rt
let AddedComplexity = 30 in
def MEMw_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memw($addr) |= $orend",
            [(store (or (load ADDRriU6_2:$addr), (i32 IntRegs:$orend)),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

//===----------------------------------------------------------------------===//
// MEMOP: Halfword
//
//  Implemented:
//     MEMh_ADDi_indexed_V4  : memw(Rs+#u6:2)+=#U5
//     MEMh_SUBi_indexed_V4  : memw(Rs+#u6:2)-=#U5
//     MEMh_ADDr_indexed_V4  : memw(Rs+#u6:2)+=Rt
//     MEMh_SUBr_indexed_V4  : memw(Rs+#u6:2)-=Rt
//     MEMh_CLRr_indexed_V4  : memw(Rs+#u6:2)&=Rt
//     MEMh_SETr_indexed_V4  : memw(Rs+#u6:2)|=Rt
//     MEMh_ADDi_V4          : memw(Rs+#u6:2)+=#U5
//     MEMh_SUBi_V4          : memw(Rs+#u6:2)-=#U5
//     MEMh_ADDr_V4          : memw(Rs+#u6:2)+=Rt
//     MEMh_SUBr_V4          : memw(Rs+#u6:2)-=Rt
//     MEMh_CLRr_V4          : memw(Rs+#u6:2)&=Rt
//     MEMh_SETr_V4          : memw(Rs+#u6:2)|=Rt
//
//   Not implemented:
//     MEMh_CLRi_indexed_V4  : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMh_SETi_indexed_V4  : memw(Rs+#u6:2)=setbit(#U5)
//     MEMh_CLRi_V4          : memw(Rs+#u6:2)=clrbit(#U5)
//     MEMh_SETi_V4          : memw(Rs+#u6:2)=setbit(#U5)
//===----------------------------------------------------------------------===//


// memh(Rs+#u6:1) += #U5
let AddedComplexity = 30 in
def MEMh_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, u5Imm:$addend),
            "memh($base+#$offset) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= #U5
let AddedComplexity = 30 in
def MEMh_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, u5Imm:$subend),
            "memh($base+#$offset) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += Rt
let AddedComplexity = 30 in
def MEMh_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$addend),
            "memh($base+#$offset) += $addend",
            [(truncstorei16 (add (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 (i32 IntRegs:$addend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= Rt
let AddedComplexity = 30 in
def MEMh_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$subend),
            "memh($base+#$offset) -= $subend",
            [(truncstorei16 (sub (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 (i32 IntRegs:$subend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) &= Rt
let AddedComplexity = 30 in
def MEMh_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$andend),
            "memh($base+#$offset) += $andend",
            [(truncstorei16 (and (sextloadi16 (add (i32 IntRegs:$base),
                                                   u6_1ImmPred:$offset)),
                                 (i32 IntRegs:$andend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) |= Rt
let AddedComplexity = 30 in
def MEMh_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$orend),
            "memh($base+#$offset) |= $orend",
            [(truncstorei16 (or (sextloadi16 (add (i32 IntRegs:$base),
                                              u6_1ImmPred:$offset)),
                             (i32 IntRegs:$orend)),
                            (add (i32 IntRegs:$base), u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += #U5
let AddedComplexity = 30 in
def MEMh_ADDi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$addend),
            "memh($addr) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= #U5
let AddedComplexity = 30 in
def MEMh_SUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$subend),
            "memh($addr) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) += Rt
let AddedComplexity = 30 in
def MEMh_ADDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$addend),
            "memh($addr) += $addend",
            [(truncstorei16 (add (sextloadi16 ADDRriU6_1:$addr),
                                 (i32 IntRegs:$addend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= Rt
let AddedComplexity = 30 in
def MEMh_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memh($addr) -= $subend",
            [(truncstorei16 (sub (sextloadi16 ADDRriU6_1:$addr),
                                 (i32 IntRegs:$subend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) &= Rt
let AddedComplexity = 30 in
def MEMh_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memh($addr) &= $andend",
            [(truncstorei16 (and (sextloadi16 ADDRriU6_1:$addr),
                                 (i32 IntRegs:$andend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) |= Rt
let AddedComplexity = 30 in
def MEMh_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memh($addr) |= $orend",
            [(truncstorei16 (or (sextloadi16 ADDRriU6_1:$addr),
                                (i32 IntRegs:$orend)), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;


//===----------------------------------------------------------------------===//
// MEMOP: Byte
//
//  Implemented:
//     MEMb_ADDi_indexed_V4  : memb(Rs+#u6:0)+=#U5
//     MEMb_SUBi_indexed_V4  : memb(Rs+#u6:0)-=#U5
//     MEMb_ADDr_indexed_V4  : memb(Rs+#u6:0)+=Rt
//     MEMb_SUBr_indexed_V4  : memb(Rs+#u6:0)-=Rt
//     MEMb_CLRr_indexed_V4  : memb(Rs+#u6:0)&=Rt
//     MEMb_SETr_indexed_V4  : memb(Rs+#u6:0)|=Rt
//     MEMb_ADDi_V4          : memb(Rs+#u6:0)+=#U5
//     MEMb_SUBi_V4          : memb(Rs+#u6:0)-=#U5
//     MEMb_ADDr_V4          : memb(Rs+#u6:0)+=Rt
//     MEMb_SUBr_V4          : memb(Rs+#u6:0)-=Rt
//     MEMb_CLRr_V4          : memb(Rs+#u6:0)&=Rt
//     MEMb_SETr_V4          : memb(Rs+#u6:0)|=Rt
//
//   Not implemented:
//     MEMb_CLRi_indexed_V4  : memb(Rs+#u6:0)=clrbit(#U5)
//     MEMb_SETi_indexed_V4  : memb(Rs+#u6:0)=setbit(#U5)
//     MEMb_CLRi_V4          : memb(Rs+#u6:0)=clrbit(#U5)
//     MEMb_SETi_V4          : memb(Rs+#u6:0)=setbit(#U5)
//===----------------------------------------------------------------------===//

// memb(Rs+#u6:0) += #U5
let AddedComplexity = 30 in
def MEMb_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, u5Imm:$addend),
            "memb($base+#$offset) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= #U5
let AddedComplexity = 30 in
def MEMb_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, u5Imm:$subend),
            "memb($base+#$offset) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += Rt
let AddedComplexity = 30 in
def MEMb_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$addend),
            "memb($base+#$offset) += $addend",
            [(truncstorei8 (add (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                (i32 IntRegs:$addend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= Rt
let AddedComplexity = 30 in
def MEMb_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$subend),
            "memb($base+#$offset) -= $subend",
            [(truncstorei8 (sub (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                (i32 IntRegs:$subend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) &= Rt
let AddedComplexity = 30 in
def MEMb_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$andend),
            "memb($base+#$offset) += $andend",
            [(truncstorei8 (and (sextloadi8 (add (i32 IntRegs:$base),
                                                 u6_0ImmPred:$offset)),
                                (i32 IntRegs:$andend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) |= Rt
let AddedComplexity = 30 in
def MEMb_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$orend),
            "memb($base+#$offset) |= $orend",
            [(truncstorei8 (or (sextloadi8 (add (i32 IntRegs:$base),
                                                u6_0ImmPred:$offset)),
                               (i32 IntRegs:$orend)),
                           (add (i32 IntRegs:$base), u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += #U5
let AddedComplexity = 30 in
def MEMb_ADDi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$addend),
            "memb($addr) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= #U5
let AddedComplexity = 30 in
def MEMb_SUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, u5Imm:$subend),
            "memb($addr) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) += Rt
let AddedComplexity = 30 in
def MEMb_ADDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$addend),
            "memb($addr) += $addend",
            [(truncstorei8 (add (sextloadi8 ADDRriU6_0:$addr),
                                (i32 IntRegs:$addend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= Rt
let AddedComplexity = 30 in
def MEMb_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memb($addr) -= $subend",
            [(truncstorei8 (sub (sextloadi8 ADDRriU6_0:$addr),
                                (i32 IntRegs:$subend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) &= Rt
let AddedComplexity = 30 in
def MEMb_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memb($addr) &= $andend",
            [(truncstorei8 (and (sextloadi8 ADDRriU6_0:$addr),
                                (i32 IntRegs:$andend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) |= Rt
let AddedComplexity = 30 in
def MEMb_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memb($addr) |= $orend",
            [(truncstorei8 (or (sextloadi8 ADDRriU6_0:$addr),
                               (i32 IntRegs:$orend)), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;


//===----------------------------------------------------------------------===//
// XTYPE/PRED +
//===----------------------------------------------------------------------===//

// Hexagon V4 only supports these flavors of byte/half compare instructions:
// EQ/GT/GTU. Other flavors like GE/GEU/LT/LTU/LE/LEU are not supported by
// hardware. However, compiler can still implement these patterns through
// appropriate patterns combinations based on current implemented patterns.
// The implemented patterns are: EQ/GT/GTU.
// Missing patterns are: GE/GEU/LT/LTU/LE/LEU.

// Following instruction is not being extended as it results into the
// incorrect code for negative numbers.
// Pd=cmpb.eq(Rs,#u8)

// p=!cmp.eq(r1,r2)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPnotEQ_rr : ALU32_rr<(outs PredRegs:$dst),
                           (ins IntRegs:$src1, IntRegs:$src2),
      "$dst = !cmp.eq($src1, $src2)",
      [(set (i1 PredRegs:$dst),
            (setne (i32 IntRegs:$src1), (i32 IntRegs:$src2)))]>,
      Requires<[HasV4T]>;

// p=!cmp.eq(r1,#s10)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPnotEQ_ri : ALU32_ri<(outs PredRegs:$dst),
                           (ins IntRegs:$src1, s10Ext:$src2),
      "$dst = !cmp.eq($src1, #$src2)",
      [(set (i1 PredRegs:$dst),
            (setne (i32 IntRegs:$src1), s10ImmPred:$src2))]>,
      Requires<[HasV4T]>;

// p=!cmp.gt(r1,r2)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPnotGT_rr : ALU32_rr<(outs PredRegs:$dst),
                           (ins IntRegs:$src1, IntRegs:$src2),
      "$dst = !cmp.gt($src1, $src2)",
      [(set (i1 PredRegs:$dst),
            (not (setgt (i32 IntRegs:$src1), (i32 IntRegs:$src2))))]>,
      Requires<[HasV4T]>;

// p=!cmp.gt(r1,#s10)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPnotGT_ri : ALU32_ri<(outs PredRegs:$dst),
                           (ins IntRegs:$src1, s10Ext:$src2),
      "$dst = !cmp.gt($src1, #$src2)",
      [(set (i1 PredRegs:$dst),
            (not (setgt (i32 IntRegs:$src1), s10ImmPred:$src2)))]>,
      Requires<[HasV4T]>;

// p=!cmp.gtu(r1,r2)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPnotGTU_rr : ALU32_rr<(outs PredRegs:$dst),
                            (ins IntRegs:$src1, IntRegs:$src2),
      "$dst = !cmp.gtu($src1, $src2)",
      [(set (i1 PredRegs:$dst),
            (not (setugt (i32 IntRegs:$src1), (i32 IntRegs:$src2))))]>,
      Requires<[HasV4T]>;

// p=!cmp.gtu(r1,#u9)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPnotGTU_ri : ALU32_ri<(outs PredRegs:$dst),
                            (ins IntRegs:$src1, u9Ext:$src2),
      "$dst = !cmp.gtu($src1, #$src2)",
      [(set (i1 PredRegs:$dst),
            (not (setugt (i32 IntRegs:$src1), u9ImmPred:$src2)))]>,
      Requires<[HasV4T]>;

let isCompare = 1, validSubTargets = HasV4SubT in
def CMPbEQri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u8Imm:$src2),
            "$dst = cmpb.eq($src1, #$src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (and (i32 IntRegs:$src1), 255), u8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

def : Pat <(brcond (i1 (setne (and (i32 IntRegs:$src1), 255), u8ImmPred:$src2)),
                       bb:$offset),
      (JMP_cNot (CMPbEQri_V4 (i32 IntRegs:$src1), u8ImmPred:$src2),
                bb:$offset)>,
      Requires<[HasV4T]>;

// Pd=cmpb.eq(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPbEQrr_ubub_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (and (xor (i32 IntRegs:$src1),
                                   (i32 IntRegs:$src2)), 255), 0))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.eq(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPbEQrr_sbsb_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (shl (i32 IntRegs:$src1), (i32 24)),
                         (shl (i32 IntRegs:$src2), (i32 24))))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gt(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPbGTrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.gt($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (setgt (shl (i32 IntRegs:$src1), (i32 24)),
                         (shl (i32 IntRegs:$src2), (i32 24))))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gtu(Rs,#u7)
let isExtendable = 1, opExtendable = 2, isExtentSigned = 0, opExtentBits = 7,
isCompare = 1, validSubTargets = HasV4SubT, CextOpcode = "CMPbGTU", InputType = "imm" in
def CMPbGTUri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u7Ext:$src2),
            "$dst = cmpb.gtu($src1, #$src2)",
            [(set (i1 PredRegs:$dst), (setugt (and (i32 IntRegs:$src1), 255),
                                              u7ExtPred:$src2))]>,
            Requires<[HasV4T]>, ImmRegRel;

// SDNode for converting immediate C to C-1.
def DEC_CONST_BYTE : SDNodeXForm<imm, [{
   // Return the byte immediate const-1 as an SDNode.
   int32_t imm = N->getSExtValue();
   return XformU7ToU7M1Imm(imm);
}]>;

// For the sequence
//   zext( seteq ( and(Rs, 255), u8))
// Generate
//   Pd=cmpb.eq(Rs, #u8)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
def : Pat <(i32 (zext (i1 (seteq (i32 (and (i32 IntRegs:$Rs), 255)),
                                           u8ExtPred:$u8)))),
           (i32 (TFR_condset_ii (i1 (CMPbEQri_V4 (i32 IntRegs:$Rs),
                                                 (u8ExtPred:$u8))),
                                1, 0))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setne ( and(Rs, 255), u8))
// Generate
//   Pd=cmpb.eq(Rs, #u8)
//   if (Pd.new) Rd=#0
//   if (!Pd.new) Rd=#1
def : Pat <(i32 (zext (i1 (setne (i32 (and (i32 IntRegs:$Rs), 255)),
                                           u8ExtPred:$u8)))),
           (i32 (TFR_condset_ii (i1 (CMPbEQri_V4 (i32 IntRegs:$Rs),
                                                 (u8ExtPred:$u8))),
                                0, 1))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( seteq (Rs, and(Rt, 255)))
// Generate
//   Pd=cmpb.eq(Rs, Rt)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
def : Pat <(i32 (zext (i1 (seteq (i32 IntRegs:$Rt),
                                 (i32 (and (i32 IntRegs:$Rs), 255)))))),
           (i32 (TFR_condset_ii (i1 (CMPbEQrr_ubub_V4 (i32 IntRegs:$Rs),
                                                      (i32 IntRegs:$Rt))),
                                1, 0))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setne (Rs, and(Rt, 255)))
// Generate
//   Pd=cmpb.eq(Rs, Rt)
//   if (Pd.new) Rd=#0
//   if (!Pd.new) Rd=#1
def : Pat <(i32 (zext (i1 (setne (i32 IntRegs:$Rt),
                                 (i32 (and (i32 IntRegs:$Rs), 255)))))),
           (i32 (TFR_condset_ii (i1 (CMPbEQrr_ubub_V4 (i32 IntRegs:$Rs),
                                                      (i32 IntRegs:$Rt))),
                                0, 1))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setugt ( and(Rs, 255), u8))
// Generate
//   Pd=cmpb.gtu(Rs, #u8)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
def : Pat <(i32 (zext (i1 (setugt (i32 (and (i32 IntRegs:$Rs), 255)),
                                            u8ExtPred:$u8)))),
           (i32 (TFR_condset_ii (i1 (CMPbGTUri_V4 (i32 IntRegs:$Rs),
                                                  (u8ExtPred:$u8))),
                                1, 0))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setugt ( and(Rs, 254), u8))
// Generate
//   Pd=cmpb.gtu(Rs, #u8)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
def : Pat <(i32 (zext (i1 (setugt (i32 (and (i32 IntRegs:$Rs), 254)),
                                            u8ExtPred:$u8)))),
           (i32 (TFR_condset_ii (i1 (CMPbGTUri_V4 (i32 IntRegs:$Rs),
                                                  (u8ExtPred:$u8))),
                                1, 0))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setult ( Rs, Rt))
// Generate
//   Pd=cmp.ltu(Rs, Rt)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
// cmp.ltu(Rs, Rt) -> cmp.gtu(Rt, Rs)
def : Pat <(i32 (zext (i1 (setult (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTUrr (i32 IntRegs:$Rt),
                                              (i32 IntRegs:$Rs))),
                                1, 0))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setlt ( Rs, Rt))
// Generate
//   Pd=cmp.lt(Rs, Rt)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
// cmp.lt(Rs, Rt) -> cmp.gt(Rt, Rs)
def : Pat <(i32 (zext (i1 (setlt (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTrr (i32 IntRegs:$Rt),
                                             (i32 IntRegs:$Rs))),
                                1, 0))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setugt ( Rs, Rt))
// Generate
//   Pd=cmp.gtu(Rs, Rt)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0
def : Pat <(i32 (zext (i1 (setugt (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTUrr (i32 IntRegs:$Rs),
                                              (i32 IntRegs:$Rt))),
                                1, 0))>,
           Requires<[HasV4T]>;

// This pattern interefers with coremark performance, not implementing at this
// time.
// For the sequence
//   zext( setgt ( Rs, Rt))
// Generate
//   Pd=cmp.gt(Rs, Rt)
//   if (Pd.new) Rd=#1
//   if (!Pd.new) Rd=#0

// For the sequence
//   zext( setuge ( Rs, Rt))
// Generate
//   Pd=cmp.ltu(Rs, Rt)
//   if (Pd.new) Rd=#0
//   if (!Pd.new) Rd=#1
// cmp.ltu(Rs, Rt) -> cmp.gtu(Rt, Rs)
def : Pat <(i32 (zext (i1 (setuge (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTUrr (i32 IntRegs:$Rt),
                                              (i32 IntRegs:$Rs))),
                                0, 1))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setge ( Rs, Rt))
// Generate
//   Pd=cmp.lt(Rs, Rt)
//   if (Pd.new) Rd=#0
//   if (!Pd.new) Rd=#1
// cmp.lt(Rs, Rt) -> cmp.gt(Rt, Rs)
def : Pat <(i32 (zext (i1 (setge (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTrr (i32 IntRegs:$Rt),
                                             (i32 IntRegs:$Rs))),
                                0, 1))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setule ( Rs, Rt))
// Generate
//   Pd=cmp.gtu(Rs, Rt)
//   if (Pd.new) Rd=#0
//   if (!Pd.new) Rd=#1
def : Pat <(i32 (zext (i1 (setule (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTUrr (i32 IntRegs:$Rs),
                                              (i32 IntRegs:$Rt))),
                                0, 1))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setle ( Rs, Rt))
// Generate
//   Pd=cmp.gt(Rs, Rt)
//   if (Pd.new) Rd=#0
//   if (!Pd.new) Rd=#1
def : Pat <(i32 (zext (i1 (setle (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))),
           (i32 (TFR_condset_ii (i1 (CMPGTrr (i32 IntRegs:$Rs),
                                             (i32 IntRegs:$Rt))),
                                0, 1))>,
           Requires<[HasV4T]>;

// For the sequence
//   zext( setult ( and(Rs, 255), u8))
// Use the isdigit transformation below

// Generate code of the form 'mux_ii(cmpbgtu(Rdd, C-1),0,1)'
// for C code of the form r = ((c>='0') & (c<='9')) ? 1 : 0;.
// The isdigit transformation relies on two 'clever' aspects:
// 1) The data type is unsigned which allows us to eliminate a zero test after
//    biasing the expression by 48. We are depending on the representation of
//    the unsigned types, and semantics.
// 2) The front end has converted <= 9 into < 10 on entry to LLVM
//
// For the C code:
//   retval = ((c>='0') & (c<='9')) ? 1 : 0;
// The code is transformed upstream of llvm into
//   retval = (c-48) < 10 ? 1 : 0;
let AddedComplexity = 139 in
def : Pat <(i32 (zext (i1 (setult (i32 (and (i32 IntRegs:$src1), 255)),
                                  u7StrictPosImmPred:$src2)))),
  (i32 (MUX_ii (i1 (CMPbGTUri_V4 (i32 IntRegs:$src1),
                                 (DEC_CONST_BYTE u7StrictPosImmPred:$src2))),
                   0, 1))>,
                   Requires<[HasV4T]>;

// Pd=cmpb.gtu(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT, CextOpcode = "CMPbGTU",
InputType = "reg" in
def CMPbGTUrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.gtu($src1, $src2)",
            [(set (i1 PredRegs:$dst), (setugt (and (i32 IntRegs:$src1), 255),
                                             (and (i32 IntRegs:$src2), 255)))]>,
            Requires<[HasV4T]>, ImmRegRel;

// Following instruction is not being extended as it results into the incorrect
// code for negative numbers.

// Signed half compare(.eq) ri.
// Pd=cmph.eq(Rs,#s8)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPhEQri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, s8Imm:$src2),
            "$dst = cmph.eq($src1, #$src2)",
            [(set (i1 PredRegs:$dst), (seteq (and (i32 IntRegs:$src1), 65535),
                                             s8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) rr.
// Case 1: xor + and, then compare:
//   r0=xor(r0,r1)
//   r0=and(r0,#0xffff)
//   p0=cmp.eq(r0,#0)
// Pd=cmph.eq(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPhEQrr_xor_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst), (seteq (and (xor (i32 IntRegs:$src1),
                                                       (i32 IntRegs:$src2)),
                                                  65535), 0))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) rr.
// Case 2: shift left 16 bits then compare:
//   r0=asl(r0,16)
//   r1=asl(r1,16)
//   p0=cmp.eq(r0,r1)
// Pd=cmph.eq(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPhEQrr_shl_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.eq($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (seteq (shl (i32 IntRegs:$src1), (i32 16)),
                         (shl (i32 IntRegs:$src2), (i32 16))))]>,
            Requires<[HasV4T]>;

/* Incorrect Pattern -- immediate should be right shifted before being
used in the cmph.gt instruction.
// Signed half compare(.gt) ri.
// Pd=cmph.gt(Rs,#s8)

let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 8,
isCompare = 1, validSubTargets = HasV4SubT in
def CMPhGTri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, s8Ext:$src2),
            "$dst = cmph.gt($src1, #$src2)",
            [(set (i1 PredRegs:$dst),
                  (setgt (shl (i32 IntRegs:$src1), (i32 16)),
                         s8ExtPred:$src2))]>,
            Requires<[HasV4T]>;
*/

// Signed half compare(.gt) rr.
// Pd=cmph.gt(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT in
def CMPhGTrr_shl_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.gt($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (setgt (shl (i32 IntRegs:$src1), (i32 16)),
                         (shl (i32 IntRegs:$src2), (i32 16))))]>,
            Requires<[HasV4T]>;

// Unsigned half compare rr (.gtu).
// Pd=cmph.gtu(Rs,Rt)
let isCompare = 1, validSubTargets = HasV4SubT, CextOpcode = "CMPhGTU",
InputType = "reg" in
def CMPhGTUrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.gtu($src1, $src2)",
            [(set (i1 PredRegs:$dst),
                  (setugt (and (i32 IntRegs:$src1), 65535),
                          (and (i32 IntRegs:$src2), 65535)))]>,
            Requires<[HasV4T]>, ImmRegRel;

// Unsigned half compare ri (.gtu).
// Pd=cmph.gtu(Rs,#u7)
let isExtendable = 1, opExtendable = 2, isExtentSigned = 0, opExtentBits = 7,
isCompare = 1, validSubTargets = HasV4SubT, CextOpcode = "CMPhGTU",
InputType = "imm" in
def CMPhGTUri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u7Ext:$src2),
            "$dst = cmph.gtu($src1, #$src2)",
            [(set (i1 PredRegs:$dst), (setugt (and (i32 IntRegs:$src1), 65535),
                                              u7ExtPred:$src2))]>,
            Requires<[HasV4T]>, ImmRegRel;

let validSubTargets = HasV4SubT in
def NTSTBIT_rr : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
    "$dst = !tstbit($src1, $src2)",
    [(set (i1 PredRegs:$dst),
          (seteq (and (shl 1, (i32 IntRegs:$src2)), (i32 IntRegs:$src1)), 0))]>,
    Requires<[HasV4T]>;

let validSubTargets = HasV4SubT in
def NTSTBIT_ri : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
    "$dst = !tstbit($src1, $src2)",
    [(set (i1 PredRegs:$dst),
          (seteq (and (shl 1, u5ImmPred:$src2), (i32 IntRegs:$src1)), 0))]>,
    Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// XTYPE/PRED -
//===----------------------------------------------------------------------===//

//Deallocate frame and return.
//    dealloc_return
let isReturn = 1, isTerminator = 1, isBarrier = 1, isPredicable = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_V4 : NVInst_V4<(outs), (ins i32imm:$amt1),
            "dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

// Restore registers and dealloc return function call.
let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC] in {
  def RESTORE_DEALLOC_RET_JMP_V4 : JInst<(outs),
                                   (ins calltarget:$dst),
             "jump $dst // Restore_and_dealloc_return",
             []>,
             Requires<[HasV4T]>;
}

// Restore registers and dealloc frame before a tail call.
let isCall = 1, isBarrier = 1,
  Defs = [R29, R30, R31, PC] in {
  def RESTORE_DEALLOC_BEFORE_TAILCALL_V4 : JInst<(outs),
                                           (ins calltarget:$dst),
             "call $dst // Restore_and_dealloc_before_tailcall",
             []>,
             Requires<[HasV4T]>;
}

// Save registers function call.
let isCall = 1, isBarrier = 1,
  Uses = [R29, R31] in {
  def SAVE_REGISTERS_CALL_V4 : JInst<(outs),
                               (ins calltarget:$dst),
             "call $dst // Save_calle_saved_registers",
             []>,
             Requires<[HasV4T]>;
}

//    if (Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cPt_V4 : NVInst_V4<(outs),
                           (ins PredRegs:$src1, i32imm:$amt1),
            "if ($src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cNotPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                     i32imm:$amt1),
            "if (!$src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cdnPnt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                     i32imm:$amt1),
            "if ($src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cNotdnPnt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                        i32imm:$amt1),
            "if (!$src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:t
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cdnPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                    i32imm:$amt1),
            "if ($src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1,
    isPredicated = 1 in {
  def DEALLOC_RET_cNotdnPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                       i32imm:$amt1),
            "if (!$src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}

// Load/Store with absolute addressing mode
// memw(#u6)=Rt

multiclass ST_Abs_Predbase<string mnemonic, RegisterClass RC, bit isNot,
                           bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME#_V4 : STInst2<(outs),
            (ins PredRegs:$src1, globaladdressExt:$absaddr, RC: $src2),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"(##$absaddr) = $src2",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Abs_Pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_Abs_Predbase<mnemonic, RC, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ST_Abs_Predbase<mnemonic, RC, PredNot, 1>;
  }
}

let isNVStorable = 1, isExtended = 1, neverHasSideEffects = 1 in
multiclass ST_Abs<string mnemonic, string CextOp, RegisterClass RC> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_abs in {
    let opExtendable = 0, isPredicable = 1 in
    def NAME#_V4 : STInst2<(outs),
            (ins globaladdressExt:$absaddr, RC:$src),
            mnemonic#"(##$absaddr) = $src",
            []>,
            Requires<[HasV4T]>;

    let opExtendable = 1, isPredicated = 1 in {
      defm Pt : ST_Abs_Pred<mnemonic, RC, 0>;
      defm NotPt : ST_Abs_Pred<mnemonic, RC, 1>;
    }
  }
}

multiclass ST_Abs_Predbase_nv<string mnemonic, RegisterClass RC, bit isNot,
                           bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, globaladdressExt:$absaddr, RC: $src2),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"(##$absaddr) = $src2.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Abs_Pred_nv<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : ST_Abs_Predbase_nv<mnemonic, RC, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : ST_Abs_Predbase_nv<mnemonic, RC, PredNot, 1>;
  }
}

let mayStore = 1, isNVStore = 1, isExtended = 1, neverHasSideEffects = 1 in
multiclass ST_Abs_nv<string mnemonic, string CextOp, RegisterClass RC> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_abs in {
    let opExtendable = 0, isPredicable = 1 in
    def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins globaladdressExt:$absaddr, RC:$src),
            mnemonic#"(##$absaddr) = $src.new",
            []>,
            Requires<[HasV4T]>;

    let opExtendable = 1, isPredicated = 1 in {
      defm Pt : ST_Abs_Pred_nv<mnemonic, RC, 0>;
      defm NotPt : ST_Abs_Pred_nv<mnemonic, RC, 1>;
    }
  }
}

let addrMode = Absolute in {
    defm STrib_abs : ST_Abs<"memb", "STrib", IntRegs>,
                     ST_Abs_nv<"memb", "STrib", IntRegs>, AddrModeRel;

    defm STrih_abs : ST_Abs<"memh", "STrih", IntRegs>,
                     ST_Abs_nv<"memh", "STrih", IntRegs>, AddrModeRel;

    defm STriw_abs : ST_Abs<"memw", "STriw", IntRegs>,
                     ST_Abs_nv<"memw", "STriw", IntRegs>, AddrModeRel;

  let isNVStorable = 0 in
    defm STrid_abs : ST_Abs<"memd", "STrid", DoubleRegs>, AddrModeRel;
}

let Predicates = [HasV4T], AddedComplexity = 30 in {
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
                        (HexagonCONST32 tglobaladdr:$absaddr)),
          (STrib_abs_V4 tglobaladdr: $absaddr, IntRegs: $src1)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                          (HexagonCONST32 tglobaladdr:$absaddr)),
          (STrih_abs_V4 tglobaladdr: $absaddr, IntRegs: $src1)>;

def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32 tglobaladdr:$absaddr)),
          (STriw_abs_V4 tglobaladdr: $absaddr, IntRegs: $src1)>;

def : Pat<(store (i64 DoubleRegs:$src1),
                 (HexagonCONST32 tglobaladdr:$absaddr)),
          (STrid_abs_V4 tglobaladdr: $absaddr, DoubleRegs: $src1)>;
}

multiclass LD_Abs_Predbase<string mnemonic, RegisterClass RC, bit isNot,
                           bit isPredNew> {
  let PNewValue = !if(isPredNew, "new", "") in
  def NAME : LDInst2<(outs RC:$dst),
            (ins PredRegs:$src1, globaladdressExt:$absaddr),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#"$dst = "#mnemonic#"(##$absaddr)",
            []>,
            Requires<[HasV4T]>;
}

multiclass LD_Abs_Pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let PredSense = !if(PredNot, "false", "true") in {
    defm _c#NAME : LD_Abs_Predbase<mnemonic, RC, PredNot, 0>;
    // Predicate new
    defm _cdn#NAME : LD_Abs_Predbase<mnemonic, RC, PredNot, 1>;
  }
}

let isExtended = 1, neverHasSideEffects = 1 in
multiclass LD_Abs<string mnemonic, string CextOp, RegisterClass RC> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_abs in {
    let  opExtendable = 1, isPredicable = 1 in
    def NAME#_V4 : LDInst2<(outs RC:$dst),
            (ins globaladdressExt:$absaddr),
            "$dst = "#mnemonic#"(##$absaddr)",
            []>,
            Requires<[HasV4T]>;

    let opExtendable = 2, isPredicated = 1 in {
      defm Pt_V4 : LD_Abs_Pred<mnemonic, RC, 0>;
      defm NotPt_V4 : LD_Abs_Pred<mnemonic, RC, 1>;
    }
  }
}

let addrMode = Absolute in {
    defm LDrib_abs  : LD_Abs<"memb", "LDrib", IntRegs>, AddrModeRel;
    defm LDriub_abs : LD_Abs<"memub", "LDriub", IntRegs>, AddrModeRel;
    defm LDrih_abs  : LD_Abs<"memh", "LDrih", IntRegs>, AddrModeRel;
    defm LDriuh_abs : LD_Abs<"memuh", "LDriuh", IntRegs>, AddrModeRel;
    defm LDriw_abs  : LD_Abs<"memw", "LDriw", IntRegs>, AddrModeRel;
    defm LDrid_abs : LD_Abs<"memd",  "LDrid", DoubleRegs>, AddrModeRel;
}

let Predicates = [HasV4T], AddedComplexity  = 30 in
def : Pat<(i32 (load (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriw_abs_V4 tglobaladdr: $absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (sextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDrib_abs_V4 tglobaladdr:$absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (zextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriub_abs_V4 tglobaladdr:$absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (sextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDrih_abs_V4 tglobaladdr:$absaddr)>;

let Predicates = [HasV4T], AddedComplexity=30 in
def : Pat<(i32 (zextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriuh_abs_V4 tglobaladdr:$absaddr)>;

// Transfer global address into a register
let AddedComplexity=50, isMoveImm = 1, isReMaterializable = 1 in
def TFRI_V4 : ALU32_ri<(outs IntRegs:$dst), (ins globaladdress:$src1),
           "$dst = ##$src1",
           [(set IntRegs:$dst, (HexagonCONST32 tglobaladdr:$src1))]>,
           Requires<[HasV4T]>;

// Transfer a block address into a register
def : Pat<(HexagonCONST32_GP tblockaddress:$src1),
          (TFRI_V4 tblockaddress:$src1)>,
          Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                           (ins PredRegs:$src1, globaladdress:$src2),
           "if($src1) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, globaladdress:$src2),
           "if(!$src1) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cdnPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                             (ins PredRegs:$src1, globaladdress:$src2),
           "if($src1.new) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity=50, neverHasSideEffects = 1, isPredicated = 1 in
def TFRI_cdnNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                                (ins PredRegs:$src1, globaladdress:$src2),
           "if(!$src1.new) $dst = ##$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity = 50, Predicates = [HasV4T] in
def : Pat<(HexagonCONST32_GP tglobaladdr:$src1),
           (TFRI_V4 tglobaladdr:$src1)>;


// Load - Indirect with long offset: These instructions take global address
// as an operand
let AddedComplexity = 10 in
def LDrid_ind_lo_V4 : LDInst<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$offset),
            "$dst=memd($src1<<#$src2+##$offset)",
            [(set (i64 DoubleRegs:$dst),
                  (load (add (shl IntRegs:$src1, u2ImmPred:$src2),
                        (HexagonCONST32 tglobaladdr:$offset))))]>,
            Requires<[HasV4T]>;

let AddedComplexity = 10 in
multiclass LD_indirect_lo<string OpcStr, PatFrag OpNode> {
  def _lo_V4 : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$offset),
            !strconcat("$dst = ",
            !strconcat(OpcStr, "($src1<<#$src2+##$offset)")),
            [(set IntRegs:$dst,
                  (i32 (OpNode (add (shl IntRegs:$src1, u2ImmPred:$src2),
                          (HexagonCONST32 tglobaladdr:$offset)))))]>,
            Requires<[HasV4T]>;
}

defm LDrib_ind : LD_indirect_lo<"memb", sextloadi8>;
defm LDriub_ind : LD_indirect_lo<"memub", zextloadi8>;
defm LDrih_ind : LD_indirect_lo<"memh", sextloadi16>;
defm LDriuh_ind : LD_indirect_lo<"memuh", zextloadi16>;
defm LDriw_ind : LD_indirect_lo<"memw", load>;

// Store - Indirect with long offset: These instructions take global address
// as an operand
let AddedComplexity = 10 in
def STrid_ind_lo_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$src3,
                 DoubleRegs:$src4),
            "memd($src1<<#$src2+#$src3) = $src4",
            [(store (i64 DoubleRegs:$src4),
                 (add (shl IntRegs:$src1, u2ImmPred:$src2),
                      (HexagonCONST32 tglobaladdr:$src3)))]>,
             Requires<[HasV4T]>;

let AddedComplexity = 10 in
multiclass ST_indirect_lo<string OpcStr, PatFrag OpNode> {
  def _lo_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdress:$src3,
                 IntRegs:$src4),
            !strconcat(OpcStr, "($src1<<#$src2+##$src3) = $src4"),
            [(OpNode (i32 IntRegs:$src4),
                 (add (shl IntRegs:$src1, u2ImmPred:$src2),
                      (HexagonCONST32 tglobaladdr:$src3)))]>,
             Requires<[HasV4T]>;
}

defm STrib_ind : ST_indirect_lo<"memb", truncstorei8>;
defm STrih_ind : ST_indirect_lo<"memh", truncstorei16>;
defm STriw_ind : ST_indirect_lo<"memw", store>;

// Store - absolute addressing mode: These instruction take constant
// value as the extended operand.
multiclass ST_absimm<string OpcStr> {
let isExtended = 1, opExtendable = 0, isPredicable = 1,
validSubTargets = HasV4SubT in
  def _abs_V4 : STInst2<(outs),
            (ins u0AlwaysExt:$src1, IntRegs:$src2),
            !strconcat(OpcStr, "(##$src1) = $src2"),
            []>,
            Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 1, isPredicated = 1,
validSubTargets = HasV4SubT in {
  def _abs_cPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if ($src1)", !strconcat(OpcStr, "(##$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if (!$src1)", !strconcat(OpcStr, "(##$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cdnPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if ($src1.new)",
            !strconcat(OpcStr, "(##$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cdnNotPt_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if (!$src1.new)",
            !strconcat(OpcStr, "(##$src2) = $src3")),
            []>,
            Requires<[HasV4T]>;
}

let isExtended = 1, opExtendable = 0, mayStore = 1, isNVStore = 1,
validSubTargets = HasV4SubT in
  def _abs_nv_V4 : NVInst_V4<(outs),
            (ins u0AlwaysExt:$src1, IntRegs:$src2),
            !strconcat(OpcStr, "(##$src1) = $src2.new"),
            []>,
            Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 1, mayStore = 1, isPredicated = 1,
isNVStore = 1, validSubTargets = HasV4SubT in {
  def _abs_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if ($src1)",
            !strconcat(OpcStr, "(##$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if (!$src1)",
            !strconcat(OpcStr, "(##$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if ($src1.new)",
            !strconcat(OpcStr, "(##$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$src2, IntRegs:$src3),
            !strconcat("if (!$src1.new)",
            !strconcat(OpcStr, "(##$src2) = $src3.new")),
            []>,
            Requires<[HasV4T]>;
}
}

defm STrib_imm : ST_absimm<"memb">;
defm STrih_imm : ST_absimm<"memh">;
defm STriw_imm : ST_absimm<"memw">;

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(truncstorei8 (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (STrib_imm_abs_V4 u0AlwaysExtPred:$src2, IntRegs: $src1)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (STrih_imm_abs_V4 u0AlwaysExtPred:$src2, IntRegs: $src1)>;

def : Pat<(store (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (STriw_imm_abs_V4 u0AlwaysExtPred:$src2, IntRegs: $src1)>;
}

// Load - absolute addressing mode: These instruction take constant
// value as the extended operand

multiclass LD_absimm<string OpcStr> {
let isExtended = 1, opExtendable = 1, isPredicable = 1,
validSubTargets = HasV4SubT in
  def _abs_V4 : LDInst2<(outs IntRegs:$dst),
            (ins u0AlwaysExt:$src),
            !strconcat("$dst = ",
            !strconcat(OpcStr, "(##$src)")),
            []>,
            Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, isPredicated = 1,
validSubTargets = HasV4SubT in {
  def _abs_cPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u0AlwaysExt:$src2),
            !strconcat("if ($src1) $dst = ",
            !strconcat(OpcStr, "(##$src2)")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u0AlwaysExt:$src2),
            !strconcat("if (!$src1) $dst = ",
            !strconcat(OpcStr, "(##$src2)")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cdnPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u0AlwaysExt:$src2),
            !strconcat("if ($src1.new) $dst = ",
            !strconcat(OpcStr, "(##$src2)")),
            []>,
            Requires<[HasV4T]>;

  def _abs_cdnNotPt_V4 : LDInst2<(outs IntRegs:$dst),
            (ins PredRegs:$src1, u0AlwaysExt:$src2),
            !strconcat("if (!$src1.new) $dst = ",
            !strconcat(OpcStr, "(##$src2)")),
            []>,
            Requires<[HasV4T]>;
}
}

defm LDrib_imm  : LD_absimm<"memb">;
defm LDriub_imm : LD_absimm<"memub">;
defm LDrih_imm  : LD_absimm<"memh">;
defm LDriuh_imm : LD_absimm<"memuh">;
defm LDriw_imm  : LD_absimm<"memw">;

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(i32 (load u0AlwaysExtPred:$src)),
          (LDriw_imm_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (sextloadi8 u0AlwaysExtPred:$src)),
          (LDrib_imm_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (zextloadi8 u0AlwaysExtPred:$src)),
          (LDriub_imm_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (sextloadi16 u0AlwaysExtPred:$src)),
          (LDrih_imm_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (zextloadi16 u0AlwaysExtPred:$src)),
          (LDriuh_imm_abs_V4 u0AlwaysExtPred:$src)>;
}

// Indexed store double word - global address.
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10 in
def STriw_offset_ext_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_2Imm:$src2, globaladdress:$src3),
            "memw($src1+#$src2) = ##$src3",
            [(store (HexagonCONST32 tglobaladdr:$src3),
                    (add IntRegs:$src1, u6_2ImmPred:$src2))]>,
            Requires<[HasV4T]>;


// Indexed store double word - global address.
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10 in
def STrih_offset_ext_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_1Imm:$src2, globaladdress:$src3),
            "memh($src1+#$src2) = ##$src3",
            [(truncstorei16 (HexagonCONST32 tglobaladdr:$src3),
                    (add IntRegs:$src1, u6_1ImmPred:$src2))]>,
            Requires<[HasV4T]>;
// Map from store(globaladdress + x) -> memd(#foo + x)
let AddedComplexity = 100 in
def : Pat<(store (i64 DoubleRegs:$src1),
                 FoldGlobalAddrGP:$addr),
          (STrid_abs_V4 FoldGlobalAddrGP:$addr, (i64 DoubleRegs:$src1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_64 FoldGlobalAddrGP:$addr,
                           (i64 DoubleRegs:$src1)),
          (STrid_abs_V4 FoldGlobalAddrGP:$addr, (i64 DoubleRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1), FoldGlobalAddrGP:$addr),
          (STrib_abs_V4 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

def : Pat<(atomic_store_8 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1)),
          (STrib_abs_V4 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1), FoldGlobalAddrGP:$addr),
          (STrih_abs_V4 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

def : Pat<(atomic_store_16 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1)),
          (STrih_abs_V4 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memw(#foo + x)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1), FoldGlobalAddrGP:$addr),
          (STriw_abs_V4 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_store_32 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1)),
          (STriw_abs_V4 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memd(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i64 (load FoldGlobalAddrGP:$addr)),
          (i64 (LDrid_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_64 FoldGlobalAddrGP:$addr),
          (i64 (LDrid_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (extloadi8 FoldGlobalAddrGP:$addr)),
          (i32 (LDrib_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (sextloadi8 FoldGlobalAddrGP:$addr)),
          (i32 (LDrib_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

//let AddedComplexity = 100 in
let AddedComplexity = 100 in
def : Pat<(i32 (extloadi16 FoldGlobalAddrGP:$addr)),
          (i32 (LDrih_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (sextloadi16 FoldGlobalAddrGP:$addr)),
          (i32 (LDrih_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memuh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (zextloadi16 FoldGlobalAddrGP:$addr)),
          (i32 (LDriuh_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_16 FoldGlobalAddrGP:$addr),
          (i32 (LDriuh_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memub(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (zextloadi8 FoldGlobalAddrGP:$addr)),
          (i32 (LDriub_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_8 FoldGlobalAddrGP:$addr),
          (i32 (LDriub_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memw(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (load FoldGlobalAddrGP:$addr)),
          (i32 (LDriw_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_32 FoldGlobalAddrGP:$addr),
          (i32 (LDriw_abs_V4 FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

