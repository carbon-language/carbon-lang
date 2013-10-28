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
//===----------------------------------------------------------------------===//
// Template class for load instructions with Absolute set addressing mode.
//===----------------------------------------------------------------------===//
let isExtended = 1, opExtendable = 2, neverHasSideEffects = 1,
validSubTargets = HasV4SubT, addrMode = AbsoluteSet in
class T_LD_abs_set<string mnemonic, RegisterClass RC>:
            LDInst2<(outs RC:$dst1, IntRegs:$dst2),
            (ins u0AlwaysExt:$addr),
            "$dst1 = "#mnemonic#"($dst2=##$addr)",
            []>,
            Requires<[HasV4T]>;

def LDrid_abs_set_V4  : T_LD_abs_set <"memd", DoubleRegs>;
def LDrib_abs_set_V4  : T_LD_abs_set <"memb", IntRegs>;
def LDriub_abs_set_V4 : T_LD_abs_set <"memub", IntRegs>;
def LDrih_abs_set_V4  : T_LD_abs_set <"memh", IntRegs>;
def LDriw_abs_set_V4  : T_LD_abs_set <"memw", IntRegs>;
def LDriuh_abs_set_V4 : T_LD_abs_set <"memuh", IntRegs>;


// multiclass for load instructions with base + register offset
// addressing mode
multiclass ld_idxd_shl_pbase<string mnemonic, RegisterClass RC, bit isNot,
                             bit isPredNew> {
  let isPredicatedNew = isPredNew in
  def NAME : LDInst2<(outs RC:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$offset),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#"$dst = "#mnemonic#"($src2+$src3<<#$offset)",
            []>, Requires<[HasV4T]>;
}

multiclass ld_idxd_shl_pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
  let accessSize = ByteAccess in {
    defm LDrib_indexed_shl: ld_idxd_shl<"memb", "LDrib", IntRegs>,
                                        AddrModeRel;
    defm LDriub_indexed_shl: ld_idxd_shl<"memub", "LDriub", IntRegs>,
                                        AddrModeRel;
  }
  let accessSize = HalfWordAccess in {
    defm LDrih_indexed_shl: ld_idxd_shl<"memh", "LDrih", IntRegs>, AddrModeRel;
    defm LDriuh_indexed_shl: ld_idxd_shl<"memuh", "LDriuh", IntRegs>,
                             AddrModeRel;
  }
  let accessSize = WordAccess in
     defm LDriw_indexed_shl: ld_idxd_shl<"memw", "LDriw", IntRegs>, AddrModeRel;

  let accessSize = DoubleWordAccess in
    defm LDrid_indexed_shl: ld_idxd_shl<"memd", "LDrid", DoubleRegs>,
                             AddrModeRel;
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

// zext i1->i64
def:  Pat <(i64 (zextloadi1 ADDRriS11_0:$src1)),
      (i64 (COMBINE_Ir_V4 0, (LDriub ADDRriS11_0:$src1)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (zextloadi1 (add (i32 IntRegs:$src1),
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
//===----------------------------------------------------------------------===//
// Template class for store instructions with Absolute set addressing mode.
//===----------------------------------------------------------------------===//
let isExtended = 1, opExtendable = 2, validSubTargets = HasV4SubT,
addrMode = AbsoluteSet in
class T_ST_abs_set<string mnemonic, RegisterClass RC>:
            STInst2<(outs IntRegs:$dst1),
            (ins RC:$src1, u0AlwaysExt:$src2),
            mnemonic#"($dst1=##$src2) = $src1",
            []>,
            Requires<[HasV4T]>;

def STrid_abs_set_V4 : T_ST_abs_set <"memd", DoubleRegs>;
def STrib_abs_set_V4 : T_ST_abs_set <"memb", IntRegs>;
def STrih_abs_set_V4 : T_ST_abs_set <"memh", IntRegs>;
def STriw_abs_set_V4 : T_ST_abs_set <"memw", IntRegs>;

//===----------------------------------------------------------------------===//
// multiclass for store instructions with base + register offset addressing
// mode
//===----------------------------------------------------------------------===//
multiclass ST_Idxd_shl_Pbase<string mnemonic, RegisterClass RC, bit isNot,
                             bit isPredNew> {
  let isPredicatedNew = isPredNew in
  def NAME : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 RC:$src5),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Idxd_shl_Pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
  let isPredicatedNew = isPredNew in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 RC:$src5),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Idxd_shl_Pred_nv<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
  let accessSize = ByteAccess in
    defm STrib_indexed_shl: ST_Idxd_shl<"memb", "STrib", IntRegs>,
                            ST_Idxd_shl_nv<"memb", "STrib", IntRegs>, AddrModeRel;

  let accessSize = HalfWordAccess in
    defm STrih_indexed_shl: ST_Idxd_shl<"memh", "STrih", IntRegs>,
                            ST_Idxd_shl_nv<"memh", "STrih", IntRegs>, AddrModeRel;

  let accessSize = WordAccess in
    defm STriw_indexed_shl: ST_Idxd_shl<"memw", "STriw", IntRegs>,
                            ST_Idxd_shl_nv<"memw", "STriw", IntRegs>, AddrModeRel;

  let isNVStorable = 0, accessSize = DoubleWordAccess in
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

let isExtended = 1, opExtendable = 2 in
class T_ST_LongOff <string mnemonic, PatFrag stOp, RegisterClass RC, ValueType VT> :
            STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, RC:$src4),
            mnemonic#"($src1<<#$src2+##$src3) = $src4",
            [(stOp (VT RC:$src4),
                    (add (shl (i32 IntRegs:$src1), u2ImmPred:$src2),
                         u0AlwaysExtPred:$src3))]>,
            Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, mayStore = 1, isNVStore = 1 in
class T_ST_LongOff_nv <string mnemonic> :
            NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u0AlwaysExt:$src3, IntRegs:$src4),
            mnemonic#"($src1<<#$src2+##$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

multiclass ST_LongOff <string mnemonic, string BaseOp, PatFrag stOp> {
  let  BaseOpcode = BaseOp#"_shl" in {
    let isNVStorable = 1 in
    def NAME#_V4 : T_ST_LongOff<mnemonic, stOp, IntRegs, i32>;

    def NAME#_nv_V4 : T_ST_LongOff_nv<mnemonic>;
  }
}

let AddedComplexity = 10, validSubTargets = HasV4SubT in {
  def STrid_shl_V4 : T_ST_LongOff<"memd", store, DoubleRegs, i64>;
  defm STrib_shl   : ST_LongOff <"memb", "STrib", truncstorei8>, NewValueRel;
  defm STrih_shl   : ST_LongOff <"memh", "Strih", truncstorei16>, NewValueRel;
  defm STriw_shl   : ST_LongOff <"memw", "STriw", store>, NewValueRel;
}

let AddedComplexity = 40 in
multiclass T_ST_LOff_Pats <InstHexagon I, RegisterClass RC, ValueType VT,
                           PatFrag stOp> {
 def : Pat<(stOp (VT RC:$src4),
           (add (shl IntRegs:$src1, u2ImmPred:$src2),
               (NumUsesBelowThresCONST32 tglobaladdr:$src3))),
           (I IntRegs:$src1, u2ImmPred:$src2, tglobaladdr:$src3, RC:$src4)>;

 def : Pat<(stOp (VT RC:$src4),
           (add IntRegs:$src1,
               (NumUsesBelowThresCONST32 tglobaladdr:$src3))),
           (I IntRegs:$src1, 0, tglobaladdr:$src3, RC:$src4)>;
}

defm : T_ST_LOff_Pats<STrid_shl_V4, DoubleRegs, i64, store>;
defm : T_ST_LOff_Pats<STriw_shl_V4, IntRegs, i32, store>;
defm : T_ST_LOff_Pats<STrib_shl_V4, IntRegs, i32, truncstorei8>;
defm : T_ST_LOff_Pats<STrih_shl_V4, IntRegs, i32, truncstorei16>;

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
  let isPredicatedNew = isPredNew in
  def NAME : STInst2<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, OffsetOp:$src3, s6Ext:$src4),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Imm_Pred<string mnemonic, Operand OffsetOp, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
  let accessSize = ByteAccess in
    defm STrib_imm : ST_Imm<"memb", "STrib", u6_0Imm>, ImmRegRel, PredNewRel;

  let accessSize = HalfWordAccess in
    defm STrih_imm : ST_Imm<"memh", "STrih", u6_1Imm>, ImmRegRel, PredNewRel;

  let accessSize = WordAccess in
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

// memw(Rx++#s4:2)=Rt
// memw(Rx++#s4:2:circ(Mu))=Rt
// memw(Rx++I:circ(Mu))=Rt
// memw(Rx++Mu)=Rt
// memw(Rx++Mu:brev)=Rt

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
  let isPredicatedNew = isPredNew in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, predImmOp:$src3, RC: $src4),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Idxd_Pred_nv<string mnemonic, RegisterClass RC, Operand predImmOp,
                           bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
  let accessSize = ByteAccess in
    defm STrib_indexed: ST_Idxd_nv<"memb", "STrib", IntRegs, s11_0Ext,
                                   u6_0Ext, 11, 6>, AddrModeRel;

  let accessSize = HalfWordAccess in
    defm STrih_indexed: ST_Idxd_nv<"memh", "STrih", IntRegs, s11_1Ext,
                                   u6_1Ext, 12, 7>, AddrModeRel;

  let accessSize = WordAccess in
    defm STriw_indexed: ST_Idxd_nv<"memw", "STriw", IntRegs, s11_2Ext,
                                   u6_2Ext, 13, 8>, AddrModeRel;
}

// multiclass for new-value store instructions with base + immediate offset.
// and MEMri operand.
multiclass ST_MEMri_Pbase_nv<string mnemonic, RegisterClass RC, bit isNot,
                          bit isPredNew> {
  let isPredicatedNew = isPredNew in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, RC: $src2),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_MEMri_Pred_nv<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
  let accessSize = ByteAccess in
    defm STrib: ST_MEMri_nv<"memb", "STrib", IntRegs, 11, 6>, AddrModeRel;

  let accessSize = HalfWordAccess in
    defm STrih: ST_MEMri_nv<"memh", "STrih", IntRegs, 12, 7>, AddrModeRel;

  let accessSize = WordAccess in
    defm STriw: ST_MEMri_nv<"memw", "STriw", IntRegs, 13, 8>, AddrModeRel;
}

//===----------------------------------------------------------------------===//
// Post increment store
// mem[bhwd](Rx++#s4:[0123])=Nt.new
//===----------------------------------------------------------------------===//

multiclass ST_PostInc_Pbase_nv<string mnemonic, RegisterClass RC, Operand ImmOp,
                            bit isNot, bit isPredNew> {
  let isPredicatedNew = isPredNew in
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
  let isPredicatedFalse = PredNot in {
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

let addrMode = PostInc, validSubTargets = HasV4SubT in {
defm POST_STbri: ST_PostInc_nv <"memb", "STrib", IntRegs, s4_0Imm>, AddrModeRel;
defm POST_SThri: ST_PostInc_nv <"memh", "STrih", IntRegs, s4_1Imm>, AddrModeRel;
defm POST_STwri: ST_PostInc_nv <"memw", "STriw", IntRegs, s4_2Imm>, AddrModeRel;
}

// memb(Rx++#s4:0:circ(Mu))=Nt.new
// memb(Rx++I:circ(Mu))=Nt.new
// memb(Rx++Mu)=Nt.new
// memb(Rx++Mu:brev)=Nt.new
// memh(Rx++#s4:1:circ(Mu))=Nt.new
// memh(Rx++I:circ(Mu))=Nt.new
// memh(Rx++Mu)=Nt.new
// memh(Rx++Mu:brev)=Nt.new

// memw(Rx++#s4:2:circ(Mu))=Nt.new
// memw(Rx++I:circ(Mu))=Nt.new
// memw(Rx++Mu)=Nt.new
// memw(Rx++Mu:brev)=Nt.new

//===----------------------------------------------------------------------===//
// NV/ST -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NV/J +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// multiclass/template class for the new-value compare jumps with the register
// operands.
//===----------------------------------------------------------------------===//

let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 11 in
class NVJrr_template<string mnemonic, bits<3> majOp, bit NvOpNum,
                      bit isNegCond, bit isTaken>
  : NVInst_V4<(outs),
    (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
    "if ("#!if(isNegCond, "!","")#mnemonic#
    "($src1"#!if(!eq(NvOpNum, 0),".new, ",", ")#
    "$src2"#!if(!eq(NvOpNum, 1),".new))","))")#" jump:"
    #!if(isTaken, "t","nt")#" $offset",
    []>, Requires<[HasV4T]> {

      bits<5> src1;
      bits<5> src2;
      bits<3> Ns;    // New-Value Operand
      bits<5> RegOp; // Non New-Value Operand
      bits<11> offset;

      let isBrTaken = !if(isTaken, "true", "false");
      let isPredicatedFalse = isNegCond;

      let Ns = !if(!eq(NvOpNum, 0), src1{2-0}, src2{2-0});
      let RegOp = !if(!eq(NvOpNum, 0), src2, src1);

      let IClass = 0b0010;
      let Inst{26} = 0b0;
      let Inst{25-23} = majOp;
      let Inst{22} = isNegCond;
      let Inst{18-16} = Ns;
      let Inst{13} = isTaken;
      let Inst{12-8} = RegOp;
      let Inst{21-20} = offset{10-9};
      let Inst{7-1} = offset{8-2};
}


multiclass NVJrr_cond<string mnemonic, bits<3> majOp, bit NvOpNum,
                       bit isNegCond> {
  // Branch not taken:
  def _nt_V4: NVJrr_template<mnemonic, majOp, NvOpNum, isNegCond, 0>;
  // Branch taken:
  def _t_V4: NVJrr_template<mnemonic, majOp, NvOpNum, isNegCond, 1>;
}

// NvOpNum = 0 -> First Operand is a new-value Register
// NvOpNum = 1 -> Second Operand is a new-value Register

multiclass NVJrr_base<string mnemonic, string BaseOp, bits<3> majOp,
                       bit NvOpNum> {
  let BaseOpcode = BaseOp#_NVJ in {
    defm _t_Jumpnv : NVJrr_cond<mnemonic, majOp, NvOpNum, 0>; // True cond
    defm _f_Jumpnv : NVJrr_cond<mnemonic, majOp, NvOpNum, 1>; // False cond
  }
}

// if ([!]cmp.eq(Ns.new,Rt)) jump:[n]t #r9:2
// if ([!]cmp.gt(Ns.new,Rt)) jump:[n]t #r9:2
// if ([!]cmp.gtu(Ns.new,Rt)) jump:[n]t #r9:2
// if ([!]cmp.gt(Rt,Ns.new)) jump:[n]t #r9:2
// if ([!]cmp.gtu(Rt,Ns.new)) jump:[n]t #r9:2

let isPredicated = 1, isBranch = 1, isNewValue = 1, isTerminator = 1,
  Defs = [PC], neverHasSideEffects = 1, validSubTargets = HasV4SubT in {
  defm CMPEQrr  : NVJrr_base<"cmp.eq",  "CMPEQ",  0b000, 0>, PredRel;
  defm CMPGTrr  : NVJrr_base<"cmp.gt",  "CMPGT",  0b001, 0>, PredRel;
  defm CMPGTUrr : NVJrr_base<"cmp.gtu", "CMPGTU", 0b010, 0>, PredRel;
  defm CMPLTrr  : NVJrr_base<"cmp.gt",  "CMPLT",  0b011, 1>, PredRel;
  defm CMPLTUrr : NVJrr_base<"cmp.gtu", "CMPLTU", 0b100, 1>, PredRel;
}

//===----------------------------------------------------------------------===//
// multiclass/template class for the new-value compare jumps instruction
// with a register and an unsigned immediate (U5) operand.
//===----------------------------------------------------------------------===//

let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 11 in
class NVJri_template<string mnemonic, bits<3> majOp, bit isNegCond,
                         bit isTaken>
  : NVInst_V4<(outs),
    (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset),
    "if ("#!if(isNegCond, "!","")#mnemonic#"($src1.new, #$src2)) jump:"
    #!if(isTaken, "t","nt")#" $offset",
    []>, Requires<[HasV4T]> {

      let isPredicatedFalse = isNegCond;
      let isBrTaken = !if(isTaken, "true", "false");

      bits<3> src1;
      bits<5> src2;
      bits<11> offset;

      let IClass = 0b0010;
      let Inst{26} = 0b1;
      let Inst{25-23} = majOp;
      let Inst{22} = isNegCond;
      let Inst{18-16} = src1;
      let Inst{13} = isTaken;
      let Inst{12-8} = src2;
      let Inst{21-20} = offset{10-9};
      let Inst{7-1} = offset{8-2};
}

multiclass NVJri_cond<string mnemonic, bits<3> majOp, bit isNegCond> {
  // Branch not taken:
  def _nt_V4: NVJri_template<mnemonic, majOp, isNegCond, 0>;
  // Branch taken:
  def _t_V4: NVJri_template<mnemonic, majOp, isNegCond, 1>;
}

multiclass NVJri_base<string mnemonic, string BaseOp, bits<3> majOp> {
  let BaseOpcode = BaseOp#_NVJri in {
    defm _t_Jumpnv : NVJri_cond<mnemonic, majOp, 0>; // True Cond
    defm _f_Jumpnv : NVJri_cond<mnemonic, majOp, 1>; // False cond
  }
}

// if ([!]cmp.eq(Ns.new,#U5)) jump:[n]t #r9:2
// if ([!]cmp.gt(Ns.new,#U5)) jump:[n]t #r9:2
// if ([!]cmp.gtu(Ns.new,#U5)) jump:[n]t #r9:2

let isPredicated = 1, isBranch = 1, isNewValue = 1, isTerminator = 1,
  Defs = [PC], neverHasSideEffects = 1, validSubTargets = HasV4SubT in {
  defm CMPEQri  : NVJri_base<"cmp.eq", "CMPEQ", 0b000>, PredRel;
  defm CMPGTri  : NVJri_base<"cmp.gt", "CMPGT", 0b001>, PredRel;
  defm CMPGTUri : NVJri_base<"cmp.gtu", "CMPGTU", 0b010>, PredRel;
}

//===----------------------------------------------------------------------===//
// multiclass/template class for the new-value compare jumps instruction
// with a register and an hardcoded 0/-1 immediate value.
//===----------------------------------------------------------------------===//

let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 11 in
class NVJ_ConstImm_template<string mnemonic, bits<3> majOp, string ImmVal,
                            bit isNegCond, bit isTaken>
  : NVInst_V4<(outs),
    (ins IntRegs:$src1, brtarget:$offset),
    "if ("#!if(isNegCond, "!","")#mnemonic
    #"($src1.new, #"#ImmVal#")) jump:"
    #!if(isTaken, "t","nt")#" $offset",
    []>, Requires<[HasV4T]> {

      let isPredicatedFalse = isNegCond;
      let isBrTaken = !if(isTaken, "true", "false");

      bits<3> src1;
      bits<11> offset;
      let IClass = 0b0010;
      let Inst{26} = 0b1;
      let Inst{25-23} = majOp;
      let Inst{22} = isNegCond;
      let Inst{18-16} = src1;
      let Inst{13} = isTaken;
      let Inst{21-20} = offset{10-9};
      let Inst{7-1} = offset{8-2};
}

multiclass NVJ_ConstImm_cond<string mnemonic, bits<3> majOp, string ImmVal,
                             bit isNegCond> {
  // Branch not taken:
  def _nt_V4: NVJ_ConstImm_template<mnemonic, majOp, ImmVal, isNegCond, 0>;
  // Branch taken:
  def _t_V4: NVJ_ConstImm_template<mnemonic, majOp, ImmVal, isNegCond, 1>;
}

multiclass NVJ_ConstImm_base<string mnemonic, string BaseOp, bits<3> majOp,
                             string ImmVal> {
  let BaseOpcode = BaseOp#_NVJ_ConstImm in {
  defm _t_Jumpnv : NVJ_ConstImm_cond<mnemonic, majOp, ImmVal, 0>; // True cond
  defm _f_Jumpnv : NVJ_ConstImm_cond<mnemonic, majOp, ImmVal, 1>; // False Cond
  }
}

// if ([!]tstbit(Ns.new,#0)) jump:[n]t #r9:2
// if ([!]cmp.eq(Ns.new,#-1)) jump:[n]t #r9:2
// if ([!]cmp.gt(Ns.new,#-1)) jump:[n]t #r9:2

let isPredicated = 1, isBranch = 1, isNewValue = 1, isTerminator=1,
  Defs = [PC], neverHasSideEffects = 1 in {
  defm TSTBIT0  : NVJ_ConstImm_base<"tstbit", "TSTBIT", 0b011, "0">, PredRel;
  defm CMPEQn1  : NVJ_ConstImm_base<"cmp.eq", "CMPEQ",  0b100, "-1">, PredRel;
  defm CMPGTn1  : NVJ_ConstImm_base<"cmp.gt", "CMPGT",  0b101, "-1">, PredRel;
}

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

def MEMOPIMM : SDNodeXForm<imm, [{
  // Call the transformation function XformM5ToU5Imm to get the negative
  // immediate's positive counterpart.
  int32_t imm = N->getSExtValue();
  return XformM5ToU5Imm(imm);
}]>;

def MEMOPIMM_HALF : SDNodeXForm<imm, [{
  // -1 .. -31 represented as 65535..65515
  // assigning to a short restores our desired signed value.
  // Call the transformation function XformM5ToU5Imm to get the negative
  // immediate's positive counterpart.
  int16_t imm = N->getSExtValue();
  return XformM5ToU5Imm(imm);
}]>;

def MEMOPIMM_BYTE : SDNodeXForm<imm, [{
  // -1 .. -31 represented as 255..235
  // assigning to a char restores our desired signed value.
  // Call the transformation function XformM5ToU5Imm to get the negative
  // immediate's positive counterpart.
  int8_t imm = N->getSExtValue();
  return XformM5ToU5Imm(imm);
}]>;

def SETMEMIMM : SDNodeXForm<imm, [{
   // Return the bit position we will set [0-31].
   // As an SDNode.
   int32_t imm = N->getSExtValue();
   return XformMskToBitPosU5Imm(imm);
}]>;

def CLRMEMIMM : SDNodeXForm<imm, [{
   // Return the bit position we will clear [0-31].
   // As an SDNode.
   // we bit negate the value first
   int32_t imm = ~(N->getSExtValue());
   return XformMskToBitPosU5Imm(imm);
}]>;

def SETMEMIMM_SHORT : SDNodeXForm<imm, [{
   // Return the bit position we will set [0-15].
   // As an SDNode.
   int16_t imm = N->getSExtValue();
   return XformMskToBitPosU4Imm(imm);
}]>;

def CLRMEMIMM_SHORT : SDNodeXForm<imm, [{
   // Return the bit position we will clear [0-15].
   // As an SDNode.
   // we bit negate the value first
   int16_t imm = ~(N->getSExtValue());
   return XformMskToBitPosU4Imm(imm);
}]>;

def SETMEMIMM_BYTE : SDNodeXForm<imm, [{
   // Return the bit position we will set [0-7].
   // As an SDNode.
   int8_t imm =  N->getSExtValue();
   return XformMskToBitPosU3Imm(imm);
}]>;

def CLRMEMIMM_BYTE : SDNodeXForm<imm, [{
   // Return the bit position we will clear [0-7].
   // As an SDNode.
   // we bit negate the value first
   int8_t imm = ~(N->getSExtValue());
   return XformMskToBitPosU3Imm(imm);
}]>;

//===----------------------------------------------------------------------===//
// Template class for MemOp instructions with the register value.
//===----------------------------------------------------------------------===//
class MemOp_rr_base <string opc, bits<2> opcBits, Operand ImmOp,
                     string memOp, bits<2> memOpBits> :
      MEMInst_V4<(outs),
                 (ins IntRegs:$base, ImmOp:$offset, IntRegs:$delta),
                 opc#"($base+#$offset)"#memOp#"$delta",
                 []>,
                 Requires<[HasV4T, UseMEMOP]> {

    bits<5> base;
    bits<5> delta;
    bits<32> offset;
    bits<6> offsetBits; // memb - u6:0 , memh - u6:1, memw - u6:2

    let offsetBits = !if (!eq(opcBits, 0b00), offset{5-0},
                     !if (!eq(opcBits, 0b01), offset{6-1},
                     !if (!eq(opcBits, 0b10), offset{7-2},0)));

    let IClass = 0b0011;
    let Inst{27-24} = 0b1110;
    let Inst{22-21} = opcBits;
    let Inst{20-16} = base;
    let Inst{13} = 0b0;
    let Inst{12-7} = offsetBits;
    let Inst{6-5} = memOpBits;
    let Inst{4-0} = delta;
}

//===----------------------------------------------------------------------===//
// Template class for MemOp instructions with the immediate value.
//===----------------------------------------------------------------------===//
class MemOp_ri_base <string opc, bits<2> opcBits, Operand ImmOp,
                     string memOp, bits<2> memOpBits> :
      MEMInst_V4 <(outs),
                  (ins IntRegs:$base, ImmOp:$offset, u5Imm:$delta),
                  opc#"($base+#$offset)"#memOp#"#$delta"
                  #!if(memOpBits{1},")", ""), // clrbit, setbit - include ')'
                  []>,
                  Requires<[HasV4T, UseMEMOP]> {

    bits<5> base;
    bits<5> delta;
    bits<32> offset;
    bits<6> offsetBits; // memb - u6:0 , memh - u6:1, memw - u6:2

    let offsetBits = !if (!eq(opcBits, 0b00), offset{5-0},
                     !if (!eq(opcBits, 0b01), offset{6-1},
                     !if (!eq(opcBits, 0b10), offset{7-2},0)));

    let IClass = 0b0011;
    let Inst{27-24} = 0b1111;
    let Inst{22-21} = opcBits;
    let Inst{20-16} = base;
    let Inst{13} = 0b0;
    let Inst{12-7} = offsetBits;
    let Inst{6-5} = memOpBits;
    let Inst{4-0} = delta;
}

// multiclass to define MemOp instructions with register operand.
multiclass MemOp_rr<string opc, bits<2> opcBits, Operand ImmOp> {
  def _ADD#NAME#_V4 : MemOp_rr_base <opc, opcBits, ImmOp, " += ", 0b00>; // add
  def _SUB#NAME#_V4 : MemOp_rr_base <opc, opcBits, ImmOp, " -= ", 0b01>; // sub
  def _AND#NAME#_V4 : MemOp_rr_base <opc, opcBits, ImmOp, " &= ", 0b10>; // and
  def _OR#NAME#_V4  : MemOp_rr_base <opc, opcBits, ImmOp, " |= ", 0b11>; // or
}

// multiclass to define MemOp instructions with immediate Operand.
multiclass MemOp_ri<string opc, bits<2> opcBits, Operand ImmOp> {
  def _ADD#NAME#_V4 : MemOp_ri_base <opc, opcBits, ImmOp, " += ", 0b00 >;
  def _SUB#NAME#_V4 : MemOp_ri_base <opc, opcBits, ImmOp, " -= ", 0b01 >;
  def _CLRBIT#NAME#_V4 : MemOp_ri_base<opc, opcBits, ImmOp, " =clrbit(", 0b10>;
  def _SETBIT#NAME#_V4 : MemOp_ri_base<opc, opcBits, ImmOp, " =setbit(", 0b11>;
}

multiclass MemOp_base <string opc, bits<2> opcBits, Operand ImmOp> {
  defm r : MemOp_rr <opc, opcBits, ImmOp>;
  defm i : MemOp_ri <opc, opcBits, ImmOp>;
}

// Define MemOp instructions.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0,
validSubTargets =HasV4SubT in {
  let opExtentBits = 6, accessSize = ByteAccess in
  defm MemOPb : MemOp_base <"memb", 0b00, u6_0Ext>;

  let opExtentBits = 7, accessSize = HalfWordAccess in
  defm MemOPh : MemOp_base <"memh", 0b01, u6_1Ext>;

  let opExtentBits = 8, accessSize = WordAccess in
  defm MemOPw : MemOp_base <"memw", 0b10, u6_2Ext>;
}

//===----------------------------------------------------------------------===//
// Multiclass to define 'Def Pats' for ALU operations on the memory
// Here value used for the ALU operation is an immediate value.
// mem[bh](Rs+#0) += #U5
// mem[bh](Rs+#u6) += #U5
//===----------------------------------------------------------------------===//

multiclass MemOpi_u5Pats <PatFrag ldOp, PatFrag stOp, PatLeaf ExtPred,
                          InstHexagon MI, SDNode OpNode> {
  let AddedComplexity = 180 in
  def : Pat < (stOp (OpNode (ldOp IntRegs:$addr), u5ImmPred:$addend),
                    IntRegs:$addr),
              (MI IntRegs:$addr, #0, u5ImmPred:$addend )>;

  let AddedComplexity = 190 in
  def : Pat <(stOp (OpNode (ldOp (add IntRegs:$base, ExtPred:$offset)),
                     u5ImmPred:$addend),
             (add IntRegs:$base, ExtPred:$offset)),
       (MI IntRegs:$base, ExtPred:$offset, u5ImmPred:$addend)>;
}

multiclass MemOpi_u5ALUOp<PatFrag ldOp, PatFrag stOp, PatLeaf ExtPred,
                          InstHexagon addMI, InstHexagon subMI> {
  defm : MemOpi_u5Pats<ldOp, stOp, ExtPred, addMI, add>;
  defm : MemOpi_u5Pats<ldOp, stOp, ExtPred, subMI, sub>;
}

multiclass MemOpi_u5ExtType<PatFrag ldOpByte, PatFrag ldOpHalf > {
  // Half Word
  defm : MemOpi_u5ALUOp <ldOpHalf, truncstorei16, u6_1ExtPred,
                         MemOPh_ADDi_V4, MemOPh_SUBi_V4>;
  // Byte
  defm : MemOpi_u5ALUOp <ldOpByte, truncstorei8, u6ExtPred,
                         MemOPb_ADDi_V4, MemOPb_SUBi_V4>;
}

let Predicates = [HasV4T, UseMEMOP] in {
  defm : MemOpi_u5ExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOpi_u5ExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOpi_u5ExtType<extloadi8,  extloadi16>;  // any extend

  // Word
  defm : MemOpi_u5ALUOp <load, store, u6_2ExtPred, MemOPw_ADDi_V4,
                         MemOPw_SUBi_V4>;
}

//===----------------------------------------------------------------------===//
// multiclass to define 'Def Pats' for ALU operations on the memory.
// Here value used for the ALU operation is a negative value.
// mem[bh](Rs+#0) += #m5
// mem[bh](Rs+#u6) += #m5
//===----------------------------------------------------------------------===//

multiclass MemOpi_m5Pats <PatFrag ldOp, PatFrag stOp, PatLeaf extPred,
                          PatLeaf immPred, ComplexPattern addrPred,
                          SDNodeXForm xformFunc, InstHexagon MI> {
  let AddedComplexity = 190 in
  def : Pat <(stOp (add (ldOp IntRegs:$addr), immPred:$subend),
                   IntRegs:$addr),
             (MI IntRegs:$addr, #0, (xformFunc immPred:$subend) )>;

  let AddedComplexity = 195 in
  def : Pat<(stOp (add (ldOp (add IntRegs:$base, extPred:$offset)),
                       immPred:$subend),
                  (add IntRegs:$base, extPred:$offset)),
            (MI IntRegs:$base, extPred:$offset, (xformFunc immPred:$subend))>;
}

multiclass MemOpi_m5ExtType<PatFrag ldOpByte, PatFrag ldOpHalf > {
  // Half Word
  defm : MemOpi_m5Pats <ldOpHalf, truncstorei16, u6_1ExtPred, m5HImmPred,
                        ADDRriU6_1, MEMOPIMM_HALF, MemOPh_SUBi_V4>;
  // Byte
  defm : MemOpi_m5Pats <ldOpByte, truncstorei8, u6ExtPred, m5BImmPred,
                        ADDRriU6_0, MEMOPIMM_BYTE, MemOPb_SUBi_V4>;
}

let Predicates = [HasV4T, UseMEMOP] in {
  defm : MemOpi_m5ExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOpi_m5ExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOpi_m5ExtType<extloadi8,  extloadi16>;  // any extend

  // Word
  defm : MemOpi_m5Pats <load, store, u6_2ExtPred, m5ImmPred,
                          ADDRriU6_2, MEMOPIMM, MemOPw_SUBi_V4>;
}

//===----------------------------------------------------------------------===//
// Multiclass to define 'def Pats' for bit operations on the memory.
// mem[bhw](Rs+#0) = [clrbit|setbit](#U5)
// mem[bhw](Rs+#u6) = [clrbit|setbit](#U5)
//===----------------------------------------------------------------------===//

multiclass MemOpi_bitPats <PatFrag ldOp, PatFrag stOp, PatLeaf immPred,
                     PatLeaf extPred, ComplexPattern addrPred,
                     SDNodeXForm xformFunc, InstHexagon MI, SDNode OpNode> {

  // mem[bhw](Rs+#u6:[012]) = [clrbit|setbit](#U5)
  let AddedComplexity = 250 in
  def : Pat<(stOp (OpNode (ldOp (add IntRegs:$base, extPred:$offset)),
                          immPred:$bitend),
                  (add IntRegs:$base, extPred:$offset)),
            (MI IntRegs:$base, extPred:$offset, (xformFunc immPred:$bitend))>;

  // mem[bhw](Rs+#0) = [clrbit|setbit](#U5)
  let AddedComplexity = 225 in
  def : Pat <(stOp (OpNode (ldOp addrPred:$addr), immPred:$bitend),
                   addrPred:$addr),
             (MI IntRegs:$addr, #0, (xformFunc immPred:$bitend))>;
}

multiclass MemOpi_bitExtType<PatFrag ldOpByte, PatFrag ldOpHalf > {
  // Byte - clrbit
  defm : MemOpi_bitPats<ldOpByte, truncstorei8, Clr3ImmPred, u6ExtPred,
                       ADDRriU6_0, CLRMEMIMM_BYTE, MemOPb_CLRBITi_V4, and>;
  // Byte - setbit
  defm : MemOpi_bitPats<ldOpByte, truncstorei8, Set3ImmPred,  u6ExtPred,
                       ADDRriU6_0, SETMEMIMM_BYTE, MemOPb_SETBITi_V4, or>;
  // Half Word - clrbit
  defm : MemOpi_bitPats<ldOpHalf, truncstorei16, Clr4ImmPred, u6_1ExtPred,
                       ADDRriU6_1, CLRMEMIMM_SHORT, MemOPh_CLRBITi_V4, and>;
  // Half Word - setbit
  defm : MemOpi_bitPats<ldOpHalf, truncstorei16, Set4ImmPred, u6_1ExtPred,
                       ADDRriU6_1, SETMEMIMM_SHORT, MemOPh_SETBITi_V4, or>;
}

let Predicates = [HasV4T, UseMEMOP] in {
  // mem[bh](Rs+#0) = [clrbit|setbit](#U5)
  // mem[bh](Rs+#u6:[01]) = [clrbit|setbit](#U5)
  defm : MemOpi_bitExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOpi_bitExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOpi_bitExtType<extloadi8,  extloadi16>;  // any extend

  // memw(Rs+#0) = [clrbit|setbit](#U5)
  // memw(Rs+#u6:2) = [clrbit|setbit](#U5)
  defm : MemOpi_bitPats<load, store, Clr5ImmPred, u6_2ExtPred, ADDRriU6_2,
                       CLRMEMIMM, MemOPw_CLRBITi_V4, and>;
  defm : MemOpi_bitPats<load, store, Set5ImmPred, u6_2ExtPred, ADDRriU6_2,
                       SETMEMIMM, MemOPw_SETBITi_V4, or>;
}

//===----------------------------------------------------------------------===//
// Multiclass to define 'def Pats' for ALU operations on the memory
// where addend is a register.
// mem[bhw](Rs+#0) [+-&|]= Rt
// mem[bhw](Rs+#U6:[012]) [+-&|]= Rt
//===----------------------------------------------------------------------===//

multiclass MemOpr_Pats <PatFrag ldOp, PatFrag stOp, ComplexPattern addrPred,
                     PatLeaf extPred, InstHexagon MI, SDNode OpNode> {
  let AddedComplexity = 141 in
  // mem[bhw](Rs+#0) [+-&|]= Rt
  def : Pat <(stOp (OpNode (ldOp addrPred:$addr), (i32 IntRegs:$addend)),
                   addrPred:$addr),
             (MI IntRegs:$addr, #0, (i32 IntRegs:$addend) )>;

  // mem[bhw](Rs+#U6:[012]) [+-&|]= Rt
  let AddedComplexity = 150 in
  def : Pat <(stOp (OpNode (ldOp (add IntRegs:$base, extPred:$offset)),
                           (i32 IntRegs:$orend)),
                   (add IntRegs:$base, extPred:$offset)),
             (MI IntRegs:$base, extPred:$offset, (i32 IntRegs:$orend) )>;
}

multiclass MemOPr_ALUOp<PatFrag ldOp, PatFrag stOp,
                        ComplexPattern addrPred, PatLeaf extPred,
                        InstHexagon addMI, InstHexagon subMI,
                        InstHexagon andMI, InstHexagon orMI > {

  defm : MemOpr_Pats <ldOp, stOp, addrPred, extPred, addMI, add>;
  defm : MemOpr_Pats <ldOp, stOp, addrPred, extPred, subMI, sub>;
  defm : MemOpr_Pats <ldOp, stOp, addrPred, extPred, andMI, and>;
  defm : MemOpr_Pats <ldOp, stOp, addrPred, extPred, orMI,  or>;
}

multiclass MemOPr_ExtType<PatFrag ldOpByte, PatFrag ldOpHalf > {
  // Half Word
  defm : MemOPr_ALUOp <ldOpHalf, truncstorei16, ADDRriU6_1, u6_1ExtPred,
                       MemOPh_ADDr_V4, MemOPh_SUBr_V4,
                       MemOPh_ANDr_V4, MemOPh_ORr_V4>;
  // Byte
  defm : MemOPr_ALUOp <ldOpByte, truncstorei8, ADDRriU6_0, u6ExtPred,
                       MemOPb_ADDr_V4, MemOPb_SUBr_V4,
                       MemOPb_ANDr_V4, MemOPb_ORr_V4>;
}

// Define 'def Pats' for MemOps with register addend.
let Predicates = [HasV4T, UseMEMOP] in {
  // Byte, Half Word
  defm : MemOPr_ExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOPr_ExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOPr_ExtType<extloadi8,  extloadi16>;  // any extend
  // Word
  defm : MemOPr_ALUOp <load, store, ADDRriU6_2, u6_2ExtPred, MemOPw_ADDr_V4,
                       MemOPw_SUBr_V4, MemOPw_ANDr_V4, MemOPw_ORr_V4 >;
}

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
      (JMP_f (CMPbEQri_V4 (i32 IntRegs:$src1), u8ImmPred:$src2),
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
  Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_V4 : LD0Inst<(outs), (ins),
            "dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

// Restore registers and dealloc return function call.
let isCall = 1, isBarrier = 1, isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC] in {
let validSubTargets = HasV4SubT in
  def RESTORE_DEALLOC_RET_JMP_V4 : JInst<(outs),
                                   (ins calltarget:$dst),
             "jump $dst",
             []>,
             Requires<[HasV4T]>;
}

// Restore registers and dealloc frame before a tail call.
let isCall = 1, isBarrier = 1,
  Defs = [R29, R30, R31, PC] in {
let validSubTargets = HasV4SubT in
  def RESTORE_DEALLOC_BEFORE_TAILCALL_V4 : JInst<(outs),
                                           (ins calltarget:$dst),
             "call $dst",
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
    Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1,
    isPredicated = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_cPt_V4 : LD0Inst<(outs),
                           (ins PredRegs:$src1),
            "if ($src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1,
    isPredicated = 1, isPredicatedFalse = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_cNotPt_V4 : LD0Inst<(outs), (ins PredRegs:$src1),
            "if (!$src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1,
    isPredicated = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_cdnPnt_V4 : LD0Inst<(outs), (ins PredRegs:$src1),
            "if ($src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1,
    isPredicated = 1, isPredicatedFalse = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_cNotdnPnt_V4 : LD0Inst<(outs), (ins PredRegs:$src1),
            "if (!$src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:t
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1,
    isPredicated = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_cdnPt_V4 : LD0Inst<(outs), (ins PredRegs:$src1),
            "if ($src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}

// if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R30], neverHasSideEffects = 1,
    isPredicated = 1, isPredicatedFalse = 1 in {
let validSubTargets = HasV4SubT in
  def DEALLOC_RET_cNotdnPt_V4 : LD0Inst<(outs), (ins PredRegs:$src1),
            "if (!$src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}

// Load/Store with absolute addressing mode
// memw(#u6)=Rt

multiclass ST_Abs_Predbase<string mnemonic, RegisterClass RC, bit isNot,
                           bit isPredNew> {
  let isPredicatedNew = isPredNew in
  def NAME#_V4 : STInst2<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$absaddr, RC: $src2),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"(##$absaddr) = $src2",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Abs_Pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
            (ins u0AlwaysExt:$absaddr, RC:$src),
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
  let isPredicatedNew = isPredNew in
  def NAME#_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, u0AlwaysExt:$absaddr, RC: $src2),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#mnemonic#"(##$absaddr) = $src2.new",
            []>,
            Requires<[HasV4T]>;
}

multiclass ST_Abs_Pred_nv<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
            (ins u0AlwaysExt:$absaddr, RC:$src),
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
  let accessSize = ByteAccess in
    defm STrib_abs : ST_Abs<"memb", "STrib", IntRegs>,
                     ST_Abs_nv<"memb", "STrib", IntRegs>, AddrModeRel;

  let accessSize = HalfWordAccess in
    defm STrih_abs : ST_Abs<"memh", "STrih", IntRegs>,
                     ST_Abs_nv<"memh", "STrih", IntRegs>, AddrModeRel;

  let accessSize = WordAccess in
    defm STriw_abs : ST_Abs<"memw", "STriw", IntRegs>,
                     ST_Abs_nv<"memw", "STriw", IntRegs>, AddrModeRel;

  let accessSize = DoubleWordAccess, isNVStorable = 0 in
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

//===----------------------------------------------------------------------===//
// multiclass for store instructions with GP-relative addressing mode.
// mem[bhwd](#global)=Rt
// if ([!]Pv[.new]) mem[bhwd](##global) = Rt
//===----------------------------------------------------------------------===//
let mayStore = 1, isNVStorable = 1 in
multiclass ST_GP<string mnemonic, string BaseOp, RegisterClass RC> {
  let BaseOpcode = BaseOp, isPredicable = 1 in
  def NAME#_V4 : STInst2<(outs),
          (ins globaladdress:$global, RC:$src),
          mnemonic#"(#$global) = $src",
          []>;

  // When GP-relative instructions are predicated, their addressing mode is
  // changed to absolute and they are always constant extended.
  let BaseOpcode = BaseOp, isExtended = 1, opExtendable = 1,
  isPredicated = 1 in {
    defm Pt : ST_Abs_Pred <mnemonic, RC, 0>;
    defm NotPt : ST_Abs_Pred <mnemonic, RC, 1>;
  }
}

let mayStore = 1, isNVStore = 1 in
multiclass ST_GP_nv<string mnemonic, string BaseOp, RegisterClass RC> {
  let BaseOpcode = BaseOp, isPredicable = 1 in
  def NAME#_nv_V4 : NVInst_V4<(outs),
          (ins u0AlwaysExt:$global, RC:$src),
          mnemonic#"(#$global) = $src.new",
          []>,
          Requires<[HasV4T]>;

  // When GP-relative instructions are predicated, their addressing mode is
  // changed to absolute and they are always constant extended.
  let BaseOpcode = BaseOp, isExtended = 1, opExtendable = 1,
  isPredicated = 1 in {
    defm Pt : ST_Abs_Pred_nv<mnemonic, RC, 0>;
    defm NotPt : ST_Abs_Pred_nv<mnemonic, RC, 1>;
  }
}

let validSubTargets = HasV4SubT, neverHasSideEffects = 1 in {
  let isNVStorable = 0 in
  defm STd_GP : ST_GP <"memd", "STd_GP", DoubleRegs>, PredNewRel;

  defm STb_GP : ST_GP<"memb",  "STb_GP", IntRegs>,
                ST_GP_nv<"memb", "STb_GP", IntRegs>, NewValueRel;
  defm STh_GP : ST_GP<"memh",  "STh_GP", IntRegs>,
                ST_GP_nv<"memh", "STh_GP", IntRegs>, NewValueRel;
  defm STw_GP : ST_GP<"memw",  "STw_GP", IntRegs>,
                ST_GP_nv<"memw", "STw_GP", IntRegs>, NewValueRel;
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
           (STd_GP_V4 tglobaladdr:$global, (i64 DoubleRegs:$src1))>;

// 8 bit atomic store
def : Pat < (atomic_store_8 (HexagonCONST32_GP tglobaladdr:$global),
                            (i32 IntRegs:$src1)),
            (STb_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from store(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
          (HexagonCONST32_GP tglobaladdr:$global)),
          (STb_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from "i1 = constant<-1>; memw(CONST32(#foo)) = i1"
//       to "r0 = 1; memw(#foo) = r0"
let AddedComplexity = 100 in
def : Pat<(store (i1 -1), (HexagonCONST32_GP tglobaladdr:$global)),
          (STb_GP_V4 tglobaladdr:$global, (TFRI 1))>;

def : Pat<(atomic_store_16 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (STh_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from store(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                         (HexagonCONST32_GP tglobaladdr:$global)),
          (STh_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>;

// 32 bit atomic store
def : Pat<(atomic_store_32 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (STw_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from store(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32_GP tglobaladdr:$global)),
          (STw_GP_V4 tglobaladdr:$global, (i32 IntRegs:$src1))>;

//===----------------------------------------------------------------------===//
// Multiclass for the load instructions with absolute addressing mode.
//===----------------------------------------------------------------------===//
multiclass LD_Abs_Predbase<string mnemonic, RegisterClass RC, bit isNot,
                           bit isPredNew> {
  let isPredicatedNew = isPredNew in
  def NAME : LDInst2<(outs RC:$dst),
            (ins PredRegs:$src1, u0AlwaysExt:$absaddr),
            !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
            ") ")#"$dst = "#mnemonic#"(##$absaddr)",
            []>,
            Requires<[HasV4T]>;
}

multiclass LD_Abs_Pred<string mnemonic, RegisterClass RC, bit PredNot> {
  let isPredicatedFalse = PredNot in {
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
            (ins u0AlwaysExt:$absaddr),
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
  let accessSize = ByteAccess in {
    defm LDrib_abs  : LD_Abs<"memb", "LDrib", IntRegs>, AddrModeRel;
    defm LDriub_abs : LD_Abs<"memub", "LDriub", IntRegs>, AddrModeRel;
  }
  let accessSize = HalfWordAccess in {
    defm LDrih_abs  : LD_Abs<"memh", "LDrih", IntRegs>, AddrModeRel;
    defm LDriuh_abs : LD_Abs<"memuh", "LDriuh", IntRegs>, AddrModeRel;
  }
  let accessSize = WordAccess in
    defm LDriw_abs  : LD_Abs<"memw", "LDriw", IntRegs>, AddrModeRel;

  let accessSize = DoubleWordAccess in
    defm LDrid_abs : LD_Abs<"memd",  "LDrid", DoubleRegs>, AddrModeRel;
}

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(i32 (load (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriw_abs_V4 tglobaladdr: $absaddr)>;

def : Pat<(i32 (sextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDrib_abs_V4 tglobaladdr:$absaddr)>;

def : Pat<(i32 (zextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriub_abs_V4 tglobaladdr:$absaddr)>;

def : Pat<(i32 (sextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDrih_abs_V4 tglobaladdr:$absaddr)>;

def : Pat<(i32 (zextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (LDriuh_abs_V4 tglobaladdr:$absaddr)>;
}

//===----------------------------------------------------------------------===//
// multiclass for load instructions with GP-relative addressing mode.
// Rx=mem[bhwd](##global)
// if ([!]Pv[.new]) Rx=mem[bhwd](##global)
//===----------------------------------------------------------------------===//
let neverHasSideEffects = 1, validSubTargets = HasV4SubT in
multiclass LD_GP<string mnemonic, string BaseOp, RegisterClass RC> {
  let BaseOpcode = BaseOp in {
    let isPredicable = 1 in
    def NAME#_V4 : LDInst2<(outs RC:$dst),
            (ins globaladdress:$global),
            "$dst = "#mnemonic#"(#$global)",
            []>;

    let isExtended = 1, opExtendable = 2, isPredicated = 1 in {
      defm Pt_V4 : LD_Abs_Pred<mnemonic, RC, 0>;
      defm NotPt_V4 : LD_Abs_Pred<mnemonic, RC, 1>;
    }
  }
}

defm LDd_GP  : LD_GP<"memd",  "LDd_GP",  DoubleRegs>, PredNewRel;
defm LDb_GP  : LD_GP<"memb",  "LDb_GP",  IntRegs>, PredNewRel;
defm LDub_GP : LD_GP<"memub", "LDub_GP", IntRegs>, PredNewRel;
defm LDh_GP  : LD_GP<"memh",  "LDh_GP",  IntRegs>, PredNewRel;
defm LDuh_GP : LD_GP<"memuh", "LDuh_GP", IntRegs>, PredNewRel;
defm LDw_GP  : LD_GP<"memw",  "LDw_GP",  IntRegs>, PredNewRel;

def : Pat <(atomic_load_64 (HexagonCONST32_GP tglobaladdr:$global)),
           (i64 (LDd_GP_V4 tglobaladdr:$global))>;

def : Pat <(atomic_load_32 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDw_GP_V4 tglobaladdr:$global))>;

def : Pat <(atomic_load_16 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDuh_GP_V4 tglobaladdr:$global))>;

def : Pat <(atomic_load_8 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memw(#foo + 0)
let AddedComplexity = 100 in
def : Pat <(i64 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i64 (LDd_GP_V4 tglobaladdr:$global))>;

// Map from Pd = load(globaladdress) -> Rd = memb(globaladdress), Pd = Rd
let AddedComplexity = 100 in
def : Pat <(i1 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i1 (TFR_PdRs (i32 (LDb_GP_V4 tglobaladdr:$global))))>;

// When the Interprocedural Global Variable optimizer realizes that a certain
// global variable takes only two constant values, it shrinks the global to
// a boolean. Catch those loads here in the following 3 patterns.
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>;

let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDb_GP_V4 tglobaladdr:$global))>;

let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memub(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDub_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDh_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDh_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memuh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDuh_GP_V4 tglobaladdr:$global))>;

// Map from load(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (LDw_GP_V4 tglobaladdr:$global))>;


// Transfer global address into a register
let isExtended = 1, opExtendable = 1, AddedComplexity=50, isMoveImm = 1,
isAsCheapAsAMove = 1, isReMaterializable = 1, validSubTargets = HasV4SubT in
def TFRI_V4 : ALU32_ri<(outs IntRegs:$dst), (ins s16Ext:$src1),
           "$dst = #$src1",
           [(set IntRegs:$dst, (HexagonCONST32 tglobaladdr:$src1))]>,
           Requires<[HasV4T]>;

// Transfer a block address into a register
def : Pat<(HexagonCONST32_GP tblockaddress:$src1),
          (TFRI_V4 tblockaddress:$src1)>,
          Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50,
neverHasSideEffects = 1, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                           (ins PredRegs:$src1, s16Ext:$src2),
           "if($src1) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50, isPredicatedFalse = 1,
neverHasSideEffects = 1, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, s16Ext:$src2),
           "if(!$src1) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50,
neverHasSideEffects = 1, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cdnPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                             (ins PredRegs:$src1, s16Ext:$src2),
           "if($src1.new) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50, isPredicatedFalse = 1,
neverHasSideEffects = 1, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cdnNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                                (ins PredRegs:$src1, s16Ext:$src2),
           "if(!$src1.new) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let AddedComplexity = 50, Predicates = [HasV4T] in
def : Pat<(HexagonCONST32_GP tglobaladdr:$src1),
           (TFRI_V4 tglobaladdr:$src1)>,
           Requires<[HasV4T]>;


// Load - Indirect with long offset: These instructions take global address
// as an operand
let isExtended = 1, opExtendable = 3, AddedComplexity = 40,
validSubTargets = HasV4SubT in
def LDrid_ind_lo_V4 : LDInst<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdressExt:$offset),
            "$dst=memd($src1<<#$src2+##$offset)",
            [(set (i64 DoubleRegs:$dst),
                  (load (add (shl IntRegs:$src1, u2ImmPred:$src2),
                        (HexagonCONST32 tglobaladdr:$offset))))]>,
            Requires<[HasV4T]>;

let AddedComplexity = 40 in
multiclass LD_indirect_lo<string OpcStr, PatFrag OpNode> {
let isExtended = 1, opExtendable = 3, validSubTargets = HasV4SubT in
  def _lo_V4 : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, u2Imm:$src2, globaladdressExt:$offset),
            !strconcat("$dst = ",
            !strconcat(OpcStr, "($src1<<#$src2+##$offset)")),
            [(set IntRegs:$dst,
                  (i32 (OpNode (add (shl IntRegs:$src1, u2ImmPred:$src2),
                          (HexagonCONST32 tglobaladdr:$offset)))))]>,
            Requires<[HasV4T]>;
}

defm LDrib_ind : LD_indirect_lo<"memb", sextloadi8>;
defm LDriub_ind : LD_indirect_lo<"memub", zextloadi8>;
defm LDriub_ind_anyext : LD_indirect_lo<"memub", extloadi8>;
defm LDrih_ind : LD_indirect_lo<"memh", sextloadi16>;
defm LDriuh_ind : LD_indirect_lo<"memuh", zextloadi16>;
defm LDriuh_ind_anyext : LD_indirect_lo<"memuh", extloadi16>;
defm LDriw_ind : LD_indirect_lo<"memw", load>;

let AddedComplexity = 40 in
def : Pat <(i32 (sextloadi8 (add IntRegs:$src1,
                                 (NumUsesBelowThresCONST32 tglobaladdr:$offset)))),
           (i32 (LDrib_ind_lo_V4 IntRegs:$src1, 0, tglobaladdr:$offset))>,
           Requires<[HasV4T]>;

let AddedComplexity = 40 in
def : Pat <(i32 (zextloadi8 (add IntRegs:$src1,
                                 (NumUsesBelowThresCONST32 tglobaladdr:$offset)))),
           (i32 (LDriub_ind_lo_V4 IntRegs:$src1, 0, tglobaladdr:$offset))>,
           Requires<[HasV4T]>;

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(truncstorei8 (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (STrib_abs_V4 u0AlwaysExtPred:$src2, IntRegs: $src1)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (STrih_abs_V4 u0AlwaysExtPred:$src2, IntRegs: $src1)>;

def : Pat<(store (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (STriw_abs_V4 u0AlwaysExtPred:$src2, IntRegs: $src1)>;
}

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(i32 (load u0AlwaysExtPred:$src)),
          (LDriw_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (sextloadi8 u0AlwaysExtPred:$src)),
          (LDrib_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (zextloadi8 u0AlwaysExtPred:$src)),
          (LDriub_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (sextloadi16 u0AlwaysExtPred:$src)),
          (LDrih_abs_V4 u0AlwaysExtPred:$src)>;

def : Pat<(i32 (zextloadi16 u0AlwaysExtPred:$src)),
          (LDriuh_abs_V4 u0AlwaysExtPred:$src)>;
}

// Indexed store word - global address.
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10 in
def STriw_offset_ext_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_2Imm:$src2, globaladdress:$src3),
            "memw($src1+#$src2) = ##$src3",
            [(store (HexagonCONST32 tglobaladdr:$src3),
                    (add IntRegs:$src1, u6_2ImmPred:$src2))]>,
            Requires<[HasV4T]>;

def : Pat<(i64 (ctlz (i64 DoubleRegs:$src1))),
          (i64 (COMBINE_Ir_V4 (i32 0), (i32 (CTLZ64_rr DoubleRegs:$src1))))>,
          Requires<[HasV4T]>;

def : Pat<(i64 (cttz (i64 DoubleRegs:$src1))),
          (i64 (COMBINE_Ir_V4 (i32 0), (i32 (CTTZ64_rr DoubleRegs:$src1))))>,
          Requires<[HasV4T]>;


// i8 -> i64 loads
// We need a complexity of 120 here to overide preceeding handling of
// zextloadi8.
let Predicates = [HasV4T], AddedComplexity = 120 in {
def:  Pat <(i64 (extloadi8 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (COMBINE_Ir_V4 0, (LDrib_abs_V4 tglobaladdr:$addr)))>;

def:  Pat <(i64 (zextloadi8 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (COMBINE_Ir_V4 0, (LDriub_abs_V4 tglobaladdr:$addr)))>;

def:  Pat <(i64 (sextloadi8 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (SXTW (LDrib_abs_V4 tglobaladdr:$addr)))>;

def:  Pat <(i64 (extloadi8 FoldGlobalAddr:$addr)),
      (i64 (COMBINE_Ir_V4 0, (LDrib_abs_V4 FoldGlobalAddr:$addr)))>;

def:  Pat <(i64 (zextloadi8 FoldGlobalAddr:$addr)),
      (i64 (COMBINE_Ir_V4 0, (LDriub_abs_V4 FoldGlobalAddr:$addr)))>;

def:  Pat <(i64 (sextloadi8 FoldGlobalAddr:$addr)),
      (i64 (SXTW (LDrib_abs_V4 FoldGlobalAddr:$addr)))>;
}
// i16 -> i64 loads
// We need a complexity of 120 here to overide preceeding handling of
// zextloadi16.
let AddedComplexity = 120 in {
def:  Pat <(i64 (extloadi16 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (COMBINE_Ir_V4 0, (LDrih_abs_V4 tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi16 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (COMBINE_Ir_V4 0, (LDriuh_abs_V4 tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi16 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (SXTW (LDrih_abs_V4 tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (extloadi16 FoldGlobalAddr:$addr)),
      (i64 (COMBINE_Ir_V4 0, (LDrih_abs_V4 FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi16 FoldGlobalAddr:$addr)),
      (i64 (COMBINE_Ir_V4 0, (LDriuh_abs_V4 FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi16 FoldGlobalAddr:$addr)),
      (i64 (SXTW (LDrih_abs_V4 FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;
}
// i32->i64 loads
// We need a complexity of 120 here to overide preceeding handling of
// zextloadi32.
let AddedComplexity = 120 in {
def:  Pat <(i64 (extloadi32 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (COMBINE_Ir_V4 0, (LDriw_abs_V4 tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi32 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (COMBINE_Ir_V4 0, (LDriw_abs_V4 tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi32 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (SXTW (LDriw_abs_V4 tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (extloadi32 FoldGlobalAddr:$addr)),
      (i64 (COMBINE_Ir_V4 0, (LDriw_abs_V4 FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi32 FoldGlobalAddr:$addr)),
      (i64 (COMBINE_Ir_V4 0, (LDriw_abs_V4 FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi32 FoldGlobalAddr:$addr)),
      (i64 (SXTW (LDriw_abs_V4 FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;
}

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
