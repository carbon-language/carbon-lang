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

// Sign extend.

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

// Zero exten.

let neverHasSideEffects = 1 in
def ZXTB_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTB_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTB_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTB_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = zxtb($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTH_cPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTH_cNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTH_cdnPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if ($src1.new) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;

let neverHasSideEffects = 1 in
def ZXTH_cdnNotPt_V4 : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2),
            "if (!$src1.new) $dst = zxth($src2)",
            []>,
            Requires<[HasV4T]>;


//===----------------------------------------------------------------------===//
// ALU32 -
//===----------------------------------------------------------------------===//



//===----------------------------------------------------------------------===//
// LD +
//===----------------------------------------------------------------------===//
///
/// Make sure that in post increment load, the first operand is always the post
/// increment operand.
///
//// Load doubleword.
// Rdd=memd(Re=#U6)

// Rdd=memd(Rs+Rt<<#u2)
// Special case pattern for indexed load without offset which is easier to
// match. AddedComplexity of this pattern should be lower than base+offset load
// and lower yet than the more generic version with offset/shift below
// Similar approach is taken for all other base+index loads.
let AddedComplexity = 10, isPredicable = 1 in
def LDrid_indexed_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memd($src1+$src2<<#0)",
                    [(set DoubleRegs:$dst, (load (add IntRegs:$src1,
                                                      IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDrid_indexed_shl_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memd($src1+$src2<<#$offset)",
                    [(set DoubleRegs:$dst, (load (add IntRegs:$src1,
                                                 (shl IntRegs:$src2,
                                                      u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

//// Load doubleword conditionally.
// if ([!]Pv[.new]) Rd=memd(Rs+Rt<<#u2)
// if (Pv) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrid_indexed_cPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrid_indexed_cdnPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrid_indexed_cNotPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrid_indexed_cdnNotPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memd($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrid_indexed_shl_cPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrid_indexed_shl_cdnPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrid_indexed_shl_cNotPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memd(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrid_indexed_shl_cdnNotPt_V4 : LDInst<(outs DoubleRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memd($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rdd=memd(Rt<<#u2+#U6)

//// Load byte.
// Rd=memb(Re=#U6)

// Rd=memb(Rs+Rt<<#u2)
let AddedComplexity = 10, isPredicable = 1 in
def LDrib_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memb($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (sextloadi8 (add IntRegs:$src1,
                                                         IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriub_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memub($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (zextloadi8 (add IntRegs:$src1,
                                                         IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriub_ae_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memub($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (extloadi8 (add IntRegs:$src1,
                                                        IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDrib_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memb($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst,
                          (sextloadi8 (add IntRegs:$src1,
                                           (shl IntRegs:$src2,
                                                u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriub_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memub($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst,
                          (zextloadi8 (add IntRegs:$src1,
                                           (shl IntRegs:$src2,
                                                u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriub_ae_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memub($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst, (extloadi8 (add IntRegs:$src1,
                                                   (shl IntRegs:$src2,
                                                        u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

//// Load byte conditionally.
// if ([!]Pv[.new]) Rd=memb(Rs+Rt<<#u2)
// if (Pv) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrib_indexed_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrib_indexed_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrib_indexed_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrib_indexed_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memb($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrib_indexed_shl_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrib_indexed_shl_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrib_indexed_shl_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memb(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrib_indexed_shl_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memb($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

//// Load unsigned byte conditionally.
// if ([!]Pv[.new]) Rd=memub(Rs+Rt<<#u2)
// if (Pv) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriub_indexed_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriub_indexed_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriub_indexed_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriub_indexed_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memub($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriub_indexed_shl_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriub_indexed_shl_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriub_indexed_shl_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memub(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriub_indexed_shl_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memub($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rd=memb(Rt<<#u2+#U6)

//// Load halfword
// Rd=memh(Re=#U6)

// Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 10, isPredicable = 1 in
def LDrih_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memh($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (sextloadi16 (add IntRegs:$src1,
                                                          IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriuh_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memuh($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (zextloadi16 (add IntRegs:$src1,
                                                          IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 10, isPredicable = 1 in
def LDriuh_ae_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memuh($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (extloadi16 (add IntRegs:$src1,
                                                         IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

// Rd=memh(Rs+Rt<<#u2)
let AddedComplexity = 40, isPredicable = 1 in
def LDrih_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memh($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst,
                          (sextloadi16 (add IntRegs:$src1,
                                            (shl IntRegs:$src2,
                                                 u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriuh_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memuh($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst,
                          (zextloadi16 (add IntRegs:$src1,
                                            (shl IntRegs:$src2,
                                                 u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

let AddedComplexity = 40, isPredicable = 1 in
def LDriuh_ae_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memuh($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst,
                          (extloadi16 (add IntRegs:$src1,
                                           (shl IntRegs:$src2,
                                                u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

//// Load halfword conditionally.
// if ([!]Pv[.new]) Rd=memh(Rs+Rt<<#u2)
// if (Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrih_indexed_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrih_indexed_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrih_indexed_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDrih_indexed_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrih_indexed_shl_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrih_indexed_shl_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrih_indexed_shl_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDrih_indexed_shl_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

//// Load unsigned halfword conditionally.
// if ([!]Pv[.new]) Rd=memuh(Rs+Rt<<#u2)
// if (Pv) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriuh_indexed_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriuh_indexed_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriuh_indexed_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriuh_indexed_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memuh($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriuh_indexed_shl_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriuh_indexed_shl_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriuh_indexed_shl_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memuh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriuh_indexed_shl_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memuh($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rd=memh(Rt<<#u2+#U6)

//// Load word.
// Rd=memw(Re=#U6)

// Rd=memw(Rs+Rt<<#u2)
let AddedComplexity = 10, isPredicable = 1 in
def LDriw_indexed_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2),
                    "$dst=memw($src1+$src2<<#0)",
                    [(set IntRegs:$dst, (load (add IntRegs:$src1,
                                                   IntRegs:$src2)))]>,
                    Requires<[HasV4T]>;

// Rd=memw(Rs+Rt<<#u2)
let AddedComplexity = 40, isPredicable = 1 in
def LDriw_indexed_shl_V4 : LDInst<(outs IntRegs:$dst),
                    (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$offset),
                    "$dst=memw($src1+$src2<<#$offset)",
                    [(set IntRegs:$dst, (load (add IntRegs:$src1,
                                              (shl IntRegs:$src2,
                                                   u2ImmPred:$offset))))]>,
                    Requires<[HasV4T]>;

//// Load word conditionally.
// if ([!]Pv[.new]) Rd=memw(Rs+Rt<<#u2)
// if (Pv) Rd=memw(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriw_indexed_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriw_indexed_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if ($src1.new) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriw_indexed_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 15 in
def LDriw_indexed_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3),
                    "if (!$src1.new) $dst=memw($src2+$src3<<#0)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriw_indexed_shl_cPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriw_indexed_shl_cdnPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if ($src1.new) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriw_indexed_shl_cNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// if (!Pv.new) Rd=memh(Rs+Rt<<#u2)
let mayLoad = 1, AddedComplexity = 45 in
def LDriw_indexed_shl_cdnNotPt_V4 : LDInst<(outs IntRegs:$dst),
                    (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3,
                         u2Imm:$offset),
                    "if (!$src1.new) $dst=memw($src2+$src3<<#$offset)",
                    []>,
                    Requires<[HasV4T]>;

// Rd=memw(Rt<<#u2+#U6)


// Post-inc Load, Predicated, Dot new


let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrid_cdnPt_V4 : LDInstPI<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_3Imm:$src3),
            "if ($src1.new) $dst1 = memd($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrid_cdnNotPt_V4 : LDInstPI<(outs DoubleRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_3Imm:$src3),
            "if (!$src1.new) $dst1 = memd($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrib_cdnPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if ($src1.new) $dst1 = memb($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrib_cdnNotPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if (!$src1.new) $dst1 = memb($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrih_cdnPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if ($src1.new) $dst1 = memh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDrih_cdnNotPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if (!$src1.new) $dst1 = memh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriub_cdnPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if ($src1.new) $dst1 = memub($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriub_cdnNotPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_0Imm:$src3),
            "if (!$src1.new) $dst1 = memub($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriuh_cdnPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if ($src1.new) $dst1 = memuh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriuh_cdnNotPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_1Imm:$src3),
            "if (!$src1.new) $dst1 = memuh($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriw_cdnPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_2Imm:$src3),
            "if ($src1.new) $dst1 = memw($src2++#$src3)",
            [],
            "$src2 = $dst2">,
            Requires<[HasV4T]>;

let mayLoad = 1, hasCtrlDep = 1, neverHasSideEffects = 1 in
def POST_LDriw_cdnNotPt_V4 : LDInstPI<(outs IntRegs:$dst1, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, s4_2Imm:$src3),
            "if (!$src1.new) $dst1 = memw($src2++#$src3)",
            [],
            "$src2 = $dst2">,
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

// Store doubleword.
// memd(Re=#U6)=Rtt
// TODO: needs to be implemented

// memd(Rs+#s11:3)=Rtt
// memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, isPredicable = 1 in
def STrid_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, DoubleRegs:$src4),
            "memd($src1+$src2<<#$src3) = $src4",
            [(store DoubleRegs:$src4, (add IntRegs:$src1,
                                      (shl IntRegs:$src2, u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memd(Ru<<#u2+#U6)=Rtt
let AddedComplexity = 10 in
def STrid_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, DoubleRegs:$src4),
            "memd($src1<<#$src2+#$src3) = $src4",
            [(store DoubleRegs:$src4, (shl IntRegs:$src1,
                                      (add u2ImmPred:$src2,
                                           u6ImmPred:$src3)))]>,
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

// if ([!]Pv[.new]) memd(Rs+#u6:3)=Rtt
// if (Pv) memd(Rs+#u6:3)=Rtt
// if (Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, DoubleRegs:$src2),
            "if ($src1.new) memd($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memd(Rs+#u6:3)=Rtt
// if (!Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, DoubleRegs:$src2),
            "if (!$src1.new) memd($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memd(Rs+#u6:3)=Rtt
// if (Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3,
                 DoubleRegs:$src4),
            "if ($src1.new) memd($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memd(Rs+#u6:3)=Rtt
// if (!Pv.new) memd(Rs+#u6:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_3Imm:$src3,
                 DoubleRegs:$src4),
            "if (!$src1.new) memd($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memd(Rs+Ru<<#u2)=Rtt
// if (Pv) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_shl_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if ($src1) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_shl_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if ($src1) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;
// if (!Pv) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_shl_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if (!$src1) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;
// if (!Pv.new) memd(Rs+Ru<<#u2)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def STrid_indexed_shl_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 DoubleRegs:$src5),
            "if (!$src1.new) memd($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memd(Rx++#s4:3)=Rtt
// if (Pv) memd(Rx++#s4:3)=Rtt
// if (Pv.new) memd(Rx++#s4:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def POST_STdri_cdnPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, DoubleRegs:$src2, IntRegs:$src3,
                 s4_3Imm:$offset),
            "if ($src1.new) memd($src3++#$offset) = $src2",
            [],
            "$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memd(Rx++#s4:3)=Rtt
// if (!Pv.new) memd(Rx++#s4:3)=Rtt
let AddedComplexity = 10, mayStore = 1, neverHasSideEffects = 1 in
def POST_STdri_cdnNotPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, DoubleRegs:$src2, IntRegs:$src3,
                 s4_3Imm:$offset),
            "if (!$src1.new) memd($src3++#$offset) = $src2",
            [],
            "$src3 = $dst">,
            Requires<[HasV4T]>;


// Store byte.
// memb(Re=#U6)=Rt
// TODO: needs to be implemented.
// memb(Rs+#s11:0)=Rt
// memb(Rs+#u6:0)=#S8
let AddedComplexity = 10, isPredicable = 1 in
def STrib_imm_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_0Imm:$src2, s8Imm:$src3),
            "memb($src1+#$src2) = #$src3",
            [(truncstorei8 s8ImmPred:$src3, (add IntRegs:$src1,
                                                 u6_0ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// memb(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10, isPredicable = 1 in
def STrib_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memb($src1+$src2<<#$src3) = $src4",
            [(truncstorei8 IntRegs:$src4, (add IntRegs:$src1,
                                          (shl IntRegs:$src2,
                                               u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memb(Ru<<#u2+#U6)=Rt
let AddedComplexity = 10 in
def STrib_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memb($src1<<#$src2+#$src3) = $src4",
            [(truncstorei8 IntRegs:$src4, (shl IntRegs:$src1,
                                          (add u2ImmPred:$src2,
                                               u6ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0:circ(Mu))=Rt
// memb(Rx++I:circ(Mu))=Rt
// memb(Rx++Mu)=Rt
// memb(Rx++Mu:brev)=Rt
// memb(gp+#u16:0)=Rt


// Store byte conditionally.
// if ([!]Pv[.new]) memb(#u6)=Rt
// if ([!]Pv[.new]) memb(Rs+#u6:0)=#S6
// if (Pv) memb(Rs+#u6:0)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_imm_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if ($src1) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+#u6:0)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_imm_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if ($src1.new) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_imm_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if (!$src1) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_imm_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, s6Imm:$src4),
            "if (!$src1.new) memb($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rs+#u6:0)=Rt
// if (Pv) memb(Rs+#u6:0)=Rt
// if (Pv.new) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memb($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=Rt
// if (!Pv.new) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memb($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(Rs+#u6:0)=Rt
// if (!Pv) memb(Rs+#u6:0)=Rt
// if (Pv.new) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memb($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memb($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rs+Ru<<#u2)=Rt
// if (Pv) memb(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memb($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rx++#s4:0)=Rt
// if (Pv) memb(Rx++#s4:0)=Rt
// if (Pv.new) memb(Rx++#s4:0)=Rt
let mayStore = 1, hasCtrlDep = 1 in
def POST_STbri_cdnPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1.new) memb($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rx++#s4:0)=Rt
// if (!Pv.new) memb(Rx++#s4:0)=Rt
let mayStore = 1, hasCtrlDep = 1 in
def POST_STbri_cdnNotPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1.new) memb($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store halfword.
// memh(Re=#U6)=Rt.H
// TODO: needs to be implemented

// memh(Re=#U6)=Rt
// TODO: needs to be implemented

// memh(Rs+#s11:1)=Rt.H
// memh(Rs+#s11:1)=Rt
// memh(Rs+#u6:1)=#S8
let AddedComplexity = 10, isPredicable = 1 in
def STrih_imm_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_1Imm:$src2, s8Imm:$src3),
            "memh($src1+#$src2) = #$src3",
            [(truncstorei16 s8ImmPred:$src3, (add IntRegs:$src1,
                                                  u6_1ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// memh(Rs+Ru<<#u2)=Rt.H
// TODO: needs to be implemented.

// memh(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10, isPredicable = 1 in
def STrih_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memh($src1+$src2<<#$src3) = $src4",
            [(truncstorei16 IntRegs:$src4, (add IntRegs:$src1,
                                          (shl IntRegs:$src2,
                                               u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memh(Ru<<#u2+#U6)=Rt.H
// memh(Ru<<#u2+#U6)=Rt
let AddedComplexity = 10 in
def STrih_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memh($src1<<#$src2+#$src3) = $src4",
            [(truncstorei16 IntRegs:$src4, (shl IntRegs:$src1,
                                          (add u2ImmPred:$src2,
                                               u6ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1:circ(Mu))=Rt.H
// memh(Rx++#s4:1:circ(Mu))=Rt
// memh(Rx++I:circ(Mu))=Rt.H
// memh(Rx++I:circ(Mu))=Rt
// memh(Rx++Mu)=Rt.H
// memh(Rx++Mu)=Rt
// memh(Rx++Mu:brev)=Rt.H
// memh(Rx++Mu:brev)=Rt
// memh(gp+#u16:1)=Rt.H
// memh(gp+#u16:1)=Rt


// Store halfword conditionally.
// if ([!]Pv[.new]) memh(#u6)=Rt.H
// if ([!]Pv[.new]) memh(#u6)=Rt

// if ([!]Pv[.new]) memh(Rs+#u6:1)=#S6
// if (Pv) memh(Rs+#u6:1)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_imm_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if ($src1) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_imm_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if ($src1.new) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_imm_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if (!$src1) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_imm_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, s6Imm:$src4),
            "if (!$src1.new) memh($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rs+#u6:1)=Rt.H
// TODO: needs to be implemented.

// if ([!]Pv[.new]) memh(Rs+#u6:1)=Rt
// if (Pv) memh(Rs+#u6:1)=Rt
// if (Pv.new) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memh($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=Rt
// if (!Pv.new) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memh($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memh($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memh($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rs+Ru<<#u2)=Rt.H
// if ([!]Pv[.new]) memh(Rs+Ru<<#u2)=Rt
// if (Pv) memh(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+Ru<<#u2)=Rt
def STrih_indexed_shl_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memh($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rx++#s4:1)=Rt.H
// TODO: Needs to be implemented.

// if ([!]Pv[.new]) memh(Rx++#s4:1)=Rt
// if (Pv) memh(Rx++#s4:1)=Rt
// if (Pv.new) memh(Rx++#s4:1)=Rt
let mayStore = 1, hasCtrlDep = 1 in
def POST_SThri_cdnPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1.new) memh($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rx++#s4:1)=Rt
// if (!Pv.new) memh(Rx++#s4:1)=Rt
let mayStore = 1, hasCtrlDep = 1 in
def POST_SThri_cdnNotPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1.new) memh($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store word.
// memw(Re=#U6)=Rt
// TODO: Needs to be implemented.

// memw(Rs+#s11:2)=Rt
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 10, isPredicable = 1 in
def STriw_imm_V4 : STInst<(outs),
            (ins IntRegs:$src1, u6_2Imm:$src2, s8Imm:$src3),
            "memw($src1+#$src2) = #$src3",
            [(store s8ImmPred:$src3, (add IntRegs:$src1, u6_2ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// memw(Rs+Ru<<#u2)=Rt
let AddedComplexity = 10, isPredicable = 1 in
def STriw_indexed_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memw($src1+$src2<<#$src3) = $src4",
            [(store IntRegs:$src4, (add IntRegs:$src1,
                                    (shl IntRegs:$src2, u2ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memw(Ru<<#u2+#U6)=Rt
let AddedComplexity = 10 in
def STriw_shl_V4 : STInst<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memw($src1<<#$src2+#$src3) = $src4",
            [(store IntRegs:$src4, (shl IntRegs:$src1,
                                   (add u2ImmPred:$src2, u6ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2)=Rt
// memw(Rx++#s4:2:circ(Mu))=Rt
// memw(Rx++I:circ(Mu))=Rt
// memw(Rx++Mu)=Rt
// memw(Rx++Mu:brev)=Rt
// memw(gp+#u16:2)=Rt


// Store word conditionally.
// if ([!]Pv[.new]) memw(#u6)=Rt
// TODO: Needs to be implemented.

// if ([!]Pv[.new]) memw(Rs+#u6:2)=#S6
// if (Pv) memw(Rs+#u6:2)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_imm_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if ($src1) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+#u6:2)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_imm_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if ($src1.new) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_imm_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if (!$src1) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=#S6
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_imm_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, s6Imm:$src4),
            "if (!$src1.new) memw($src2+#$src3) = #$src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rs+#u6:2)=Rt
// if (Pv) memw(Rs+#u6:2)=Rt
// if (Pv.new) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memw($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=Rt
// if (!Pv.new) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memw($addr) = $src2",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(Rs+#u6:2)=Rt
// if (!Pv) memw(Rs+#u6:2)=Rt
// if (Pv.new) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memw($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=Rt
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memw($src2+#$src3) = $src4",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rs+Ru<<#u2)=Rt
// if (Pv) memw(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cdnPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+Ru<<#u2)=Rt
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cdnNotPt_V4 : STInst<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memw($src2+$src3<<#$src4) = $src5",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rx++#s4:2)=Rt
// if (Pv) memw(Rx++#s4:2)=Rt
// if (Pv.new) memw(Rx++#s4:2)=Rt
let mayStore = 1, hasCtrlDep = 1 in
def POST_STwri_cdnPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1.new) memw($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rx++#s4:2)=Rt
// if (!Pv.new) memw(Rx++#s4:2)=Rt
let mayStore = 1, hasCtrlDep = 1 in
def POST_STwri_cdnNotPt_V4 : STInstPI<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1.new) memw($src3++#$offset) = $src2",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


//===----------------------------------------------------------------------===
// ST -
//===----------------------------------------------------------------------===


//===----------------------------------------------------------------------===//
// NV/ST +
//===----------------------------------------------------------------------===//

// Store new-value byte.

// memb(Re=#U6)=Nt.new
// memb(Rs+#s11:0)=Nt.new
let mayStore = 1, isPredicable = 1 in
def STrib_nv_V4 : NVInst_V4<(outs), (ins MEMri:$addr, IntRegs:$src1),
            "memb($addr) = $src1.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, isPredicable = 1 in
def STrib_indexed_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, s11_0Imm:$src2, IntRegs:$src3),
            "memb($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

// memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10, isPredicable = 1 in
def STrib_indexed_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memb($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memb(Ru<<#u2+#U6)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrib_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memb($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1, isPredicable = 1  in
def POST_STbri_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4_0Imm:$offset),
            "memb($src2++#$offset) = $src1.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// memb(Rx++#s4:0:circ(Mu))=Nt.new
// memb(Rx++I:circ(Mu))=Nt.new
// memb(Rx++Mu)=Nt.new
// memb(Rx++Mu:brev)=Nt.new

// memb(gp+#u16:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memb(#$global+$offset) = $src.new",
            []>,
            Requires<[HasV4T]>;


// Store new-value byte conditionally.
// if ([!]Pv[.new]) memb(#u6)=Nt.new
// if (Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memb($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+#u6:0)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrib_indexed_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_0Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memb($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;


// if ([!]Pv[.new]) memb(Rs+Ru<<#u2)=Nt.new
// if (Pv) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrib_indexed_shl_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memb($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memb(Rx++#s4:0)=Nt.new
// if (Pv) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STbri_cPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (Pv.new) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STbri_cdnPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if ($src1.new) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STbri_cNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv.new) memb(Rx++#s4:0)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STbri_cdnNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_0Imm:$offset),
            "if (!$src1.new) memb($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store new-value halfword.
// memh(Re=#U6)=Nt.new
// memh(Rs+#s11:1)=Nt.new
let mayStore = 1, isPredicable = 1 in
def STrih_nv_V4 : NVInst_V4<(outs), (ins MEMri:$addr, IntRegs:$src1),
            "memh($addr) = $src1.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, isPredicable = 1 in
def STrih_indexed_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, s11_1Imm:$src2, IntRegs:$src3),
            "memh($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

// memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10, isPredicable = 1 in
def STrih_indexed_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memh($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memh(Ru<<#u2+#U6)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrih_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memh($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1, isPredicable = 1  in
def POST_SThri_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4_1Imm:$offset),
            "memh($src2++#$offset) = $src1.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// memh(Rx++#s4:1:circ(Mu))=Nt.new
// memh(Rx++I:circ(Mu))=Nt.new
// memh(Rx++Mu)=Nt.new
// memh(Rx++Mu:brev)=Nt.new

// memh(gp+#u16:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memh(#$global+$offset) = $src.new",
            []>,
            Requires<[HasV4T]>;


// Store new-value halfword conditionally.

// if ([!]Pv[.new]) memh(#u6)=Nt.new

// if ([!]Pv[.new]) memh(Rs+#u6:1)=Nt.new
// if (Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memh($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+#u6:1)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STrih_indexed_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_1Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memh($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memh(Rs+Ru<<#u2)=Nt.new
// if (Pv) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STrih_indexed_shl_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memh($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[]) memh(Rx++#s4:1)=Nt.new
// if (Pv) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_SThri_cPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (Pv.new) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_SThri_cdnPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if ($src1.new) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_SThri_cNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv.new) memh(Rx++#s4:1)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_SThri_cdnNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_1Imm:$offset),
            "if (!$src1.new) memh($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


// Store new-value word.

// memw(Re=#U6)=Nt.new
// memw(Rs+#s11:2)=Nt.new
let mayStore = 1, isPredicable = 1 in
def STriw_nv_V4 : NVInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memw($addr) = $src1.new",
            []>,
            Requires<[HasV4T]>;

let mayStore = 1, isPredicable = 1 in
def STriw_indexed_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, s11_2Imm:$src2, IntRegs:$src3),
            "memw($src1+#$src2) = $src3.new",
            []>,
            Requires<[HasV4T]>;

// memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10, isPredicable = 1 in
def STriw_indexed_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$src3, IntRegs:$src4),
            "memw($src1+$src2<<#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memw(Ru<<#u2+#U6)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STriw_shl_nv_V4 : NVInst_V4<(outs),
            (ins IntRegs:$src1, u2Imm:$src2, u6Imm:$src3, IntRegs:$src4),
            "memw($src1<<#$src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1, isPredicable = 1  in
def POST_STwri_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, s4_2Imm:$offset),
            "memw($src2++#$offset) = $src1.new",
            [],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// memw(Rx++#s4:2:circ(Mu))=Nt.new
// memw(Rx++I:circ(Mu))=Nt.new
// memw(Rx++Mu)=Nt.new
// memw(Rx++Mu:brev)=Nt.new
// memw(gp+#u16:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_GP_nv_V4 : NVInst_V4<(outs),
            (ins globaladdress:$global, u16Imm:$offset, IntRegs:$src),
            "memw(#$global+$offset) = $src.new",
            []>,
            Requires<[HasV4T]>;


// Store new-value word conditionally.

// if ([!]Pv[.new]) memw(#u6)=Nt.new

// if ([!]Pv[.new]) memw(Rs+#u6:2)=Nt.new
// if (Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if ($src1.new) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, MEMri:$addr, IntRegs:$src2),
            "if (!$src1.new) memw($addr) = $src2.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if ($src1.new) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+#u6:2)=Nt.new
let mayStore = 1, neverHasSideEffects = 1 in
def STriw_indexed_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, u6_2Imm:$src3, IntRegs:$src4),
            "if (!$src1.new) memw($src2+#$src3) = $src4.new",
            []>,
            Requires<[HasV4T]>;


// if ([!]Pv[.new]) memw(Rs+Ru<<#u2)=Nt.new
// if (Pv) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cdnPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if ($src1.new) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rs+Ru<<#u2)=Nt.new
let mayStore = 1, AddedComplexity = 10 in
def STriw_indexed_shl_cdnNotPt_nv_V4 : NVInst_V4<(outs),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$src4,
                 IntRegs:$src5),
            "if (!$src1.new) memw($src2+$src3<<#$src4) = $src5.new",
            []>,
            Requires<[HasV4T]>;

// if ([!]Pv[.new]) memw(Rx++#s4:2)=Nt.new
// if (Pv) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STwri_cPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (Pv.new) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STwri_cdnPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if ($src1.new) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STwri_cNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;

// if (!Pv.new) memw(Rx++#s4:2)=Nt.new
let mayStore = 1, hasCtrlDep = 1 in
def POST_STwri_cdnNotPt_nv_V4 : NVInstPI_V4<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, s4_2Imm:$offset),
            "if (!$src1.new) memw($src3++#$offset) = $src2.new",
            [],"$src3 = $dst">,
            Requires<[HasV4T]>;


//===----------------------------------------------------------------------===//
// NV/ST -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// XTYPE/ALU +
//===----------------------------------------------------------------------===//

//  Add and accumulate.
//  Rd=add(Rs,add(Ru,#s6))
def ADDr_ADDri_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, IntRegs:$src2, s6Imm:$src3),
          "$dst = add($src1, add($src2, #$src3))",
          [(set IntRegs:$dst,
           (add IntRegs:$src1, (add IntRegs:$src2, s6ImmPred:$src3)))]>,
          Requires<[HasV4T]>;

//  Rd=add(Rs,sub(#s6,Ru))
def ADDr_SUBri_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, s6Imm:$src2, IntRegs:$src3),
          "$dst = add($src1, sub(#$src2, $src3))",
          [(set IntRegs:$dst,
           (add IntRegs:$src1, (sub s6ImmPred:$src2, IntRegs:$src3)))]>,
          Requires<[HasV4T]>;

// Generates the same instruction as ADDr_SUBri_V4 but matches different
// pattern.
//  Rd=add(Rs,sub(#s6,Ru))
def ADDri_SUBr_V4 : MInst<(outs IntRegs:$dst),
          (ins IntRegs:$src1, s6Imm:$src2, IntRegs:$src3),
          "$dst = add($src1, sub(#$src2, $src3))",
          [(set IntRegs:$dst,
           (sub (add IntRegs:$src1, s6ImmPred:$src2), IntRegs:$src3))]>,
          Requires<[HasV4T]>;


//  Add or subtract doublewords with carry.
//TODO:
//  Rdd=add(Rss,Rtt,Px):carry
//TODO:
//  Rdd=sub(Rss,Rtt,Px):carry


//  Logical doublewords.
//  Rdd=and(Rtt,~Rss)
def ANDd_NOTd_V4 : MInst<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2),
          "$dst = and($src1, ~$src2)",
          [(set DoubleRegs:$dst, (and DoubleRegs:$src1,
                                      (not DoubleRegs:$src2)))]>,
          Requires<[HasV4T]>;

//  Rdd=or(Rtt,~Rss)
def ORd_NOTd_V4 : MInst<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2),
          "$dst = or($src1, ~$src2)",
          [(set DoubleRegs:$dst,
           (or DoubleRegs:$src1, (not DoubleRegs:$src2)))]>,
          Requires<[HasV4T]>;


//  Logical-logical doublewords.
//  Rxx^=xor(Rss,Rtt)
def XORd_XORdd: MInst_acc<(outs DoubleRegs:$dst),
          (ins DoubleRegs:$src1, DoubleRegs:$src2, DoubleRegs:$src3),
          "$dst ^= xor($src2, $src3)",
          [(set DoubleRegs:$dst,
           (xor DoubleRegs:$src1, (xor DoubleRegs:$src2, DoubleRegs:$src3)))],
          "$src1 = $dst">,
          Requires<[HasV4T]>;


// Logical-logical words.
// Rx=or(Ru,and(Rx,#s10))
def ORr_ANDri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Imm:$src3),
            "$dst = or($src1, and($src2, #$src3))",
            [(set IntRegs:$dst,
             (or IntRegs:$src1, (and IntRegs:$src2, s10ImmPred:$src3)))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=and(Rs,Rt)
// Rx&=and(Rs,Rt)
def ANDr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= and($src2, $src3)",
            [(set IntRegs:$dst,
             (and IntRegs:$src1, (and IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,Rt)
def ORr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= and($src2, $src3)",
            [(set IntRegs:$dst,
             (or IntRegs:$src1, (and IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=and(Rs,Rt)
def XORr_ANDrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= and($src2, $src3)",
            [(set IntRegs:$dst,
             (xor IntRegs:$src1, (and IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=and(Rs,~Rt)
// Rx&=and(Rs,~Rt)
def ANDr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= and($src2, ~$src3)",
            [(set IntRegs:$dst,
             (and IntRegs:$src1, (and IntRegs:$src2, (not IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,~Rt)
def ORr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= and($src2, ~$src3)",
            [(set IntRegs:$dst,
             (or IntRegs:$src1, (and IntRegs:$src2, (not IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=and(Rs,~Rt)
def XORr_ANDr_NOTr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= and($src2, ~$src3)",
            [(set IntRegs:$dst,
             (xor IntRegs:$src1, (and IntRegs:$src2, (not IntRegs:$src3))))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=or(Rs,Rt)
// Rx&=or(Rs,Rt)
def ANDr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= or($src2, $src3)",
            [(set IntRegs:$dst,
             (and IntRegs:$src1, (or IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=or(Rs,Rt)
def ORr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= or($src2, $src3)",
            [(set IntRegs:$dst,
             (or IntRegs:$src1, (or IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=or(Rs,Rt)
def XORr_ORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= or($src2, $src3)",
            [(set IntRegs:$dst,
             (xor IntRegs:$src1, (or IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx[&|^]=xor(Rs,Rt)
// Rx&=xor(Rs,Rt)
def ANDr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst &= xor($src2, $src3)",
            [(set IntRegs:$dst,
             (and IntRegs:$src1, (xor IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=xor(Rs,Rt)
def ORr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst |= xor($src2, $src3)",
            [(set IntRegs:$dst,
             (and IntRegs:$src1, (xor IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx^=xor(Rs,Rt)
def XORr_XORrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, IntRegs:$src3),
            "$dst ^= xor($src2, $src3)",
            [(set IntRegs:$dst,
             (and IntRegs:$src1, (xor IntRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=and(Rs,#s10)
def ORr_ANDri2_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Imm:$src3),
            "$dst |= and($src2, #$src3)",
            [(set IntRegs:$dst,
             (or IntRegs:$src1, (and IntRegs:$src2, s10ImmPred:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

// Rx|=or(Rs,#s10)
def ORr_ORri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs: $src2, s10Imm:$src3),
            "$dst |= or($src2, #$src3)",
            [(set IntRegs:$dst,
             (or IntRegs:$src1, (and IntRegs:$src2, s10ImmPred:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;


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
def ADDi_MPYri_V4 : MInst<(outs IntRegs:$dst),
            (ins u6Imm:$src1, IntRegs:$src2, u6Imm:$src3),
            "$dst = add(#$src1, mpyi($src2, #$src3))",
            [(set IntRegs:$dst,
             (add (mul IntRegs:$src2, u6ImmPred:$src3), u6ImmPred:$src1))]>,
            Requires<[HasV4T]>;

// Rd=add(#u6,mpyi(Rs,Rt))

def ADDi_MPYrr_V4 : MInst<(outs IntRegs:$dst),
            (ins u6Imm:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst = add(#$src1, mpyi($src2, $src3))",
            [(set IntRegs:$dst,
             (add (mul IntRegs:$src2, IntRegs:$src3), u6ImmPred:$src1))]>,
            Requires<[HasV4T]>;

// Rd=add(Ru,mpyi(#u6:2,Rs))
def ADDr_MPYir_V4 : MInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, u6Imm:$src2, IntRegs:$src3),
            "$dst = add($src1, mpyi(#$src2, $src3))",
            [(set IntRegs:$dst,
             (add IntRegs:$src1, (mul IntRegs:$src3, u6_2ImmPred:$src2)))]>,
            Requires<[HasV4T]>;

// Rd=add(Ru,mpyi(Rs,#u6))
def ADDr_MPYri_V4 : MInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, u6Imm:$src3),
            "$dst = add($src1, mpyi($src2, #$src3))",
            [(set IntRegs:$dst,
             (add IntRegs:$src1, (mul IntRegs:$src2, u6ImmPred:$src3)))]>,
            Requires<[HasV4T]>;

// Rx=add(Ru,mpyi(Rx,Rs))
def ADDr_MPYrr_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
            "$dst = add($src1, mpyi($src2, $src3))",
            [(set IntRegs:$dst,
             (add IntRegs:$src1, (mul IntRegs:$src2, IntRegs:$src3)))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


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
def ADDi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = add(#$src1, asl($src2, #$src3))",
            [(set IntRegs:$dst,
              (add (shl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=add(#u8,lsr(Rx,#U5))
def ADDi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = add(#$src1, lsr($src2, #$src3))",
            [(set IntRegs:$dst,
             (add (srl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=sub(#u8,asl(Rx,#U5))
def SUBi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = sub(#$src1, asl($src2, #$src3))",
            [(set IntRegs:$dst,
             (sub (shl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

// Rx=sub(#u8,lsr(Rx,#U5))
def SUBi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = sub(#$src1, lsr($src2, #$src3))",
            [(set IntRegs:$dst,
             (sub (srl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


//Shift by immediate and logical.
//Rx=and(#u8,asl(Rx,#U5))
def ANDi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = and(#$src1, asl($src2, #$src3))",
            [(set IntRegs:$dst,
             (and (shl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=and(#u8,lsr(Rx,#U5))
def ANDi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = and(#$src1, lsr($src2, #$src3))",
            [(set IntRegs:$dst,
             (and (srl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=or(#u8,asl(Rx,#U5))
def ORi_ASLri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = or(#$src1, asl($src2, #$src3))",
            [(set IntRegs:$dst,
             (or (shl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;

//Rx=or(#u8,lsr(Rx,#U5))
def ORi_LSRri_V4 : MInst_acc<(outs IntRegs:$dst),
            (ins u8Imm:$src1, IntRegs:$src2, u5Imm:$src3),
            "$dst = or(#$src1, lsr($src2, #$src3))",
            [(set IntRegs:$dst,
             (or (srl IntRegs:$src2, u5ImmPred:$src3), u8ImmPred:$src1))],
            "$src2 = $dst">,
            Requires<[HasV4T]>;


//Shift by register.
//Rd=lsl(#s6,Rt)
def LSLi_V4 : MInst<(outs IntRegs:$dst), (ins s6Imm:$src1, IntRegs:$src2),
            "$dst = lsl(#$src1, $src2)",
            [(set IntRegs:$dst, (shl s6ImmPred:$src1, IntRegs:$src2))]>,
            Requires<[HasV4T]>;


//Shift by register and logical.
//Rxx^=asl(Rss,Rt)
def ASLd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= asl($src2, $src3)",
            [(set DoubleRegs:$dst,
             (xor DoubleRegs:$src1, (shl DoubleRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=asr(Rss,Rt)
def ASRd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= asr($src2, $src3)",
            [(set DoubleRegs:$dst,
             (xor DoubleRegs:$src1, (sra DoubleRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=lsl(Rss,Rt)
def LSLd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= lsl($src2, $src3)",
            [(set DoubleRegs:$dst,
             (xor DoubleRegs:$src1, (shl DoubleRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;

//Rxx^=lsr(Rss,Rt)
def LSRd_rr_xor_V4 : MInst_acc<(outs DoubleRegs:$dst),
            (ins DoubleRegs:$src1, DoubleRegs:$src2, IntRegs:$src3),
            "$dst ^= lsr($src2, $src3)",
            [(set DoubleRegs:$dst,
             (xor DoubleRegs:$src1, (srl DoubleRegs:$src2, IntRegs:$src3)))],
            "$src1 = $dst">,
            Requires<[HasV4T]>;


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


// MEMw_ADDSUBi_indexed_V4:
//   pseudo operation for MEMw_ADDi_indexed_V4 and
//   MEMw_SUBi_indexed_V4 a later pass will change it
//   to the corresponding pattern.
let AddedComplexity = 30 in
def MEMw_ADDSUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, m6Imm:$addend),
            "Error; should not emit",
            [(store (add (load (add IntRegs:$base, u6_2ImmPred:$offset)),
m6ImmPred:$addend),
                    (add IntRegs:$base, u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += #U5
let AddedComplexity = 30 in
def MEMw_ADDi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, u5Imm:$addend),
            "memw($base+#$offset) += $addend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= #U5
let AddedComplexity = 30 in
def MEMw_SUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, u5Imm:$subend),
            "memw($base+#$offset) -= $subend",
            []>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) += Rt
let AddedComplexity = 30 in
def MEMw_ADDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$addend),
            "memw($base+#$offset) += $addend",
            [(store (add (load (add IntRegs:$base, u6_2ImmPred:$offset)),
IntRegs:$addend),
                    (add IntRegs:$base, u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= Rt
let AddedComplexity = 30 in
def MEMw_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$subend),
            "memw($base+#$offset) -= $subend",
            [(store (sub (load (add IntRegs:$base, u6_2ImmPred:$offset)),
IntRegs:$subend),
                    (add IntRegs:$base, u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) &= Rt
let AddedComplexity = 30 in
def MEMw_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$andend),
            "memw($base+#$offset) += $andend",
            [(store (and (load (add IntRegs:$base, u6_2ImmPred:$offset)),
IntRegs:$andend),
                    (add IntRegs:$base, u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) |= Rt
let AddedComplexity = 30 in
def MEMw_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_2Imm:$offset, IntRegs:$orend),
            "memw($base+#$offset) |= $orend",
            [(store (or (load (add IntRegs:$base, u6_2ImmPred:$offset)),
                        IntRegs:$orend),
                    (add IntRegs:$base, u6_2ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// MEMw_ADDSUBi_V4:
//   Pseudo operation for MEMw_ADDi_V4 and MEMw_SUBi_V4
//   a later pass will change it to the right pattern.
let AddedComplexity = 30 in
def MEMw_ADDSUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, m6Imm:$addend),
            "Error; should not emit",
            [(store (add (load ADDRriU6_2:$addr), m6ImmPred:$addend),
                    ADDRriU6_2:$addr)]>,
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
            [(store (add (load ADDRriU6_2:$addr), IntRegs:$addend),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) -= Rt
let AddedComplexity = 30 in
def MEMw_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memw($addr) -= $subend",
            [(store (sub (load ADDRriU6_2:$addr), IntRegs:$subend),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) &= Rt
let AddedComplexity = 30 in
def MEMw_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memw($addr) &= $andend",
            [(store (and (load ADDRriU6_2:$addr), IntRegs:$andend),
                    ADDRriU6_2:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memw(Rs+#u6:2) |= Rt
let AddedComplexity = 30 in
def MEMw_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memw($addr) |= $orend",
            [(store (or (load ADDRriU6_2:$addr), IntRegs:$orend),
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


// MEMh_ADDSUBi_indexed_V4:
//   Pseudo operation for MEMh_ADDi_indexed_V4 and
//   MEMh_SUBi_indexed_V4 a later pass will change it
//   to the corresponding pattern.
let AddedComplexity = 30 in
def MEMh_ADDSUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei16 (add (sextloadi16 (add IntRegs:$base,
                                                   u6_1ImmPred:$offset)),
                                 m6ImmPred:$addend),
                            (add IntRegs:$base, u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

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
            [(truncstorei16 (add (sextloadi16 (add IntRegs:$base,
                                                   u6_1ImmPred:$offset)),
                                 IntRegs:$addend),
                            (add IntRegs:$base, u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= Rt
let AddedComplexity = 30 in
def MEMh_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$subend),
            "memh($base+#$offset) -= $subend",
            [(truncstorei16 (sub (sextloadi16 (add IntRegs:$base,
                                                   u6_1ImmPred:$offset)),
                                 IntRegs:$subend),
                            (add IntRegs:$base, u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) &= Rt
let AddedComplexity = 30 in
def MEMh_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$andend),
            "memh($base+#$offset) += $andend",
            [(truncstorei16 (and (sextloadi16 (add IntRegs:$base,
                                                   u6_1ImmPred:$offset)),
                                 IntRegs:$andend),
                            (add IntRegs:$base, u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) |= Rt
let AddedComplexity = 30 in
def MEMh_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_1Imm:$offset, IntRegs:$orend),
            "memh($base+#$offset) |= $orend",
            [(truncstorei16 (or (sextloadi16 (add IntRegs:$base,
                                              u6_1ImmPred:$offset)),
                             IntRegs:$orend),
                            (add IntRegs:$base, u6_1ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// MEMh_ADDSUBi_V4:
//   Pseudo operation for MEMh_ADDi_V4 and MEMh_SUBi_V4
//   a later pass will change it to the right pattern.
let AddedComplexity = 30 in
def MEMh_ADDSUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei16 (add (sextloadi16 ADDRriU6_1:$addr),
                                 m6ImmPred:$addend), ADDRriU6_1:$addr)]>,
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
                                 IntRegs:$addend), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) -= Rt
let AddedComplexity = 30 in
def MEMh_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memh($addr) -= $subend",
            [(truncstorei16 (sub (sextloadi16 ADDRriU6_1:$addr),
                                 IntRegs:$subend), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) &= Rt
let AddedComplexity = 30 in
def MEMh_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memh($addr) &= $andend",
            [(truncstorei16 (and (sextloadi16 ADDRriU6_1:$addr),
                                 IntRegs:$andend), ADDRriU6_1:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memh(Rs+#u6:1) |= Rt
let AddedComplexity = 30 in
def MEMh_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memh($addr) |= $orend",
            [(truncstorei16 (or (sextloadi16 ADDRriU6_1:$addr),
                                IntRegs:$orend), ADDRriU6_1:$addr)]>,
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


// MEMb_ADDSUBi_indexed_V4:
//   Pseudo operation for MEMb_ADDi_indexed_V4 and
//   MEMb_SUBi_indexed_V4 a later pass will change it
//   to the corresponding pattern.
let AddedComplexity = 30 in
def MEMb_ADDSUBi_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei8 (add (sextloadi8 (add IntRegs:$base,
                                                 u6_0ImmPred:$offset)),
                                m6ImmPred:$addend),
                           (add IntRegs:$base, u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

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
            [(truncstorei8 (add (sextloadi8 (add IntRegs:$base,
                                                 u6_0ImmPred:$offset)),
                                IntRegs:$addend),
                           (add IntRegs:$base, u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= Rt
let AddedComplexity = 30 in
def MEMb_SUBr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$subend),
            "memb($base+#$offset) -= $subend",
            [(truncstorei8 (sub (sextloadi8 (add IntRegs:$base,
                                                 u6_0ImmPred:$offset)),
                                IntRegs:$subend),
                           (add IntRegs:$base, u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) &= Rt
let AddedComplexity = 30 in
def MEMb_ANDr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$andend),
            "memb($base+#$offset) += $andend",
            [(truncstorei8 (and (sextloadi8 (add IntRegs:$base,
                                                 u6_0ImmPred:$offset)),
                                IntRegs:$andend),
                           (add IntRegs:$base, u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) |= Rt
let AddedComplexity = 30 in
def MEMb_ORr_indexed_MEM_V4 : MEMInst_V4<(outs),
            (ins IntRegs:$base, u6_0Imm:$offset, IntRegs:$orend),
            "memb($base+#$offset) |= $orend",
            [(truncstorei8 (or (sextloadi8 (add IntRegs:$base,
                                                u6_0ImmPred:$offset)),
                                IntRegs:$orend),
                           (add IntRegs:$base, u6_0ImmPred:$offset))]>,
            Requires<[HasV4T, UseMEMOP]>;

// MEMb_ADDSUBi_V4:
//   Pseudo operation for MEMb_ADDi_V4 and MEMb_SUBi_V4
//   a later pass will change it to the right pattern.
let AddedComplexity = 30 in
def MEMb_ADDSUBi_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, m6Imm:$addend),
            "Error; should not emit",
            [(truncstorei8 (add (sextloadi8 ADDRriU6_0:$addr),
                                m6ImmPred:$addend), ADDRriU6_0:$addr)]>,
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
                                IntRegs:$addend), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) -= Rt
let AddedComplexity = 30 in
def MEMb_SUBr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$subend),
            "memb($addr) -= $subend",
            [(truncstorei8 (sub (sextloadi8 ADDRriU6_0:$addr),
                                IntRegs:$subend), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) &= Rt
let AddedComplexity = 30 in
def MEMb_ANDr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$andend),
            "memb($addr) &= $andend",
            [(truncstorei8 (and (sextloadi8 ADDRriU6_0:$addr),
                                IntRegs:$andend), ADDRriU6_0:$addr)]>,
            Requires<[HasV4T, UseMEMOP]>;

// memb(Rs+#u6:0) |= Rt
let AddedComplexity = 30 in
def MEMb_ORr_MEM_V4 : MEMInst_V4<(outs),
            (ins MEMri:$addr, IntRegs:$orend),
            "memb($addr) |= $orend",
            [(truncstorei8 (or (sextloadi8 ADDRriU6_0:$addr),
                                IntRegs:$orend), ADDRriU6_0:$addr)]>,
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

// Pd=cmpb.eq(Rs,#u8)
let isCompare = 1 in
def CMPbEQri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u8Imm:$src2),
            "$dst = cmpb.eq($src1, #$src2)",
            [(set PredRegs:$dst, (seteq (and IntRegs:$src1, 255),
                                        u8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.eq(Rs,Rt)
let isCompare = 1 in
def CMPbEQrr_ubub_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.eq($src1, $src2)",
            [(set PredRegs:$dst, (seteq (and (xor IntRegs:$src1,
                                                  IntRegs:$src2),
                                             255),
                                        0))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.eq(Rs,Rt)
let isCompare = 1 in
def CMPbEQrr_sbsb_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.eq($src1, $src2)",
            [(set PredRegs:$dst, (seteq (shl IntRegs:$src1, (i32 24)),
                                        (shl IntRegs:$src2, (i32 24))))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gt(Rs,#s8)
let isCompare = 1 in
def CMPbGTri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, s32Imm:$src2),
            "$dst = cmpb.gt($src1, #$src2)",
            [(set PredRegs:$dst, (setgt (shl IntRegs:$src1, (i32 24)),
                                        s32_24ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gt(Rs,Rt)
let isCompare = 1 in
def CMPbGTrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.gt($src1, $src2)",
            [(set PredRegs:$dst, (setgt (shl IntRegs:$src1, (i32 24)),
                                        (shl IntRegs:$src2, (i32 24))))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gtu(Rs,#u7)
let isCompare = 1 in
def CMPbGTUri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u7Imm:$src2),
            "$dst = cmpb.gtu($src1, #$src2)",
            [(set PredRegs:$dst, (setugt (and IntRegs:$src1, 255),
                                         u7ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Pd=cmpb.gtu(Rs,Rt)
let isCompare = 1 in
def CMPbGTUrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmpb.gtu($src1, $src2)",
            [(set PredRegs:$dst, (setugt (and IntRegs:$src1, 255),
                                         (and IntRegs:$src2, 255)))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) ri.
// Pd=cmph.eq(Rs,#s8)
let isCompare = 1 in
def CMPhEQri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u16Imm:$src2),
            "$dst = cmph.eq($src1, #$src2)",
            [(set PredRegs:$dst, (seteq (and IntRegs:$src1, 65535),
                                        u16_s8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) rr.
// Case 1: xor + and, then compare:
//   r0=xor(r0,r1)
//   r0=and(r0,#0xffff)
//   p0=cmp.eq(r0,#0)
// Pd=cmph.eq(Rs,Rt)
let isCompare = 1 in
def CMPhEQrr_xor_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.eq($src1, $src2)",
            [(set PredRegs:$dst, (seteq (and (xor IntRegs:$src1,
                                                  IntRegs:$src2),
                                             65535),
                                        0))]>,
            Requires<[HasV4T]>;

// Signed half compare(.eq) rr.
// Case 2: shift left 16 bits then compare:
//   r0=asl(r0,16)
//   r1=asl(r1,16)
//   p0=cmp.eq(r0,r1)
// Pd=cmph.eq(Rs,Rt)
let isCompare = 1 in
def CMPhEQrr_shl_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.eq($src1, $src2)",
            [(set PredRegs:$dst, (seteq (shl IntRegs:$src1, (i32 16)),
                                        (shl IntRegs:$src2, (i32 16))))]>,
            Requires<[HasV4T]>;

// Signed half compare(.gt) ri.
// Pd=cmph.gt(Rs,#s8)
let isCompare = 1 in
def CMPhGTri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, s32Imm:$src2),
            "$dst = cmph.gt($src1, #$src2)",
            [(set PredRegs:$dst, (setgt (shl IntRegs:$src1, (i32 16)),
                                        s32_16s8ImmPred:$src2))]>,
            Requires<[HasV4T]>;

// Signed half compare(.gt) rr.
// Pd=cmph.gt(Rs,Rt)
let isCompare = 1 in
def CMPhGTrr_shl_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.gt($src1, $src2)",
            [(set PredRegs:$dst, (setgt (shl IntRegs:$src1, (i32 16)),
                                        (shl IntRegs:$src2, (i32 16))))]>,
            Requires<[HasV4T]>;

// Unsigned half compare rr (.gtu).
// Pd=cmph.gtu(Rs,Rt)
let isCompare = 1 in
def CMPhGTUrr_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = cmph.gtu($src1, $src2)",
            [(set PredRegs:$dst, (setugt (and IntRegs:$src1, 65535),
                                         (and IntRegs:$src2, 65535)))]>,
            Requires<[HasV4T]>;

// Unsigned half compare ri (.gtu).
// Pd=cmph.gtu(Rs,#u7)
let isCompare = 1 in
def CMPhGTUri_V4 : MInst<(outs PredRegs:$dst),
            (ins IntRegs:$src1, u7Imm:$src2),
            "$dst = cmph.gtu($src1, #$src2)",
            [(set PredRegs:$dst, (setugt (and IntRegs:$src1, 65535),
                                         u7ImmPred:$src2))]>,
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

//    if (Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_cPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1, i32imm:$amt1),
            "if ($src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps) dealloc_return
let isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_cNotPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                     i32imm:$amt1),
            "if (!$src1) dealloc_return",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_cdnPnt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                     i32imm:$amt1),
            "if ($src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_cNotdnPnt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                        i32imm:$amt1),
            "if (!$src1.new) dealloc_return:nt",
            []>,
            Requires<[HasV4T]>;
}

//    if (Ps.new) dealloc_return:t
let isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_cdnPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                    i32imm:$amt1),
            "if ($src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}

//    if (!Ps.new) dealloc_return:nt
let isReturn = 1, isTerminator = 1,
  Defs = [R29, R30, R31, PC], Uses = [R29, R31], neverHasSideEffects = 1 in {
  def DEALLOC_RET_cNotdnPt_V4 : NVInst_V4<(outs), (ins PredRegs:$src1,
                                                       i32imm:$amt1),
            "if (!$src1.new) dealloc_return:t",
            []>,
            Requires<[HasV4T]>;
}
