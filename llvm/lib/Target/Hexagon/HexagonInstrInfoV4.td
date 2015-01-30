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

let hasSideEffects = 0 in
class T_Immext<Operand ImmType>
  : EXTENDERInst<(outs), (ins ImmType:$imm),
                 "immext(#$imm)", []> {
    bits<32> imm;
    let IClass = 0b0000;

    let Inst{27-16} = imm{31-20};
    let Inst{13-0} = imm{19-6};
  }

def A4_ext : T_Immext<u26_6Imm>;
let isCodeGenOnly = 1 in {
  let isBranch = 1 in
    def A4_ext_b : T_Immext<brtarget>;
  let isCall = 1 in
    def A4_ext_c : T_Immext<calltarget>;
  def A4_ext_g : T_Immext<globaladdress>;
}

def BITPOS32 : SDNodeXForm<imm, [{
   // Return the bit position we will set [0-31].
   // As an SDNode.
   int32_t imm = N->getSExtValue();
   return XformMskToBitPosU5Imm(imm);
}]>;

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

class T_ALU32_3op_not<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                      bit OpsRev>
  : T_ALU32_3op<mnemonic, MajOp, MinOp, OpsRev, 0> {
  let AsmString = "$Rd = "#mnemonic#"($Rs, ~$Rt)";
}

let BaseOpcode = "andn_rr", CextOpcode = "andn", isCodeGenOnly = 0 in
def A4_andn    : T_ALU32_3op_not<"and", 0b001, 0b100, 1>;
let BaseOpcode = "orn_rr", CextOpcode = "orn", isCodeGenOnly = 0 in
def A4_orn     : T_ALU32_3op_not<"or",  0b001, 0b101, 1>;

let CextOpcode = "rcmp.eq", isCodeGenOnly = 0 in
def A4_rcmpeq  : T_ALU32_3op<"cmp.eq",  0b011, 0b010, 0, 1>;
let CextOpcode = "!rcmp.eq", isCodeGenOnly = 0 in
def A4_rcmpneq : T_ALU32_3op<"!cmp.eq", 0b011, 0b011, 0, 1>;

let isCodeGenOnly = 0 in {
def C4_cmpneq  : T_ALU32_3op_cmp<"!cmp.eq",  0b00, 1, 1>;
def C4_cmplte  : T_ALU32_3op_cmp<"!cmp.gt",  0b10, 1, 0>;
def C4_cmplteu : T_ALU32_3op_cmp<"!cmp.gtu", 0b11, 1, 0>;
}

// Pats for instruction selection.

// A class to embed the usual comparison patfrags within a zext to i32.
// The seteq/setne frags use "lhs" and "rhs" as operands, so use the same
// names, or else the frag's "body" won't match the operands.
class CmpInReg<PatFrag Op>
  : PatFrag<(ops node:$lhs, node:$rhs),(i32 (zext (i1 Op.Fragment)))>;

def: T_cmp32_rr_pat<A4_rcmpeq,  CmpInReg<seteq>, i32>;
def: T_cmp32_rr_pat<A4_rcmpneq, CmpInReg<setne>, i32>;

def: T_cmp32_rr_pat<C4_cmpneq,  setne,  i1>;

class T_CMP_rrbh<string mnemonic, bits<3> MinOp, bit IsComm>
  : SInst<(outs PredRegs:$Pd), (ins IntRegs:$Rs, IntRegs:$Rt),
    "$Pd = "#mnemonic#"($Rs, $Rt)", [], "", S_3op_tc_2early_SLOT23>,
    ImmRegRel {
  let validSubTargets = HasV4SubT;
  let InputType = "reg";
  let CextOpcode = mnemonic;
  let isCompare = 1;
  let isCommutable = IsComm;
  let hasSideEffects = 0;

  bits<2> Pd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1100;
  let Inst{27-21} = 0b0111110;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{7-5} = MinOp;
  let Inst{1-0} = Pd;
}

let isCodeGenOnly = 0 in {
def A4_cmpbeq  : T_CMP_rrbh<"cmpb.eq",  0b110, 1>;
def A4_cmpbgt  : T_CMP_rrbh<"cmpb.gt",  0b010, 0>;
def A4_cmpbgtu : T_CMP_rrbh<"cmpb.gtu", 0b111, 0>;
def A4_cmpheq  : T_CMP_rrbh<"cmph.eq",  0b011, 1>;
def A4_cmphgt  : T_CMP_rrbh<"cmph.gt",  0b100, 0>;
def A4_cmphgtu : T_CMP_rrbh<"cmph.gtu", 0b101, 0>;
}

class T_CMP_ribh<string mnemonic, bits<2> MajOp, bit IsHalf, bit IsComm,
                 Operand ImmType, bit IsImmExt, bit IsImmSigned, int ImmBits>
  : ALU64Inst<(outs PredRegs:$Pd), (ins IntRegs:$Rs, ImmType:$Imm),
    "$Pd = "#mnemonic#"($Rs, #$Imm)", [], "", ALU64_tc_2early_SLOT23>,
    ImmRegRel {
  let validSubTargets = HasV4SubT;
  let InputType = "imm";
  let CextOpcode = mnemonic;
  let isCompare = 1;
  let isCommutable = IsComm;
  let hasSideEffects = 0;
  let isExtendable = IsImmExt;
  let opExtendable = !if (IsImmExt, 2, 0);
  let isExtentSigned = IsImmSigned;
  let opExtentBits = ImmBits;

  bits<2> Pd;
  bits<5> Rs;
  bits<8> Imm;

  let IClass = 0b1101;
  let Inst{27-24} = 0b1101;
  let Inst{22-21} = MajOp;
  let Inst{20-16} = Rs;
  let Inst{12-5} = Imm;
  let Inst{4} = 0b0;
  let Inst{3} = IsHalf;
  let Inst{1-0} = Pd;
}

let isCodeGenOnly = 0 in {
def A4_cmpbeqi  : T_CMP_ribh<"cmpb.eq",  0b00, 0, 1, u8Imm, 0, 0, 8>;
def A4_cmpbgti  : T_CMP_ribh<"cmpb.gt",  0b01, 0, 0, s8Imm, 0, 1, 8>;
def A4_cmpbgtui : T_CMP_ribh<"cmpb.gtu", 0b10, 0, 0, u7Ext, 1, 0, 7>;
def A4_cmpheqi  : T_CMP_ribh<"cmph.eq",  0b00, 1, 1, s8Ext, 1, 1, 8>;
def A4_cmphgti  : T_CMP_ribh<"cmph.gt",  0b01, 1, 0, s8Ext, 1, 1, 8>;
def A4_cmphgtui : T_CMP_ribh<"cmph.gtu", 0b10, 1, 0, u7Ext, 1, 0, 7>;
}
class T_RCMP_EQ_ri<string mnemonic, bit IsNeg>
  : ALU32_ri<(outs IntRegs:$Rd), (ins IntRegs:$Rs, s8Ext:$s8),
    "$Rd = "#mnemonic#"($Rs, #$s8)", [], "", ALU32_2op_tc_1_SLOT0123>,
    ImmRegRel {
  let validSubTargets = HasV4SubT;
  let InputType = "imm";
  let CextOpcode = !if (IsNeg, "!rcmp.eq", "rcmp.eq");
  let isExtendable = 1;
  let opExtendable = 2;
  let isExtentSigned = 1;
  let opExtentBits = 8;
  let hasNewValue = 1;

  bits<5> Rd;
  bits<5> Rs;
  bits<8> s8;

  let IClass = 0b0111;
  let Inst{27-24} = 0b0011;
  let Inst{22} = 0b1;
  let Inst{21} = IsNeg;
  let Inst{20-16} = Rs;
  let Inst{13} = 0b1;
  let Inst{12-5} = s8;
  let Inst{4-0} = Rd;
}

let isCodeGenOnly = 0 in {
def A4_rcmpeqi  : T_RCMP_EQ_ri<"cmp.eq",  0>;
def A4_rcmpneqi : T_RCMP_EQ_ri<"!cmp.eq", 1>;
}

def: Pat<(i32 (zext (i1 (seteq (i32 IntRegs:$Rs), s8ExtPred:$s8)))),
         (A4_rcmpeqi IntRegs:$Rs, s8ExtPred:$s8)>;
def: Pat<(i32 (zext (i1 (setne (i32 IntRegs:$Rs), s8ExtPred:$s8)))),
         (A4_rcmpneqi IntRegs:$Rs, s8ExtPred:$s8)>;

// Preserve the S2_tstbit_r generation
def: Pat<(i32 (zext (i1 (setne (i32 (and (i32 (shl 1, (i32 IntRegs:$src2))),
                                         (i32 IntRegs:$src1))), 0)))),
         (C2_muxii (S2_tstbit_r IntRegs:$src1, IntRegs:$src2), 1, 0)>;


//===----------------------------------------------------------------------===//
// ALU32 -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//

// Combine a word and an immediate into a register pair.
let hasSideEffects = 0, isExtentSigned = 1, isExtendable = 1,
    opExtentBits = 8 in
class T_Combine1 <bits<2> MajOp, dag ins, string AsmStr>
  : ALU32Inst <(outs DoubleRegs:$Rdd), ins, AsmStr> {
    bits<5> Rdd;
    bits<5> Rs;
    bits<8> s8;

    let IClass      = 0b0111;
    let Inst{27-24} = 0b0011;
    let Inst{22-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b1;
    let Inst{12-5}  = s8;
    let Inst{4-0}   = Rdd;
  }

let opExtendable = 2, isCodeGenOnly = 0 in
def A4_combineri : T_Combine1<0b00, (ins IntRegs:$Rs, s8Ext:$s8),
                                    "$Rdd = combine($Rs, #$s8)">;

let opExtendable = 1, isCodeGenOnly = 0 in
def A4_combineir : T_Combine1<0b01, (ins s8Ext:$s8, IntRegs:$Rs),
                                    "$Rdd = combine(#$s8, $Rs)">;

def HexagonWrapperCombineRI_V4 :
  SDNode<"HexagonISD::WrapperCombineRI_V4", SDTHexagonI64I32I32>;
def HexagonWrapperCombineIR_V4 :
  SDNode<"HexagonISD::WrapperCombineIR_V4", SDTHexagonI64I32I32>;

def : Pat <(HexagonWrapperCombineRI_V4 IntRegs:$r, s8ExtPred:$i),
           (A4_combineri IntRegs:$r, s8ExtPred:$i)>,
          Requires<[HasV4T]>;

def : Pat <(HexagonWrapperCombineIR_V4 s8ExtPred:$i, IntRegs:$r),
           (A4_combineir s8ExtPred:$i, IntRegs:$r)>,
          Requires<[HasV4T]>;

// A4_combineii: Set two small immediates.
let hasSideEffects = 0, isExtendable = 1, opExtentBits = 6, opExtendable = 2 in
def A4_combineii: ALU32Inst<(outs DoubleRegs:$Rdd), (ins s8Imm:$s8, u6Ext:$U6),
  "$Rdd = combine(#$s8, #$U6)"> {
    bits<5> Rdd;
    bits<8> s8;
    bits<6> U6;

    let IClass = 0b0111;
    let Inst{27-23} = 0b11001;
    let Inst{20-16} = U6{5-1};
    let Inst{13}    = U6{0};
    let Inst{12-5}  = s8;
    let Inst{4-0}   = Rdd;
  }

//===----------------------------------------------------------------------===//
// ALU32/PERM -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LD +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Template class for load instructions with Absolute set addressing mode.
//===----------------------------------------------------------------------===//
let isExtended = 1, opExtendable = 2, opExtentBits = 6, addrMode = AbsoluteSet,
    hasSideEffects = 0 in
class T_LD_abs_set<string mnemonic, RegisterClass RC, bits<4>MajOp>:
            LDInst<(outs RC:$dst1, IntRegs:$dst2),
            (ins u6Ext:$addr),
            "$dst1 = "#mnemonic#"($dst2 = #$addr)",
            []> {
  bits<7> name;
  bits<5> dst1;
  bits<5> dst2;
  bits<6> addr;

  let IClass = 0b1001;
  let Inst{27-25} = 0b101;
  let Inst{24-21} = MajOp;
  let Inst{13-12} = 0b01;
  let Inst{4-0}   = dst1;
  let Inst{20-16} = dst2;
  let Inst{11-8}  = addr{5-2};
  let Inst{6-5}   = addr{1-0};
}

let accessSize = ByteAccess, hasNewValue = 1, isCodeGenOnly = 0 in {
  def L4_loadrb_ap   : T_LD_abs_set <"memb",   IntRegs, 0b1000>;
  def L4_loadrub_ap  : T_LD_abs_set <"memub",  IntRegs, 0b1001>;
}

let accessSize = HalfWordAccess, hasNewValue = 1, isCodeGenOnly = 0 in {
  def L4_loadrh_ap  : T_LD_abs_set <"memh",  IntRegs, 0b1010>;
  def L4_loadruh_ap : T_LD_abs_set <"memuh", IntRegs, 0b1011>;
}

let accessSize = WordAccess, hasNewValue = 1, isCodeGenOnly = 0 in
  def L4_loadri_ap : T_LD_abs_set <"memw", IntRegs, 0b1100>;

let accessSize = DoubleWordAccess, isCodeGenOnly = 0 in
def L4_loadrd_ap : T_LD_abs_set <"memd", DoubleRegs, 0b1110>;

//===----------------------------------------------------------------------===//
// Template classes for the non-predicated load instructions with
// base + register offset addressing mode
//===----------------------------------------------------------------------===//
class T_load_rr <string mnemonic, RegisterClass RC, bits<3> MajOp>:
   LDInst<(outs RC:$dst), (ins IntRegs:$src1, IntRegs:$src2, u2Imm:$u2),
  "$dst = "#mnemonic#"($src1 + $src2<<#$u2)",
  [], "", V4LDST_tc_ld_SLOT01>, ImmRegShl, AddrModeRel {
    bits<5> dst;
    bits<5> src1;
    bits<5> src2;
    bits<2> u2;

    let IClass = 0b0011;

    let Inst{27-24} = 0b1010;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{12-8}  = src2;
    let Inst{13}    = u2{1};
    let Inst{7}     = u2{0};
    let Inst{4-0}   = dst;
  }

//===----------------------------------------------------------------------===//
// Template classes for the predicated load instructions with
// base + register offset addressing mode
//===----------------------------------------------------------------------===//
let isPredicated =  1 in
class T_pload_rr <string mnemonic, RegisterClass RC, bits<3> MajOp,
                  bit isNot, bit isPredNew>:
   LDInst <(outs RC:$dst),
           (ins PredRegs:$src1, IntRegs:$src2, IntRegs:$src3, u2Imm:$u2),
  !if(isNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#"$dst = "#mnemonic#"($src2+$src3<<#$u2)",
  [], "", V4LDST_tc_ld_SLOT01>, AddrModeRel {
    bits<5> dst;
    bits<2> src1;
    bits<5> src2;
    bits<5> src3;
    bits<2> u2;

    let isPredicatedFalse = isNot;
    let isPredicatedNew = isPredNew;

    let IClass = 0b0011;

    let Inst{27-26} = 0b00;
    let Inst{25}    = isPredNew;
    let Inst{24}    = isNot;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{12-8}  = src3;
    let Inst{13}    = u2{1};
    let Inst{7}     = u2{0};
    let Inst{6-5}   = src1;
    let Inst{4-0}   = dst;
  }

//===----------------------------------------------------------------------===//
// multiclass for load instructions with base + register offset
// addressing mode
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, addrMode = BaseRegOffset in
multiclass ld_idxd_shl <string mnemonic, string CextOp, RegisterClass RC,
                        bits<3> MajOp > {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed_shl,
      InputType = "reg" in {
    let isPredicable = 1 in
    def L4_#NAME#_rr : T_load_rr <mnemonic, RC, MajOp>;

    // Predicated
    def L4_p#NAME#t_rr : T_pload_rr <mnemonic, RC, MajOp, 0, 0>;
    def L4_p#NAME#f_rr : T_pload_rr <mnemonic, RC, MajOp, 1, 0>;

    // Predicated new
    def L4_p#NAME#tnew_rr : T_pload_rr <mnemonic, RC, MajOp, 0, 1>;
    def L4_p#NAME#fnew_rr : T_pload_rr <mnemonic, RC, MajOp, 1, 1>;
  }
}

let hasNewValue = 1, accessSize = ByteAccess, isCodeGenOnly = 0 in {
  defm loadrb  : ld_idxd_shl<"memb", "LDrib", IntRegs, 0b000>;
  defm loadrub : ld_idxd_shl<"memub", "LDriub", IntRegs, 0b001>;
}

let hasNewValue = 1, accessSize = HalfWordAccess, isCodeGenOnly = 0 in {
  defm loadrh  : ld_idxd_shl<"memh", "LDrih", IntRegs, 0b010>;
  defm loadruh : ld_idxd_shl<"memuh", "LDriuh", IntRegs, 0b011>;
}

let hasNewValue = 1, accessSize = WordAccess, isCodeGenOnly = 0 in
defm loadri : ld_idxd_shl<"memw", "LDriw", IntRegs, 0b100>;

let accessSize = DoubleWordAccess, isCodeGenOnly = 0 in
defm loadrd  : ld_idxd_shl<"memd", "LDrid", DoubleRegs, 0b110>;

// 'def pats' for load instructions with base + register offset and non-zero
// immediate value. Immediate value is used to left-shift the second
// register operand.
let AddedComplexity = 40 in {
def : Pat <(i32 (sextloadi8 (add IntRegs:$src1,
                                 (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadrb_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (zextloadi8 (add IntRegs:$src1,
                                 (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadrub_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (extloadi8 (add IntRegs:$src1,
                                (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadrub_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (sextloadi16 (add IntRegs:$src1,
                                  (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadrh_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (zextloadi16 (add IntRegs:$src1,
                                  (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadruh_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (extloadi16 (add IntRegs:$src1,
                                 (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadruh_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i32 (load (add IntRegs:$src1,
                           (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadri_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;

def : Pat <(i64 (load (add IntRegs:$src1,
                           (shl IntRegs:$src2, u2ImmPred:$offset)))),
           (L4_loadrd_rr IntRegs:$src1,
            IntRegs:$src2, u2ImmPred:$offset)>,
            Requires<[HasV4T]>;
}

// 'def pats' for load instruction base + register offset and
// zero immediate value.
class Loadxs_simple_pat<PatFrag Load, ValueType VT, InstHexagon MI>
  : Pat<(VT (Load (add (i32 IntRegs:$Rs), (i32 IntRegs:$Rt)))),
        (VT (MI IntRegs:$Rs, IntRegs:$Rt, 0))>;

let AddedComplexity = 20 in {
  def: Loadxs_simple_pat<extloadi8,   i32, L4_loadrub_rr>;
  def: Loadxs_simple_pat<zextloadi8,  i32, L4_loadrub_rr>;
  def: Loadxs_simple_pat<sextloadi8,  i32, L4_loadrb_rr>;
  def: Loadxs_simple_pat<extloadi16,  i32, L4_loadruh_rr>;
  def: Loadxs_simple_pat<zextloadi16, i32, L4_loadruh_rr>;
  def: Loadxs_simple_pat<sextloadi16, i32, L4_loadrh_rr>;
  def: Loadxs_simple_pat<load,        i32, L4_loadri_rr>;
  def: Loadxs_simple_pat<load,        i64, L4_loadrd_rr>;
}

// zext i1->i64
def : Pat <(i64 (zext (i1 PredRegs:$src1))),
      (i64 (A4_combineir 0, (C2_muxii (i1 PredRegs:$src1), 1, 0)))>,
      Requires<[HasV4T]>;

// zext i32->i64
def : Pat <(i64 (zext (i32 IntRegs:$src1))),
      (i64 (A4_combineir 0, (i32 IntRegs:$src1)))>,
      Requires<[HasV4T]>;
// zext i8->i64
def:  Pat <(i64 (zextloadi8 ADDRriS11_0:$src1)),
      (i64 (A4_combineir 0, (L2_loadrub_io AddrFI:$src1, 0)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (zextloadi8 (add (i32 IntRegs:$src1),
                                s11_0ExtPred:$offset))),
      (i64 (A4_combineir 0, (L2_loadrub_io IntRegs:$src1,
                                  s11_0ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// zext i1->i64
def:  Pat <(i64 (zextloadi1 ADDRriS11_0:$src1)),
      (i64 (A4_combineir 0, (L2_loadrub_io AddrFI:$src1, 0)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (zextloadi1 (add (i32 IntRegs:$src1),
                                s11_0ExtPred:$offset))),
      (i64 (A4_combineir 0, (L2_loadrub_io IntRegs:$src1,
                                  s11_0ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// zext i16->i64
def:  Pat <(i64 (zextloadi16 ADDRriS11_1:$src1)),
      (i64 (A4_combineir 0, (L2_loadruh_io AddrFI:$src1, 0)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (zextloadi16 (add (i32 IntRegs:$src1),
                                  s11_1ExtPred:$offset))),
      (i64 (A4_combineir 0, (L2_loadruh_io IntRegs:$src1,
                                  s11_1ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// anyext i16->i64
def:  Pat <(i64 (extloadi16 ADDRriS11_2:$src1)),
      (i64 (A4_combineir 0, (L2_loadrh_io AddrFI:$src1, 0)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 20 in
def:  Pat <(i64 (extloadi16 (add (i32 IntRegs:$src1),
                                  s11_1ExtPred:$offset))),
      (i64 (A4_combineir 0, (L2_loadrh_io IntRegs:$src1,
                                  s11_1ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// zext i32->i64
def:  Pat <(i64 (zextloadi32 ADDRriS11_2:$src1)),
      (i64 (A4_combineir 0, (L2_loadri_io AddrFI:$src1, 0)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 100 in
def:  Pat <(i64 (zextloadi32 (i32 (add IntRegs:$src1, s11_2ExtPred:$offset)))),
      (i64 (A4_combineir 0, (L2_loadri_io IntRegs:$src1,
                                  s11_2ExtPred:$offset)))>,
      Requires<[HasV4T]>;

// anyext i32->i64
def:  Pat <(i64 (extloadi32 ADDRriS11_2:$src1)),
      (i64 (A4_combineir 0, (L2_loadri_io AddrFI:$src1, 0)))>,
      Requires<[HasV4T]>;

let AddedComplexity = 100 in
def:  Pat <(i64 (extloadi32 (i32 (add IntRegs:$src1, s11_2ExtPred:$offset)))),
      (i64 (A4_combineir 0, (L2_loadri_io IntRegs:$src1,
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
// Template classes for the non-predicated store instructions with
// base + register offset addressing mode
//===----------------------------------------------------------------------===//
let isPredicable = 1 in
class T_store_rr <string mnemonic, RegisterClass RC, bits<3> MajOp, bit isH>
  : STInst < (outs ), (ins IntRegs:$Rs, IntRegs:$Ru, u2Imm:$u2, RC:$Rt),
  mnemonic#"($Rs + $Ru<<#$u2) = $Rt"#!if(isH, ".h",""),
  [],"",V4LDST_tc_st_SLOT01>, ImmRegShl, AddrModeRel {

    bits<5> Rs;
    bits<5> Ru;
    bits<2> u2;
    bits<5> Rt;

    let IClass = 0b0011;

    let Inst{27-24} = 0b1011;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Ru;
    let Inst{13}    = u2{1};
    let Inst{7}     = u2{0};
    let Inst{4-0}   = Rt;
  }

//===----------------------------------------------------------------------===//
// Template classes for the predicated store instructions with
// base + register offset addressing mode
//===----------------------------------------------------------------------===//
let isPredicated = 1 in
class T_pstore_rr <string mnemonic, RegisterClass RC, bits<3> MajOp,
                   bit isNot, bit isPredNew, bit isH>
  : STInst <(outs),
            (ins PredRegs:$Pv, IntRegs:$Rs, IntRegs:$Ru, u2Imm:$u2, RC:$Rt),

  !if(isNot, "if (!$Pv", "if ($Pv")#!if(isPredNew, ".new) ",
  ") ")#mnemonic#"($Rs+$Ru<<#$u2) = $Rt"#!if(isH, ".h",""),
  [], "", V4LDST_tc_st_SLOT01> , AddrModeRel{
    bits<2> Pv;
    bits<5> Rs;
    bits<5> Ru;
    bits<2> u2;
    bits<5> Rt;

    let isPredicatedFalse = isNot;
    let isPredicatedNew = isPredNew;

    let IClass = 0b0011;

    let Inst{27-26} = 0b01;
    let Inst{25}    = isPredNew;
    let Inst{24}    = isNot;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Ru;
    let Inst{13}    = u2{1};
    let Inst{7}     = u2{0};
    let Inst{6-5}   = Pv;
    let Inst{4-0}   = Rt;
  }

//===----------------------------------------------------------------------===//
// Template classes for the new-value store instructions with
// base + register offset addressing mode
//===----------------------------------------------------------------------===//
let isPredicable = 1, isNewValue = 1, opNewValue = 3 in
class T_store_new_rr <string mnemonic, bits<2> MajOp> :
  NVInst < (outs ), (ins IntRegs:$Rs, IntRegs:$Ru, u2Imm:$u2, IntRegs:$Nt),
  mnemonic#"($Rs + $Ru<<#$u2) = $Nt.new",
  [],"",V4LDST_tc_st_SLOT0>, ImmRegShl, AddrModeRel {

    bits<5> Rs;
    bits<5> Ru;
    bits<2> u2;
    bits<3> Nt;

    let IClass = 0b0011;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Ru;
    let Inst{13}    = u2{1};
    let Inst{7}     = u2{0};
    let Inst{4-3}   = MajOp;
    let Inst{2-0}   = Nt;
  }

//===----------------------------------------------------------------------===//
// Template classes for the predicated new-value store instructions with
// base + register offset addressing mode
//===----------------------------------------------------------------------===//
let isPredicated = 1, isNewValue = 1, opNewValue = 4 in
class T_pstore_new_rr <string mnemonic, bits<2> MajOp, bit isNot, bit isPredNew>
  : NVInst<(outs),
           (ins PredRegs:$Pv, IntRegs:$Rs, IntRegs:$Ru, u2Imm:$u2, IntRegs:$Nt),
   !if(isNot, "if (!$Pv", "if ($Pv")#!if(isPredNew, ".new) ",
   ") ")#mnemonic#"($Rs+$Ru<<#$u2) = $Nt.new",
   [], "", V4LDST_tc_st_SLOT0>, AddrModeRel {
    bits<2> Pv;
    bits<5> Rs;
    bits<5> Ru;
    bits<2> u2;
    bits<3> Nt;

    let isPredicatedFalse = isNot;
    let isPredicatedNew = isPredNew;

    let IClass = 0b0011;
    let Inst{27-26} = 0b01;
    let Inst{25}    = isPredNew;
    let Inst{24}    = isNot;
    let Inst{23-21} = 0b101;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Ru;
    let Inst{13}    = u2{1};
    let Inst{7}     = u2{0};
    let Inst{6-5}   = Pv;
    let Inst{4-3}   = MajOp;
    let Inst{2-0}   = Nt;
  }

//===----------------------------------------------------------------------===//
// multiclass for store instructions with base + register offset addressing
// mode
//===----------------------------------------------------------------------===//
let isNVStorable = 1 in
multiclass ST_Idxd_shl<string mnemonic, string CextOp, RegisterClass RC,
                       bits<3> MajOp, bit isH = 0> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed_shl in {
    def S4_#NAME#_rr : T_store_rr <mnemonic, RC, MajOp, isH>;

    // Predicated
    def S4_p#NAME#t_rr : T_pstore_rr <mnemonic, RC, MajOp, 0, 0, isH>;
    def S4_p#NAME#f_rr : T_pstore_rr <mnemonic, RC, MajOp, 1, 0, isH>;

    // Predicated new
    def S4_p#NAME#tnew_rr : T_pstore_rr <mnemonic, RC, MajOp, 0, 1, isH>;
    def S4_p#NAME#fnew_rr : T_pstore_rr <mnemonic, RC, MajOp, 1, 1, isH>;
  }
}

//===----------------------------------------------------------------------===//
// multiclass for new-value store instructions with base + register offset
// addressing mode.
//===----------------------------------------------------------------------===//
let mayStore = 1, isNVStore = 1 in
multiclass ST_Idxd_shl_nv <string mnemonic, string CextOp, RegisterClass RC,
                           bits<2> MajOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed_shl in {
    def S4_#NAME#new_rr : T_store_new_rr<mnemonic, MajOp>;

    // Predicated
    def S4_p#NAME#newt_rr : T_pstore_new_rr <mnemonic, MajOp, 0, 0>;
    def S4_p#NAME#newf_rr : T_pstore_new_rr <mnemonic, MajOp, 1, 0>;

    // Predicated new
    def S4_p#NAME#newtnew_rr : T_pstore_new_rr <mnemonic, MajOp, 0, 1>;
    def S4_p#NAME#newfnew_rr : T_pstore_new_rr <mnemonic, MajOp, 1, 1>;
  }
}

let addrMode = BaseRegOffset, InputType = "reg", hasSideEffects = 0,
    isCodeGenOnly = 0 in {
  let accessSize = ByteAccess in
  defm storerb: ST_Idxd_shl<"memb", "STrib", IntRegs, 0b000>,
                ST_Idxd_shl_nv<"memb", "STrib", IntRegs, 0b00>;

  let accessSize = HalfWordAccess in
  defm storerh: ST_Idxd_shl<"memh", "STrih", IntRegs, 0b010>,
                ST_Idxd_shl_nv<"memh", "STrih", IntRegs, 0b01>;

  let accessSize = WordAccess in
  defm storeri: ST_Idxd_shl<"memw", "STriw", IntRegs, 0b100>,
                ST_Idxd_shl_nv<"memw", "STriw", IntRegs, 0b10>;

  let isNVStorable = 0, accessSize = DoubleWordAccess in
  defm storerd: ST_Idxd_shl<"memd", "STrid", DoubleRegs, 0b110>;

  let isNVStorable = 0, accessSize = HalfWordAccess in
  defm storerf: ST_Idxd_shl<"memh", "STrif", IntRegs, 0b011, 1>;
}

let Predicates = [HasV4T], AddedComplexity = 10 in {
def : Pat<(truncstorei8 (i32 IntRegs:$src4),
                       (add IntRegs:$src1, (shl IntRegs:$src2,
                                                u2ImmPred:$src3))),
          (S4_storerb_rr IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, IntRegs:$src4)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src4),
                        (add IntRegs:$src1, (shl IntRegs:$src2,
                                                 u2ImmPred:$src3))),
          (S4_storerh_rr IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, IntRegs:$src4)>;

def : Pat<(store (i32 IntRegs:$src4),
                 (add IntRegs:$src1, (shl IntRegs:$src2, u2ImmPred:$src3))),
          (S4_storeri_rr IntRegs:$src1, IntRegs:$src2,
                                u2ImmPred:$src3, IntRegs:$src4)>;

def : Pat<(store (i64 DoubleRegs:$src4),
                (add IntRegs:$src1, (shl IntRegs:$src2, u2ImmPred:$src3))),
          (S4_storerd_rr IntRegs:$src1, IntRegs:$src2,
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
// Template class
//===----------------------------------------------------------------------===//
let isPredicable = 1, isExtendable = 1, isExtentSigned = 1, opExtentBits = 8,
    opExtendable = 2 in
class T_StoreImm <string mnemonic, Operand OffsetOp, bits<2> MajOp >
  : STInst <(outs ), (ins IntRegs:$Rs, OffsetOp:$offset, s8Ext:$S8),
  mnemonic#"($Rs+#$offset)=#$S8",
  [], "", V4LDST_tc_st_SLOT01>,
  ImmRegRel, PredNewRel {
    bits<5> Rs;
    bits<8> S8;
    bits<8> offset;
    bits<6> offsetBits;

    string OffsetOpStr = !cast<string>(OffsetOp);
    let offsetBits = !if (!eq(OffsetOpStr, "u6_2Imm"), offset{7-2},
                     !if (!eq(OffsetOpStr, "u6_1Imm"), offset{6-1},
                                         /* u6_0Imm */ offset{5-0}));

    let IClass = 0b0011;

    let Inst{27-25} = 0b110;
    let Inst{22-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{12-7}  = offsetBits;
    let Inst{13}    = S8{7};
    let Inst{6-0}   = S8{6-0};
  }

let isPredicated = 1, isExtendable = 1, isExtentSigned = 1, opExtentBits = 6,
    opExtendable = 3 in
class T_StoreImm_pred <string mnemonic, Operand OffsetOp, bits<2> MajOp,
                       bit isPredNot, bit isPredNew >
  : STInst <(outs ),
            (ins PredRegs:$Pv, IntRegs:$Rs, OffsetOp:$offset, s6Ext:$S6),
  !if(isPredNot, "if (!$Pv", "if ($Pv")#!if(isPredNew, ".new) ",
  ") ")#mnemonic#"($Rs+#$offset)=#$S6",
  [], "", V4LDST_tc_st_SLOT01>,
  ImmRegRel, PredNewRel {
    bits<2> Pv;
    bits<5> Rs;
    bits<6> S6;
    bits<8> offset;
    bits<6> offsetBits;

    string OffsetOpStr = !cast<string>(OffsetOp);
    let offsetBits = !if (!eq(OffsetOpStr, "u6_2Imm"), offset{7-2},
                     !if (!eq(OffsetOpStr, "u6_1Imm"), offset{6-1},
                                         /* u6_0Imm */ offset{5-0}));
    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = isPredNot;

    let IClass = 0b0011;

    let Inst{27-25} = 0b100;
    let Inst{24}    = isPredNew;
    let Inst{23}    = isPredNot;
    let Inst{22-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{13}    = S6{5};
    let Inst{12-7}  = offsetBits;
    let Inst{6-5}   = Pv;
    let Inst{4-0}   = S6{4-0};
  }


//===----------------------------------------------------------------------===//
// multiclass for store instructions with base + immediate offset
// addressing mode and immediate stored value.
// mem[bhw](Rx++#s4:3)=#s8
// if ([!]Pv[.new]) mem[bhw](Rx++#s4:3)=#s6
//===----------------------------------------------------------------------===//

multiclass ST_Imm_Pred <string mnemonic, Operand OffsetOp, bits<2> MajOp,
                        bit PredNot> {
  def _io    : T_StoreImm_pred <mnemonic, OffsetOp, MajOp, PredNot, 0>;
  // Predicate new
  def new_io : T_StoreImm_pred <mnemonic, OffsetOp, MajOp, PredNot, 1>;
}

multiclass ST_Imm <string mnemonic, string CextOp, Operand OffsetOp,
                   bits<2> MajOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_imm in {
    def _io : T_StoreImm <mnemonic, OffsetOp, MajOp>;

    defm t : ST_Imm_Pred <mnemonic, OffsetOp, MajOp, 0>;
    defm f : ST_Imm_Pred <mnemonic, OffsetOp, MajOp, 1>;
  }
}

let hasSideEffects = 0, validSubTargets = HasV4SubT, addrMode = BaseImmOffset,
    InputType = "imm", isCodeGenOnly = 0 in {
  let accessSize = ByteAccess in
  defm S4_storeirb : ST_Imm<"memb", "STrib", u6_0Imm, 0b00>;

  let accessSize = HalfWordAccess in
  defm S4_storeirh : ST_Imm<"memh", "STrih", u6_1Imm, 0b01>;

  let accessSize = WordAccess in
  defm S4_storeiri : ST_Imm<"memw", "STriw", u6_2Imm, 0b10>;
}

let Predicates = [HasV4T], AddedComplexity = 10 in {
def: Pat<(truncstorei8 s8ExtPred:$src3, (add IntRegs:$src1, u6_0ImmPred:$src2)),
            (S4_storeirb_io IntRegs:$src1, u6_0ImmPred:$src2, s8ExtPred:$src3)>;

def: Pat<(truncstorei16 s8ExtPred:$src3, (add IntRegs:$src1,
                                              u6_1ImmPred:$src2)),
            (S4_storeirh_io IntRegs:$src1, u6_1ImmPred:$src2, s8ExtPred:$src3)>;

def: Pat<(store s8ExtPred:$src3, (add IntRegs:$src1, u6_2ImmPred:$src2)),
            (S4_storeiri_io IntRegs:$src1, u6_2ImmPred:$src2, s8ExtPred:$src3)>;
}

let AddedComplexity = 6 in
def : Pat <(truncstorei8 s8ExtPred:$src2, (i32 IntRegs:$src1)),
           (S4_storeirb_io IntRegs:$src1, 0, s8ExtPred:$src2)>,
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
           (S4_storeirh_io IntRegs:$src1, 0, s8ExtPred:$src2)>,
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
let hasSideEffects = 0 in
def STriw_pred_V4 : STInst2<(outs),
            (ins MEMri:$addr, PredRegs:$src1),
            "Error; should not emit",
            []>,
            Requires<[HasV4T]>;

let AddedComplexity = 6 in
def : Pat <(store s8ExtPred:$src2, (i32 IntRegs:$src1)),
           (S4_storeiri_io IntRegs:$src1, 0, s8ExtPred:$src2)>,
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

let opNewValue = 2, opExtendable = 1, isExtentSigned = 1, isPredicable = 1 in
class T_store_io_nv <string mnemonic, RegisterClass RC,
                    Operand ImmOp, bits<2>MajOp>
  : NVInst_V4 <(outs),
               (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
  mnemonic#"($src1+#$src2) = $src3.new",
  [],"",ST_tc_st_SLOT0> {
    bits<5> src1;
    bits<13> src2; // Actual address offset
    bits<3> src3;
    bits<11> offsetBits; // Represents offset encoding

    let opExtentBits = !if (!eq(mnemonic, "memb"), 11,
                       !if (!eq(mnemonic, "memh"), 12,
                       !if (!eq(mnemonic, "memw"), 13, 0)));

    let opExtentAlign = !if (!eq(mnemonic, "memb"), 0,
                        !if (!eq(mnemonic, "memh"), 1,
                        !if (!eq(mnemonic, "memw"), 2, 0)));

    let offsetBits = !if (!eq(mnemonic, "memb"),  src2{10-0},
                     !if (!eq(mnemonic, "memh"),  src2{11-1},
                     !if (!eq(mnemonic, "memw"),  src2{12-2}, 0)));

    let IClass = 0b1010;

    let Inst{27} = 0b0;
    let Inst{26-25} = offsetBits{10-9};
    let Inst{24-21} = 0b1101;
    let Inst{20-16} = src1;
    let Inst{13} = offsetBits{8};
    let Inst{12-11} = MajOp;
    let Inst{10-8} = src3;
    let Inst{7-0} = offsetBits{7-0};
  }

let opExtendable = 2, opNewValue = 3, isPredicated = 1 in
class T_pstore_io_nv <string mnemonic, RegisterClass RC, Operand predImmOp,
                         bits<2>MajOp, bit PredNot, bit isPredNew>
  : NVInst_V4 <(outs),
               (ins PredRegs:$src1, IntRegs:$src2, predImmOp:$src3, RC:$src4),
  !if(PredNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#mnemonic#"($src2+#$src3) = $src4.new",
  [],"",V2LDST_tc_st_SLOT0> {
    bits<2> src1;
    bits<5> src2;
    bits<9> src3;
    bits<3> src4;
    bits<6> offsetBits; // Represents offset encoding

    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = PredNot;
    let opExtentBits = !if (!eq(mnemonic, "memb"), 6,
                       !if (!eq(mnemonic, "memh"), 7,
                       !if (!eq(mnemonic, "memw"), 8, 0)));

    let opExtentAlign = !if (!eq(mnemonic, "memb"), 0,
                        !if (!eq(mnemonic, "memh"), 1,
                        !if (!eq(mnemonic, "memw"), 2, 0)));

    let offsetBits = !if (!eq(mnemonic, "memb"), src3{5-0},
                     !if (!eq(mnemonic, "memh"), src3{6-1},
                     !if (!eq(mnemonic, "memw"), src3{7-2}, 0)));

    let IClass = 0b0100;

    let Inst{27}    = 0b0;
    let Inst{26}    = PredNot;
    let Inst{25}    = isPredNew;
    let Inst{24-21} = 0b0101;
    let Inst{20-16} = src2;
    let Inst{13}    = offsetBits{5};
    let Inst{12-11} = MajOp;
    let Inst{10-8}  = src4;
    let Inst{7-3}   = offsetBits{4-0};
    let Inst{2}     = 0b0;
    let Inst{1-0}   = src1;
  }

// multiclass for new-value store instructions with base + immediate offset.
//
let mayStore = 1, isNVStore = 1, isNewValue = 1, hasSideEffects = 0,
    isExtendable = 1 in
multiclass ST_Idxd_nv<string mnemonic, string CextOp, RegisterClass RC,
                   Operand ImmOp, Operand predImmOp, bits<2> MajOp> {

  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed in {
    def S2_#NAME#new_io : T_store_io_nv <mnemonic, RC, ImmOp, MajOp>;
    // Predicated
    def S2_p#NAME#newt_io :T_pstore_io_nv <mnemonic, RC, predImmOp, MajOp, 0, 0>;
    def S2_p#NAME#newf_io :T_pstore_io_nv <mnemonic, RC, predImmOp, MajOp, 1, 0>;
    // Predicated new
    def S4_p#NAME#newtnew_io :T_pstore_io_nv <mnemonic, RC, predImmOp,
                                              MajOp, 0, 1>;
    def S4_p#NAME#newfnew_io :T_pstore_io_nv <mnemonic, RC, predImmOp,
                                              MajOp, 1, 1>;
  }
}

let addrMode = BaseImmOffset, InputType = "imm", isCodeGenOnly = 0 in {
  let accessSize = ByteAccess in
  defm storerb: ST_Idxd_nv<"memb", "STrib", IntRegs, s11_0Ext,
                           u6_0Ext, 0b00>, AddrModeRel;

  let accessSize = HalfWordAccess, opExtentAlign = 1 in
  defm storerh: ST_Idxd_nv<"memh", "STrih", IntRegs, s11_1Ext,
                           u6_1Ext, 0b01>, AddrModeRel;

  let accessSize = WordAccess, opExtentAlign = 2 in
  defm storeri: ST_Idxd_nv<"memw", "STriw", IntRegs, s11_2Ext,
                           u6_2Ext, 0b10>, AddrModeRel;
}

//===----------------------------------------------------------------------===//
// Post increment loads with register offset.
//===----------------------------------------------------------------------===//

let hasNewValue = 1, isCodeGenOnly = 0 in
def L2_loadbsw2_pr : T_load_pr <"membh", IntRegs, 0b0001, HalfWordAccess>;

let isCodeGenOnly = 0 in
def L2_loadbsw4_pr : T_load_pr <"membh", DoubleRegs, 0b0111, WordAccess>;

//===----------------------------------------------------------------------===//
// Template class for non-predicated post increment .new stores
// mem[bhwd](Rx++#s4:[0123])=Nt.new
//===----------------------------------------------------------------------===//
let isPredicable = 1, hasSideEffects = 0, validSubTargets = HasV4SubT,
    addrMode = PostInc, isNVStore = 1, isNewValue = 1, opNewValue = 3 in
class T_StorePI_nv <string mnemonic, Operand ImmOp, bits<2> MajOp >
  : NVInstPI_V4 <(outs IntRegs:$_dst_),
                 (ins IntRegs:$src1, ImmOp:$offset, IntRegs:$src2),
  mnemonic#"($src1++#$offset) = $src2.new",
  [], "$src1 = $_dst_">,
  AddrModeRel {
    bits<5> src1;
    bits<3> src2;
    bits<7> offset;
    bits<4> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0}));
    let IClass = 0b1010;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = src1;
    let Inst{13} = 0b0;
    let Inst{12-11} = MajOp;
    let Inst{10-8} = src2;
    let Inst{7} = 0b0;
    let Inst{6-3} = offsetBits;
    let Inst{1} = 0b0;
  }

//===----------------------------------------------------------------------===//
// Template class for predicated post increment .new stores
// if([!]Pv[.new]) mem[bhwd](Rx++#s4:[0123])=Nt.new
//===----------------------------------------------------------------------===//
let isPredicated = 1, hasSideEffects = 0, validSubTargets = HasV4SubT,
    addrMode = PostInc, isNVStore = 1, isNewValue = 1, opNewValue = 4 in
class T_StorePI_nv_pred <string mnemonic, Operand ImmOp,
                         bits<2> MajOp, bit isPredNot, bit isPredNew >
  : NVInstPI_V4 <(outs IntRegs:$_dst_),
                 (ins PredRegs:$src1, IntRegs:$src2,
                      ImmOp:$offset, IntRegs:$src3),
  !if(isPredNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#mnemonic#"($src2++#$offset) = $src3.new",
  [], "$src2 = $_dst_">,
  AddrModeRel {
    bits<2> src1;
    bits<5> src2;
    bits<3> src3;
    bits<7> offset;
    bits<4> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0}));
    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = isPredNot;

    let IClass = 0b1010;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = src2;
    let Inst{13} = 0b1;
    let Inst{12-11} = MajOp;
    let Inst{10-8} = src3;
    let Inst{7} = isPredNew;
    let Inst{6-3} = offsetBits;
    let Inst{2} = isPredNot;
    let Inst{1-0} = src1;
  }

multiclass ST_PostInc_Pred_nv<string mnemonic, Operand ImmOp,
                              bits<2> MajOp, bit PredNot> {
  def _pi : T_StorePI_nv_pred <mnemonic, ImmOp, MajOp, PredNot, 0>;

  // Predicate new
  def new_pi : T_StorePI_nv_pred <mnemonic, ImmOp, MajOp, PredNot, 1>;
}

multiclass ST_PostInc_nv<string mnemonic, string BaseOp, Operand ImmOp,
                         bits<2> MajOp> {
  let BaseOpcode = "POST_"#BaseOp in {
    def S2_#NAME#_pi : T_StorePI_nv <mnemonic, ImmOp, MajOp>;

    // Predicated
    defm S2_p#NAME#t : ST_PostInc_Pred_nv <mnemonic, ImmOp, MajOp, 0>;
    defm S2_p#NAME#f : ST_PostInc_Pred_nv <mnemonic, ImmOp, MajOp, 1>;
  }
}

let accessSize = ByteAccess, isCodeGenOnly = 0 in
defm storerbnew: ST_PostInc_nv <"memb", "STrib", s4_0Imm, 0b00>;

let accessSize = HalfWordAccess, isCodeGenOnly = 0 in
defm storerhnew: ST_PostInc_nv <"memh", "STrih", s4_1Imm, 0b01>;

let accessSize = WordAccess, isCodeGenOnly = 0 in
defm storerinew: ST_PostInc_nv <"memw", "STriw", s4_2Imm, 0b10>;

//===----------------------------------------------------------------------===//
// Template class for post increment .new stores with register offset
//===----------------------------------------------------------------------===//
let isNewValue = 1, mayStore = 1, isNVStore = 1, opNewValue = 3 in
class T_StorePI_RegNV <string mnemonic, bits<2> MajOp, MemAccessSize AccessSz>
  : NVInstPI_V4 <(outs IntRegs:$_dst_),
                 (ins IntRegs:$src1, ModRegs:$src2, IntRegs:$src3),
  #mnemonic#"($src1++$src2) = $src3.new",
  [], "$src1 = $_dst_"> {
    bits<5> src1;
    bits<1> src2;
    bits<3> src3;
    let accessSize = AccessSz;

    let IClass = 0b1010;

    let Inst{27-21} = 0b1101101;
    let Inst{20-16} = src1;
    let Inst{13}    = src2;
    let Inst{12-11} = MajOp;
    let Inst{10-8}  = src3;
    let Inst{7}     = 0b0;
  }

let isCodeGenOnly = 0 in {
def S2_storerbnew_pr : T_StorePI_RegNV<"memb", 0b00, ByteAccess>;
def S2_storerhnew_pr : T_StorePI_RegNV<"memh", 0b01, HalfWordAccess>;
def S2_storerinew_pr : T_StorePI_RegNV<"memw", 0b10, WordAccess>;
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

let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 11,
    opExtentAlign = 2 in
class NVJrr_template<string mnemonic, bits<3> majOp, bit NvOpNum,
                      bit isNegCond, bit isTak>
  : NVInst_V4<(outs),
    (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset),
    "if ("#!if(isNegCond, "!","")#mnemonic#
    "($src1"#!if(!eq(NvOpNum, 0),".new, ",", ")#
    "$src2"#!if(!eq(NvOpNum, 1),".new))","))")#" jump:"
    #!if(isTak, "t","nt")#" $offset", []> {

      bits<5> src1;
      bits<5> src2;
      bits<3> Ns;    // New-Value Operand
      bits<5> RegOp; // Non-New-Value Operand
      bits<11> offset;

      let isTaken = isTak;
      let isPredicatedFalse = isNegCond;
      let opNewValue{0} = NvOpNum;

      let Ns = !if(!eq(NvOpNum, 0), src1{2-0}, src2{2-0});
      let RegOp = !if(!eq(NvOpNum, 0), src2, src1);

      let IClass = 0b0010;
      let Inst{26} = 0b0;
      let Inst{25-23} = majOp;
      let Inst{22} = isNegCond;
      let Inst{18-16} = Ns;
      let Inst{13} = isTak;
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
    Defs = [PC], hasSideEffects = 0, validSubTargets = HasV4SubT,
    isCodeGenOnly = 0 in {
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

let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 11,
    opExtentAlign = 2 in
class NVJri_template<string mnemonic, bits<3> majOp, bit isNegCond,
                         bit isTak>
  : NVInst_V4<(outs),
    (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset),
    "if ("#!if(isNegCond, "!","")#mnemonic#"($src1.new, #$src2)) jump:"
    #!if(isTak, "t","nt")#" $offset", []> {

      let isTaken = isTak;
      let isPredicatedFalse = isNegCond;
      let isTaken = isTak;

      bits<3> src1;
      bits<5> src2;
      bits<11> offset;

      let IClass = 0b0010;
      let Inst{26} = 0b1;
      let Inst{25-23} = majOp;
      let Inst{22} = isNegCond;
      let Inst{18-16} = src1;
      let Inst{13} = isTak;
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
    Defs = [PC], hasSideEffects = 0, validSubTargets = HasV4SubT,
    isCodeGenOnly = 0 in {
  defm CMPEQri  : NVJri_base<"cmp.eq", "CMPEQ", 0b000>, PredRel;
  defm CMPGTri  : NVJri_base<"cmp.gt", "CMPGT", 0b001>, PredRel;
  defm CMPGTUri : NVJri_base<"cmp.gtu", "CMPGTU", 0b010>, PredRel;
}

//===----------------------------------------------------------------------===//
// multiclass/template class for the new-value compare jumps instruction
// with a register and an hardcoded 0/-1 immediate value.
//===----------------------------------------------------------------------===//

let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 11,
    opExtentAlign = 2 in
class NVJ_ConstImm_template<string mnemonic, bits<3> majOp, string ImmVal,
                            bit isNegCond, bit isTak>
  : NVInst_V4<(outs),
    (ins IntRegs:$src1, brtarget:$offset),
    "if ("#!if(isNegCond, "!","")#mnemonic
    #"($src1.new, #"#ImmVal#")) jump:"
    #!if(isTak, "t","nt")#" $offset", []> {

      let isTaken = isTak;
      let isPredicatedFalse = isNegCond;
      let isTaken = isTak;

      bits<3> src1;
      bits<11> offset;
      let IClass = 0b0010;
      let Inst{26} = 0b1;
      let Inst{25-23} = majOp;
      let Inst{22} = isNegCond;
      let Inst{18-16} = src1;
      let Inst{13} = isTak;
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
    defm _t_Jumpnv : NVJ_ConstImm_cond<mnemonic, majOp, ImmVal, 0>; // True
    defm _f_Jumpnv : NVJ_ConstImm_cond<mnemonic, majOp, ImmVal, 1>; // False
  }
}

// if ([!]tstbit(Ns.new,#0)) jump:[n]t #r9:2
// if ([!]cmp.eq(Ns.new,#-1)) jump:[n]t #r9:2
// if ([!]cmp.gt(Ns.new,#-1)) jump:[n]t #r9:2

let isPredicated = 1, isBranch = 1, isNewValue = 1, isTerminator=1,
    Defs = [PC], hasSideEffects = 0, isCodeGenOnly = 0 in {
  defm TSTBIT0  : NVJ_ConstImm_base<"tstbit", "TSTBIT", 0b011, "0">, PredRel;
  defm CMPEQn1  : NVJ_ConstImm_base<"cmp.eq", "CMPEQ",  0b100, "-1">, PredRel;
  defm CMPGTn1  : NVJ_ConstImm_base<"cmp.gt", "CMPGT",  0b101, "-1">, PredRel;
}

// J4_hintjumpr: Hint indirect conditional jump.
let isBranch = 1, isIndirectBranch = 1, hasSideEffects = 0, isCodeGenOnly = 0 in
def J4_hintjumpr: JRInst <
  (outs),
  (ins IntRegs:$Rs),
  "hintjr($Rs)"> {
    bits<5> Rs;
    let IClass = 0b0101;
    let Inst{27-21} = 0b0010101;
    let Inst{20-16} = Rs;
  }

//===----------------------------------------------------------------------===//
// NV/J -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CR +
//===----------------------------------------------------------------------===//

// PC-relative add
let hasNewValue = 1, isExtendable = 1, opExtendable = 1,
    isExtentSigned = 0, opExtentBits = 6, hasSideEffects = 0,
    Uses = [PC], validSubTargets = HasV4SubT, isCodeGenOnly = 0 in
def C4_addipc : CRInst <(outs IntRegs:$Rd), (ins u6Ext:$u6),
  "$Rd = add(pc, #$u6)", [], "", CR_tc_2_SLOT3 > {
    bits<5> Rd;
    bits<6> u6;

    let IClass = 0b0110;
    let Inst{27-16} = 0b101001001001;
    let Inst{12-7} = u6;
    let Inst{4-0} = Rd;
  }



let hasSideEffects = 0 in
class T_LOGICAL_3OP<string MnOp1, string MnOp2, bits<2> OpBits, bit IsNeg>
    : CRInst<(outs PredRegs:$Pd),
             (ins PredRegs:$Ps, PredRegs:$Pt, PredRegs:$Pu),
             "$Pd = " # MnOp1 # "($Ps, " # MnOp2 # "($Pt, " #
                   !if (IsNeg,"!","") # "$Pu))",
             [], "", CR_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<2> Ps;
  bits<2> Pt;
  bits<2> Pu;

  let IClass = 0b0110;
  let Inst{27-24} = 0b1011;
  let Inst{23} = IsNeg;
  let Inst{22-21} = OpBits;
  let Inst{20} = 0b1;
  let Inst{17-16} = Ps;
  let Inst{13} = 0b0;
  let Inst{9-8} = Pt;
  let Inst{7-6} = Pu;
  let Inst{1-0} = Pd;
}

let isCodeGenOnly = 0 in {
def C4_and_and  : T_LOGICAL_3OP<"and", "and", 0b00, 0>;
def C4_and_or   : T_LOGICAL_3OP<"and", "or",  0b01, 0>;
def C4_or_and   : T_LOGICAL_3OP<"or",  "and", 0b10, 0>;
def C4_or_or    : T_LOGICAL_3OP<"or",  "or",  0b11, 0>;
def C4_and_andn : T_LOGICAL_3OP<"and", "and", 0b00, 1>;
def C4_and_orn  : T_LOGICAL_3OP<"and", "or",  0b01, 1>;
def C4_or_andn  : T_LOGICAL_3OP<"or",  "and", 0b10, 1>;
def C4_or_orn   : T_LOGICAL_3OP<"or",  "or",  0b11, 1>;
}

//===----------------------------------------------------------------------===//
// CR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/ALU +
//===----------------------------------------------------------------------===//

// Logical with-not instructions.
let validSubTargets = HasV4SubT, isCodeGenOnly = 0 in {
  def A4_andnp : T_ALU64_logical<"and", 0b001, 1, 0, 1>;
  def A4_ornp  : T_ALU64_logical<"or",  0b011, 1, 0, 1>;
}

let hasNewValue = 1, hasSideEffects = 0, isCodeGenOnly = 0 in
def S4_parity: ALU64Inst<(outs IntRegs:$Rd), (ins IntRegs:$Rs, IntRegs:$Rt),
      "$Rd = parity($Rs, $Rt)", [], "", ALU64_tc_2_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;
  let Inst{27-21} = 0b0101111;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{4-0} = Rd;
}
//  Add and accumulate.
//  Rd=add(Rs,add(Ru,#s6))
let isExtentSigned = 1, hasNewValue = 1, isExtendable = 1, opExtentBits = 6,
    opExtendable = 3, isCodeGenOnly = 0 in
def S4_addaddi : ALU64Inst <(outs IntRegs:$Rd),
                            (ins IntRegs:$Rs, IntRegs:$Ru, s6Ext:$s6),
  "$Rd = add($Rs, add($Ru, #$s6))" ,
  [(set (i32 IntRegs:$Rd), (add (i32 IntRegs:$Rs),
                           (add (i32 IntRegs:$Ru), s6_16ExtPred:$s6)))],
  "", ALU64_tc_2_SLOT23> {
    bits<5> Rd;
    bits<5> Rs;
    bits<5> Ru;
    bits<6> s6;

    let IClass = 0b1101;

    let Inst{27-23} = 0b10110;
    let Inst{22-21} = s6{5-4};
    let Inst{20-16} = Rs;
    let Inst{13}    = s6{3};
    let Inst{12-8}  = Rd;
    let Inst{7-5}   = s6{2-0};
    let Inst{4-0}   = Ru;
  }

let isExtentSigned = 1, hasSideEffects = 0, hasNewValue = 1, isExtendable = 1,
    opExtentBits = 6, opExtendable = 2, isCodeGenOnly = 0 in
def S4_subaddi: ALU64Inst <(outs IntRegs:$Rd),
                           (ins IntRegs:$Rs, s6Ext:$s6, IntRegs:$Ru),
  "$Rd = add($Rs, sub(#$s6, $Ru))",
  [], "", ALU64_tc_2_SLOT23> {
    bits<5> Rd;
    bits<5> Rs;
    bits<6> s6;
    bits<5> Ru;

    let IClass = 0b1101;

    let Inst{27-23} = 0b10111;
    let Inst{22-21} = s6{5-4};
    let Inst{20-16} = Rs;
    let Inst{13}    = s6{3};
    let Inst{12-8}  = Rd;
    let Inst{7-5}   = s6{2-0};
    let Inst{4-0}   = Ru;
  }
  
// Extract bitfield
// Rdd=extract(Rss,#u6,#U6)
// Rdd=extract(Rss,Rtt)
// Rd=extract(Rs,Rtt)
// Rd=extract(Rs,#u5,#U5)

let isCodeGenOnly = 0 in {
def S4_extractp_rp : T_S3op_64 < "extract",  0b11, 0b100, 0>;
def S4_extractp    : T_S2op_extract <"extract",  0b1010, DoubleRegs, u6Imm>;
}

let hasNewValue = 1, isCodeGenOnly = 0 in {
  def S4_extract_rp : T_S3op_extract<"extract",  0b01>;
  def S4_extract    : T_S2op_extract <"extract",  0b1101, IntRegs, u5Imm>;
}

// Complex add/sub halfwords/words
let Defs = [USR_OVF], isCodeGenOnly = 0 in {
  def S4_vxaddsubh : T_S3op_64 < "vxaddsubh", 0b01, 0b100, 0, 1>;
  def S4_vxaddsubw : T_S3op_64 < "vxaddsubw", 0b01, 0b000, 0, 1>;
  def S4_vxsubaddh : T_S3op_64 < "vxsubaddh", 0b01, 0b110, 0, 1>;
  def S4_vxsubaddw : T_S3op_64 < "vxsubaddw", 0b01, 0b010, 0, 1>;
}

let Defs = [USR_OVF], isCodeGenOnly = 0 in {
  def S4_vxaddsubhr : T_S3op_64 < "vxaddsubh", 0b11, 0b000, 0, 1, 1, 1>;
  def S4_vxsubaddhr : T_S3op_64 < "vxsubaddh", 0b11, 0b010, 0, 1, 1, 1>;
}

let Itinerary = M_tc_3x_SLOT23, Defs = [USR_OVF], isCodeGenOnly = 0 in {
  def M4_mac_up_s1_sat: T_MType_acc_rr<"+= mpy", 0b011, 0b000, 0, [], 0, 1, 1>;
  def M4_nac_up_s1_sat: T_MType_acc_rr<"-= mpy", 0b011, 0b001, 0, [], 0, 1, 1>;
}

// Logical xor with xor accumulation.
// Rxx^=xor(Rss,Rtt)
let hasSideEffects = 0, isCodeGenOnly = 0 in
def M4_xor_xacc
  : SInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rxx ^= xor($Rss, $Rtt)",
  [(set (i64 DoubleRegs:$Rxx),
   (xor (i64 DoubleRegs:$dst2), (xor (i64 DoubleRegs:$Rss),
                                     (i64 DoubleRegs:$Rtt))))],
  "$dst2 = $Rxx", S_3op_tc_1_SLOT23> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1100;

    let Inst{27-23} = 0b10101;
    let Inst{20-16} = Rss;
    let Inst{12-8}  = Rtt;
    let Inst{4-0}   = Rxx;
  }

// Rotate and reduce bytes
// Rdd=vrcrotate(Rss,Rt,#u2)
let hasSideEffects = 0, isCodeGenOnly = 0 in
def S4_vrcrotate
  : SInst <(outs DoubleRegs:$Rdd),
           (ins DoubleRegs:$Rss, IntRegs:$Rt, u2Imm:$u2),
  "$Rdd = vrcrotate($Rss, $Rt, #$u2)",
  [], "", S_3op_tc_3x_SLOT23> {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rt;
    bits<2> u2;

    let IClass = 0b1100;

    let Inst{27-22} = 0b001111;
    let Inst{20-16} = Rss;
    let Inst{13}    = u2{1};
    let Inst{12-8}  = Rt;
    let Inst{7-6}   = 0b11;
    let Inst{5}     = u2{0};
    let Inst{4-0}   = Rdd;
  }

// Rotate and reduce bytes with accumulation
// Rxx+=vrcrotate(Rss,Rt,#u2)
let hasSideEffects = 0, isCodeGenOnly = 0 in
def S4_vrcrotate_acc
  : SInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, IntRegs:$Rt, u2Imm:$u2),
  "$Rxx += vrcrotate($Rss, $Rt, #$u2)", [],
  "$dst2 = $Rxx", S_3op_tc_3x_SLOT23> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rt;
    bits<2> u2;

    let IClass = 0b1100;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = Rss;
    let Inst{13}    = u2{1};
    let Inst{12-8}  = Rt;
    let Inst{5}     = u2{0};
    let Inst{4-0}   = Rxx;
  }


// Vector reduce conditional negate halfwords
let hasSideEffects = 0, isCodeGenOnly = 0 in
def S2_vrcnegh
  : SInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, IntRegs:$Rt),
  "$Rxx += vrcnegh($Rss, $Rt)", [],
  "$dst2 = $Rxx", S_3op_tc_3x_SLOT23> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rt;

    let IClass = 0b1100;

    let Inst{27-21} = 0b1011001;
    let Inst{20-16} = Rss;
    let Inst{13}    = 0b1;
    let Inst{12-8}  = Rt;
    let Inst{7-5}   = 0b111;
    let Inst{4-0}   = Rxx;
  }

// Split bitfield
let isCodeGenOnly = 0 in
def A4_bitspliti : T_S2op_2_di <"bitsplit", 0b110, 0b100>;

// Arithmetic/Convergent round
let isCodeGenOnly = 0 in
def A4_cround_ri : T_S2op_2_ii <"cround", 0b111, 0b000>;

let isCodeGenOnly = 0 in
def A4_round_ri  : T_S2op_2_ii <"round", 0b111, 0b100>;

let Defs = [USR_OVF], isCodeGenOnly = 0 in
def A4_round_ri_sat : T_S2op_2_ii <"round", 0b111, 0b110, 1>;

// Logical-logical words.
// Compound or-and -- Rx=or(Ru,and(Rx,#s10))
let isExtentSigned = 1, hasNewValue = 1, isExtendable = 1, opExtentBits = 10,
    opExtendable = 3, isCodeGenOnly = 0 in
def S4_or_andix:
  ALU64Inst<(outs IntRegs:$Rx),
            (ins IntRegs:$Ru, IntRegs:$_src_, s10Ext:$s10),
  "$Rx = or($Ru, and($_src_, #$s10))" ,
  [(set (i32 IntRegs:$Rx),
        (or (i32 IntRegs:$Ru), (and (i32 IntRegs:$_src_), s10ExtPred:$s10)))] ,
  "$_src_ = $Rx", ALU64_tc_2_SLOT23> {
    bits<5> Rx;
    bits<5> Ru;
    bits<10> s10;

    let IClass = 0b1101;

    let Inst{27-22} = 0b101001;
    let Inst{20-16} = Rx;
    let Inst{21}    = s10{9};
    let Inst{13-5}  = s10{8-0};
    let Inst{4-0}   = Ru;
  }

// Miscellaneous ALU64 instructions.
//
let hasNewValue = 1, hasSideEffects = 0, isCodeGenOnly = 0 in
def A4_modwrapu: ALU64Inst<(outs IntRegs:$Rd), (ins IntRegs:$Rs, IntRegs:$Rt),
      "$Rd = modwrap($Rs, $Rt)", [], "", ALU64_tc_2_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;
  let Inst{27-21} = 0b0011111;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{7-5} = 0b111;
  let Inst{4-0} = Rd;
}

let hasSideEffects = 0, isCodeGenOnly = 0 in
def A4_bitsplit: ALU64Inst<(outs DoubleRegs:$Rd),
      (ins IntRegs:$Rs, IntRegs:$Rt),
      "$Rd = bitsplit($Rs, $Rt)", [], "", ALU64_tc_1_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;
  let Inst{27-24} = 0b0100;
  let Inst{21} = 0b1;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{4-0} = Rd;
}

let isCodeGenOnly = 0 in {
// Rx[&|]=xor(Rs,Rt)
def M4_or_xor   : T_MType_acc_rr < "|= xor", 0b110, 0b001, 0>;
def M4_and_xor  : T_MType_acc_rr < "&= xor", 0b010, 0b010, 0>;

// Rx[&|^]=or(Rs,Rt)
def M4_xor_or   : T_MType_acc_rr < "^= or",  0b110, 0b011, 0>;

let CextOpcode = "ORr_ORr" in
def M4_or_or    : T_MType_acc_rr < "|= or",  0b110, 0b000, 0>;
def M4_and_or   : T_MType_acc_rr < "&= or",  0b010, 0b001, 0>;

// Rx[&|^]=and(Rs,Rt)
def M4_xor_and  : T_MType_acc_rr < "^= and", 0b110, 0b010, 0>;

let CextOpcode = "ORr_ANDr" in
def M4_or_and   : T_MType_acc_rr < "|= and", 0b010, 0b011, 0>;
def M4_and_and  : T_MType_acc_rr < "&= and", 0b010, 0b000, 0>;

// Rx[&|^]=and(Rs,~Rt)
def M4_xor_andn : T_MType_acc_rr < "^= and", 0b001, 0b010, 0, [], 1>;
def M4_or_andn  : T_MType_acc_rr < "|= and", 0b001, 0b000, 0, [], 1>;
def M4_and_andn : T_MType_acc_rr < "&= and", 0b001, 0b001, 0, [], 1>;
}

// Compound or-or and or-and
let isExtentSigned = 1, InputType = "imm", hasNewValue = 1, isExtendable = 1,
    opExtentBits = 10, opExtendable = 3 in
class T_CompOR <string mnemonic, bits<2> MajOp, SDNode OpNode>
  : MInst_acc <(outs IntRegs:$Rx),
               (ins IntRegs:$src1, IntRegs:$Rs, s10Ext:$s10),
  "$Rx |= "#mnemonic#"($Rs, #$s10)",
  [(set (i32 IntRegs:$Rx), (or (i32 IntRegs:$src1),
                           (OpNode (i32 IntRegs:$Rs), s10ExtPred:$s10)))],
  "$src1 = $Rx", ALU64_tc_2_SLOT23>, ImmRegRel {
    bits<5> Rx;
    bits<5> Rs;
    bits<10> s10;

    let IClass = 0b1101;

    let Inst{27-24} = 0b1010;
    let Inst{23-22} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{21}    = s10{9};
    let Inst{13-5}  = s10{8-0};
    let Inst{4-0}   = Rx;
  }

let CextOpcode = "ORr_ANDr", isCodeGenOnly = 0 in
def S4_or_andi : T_CompOR <"and", 0b00, and>;

let CextOpcode = "ORr_ORr", isCodeGenOnly = 0 in
def S4_or_ori : T_CompOR <"or", 0b10, or>;

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
// XTYPE/BIT +
//===----------------------------------------------------------------------===//

// Bit reverse
let isCodeGenOnly = 0 in
def S2_brevp : T_S2op_3 <"brev", 0b11, 0b110>;

// Bit count
let isCodeGenOnly = 0 in {
def S2_ct0p : T_COUNT_LEADING_64<"ct0", 0b111, 0b010>;
def S2_ct1p : T_COUNT_LEADING_64<"ct1", 0b111, 0b100>;
def S4_clbpnorm : T_COUNT_LEADING_64<"normamt", 0b011, 0b000>;
}

def: Pat<(i32 (trunc (cttz (i64 DoubleRegs:$Rss)))),
         (S2_ct0p (i64 DoubleRegs:$Rss))>;
def: Pat<(i32 (trunc (cttz (not (i64 DoubleRegs:$Rss))))),
         (S2_ct1p (i64 DoubleRegs:$Rss))>;

let hasSideEffects = 0, hasNewValue = 1, isCodeGenOnly = 0 in
def S4_clbaddi : SInst<(outs IntRegs:$Rd), (ins IntRegs:$Rs, s6Imm:$s6),
    "$Rd = add(clb($Rs), #$s6)", [], "", S_2op_tc_2_SLOT23> {
  bits<5> Rs;
  bits<5> Rd;
  bits<6> s6;
  let IClass = 0b1000;
  let Inst{27-24} = 0b1100;
  let Inst{23-21} = 0b001;
  let Inst{20-16} = Rs;
  let Inst{13-8} = s6;
  let Inst{7-5} = 0b000;
  let Inst{4-0} = Rd;
}

let hasSideEffects = 0, hasNewValue = 1, isCodeGenOnly = 0 in
def S4_clbpaddi : SInst<(outs IntRegs:$Rd), (ins DoubleRegs:$Rs, s6Imm:$s6),
    "$Rd = add(clb($Rs), #$s6)", [], "", S_2op_tc_2_SLOT23> {
  bits<5> Rs;
  bits<5> Rd;
  bits<6> s6;
  let IClass = 0b1000;
  let Inst{27-24} = 0b1000;
  let Inst{23-21} = 0b011;
  let Inst{20-16} = Rs;
  let Inst{13-8} = s6;
  let Inst{7-5} = 0b010;
  let Inst{4-0} = Rd;
}


// Bit test/set/clear
let isCodeGenOnly = 0 in {
def S4_ntstbit_i : T_TEST_BIT_IMM<"!tstbit", 0b001>;
def S4_ntstbit_r : T_TEST_BIT_REG<"!tstbit", 1>;
}

let AddedComplexity = 20 in {   // Complexity greater than cmp reg-imm.
  def: Pat<(i1 (seteq (and (shl 1, u5ImmPred:$u5), (i32 IntRegs:$Rs)), 0)),
           (S4_ntstbit_i (i32 IntRegs:$Rs), u5ImmPred:$u5)>;
  def: Pat<(i1 (seteq (and (shl 1, (i32 IntRegs:$Rt)), (i32 IntRegs:$Rs)), 0)),
           (S4_ntstbit_r (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))>;
}

// Add extra complexity to prefer these instructions over bitsset/bitsclr.
// The reason is that tstbit/ntstbit can be folded into a compound instruction:
//   if ([!]tstbit(...)) jump ...
let AddedComplexity = 100 in
def: Pat<(i1 (setne (and (i32 IntRegs:$Rs), (i32 Set5ImmPred:$u5)), (i32 0))),
         (S2_tstbit_i (i32 IntRegs:$Rs), (BITPOS32 Set5ImmPred:$u5))>;

let AddedComplexity = 100 in
def: Pat<(i1 (seteq (and (i32 IntRegs:$Rs), (i32 Set5ImmPred:$u5)), (i32 0))),
         (S4_ntstbit_i (i32 IntRegs:$Rs), (BITPOS32 Set5ImmPred:$u5))>;

let isCodeGenOnly = 0 in {
def C4_nbitsset  : T_TEST_BITS_REG<"!bitsset", 0b01, 1>;
def C4_nbitsclr  : T_TEST_BITS_REG<"!bitsclr", 0b10, 1>;
def C4_nbitsclri : T_TEST_BITS_IMM<"!bitsclr", 0b10, 1>;
}

// Do not increase complexity of these patterns. In the DAG, "cmp i8" may be
// represented as a compare against "value & 0xFF", which is an exact match
// for cmpb (same for cmph). The patterns below do not contain any additional
// complexity that would make them preferable, and if they were actually used
// instead of cmpb/cmph, they would result in a compare against register that
// is loaded with the byte/half mask (i.e. 0xFF or 0xFFFF).
def: Pat<(i1 (setne (and I32:$Rs, u6ImmPred:$u6), 0)),
         (C4_nbitsclri I32:$Rs, u6ImmPred:$u6)>;
def: Pat<(i1 (setne (and I32:$Rs, I32:$Rt), 0)),
         (C4_nbitsclr I32:$Rs, I32:$Rt)>;
def: Pat<(i1 (setne (and I32:$Rs, I32:$Rt), I32:$Rt)),
         (C4_nbitsset I32:$Rs, I32:$Rt)>;

//===----------------------------------------------------------------------===//
// XTYPE/BIT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/MPY +
//===----------------------------------------------------------------------===//

// Rd=add(#u6,mpyi(Rs,#U6)) -- Multiply by immed and add immed.

let hasNewValue = 1, isExtendable = 1, opExtentBits = 6, opExtendable = 1,
    isCodeGenOnly = 0 in
def M4_mpyri_addi : MInst<(outs IntRegs:$Rd),
  (ins u6Ext:$u6, IntRegs:$Rs, u6Imm:$U6),
  "$Rd = add(#$u6, mpyi($Rs, #$U6))" ,
  [(set (i32 IntRegs:$Rd),
        (add (mul (i32 IntRegs:$Rs), u6ImmPred:$U6),
             u6ExtPred:$u6))] ,"",ALU64_tc_3x_SLOT23> {
    bits<5> Rd;
    bits<6> u6;
    bits<5> Rs;
    bits<6> U6;

    let IClass = 0b1101;

    let Inst{27-24} = 0b1000;
    let Inst{23}    = U6{5};
    let Inst{22-21} = u6{5-4};
    let Inst{20-16} = Rs;
    let Inst{13}    = u6{3};
    let Inst{12-8}  = Rd;
    let Inst{7-5}   = u6{2-0};
    let Inst{4-0}   = U6{4-0};
  }

// Rd=add(#u6,mpyi(Rs,Rt))
let CextOpcode = "ADD_MPY", InputType = "imm", hasNewValue = 1,
    isExtendable = 1, opExtentBits = 6, opExtendable = 1, isCodeGenOnly = 0 in
def M4_mpyrr_addi : MInst <(outs IntRegs:$Rd),
  (ins u6Ext:$u6, IntRegs:$Rs, IntRegs:$Rt),
  "$Rd = add(#$u6, mpyi($Rs, $Rt))" ,
  [(set (i32 IntRegs:$Rd),
        (add (mul (i32 IntRegs:$Rs), (i32 IntRegs:$Rt)), u6ExtPred:$u6))],
  "", ALU64_tc_3x_SLOT23>, ImmRegRel {
    bits<5> Rd;
    bits<6> u6;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1101;

    let Inst{27-23} = 0b01110;
    let Inst{22-21} = u6{5-4};
    let Inst{20-16} = Rs;
    let Inst{13}    = u6{3};
    let Inst{12-8}  = Rt;
    let Inst{7-5}   = u6{2-0};
    let Inst{4-0}   = Rd;
  }

let hasNewValue = 1 in
class T_AddMpy <bit MajOp, PatLeaf ImmPred, dag ins>
  : ALU64Inst <(outs IntRegs:$dst), ins,
  "$dst = add($src1, mpyi("#!if(MajOp,"$src3, #$src2))",
                                      "#$src2, $src3))"),
  [(set (i32 IntRegs:$dst),
        (add (i32 IntRegs:$src1), (mul (i32 IntRegs:$src3), ImmPred:$src2)))],
  "", ALU64_tc_3x_SLOT23> {
    bits<5> dst;
    bits<5> src1;
    bits<8> src2;
    bits<5> src3;

    let IClass = 0b1101;

    bits<6> ImmValue = !if(MajOp, src2{5-0}, src2{7-2});

    let Inst{27-24} = 0b1111;
    let Inst{23}    = MajOp;
    let Inst{22-21} = ImmValue{5-4};
    let Inst{20-16} = src3;
    let Inst{13}    = ImmValue{3};
    let Inst{12-8}  = dst;
    let Inst{7-5}   = ImmValue{2-0};
    let Inst{4-0}   = src1;
  }

let isCodeGenOnly = 0 in
def M4_mpyri_addr_u2 : T_AddMpy<0b0, u6_2ImmPred,
                       (ins IntRegs:$src1, u6_2Imm:$src2, IntRegs:$src3)>;

let isExtendable = 1, opExtentBits = 6, opExtendable = 3,
    CextOpcode = "ADD_MPY", InputType = "imm", isCodeGenOnly = 0 in
def M4_mpyri_addr : T_AddMpy<0b1, u6ExtPred,
                    (ins IntRegs:$src1, IntRegs:$src3, u6Ext:$src2)>, ImmRegRel;

// Rx=add(Ru,mpyi(Rx,Rs))
let validSubTargets = HasV4SubT, CextOpcode = "ADD_MPY", InputType = "reg",
    hasNewValue = 1, isCodeGenOnly = 0 in
def M4_mpyrr_addr: MInst_acc <(outs IntRegs:$Rx),
                              (ins IntRegs:$Ru, IntRegs:$_src_, IntRegs:$Rs),
  "$Rx = add($Ru, mpyi($_src_, $Rs))",
  [(set (i32 IntRegs:$Rx), (add (i32 IntRegs:$Ru),
                           (mul (i32 IntRegs:$_src_), (i32 IntRegs:$Rs))))],
  "$_src_ = $Rx", M_tc_3x_SLOT23>, ImmRegRel {
    bits<5> Rx;
    bits<5> Ru;
    bits<5> Rs;

    let IClass = 0b1110;

    let Inst{27-21} = 0b0011000;
    let Inst{12-8} = Rx;
    let Inst{4-0} = Ru;
    let Inst{20-16} = Rs;
  }

// Rd=add(##,mpyi(Rs,#U6))
def : Pat <(add (mul (i32 IntRegs:$src2), u6ImmPred:$src3),
                     (HexagonCONST32 tglobaladdr:$src1)),
           (i32 (M4_mpyri_addi tglobaladdr:$src1, IntRegs:$src2,
                               u6ImmPred:$src3))>;

// Rd=add(##,mpyi(Rs,Rt))
def : Pat <(add (mul (i32 IntRegs:$src2), (i32 IntRegs:$src3)),
                     (HexagonCONST32 tglobaladdr:$src1)),
           (i32 (M4_mpyrr_addi tglobaladdr:$src1, IntRegs:$src2,
                               IntRegs:$src3))>;

// Vector reduce multiply word by signed half (32x16)
//Rdd=vrmpyweh(Rss,Rtt)[:<<1]
let isCodeGenOnly = 0 in {
def M4_vrmpyeh_s0 : T_M2_vmpy<"vrmpyweh", 0b010, 0b100, 0, 0, 0>;
def M4_vrmpyeh_s1 : T_M2_vmpy<"vrmpyweh", 0b110, 0b100, 1, 0, 0>;
}

//Rdd=vrmpywoh(Rss,Rtt)[:<<1]
let isCodeGenOnly = 0 in {
def M4_vrmpyoh_s0 : T_M2_vmpy<"vrmpywoh", 0b001, 0b010, 0, 0, 0>;
def M4_vrmpyoh_s1 : T_M2_vmpy<"vrmpywoh", 0b101, 0b010, 1, 0, 0>;
}
//Rdd+=vrmpyweh(Rss,Rtt)[:<<1]
let isCodeGenOnly = 0 in {
def M4_vrmpyeh_acc_s0: T_M2_vmpy_acc<"vrmpyweh", 0b001, 0b110, 0, 0>;
def M4_vrmpyeh_acc_s1: T_M2_vmpy_acc<"vrmpyweh", 0b101, 0b110, 1, 0>;
}

//Rdd=vrmpywoh(Rss,Rtt)[:<<1]
let isCodeGenOnly = 0 in {
def M4_vrmpyoh_acc_s0: T_M2_vmpy_acc<"vrmpywoh", 0b011, 0b110, 0, 0>;
def M4_vrmpyoh_acc_s1: T_M2_vmpy_acc<"vrmpywoh", 0b111, 0b110, 1, 0>;
}

// Vector multiply halfwords, signed by unsigned
// Rdd=vmpyhsu(Rs,Rt)[:<<]:sat
let isCodeGenOnly = 0 in {
def M2_vmpy2su_s0 : T_XTYPE_mpy64 < "vmpyhsu", 0b000, 0b111, 1, 0, 0>;
def M2_vmpy2su_s1 : T_XTYPE_mpy64 < "vmpyhsu", 0b100, 0b111, 1, 1, 0>;
}

// Rxx+=vmpyhsu(Rs,Rt)[:<<1]:sat
let isCodeGenOnly = 0 in {
def M2_vmac2su_s0 : T_XTYPE_mpy64_acc < "vmpyhsu", "+", 0b011, 0b101, 1, 0, 0>;
def M2_vmac2su_s1 : T_XTYPE_mpy64_acc < "vmpyhsu", "+", 0b111, 0b101, 1, 1, 0>;
}

// Vector polynomial multiply halfwords
// Rdd=vpmpyh(Rs,Rt)
let isCodeGenOnly = 0 in
def M4_vpmpyh : T_XTYPE_mpy64 < "vpmpyh", 0b110, 0b111, 0, 0, 0>;

// Rxx^=vpmpyh(Rs,Rt)
let isCodeGenOnly = 0 in
def M4_vpmpyh_acc : T_XTYPE_mpy64_acc < "vpmpyh", "^", 0b101, 0b111, 0, 0, 0>;

// Polynomial multiply words
// Rdd=pmpyw(Rs,Rt)
let isCodeGenOnly = 0 in
def M4_pmpyw : T_XTYPE_mpy64 < "pmpyw", 0b010, 0b111, 0, 0, 0>;

// Rxx^=pmpyw(Rs,Rt)
let isCodeGenOnly = 0 in
def M4_pmpyw_acc  : T_XTYPE_mpy64_acc < "pmpyw", "^", 0b001, 0b111, 0, 0, 0>;

//===----------------------------------------------------------------------===//
// XTYPE/MPY -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU64/Vector compare
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Template class for vector compare
//===----------------------------------------------------------------------===//

let hasSideEffects = 0 in
class T_vcmpImm <string Str, bits<2> cmpOp, bits<2> minOp, Operand ImmOprnd>
  : ALU64_rr <(outs PredRegs:$Pd),
              (ins DoubleRegs:$Rss, ImmOprnd:$Imm),
  "$Pd = "#Str#"($Rss, #$Imm)",
  [], "", ALU64_tc_2early_SLOT23> {
    bits<2> Pd;
    bits<5> Rss;
    bits<32> Imm;
    bits<8> ImmBits;
    let ImmBits{6-0} = Imm{6-0};
    let ImmBits{7} = !if (!eq(cmpOp,0b10), 0b0, Imm{7}); // 0 for vcmp[bhw].gtu

    let IClass = 0b1101;

    let Inst{27-24} = 0b1100;
    let Inst{22-21} = cmpOp;
    let Inst{20-16} = Rss;
    let Inst{12-5} = ImmBits;
    let Inst{4-3} = minOp;
    let Inst{1-0} = Pd;
  }

// Vector compare bytes
let isCodeGenOnly = 0 in
def A4_vcmpbgt   : T_vcmp <"vcmpb.gt", 0b1010>;
def: T_vcmp_pat<A4_vcmpbgt, setgt, v8i8>;

let AsmString = "$Pd = any8(vcmpb.eq($Rss, $Rtt))" in
let isCodeGenOnly = 0 in
def A4_vcmpbeq_any : T_vcmp <"any8(vcmpb.gt", 0b1000>;

let isCodeGenOnly = 0 in {
def A4_vcmpbeqi  : T_vcmpImm <"vcmpb.eq",  0b00, 0b00, u8Imm>;
def A4_vcmpbgti  : T_vcmpImm <"vcmpb.gt",  0b01, 0b00, s8Imm>;
def A4_vcmpbgtui : T_vcmpImm <"vcmpb.gtu", 0b10, 0b00, u7Imm>;
}

// Vector compare halfwords
let isCodeGenOnly = 0 in {
def A4_vcmpheqi  : T_vcmpImm <"vcmph.eq",  0b00, 0b01, s8Imm>;
def A4_vcmphgti  : T_vcmpImm <"vcmph.gt",  0b01, 0b01, s8Imm>;
def A4_vcmphgtui : T_vcmpImm <"vcmph.gtu", 0b10, 0b01, u7Imm>;
}

// Vector compare words
let isCodeGenOnly = 0 in {
def A4_vcmpweqi  : T_vcmpImm <"vcmpw.eq",  0b00, 0b10, s8Imm>;
def A4_vcmpwgti  : T_vcmpImm <"vcmpw.gt",  0b01, 0b10, s8Imm>;
def A4_vcmpwgtui : T_vcmpImm <"vcmpw.gtu", 0b10, 0b10, u7Imm>;
}

//===----------------------------------------------------------------------===//
// XTYPE/SHIFT +
//===----------------------------------------------------------------------===//
// Shift by immediate and accumulate/logical.
// Rx=add(#u8,asl(Rx,#U5))  Rx=add(#u8,lsr(Rx,#U5))
// Rx=sub(#u8,asl(Rx,#U5))  Rx=sub(#u8,lsr(Rx,#U5))
// Rx=and(#u8,asl(Rx,#U5))  Rx=and(#u8,lsr(Rx,#U5))
// Rx=or(#u8,asl(Rx,#U5))   Rx=or(#u8,lsr(Rx,#U5))
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0, opExtentBits = 8,
    hasNewValue = 1, opNewValue = 0, validSubTargets = HasV4SubT in
class T_S4_ShiftOperate<string MnOp, string MnSh, SDNode Op, SDNode Sh,
                        bit asl_lsr, bits<2> MajOp, InstrItinClass Itin>
  : MInst_acc<(outs IntRegs:$Rd), (ins u8Ext:$u8, IntRegs:$Rx, u5Imm:$U5),
      "$Rd = "#MnOp#"(#$u8, "#MnSh#"($Rx, #$U5))",
      [(set (i32 IntRegs:$Rd),
            (Op (Sh I32:$Rx, u5ImmPred:$U5), u8ExtPred:$u8))],
      "$Rd = $Rx", Itin> {

  bits<5> Rd;
  bits<8> u8;
  bits<5> Rx;
  bits<5> U5;

  let IClass = 0b1101;
  let Inst{27-24} = 0b1110;
  let Inst{23-21} = u8{7-5};
  let Inst{20-16} = Rd;
  let Inst{13} = u8{4};
  let Inst{12-8} = U5;
  let Inst{7-5} = u8{3-1};
  let Inst{4} = asl_lsr;
  let Inst{3} = u8{0};
  let Inst{2-1} = MajOp;
}

multiclass T_ShiftOperate<string mnemonic, SDNode Op, bits<2> MajOp,
                          InstrItinClass Itin> {
  def _asl_ri : T_S4_ShiftOperate<mnemonic, "asl", Op, shl, 0, MajOp, Itin>;
  def _lsr_ri : T_S4_ShiftOperate<mnemonic, "lsr", Op, srl, 1, MajOp, Itin>;
}

let AddedComplexity = 200, isCodeGenOnly = 0 in {
  defm S4_addi : T_ShiftOperate<"add", add, 0b10, ALU64_tc_2_SLOT23>;
  defm S4_andi : T_ShiftOperate<"and", and, 0b00, ALU64_tc_2_SLOT23>;
}

let AddedComplexity = 30, isCodeGenOnly = 0 in
defm S4_ori  : T_ShiftOperate<"or",  or,  0b01, ALU64_tc_1_SLOT23>;

let isCodeGenOnly = 0 in
defm S4_subi : T_ShiftOperate<"sub", sub, 0b11, ALU64_tc_1_SLOT23>;

// Vector conditional negate
// Rdd=vcnegh(Rss,Rt)
let Defs = [USR_OVF], Itinerary = S_3op_tc_2_SLOT23, isCodeGenOnly = 0 in
def S2_vcnegh   : T_S3op_shiftVect < "vcnegh",   0b11, 0b01>;

// Rd=[cround|round](Rs,Rt)
let hasNewValue = 1, Itinerary = S_3op_tc_2_SLOT23, isCodeGenOnly = 0 in {
  def A4_cround_rr    : T_S3op_3 < "cround", IntRegs, 0b11, 0b00>;
  def A4_round_rr     : T_S3op_3 < "round", IntRegs, 0b11, 0b10>;
}

// Rd=round(Rs,Rt):sat
let hasNewValue = 1, Defs = [USR_OVF], Itinerary = S_3op_tc_2_SLOT23,
    isCodeGenOnly = 0 in
def A4_round_rr_sat : T_S3op_3 < "round", IntRegs, 0b11, 0b11, 1>;

// Rd=[cmpyiwh|cmpyrwh](Rss,Rt):<<1:rnd:sat
let Defs = [USR_OVF], Itinerary = S_3op_tc_3x_SLOT23, isCodeGenOnly = 0 in {
  def M4_cmpyi_wh     : T_S3op_8<"cmpyiwh", 0b100, 1, 1, 1>;
  def M4_cmpyr_wh     : T_S3op_8<"cmpyrwh", 0b110, 1, 1, 1>;
}

// Rdd=[add|sub](Rss,Rtt,Px):carry
let isPredicateLate = 1, hasSideEffects = 0 in
class T_S3op_carry <string mnemonic, bits<3> MajOp>
  : SInst < (outs DoubleRegs:$Rdd, PredRegs:$Px),
            (ins DoubleRegs:$Rss, DoubleRegs:$Rtt, PredRegs:$Pu),
  "$Rdd = "#mnemonic#"($Rss, $Rtt, $Pu):carry",
  [], "$Px = $Pu", S_3op_tc_1_SLOT23 > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;
    bits<2> Pu;

    let IClass = 0b1100;

    let Inst{27-24} = 0b0010;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rss;
    let Inst{12-8}  = Rtt;
    let Inst{6-5}   = Pu;
    let Inst{4-0}   = Rdd;
  }

let isCodeGenOnly = 0 in {
def A4_addp_c : T_S3op_carry < "add", 0b110 >;
def A4_subp_c : T_S3op_carry < "sub", 0b111 >;
}

let Itinerary = S_3op_tc_3_SLOT23, hasSideEffects = 0 in
class T_S3op_6 <string mnemonic, bits<3> MinOp, bit isUnsigned>
  : SInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, IntRegs:$Ru),
  "$Rxx = "#mnemonic#"($Rss, $Ru)" ,
  [] , "$dst2 = $Rxx"> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Ru;

    let IClass = 0b1100;

    let Inst{27-21} = 0b1011001;
    let Inst{20-16} = Rss;
    let Inst{13}    = isUnsigned;
    let Inst{12-8}  = Rxx;
    let Inst{7-5}   = MinOp;
    let Inst{4-0}   = Ru;
  }

// Vector reduce maximum halfwords
// Rxx=vrmax[u]h(Rss,Ru)
let isCodeGenOnly = 0 in {
def A4_vrmaxh  : T_S3op_6 < "vrmaxh",  0b001, 0>;
def A4_vrmaxuh : T_S3op_6 < "vrmaxuh", 0b001, 1>;
}
// Vector reduce maximum words
// Rxx=vrmax[u]w(Rss,Ru)
let isCodeGenOnly = 0 in {
def A4_vrmaxw  : T_S3op_6 < "vrmaxw",  0b010, 0>;
def A4_vrmaxuw : T_S3op_6 < "vrmaxuw", 0b010, 1>;
}
// Vector reduce minimum halfwords
// Rxx=vrmin[u]h(Rss,Ru)
let isCodeGenOnly = 0 in {
def A4_vrminh  : T_S3op_6 < "vrminh",  0b101, 0>;
def A4_vrminuh : T_S3op_6 < "vrminuh", 0b101, 1>;
}

// Vector reduce minimum words
// Rxx=vrmin[u]w(Rss,Ru)
let isCodeGenOnly = 0 in {
def A4_vrminw  : T_S3op_6 < "vrminw",  0b110, 0>;
def A4_vrminuw : T_S3op_6 < "vrminuw", 0b110, 1>;
}

// Shift an immediate left by register amount.
let hasNewValue = 1, hasSideEffects = 0, isCodeGenOnly = 0 in
def S4_lsli: SInst <(outs IntRegs:$Rd), (ins s6Imm:$s6, IntRegs:$Rt),
  "$Rd = lsl(#$s6, $Rt)" ,
  [(set (i32 IntRegs:$Rd), (shl s6ImmPred:$s6,
                                 (i32 IntRegs:$Rt)))],
  "", S_3op_tc_1_SLOT23> {
    bits<5> Rd;
    bits<6> s6;
    bits<5> Rt;

    let IClass = 0b1100;

    let Inst{27-22} = 0b011010;
    let Inst{20-16} = s6{5-1};
    let Inst{12-8}  = Rt;
    let Inst{7-6}   = 0b11;
    let Inst{4-0}   = Rd;
    let Inst{5}     = s6{0};
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
                 Requires<[UseMEMOP]> {

    bits<5> base;
    bits<5> delta;
    bits<32> offset;
    bits<6> offsetBits; // memb - u6:0 , memh - u6:1, memw - u6:2

    let offsetBits = !if (!eq(opcBits, 0b00), offset{5-0},
                     !if (!eq(opcBits, 0b01), offset{6-1},
                     !if (!eq(opcBits, 0b10), offset{7-2},0)));

    let opExtentAlign = opcBits;
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
                  Requires<[UseMEMOP]> {

    bits<5> base;
    bits<5> delta;
    bits<32> offset;
    bits<6> offsetBits; // memb - u6:0 , memh - u6:1, memw - u6:2

    let offsetBits = !if (!eq(opcBits, 0b00), offset{5-0},
                     !if (!eq(opcBits, 0b01), offset{6-1},
                     !if (!eq(opcBits, 0b10), offset{7-2},0)));

    let opExtentAlign = opcBits;
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
  def L4_add#NAME : MemOp_rr_base <opc, opcBits, ImmOp, " += ", 0b00>; // add
  def L4_sub#NAME : MemOp_rr_base <opc, opcBits, ImmOp, " -= ", 0b01>; // sub
  def L4_and#NAME : MemOp_rr_base <opc, opcBits, ImmOp, " &= ", 0b10>; // and
  def L4_or#NAME  : MemOp_rr_base <opc, opcBits, ImmOp, " |= ", 0b11>; // or
}

// multiclass to define MemOp instructions with immediate Operand.
multiclass MemOp_ri<string opc, bits<2> opcBits, Operand ImmOp> {
  def L4_iadd#NAME : MemOp_ri_base <opc, opcBits, ImmOp, " += ", 0b00 >;
  def L4_isub#NAME : MemOp_ri_base <opc, opcBits, ImmOp, " -= ", 0b01 >;
  def L4_iand#NAME : MemOp_ri_base<opc, opcBits, ImmOp, " = clrbit(", 0b10>;
  def L4_ior#NAME : MemOp_ri_base<opc, opcBits, ImmOp, " = setbit(", 0b11>;
}

multiclass MemOp_base <string opc, bits<2> opcBits, Operand ImmOp> {
  defm _#NAME : MemOp_rr <opc, opcBits, ImmOp>;
  defm _#NAME : MemOp_ri <opc, opcBits, ImmOp>;
}

// Define MemOp instructions.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 0,
    validSubTargets =HasV4SubT in {
  let opExtentBits = 6, accessSize = ByteAccess, isCodeGenOnly = 0 in
  defm memopb_io : MemOp_base <"memb", 0b00, u6_0Ext>;

  let opExtentBits = 7, accessSize = HalfWordAccess, isCodeGenOnly = 0 in
  defm memoph_io : MemOp_base <"memh", 0b01, u6_1Ext>;

  let opExtentBits = 8, accessSize = WordAccess, isCodeGenOnly = 0 in
  defm memopw_io : MemOp_base <"memw", 0b10, u6_2Ext>;
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
                         L4_iadd_memoph_io, L4_isub_memoph_io>;
  // Byte
  defm : MemOpi_u5ALUOp <ldOpByte, truncstorei8, u6ExtPred,
                         L4_iadd_memopb_io, L4_isub_memopb_io>;
}

let Predicates = [HasV4T, UseMEMOP] in {
  defm : MemOpi_u5ExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOpi_u5ExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOpi_u5ExtType<extloadi8,  extloadi16>;  // any extend

  // Word
  defm : MemOpi_u5ALUOp <load, store, u6_2ExtPred, L4_iadd_memopw_io,
                         L4_isub_memopw_io>;
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
                        ADDRriU6_1, MEMOPIMM_HALF, L4_isub_memoph_io>;
  // Byte
  defm : MemOpi_m5Pats <ldOpByte, truncstorei8, u6ExtPred, m5BImmPred,
                        ADDRriU6_0, MEMOPIMM_BYTE, L4_isub_memopb_io>;
}

let Predicates = [HasV4T, UseMEMOP] in {
  defm : MemOpi_m5ExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOpi_m5ExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOpi_m5ExtType<extloadi8,  extloadi16>;  // any extend

  // Word
  defm : MemOpi_m5Pats <load, store, u6_2ExtPred, m5ImmPred,
                          ADDRriU6_2, MEMOPIMM, L4_isub_memopw_io>;
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
  def : Pat <(stOp (OpNode (ldOp (addrPred IntRegs:$addr, extPred:$offset)),
                           immPred:$bitend),
                   (addrPred (i32 IntRegs:$addr), extPred:$offset)),
             (MI IntRegs:$addr, extPred:$offset, (xformFunc immPred:$bitend))>;
}

multiclass MemOpi_bitExtType<PatFrag ldOpByte, PatFrag ldOpHalf > {
  // Byte - clrbit
  defm : MemOpi_bitPats<ldOpByte, truncstorei8, Clr3ImmPred, u6ExtPred,
                       ADDRriU6_0, CLRMEMIMM_BYTE, L4_iand_memopb_io, and>;
  // Byte - setbit
  defm : MemOpi_bitPats<ldOpByte, truncstorei8, Set3ImmPred,  u6ExtPred,
                       ADDRriU6_0, SETMEMIMM_BYTE, L4_ior_memopb_io, or>;
  // Half Word - clrbit
  defm : MemOpi_bitPats<ldOpHalf, truncstorei16, Clr4ImmPred, u6_1ExtPred,
                       ADDRriU6_1, CLRMEMIMM_SHORT, L4_iand_memoph_io, and>;
  // Half Word - setbit
  defm : MemOpi_bitPats<ldOpHalf, truncstorei16, Set4ImmPred, u6_1ExtPred,
                       ADDRriU6_1, SETMEMIMM_SHORT, L4_ior_memoph_io, or>;
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
                       CLRMEMIMM, L4_iand_memopw_io, and>;
  defm : MemOpi_bitPats<load, store, Set5ImmPred, u6_2ExtPred, ADDRriU6_2,
                       SETMEMIMM, L4_ior_memopw_io, or>;
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
  def : Pat <(stOp (OpNode (ldOp (addrPred IntRegs:$addr, extPred:$offset)),
                           (i32 IntRegs:$addend)),
                   (addrPred (i32 IntRegs:$addr), extPred:$offset)),
             (MI IntRegs:$addr, extPred:$offset, (i32 IntRegs:$addend) )>;

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
                       L4_add_memoph_io, L4_sub_memoph_io,
                       L4_and_memoph_io, L4_or_memoph_io>;
  // Byte
  defm : MemOPr_ALUOp <ldOpByte, truncstorei8, ADDRriU6_0, u6ExtPred,
                       L4_add_memopb_io, L4_sub_memopb_io,
                       L4_and_memopb_io, L4_or_memopb_io>;
}

// Define 'def Pats' for MemOps with register addend.
let Predicates = [HasV4T, UseMEMOP] in {
  // Byte, Half Word
  defm : MemOPr_ExtType<zextloadi8, zextloadi16>; // zero extend
  defm : MemOPr_ExtType<sextloadi8, sextloadi16>; // sign extend
  defm : MemOPr_ExtType<extloadi8,  extloadi16>;  // any extend
  // Word
  defm : MemOPr_ALUOp <load, store, ADDRriU6_2, u6_2ExtPred, L4_add_memopw_io,
                       L4_sub_memopw_io, L4_and_memopw_io, L4_or_memopw_io >;
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

// p=!cmp.eq(r1,#s10)
let isCodeGenOnly = 0 in {
def C4_cmpneqi  : T_CMP <"cmp.eq",  0b00, 1, s10Ext>;
def C4_cmpltei  : T_CMP <"cmp.gt",  0b01, 1, s10Ext>;
def C4_cmplteui : T_CMP <"cmp.gtu", 0b10, 1, u9Ext>;
}

def : T_CMP_pat <C4_cmpneqi,  setne,  s10ExtPred>;
def : T_CMP_pat <C4_cmpltei,  setle,  s10ExtPred>;
def : T_CMP_pat <C4_cmplteui, setule, u9ImmPred>;

// rs <= rt -> !(rs > rt).
/*
def: Pat<(i1 (setle (i32 IntRegs:$src1), s10ExtPred:$src2)),
         (C2_not (C2_cmpgti IntRegs:$src1, s10ExtPred:$src2))>;
//         (C4_cmpltei IntRegs:$src1, s10ExtPred:$src2)>;
*/
// Map cmplt(Rs, Imm) -> !cmpgt(Rs, Imm-1).
def: Pat<(i1 (setlt (i32 IntRegs:$src1), s8ExtPred:$src2)),
         (C4_cmpltei IntRegs:$src1, (DEC_CONST_SIGNED s8ExtPred:$src2))>;

// rs != rt -> !(rs == rt).
def: Pat<(i1 (setne (i32 IntRegs:$src1), s10ExtPred:$src2)),
         (C4_cmpneqi IntRegs:$src1, s10ExtPred:$src2)>;

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
           (i32 (TFR_condset_ii (i1 (A4_cmpbeqi (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (A4_cmpbeqi (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (A4_cmpbeq (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (A4_cmpbeq (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (A4_cmpbgtui (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (A4_cmpbgtui (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgtu (i32 IntRegs:$Rt),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgt (i32 IntRegs:$Rt),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgtu (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgtu (i32 IntRegs:$Rt),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgt (i32 IntRegs:$Rt),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgtu (i32 IntRegs:$Rs),
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
           (i32 (TFR_condset_ii (i1 (C2_cmpgt (i32 IntRegs:$Rs),
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
  (i32 (C2_muxii (i1 (A4_cmpbgtui (i32 IntRegs:$src1),
                                 (DEC_CONST_BYTE u7StrictPosImmPred:$src2))),
                   0, 1))>,
                   Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// XTYPE/PRED -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Multiclass for DeallocReturn
//===----------------------------------------------------------------------===//
class L4_RETURN<string mnemonic, bit isNot, bit isPredNew, bit isTak>
  : LD0Inst<(outs), (ins PredRegs:$src),
  !if(isNot, "if (!$src", "if ($src")#
  !if(isPredNew, ".new) ", ") ")#mnemonic#
  !if(isPredNew, #!if(isTak,":t", ":nt"),""),
  [], "", LD_tc_3or4stall_SLOT0> {

    bits<2> src;
    let BaseOpcode = "L4_RETURN";
    let isPredicatedFalse = isNot;
    let isPredicatedNew = isPredNew;
    let isTaken = isTak;
    let IClass = 0b1001;

    let Inst{27-16} = 0b011000011110;

    let Inst{13} = isNot;
    let Inst{12} = isTak;
    let Inst{11} = isPredNew;
    let Inst{10} = 0b0;
    let Inst{9-8} = src;
    let Inst{4-0} = 0b11110;
  }

// Produce all predicated forms, p, !p, p.new, !p.new, :t, :nt
multiclass L4_RETURN_PRED<string mnemonic, bit PredNot> {
  let isPredicated = 1 in {
    def _#NAME# : L4_RETURN <mnemonic, PredNot, 0, 1>;
    def _#NAME#new_pnt : L4_RETURN <mnemonic, PredNot, 1, 0>;
    def _#NAME#new_pt : L4_RETURN <mnemonic, PredNot, 1, 1>;
  }
}

multiclass LD_MISC_L4_RETURN<string mnemonic> {
  let isBarrier = 1, isPredicable = 1 in
    def NAME : LD0Inst <(outs), (ins), mnemonic, [], "",
                        LD_tc_3or4stall_SLOT0> {
      let BaseOpcode = "L4_RETURN";
      let IClass = 0b1001;
      let Inst{27-16} = 0b011000011110;
      let Inst{13-10} = 0b0000;
      let Inst{4-0} = 0b11110;
    }
  defm t : L4_RETURN_PRED<mnemonic, 0 >;
  defm f : L4_RETURN_PRED<mnemonic, 1 >;
}

let isReturn = 1, isTerminator = 1,
    Defs = [R29, R30, R31, PC], Uses = [R30], hasSideEffects = 0,
    validSubTargets = HasV4SubT, isCodeGenOnly = 0 in
defm L4_return: LD_MISC_L4_RETURN <"dealloc_return">, PredNewRel;

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

//===----------------------------------------------------------------------===//
// Template class for non predicated store instructions with
// GP-Relative or absolute addressing.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, isPredicable = 1, isNVStorable = 1 in
class T_StoreAbsGP <string mnemonic, RegisterClass RC, Operand ImmOp,
                    bits<2>MajOp, Operand AddrOp, bit isAbs, bit isHalf>
  : STInst<(outs), (ins AddrOp:$addr, RC:$src),
  mnemonic # !if(isAbs, "(##", "(#")#"$addr) = $src"#!if(isHalf, ".h",""),
  [], "", V2LDST_tc_st_SLOT01> {
    bits<19> addr;
    bits<5> src;
    bits<16> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "u16_3Imm"), addr{18-3},
                     !if (!eq(ImmOpStr, "u16_2Imm"), addr{17-2},
                     !if (!eq(ImmOpStr, "u16_1Imm"), addr{16-1},
                                      /* u16_0Imm */ addr{15-0})));
    let IClass = 0b0100;
    let Inst{27} = 1;
    let Inst{26-25} = offsetBits{15-14};
    let Inst{24}    = 0b0;
    let Inst{23-22} = MajOp;
    let Inst{21}    = isHalf;
    let Inst{20-16} = offsetBits{13-9};
    let Inst{13}    = offsetBits{8};
    let Inst{12-8}  = src;
    let Inst{7-0}   = offsetBits{7-0};
  }

//===----------------------------------------------------------------------===//
// Template class for predicated store instructions with
// GP-Relative or absolute addressing.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, isPredicated = 1, isNVStorable = 1, opExtentBits = 6,
    opExtendable = 1 in
class T_StoreAbs_Pred <string mnemonic, RegisterClass RC, bits<2> MajOp,
                       bit isHalf, bit isNot, bit isNew>
  : STInst<(outs), (ins PredRegs:$src1, u6Ext:$absaddr, RC: $src2),
  !if(isNot, "if (!$src1", "if ($src1")#!if(isNew, ".new) ",
  ") ")#mnemonic#"(#$absaddr) = $src2"#!if(isHalf, ".h",""),
  [], "", ST_tc_st_SLOT01>, AddrModeRel {
    bits<2> src1;
    bits<6> absaddr;
    bits<5> src2;

    let isPredicatedNew = isNew;
    let isPredicatedFalse = isNot;

    let IClass = 0b1010;

    let Inst{27-24} = 0b1111;
    let Inst{23-22} = MajOp;
    let Inst{21}    = isHalf;
    let Inst{17-16} = absaddr{5-4};
    let Inst{13}    = isNew;
    let Inst{12-8}  = src2;
    let Inst{7}     = 0b1;
    let Inst{6-3}   = absaddr{3-0};
    let Inst{2}     = isNot;
    let Inst{1-0}   = src1;
  }

//===----------------------------------------------------------------------===//
// Template class for predicated store instructions with absolute addressing.
//===----------------------------------------------------------------------===//
class T_StoreAbs <string mnemonic, RegisterClass RC, Operand ImmOp,
                 bits<2> MajOp, bit isHalf>
  : T_StoreAbsGP <mnemonic, RC, ImmOp, MajOp, u0AlwaysExt, 1, isHalf>,
                  AddrModeRel {
  string ImmOpStr = !cast<string>(ImmOp);
  let opExtentBits = !if (!eq(ImmOpStr, "u16_3Imm"), 19,
                     !if (!eq(ImmOpStr, "u16_2Imm"), 18,
                     !if (!eq(ImmOpStr, "u16_1Imm"), 17,
                                      /* u16_0Imm */ 16)));

  let opExtentAlign = !if (!eq(ImmOpStr, "u16_3Imm"), 3,
                      !if (!eq(ImmOpStr, "u16_2Imm"), 2,
                      !if (!eq(ImmOpStr, "u16_1Imm"), 1,
                                       /* u16_0Imm */ 0)));
}

//===----------------------------------------------------------------------===//
// Multiclass for store instructions with absolute addressing.
//===----------------------------------------------------------------------===//
let validSubTargets = HasV4SubT, addrMode = Absolute, isExtended = 1 in
multiclass ST_Abs<string mnemonic, string CextOp, RegisterClass RC,
                  Operand ImmOp, bits<2> MajOp, bit isHalf = 0> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_abs in {
    let opExtendable = 0, isPredicable = 1 in
    def S2_#NAME#abs : T_StoreAbs <mnemonic, RC, ImmOp, MajOp, isHalf>;

    // Predicated
    def S4_p#NAME#t_abs : T_StoreAbs_Pred<mnemonic, RC, MajOp, isHalf, 0, 0>;
    def S4_p#NAME#f_abs : T_StoreAbs_Pred<mnemonic, RC, MajOp, isHalf, 1, 0>;

    // .new Predicated
    def S4_p#NAME#tnew_abs : T_StoreAbs_Pred<mnemonic, RC, MajOp, isHalf, 0, 1>;
    def S4_p#NAME#fnew_abs : T_StoreAbs_Pred<mnemonic, RC, MajOp, isHalf, 1, 1>;
  }
}

//===----------------------------------------------------------------------===//
// Template class for non predicated new-value store instructions with
// GP-Relative or absolute addressing.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, isPredicable = 1, mayStore = 1, isNVStore = 1,
    isNewValue = 1, opNewValue = 1 in
class T_StoreAbsGP_NV <string mnemonic, Operand ImmOp, bits<2>MajOp, bit isAbs>
  : NVInst_V4<(outs), (ins u0AlwaysExt:$addr, IntRegs:$src),
  mnemonic # !if(isAbs, "(##", "(#")#"$addr) = $src.new",
  [], "", V2LDST_tc_st_SLOT0> {
    bits<19> addr;
    bits<3> src;
    bits<16> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "u16_3Imm"), addr{18-3},
                     !if (!eq(ImmOpStr, "u16_2Imm"), addr{17-2},
                     !if (!eq(ImmOpStr, "u16_1Imm"), addr{16-1},
                                      /* u16_0Imm */ addr{15-0})));
    let IClass = 0b0100;

    let Inst{27} = 1;
    let Inst{26-25} = offsetBits{15-14};
    let Inst{24-21} = 0b0101;
    let Inst{20-16} = offsetBits{13-9};
    let Inst{13}    = offsetBits{8};
    let Inst{12-11} = MajOp;
    let Inst{10-8}  = src;
    let Inst{7-0}   = offsetBits{7-0};
  }

//===----------------------------------------------------------------------===//
// Template class for predicated new-value store instructions with
// absolute addressing.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, isPredicated = 1, mayStore = 1, isNVStore = 1,
    isNewValue = 1, opNewValue = 2, opExtentBits = 6, opExtendable = 1 in
class T_StoreAbs_NV_Pred <string mnemonic, bits<2> MajOp, bit isNot, bit isNew>
  : NVInst_V4<(outs), (ins PredRegs:$src1, u6Ext:$absaddr, IntRegs:$src2),
  !if(isNot, "if (!$src1", "if ($src1")#!if(isNew, ".new) ",
  ") ")#mnemonic#"(#$absaddr) = $src2.new",
  [], "", ST_tc_st_SLOT0>, AddrModeRel {
    bits<2> src1;
    bits<6> absaddr;
    bits<3> src2;

    let isPredicatedNew = isNew;
    let isPredicatedFalse = isNot;

    let IClass = 0b1010;

    let Inst{27-24} = 0b1111;
    let Inst{23-21} = 0b101;
    let Inst{17-16} = absaddr{5-4};
    let Inst{13}    = isNew;
    let Inst{12-11} = MajOp;
    let Inst{10-8}  = src2;
    let Inst{7}     = 0b1;
    let Inst{6-3}   = absaddr{3-0};
    let Inst{2}     = isNot;
    let Inst{1-0}   = src1;
}

//===----------------------------------------------------------------------===//
// Template class for non-predicated new-value store instructions with
// absolute addressing.
//===----------------------------------------------------------------------===//
class T_StoreAbs_NV <string mnemonic, Operand ImmOp, bits<2> MajOp>
  : T_StoreAbsGP_NV <mnemonic, ImmOp, MajOp, 1>, AddrModeRel {

  string ImmOpStr = !cast<string>(ImmOp);
  let opExtentBits = !if (!eq(ImmOpStr, "u16_3Imm"), 19,
                     !if (!eq(ImmOpStr, "u16_2Imm"), 18,
                     !if (!eq(ImmOpStr, "u16_1Imm"), 17,
                                      /* u16_0Imm */ 16)));

  let opExtentAlign = !if (!eq(ImmOpStr, "u16_3Imm"), 3,
                      !if (!eq(ImmOpStr, "u16_2Imm"), 2,
                      !if (!eq(ImmOpStr, "u16_1Imm"), 1,
                                       /* u16_0Imm */ 0)));
}

//===----------------------------------------------------------------------===//
// Multiclass for new-value store instructions with absolute addressing.
//===----------------------------------------------------------------------===//
let validSubTargets = HasV4SubT, addrMode = Absolute, isExtended = 1  in
multiclass ST_Abs_NV <string mnemonic, string CextOp, Operand ImmOp,
                   bits<2> MajOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_abs in {
    let opExtendable = 0, isPredicable = 1 in
    def S2_#NAME#newabs : T_StoreAbs_NV <mnemonic, ImmOp, MajOp>;

    // Predicated
    def S4_p#NAME#newt_abs  : T_StoreAbs_NV_Pred <mnemonic, MajOp, 0, 0>;
    def S4_p#NAME#newf_abs  : T_StoreAbs_NV_Pred <mnemonic, MajOp, 1, 0>;

    // .new Predicated
    def S4_p#NAME#newtnew_abs : T_StoreAbs_NV_Pred <mnemonic, MajOp, 0, 1>;
    def S4_p#NAME#newfnew_abs : T_StoreAbs_NV_Pred <mnemonic, MajOp, 1, 1>;
  }
}

//===----------------------------------------------------------------------===//
// Stores with absolute addressing
//===----------------------------------------------------------------------===//
let accessSize = ByteAccess, isCodeGenOnly = 0 in
defm storerb : ST_Abs    <"memb", "STrib", IntRegs, u16_0Imm, 0b00>,
               ST_Abs_NV <"memb", "STrib", u16_0Imm, 0b00>;

let accessSize = HalfWordAccess, isCodeGenOnly = 0 in
defm storerh : ST_Abs    <"memh", "STrih", IntRegs, u16_1Imm, 0b01>,
               ST_Abs_NV <"memh", "STrih", u16_1Imm, 0b01>;

let accessSize = WordAccess, isCodeGenOnly = 0 in
defm storeri : ST_Abs    <"memw", "STriw", IntRegs, u16_2Imm, 0b10>,
               ST_Abs_NV <"memw", "STriw", u16_2Imm, 0b10>;

let isNVStorable = 0, accessSize = DoubleWordAccess, isCodeGenOnly = 0 in
defm storerd : ST_Abs <"memd", "STrid", DoubleRegs, u16_3Imm, 0b11>;

let isNVStorable = 0, accessSize = HalfWordAccess, isCodeGenOnly = 0 in
defm storerf : ST_Abs <"memh", "STrif", IntRegs, u16_1Imm, 0b01, 1>;

//===----------------------------------------------------------------------===//
// GP-relative stores.
// mem[bhwd](#global)=Rt
// Once predicated, these instructions map to absolute addressing mode.
// if ([!]Pv[.new]) mem[bhwd](##global)=Rt
//===----------------------------------------------------------------------===//

let validSubTargets = HasV4SubT in
class T_StoreGP <string mnemonic, string BaseOp, RegisterClass RC,
                 Operand ImmOp, bits<2> MajOp, bit isHalf = 0>
  : T_StoreAbsGP <mnemonic, RC, ImmOp, MajOp, globaladdress, 0, isHalf> {
    // Set BaseOpcode same as absolute addressing instructions so that
    // non-predicated GP-Rel instructions can have relate with predicated
    // Absolute instruction.
    let BaseOpcode = BaseOp#_abs;
  }

let validSubTargets = HasV4SubT in
multiclass ST_GP <string mnemonic, string BaseOp, Operand ImmOp,
                  bits<2> MajOp, bit isHalf = 0> {
  // Set BaseOpcode same as absolute addressing instructions so that
  // non-predicated GP-Rel instructions can have relate with predicated
  // Absolute instruction.
  let BaseOpcode = BaseOp#_abs in {
    def NAME#gp : T_StoreAbsGP <mnemonic, IntRegs, ImmOp, MajOp,
                                globaladdress, 0, isHalf>;
    // New-value store
    def NAME#newgp : T_StoreAbsGP_NV <mnemonic, ImmOp, MajOp, 0> ;
  }
}

let accessSize = ByteAccess in
defm S2_storerb : ST_GP<"memb", "STrib", u16_0Imm, 0b00>, NewValueRel;

let accessSize = HalfWordAccess in
defm S2_storerh : ST_GP<"memh", "STrih", u16_1Imm, 0b01>, NewValueRel;

let accessSize = WordAccess in
defm S2_storeri : ST_GP<"memw", "STriw", u16_2Imm, 0b10>, NewValueRel;

let isNVStorable = 0, accessSize = DoubleWordAccess in
def S2_storerdgp : T_StoreGP <"memd", "STrid", DoubleRegs,
                              u16_3Imm, 0b11>, PredNewRel;

let isNVStorable = 0, accessSize = HalfWordAccess in
def S2_storerfgp : T_StoreGP <"memh", "STrif", IntRegs,
                              u16_1Imm, 0b01, 1>, PredNewRel;

let Predicates = [HasV4T], AddedComplexity = 30 in {
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
                        (HexagonCONST32 tglobaladdr:$absaddr)),
          (S2_storerbabs tglobaladdr: $absaddr, IntRegs: $src1)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                          (HexagonCONST32 tglobaladdr:$absaddr)),
          (S2_storerhabs tglobaladdr: $absaddr, IntRegs: $src1)>;

def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32 tglobaladdr:$absaddr)),
          (S2_storeriabs tglobaladdr: $absaddr, IntRegs: $src1)>;

def : Pat<(store (i64 DoubleRegs:$src1),
                 (HexagonCONST32 tglobaladdr:$absaddr)),
          (S2_storerdabs tglobaladdr: $absaddr, DoubleRegs: $src1)>;
}

// 64 bit atomic store
def : Pat <(atomic_store_64 (HexagonCONST32_GP tglobaladdr:$global),
                            (i64 DoubleRegs:$src1)),
           (S2_storerdgp tglobaladdr:$global, (i64 DoubleRegs:$src1))>,
           Requires<[HasV4T]>;

// Map from store(globaladdress) -> memd(#foo)
let AddedComplexity = 100 in
def : Pat <(store (i64 DoubleRegs:$src1),
                  (HexagonCONST32_GP tglobaladdr:$global)),
           (S2_storerdgp tglobaladdr:$global, (i64 DoubleRegs:$src1))>;

// 8 bit atomic store
def : Pat < (atomic_store_8 (HexagonCONST32_GP tglobaladdr:$global),
                            (i32 IntRegs:$src1)),
            (S2_storerbgp tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from store(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1),
          (HexagonCONST32_GP tglobaladdr:$global)),
          (S2_storerbgp tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from "i1 = constant<-1>; memw(CONST32(#foo)) = i1"
//       to "r0 = 1; memw(#foo) = r0"
let AddedComplexity = 100 in
def : Pat<(store (i1 -1), (HexagonCONST32_GP tglobaladdr:$global)),
          (S2_storerbgp tglobaladdr:$global, (A2_tfrsi 1))>;

def : Pat<(atomic_store_16 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (S2_storerhgp tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from store(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1),
                         (HexagonCONST32_GP tglobaladdr:$global)),
          (S2_storerhgp tglobaladdr:$global, (i32 IntRegs:$src1))>;

// 32 bit atomic store
def : Pat<(atomic_store_32 (HexagonCONST32_GP tglobaladdr:$global),
                           (i32 IntRegs:$src1)),
          (S2_storerigp tglobaladdr:$global, (i32 IntRegs:$src1))>;

// Map from store(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1), (HexagonCONST32_GP tglobaladdr:$global)),
          (S2_storerigp tglobaladdr:$global, (i32 IntRegs:$src1))>;

//===----------------------------------------------------------------------===//
// Template class for non predicated load instructions with
// absolute addressing mode.
//===----------------------------------------------------------------------===//
let isPredicable = 1, hasSideEffects = 0, validSubTargets = HasV4SubT in
class T_LoadAbsGP <string mnemonic, RegisterClass RC, Operand ImmOp,
                   bits<3> MajOp, Operand AddrOp, bit isAbs>
  : LDInst <(outs RC:$dst), (ins AddrOp:$addr),
  "$dst = "#mnemonic# !if(isAbs, "(##", "(#")#"$addr)",
  [], "", V2LDST_tc_ld_SLOT01> {
    bits<5> dst;
    bits<19> addr;
    bits<16> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "u16_3Imm"), addr{18-3},
                     !if (!eq(ImmOpStr, "u16_2Imm"), addr{17-2},
                     !if (!eq(ImmOpStr, "u16_1Imm"), addr{16-1},
                                      /* u16_0Imm */ addr{15-0})));

    let IClass = 0b0100;

    let Inst{27}    = 0b1;
    let Inst{26-25} = offsetBits{15-14};
    let Inst{24}    = 0b1;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = offsetBits{13-9};
    let Inst{13-5}  = offsetBits{8-0};
    let Inst{4-0}   = dst;
  }

class T_LoadAbs <string mnemonic, RegisterClass RC, Operand ImmOp,
                 bits<3> MajOp>
  : T_LoadAbsGP <mnemonic, RC, ImmOp, MajOp, u0AlwaysExt, 1>, AddrModeRel {

    string ImmOpStr = !cast<string>(ImmOp);
    let opExtentBits = !if (!eq(ImmOpStr, "u16_3Imm"), 19,
                       !if (!eq(ImmOpStr, "u16_2Imm"), 18,
                       !if (!eq(ImmOpStr, "u16_1Imm"), 17,
                                        /* u16_0Imm */ 16)));

    let opExtentAlign = !if (!eq(ImmOpStr, "u16_3Imm"), 3,
                        !if (!eq(ImmOpStr, "u16_2Imm"), 2,
                        !if (!eq(ImmOpStr, "u16_1Imm"), 1,
                                        /* u16_0Imm */ 0)));
  }
//===----------------------------------------------------------------------===//
// Template class for predicated load instructions with
// absolute addressing mode.
//===----------------------------------------------------------------------===//
let isPredicated = 1, hasNewValue = 1, opExtentBits = 6, opExtendable = 2 in
class T_LoadAbs_Pred <string mnemonic, RegisterClass RC, bits<3> MajOp,
                      bit isPredNot, bit isPredNew>
  : LDInst <(outs RC:$dst), (ins PredRegs:$src1, u6Ext:$absaddr),
  !if(isPredNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#"$dst = "#mnemonic#"(#$absaddr)">, AddrModeRel {
    bits<5> dst;
    bits<2> src1;
    bits<6> absaddr;

    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = isPredNot;

    let IClass = 0b1001;

    let Inst{27-24} = 0b1111;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = absaddr{5-1};
    let Inst{13} = 0b1;
    let Inst{12} = isPredNew;
    let Inst{11} = isPredNot;
    let Inst{10-9} = src1;
    let Inst{8} = absaddr{0};
    let Inst{7} = 0b1;
    let Inst{4-0} = dst;
  }

//===----------------------------------------------------------------------===//
// Multiclass for the load instructions with absolute addressing mode.
//===----------------------------------------------------------------------===//
multiclass LD_Abs_Pred<string mnemonic, RegisterClass RC, bits<3> MajOp,
                       bit PredNot> {
  def _abs : T_LoadAbs_Pred <mnemonic, RC, MajOp, PredNot, 0>;
  // Predicate new
  def new_abs : T_LoadAbs_Pred <mnemonic, RC, MajOp, PredNot, 1>;
}

let addrMode = Absolute, isExtended = 1 in
multiclass LD_Abs<string mnemonic, string CextOp, RegisterClass RC,
                  Operand ImmOp, bits<3> MajOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_abs in {
    let opExtendable = 1, isPredicable = 1 in
    def L4_#NAME#_abs: T_LoadAbs <mnemonic, RC, ImmOp, MajOp>;

    // Predicated
    defm L4_p#NAME#t : LD_Abs_Pred<mnemonic, RC, MajOp, 0>;
    defm L4_p#NAME#f : LD_Abs_Pred<mnemonic, RC, MajOp, 1>;
  }
}

let accessSize = ByteAccess, hasNewValue = 1, isCodeGenOnly = 0 in {
  defm loadrb  : LD_Abs<"memb",  "LDrib",  IntRegs, u16_0Imm, 0b000>;
  defm loadrub : LD_Abs<"memub", "LDriub", IntRegs, u16_0Imm, 0b001>;
}

let accessSize = HalfWordAccess, hasNewValue = 1, isCodeGenOnly = 0 in {
  defm loadrh  : LD_Abs<"memh",  "LDrih",  IntRegs, u16_1Imm, 0b010>;
  defm loadruh : LD_Abs<"memuh", "LDriuh", IntRegs, u16_1Imm, 0b011>;
}

let accessSize = WordAccess, hasNewValue = 1, isCodeGenOnly = 0 in
defm loadri  : LD_Abs<"memw",  "LDriw",  IntRegs, u16_2Imm, 0b100>;

let accessSize = DoubleWordAccess, isCodeGenOnly = 0 in
defm loadrd  : LD_Abs<"memd",  "LDrid", DoubleRegs, u16_3Imm, 0b110>;

//===----------------------------------------------------------------------===//
// multiclass for load instructions with GP-relative addressing mode.
// Rx=mem[bhwd](##global)
// Once predicated, these instructions map to absolute addressing mode.
// if ([!]Pv[.new]) Rx=mem[bhwd](##global)
//===----------------------------------------------------------------------===//

class T_LoadGP <string mnemonic, string BaseOp, RegisterClass RC, Operand ImmOp,
                bits<3> MajOp>
  : T_LoadAbsGP <mnemonic, RC, ImmOp, MajOp, globaladdress, 0>, PredNewRel {
    let BaseOpcode = BaseOp#_abs;
  }

let accessSize = ByteAccess, hasNewValue = 1 in {
  def L2_loadrbgp  : T_LoadGP<"memb",  "LDrib",  IntRegs, u16_0Imm, 0b000>;
  def L2_loadrubgp : T_LoadGP<"memub", "LDriub", IntRegs, u16_0Imm, 0b001>;
}

let accessSize = HalfWordAccess, hasNewValue = 1 in {
  def L2_loadrhgp  : T_LoadGP<"memh",  "LDrih",  IntRegs, u16_1Imm, 0b010>;
  def L2_loadruhgp : T_LoadGP<"memuh", "LDriuh", IntRegs, u16_1Imm, 0b011>;
}

let accessSize = WordAccess, hasNewValue = 1 in
def L2_loadrigp  : T_LoadGP<"memw",  "LDriw",  IntRegs, u16_2Imm, 0b100>;

let accessSize = DoubleWordAccess in
def L2_loadrdgp  : T_LoadGP<"memd", "LDrid", DoubleRegs, u16_3Imm, 0b110>;

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(i32 (load (HexagonCONST32 tglobaladdr:$absaddr))),
          (L4_loadri_abs tglobaladdr: $absaddr)>;

def : Pat<(i32 (sextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (L4_loadrb_abs tglobaladdr:$absaddr)>;

def : Pat<(i32 (zextloadi8 (HexagonCONST32 tglobaladdr:$absaddr))),
          (L4_loadrub_abs tglobaladdr:$absaddr)>;

def : Pat<(i32 (sextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (L4_loadrh_abs tglobaladdr:$absaddr)>;

def : Pat<(i32 (zextloadi16 (HexagonCONST32 tglobaladdr:$absaddr))),
          (L4_loadruh_abs tglobaladdr:$absaddr)>;
}

def : Pat <(atomic_load_64 (HexagonCONST32_GP tglobaladdr:$global)),
           (i64 (L2_loadrdgp tglobaladdr:$global))>;

def : Pat <(atomic_load_32 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (L2_loadrigp tglobaladdr:$global))>;

def : Pat <(atomic_load_16 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (L2_loadruhgp tglobaladdr:$global))>;

def : Pat <(atomic_load_8 (HexagonCONST32_GP tglobaladdr:$global)),
           (i32 (L2_loadrubgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memw(#foo + 0)
let AddedComplexity = 100 in
def : Pat <(i64 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i64 (L2_loadrdgp tglobaladdr:$global))>;

// Map from Pd = load(globaladdress) -> Rd = memb(globaladdress), Pd = Rd
let AddedComplexity = 100 in
def : Pat <(i1 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i1 (C2_tfrrp (i32 (L2_loadrbgp tglobaladdr:$global))))>;

// When the Interprocedural Global Variable optimizer realizes that a certain
// global variable takes only two constant values, it shrinks the global to
// a boolean. Catch those loads here in the following 3 patterns.
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrbgp tglobaladdr:$global))>;

let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrbgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrbgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memb(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrbgp tglobaladdr:$global))>;

let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi1 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrubgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memub(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi8 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrubgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (extloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrhgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (sextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrhgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memuh(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (zextloadi16 (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadruhgp tglobaladdr:$global))>;

// Map from load(globaladdress) -> memw(#foo)
let AddedComplexity = 100 in
def : Pat <(i32 (load (HexagonCONST32_GP tglobaladdr:$global))),
           (i32 (L2_loadrigp tglobaladdr:$global))>;


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
hasSideEffects = 0, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                           (ins PredRegs:$src1, s16Ext:$src2),
           "if($src1) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50, isPredicatedFalse = 1,
hasSideEffects = 0, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cNotPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, s16Ext:$src2),
           "if(!$src1) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50,
hasSideEffects = 0, isPredicated = 1, validSubTargets = HasV4SubT in
def TFRI_cdnPt_V4 : ALU32_ri<(outs IntRegs:$dst),
                             (ins PredRegs:$src1, s16Ext:$src2),
           "if($src1.new) $dst = #$src2",
           []>,
           Requires<[HasV4T]>;

let isExtended = 1, opExtendable = 2, AddedComplexity=50, isPredicatedFalse = 1,
hasSideEffects = 0, isPredicated = 1, validSubTargets = HasV4SubT in
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
          (S2_storerbabs u0AlwaysExtPred:$src2, IntRegs: $src1)>;

def : Pat<(truncstorei16 (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (S2_storerhabs u0AlwaysExtPred:$src2, IntRegs: $src1)>;

def : Pat<(store (i32 IntRegs:$src1), u0AlwaysExtPred:$src2),
          (S2_storeriabs u0AlwaysExtPred:$src2, IntRegs: $src1)>;
}

let Predicates = [HasV4T], AddedComplexity  = 30 in {
def : Pat<(i32 (load u0AlwaysExtPred:$src)),
          (L4_loadri_abs u0AlwaysExtPred:$src)>;

def : Pat<(i32 (sextloadi8 u0AlwaysExtPred:$src)),
          (L4_loadrb_abs u0AlwaysExtPred:$src)>;

def : Pat<(i32 (zextloadi8 u0AlwaysExtPred:$src)),
          (L4_loadrub_abs u0AlwaysExtPred:$src)>;

def : Pat<(i32 (sextloadi16 u0AlwaysExtPred:$src)),
          (L4_loadrh_abs u0AlwaysExtPred:$src)>;

def : Pat<(i32 (zextloadi16 u0AlwaysExtPred:$src)),
          (L4_loadruh_abs u0AlwaysExtPred:$src)>;
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
          (i64 (A4_combineir (i32 0), (i32 (S2_cl0p DoubleRegs:$src1))))>,
          Requires<[HasV4T]>;

def : Pat<(i64 (cttz (i64 DoubleRegs:$src1))),
          (i64 (A4_combineir (i32 0), (i32 (S2_ct0p DoubleRegs:$src1))))>,
          Requires<[HasV4T]>;


// i8 -> i64 loads
// We need a complexity of 120 here to override preceding handling of
// zextloadi8.
let Predicates = [HasV4T], AddedComplexity = 120 in {
def:  Pat <(i64 (extloadi8 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A4_combineir 0, (L4_loadrb_abs tglobaladdr:$addr)))>;

def:  Pat <(i64 (zextloadi8 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A4_combineir 0, (L4_loadrub_abs tglobaladdr:$addr)))>;

def:  Pat <(i64 (sextloadi8 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A2_sxtw (L4_loadrb_abs tglobaladdr:$addr)))>;

def:  Pat <(i64 (extloadi8 FoldGlobalAddr:$addr)),
      (i64 (A4_combineir 0, (L4_loadrb_abs FoldGlobalAddr:$addr)))>;

def:  Pat <(i64 (zextloadi8 FoldGlobalAddr:$addr)),
      (i64 (A4_combineir 0, (L4_loadrub_abs FoldGlobalAddr:$addr)))>;

def:  Pat <(i64 (sextloadi8 FoldGlobalAddr:$addr)),
      (i64 (A2_sxtw (L4_loadrb_abs FoldGlobalAddr:$addr)))>;
}
// i16 -> i64 loads
// We need a complexity of 120 here to override preceding handling of
// zextloadi16.
let AddedComplexity = 120 in {
def:  Pat <(i64 (extloadi16 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A4_combineir 0, (L4_loadrh_abs tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi16 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A4_combineir 0, (L4_loadruh_abs tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi16 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A2_sxtw (L4_loadrh_abs tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (extloadi16 FoldGlobalAddr:$addr)),
      (i64 (A4_combineir 0, (L4_loadrh_abs FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi16 FoldGlobalAddr:$addr)),
      (i64 (A4_combineir 0, (L4_loadruh_abs FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi16 FoldGlobalAddr:$addr)),
      (i64 (A2_sxtw (L4_loadrh_abs FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;
}
// i32->i64 loads
// We need a complexity of 120 here to override preceding handling of
// zextloadi32.
let AddedComplexity = 120 in {
def:  Pat <(i64 (extloadi32 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A4_combineir 0, (L4_loadri_abs tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi32 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A4_combineir 0, (L4_loadri_abs tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi32 (NumUsesBelowThresCONST32 tglobaladdr:$addr))),
      (i64 (A2_sxtw (L4_loadri_abs tglobaladdr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (extloadi32 FoldGlobalAddr:$addr)),
      (i64 (A4_combineir 0, (L4_loadri_abs FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (zextloadi32 FoldGlobalAddr:$addr)),
      (i64 (A4_combineir 0, (L4_loadri_abs FoldGlobalAddr:$addr)))>,
      Requires<[HasV4T]>;

def:  Pat <(i64 (sextloadi32 FoldGlobalAddr:$addr)),
      (i64 (A2_sxtw (L4_loadri_abs FoldGlobalAddr:$addr)))>,
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
          (S2_storerdabs FoldGlobalAddrGP:$addr, (i64 DoubleRegs:$src1))>,
          Requires<[HasV4T]>;

def : Pat<(atomic_store_64 FoldGlobalAddrGP:$addr,
                           (i64 DoubleRegs:$src1)),
          (S2_storerdabs FoldGlobalAddrGP:$addr, (i64 DoubleRegs:$src1))>,
          Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(truncstorei8 (i32 IntRegs:$src1), FoldGlobalAddrGP:$addr),
          (S2_storerbabs FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

def : Pat<(atomic_store_8 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1)),
          (S2_storerbabs FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(truncstorei16 (i32 IntRegs:$src1), FoldGlobalAddrGP:$addr),
          (S2_storerhabs FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

def : Pat<(atomic_store_16 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1)),
          (S2_storerhabs FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

// Map from store(globaladdress + x) -> memw(#foo + x)
let AddedComplexity = 100 in
def : Pat<(store (i32 IntRegs:$src1), FoldGlobalAddrGP:$addr),
          (S2_storeriabs FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_store_32 FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1)),
          (S2_storeriabs FoldGlobalAddrGP:$addr, (i32 IntRegs:$src1))>,
            Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memd(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i64 (load FoldGlobalAddrGP:$addr)),
          (i64 (L4_loadrd_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_64 FoldGlobalAddrGP:$addr),
          (i64 (L4_loadrd_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (extloadi8 FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadrb_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memb(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (sextloadi8 FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadrb_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

//let AddedComplexity = 100 in
let AddedComplexity = 100 in
def : Pat<(i32 (extloadi16 FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadrh_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (sextloadi16 FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadrh_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memuh(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (zextloadi16 FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadruh_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_16 FoldGlobalAddrGP:$addr),
          (i32 (L4_loadruh_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memub(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (zextloadi8 FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadrub_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_8 FoldGlobalAddrGP:$addr),
          (i32 (L4_loadrub_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

// Map from load(globaladdress + x) -> memw(#foo + x)
let AddedComplexity = 100 in
def : Pat<(i32 (load FoldGlobalAddrGP:$addr)),
          (i32 (L4_loadri_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

def : Pat<(atomic_load_32 FoldGlobalAddrGP:$addr),
          (i32 (L4_loadri_abs FoldGlobalAddrGP:$addr))>,
           Requires<[HasV4T]>;

//===----------------------------------------------------------------------===//
// :raw for of boundscheck:hi:lo insns
//===----------------------------------------------------------------------===//

// A4_boundscheck_lo: Detect if a register is within bounds.
let hasSideEffects = 0, isCodeGenOnly = 0 in
def A4_boundscheck_lo: ALU64Inst <
  (outs PredRegs:$Pd),
  (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Pd = boundscheck($Rss, $Rtt):raw:lo"> {
    bits<2> Pd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1101;

    let Inst{27-23} = 0b00100;
    let Inst{13} = 0b1;
    let Inst{7-5} = 0b100;
    let Inst{1-0} = Pd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

// A4_boundscheck_hi: Detect if a register is within bounds.
let hasSideEffects = 0, isCodeGenOnly = 0 in
def A4_boundscheck_hi: ALU64Inst <
  (outs PredRegs:$Pd),
  (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Pd = boundscheck($Rss, $Rtt):raw:hi"> {
    bits<2> Pd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1101;

    let Inst{27-23} = 0b00100;
    let Inst{13} = 0b1;
    let Inst{7-5} = 0b101;
    let Inst{1-0} = Pd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

let hasSideEffects = 0 in
def A4_boundscheck : MInst <
  (outs PredRegs:$Pd), (ins IntRegs:$Rs, DoubleRegs:$Rtt),
  "$Pd=boundscheck($Rs,$Rtt)">;

// A4_tlbmatch: Detect if a VA/ASID matches a TLB entry.
let isPredicateLate = 1, hasSideEffects = 0, isCodeGenOnly = 0 in
def A4_tlbmatch : ALU64Inst<(outs PredRegs:$Pd),
  (ins DoubleRegs:$Rs, IntRegs:$Rt),
  "$Pd = tlbmatch($Rs, $Rt)",
  [], "", ALU64_tc_2early_SLOT23> {
    bits<2> Pd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1101;
    let Inst{27-23} = 0b00100;
    let Inst{20-16} = Rs;
    let Inst{13} = 0b1;
    let Inst{12-8} = Rt;
    let Inst{7-5} = 0b011;
    let Inst{1-0} = Pd;
  }

// We need custom lowering of ISD::PREFETCH into HexagonISD::DCFETCH
// because the SDNode ISD::PREFETCH has properties MayLoad and MayStore.
// We don't really want either one here.
def SDTHexagonDCFETCH : SDTypeProfile<0, 2, [SDTCisPtrTy<0>,SDTCisInt<1>]>;
def HexagonDCFETCH : SDNode<"HexagonISD::DCFETCH", SDTHexagonDCFETCH,
                            [SDNPHasChain]>;

// Use LD0Inst for dcfetch, but set "mayLoad" to 0 because this doesn't
// really do a load.
let hasSideEffects = 1, mayLoad = 0, isCodeGenOnly = 0 in
def Y2_dcfetchbo : LD0Inst<(outs), (ins IntRegs:$Rs, u11_3Imm:$u11_3),
      "dcfetch($Rs + #$u11_3)",
      [(HexagonDCFETCH IntRegs:$Rs, u11_3ImmPred:$u11_3)],
      "", LD_tc_ld_SLOT0> {
  bits<5> Rs;
  bits<14> u11_3;

  let IClass = 0b1001;
  let Inst{27-21} = 0b0100000;
  let Inst{20-16} = Rs;
  let Inst{13} = 0b0;
  let Inst{10-0} = u11_3{13-3};
}

//===----------------------------------------------------------------------===//
// Compound instructions
//===----------------------------------------------------------------------===//

let isBranch = 1, hasSideEffects = 0, isExtentSigned = 1,
    isPredicated = 1, isPredicatedNew = 1, isExtendable = 1,
    opExtentBits = 11, opExtentAlign = 2, opExtendable = 1,
    isTerminator = 1, validSubTargets = HasV4SubT in
class CJInst_tstbit_R0<string px, bit np, string tnt>
  : InstHexagon<(outs), (ins IntRegs:$Rs, brtarget:$r9_2),
  ""#px#" = tstbit($Rs, #0); if ("
    #!if(np, "!","")#""#px#".new) jump:"#tnt#" $r9_2",
  [], "", COMPOUND, TypeCOMPOUND> {
  bits<4> Rs;
  bits<11> r9_2;

  // np: !p[01]
  let isPredicatedFalse = np;
  // tnt: Taken/Not Taken
  let isBrTaken = !if (!eq(tnt, "t"), "true", "false");
  let isTaken   = !if (!eq(tnt, "t"), 1, 0);

  let IClass = 0b0001;
  let Inst{27-26} = 0b00;
  let Inst{25} = !if (!eq(px, "!p1"), 1,
                 !if (!eq(px,  "p1"), 1, 0));
  let Inst{24-23} = 0b11;
  let Inst{22} = np;
  let Inst{21-20} = r9_2{10-9};
  let Inst{19-16} = Rs;
  let Inst{13} = !if (!eq(tnt, "t"), 1, 0);
  let Inst{9-8} = 0b11;
  let Inst{7-1} = r9_2{8-2};
}

let Defs = [PC, P0], Uses = [P0], isCodeGenOnly = 0 in {
  def J4_tstbit0_tp0_jump_nt : CJInst_tstbit_R0<"p0", 0, "nt">;
  def J4_tstbit0_tp0_jump_t : CJInst_tstbit_R0<"p0", 0, "t">;
  def J4_tstbit0_fp0_jump_nt : CJInst_tstbit_R0<"p0", 1, "nt">;
  def J4_tstbit0_fp0_jump_t : CJInst_tstbit_R0<"p0", 1, "t">;
}

let Defs = [PC, P1], Uses = [P1], isCodeGenOnly = 0 in {
  def J4_tstbit0_tp1_jump_nt : CJInst_tstbit_R0<"p1", 0, "nt">;
  def J4_tstbit0_tp1_jump_t : CJInst_tstbit_R0<"p1", 0, "t">;
  def J4_tstbit0_fp1_jump_nt : CJInst_tstbit_R0<"p1", 1, "nt">;
  def J4_tstbit0_fp1_jump_t : CJInst_tstbit_R0<"p1", 1, "t">;
}


let isBranch = 1, hasSideEffects = 0,
    isExtentSigned = 1, isPredicated = 1, isPredicatedNew = 1,
    isExtendable = 1, opExtentBits = 11, opExtentAlign = 2,
    opExtendable = 2, isTerminator = 1, validSubTargets = HasV4SubT in
class CJInst_RR<string px, string op, bit np, string tnt>
  : InstHexagon<(outs), (ins IntRegs:$Rs, IntRegs:$Rt, brtarget:$r9_2),
  ""#px#" = cmp."#op#"($Rs, $Rt); if ("
   #!if(np, "!","")#""#px#".new) jump:"#tnt#" $r9_2",
  [], "", COMPOUND, TypeCOMPOUND> {
  bits<4> Rs;
  bits<4> Rt;
  bits<11> r9_2;

  // np: !p[01]
  let isPredicatedFalse = np;
  // tnt: Taken/Not Taken
  let isBrTaken = !if (!eq(tnt, "t"), "true", "false");
  let isTaken   = !if (!eq(tnt, "t"), 1, 0);

  let IClass = 0b0001;
  let Inst{27-23} = !if (!eq(op, "eq"),  0b01000,
                    !if (!eq(op, "gt"),  0b01001,
                    !if (!eq(op, "gtu"), 0b01010, 0)));
  let Inst{22} = np;
  let Inst{21-20} = r9_2{10-9};
  let Inst{19-16} = Rs;
  let Inst{13} = !if (!eq(tnt, "t"), 1, 0);
  // px: Predicate reg 0/1
  let Inst{12} = !if (!eq(px, "!p1"), 1,
                 !if (!eq(px,  "p1"), 1, 0));
  let Inst{11-8} = Rt;
  let Inst{7-1} = r9_2{8-2};
}

// P[10] taken/not taken.
multiclass T_tnt_CJInst_RR<string op, bit np> {
  let Defs = [PC, P0], Uses = [P0] in {
    def NAME#p0_jump_nt : CJInst_RR<"p0", op, np, "nt">;
    def NAME#p0_jump_t : CJInst_RR<"p0", op, np, "t">;
  }
  let Defs = [PC, P1], Uses = [P1] in {
    def NAME#p1_jump_nt : CJInst_RR<"p1", op, np, "nt">;
    def NAME#p1_jump_t : CJInst_RR<"p1", op, np, "t">;
  }
}
// Predicate / !Predicate
multiclass T_pnp_CJInst_RR<string op>{
  defm J4_cmp#NAME#_t : T_tnt_CJInst_RR<op, 0>;
  defm J4_cmp#NAME#_f : T_tnt_CJInst_RR<op, 1>;
}
// TypeCJ Instructions compare RR and jump
let isCodeGenOnly = 0 in {
defm eq : T_pnp_CJInst_RR<"eq">;
defm gt : T_pnp_CJInst_RR<"gt">;
defm gtu : T_pnp_CJInst_RR<"gtu">;
}

let isBranch = 1, hasSideEffects = 0, isExtentSigned = 1,
    isPredicated = 1, isPredicatedNew = 1, isExtendable = 1, opExtentBits = 11,
    opExtentAlign = 2, opExtendable = 2, isTerminator = 1,
    validSubTargets = HasV4SubT in
class CJInst_RU5<string px, string op, bit np, string tnt>
  : InstHexagon<(outs), (ins IntRegs:$Rs, u5Imm:$U5, brtarget:$r9_2),
  ""#px#" = cmp."#op#"($Rs, #$U5); if ("
    #!if(np, "!","")#""#px#".new) jump:"#tnt#" $r9_2",
  [], "", COMPOUND, TypeCOMPOUND> {
  bits<4> Rs;
  bits<5> U5;
  bits<11> r9_2;

  // np: !p[01]
  let isPredicatedFalse = np;
  // tnt: Taken/Not Taken
  let isBrTaken = !if (!eq(tnt, "t"), "true", "false");
  let isTaken   = !if (!eq(tnt, "t"), 1, 0);

  let IClass = 0b0001;
  let Inst{27-26} = 0b00;
  // px: Predicate reg 0/1
  let Inst{25} = !if (!eq(px, "!p1"), 1,
                 !if (!eq(px,  "p1"), 1, 0));
  let Inst{24-23} = !if (!eq(op, "eq"),  0b00,
                    !if (!eq(op, "gt"),  0b01,
                    !if (!eq(op, "gtu"), 0b10, 0)));
  let Inst{22} = np;
  let Inst{21-20} = r9_2{10-9};
  let Inst{19-16} = Rs;
  let Inst{13} = !if (!eq(tnt, "t"), 1, 0);
  let Inst{12-8} = U5;
  let Inst{7-1} = r9_2{8-2};
}
// P[10] taken/not taken.
multiclass T_tnt_CJInst_RU5<string op, bit np> {
  let Defs = [PC, P0], Uses = [P0] in {
    def NAME#p0_jump_nt : CJInst_RU5<"p0", op, np, "nt">;
    def NAME#p0_jump_t : CJInst_RU5<"p0", op, np, "t">;
  }
  let Defs = [PC, P1], Uses = [P1] in {
    def NAME#p1_jump_nt : CJInst_RU5<"p1", op, np, "nt">;
    def NAME#p1_jump_t : CJInst_RU5<"p1", op, np, "t">;
  }
}
// Predicate / !Predicate
multiclass T_pnp_CJInst_RU5<string op>{
  defm J4_cmp#NAME#i_t : T_tnt_CJInst_RU5<op, 0>;
  defm J4_cmp#NAME#i_f : T_tnt_CJInst_RU5<op, 1>;
}
// TypeCJ Instructions compare RI and jump
let isCodeGenOnly = 0 in {
defm eq : T_pnp_CJInst_RU5<"eq">;
defm gt : T_pnp_CJInst_RU5<"gt">;
defm gtu : T_pnp_CJInst_RU5<"gtu">;
}

let isBranch = 1, hasSideEffects = 0, isExtentSigned = 1,
    isPredicated = 1, isPredicatedFalse = 1, isPredicatedNew = 1,
    isExtendable = 1, opExtentBits = 11, opExtentAlign = 2, opExtendable = 1,
    isTerminator = 1, validSubTargets = HasV4SubT in
class CJInst_Rn1<string px, string op, bit np, string tnt>
  : InstHexagon<(outs), (ins IntRegs:$Rs, brtarget:$r9_2),
  ""#px#" = cmp."#op#"($Rs,#-1); if ("
  #!if(np, "!","")#""#px#".new) jump:"#tnt#" $r9_2",
  [], "", COMPOUND, TypeCOMPOUND> {
  bits<4> Rs;
  bits<11> r9_2;

  // np: !p[01]
  let isPredicatedFalse = np;
  // tnt: Taken/Not Taken
  let isBrTaken = !if (!eq(tnt, "t"), "true", "false");
  let isTaken   = !if (!eq(tnt, "t"), 1, 0);

  let IClass = 0b0001;
  let Inst{27-26} = 0b00;
  let Inst{25} = !if (!eq(px, "!p1"), 1,
                 !if (!eq(px,  "p1"), 1, 0));

  let Inst{24-23} = 0b11;
  let Inst{22} = np;
  let Inst{21-20} = r9_2{10-9};
  let Inst{19-16} = Rs;
  let Inst{13} = !if (!eq(tnt, "t"), 1, 0);
  let Inst{9-8} = !if (!eq(op, "eq"),  0b00,
                  !if (!eq(op, "gt"),  0b01, 0));
  let Inst{7-1} = r9_2{8-2};
}

// P[10] taken/not taken.
multiclass T_tnt_CJInst_Rn1<string op, bit np> {
  let Defs = [PC, P0], Uses = [P0] in {
    def NAME#p0_jump_nt : CJInst_Rn1<"p0", op, np, "nt">;
    def NAME#p0_jump_t : CJInst_Rn1<"p0", op, np, "t">;
  }
  let Defs = [PC, P1], Uses = [P1] in {
    def NAME#p1_jump_nt : CJInst_Rn1<"p1", op, np, "nt">;
    def NAME#p1_jump_t : CJInst_Rn1<"p1", op, np, "t">;
  }
}
// Predicate / !Predicate
multiclass T_pnp_CJInst_Rn1<string op>{
  defm J4_cmp#NAME#n1_t : T_tnt_CJInst_Rn1<op, 0>;
  defm J4_cmp#NAME#n1_f : T_tnt_CJInst_Rn1<op, 1>;
}
// TypeCJ Instructions compare -1 and jump
let isCodeGenOnly = 0 in {
defm eq : T_pnp_CJInst_Rn1<"eq">;
defm gt : T_pnp_CJInst_Rn1<"gt">;
}

// J4_jumpseti: Direct unconditional jump and set register to immediate.
let Defs = [PC], isBranch = 1, hasSideEffects = 0, hasNewValue = 1,
    isExtentSigned = 1, opNewValue = 0, isExtendable = 1, opExtentBits = 11,
    opExtentAlign = 2, opExtendable = 2, validSubTargets = HasV4SubT,
    isCodeGenOnly = 0 in
def J4_jumpseti: CJInst <
  (outs IntRegs:$Rd),
  (ins u6Imm:$U6, brtarget:$r9_2),
  "$Rd = #$U6 ; jump $r9_2"> {
    bits<4> Rd;
    bits<6> U6;
    bits<11> r9_2;

    let IClass = 0b0001;
    let Inst{27-24} = 0b0110;
    let Inst{21-20} = r9_2{10-9};
    let Inst{19-16} = Rd;
    let Inst{13-8} = U6;
    let Inst{7-1} = r9_2{8-2};
  }

// J4_jumpsetr: Direct unconditional jump and transfer register.
let Defs = [PC], isBranch = 1, hasSideEffects = 0, hasNewValue = 1,
    isExtentSigned = 1, opNewValue = 0, isExtendable = 1, opExtentBits = 11,
    opExtentAlign = 2, opExtendable = 2, validSubTargets = HasV4SubT,
    isCodeGenOnly = 0 in
def J4_jumpsetr: CJInst <
  (outs IntRegs:$Rd),
  (ins IntRegs:$Rs, brtarget:$r9_2),
  "$Rd = $Rs ; jump $r9_2"> {
    bits<4> Rd;
    bits<4> Rs;
    bits<11> r9_2;

    let IClass = 0b0001;
    let Inst{27-24} = 0b0111;
    let Inst{21-20} = r9_2{10-9};
    let Inst{11-8} = Rd;
    let Inst{19-16} = Rs;
    let Inst{7-1} = r9_2{8-2};
  }
