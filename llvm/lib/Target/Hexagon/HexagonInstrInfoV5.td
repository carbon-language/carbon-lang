//=- HexagonInstrInfoV5.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V5 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/MPY
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 0 in
def S2_asr_i_p_rnd : S_2OpInstImm<"asr", 0b110, 0b111, u6Imm,
      [(set I64:$dst,
            (sra (i64 (add (i64 (sra I64:$src1, u6ImmPred:$src2)), 1)),
                 (i32 1)))], 1>,
      Requires<[HasV5T]> {
  bits<6> src2;
  let Inst{13-8} = src2;
}

let isCodeGenOnly = 0 in
def C4_fastcorner9 : T_LOGICAL_2OP<"fastcorner9", 0b000, 0, 0>,
  Requires<[HasV5T]> {
  let Inst{13,7,4} = 0b111;
}

let isCodeGenOnly = 0 in
def C4_fastcorner9_not : T_LOGICAL_2OP<"!fastcorner9", 0b000, 0, 0>,
  Requires<[HasV5T]> {
  let Inst{20,13,7,4} = 0b1111;
}

def SDTHexagonFCONST32 : SDTypeProfile<1, 1, [
                                            SDTCisVT<0, f32>,
                                            SDTCisPtrTy<1>]>;
def HexagonFCONST32 : SDNode<"HexagonISD::FCONST32",     SDTHexagonFCONST32>;

let isReMaterializable = 1, isMoveImm = 1 in
def FCONST32_nsdata : LDInst<(outs IntRegs:$dst), (ins globaladdress:$global),
              "$dst = CONST32(#$global)",
              [(set (f32 IntRegs:$dst),
              (HexagonFCONST32 tglobaladdr:$global))]>,
               Requires<[HasV5T]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST64_Float_Real : LDInst<(outs DoubleRegs:$dst), (ins f64imm:$src1),
                       "$dst = CONST64(#$src1)",
                       [(set DoubleRegs:$dst, fpimm:$src1)]>,
          Requires<[HasV5T]>;

let isReMaterializable = 1, isMoveImm = 1 in
def CONST32_Float_Real : LDInst<(outs IntRegs:$dst), (ins f32imm:$src1),
                       "$dst = CONST32(#$src1)",
                       [(set IntRegs:$dst, fpimm:$src1)]>,
          Requires<[HasV5T]>;

// Transfer immediate float.
// Only works with single precision fp value.
// For double precision, use CONST64_float_real, as 64bit transfer
// can only hold 40-bit values - 32 from const ext + 8 bit immediate.
// Make sure that complexity is more than the CONST32 pattern in
// HexagonInstrInfo.td patterns.
let isExtended = 1, opExtendable = 1, isMoveImm = 1, isReMaterializable = 1,
isPredicable = 1, AddedComplexity = 30, validSubTargets = HasV5SubT,
isCodeGenOnly = 1 in
def TFRI_f : ALU32_ri<(outs IntRegs:$dst), (ins f32Ext:$src1),
           "$dst = #$src1",
           [(set IntRegs:$dst, fpimm:$src1)]>,
          Requires<[HasV5T]>;

let isExtended = 1, opExtendable = 2, isPredicated = 1,
hasSideEffects = 0, validSubTargets = HasV5SubT in
def TFRI_cPt_f : ALU32_ri<(outs IntRegs:$dst),
                          (ins PredRegs:$src1, f32Ext:$src2),
           "if ($src1) $dst = #$src2",
           []>,
          Requires<[HasV5T]>;

let isExtended = 1, opExtendable = 2, isPredicated = 1, isPredicatedFalse = 1,
hasSideEffects = 0, validSubTargets = HasV5SubT in
def TFRI_cNotPt_f : ALU32_ri<(outs IntRegs:$dst),
                             (ins PredRegs:$src1, f32Ext:$src2),
           "if (!$src1) $dst =#$src2",
           []>,
          Requires<[HasV5T]>;

def SDTHexagonI32I64: SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                           SDTCisVT<1, i64>]>;

def HexagonPOPCOUNT: SDNode<"HexagonISD::POPCOUNT", SDTHexagonI32I64>;

let hasNewValue = 1, validSubTargets = HasV5SubT, isCodeGenOnly = 0 in
def S5_popcountp : ALU64_rr<(outs IntRegs:$Rd), (ins DoubleRegs:$Rss),
  "$Rd = popcount($Rss)",
  [(set I32:$Rd, (HexagonPOPCOUNT I64:$Rss))], "", S_2op_tc_2_SLOT23>,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rss;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1000011;
    let Inst{7-5} = 0b011;
    let Inst{4-0} = Rd;
    let Inst{20-16} = Rss;
  }

let isFP = 1, hasNewValue = 1, opNewValue = 0 in
class T_MInstFloat <string mnemonic, bits<3> MajOp, bits<3> MinOp>
  : MInst<(outs IntRegs:$Rd),
          (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd = "#mnemonic#"($Rs, $Rt)", [],
  "" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1011;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{13} = 0b0;
    let Inst{12-8} = Rt;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rd;
  }

let isCommutable = 1, isCodeGenOnly = 0 in {
  def F2_sfadd : T_MInstFloat < "sfadd", 0b000, 0b000>;
  def F2_sfmpy : T_MInstFloat < "sfmpy", 0b010, 0b000>;
}

let isCodeGenOnly = 0 in
def F2_sfsub : T_MInstFloat < "sfsub", 0b000, 0b001>;

let Itinerary = M_tc_3x_SLOT23, isCodeGenOnly = 0 in {
  def F2_sfmax : T_MInstFloat < "sfmax", 0b100, 0b000>;
  def F2_sfmin : T_MInstFloat < "sfmin", 0b100, 0b001>;
}

let isCodeGenOnly = 0 in {
def F2_sffixupn : T_MInstFloat < "sffixupn", 0b110, 0b000>;
def F2_sffixupd : T_MInstFloat < "sffixupd", 0b110, 0b001>;
}

// F2_sfrecipa: Reciprocal approximation for division.
let isPredicateLate = 1, isFP = 1,
hasSideEffects = 0, hasNewValue = 1, isCodeGenOnly = 0 in
def F2_sfrecipa: MInst <
  (outs IntRegs:$Rd, PredRegs:$Pe),
  (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd, $Pe = sfrecipa($Rs, $Rt)">,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<2> Pe;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;
    let Inst{27-21} = 0b1011111;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6-5}   = Pe;
    let Inst{4-0}   = Rd;
  }

// F2_dfcmpeq: Floating point compare for equal.
let isCompare = 1, isFP = 1 in
class T_fcmp <string mnemonic, RegisterClass RC, bits<3> MinOp,
              list<dag> pattern = [] >
  : ALU64Inst <(outs PredRegs:$dst), (ins RC:$src1, RC:$src2),
  "$dst = "#mnemonic#"($src1, $src2)", pattern,
  "" , ALU64_tc_2early_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<2> dst;
    bits<5> src1;
    bits<5> src2;

    let IClass = 0b1101;

    let Inst{27-21} = 0b0010111;
    let Inst{20-16} = src1;
    let Inst{12-8}  = src2;
    let Inst{7-5}   = MinOp;
    let Inst{1-0}   = dst;
  }

class T_fcmp64 <string mnemonic, PatFrag OpNode, bits<3> MinOp>
  : T_fcmp <mnemonic, DoubleRegs, MinOp,
  [(set  I1:$dst, (OpNode F64:$src1, F64:$src2))]> {
  let IClass = 0b1101;
  let Inst{27-21} = 0b0010111;
}

class T_fcmp32 <string mnemonic, PatFrag OpNode, bits<3> MinOp>
  : T_fcmp <mnemonic, IntRegs, MinOp,
  [(set  I1:$dst, (OpNode F32:$src1, F32:$src2))]> {
  let IClass = 0b1100;
  let Inst{27-21} = 0b0111111;
}

let isCodeGenOnly = 0 in {
def F2_dfcmpeq : T_fcmp64<"dfcmp.eq", setoeq, 0b000>;
def F2_dfcmpgt : T_fcmp64<"dfcmp.gt", setogt, 0b001>;
def F2_dfcmpge : T_fcmp64<"dfcmp.ge", setoge, 0b010>;
def F2_dfcmpuo : T_fcmp64<"dfcmp.uo", setuo,  0b011>;

def F2_sfcmpge : T_fcmp32<"sfcmp.ge", setoge, 0b000>;
def F2_sfcmpuo : T_fcmp32<"sfcmp.uo", setuo,  0b001>;
def F2_sfcmpeq : T_fcmp32<"sfcmp.eq", setoeq, 0b011>;
def F2_sfcmpgt : T_fcmp32<"sfcmp.gt", setogt, 0b100>;
}

// F2 convert template classes:
let isFP = 1 in
class F2_RDD_RSS_CONVERT<string mnemonic, bits<3> MinOp,
                         SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                         string chop ="">
  : SInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss),
   "$Rdd = "#mnemonic#"($Rss)"#chop,
   [(set RCOut:$Rdd, (Op RCIn:$Rss))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rdd;
     bits<5> Rss;

     let IClass = 0b1000;

     let Inst{27-21} = 0b0000111;
     let Inst{20-16} = Rss;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rdd;
  }

let isFP = 1 in
class F2_RDD_RS_CONVERT<string mnemonic, bits<3> MinOp,
                        SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                        string chop ="">
  : SInst <(outs DoubleRegs:$Rdd), (ins IntRegs:$Rs),
   "$Rdd = "#mnemonic#"($Rs)"#chop,
   [(set RCOut:$Rdd, (Op RCIn:$Rs))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rdd;
     bits<5> Rs;

     let IClass = 0b1000;

     let Inst{27-21} = 0b0100100;
     let Inst{20-16} = Rs;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rdd;
  }

let isFP = 1, hasNewValue = 1 in
class F2_RD_RSS_CONVERT<string mnemonic, bits<3> MinOp,
                        SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                        string chop ="">
  : SInst <(outs IntRegs:$Rd), (ins DoubleRegs:$Rss),
   "$Rd = "#mnemonic#"($Rss)"#chop,
   [(set RCOut:$Rd, (Op RCIn:$Rss))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rd;
     bits<5> Rss;

     let IClass = 0b1000;

     let Inst{27-24} = 0b1000;
     let Inst{23-21} = MinOp;
     let Inst{20-16} = Rss;
     let Inst{7-5} = 0b001;
     let Inst{4-0} = Rd;
  }

let isFP = 1, hasNewValue = 1 in
class F2_RD_RS_CONVERT<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                        SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                        string chop ="">
  : SInst <(outs IntRegs:$Rd), (ins IntRegs:$Rs),
   "$Rd = "#mnemonic#"($Rs)"#chop,
   [(set RCOut:$Rd, (Op RCIn:$Rs))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rd;
     bits<5> Rs;

     let IClass = 0b1000;

     let Inst{27-24} = 0b1011;
     let Inst{23-21} = MajOp;
     let Inst{20-16} = Rs;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rd;
  }

// Convert single precision to double precision and vice-versa.
let isCodeGenOnly = 0 in {
def F2_conv_sf2df : F2_RDD_RS_CONVERT <"convert_sf2df", 0b000,
                                       fextend, F64, F32>;

def F2_conv_df2sf : F2_RD_RSS_CONVERT <"convert_df2sf", 0b000,
                                       fround, F32, F64>;

// Convert Integer to Floating Point.
def F2_conv_d2sf : F2_RD_RSS_CONVERT <"convert_d2sf", 0b010,
                                       sint_to_fp, F32, I64>;
def F2_conv_ud2sf : F2_RD_RSS_CONVERT <"convert_ud2sf", 0b001,
                                       uint_to_fp, F32, I64>;
def F2_conv_uw2sf : F2_RD_RS_CONVERT <"convert_uw2sf", 0b001, 0b000,
                                       uint_to_fp, F32, I32>;
def F2_conv_w2sf : F2_RD_RS_CONVERT <"convert_w2sf", 0b010, 0b000,
                                       sint_to_fp, F32, I32>;
def F2_conv_d2df : F2_RDD_RSS_CONVERT <"convert_d2df", 0b011,
                                       sint_to_fp, F64, I64>;
def F2_conv_ud2df : F2_RDD_RSS_CONVERT <"convert_ud2df", 0b010,
                                        uint_to_fp, F64, I64>;
def F2_conv_uw2df : F2_RDD_RS_CONVERT <"convert_uw2df", 0b001,
                                       uint_to_fp, F64, I32>;
def F2_conv_w2df : F2_RDD_RS_CONVERT <"convert_w2df", 0b010,
                                       sint_to_fp, F64, I32>;

// Convert Floating Point to Integer - default.
def F2_conv_df2uw_chop : F2_RD_RSS_CONVERT <"convert_df2uw", 0b101,
                                            fp_to_uint, I32, F64, ":chop">;
def F2_conv_df2w_chop : F2_RD_RSS_CONVERT <"convert_df2w", 0b111,
                                            fp_to_sint, I32, F64, ":chop">;
def F2_conv_sf2uw_chop : F2_RD_RS_CONVERT <"convert_sf2uw", 0b011, 0b001,
                                       fp_to_uint, I32, F32, ":chop">;
def F2_conv_sf2w_chop : F2_RD_RS_CONVERT <"convert_sf2w", 0b100, 0b001,
                                       fp_to_sint, I32, F32, ":chop">;
def F2_conv_df2d_chop : F2_RDD_RSS_CONVERT <"convert_df2d", 0b110,
                                            fp_to_sint, I64, F64, ":chop">;
def F2_conv_df2ud_chop : F2_RDD_RSS_CONVERT <"convert_df2ud", 0b111,
                                             fp_to_uint, I64, F64, ":chop">;
def F2_conv_sf2d_chop : F2_RDD_RS_CONVERT <"convert_sf2d", 0b110,
                                       fp_to_sint, I64, F32, ":chop">;
def F2_conv_sf2ud_chop : F2_RDD_RS_CONVERT <"convert_sf2ud", 0b101,
                                            fp_to_uint, I64, F32, ":chop">;

// Convert Floating Point to Integer: non-chopped.
let AddedComplexity = 20, Predicates = [HasV5T, IEEERndNearV5T] in {
  def F2_conv_df2d : F2_RDD_RSS_CONVERT <"convert_df2d", 0b000,
                                         fp_to_sint, I64, F64>;
  def F2_conv_df2ud : F2_RDD_RSS_CONVERT <"convert_df2ud", 0b001,
                                          fp_to_uint, I64, F64>;
  def F2_conv_sf2ud : F2_RDD_RS_CONVERT <"convert_sf2ud", 0b011,
                                         fp_to_uint, I64, F32>;
  def F2_conv_sf2d : F2_RDD_RS_CONVERT <"convert_sf2d", 0b100,
                                         fp_to_sint, I64, F32>;
  def F2_conv_df2uw : F2_RD_RSS_CONVERT <"convert_df2uw", 0b011,
                                         fp_to_uint, I32, F64>;
  def F2_conv_df2w : F2_RD_RSS_CONVERT <"convert_df2w", 0b100,
                                         fp_to_sint, I32, F64>;
  def F2_conv_sf2uw : F2_RD_RS_CONVERT <"convert_sf2uw", 0b011, 0b000,
                                         fp_to_uint, I32, F32>;
  def F2_conv_sf2w : F2_RD_RS_CONVERT <"convert_sf2w", 0b100, 0b000,
                                         fp_to_sint, I32, F32>;
}
}

// Fix up radicand.
let isFP = 1, hasNewValue = 1, isCodeGenOnly = 0 in
def F2_sffixupr: SInst<(outs IntRegs:$Rd), (ins IntRegs:$Rs),
  "$Rd = sffixupr($Rs)",
  [], "" , S_2op_tc_3or4x_SLOT23>, Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rs;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = Rs;
    let Inst{7-5}   = 0b000;
    let Inst{4-0}   = Rd;
  }

// F2_sffma: Floating-point fused multiply add.
let isFP = 1, hasNewValue = 1 in
class T_sfmpy_acc <bit isSub, bit isLib>
  : MInst<(outs IntRegs:$Rx),
          (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt),
  "$Rx "#!if(isSub, "-=","+=")#" sfmpy($Rs, $Rt)"#!if(isLib, ":lib",""),
  [], "$dst2 = $Rx" , M_tc_3_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-21} = 0b1111000;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6}     = isLib;
    let Inst{5}     = isSub;
    let Inst{4-0}   = Rx;
  }

let isCodeGenOnly = 0 in {
def F2_sffma: T_sfmpy_acc <0, 0>;
def F2_sffms: T_sfmpy_acc <1, 0>;
def F2_sffma_lib: T_sfmpy_acc <0, 1>;
def F2_sffms_lib: T_sfmpy_acc <1, 1>;
}

// Floating-point fused multiply add w/ additional scaling (2**pu).
let isFP = 1, hasNewValue = 1, isCodeGenOnly = 0 in
def F2_sffma_sc: MInst <
  (outs IntRegs:$Rx),
  (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt, PredRegs:$Pu),
  "$Rx += sfmpy($Rs, $Rt, $Pu):scale" ,
  [], "$dst2 = $Rx" , M_tc_3_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;
    bits<2> Pu;

    let IClass = 0b1110;

    let Inst{27-21} = 0b1111011;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6-5}   = Pu;
    let Inst{4-0}   = Rx;
  }

// Classify floating-point value
let isFP = 1, isCodeGenOnly = 0 in
 def F2_sfclass : T_TEST_BIT_IMM<"sfclass", 0b111>;

let isFP = 1, isCodeGenOnly = 0 in
def F2_dfclass: ALU64Inst<(outs PredRegs:$Pd), (ins DoubleRegs:$Rss, u5Imm:$u5),
  "$Pd = dfclass($Rss, #$u5)",
  [], "" , ALU64_tc_2early_SLOT23 > , Requires<[HasV5T]> {
    bits<2> Pd;
    bits<5> Rss;
    bits<5> u5;

    let IClass = 0b1101;
    let Inst{27-21} = 0b1100100;
    let Inst{20-16} = Rss;
    let Inst{12-10} = 0b000;
    let Inst{9-5}   = u5;
    let Inst{4-3}   = 0b10;
    let Inst{1-0}   = Pd;
  }

// Instructions to create floating point constant
let hasNewValue = 1, opNewValue = 0 in
class T_fimm <string mnemonic, RegisterClass RC, bits<4> RegType, bit isNeg>
  : ALU64Inst<(outs RC:$dst), (ins u10Imm:$src),
  "$dst = "#mnemonic#"(#$src)"#!if(isNeg, ":neg", ":pos"),
  [], "", ALU64_tc_3x_SLOT23>, Requires<[HasV5T]> {
    bits<5> dst;
    bits<10> src;

    let IClass = 0b1101;
    let Inst{27-24} = RegType;
    let Inst{23}    = 0b0;
    let Inst{22}    = isNeg;
    let Inst{21}    = src{9};
    let Inst{13-5}  = src{8-0};
    let Inst{4-0}   = dst;
  }

let isCodeGenOnly = 0 in {
def F2_sfimm_p : T_fimm <"sfmake", IntRegs, 0b0110, 0>;
def F2_sfimm_n : T_fimm <"sfmake", IntRegs, 0b0110, 1>;
def F2_dfimm_p : T_fimm <"dfmake", DoubleRegs, 0b1001, 0>;
def F2_dfimm_n : T_fimm <"dfmake", DoubleRegs, 0b1001, 1>;
}

// Convert single precision to double precision and vice-versa.
def CONVERT_sf2df : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
                "$dst = convert_sf2df($src)",
                [(set DoubleRegs:$dst, (fextend IntRegs:$src))]>,
          Requires<[HasV5T]>;

def CONVERT_df2sf : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
                "$dst = convert_df2sf($src)",
                [(set IntRegs:$dst, (fround DoubleRegs:$src))]>,
          Requires<[HasV5T]>;


// Load.
def LDrid_f : LDInst<(outs DoubleRegs:$dst),
            (ins MEMri:$addr),
            "$dst = memd($addr)",
            [(set DoubleRegs:$dst, (f64 (load ADDRriS11_3:$addr)))]>,
          Requires<[HasV5T]>;


let AddedComplexity = 20 in
def LDrid_indexed_f : LDInst<(outs DoubleRegs:$dst),
            (ins IntRegs:$src1, s11_3Imm:$offset),
            "$dst = memd($src1+#$offset)",
            [(set DoubleRegs:$dst, (f64 (load (add IntRegs:$src1,
                                              s11_3ImmPred:$offset))))]>,
          Requires<[HasV5T]>;

def LDriw_f : LDInst<(outs IntRegs:$dst),
            (ins MEMri:$addr), "$dst = memw($addr)",
            [(set IntRegs:$dst, (f32 (load ADDRriS11_2:$addr)))]>,
          Requires<[HasV5T]>;


let AddedComplexity = 20 in
def LDriw_indexed_f : LDInst<(outs IntRegs:$dst),
            (ins IntRegs:$src1, s11_2Imm:$offset),
            "$dst = memw($src1+#$offset)",
            [(set IntRegs:$dst, (f32 (load (add IntRegs:$src1,
                                           s11_2ImmPred:$offset))))]>,
          Requires<[HasV5T]>;

// Store.
def STriw_f : STInst<(outs),
            (ins MEMri:$addr, IntRegs:$src1),
            "memw($addr) = $src1",
            [(store (f32 IntRegs:$src1), ADDRriS11_2:$addr)]>,
          Requires<[HasV5T]>;

let AddedComplexity = 10 in
def STriw_indexed_f : STInst<(outs),
            (ins IntRegs:$src1, s11_2Imm:$src2, IntRegs:$src3),
            "memw($src1+#$src2) = $src3",
            [(store (f32 IntRegs:$src3),
                (add IntRegs:$src1, s11_2ImmPred:$src2))]>,
          Requires<[HasV5T]>;

def STrid_f : STInst<(outs),
            (ins MEMri:$addr, DoubleRegs:$src1),
            "memd($addr) = $src1",
            [(store (f64 DoubleRegs:$src1), ADDRriS11_2:$addr)]>,
          Requires<[HasV5T]>;

// Indexed store double word.
let AddedComplexity = 10 in
def STrid_indexed_f : STInst<(outs),
            (ins IntRegs:$src1, s11_3Imm:$src2,  DoubleRegs:$src3),
            "memd($src1+#$src2) = $src3",
            [(store (f64 DoubleRegs:$src3),
                                (add IntRegs:$src1, s11_3ImmPred:$src2))]>,
          Requires<[HasV5T]>;


// Add
let isCommutable = 1 in
def fADD_rr : ALU64_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sfadd($src1, $src2)",
            [(set IntRegs:$dst, (fadd IntRegs:$src1, IntRegs:$src2))]>,
          Requires<[HasV5T]>;

let isCommutable = 1 in
def fADD64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = dfadd($src1, $src2)",
               [(set DoubleRegs:$dst, (fadd DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>,
          Requires<[HasV5T]>;

def fSUB_rr : ALU64_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sfsub($src1, $src2)",
            [(set IntRegs:$dst, (fsub IntRegs:$src1, IntRegs:$src2))]>,
          Requires<[HasV5T]>;

def fSUB64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = dfsub($src1, $src2)",
               [(set DoubleRegs:$dst, (fsub DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>,
               Requires<[HasV5T]>;

let isCommutable = 1 in
def fMUL_rr : ALU64_rr<(outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2),
            "$dst = sfmpy($src1, $src2)",
            [(set IntRegs:$dst, (fmul IntRegs:$src1, IntRegs:$src2))]>,
            Requires<[HasV5T]>;

let isCommutable = 1 in
def fMUL64_rr : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                     DoubleRegs:$src2),
               "$dst = dfmpy($src1, $src2)",
               [(set DoubleRegs:$dst, (fmul DoubleRegs:$src1,
                                           DoubleRegs:$src2))]>,
               Requires<[HasV5T]>;

// Compare.
let isCompare = 1 in {
multiclass FCMP64_rr<string OpcStr, PatFrag OpNode> {
  def _rr : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$b, DoubleRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst,
                        (OpNode (f64 DoubleRegs:$b), (f64 DoubleRegs:$c)))]>,
                 Requires<[HasV5T]>;
}

multiclass FCMP32_rr<string OpcStr, PatFrag OpNode> {
  def _rr : ALU64_rr<(outs PredRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat("$dst = ", !strconcat(OpcStr, "($b, $c)")),
                 [(set PredRegs:$dst,
                        (OpNode (f32 IntRegs:$b), (f32 IntRegs:$c)))]>,
                 Requires<[HasV5T]>;
}
}

defm FCMPOEQ64 : FCMP64_rr<"dfcmp.eq", setoeq>;
defm FCMPUEQ64 : FCMP64_rr<"dfcmp.eq", setueq>;
defm FCMPOGT64 : FCMP64_rr<"dfcmp.gt", setogt>;
defm FCMPUGT64 : FCMP64_rr<"dfcmp.gt", setugt>;
defm FCMPOGE64 : FCMP64_rr<"dfcmp.ge", setoge>;
defm FCMPUGE64 : FCMP64_rr<"dfcmp.ge", setuge>;

defm FCMPOEQ32 : FCMP32_rr<"sfcmp.eq", setoeq>;
defm FCMPUEQ32 : FCMP32_rr<"sfcmp.eq", setueq>;
defm FCMPOGT32 : FCMP32_rr<"sfcmp.gt", setogt>;
defm FCMPUGT32 : FCMP32_rr<"sfcmp.gt", setugt>;
defm FCMPOGE32 : FCMP32_rr<"sfcmp.ge", setoge>;
defm FCMPUGE32 : FCMP32_rr<"sfcmp.ge", setuge>;

// olt.
def : Pat <(i1 (setolt (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPOGT32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setolt (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGT32_rr (f32 (TFRI_f fpimm:$src2)), (f32 IntRegs:$src1)))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setolt (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPOGT64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setolt (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGT64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                        (f64 DoubleRegs:$src1)))>,
      Requires<[HasV5T]>;

// gt.
def : Pat <(i1 (setugt (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT64_rr (f64 DoubleRegs:$src1),
                        (f64 (CONST64_Float_Real fpimm:$src2))))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setugt (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT32_rr (f32 IntRegs:$src1), (f32 (TFRI_f fpimm:$src2))))>,
      Requires<[HasV5T]>;

// ult.
def : Pat <(i1 (setult (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPUGT32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setult (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT32_rr (f32 (TFRI_f fpimm:$src2)), (f32 IntRegs:$src1)))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setult (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPUGT64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat <(i1 (setult (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGT64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                        (f64 DoubleRegs:$src1)))>,
      Requires<[HasV5T]>;

// le.
// rs <= rt -> rt >= rs.
def : Pat<(i1 (setole (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPOGE32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setole (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGE32_rr (f32 (TFRI_f fpimm:$src2)), IntRegs:$src1))>,
      Requires<[HasV5T]>;


// Rss <= Rtt -> Rtt >= Rss.
def : Pat<(i1 (setole (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPOGE64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setole (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPOGE64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                                DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

// rs <= rt -> rt >= rs.
def : Pat<(i1 (setule (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (FCMPUGE32_rr IntRegs:$src2, IntRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setule (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGE32_rr (f32 (TFRI_f fpimm:$src2)), IntRegs:$src1))>,
      Requires<[HasV5T]>;

// Rss <= Rtt -> Rtt >= Rss.
def : Pat<(i1 (setule (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (FCMPUGE64_rr DoubleRegs:$src2, DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setule (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (FCMPUGE64_rr (f64 (CONST64_Float_Real fpimm:$src2)),
                                DoubleRegs:$src1))>,
      Requires<[HasV5T]>;

// ne.
def : Pat<(i1 (setone (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (C2_not (FCMPOEQ32_rr IntRegs:$src1, IntRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setone (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (C2_not (FCMPOEQ64_rr DoubleRegs:$src1, DoubleRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
      (i1 (C2_not (FCMPUEQ32_rr IntRegs:$src1, IntRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
      (i1 (C2_not (FCMPUEQ64_rr DoubleRegs:$src1, DoubleRegs:$src2)))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setone (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (C2_not (FCMPOEQ32_rr IntRegs:$src1, (f32 (TFRI_f fpimm:$src2)))))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setone (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (C2_not (FCMPOEQ64_rr DoubleRegs:$src1,
                              (f64 (CONST64_Float_Real fpimm:$src2)))))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f32 IntRegs:$src1), (fpimm:$src2))),
      (i1 (C2_not (FCMPUEQ32_rr IntRegs:$src1,  (f32 (TFRI_f fpimm:$src2)))))>,
      Requires<[HasV5T]>;

def : Pat<(i1 (setune (f64 DoubleRegs:$src1), (fpimm:$src2))),
      (i1 (C2_not (FCMPUEQ64_rr DoubleRegs:$src1,
                              (f64 (CONST64_Float_Real fpimm:$src2)))))>,
      Requires<[HasV5T]>;

// Convert Integer to Floating Point.
def CONVERT_d2sf : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_d2sf($src)",
              [(set (f32 IntRegs:$dst), (sint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_ud2sf : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_ud2sf($src)",
              [(set (f32 IntRegs:$dst), (uint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_uw2sf : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_uw2sf($src)",
              [(set (f32 IntRegs:$dst), (uint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_w2sf : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_w2sf($src)",
              [(set (f32 IntRegs:$dst), (sint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_d2df : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_d2df($src)",
              [(set (f64 DoubleRegs:$dst), (sint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_ud2df : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_ud2df($src)",
              [(set (f64 DoubleRegs:$dst), (uint_to_fp (i64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_uw2df : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_uw2df($src)",
              [(set (f64 DoubleRegs:$dst), (uint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_w2df : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_w2df($src)",
              [(set (f64 DoubleRegs:$dst), (sint_to_fp (i32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

// Convert Floating Point to Integer - default.
def CONVERT_df2uw : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2uw($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_df2w : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2w($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2uw : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2uw($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2w : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2w($src):chop",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_df2d : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2d($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_df2ud : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2ud($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2d : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2d($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

def CONVERT_sf2ud : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2ud($src):chop",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T]>;

// Convert Floating Point to Integer: non-chopped.
let AddedComplexity = 20 in
def CONVERT_df2uw_nchop : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2uw($src)",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_df2w_nchop : ALU64_rr<(outs IntRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2w($src)",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2uw_nchop : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2uw($src)",
              [(set (i32 IntRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2w_nchop : ALU64_rr<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2w($src)",
              [(set (i32 IntRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_df2d_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2d($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_df2ud_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src),
              "$dst = convert_df2ud($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f64 DoubleRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2d_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2d($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_sint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;

let AddedComplexity = 20 in
def CONVERT_sf2ud_nchop : ALU64_rr<(outs DoubleRegs:$dst), (ins IntRegs:$src),
              "$dst = convert_sf2ud($src)",
              [(set (i64 DoubleRegs:$dst), (fp_to_uint (f32 IntRegs:$src)))]>,
              Requires<[HasV5T, IEEERndNearV5T]>;



// Bitcast is different than [fp|sint|uint]_to_[sint|uint|fp].
def : Pat <(i32 (bitconvert (f32 IntRegs:$src))),
           (i32 (A2_tfr IntRegs:$src))>,
          Requires<[HasV5T]>;

def : Pat <(f32 (bitconvert (i32 IntRegs:$src))),
           (f32 (A2_tfr IntRegs:$src))>,
          Requires<[HasV5T]>;

def : Pat <(i64 (bitconvert (f64 DoubleRegs:$src))),
           (i64 (A2_tfrp DoubleRegs:$src))>,
          Requires<[HasV5T]>;

def : Pat <(f64 (bitconvert (i64 DoubleRegs:$src))),
           (f64 (A2_tfrp DoubleRegs:$src))>,
          Requires<[HasV5T]>;

def FMADD_sp : ALU64_acc<(outs IntRegs:$dst),
                  (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
              "$dst += sfmpy($src2, $src3)",
              [(set (f32 IntRegs:$dst),
                  (fma IntRegs:$src2, IntRegs:$src3, IntRegs:$src1))],
                  "$src1 = $dst">,
              Requires<[HasV5T]>;


// Floating point max/min.

let AddedComplexity = 100 in
def FMAX_sp : ALU64_rr<(outs IntRegs:$dst),
                  (ins IntRegs:$src1, IntRegs:$src2),
              "$dst = sfmax($src1, $src2)",
              [(set IntRegs:$dst, (f32 (select (i1 (setolt IntRegs:$src2,
                                                        IntRegs:$src1)),
                                             IntRegs:$src1,
                                             IntRegs:$src2)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100 in
def FMIN_sp : ALU64_rr<(outs IntRegs:$dst),
                  (ins IntRegs:$src1, IntRegs:$src2),
              "$dst = sfmin($src1, $src2)",
              [(set IntRegs:$dst, (f32 (select (i1 (setogt IntRegs:$src2,
                                                        IntRegs:$src1)),
                                             IntRegs:$src1,
                                             IntRegs:$src2)))]>,
               Requires<[HasV5T]>;

// Pseudo instruction to encode a set of conditional transfers.
// This instruction is used instead of a mux and trades-off codesize
// for performance. We conduct this transformation optimistically in
// the hope that these instructions get promoted to dot-new transfers.
let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_rr_f : ALU32_rr<(outs IntRegs:$dst), (ins PredRegs:$src1,
                                                        IntRegs:$src2,
                                                        IntRegs:$src3),
                     "Error; should not emit",
                     [(set IntRegs:$dst, (f32 (select PredRegs:$src1,
                                                 IntRegs:$src2,
                                                 IntRegs:$src3)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_rr64_f : ALU32_rr<(outs DoubleRegs:$dst), (ins PredRegs:$src1,
                                                        DoubleRegs:$src2,
                                                        DoubleRegs:$src3),
                     "Error; should not emit",
                     [(set DoubleRegs:$dst, (f64 (select PredRegs:$src1,
                                                 DoubleRegs:$src2,
                                                 DoubleRegs:$src3)))]>,
               Requires<[HasV5T]>;



let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_ri_f : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, IntRegs:$src2, f32imm:$src3),
            "Error; should not emit",
            [(set IntRegs:$dst,
             (f32 (select PredRegs:$src1, IntRegs:$src2, fpimm:$src3)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_ir_f : ALU32_rr<(outs IntRegs:$dst),
            (ins PredRegs:$src1, f32imm:$src2, IntRegs:$src3),
            "Error; should not emit",
            [(set IntRegs:$dst,
             (f32 (select PredRegs:$src1, fpimm:$src2, IntRegs:$src3)))]>,
               Requires<[HasV5T]>;

let AddedComplexity = 100, isPredicated = 1 in
def TFR_condset_ii_f : ALU32_rr<(outs IntRegs:$dst),
                              (ins PredRegs:$src1, f32imm:$src2, f32imm:$src3),
                     "Error; should not emit",
                     [(set IntRegs:$dst, (f32 (select PredRegs:$src1,
                                                 fpimm:$src2,
                                                 fpimm:$src3)))]>,
               Requires<[HasV5T]>;


def : Pat <(select (i1 (setult (f32 IntRegs:$src1), (f32 IntRegs:$src2))),
                   (f32 IntRegs:$src3),
                   (f32 IntRegs:$src4)),
    (TFR_condset_rr_f (FCMPUGT32_rr IntRegs:$src2, IntRegs:$src1), IntRegs:$src4,
                      IntRegs:$src3)>, Requires<[HasV5T]>;

def : Pat <(select (i1 (setult (f64 DoubleRegs:$src1), (f64 DoubleRegs:$src2))),
                   (f64 DoubleRegs:$src3),
                   (f64 DoubleRegs:$src4)),
      (TFR_condset_rr64_f (FCMPUGT64_rr DoubleRegs:$src2, DoubleRegs:$src1),
                DoubleRegs:$src4, DoubleRegs:$src3)>, Requires<[HasV5T]>;

// Map from p0 = pnot(p0); r0 = mux(p0, #i, #j) => r0 = mux(p0, #j, #i).
def : Pat <(select (not PredRegs:$src1), fpimm:$src2, fpimm:$src3),
      (TFR_condset_ii_f PredRegs:$src1, fpimm:$src3, fpimm:$src2)>;

// Map from p0 = pnot(p0); r0 = select(p0, #i, r1)
// => r0 = TFR_condset_ri(p0, r1, #i)
def : Pat <(select (not PredRegs:$src1), fpimm:$src2, IntRegs:$src3),
      (TFR_condset_ri_f PredRegs:$src1, IntRegs:$src3, fpimm:$src2)>;

// Map from p0 = pnot(p0); r0 = mux(p0, r1, #i)
// => r0 = TFR_condset_ir(p0, #i, r1)
def : Pat <(select (not PredRegs:$src1), IntRegs:$src2, fpimm:$src3),
      (TFR_condset_ir_f PredRegs:$src1, fpimm:$src3, IntRegs:$src2)>;

def : Pat <(i32 (fp_to_sint (f64 DoubleRegs:$src1))),
          (i32 (EXTRACT_SUBREG (i64 (CONVERT_df2d (f64 DoubleRegs:$src1))), subreg_loreg))>,
          Requires<[HasV5T]>;

def : Pat <(fabs (f32 IntRegs:$src1)),
           (S2_clrbit_i (f32 IntRegs:$src1), 31)>,
          Requires<[HasV5T]>;

def : Pat <(fneg (f32 IntRegs:$src1)),
           (S2_togglebit_i (f32 IntRegs:$src1), 31)>,
          Requires<[HasV5T]>;

/*
def : Pat <(fabs (f64 DoubleRegs:$src1)),
          (S2_clrbit_i (f32 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg)), 31)>,
          Requires<[HasV5T]>;

def : Pat <(fabs (f64 DoubleRegs:$src1)),
          (S2_clrbit_i (f32 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg)), 31)>,
          Requires<[HasV5T]>;
          */
