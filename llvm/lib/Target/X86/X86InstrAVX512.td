//===-- X86InstrAVX512.td - AVX512 Instruction Set ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 AVX512 instruction set, defining the
// instructions, and properties of the instructions which are needed for code
// generation, machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

// Group template arguments that can be derived from the vector type (EltNum x
// EltVT).  These are things like the register class for the writemask, etc.
// The idea is to pass one of these as the template argument rather than the
// individual arguments.
// The template is also used for scalar types, in this case numelts is 1.
class X86VectorVTInfo<int numelts, ValueType eltvt, RegisterClass rc,
                      string suffix = ""> {
  RegisterClass RC = rc;
  ValueType EltVT = eltvt;
  int NumElts = numelts;

  // Corresponding mask register class.
  RegisterClass KRC = !cast<RegisterClass>("VK" # NumElts);

  // Corresponding write-mask register class.
  RegisterClass KRCWM = !cast<RegisterClass>("VK" # NumElts # "WM");

  // The GPR register class that can hold the write mask.  Use GR8 for fewer
  // than 8 elements.  Use shift-right and equal to work around the lack of
  // !lt in tablegen.
  RegisterClass MRC =
    !cast<RegisterClass>("GR" #
                         !if (!eq (!srl(NumElts, 3), 0), 8, NumElts));

  // Suffix used in the instruction mnemonic.
  string Suffix = suffix;

  // VTName is a string name for vector VT. For vector types it will be
  // v # NumElts # EltVT, so for vector of 8 elements of i32 it will be v8i32
  // It is a little bit complex for scalar types, where NumElts = 1.
  // In this case we build v4f32 or v2f64
  string VTName = "v" # !if (!eq (NumElts, 1),
                        !if (!eq (EltVT.Size, 32), 4,
                        !if (!eq (EltVT.Size, 64), 2, NumElts)), NumElts) # EltVT;

  // The vector VT.
  ValueType VT = !cast<ValueType>(VTName);

  string EltTypeName = !cast<string>(EltVT);
  // Size of the element type in bits, e.g. 32 for v16i32.
  string EltSizeName = !subst("i", "", !subst("f", "", EltTypeName));
  int EltSize = EltVT.Size;

  // "i" for integer types and "f" for floating-point types
  string TypeVariantName = !subst(EltSizeName, "", EltTypeName);

  // Size of RC in bits, e.g. 512 for VR512.
  int Size = VT.Size;

  // The corresponding memory operand, e.g. i512mem for VR512.
  X86MemOperand MemOp = !cast<X86MemOperand>(TypeVariantName # Size # "mem");
  X86MemOperand ScalarMemOp = !cast<X86MemOperand>(EltVT # "mem");

  // Load patterns
  // Note: For 128/256-bit integer VT we choose loadv2i64/loadv4i64
  //       due to load promotion during legalization
  PatFrag LdFrag = !cast<PatFrag>("load" #
                                  !if (!eq (TypeVariantName, "i"),
                                       !if (!eq (Size, 128), "v2i64",
                                       !if (!eq (Size, 256), "v4i64",
                                            VTName)), VTName));

  PatFrag AlignedLdFrag = !cast<PatFrag>("alignedload" #
                          !if (!eq (TypeVariantName, "i"),
                                !if (!eq (Size, 128), "v2i64",
                                !if (!eq (Size, 256), "v4i64",
                                !if (!eq (Size, 512), 
                                    !if (!eq (EltSize, 64), "v8i64", "v16i32"),
                                    VTName))), VTName));

  PatFrag ScalarLdFrag = !cast<PatFrag>("load" # EltVT);

  // The corresponding float type, e.g. v16f32 for v16i32
  // Note: For EltSize < 32, FloatVT is illegal and TableGen
  //       fails to compile, so we choose FloatVT = VT
  ValueType FloatVT = !cast<ValueType>(
                        !if (!eq (!srl(EltSize,5),0),
                             VTName,
                             !if (!eq(TypeVariantName, "i"),
                                  "v" # NumElts # "f" # EltSize,
                                  VTName)));

  // The string to specify embedded broadcast in assembly.
  string BroadcastStr = "{1to" # NumElts # "}";

  // 8-bit compressed displacement tuple/subvector format.  This is only
  // defined for NumElts <= 8.
  CD8VForm CD8TupleForm = !if (!eq (!srl(NumElts, 4), 0),
                               !cast<CD8VForm>("CD8VT" # NumElts), ?);

  SubRegIndex SubRegIdx = !if (!eq (Size, 128), sub_xmm,
                          !if (!eq (Size, 256), sub_ymm, ?));

  Domain ExeDomain = !if (!eq (EltTypeName, "f32"), SSEPackedSingle,
                     !if (!eq (EltTypeName, "f64"), SSEPackedDouble,
                     SSEPackedInt));

  RegisterClass FRC = !if (!eq (EltTypeName, "f32"), FR32X, FR64X);

  // A vector type of the same width with element type i32.  This is used to
  // create the canonical constant zero node ImmAllZerosV.
  ValueType i32VT = !cast<ValueType>("v" # !srl(Size, 5) # "i32");
  dag ImmAllZerosV = (VT (bitconvert (i32VT immAllZerosV)));

  string ZSuffix = !if (!eq (Size, 128), "Z128",
                   !if (!eq (Size, 256), "Z256", "Z"));
}

def v64i8_info  : X86VectorVTInfo<64,  i8, VR512, "b">;
def v32i16_info : X86VectorVTInfo<32, i16, VR512, "w">;
def v16i32_info : X86VectorVTInfo<16, i32, VR512, "d">;
def v8i64_info  : X86VectorVTInfo<8,  i64, VR512, "q">;
def v16f32_info : X86VectorVTInfo<16, f32, VR512, "ps">;
def v8f64_info  : X86VectorVTInfo<8,  f64, VR512, "pd">;

// "x" in v32i8x_info means RC = VR256X
def v32i8x_info  : X86VectorVTInfo<32,  i8, VR256X, "b">;
def v16i16x_info : X86VectorVTInfo<16, i16, VR256X, "w">;
def v8i32x_info  : X86VectorVTInfo<8,  i32, VR256X, "d">;
def v4i64x_info  : X86VectorVTInfo<4,  i64, VR256X, "q">;
def v8f32x_info  : X86VectorVTInfo<8,  f32, VR256X, "ps">;
def v4f64x_info  : X86VectorVTInfo<4,  f64, VR256X, "pd">;

def v16i8x_info  : X86VectorVTInfo<16,  i8, VR128X, "b">;
def v8i16x_info  : X86VectorVTInfo<8,  i16, VR128X, "w">;
def v4i32x_info  : X86VectorVTInfo<4,  i32, VR128X, "d">;
def v2i64x_info  : X86VectorVTInfo<2,  i64, VR128X, "q">;
def v4f32x_info  : X86VectorVTInfo<4,  f32, VR128X, "ps">;
def v2f64x_info  : X86VectorVTInfo<2,  f64, VR128X, "pd">;

// We map scalar types to the smallest (128-bit) vector type
// with the appropriate element type. This allows to use the same masking logic.
def f32x_info    : X86VectorVTInfo<1,  f32, VR128X, "ss">;
def f64x_info    : X86VectorVTInfo<1,  f64, VR128X, "sd">;

class AVX512VLVectorVTInfo<X86VectorVTInfo i512, X86VectorVTInfo i256,
                           X86VectorVTInfo i128> {
  X86VectorVTInfo info512 = i512;
  X86VectorVTInfo info256 = i256;
  X86VectorVTInfo info128 = i128;
}

def avx512vl_i8_info  : AVX512VLVectorVTInfo<v64i8_info, v32i8x_info,
                                             v16i8x_info>;
def avx512vl_i16_info : AVX512VLVectorVTInfo<v32i16_info, v16i16x_info,
                                             v8i16x_info>;
def avx512vl_i32_info : AVX512VLVectorVTInfo<v16i32_info, v8i32x_info,
                                             v4i32x_info>;
def avx512vl_i64_info : AVX512VLVectorVTInfo<v8i64_info, v4i64x_info,
                                             v2i64x_info>;
def avx512vl_f32_info : AVX512VLVectorVTInfo<v16f32_info, v8f32x_info,
                                             v4f32x_info>;
def avx512vl_f64_info : AVX512VLVectorVTInfo<v8f64_info, v4f64x_info,
                                             v2f64x_info>;

// This multiclass generates the masking variants from the non-masking
// variant.  It only provides the assembly pieces for the masking variants.
// It assumes custom ISel patterns for masking which can be provided as
// template arguments.
multiclass AVX512_maskable_custom<bits<8> O, Format F,
                                  dag Outs,
                                  dag Ins, dag MaskingIns, dag ZeroMaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  list<dag> Pattern,
                                  list<dag> MaskingPattern,
                                  list<dag> ZeroMaskingPattern,
                                  string Round = "",
                                  string MaskingConstraint = "",
                                  InstrItinClass itin = NoItinerary,
                                  bit IsCommutable = 0> {
  let isCommutable = IsCommutable in
    def NAME: AVX512<O, F, Outs, Ins,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst "#Round#"|"#
                                     "$dst "#Round#", "#IntelSrcAsm#"}",
                       Pattern, itin>;

  // Prefer over VMOV*rrk Pat<>
  let AddedComplexity = 20 in
    def NAME#k: AVX512<O, F, Outs, MaskingIns,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst {${mask}}"#Round#"|"#
                                     "$dst {${mask}}"#Round#", "#IntelSrcAsm#"}",
                       MaskingPattern, itin>,
              EVEX_K {
      // In case of the 3src subclass this is overridden with a let.
      string Constraints = MaskingConstraint;
  }
  let AddedComplexity = 30 in // Prefer over VMOV*rrkz Pat<>
    def NAME#kz: AVX512<O, F, Outs, ZeroMaskingIns,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst {${mask}} {z}"#Round#"|"#
                                     "$dst {${mask}} {z}"#Round#", "#IntelSrcAsm#"}",
                       ZeroMaskingPattern,
                       itin>,
              EVEX_KZ;
}


// Common base class of AVX512_maskable and AVX512_maskable_3src.
multiclass AVX512_maskable_common<bits<8> O, Format F, X86VectorVTInfo _,
                                  dag Outs,
                                  dag Ins, dag MaskingIns, dag ZeroMaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  dag RHS, dag MaskingRHS,
                                  SDNode Select = vselect, string Round = "",
                                  string MaskingConstraint = "",
                                  InstrItinClass itin = NoItinerary,
                                  bit IsCommutable = 0> :
  AVX512_maskable_custom<O, F, Outs, Ins, MaskingIns, ZeroMaskingIns, OpcodeStr,
                         AttSrcAsm, IntelSrcAsm,
                         [(set _.RC:$dst, RHS)],
                         [(set _.RC:$dst, MaskingRHS)],
                         [(set _.RC:$dst,
                               (Select _.KRCWM:$mask, RHS, _.ImmAllZerosV))],
                         Round, MaskingConstraint, NoItinerary, IsCommutable>;

// This multiclass generates the unconditional/non-masking, the masking and
// the zero-masking variant of the vector instruction.  In the masking case, the
// perserved vector elements come from a new dummy input operand tied to $dst.
multiclass AVX512_maskable<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm,
                           dag RHS, string Round = "",
                           InstrItinClass itin = NoItinerary,
                           bit IsCommutable = 0> :
   AVX512_maskable_common<O, F, _, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (vselect _.KRCWM:$mask, RHS, _.RC:$src0), vselect,
                          Round, "$src0 = $dst", itin, IsCommutable>;

// This multiclass generates the unconditional/non-masking, the masking and
// the zero-masking variant of the scalar instruction.
multiclass AVX512_maskable_scalar<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm,
                           dag RHS, string Round = "",
                           InstrItinClass itin = NoItinerary,
                           bit IsCommutable = 0> :
   AVX512_maskable_common<O, F, _, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (X86select _.KRCWM:$mask, RHS, _.RC:$src0), X86select,
                          Round, "$src0 = $dst", itin, IsCommutable>;

// Similar to AVX512_maskable but in this case one of the source operands
// ($src1) is already tied to $dst so we just use that for the preserved
// vector elements.  NOTE that the NonTiedIns (the ins dag) should exclude
// $src1.
multiclass AVX512_maskable_3src<bits<8> O, Format F, X86VectorVTInfo _,
                                dag Outs, dag NonTiedIns, string OpcodeStr,
                                string AttSrcAsm, string IntelSrcAsm,
                                dag RHS> :
   AVX512_maskable_common<O, F, _, Outs,
                          !con((ins _.RC:$src1), NonTiedIns),
                          !con((ins _.RC:$src1, _.KRCWM:$mask), NonTiedIns),
                          !con((ins _.RC:$src1, _.KRCWM:$mask), NonTiedIns),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (vselect _.KRCWM:$mask, RHS, _.RC:$src1)>;


multiclass AVX512_maskable_in_asm<bits<8> O, Format F, X86VectorVTInfo _,
                                  dag Outs, dag Ins,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  list<dag> Pattern> :
   AVX512_maskable_custom<O, F, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, Pattern, [], [], "",
                          "$src0 = $dst">;

// Bitcasts between 512-bit vector types. Return the original type since
// no instruction is needed for the conversion
let Predicates = [HasAVX512] in {
  def : Pat<(v8f64  (bitconvert (v8i64 VR512:$src))),  (v8f64 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v16i32 VR512:$src))), (v8f64 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v32i16 VR512:$src))),  (v8f64 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v64i8 VR512:$src))), (v8f64 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v16f32 VR512:$src))), (v8f64 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v8i64 VR512:$src))),  (v16f32 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v16i32 VR512:$src))), (v16f32 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v32i16 VR512:$src))), (v16f32 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v64i8 VR512:$src))), (v16f32 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v8f64 VR512:$src))),  (v16f32 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v16i32 VR512:$src))), (v8i64 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v32i16 VR512:$src))), (v8i64 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v64i8 VR512:$src))), (v8i64 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v8f64 VR512:$src))),  (v8i64 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v16f32 VR512:$src))), (v8i64 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v8i64 VR512:$src))), (v16i32 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v16f32 VR512:$src))), (v16i32 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v32i16 VR512:$src))),  (v16i32 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v64i8 VR512:$src))),  (v16i32 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v8f64 VR512:$src))),  (v16i32 VR512:$src)>;
  def : Pat<(v32i16 (bitconvert (v8i64 VR512:$src))), (v32i16 VR512:$src)>;
  def : Pat<(v32i16 (bitconvert (v16i32 VR512:$src))),  (v32i16 VR512:$src)>;
  def : Pat<(v32i16 (bitconvert (v64i8 VR512:$src))),  (v32i16 VR512:$src)>;
  def : Pat<(v32i16 (bitconvert (v8f64 VR512:$src))),  (v32i16 VR512:$src)>;
  def : Pat<(v32i16 (bitconvert (v16f32 VR512:$src))), (v32i16 VR512:$src)>;
  def : Pat<(v32i16 (bitconvert (v16f32 VR512:$src))), (v32i16 VR512:$src)>;
  def : Pat<(v64i8  (bitconvert (v8i64 VR512:$src))), (v64i8 VR512:$src)>;
  def : Pat<(v64i8  (bitconvert (v16i32 VR512:$src))), (v64i8 VR512:$src)>;
  def : Pat<(v64i8  (bitconvert (v32i16 VR512:$src))), (v64i8 VR512:$src)>;
  def : Pat<(v64i8  (bitconvert (v8f64 VR512:$src))),  (v64i8 VR512:$src)>;
  def : Pat<(v64i8  (bitconvert (v16f32 VR512:$src))), (v64i8 VR512:$src)>;

  def : Pat<(v2i64 (bitconvert (v4i32 VR128X:$src))), (v2i64 VR128X:$src)>;
  def : Pat<(v2i64 (bitconvert (v8i16 VR128X:$src))), (v2i64 VR128X:$src)>;
  def : Pat<(v2i64 (bitconvert (v16i8 VR128X:$src))), (v2i64 VR128X:$src)>;
  def : Pat<(v2i64 (bitconvert (v2f64 VR128X:$src))), (v2i64 VR128X:$src)>;
  def : Pat<(v2i64 (bitconvert (v4f32 VR128X:$src))), (v2i64 VR128X:$src)>;
  def : Pat<(v4i32 (bitconvert (v2i64 VR128X:$src))), (v4i32 VR128X:$src)>;
  def : Pat<(v4i32 (bitconvert (v8i16 VR128X:$src))), (v4i32 VR128X:$src)>;
  def : Pat<(v4i32 (bitconvert (v16i8 VR128X:$src))), (v4i32 VR128X:$src)>;
  def : Pat<(v4i32 (bitconvert (v2f64 VR128X:$src))), (v4i32 VR128X:$src)>;
  def : Pat<(v4i32 (bitconvert (v4f32 VR128X:$src))), (v4i32 VR128X:$src)>;
  def : Pat<(v8i16 (bitconvert (v2i64 VR128X:$src))), (v8i16 VR128X:$src)>;
  def : Pat<(v8i16 (bitconvert (v4i32 VR128X:$src))), (v8i16 VR128X:$src)>;
  def : Pat<(v8i16 (bitconvert (v16i8 VR128X:$src))), (v8i16 VR128X:$src)>;
  def : Pat<(v8i16 (bitconvert (v2f64 VR128X:$src))), (v8i16 VR128X:$src)>;
  def : Pat<(v8i16 (bitconvert (v4f32 VR128X:$src))), (v8i16 VR128X:$src)>;
  def : Pat<(v16i8 (bitconvert (v2i64 VR128X:$src))), (v16i8 VR128X:$src)>;
  def : Pat<(v16i8 (bitconvert (v4i32 VR128X:$src))), (v16i8 VR128X:$src)>;
  def : Pat<(v16i8 (bitconvert (v8i16 VR128X:$src))), (v16i8 VR128X:$src)>;
  def : Pat<(v16i8 (bitconvert (v2f64 VR128X:$src))), (v16i8 VR128X:$src)>;
  def : Pat<(v16i8 (bitconvert (v4f32 VR128X:$src))), (v16i8 VR128X:$src)>;
  def : Pat<(v4f32 (bitconvert (v2i64 VR128X:$src))), (v4f32 VR128X:$src)>;
  def : Pat<(v4f32 (bitconvert (v4i32 VR128X:$src))), (v4f32 VR128X:$src)>;
  def : Pat<(v4f32 (bitconvert (v8i16 VR128X:$src))), (v4f32 VR128X:$src)>;
  def : Pat<(v4f32 (bitconvert (v16i8 VR128X:$src))), (v4f32 VR128X:$src)>;
  def : Pat<(v4f32 (bitconvert (v2f64 VR128X:$src))), (v4f32 VR128X:$src)>;
  def : Pat<(v2f64 (bitconvert (v2i64 VR128X:$src))), (v2f64 VR128X:$src)>;
  def : Pat<(v2f64 (bitconvert (v4i32 VR128X:$src))), (v2f64 VR128X:$src)>;
  def : Pat<(v2f64 (bitconvert (v8i16 VR128X:$src))), (v2f64 VR128X:$src)>;
  def : Pat<(v2f64 (bitconvert (v16i8 VR128X:$src))), (v2f64 VR128X:$src)>;
  def : Pat<(v2f64 (bitconvert (v4f32 VR128X:$src))), (v2f64 VR128X:$src)>;

// Bitcasts between 256-bit vector types. Return the original type since
// no instruction is needed for the conversion
  def : Pat<(v4f64  (bitconvert (v8f32 VR256X:$src))),  (v4f64 VR256X:$src)>;
  def : Pat<(v4f64  (bitconvert (v8i32 VR256X:$src))),  (v4f64 VR256X:$src)>;
  def : Pat<(v4f64  (bitconvert (v4i64 VR256X:$src))),  (v4f64 VR256X:$src)>;
  def : Pat<(v4f64  (bitconvert (v16i16 VR256X:$src))), (v4f64 VR256X:$src)>;
  def : Pat<(v4f64  (bitconvert (v32i8 VR256X:$src))),  (v4f64 VR256X:$src)>;
  def : Pat<(v8f32  (bitconvert (v8i32 VR256X:$src))),  (v8f32 VR256X:$src)>;
  def : Pat<(v8f32  (bitconvert (v4i64 VR256X:$src))),  (v8f32 VR256X:$src)>;
  def : Pat<(v8f32  (bitconvert (v4f64 VR256X:$src))),  (v8f32 VR256X:$src)>;
  def : Pat<(v8f32  (bitconvert (v32i8 VR256X:$src))),  (v8f32 VR256X:$src)>;
  def : Pat<(v8f32  (bitconvert (v16i16 VR256X:$src))), (v8f32 VR256X:$src)>;
  def : Pat<(v4i64  (bitconvert (v8f32 VR256X:$src))),  (v4i64 VR256X:$src)>;
  def : Pat<(v4i64  (bitconvert (v8i32 VR256X:$src))),  (v4i64 VR256X:$src)>;
  def : Pat<(v4i64  (bitconvert (v4f64 VR256X:$src))),  (v4i64 VR256X:$src)>;
  def : Pat<(v4i64  (bitconvert (v32i8 VR256X:$src))),  (v4i64 VR256X:$src)>;
  def : Pat<(v4i64  (bitconvert (v16i16 VR256X:$src))), (v4i64 VR256X:$src)>;
  def : Pat<(v32i8  (bitconvert (v4f64 VR256X:$src))),  (v32i8 VR256X:$src)>;
  def : Pat<(v32i8  (bitconvert (v4i64 VR256X:$src))),  (v32i8 VR256X:$src)>;
  def : Pat<(v32i8  (bitconvert (v8f32 VR256X:$src))),  (v32i8 VR256X:$src)>;
  def : Pat<(v32i8  (bitconvert (v8i32 VR256X:$src))),  (v32i8 VR256X:$src)>;
  def : Pat<(v32i8  (bitconvert (v16i16 VR256X:$src))), (v32i8 VR256X:$src)>;
  def : Pat<(v8i32  (bitconvert (v32i8 VR256X:$src))),  (v8i32 VR256X:$src)>;
  def : Pat<(v8i32  (bitconvert (v16i16 VR256X:$src))), (v8i32 VR256X:$src)>;
  def : Pat<(v8i32  (bitconvert (v8f32 VR256X:$src))),  (v8i32 VR256X:$src)>;
  def : Pat<(v8i32  (bitconvert (v4i64 VR256X:$src))),  (v8i32 VR256X:$src)>;
  def : Pat<(v8i32  (bitconvert (v4f64 VR256X:$src))),  (v8i32 VR256X:$src)>;
  def : Pat<(v16i16 (bitconvert (v8f32 VR256X:$src))),  (v16i16 VR256X:$src)>;
  def : Pat<(v16i16 (bitconvert (v8i32 VR256X:$src))),  (v16i16 VR256X:$src)>;
  def : Pat<(v16i16 (bitconvert (v4i64 VR256X:$src))),  (v16i16 VR256X:$src)>;
  def : Pat<(v16i16 (bitconvert (v4f64 VR256X:$src))),  (v16i16 VR256X:$src)>;
  def : Pat<(v16i16 (bitconvert (v32i8 VR256X:$src))),  (v16i16 VR256X:$src)>;
}

//
// AVX-512: VPXOR instruction writes zero to its upper part, it's safe build zeros.
//

let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, Predicates = [HasAVX512] in {
def AVX512_512_SET0 : I<0, Pseudo, (outs VR512:$dst), (ins), "",
               [(set VR512:$dst, (v16f32 immAllZerosV))]>;
}

let Predicates = [HasAVX512] in {
def : Pat<(v8i64 immAllZerosV), (AVX512_512_SET0)>;
def : Pat<(v16i32 immAllZerosV), (AVX512_512_SET0)>;
def : Pat<(v8f64 immAllZerosV), (AVX512_512_SET0)>;
}

//===----------------------------------------------------------------------===//
// AVX-512 - VECTOR INSERT
//

multiclass vinsert_for_size_no_alt<int Opcode,
                                   X86VectorVTInfo From, X86VectorVTInfo To,
                                   PatFrag vinsert_insert,
                                   SDNodeXForm INSERT_get_vinsert_imm> {
  let hasSideEffects = 0, ExeDomain = To.ExeDomain in {
    def rr : AVX512AIi8<Opcode, MRMSrcReg, (outs VR512:$dst),
               (ins VR512:$src1, From.RC:$src2, u8imm:$src3),
               "vinsert" # From.EltTypeName # "x" # From.NumElts #
                                                "\t{$src3, $src2, $src1, $dst|"
                                                   "$dst, $src1, $src2, $src3}",
               [(set To.RC:$dst, (vinsert_insert:$src3 (To.VT VR512:$src1),
                                                       (From.VT From.RC:$src2),
                                                       (iPTR imm)))]>,
             EVEX_4V, EVEX_V512;

    let mayLoad = 1 in
    def rm : AVX512AIi8<Opcode, MRMSrcMem, (outs VR512:$dst),
               (ins VR512:$src1, From.MemOp:$src2, u8imm:$src3),
               "vinsert" # From.EltTypeName # "x" # From.NumElts #
                                                "\t{$src3, $src2, $src1, $dst|"
                                                   "$dst, $src1, $src2, $src3}",
               []>,
             EVEX_4V, EVEX_V512, EVEX_CD8<From.EltSize, From.CD8TupleForm>;
  }
}

multiclass vinsert_for_size<int Opcode,
                            X86VectorVTInfo From, X86VectorVTInfo To,
                            X86VectorVTInfo AltFrom, X86VectorVTInfo AltTo,
                            PatFrag vinsert_insert,
                            SDNodeXForm INSERT_get_vinsert_imm> :
  vinsert_for_size_no_alt<Opcode, From, To,
                          vinsert_insert, INSERT_get_vinsert_imm> {
  // Codegen pattern with the alternative types, e.g. v2i64 -> v8i64 for
  // vinserti32x4.  Only add this if 64x2 and friends are not supported
  // natively via AVX512DQ.
  let Predicates = [NoDQI] in
    def : Pat<(vinsert_insert:$ins
                 (AltTo.VT VR512:$src1), (AltFrom.VT From.RC:$src2), (iPTR imm)),
              (AltTo.VT (!cast<Instruction>(NAME # From.EltSize # "x4rr")
                            VR512:$src1, From.RC:$src2,
                            (INSERT_get_vinsert_imm VR512:$ins)))>;
}

multiclass vinsert_for_type<ValueType EltVT32, int Opcode128,
                            ValueType EltVT64, int Opcode256> {
  defm NAME # "32x4" : vinsert_for_size<Opcode128,
                                 X86VectorVTInfo< 4, EltVT32, VR128X>,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 X86VectorVTInfo< 2, EltVT64, VR128X>,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 vinsert128_insert,
                                 INSERT_get_vinsert128_imm>;
  let Predicates = [HasDQI] in
    defm NAME # "64x2" : vinsert_for_size_no_alt<Opcode128,
                                 X86VectorVTInfo< 2, EltVT64, VR128X>,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 vinsert128_insert,
                                 INSERT_get_vinsert128_imm>, VEX_W;
  defm NAME # "64x4" : vinsert_for_size<Opcode256,
                                 X86VectorVTInfo< 4, EltVT64, VR256X>,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 X86VectorVTInfo< 8, EltVT32, VR256>,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 vinsert256_insert,
                                 INSERT_get_vinsert256_imm>, VEX_W;
  let Predicates = [HasDQI] in
    defm NAME # "32x8" : vinsert_for_size_no_alt<Opcode256,
                                 X86VectorVTInfo< 8, EltVT32, VR256X>,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 vinsert256_insert,
                                 INSERT_get_vinsert256_imm>;
}

defm VINSERTF : vinsert_for_type<f32, 0x18, f64, 0x1a>;
defm VINSERTI : vinsert_for_type<i32, 0x38, i64, 0x3a>;

// vinsertps - insert f32 to XMM
def VINSERTPSzrr : AVX512AIi8<0x21, MRMSrcReg, (outs VR128X:$dst),
      (ins VR128X:$src1, VR128X:$src2, u8imm:$src3),
      "vinsertps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insertps VR128X:$src1, VR128X:$src2, imm:$src3))]>,
      EVEX_4V;
def VINSERTPSzrm: AVX512AIi8<0x21, MRMSrcMem, (outs VR128X:$dst),
      (ins VR128X:$src1, f32mem:$src2, u8imm:$src3),
      "vinsertps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insertps VR128X:$src1,
                          (v4f32 (scalar_to_vector (loadf32 addr:$src2))),
                          imm:$src3))]>, EVEX_4V, EVEX_CD8<32, CD8VT1>;

//===----------------------------------------------------------------------===//
// AVX-512 VECTOR EXTRACT
//---

multiclass vextract_for_size<int Opcode,
                             X86VectorVTInfo From, X86VectorVTInfo To,
                             X86VectorVTInfo AltFrom, X86VectorVTInfo AltTo,
                             PatFrag vextract_extract,
                             SDNodeXForm EXTRACT_get_vextract_imm> {
  let hasSideEffects = 0, ExeDomain = To.ExeDomain in {
    defm rr : AVX512_maskable_in_asm<Opcode, MRMDestReg, To, (outs To.RC:$dst),
                (ins VR512:$src1, u8imm:$idx),
                "vextract" # To.EltTypeName # "x4",
                "$idx, $src1", "$src1, $idx",
                [(set To.RC:$dst, (vextract_extract:$idx (From.VT VR512:$src1),
                                                         (iPTR imm)))]>,
              AVX512AIi8Base, EVEX, EVEX_V512;
    let mayStore = 1 in
    def rm : AVX512AIi8<Opcode, MRMDestMem, (outs),
            (ins To.MemOp:$dst, VR512:$src1, u8imm:$src2),
            "vextract" # To.EltTypeName # "x4\t{$src2, $src1, $dst|"
                                               "$dst, $src1, $src2}",
            []>, EVEX, EVEX_V512, EVEX_CD8<To.EltSize, CD8VT4>;
  }

  // Codegen pattern with the alternative types, e.g. v8i64 -> v2i64 for
  // vextracti32x4
  def : Pat<(vextract_extract:$ext (AltFrom.VT VR512:$src1), (iPTR imm)),
            (AltTo.VT (!cast<Instruction>(NAME # To.EltSize # "x4rr")
                          VR512:$src1,
                          (EXTRACT_get_vextract_imm To.RC:$ext)))>;

  // A 128/256-bit subvector extract from the first 512-bit vector position is
  // a subregister copy that needs no instruction.
  def : Pat<(To.VT (extract_subvector (From.VT VR512:$src), (iPTR 0))),
            (To.VT
               (EXTRACT_SUBREG (From.VT VR512:$src), To.SubRegIdx))>;

  // And for the alternative types.
  def : Pat<(AltTo.VT (extract_subvector (AltFrom.VT VR512:$src), (iPTR 0))),
            (AltTo.VT
               (EXTRACT_SUBREG (AltFrom.VT VR512:$src), AltTo.SubRegIdx))>;

  // Intrinsic call with masking.
  def : Pat<(!cast<Intrinsic>("int_x86_avx512_mask_vextract" # To.EltTypeName #
                              "x4_512")
                VR512:$src1, (iPTR imm:$idx), To.RC:$src0, GR8:$mask),
            (!cast<Instruction>(NAME # To.EltSize # "x4rrk") To.RC:$src0,
                (v4i1 (COPY_TO_REGCLASS GR8:$mask, VK4WM)),
                VR512:$src1, imm:$idx)>;

  // Intrinsic call with zero-masking.
  def : Pat<(!cast<Intrinsic>("int_x86_avx512_mask_vextract" # To.EltTypeName #
                              "x4_512")
                VR512:$src1, (iPTR imm:$idx), To.ImmAllZerosV, GR8:$mask),
            (!cast<Instruction>(NAME # To.EltSize # "x4rrkz")
                (v4i1 (COPY_TO_REGCLASS GR8:$mask, VK4WM)),
                VR512:$src1, imm:$idx)>;

  // Intrinsic call without masking.
  def : Pat<(!cast<Intrinsic>("int_x86_avx512_mask_vextract" # To.EltTypeName #
                              "x4_512")
                VR512:$src1, (iPTR imm:$idx), To.ImmAllZerosV, (i8 -1)),
            (!cast<Instruction>(NAME # To.EltSize # "x4rr")
                VR512:$src1, imm:$idx)>;
}

multiclass vextract_for_type<ValueType EltVT32, int Opcode32,
                             ValueType EltVT64, int Opcode64> {
  defm NAME # "32x4" : vextract_for_size<Opcode32,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 X86VectorVTInfo< 4, EltVT32, VR128X>,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 X86VectorVTInfo< 2, EltVT64, VR128X>,
                                 vextract128_extract,
                                 EXTRACT_get_vextract128_imm>;
  defm NAME # "64x4" : vextract_for_size<Opcode64,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 X86VectorVTInfo< 4, EltVT64, VR256X>,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 X86VectorVTInfo< 8, EltVT32, VR256>,
                                 vextract256_extract,
                                 EXTRACT_get_vextract256_imm>, VEX_W;
}

defm VEXTRACTF : vextract_for_type<f32, 0x19, f64, 0x1b>;
defm VEXTRACTI : vextract_for_type<i32, 0x39, i64, 0x3b>;

// A 128-bit subvector insert to the first 512-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(insert_subvector undef, (v2i64 VR128X:$src), (iPTR 0)),
          (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)),
          (INSERT_SUBREG (v4i64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
          sub_ymm)>;
def : Pat<(insert_subvector undef, (v2f64 VR128X:$src), (iPTR 0)),
          (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)),
          (INSERT_SUBREG (v4f64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
          sub_ymm)>;
def : Pat<(insert_subvector undef, (v4i32 VR128X:$src), (iPTR 0)),
          (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)),
          (INSERT_SUBREG (v8i32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
          sub_ymm)>;
def : Pat<(insert_subvector undef, (v4f32 VR128X:$src), (iPTR 0)),
          (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)),
          (INSERT_SUBREG (v8f32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
          sub_ymm)>;

def : Pat<(insert_subvector undef, (v4i64 VR256X:$src), (iPTR 0)),
          (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(insert_subvector undef, (v4f64 VR256X:$src), (iPTR 0)),
          (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(insert_subvector undef, (v8i32 VR256X:$src), (iPTR 0)),
          (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(insert_subvector undef, (v8f32 VR256X:$src), (iPTR 0)),
          (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;

// vextractps - extract 32 bits from XMM
def VEXTRACTPSzrr : AVX512AIi8<0x17, MRMDestReg, (outs GR32:$dst),
      (ins VR128X:$src1, u8imm:$src2),
      "vextractps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set GR32:$dst, (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2))]>,
      EVEX;

def VEXTRACTPSzmr : AVX512AIi8<0x17, MRMDestMem, (outs),
      (ins f32mem:$dst, VR128X:$src1, u8imm:$src2),
      "vextractps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(store (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2),
                          addr:$dst)]>, EVEX, EVEX_CD8<32, CD8VT1>;

//===---------------------------------------------------------------------===//
// AVX-512 BROADCAST
//---
multiclass avx512_fp_broadcast<bits<8> opc, SDNode OpNode, RegisterClass SrcRC,
                              ValueType svt, X86VectorVTInfo _> {
  defm r : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                   (ins SrcRC:$src), "vbroadcast"## !subst("p", "s", _.Suffix),
                   "$src", "$src", (_.VT (OpNode (svt SrcRC:$src)))>,
                   T8PD, EVEX;

  let mayLoad = 1 in {
    defm m : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                     (ins _.ScalarMemOp:$src),
                     "vbroadcast"##!subst("p", "s", _.Suffix), "$src", "$src",
                     (_.VT (OpNode (_.ScalarLdFrag addr:$src)))>,
                     T8PD, EVEX;
  }
}

multiclass avx512_fp_broadcast_vl<bits<8> opc, SDNode OpNode,
                                  AVX512VLVectorVTInfo _> {
  defm Z  : avx512_fp_broadcast<opc, OpNode, VR128X, _.info128.VT, _.info512>,
                             EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256  : avx512_fp_broadcast<opc, OpNode, VR128X, _.info128.VT, _.info256>,
                                  EVEX_V256;
  }
}

let ExeDomain = SSEPackedSingle in {
  defm VBROADCASTSS  : avx512_fp_broadcast_vl<0x18, X86VBroadcast,
                              avx512vl_f32_info>, EVEX_CD8<32, CD8VT1>;
   let Predicates = [HasVLX] in {
     defm VBROADCASTSSZ128  : avx512_fp_broadcast<0x18, X86VBroadcast, VR128X,
                                     v4f32, v4f32x_info>, EVEX_V128,
                                     EVEX_CD8<32, CD8VT1>;
   }
}

let ExeDomain = SSEPackedDouble in {
  defm VBROADCASTSD  : avx512_fp_broadcast_vl<0x19, X86VBroadcast,
                              avx512vl_f64_info>, VEX_W, EVEX_CD8<64, CD8VT1>;
}

// avx512_broadcast_pat introduces patterns for broadcast with a scalar argument.
// Later, we can canonize broadcast instructions before ISel phase and 
// eliminate additional patterns on ISel.
// SrcRC_v and SrcRC_s are RegisterClasses for vector and scalar
// representations of source
multiclass avx512_broadcast_pat<string InstName, SDNode OpNode,
                                X86VectorVTInfo _, RegisterClass SrcRC_v,
                                RegisterClass SrcRC_s> {
  def : Pat<(_.VT (OpNode  (_.EltVT SrcRC_s:$src))),
            (!cast<Instruction>(InstName##"r")
              (COPY_TO_REGCLASS SrcRC_s:$src, SrcRC_v))>;

  let AddedComplexity = 30 in {
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                (OpNode (_.EltVT SrcRC_s:$src)), _.RC:$src0)),
              (!cast<Instruction>(InstName##"rk") _.RC:$src0, _.KRCWM:$mask,
                (COPY_TO_REGCLASS SrcRC_s:$src, SrcRC_v))>;

    def : Pat<(_.VT(vselect _.KRCWM:$mask,
                (OpNode (_.EltVT SrcRC_s:$src)), _.ImmAllZerosV)),
              (!cast<Instruction>(InstName##"rkz") _.KRCWM:$mask,
                (COPY_TO_REGCLASS SrcRC_s:$src, SrcRC_v))>;
  }
}

defm : avx512_broadcast_pat<"VBROADCASTSSZ", X86VBroadcast, v16f32_info,
                            VR128X, FR32X>;
defm : avx512_broadcast_pat<"VBROADCASTSDZ", X86VBroadcast, v8f64_info,
                            VR128X, FR64X>;

let Predicates = [HasVLX] in {
  defm : avx512_broadcast_pat<"VBROADCASTSSZ256", X86VBroadcast,
                              v8f32x_info, VR128X, FR32X>;
  defm : avx512_broadcast_pat<"VBROADCASTSSZ128", X86VBroadcast,
                              v4f32x_info, VR128X, FR32X>;
  defm : avx512_broadcast_pat<"VBROADCASTSDZ256", X86VBroadcast,
                              v4f64x_info, VR128X, FR64X>;
}

def : Pat<(v16f32 (X86VBroadcast (loadf32 addr:$src))),
          (VBROADCASTSSZm addr:$src)>;
def : Pat<(v8f64 (X86VBroadcast (loadf64 addr:$src))),
          (VBROADCASTSDZm addr:$src)>;

def : Pat<(int_x86_avx512_vbroadcast_ss_512 addr:$src),
          (VBROADCASTSSZm addr:$src)>;
def : Pat<(int_x86_avx512_vbroadcast_sd_512 addr:$src),
          (VBROADCASTSDZm addr:$src)>;

multiclass avx512_int_broadcast_reg<bits<8> opc, X86VectorVTInfo _,
                                    RegisterClass SrcRC> {
  defm r : AVX512_maskable_in_asm<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins SrcRC:$src),  "vpbroadcast"##_.Suffix,
                           "$src", "$src", []>, T8PD, EVEX;
}

multiclass avx512_int_broadcast_reg_vl<bits<8> opc, AVX512VLVectorVTInfo _,
                                       RegisterClass SrcRC, Predicate prd> {
  let Predicates = [prd] in
    defm Z : avx512_int_broadcast_reg<opc, _.info512, SrcRC>, EVEX_V512;
  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_int_broadcast_reg<opc, _.info256, SrcRC>, EVEX_V256;
    defm Z128 : avx512_int_broadcast_reg<opc, _.info128, SrcRC>, EVEX_V128;
  }
}

defm VPBROADCASTBr : avx512_int_broadcast_reg_vl<0x7A, avx512vl_i8_info, GR32,
                                                 HasBWI>;
defm VPBROADCASTWr : avx512_int_broadcast_reg_vl<0x7B, avx512vl_i16_info, GR32,
                                                 HasBWI>;
defm VPBROADCASTDr : avx512_int_broadcast_reg_vl<0x7C, avx512vl_i32_info, GR32,
                                                 HasAVX512>;
defm VPBROADCASTQr : avx512_int_broadcast_reg_vl<0x7C, avx512vl_i64_info, GR64,
                                                 HasAVX512>, VEX_W;

def : Pat <(v16i32 (X86vzext VK16WM:$mask)),
           (VPBROADCASTDrZrkz VK16WM:$mask, (i32 (MOV32ri 0x1)))>;

def : Pat <(v8i64 (X86vzext VK8WM:$mask)),
           (VPBROADCASTQrZrkz VK8WM:$mask, (i64 (MOV64ri 0x1)))>;

def : Pat<(v16i32 (X86VBroadcast (i32 GR32:$src))),
        (VPBROADCASTDrZr GR32:$src)>;
def : Pat<(v16i32 (X86VBroadcastm VK16WM:$mask, (i32 GR32:$src))),
        (VPBROADCASTDrZrkz VK16WM:$mask, GR32:$src)>;
def : Pat<(v8i64 (X86VBroadcast (i64 GR64:$src))),
        (VPBROADCASTQrZr GR64:$src)>;
def : Pat<(v8i64 (X86VBroadcastm VK8WM:$mask, (i64 GR64:$src))),
        (VPBROADCASTQrZrkz VK8WM:$mask, GR64:$src)>;

def : Pat<(v16i32 (int_x86_avx512_pbroadcastd_i32_512 (i32 GR32:$src))),
        (VPBROADCASTDrZr GR32:$src)>;
def : Pat<(v8i64 (int_x86_avx512_pbroadcastq_i64_512 (i64 GR64:$src))),
        (VPBROADCASTQrZr GR64:$src)>;

def : Pat<(v16i32 (int_x86_avx512_mask_pbroadcast_d_gpr_512 (i32 GR32:$src),
                   (v16i32 immAllZerosV), (i16 GR16:$mask))),
          (VPBROADCASTDrZrkz (COPY_TO_REGCLASS GR16:$mask, VK16WM), GR32:$src)>;
def : Pat<(v8i64 (int_x86_avx512_mask_pbroadcast_q_gpr_512 (i64 GR64:$src),
                   (bc_v8i64 (v16i32 immAllZerosV)), (i8 GR8:$mask))),
          (VPBROADCASTQrZrkz (COPY_TO_REGCLASS GR8:$mask, VK8WM), GR64:$src)>;

multiclass avx512_int_broadcast_rm<bits<8> opc, string OpcodeStr,
                          X86MemOperand x86memop, PatFrag ld_frag,
                          RegisterClass DstRC, ValueType OpVT, ValueType SrcVT,
                          RegisterClass KRC> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst), (ins VR128X:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set DstRC:$dst,
                    (OpVT (X86VBroadcast (SrcVT VR128X:$src))))]>, EVEX;
  def krr : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst), (ins KRC:$mask,
                                                         VR128X:$src),
                    !strconcat(OpcodeStr,
                    "\t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
                    [(set DstRC:$dst,
                      (OpVT (X86VBroadcastm KRC:$mask, (SrcVT VR128X:$src))))]>,
                    EVEX, EVEX_KZ;
  let mayLoad = 1 in {
  def rm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set DstRC:$dst,
                    (OpVT (X86VBroadcast (ld_frag addr:$src))))]>, EVEX;
  def krm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst), (ins KRC:$mask,
                                                         x86memop:$src),
                  !strconcat(OpcodeStr,
                      "\t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
                  [(set DstRC:$dst, (OpVT (X86VBroadcastm KRC:$mask,
                                     (ld_frag addr:$src))))]>, EVEX, EVEX_KZ;
  }
}

defm VPBROADCASTDZ  : avx512_int_broadcast_rm<0x58, "vpbroadcastd", i32mem,
                      loadi32, VR512, v16i32, v4i32, VK16WM>,
                      EVEX_V512, EVEX_CD8<32, CD8VT1>;
defm VPBROADCASTQZ  : avx512_int_broadcast_rm<0x59, "vpbroadcastq", i64mem,
                      loadi64, VR512, v8i64, v2i64, VK8WM>,  EVEX_V512, VEX_W,
                      EVEX_CD8<64, CD8VT1>;

multiclass avx512_int_subvec_broadcast_rm<bits<8> opc, string OpcodeStr,
                          X86MemOperand x86memop, PatFrag ld_frag,
                          RegisterClass KRC> {
  let mayLoad = 1 in {
  def rm : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst), (ins x86memop:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  []>, EVEX;
  def krm : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst), (ins KRC:$mask,
                                                         x86memop:$src),
                  !strconcat(OpcodeStr,
                      "\t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
                  []>, EVEX, EVEX_KZ;
  }
}

defm VBROADCASTI32X4 : avx512_int_subvec_broadcast_rm<0x5a, "vbroadcasti32x4",
                       i128mem, loadv2i64, VK16WM>,
                       EVEX_V512, EVEX_CD8<32, CD8VT4>;
defm VBROADCASTI64X4 : avx512_int_subvec_broadcast_rm<0x5b, "vbroadcasti64x4",
                       i256mem, loadv4i64, VK16WM>, VEX_W,
                       EVEX_V512, EVEX_CD8<64, CD8VT4>;

def : Pat<(v16i32 (int_x86_avx512_pbroadcastd_512 (v4i32 VR128X:$src))),
          (VPBROADCASTDZrr VR128X:$src)>;
def : Pat<(v8i64 (int_x86_avx512_pbroadcastq_512 (v2i64 VR128X:$src))),
          (VPBROADCASTQZrr VR128X:$src)>;

def : Pat<(v16f32 (X86VBroadcast (v16f32 VR512:$src))),
          (VBROADCASTSSZr (EXTRACT_SUBREG (v16f32 VR512:$src), sub_xmm))>;
def : Pat<(v8f64 (X86VBroadcast (v8f64 VR512:$src))),
          (VBROADCASTSDZr (EXTRACT_SUBREG (v8f64 VR512:$src), sub_xmm))>;

def : Pat<(v16i32 (X86VBroadcast (v16i32 VR512:$src))),
          (VPBROADCASTDZrr (EXTRACT_SUBREG (v16i32 VR512:$src), sub_xmm))>;
def : Pat<(v8i64 (X86VBroadcast (v8i64 VR512:$src))),
          (VPBROADCASTQZrr (EXTRACT_SUBREG (v8i64 VR512:$src), sub_xmm))>;

def : Pat<(v16f32 (int_x86_avx512_vbroadcast_ss_ps_512 (v4f32 VR128X:$src))),
          (VBROADCASTSSZr VR128X:$src)>;
def : Pat<(v8f64 (int_x86_avx512_vbroadcast_sd_pd_512 (v2f64 VR128X:$src))),
          (VBROADCASTSDZr VR128X:$src)>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v16f32 (X86VBroadcast FR32X:$src)),
          (VBROADCASTSSZr (COPY_TO_REGCLASS FR32X:$src, VR128X))>;
def : Pat<(v8f64 (X86VBroadcast FR64X:$src)),
          (VBROADCASTSDZr (COPY_TO_REGCLASS FR64X:$src, VR128X))>;


let Predicates = [HasAVX512] in {
def : Pat<(v8i32 (X86VBroadcastm (v8i1 VK8WM:$mask), (loadi32 addr:$src))),
           (EXTRACT_SUBREG
              (v16i32 (VPBROADCASTDZkrm (COPY_TO_REGCLASS VK8WM:$mask, VK16WM),
                       addr:$src)), sub_ymm)>;
}
//===----------------------------------------------------------------------===//
// AVX-512 BROADCAST MASK TO VECTOR REGISTER
//---

multiclass avx512_mask_broadcast<bits<8> opc, string OpcodeStr,
                       RegisterClass KRC> {
let Predicates = [HasCDI] in
def Zrr : AVX512XS8I<opc, MRMSrcReg, (outs VR512:$dst), (ins KRC:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  []>, EVEX, EVEX_V512;

let Predicates = [HasCDI, HasVLX] in {
def Z128rr : AVX512XS8I<opc, MRMSrcReg, (outs VR128:$dst), (ins KRC:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  []>, EVEX, EVEX_V128;
def Z256rr : AVX512XS8I<opc, MRMSrcReg, (outs VR256:$dst), (ins KRC:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  []>, EVEX, EVEX_V256;
}
}

let Predicates = [HasCDI] in {
defm VPBROADCASTMW2D : avx512_mask_broadcast<0x3A, "vpbroadcastmw2d",
                                             VK16>;
defm VPBROADCASTMB2Q : avx512_mask_broadcast<0x2A, "vpbroadcastmb2q",
                                             VK8>, VEX_W;
}

//===----------------------------------------------------------------------===//
// AVX-512 - VPERM
//
// -- immediate form --
multiclass avx512_perm_imm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  def ri : AVX512AIi8<opc, MRMSrcReg, (outs _.RC:$dst),
                     (ins _.RC:$src1, u8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set _.RC:$dst,
                       (_.VT (OpNode _.RC:$src1, (i8 imm:$src2))))]>,
                     EVEX;
  def mi : AVX512AIi8<opc, MRMSrcMem, (outs _.RC:$dst),
                     (ins _.MemOp:$src1, u8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set _.RC:$dst,
                       (_.VT (OpNode (_.LdFrag addr:$src1),
                              (i8 imm:$src2))))]>,
           EVEX, EVEX_CD8<_.EltSize, CD8VF>;
}
}

multiclass avx512_permil<bits<8> OpcImm, bits<8> OpcVar, X86VectorVTInfo _,
                         X86VectorVTInfo Ctrl> :
     avx512_perm_imm<OpcImm, "vpermil" # _.Suffix, X86VPermilpi, _> {
  let ExeDomain = _.ExeDomain in {
    def rr : AVX5128I<OpcVar, MRMSrcReg, (outs _.RC:$dst),
                     (ins _.RC:$src1, _.RC:$src2),
                     !strconcat("vpermil" # _.Suffix,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set _.RC:$dst,
                         (_.VT (X86VPermilpv _.RC:$src1,
                                  (Ctrl.VT Ctrl.RC:$src2))))]>,
             EVEX_4V;
    def rm : AVX5128I<OpcVar, MRMSrcMem, (outs _.RC:$dst),
                     (ins _.RC:$src1, Ctrl.MemOp:$src2),
                     !strconcat("vpermil" # _.Suffix,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set _.RC:$dst,
                         (_.VT (X86VPermilpv _.RC:$src1,
                                  (Ctrl.VT (Ctrl.LdFrag addr:$src2)))))]>,
             EVEX_4V;
  }
}

defm VPERMQZ :    avx512_perm_imm<0x00, "vpermq", X86VPermi, v8i64_info>,
                  EVEX_V512, VEX_W;
defm VPERMPDZ :   avx512_perm_imm<0x01, "vpermpd", X86VPermi, v8f64_info>,
                  EVEX_V512, VEX_W;

defm VPERMILPSZ : avx512_permil<0x04, 0x0C, v16f32_info, v16i32_info>,
                  EVEX_V512;
defm VPERMILPDZ : avx512_permil<0x05, 0x0D, v8f64_info, v8i64_info>,
                  EVEX_V512, VEX_W;

def : Pat<(v16i32 (X86VPermilpi VR512:$src1, (i8 imm:$imm))),
          (VPERMILPSZri VR512:$src1, imm:$imm)>;
def : Pat<(v8i64 (X86VPermilpi VR512:$src1, (i8 imm:$imm))),
          (VPERMILPDZri VR512:$src1, imm:$imm)>;

// -- VPERM - register form --
multiclass avx512_perm<bits<8> opc, string OpcodeStr, RegisterClass RC,
                     PatFrag mem_frag, X86MemOperand x86memop, ValueType OpVT> {

  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2),
                   !strconcat(OpcodeStr,
                       "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set RC:$dst,
                     (OpVT (X86VPermv RC:$src1, RC:$src2)))]>, EVEX_4V;

  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, x86memop:$src2),
                   !strconcat(OpcodeStr,
                       "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set RC:$dst,
                     (OpVT (X86VPermv RC:$src1, (mem_frag addr:$src2))))]>,
                     EVEX_4V;
}

defm VPERMDZ   : avx512_perm<0x36, "vpermd",  VR512,  loadv16i32, i512mem,
                           v16i32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMQZ   : avx512_perm<0x36, "vpermq",  VR512,  loadv8i64,  i512mem,
                           v8i64>,  EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
let ExeDomain = SSEPackedSingle in
defm VPERMPSZ  : avx512_perm<0x16, "vpermps", VR512,  loadv16f32, f512mem,
                           v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
let ExeDomain = SSEPackedDouble in
defm VPERMPDZ  : avx512_perm<0x16, "vpermpd", VR512,  loadv8f64, f512mem,
                           v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

// -- VPERM2I - 3 source operands form --
multiclass avx512_perm_3src<bits<8> opc, string OpcodeStr, RegisterClass RC,
                          PatFrag mem_frag, X86MemOperand x86memop,
                          SDNode OpNode, ValueType OpVT, RegisterClass KRC> {
let Constraints = "$src1 = $dst" in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                       "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src1, RC:$src2, RC:$src3)))]>,
                    EVEX_4V;

  def rrk : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, KRC:$mask, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                       "\t{$src3, $src2, $dst {${mask}}|"
                       "$dst {${mask}}, $src2, $src3}"),
                   [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                           (OpNode RC:$src1, RC:$src2,
                                              RC:$src3),
                                           RC:$src1)))]>,
                    EVEX_4V, EVEX_K;

  let AddedComplexity = 30 in // Prefer over VMOV*rrkz Pat<>
    def rrkz : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, KRC:$mask, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                       "\t{$src3, $src2, $dst {${mask}} {z} |",
                       "$dst {${mask}} {z}, $src2, $src3}"),
                   [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                           (OpNode RC:$src1, RC:$src2,
                                              RC:$src3),
                                           (OpVT (bitconvert
                                              (v16i32 immAllZerosV))))))]>,
                    EVEX_4V, EVEX_KZ;

  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src1, RC:$src2,
                      (mem_frag addr:$src3))))]>, EVEX_4V;

  def rmk : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, KRC:$mask, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst {${mask}}|"
                    "$dst {${mask}}, $src2, $src3}"),
                   [(set RC:$dst,
                       (OpVT (vselect KRC:$mask,
                                      (OpNode RC:$src1, RC:$src2,
                                         (mem_frag addr:$src3)),
                                      RC:$src1)))]>,
                    EVEX_4V, EVEX_K;

  let AddedComplexity = 10 in // Prefer over the rrkz variant
    def rmkz : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, KRC:$mask, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst {${mask}} {z}|"
                    "$dst {${mask}} {z}, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (vselect KRC:$mask,
                                    (OpNode RC:$src1, RC:$src2,
                                            (mem_frag addr:$src3)),
                                    (OpVT (bitconvert
                                       (v16i32 immAllZerosV))))))]>,
                    EVEX_4V, EVEX_KZ;
  }
}
defm VPERMI2D  : avx512_perm_3src<0x76, "vpermi2d",  VR512, loadv16i32,
                                  i512mem, X86VPermiv3, v16i32, VK16WM>,
                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMI2Q  : avx512_perm_3src<0x76, "vpermi2q",  VR512, loadv8i64,
                                  i512mem, X86VPermiv3, v8i64, VK8WM>,
                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMI2PS : avx512_perm_3src<0x77, "vpermi2ps",  VR512, loadv16f32,
                                  i512mem, X86VPermiv3, v16f32, VK16WM>,
                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMI2PD : avx512_perm_3src<0x77, "vpermi2pd",  VR512, loadv8f64,
                                  i512mem, X86VPermiv3, v8f64, VK8WM>,
                  EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

multiclass avx512_perm_table_3src<bits<8> opc, string Suffix, RegisterClass RC,
                          PatFrag mem_frag, X86MemOperand x86memop,
                          SDNode OpNode, ValueType OpVT, RegisterClass KRC,
                          ValueType MaskVT, RegisterClass MRC> :
        avx512_perm_3src<opc, "vpermt2"##Suffix, RC, mem_frag, x86memop, OpNode,
                         OpVT, KRC> {
  def : Pat<(OpVT (!cast<Intrinsic>("int_x86_avx512_mask_vpermt_"##Suffix##"_512")
                     VR512:$idx, VR512:$src1, VR512:$src2, -1)),
            (!cast<Instruction>(NAME#rr) VR512:$src1, VR512:$idx, VR512:$src2)>;

  def : Pat<(OpVT (!cast<Intrinsic>("int_x86_avx512_mask_vpermt_"##Suffix##"_512")
                     VR512:$idx, VR512:$src1, VR512:$src2, MRC:$mask)),
            (!cast<Instruction>(NAME#rrk) VR512:$src1,
              (MaskVT (COPY_TO_REGCLASS MRC:$mask, KRC)), VR512:$idx, VR512:$src2)>;
}

defm VPERMT2D  : avx512_perm_table_3src<0x7E, "d",  VR512, loadv16i32, i512mem,
                               X86VPermv3, v16i32, VK16WM, v16i1, GR16>,
                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMT2Q  : avx512_perm_table_3src<0x7E, "q",  VR512, loadv8i64, i512mem,
                               X86VPermv3, v8i64, VK8WM, v8i1, GR8>,
                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMT2PS : avx512_perm_table_3src<0x7F, "ps",  VR512, loadv16f32, i512mem,
                               X86VPermv3, v16f32, VK16WM, v16i1, GR16>,
                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMT2PD : avx512_perm_table_3src<0x7F, "pd",  VR512, loadv8f64, i512mem,
                               X86VPermv3, v8f64, VK8WM, v8i1, GR8>,
                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512 - BLEND using mask
//
multiclass avx512_blendmask<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} |${dst}, $src1, $src2}"),
             []>, EVEX_4V;
  def rrk : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}}|${dst} {${mask}}, $src1, $src2}"),
             [(set _.RC:$dst, (X86select _.KRCWM:$mask, (_.VT _.RC:$src1),
                 (_.VT _.RC:$src2)))]>, EVEX_4V, EVEX_K;
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_KZ;
  let mayLoad = 1 in {
  def rm  : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
             (ins _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} |${dst},  $src1, $src2}"),
             []>, EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
  def rmk : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}}|${dst} {${mask}}, $src1, $src2}"),
             [(set _.RC:$dst, (X86select _.KRCWM:$mask, (_.VT _.RC:$src1),
              (_.VT (bitconvert (_.LdFrag addr:$src2)))))]>,
              EVEX_4V, EVEX_K, EVEX_CD8<_.EltSize, CD8VF>;
  def rmkz : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_KZ, EVEX_CD8<_.EltSize, CD8VF>;
  }
  }
}
multiclass avx512_blendmask_rmb<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {

  def rmbk : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
      (ins _.KRCWM:$mask, _.RC:$src1, _.ScalarMemOp:$src2),
       !strconcat(OpcodeStr,
            "\t{${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
            "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, "}"),
      [(set _.RC:$dst,(X86select _.KRCWM:$mask, (_.VT _.RC:$src1),
                       (X86VBroadcast (_.ScalarLdFrag addr:$src2))))]>,
      EVEX_4V, EVEX_K, EVEX_B, EVEX_CD8<_.EltSize, CD8VF>;

  def rmb : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
      (ins _.RC:$src1, _.ScalarMemOp:$src2),
       !strconcat(OpcodeStr,
            "\t{${src2}", _.BroadcastStr, ", $src1, $dst|",
            "$dst, $src1, ${src2}", _.BroadcastStr, "}"),
      []>,  EVEX_4V, EVEX_B, EVEX_CD8<_.EltSize, CD8VF>;

}

multiclass blendmask_dq <bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo> {
  defm Z : avx512_blendmask      <opc, OpcodeStr, VTInfo.info512>,
           avx512_blendmask_rmb  <opc, OpcodeStr, VTInfo.info512>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256 : avx512_blendmask<opc, OpcodeStr, VTInfo.info256>,
                avx512_blendmask_rmb  <opc, OpcodeStr, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_blendmask<opc, OpcodeStr, VTInfo.info128>,
                avx512_blendmask_rmb  <opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  }
}

multiclass blendmask_bw <bits<8> opc, string OpcodeStr,
                         AVX512VLVectorVTInfo VTInfo> {
  let Predicates = [HasBWI] in
    defm Z : avx512_blendmask    <opc, OpcodeStr, VTInfo.info512>, EVEX_V512;

  let Predicates = [HasBWI, HasVLX] in {
    defm Z256 : avx512_blendmask <opc, OpcodeStr, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_blendmask <opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  }
}


defm VBLENDMPS : blendmask_dq <0x65, "vblendmps", avx512vl_f32_info>;
defm VBLENDMPD : blendmask_dq <0x65, "vblendmpd", avx512vl_f64_info>, VEX_W;
defm VPBLENDMD : blendmask_dq <0x64, "vpblendmd", avx512vl_i32_info>;
defm VPBLENDMQ : blendmask_dq <0x64, "vpblendmq", avx512vl_i64_info>, VEX_W;
defm VPBLENDMB : blendmask_bw <0x66, "vpblendmb", avx512vl_i8_info>;
defm VPBLENDMW : blendmask_bw <0x66, "vpblendmw", avx512vl_i16_info>, VEX_W;


let Predicates = [HasAVX512] in {
def : Pat<(v8f32 (vselect (v8i1 VK8WM:$mask), (v8f32 VR256X:$src1),
                            (v8f32 VR256X:$src2))),
            (EXTRACT_SUBREG
              (v16f32 (VBLENDMPSZrrk (COPY_TO_REGCLASS VK8WM:$mask, VK16WM),
            (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v8i32 (vselect (v8i1 VK8WM:$mask), (v8i32 VR256X:$src1),
                            (v8i32 VR256X:$src2))),
            (EXTRACT_SUBREG
                (v16i32 (VPBLENDMDZrrk (COPY_TO_REGCLASS VK8WM:$mask, VK16WM),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;
}
//===----------------------------------------------------------------------===//
// Compare Instructions
//===----------------------------------------------------------------------===//

// avx512_cmp_scalar - AVX512 CMPSS and CMPSD
multiclass avx512_cmp_scalar<RegisterClass RC, X86MemOperand x86memop,
                            SDNode OpNode, ValueType VT,
                            PatFrag ld_frag, string Suffix> {
  def rr : AVX512Ii8<0xC2, MRMSrcReg,
                (outs VK1:$dst), (ins RC:$src1, RC:$src2, AVXCC:$cc),
                !strconcat("vcmp${cc}", Suffix,
                           "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                [(set VK1:$dst, (OpNode (VT RC:$src1), RC:$src2, imm:$cc))],
                IIC_SSE_ALU_F32S_RR>, EVEX_4V;
  def rm : AVX512Ii8<0xC2, MRMSrcMem,
                (outs VK1:$dst), (ins RC:$src1, x86memop:$src2, AVXCC:$cc),
                !strconcat("vcmp${cc}", Suffix,
                           "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                [(set VK1:$dst, (OpNode (VT RC:$src1),
                (ld_frag addr:$src2), imm:$cc))], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512Ii8<0xC2, MRMSrcReg,
               (outs VK1:$dst), (ins RC:$src1, RC:$src2, u8imm:$cc),
               !strconcat("vcmp", Suffix,
                          "\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32S_RR>, EVEX_4V;
    let mayLoad = 1 in
    def rmi_alt : AVX512Ii8<0xC2, MRMSrcMem,
               (outs VK1:$dst), (ins RC:$src1, x86memop:$src2, u8imm:$cc),
               !strconcat("vcmp", Suffix,
                          "\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  }
}

let Predicates = [HasAVX512] in {
defm VCMPSSZ : avx512_cmp_scalar<FR32X, f32mem, X86cmpms, f32, loadf32, "ss">,
                                 XS;
defm VCMPSDZ : avx512_cmp_scalar<FR64X, f64mem, X86cmpms, f64, loadf64, "sd">,
                                 XD, VEX_W;
}

multiclass avx512_icmp_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
              X86VectorVTInfo _> {
  def rr : AVX512BI<opc, MRMSrcReg,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2)))],
             IIC_SSE_ALU_F32P_RR>, EVEX_4V;
  let mayLoad = 1 in
  def rm : AVX512BI<opc, MRMSrcMem,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                                     (_.VT (bitconvert (_.LdFrag addr:$src2)))))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  def rrk : AVX512BI<opc, MRMSrcReg,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
              !strconcat(OpcodeStr, "\t{$src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                   (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2))))],
              IIC_SSE_ALU_F32P_RR>, EVEX_4V, EVEX_K;
  let mayLoad = 1 in
  def rmk : AVX512BI<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2),
              !strconcat(OpcodeStr, "\t{$src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                   (OpNode (_.VT _.RC:$src1),
                                       (_.VT (bitconvert
                                              (_.LdFrag addr:$src2))))))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K;
}

multiclass avx512_icmp_packed_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
              X86VectorVTInfo _> :
           avx512_icmp_packed<opc, OpcodeStr, OpNode, _> {
  let mayLoad = 1 in {
  def rmb : AVX512BI<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.RC:$src1, _.ScalarMemOp:$src2),
              !strconcat(OpcodeStr, "\t{${src2}", _.BroadcastStr, ", $src1, $dst",
                                    "|$dst, $src1, ${src2}", _.BroadcastStr, "}"),
              [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                              (X86VBroadcast (_.ScalarLdFrag addr:$src2))))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_B;
  def rmbk : AVX512BI<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1,
                                       _.ScalarMemOp:$src2),
               !strconcat(OpcodeStr,
                          "\t{${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, "}"),
               [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                      (OpNode (_.VT _.RC:$src1),
                                        (X86VBroadcast
                                          (_.ScalarLdFrag addr:$src2)))))],
               IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K, EVEX_B;
  }
}

multiclass avx512_icmp_packed_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_packed<opc, OpcodeStr, OpNode, VTInfo.info512>,
           EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_packed<opc, OpcodeStr, OpNode, VTInfo.info256>,
                EVEX_V256;
    defm Z128 : avx512_icmp_packed<opc, OpcodeStr, OpNode, VTInfo.info128>,
                EVEX_V128;
  }
}

multiclass avx512_icmp_packed_rmb_vl<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, AVX512VLVectorVTInfo VTInfo,
                                  Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_packed_rmb<opc, OpcodeStr, OpNode, VTInfo.info512>,
           EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_packed_rmb<opc, OpcodeStr, OpNode, VTInfo.info256>,
                EVEX_V256;
    defm Z128 : avx512_icmp_packed_rmb<opc, OpcodeStr, OpNode, VTInfo.info128>,
                EVEX_V128;
  }
}

defm VPCMPEQB : avx512_icmp_packed_vl<0x74, "vpcmpeqb", X86pcmpeqm,
                      avx512vl_i8_info, HasBWI>,
                EVEX_CD8<8, CD8VF>;

defm VPCMPEQW : avx512_icmp_packed_vl<0x75, "vpcmpeqw", X86pcmpeqm,
                      avx512vl_i16_info, HasBWI>,
                EVEX_CD8<16, CD8VF>;

defm VPCMPEQD : avx512_icmp_packed_rmb_vl<0x76, "vpcmpeqd", X86pcmpeqm,
                      avx512vl_i32_info, HasAVX512>,
                EVEX_CD8<32, CD8VF>;

defm VPCMPEQQ : avx512_icmp_packed_rmb_vl<0x29, "vpcmpeqq", X86pcmpeqm,
                      avx512vl_i64_info, HasAVX512>,
                T8PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPCMPGTB : avx512_icmp_packed_vl<0x64, "vpcmpgtb", X86pcmpgtm,
                      avx512vl_i8_info, HasBWI>,
                EVEX_CD8<8, CD8VF>;

defm VPCMPGTW : avx512_icmp_packed_vl<0x65, "vpcmpgtw", X86pcmpgtm,
                      avx512vl_i16_info, HasBWI>,
                EVEX_CD8<16, CD8VF>;

defm VPCMPGTD : avx512_icmp_packed_rmb_vl<0x66, "vpcmpgtd", X86pcmpgtm,
                      avx512vl_i32_info, HasAVX512>,
                EVEX_CD8<32, CD8VF>;

defm VPCMPGTQ : avx512_icmp_packed_rmb_vl<0x37, "vpcmpgtq", X86pcmpgtm,
                      avx512vl_i64_info, HasAVX512>,
                T8PD, VEX_W, EVEX_CD8<64, CD8VF>;

def : Pat<(v8i1 (X86pcmpgtm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2))),
            (COPY_TO_REGCLASS (VPCMPGTDZrr
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm))), VK8)>;

def : Pat<(v8i1 (X86pcmpeqm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2))),
            (COPY_TO_REGCLASS (VPCMPEQDZrr
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm))), VK8)>;

multiclass avx512_icmp_cc<bits<8> opc, string Suffix, SDNode OpNode,
                          X86VectorVTInfo _> {
  def rri : AVX512AIi8<opc, MRMSrcReg,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2, AVX512ICC:$cc),
             !strconcat("vpcmp${cc}", Suffix,
                        "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                       imm:$cc))],
             IIC_SSE_ALU_F32P_RR>, EVEX_4V;
  let mayLoad = 1 in
  def rmi : AVX512AIi8<opc, MRMSrcMem,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.MemOp:$src2, AVX512ICC:$cc),
             !strconcat("vpcmp${cc}", Suffix,
                        "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                              (_.VT (bitconvert (_.LdFrag addr:$src2))),
                              imm:$cc))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  def rrik : AVX512AIi8<opc, MRMSrcReg,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2,
                                      AVX512ICC:$cc),
              !strconcat("vpcmp${cc}", Suffix,
                         "\t{$src2, $src1, $dst {${mask}}|",
                         "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                  (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                          imm:$cc)))],
              IIC_SSE_ALU_F32P_RR>, EVEX_4V, EVEX_K;
  let mayLoad = 1 in
  def rmik : AVX512AIi8<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2,
                                    AVX512ICC:$cc),
              !strconcat("vpcmp${cc}", Suffix,
                         "\t{$src2, $src1, $dst {${mask}}|",
                         "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                   (OpNode (_.VT _.RC:$src1),
                                      (_.VT (bitconvert (_.LdFrag addr:$src2))),
                                      imm:$cc)))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K;

  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512AIi8<opc, MRMSrcReg,
               (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
               !strconcat("vpcmp", Suffix, "\t{$cc, $src2, $src1, $dst|",
                          "$dst, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RR>, EVEX_4V;
    let mayLoad = 1 in
    def rmi_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.RC:$src1, _.MemOp:$src2, u8imm:$cc),
               !strconcat("vpcmp", Suffix, "\t{$cc, $src2, $src1, $dst|",
                          "$dst, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
    def rrik_alt : AVX512AIi8<opc, MRMSrcReg,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2,
                                       u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                          "\t{$cc, $src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RR>, EVEX_4V, EVEX_K;
    let mayLoad = 1 in
    def rmik_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2,
                                       u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                          "\t{$cc, $src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K;
  }
}

multiclass avx512_icmp_cc_rmb<bits<8> opc, string Suffix, SDNode OpNode,
                              X86VectorVTInfo _> :
           avx512_icmp_cc<opc, Suffix, OpNode, _> {
  def rmib : AVX512AIi8<opc, MRMSrcMem,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.ScalarMemOp:$src2,
                                     AVX512ICC:$cc),
             !strconcat("vpcmp${cc}", Suffix,
                        "\t{${src2}", _.BroadcastStr, ", $src1, $dst|",
                        "$dst, $src1, ${src2}", _.BroadcastStr, "}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                               (X86VBroadcast (_.ScalarLdFrag addr:$src2)),
                               imm:$cc))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_B;
  def rmibk : AVX512AIi8<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1,
                                       _.ScalarMemOp:$src2, AVX512ICC:$cc),
              !strconcat("vpcmp${cc}", Suffix,
                       "\t{${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
                       "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, "}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                  (OpNode (_.VT _.RC:$src1),
                                    (X86VBroadcast (_.ScalarLdFrag addr:$src2)),
                                    imm:$cc)))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K, EVEX_B;

  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 1 in {
    def rmib_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.RC:$src1, _.ScalarMemOp:$src2,
                                       u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                   "\t{$cc, ${src2}", _.BroadcastStr, ", $src1, $dst|",
                   "$dst, $src1, ${src2}", _.BroadcastStr, ", $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_B;
    def rmibk_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1,
                                       _.ScalarMemOp:$src2, u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                  "\t{$cc, ${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
                  "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, ", $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K, EVEX_B;
  }
}

multiclass avx512_icmp_cc_vl<bits<8> opc, string Suffix, SDNode OpNode,
                             AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_cc<opc, Suffix, OpNode, VTInfo.info512>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_cc<opc, Suffix, OpNode, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_icmp_cc<opc, Suffix, OpNode, VTInfo.info128>, EVEX_V128;
  }
}

multiclass avx512_icmp_cc_rmb_vl<bits<8> opc, string Suffix, SDNode OpNode,
                                AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_cc_rmb<opc, Suffix, OpNode, VTInfo.info512>,
           EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_cc_rmb<opc, Suffix, OpNode, VTInfo.info256>,
                EVEX_V256;
    defm Z128 : avx512_icmp_cc_rmb<opc, Suffix, OpNode, VTInfo.info128>,
                EVEX_V128;
  }
}

defm VPCMPB : avx512_icmp_cc_vl<0x3F, "b", X86cmpm, avx512vl_i8_info,
                                HasBWI>, EVEX_CD8<8, CD8VF>;
defm VPCMPUB : avx512_icmp_cc_vl<0x3E, "ub", X86cmpmu, avx512vl_i8_info,
                                 HasBWI>, EVEX_CD8<8, CD8VF>;

defm VPCMPW : avx512_icmp_cc_vl<0x3F, "w", X86cmpm, avx512vl_i16_info,
                                HasBWI>, VEX_W, EVEX_CD8<16, CD8VF>;
defm VPCMPUW : avx512_icmp_cc_vl<0x3E, "uw", X86cmpmu, avx512vl_i16_info,
                                 HasBWI>, VEX_W, EVEX_CD8<16, CD8VF>;

defm VPCMPD : avx512_icmp_cc_rmb_vl<0x1F, "d", X86cmpm, avx512vl_i32_info,
                                    HasAVX512>, EVEX_CD8<32, CD8VF>;
defm VPCMPUD : avx512_icmp_cc_rmb_vl<0x1E, "ud", X86cmpmu, avx512vl_i32_info,
                                     HasAVX512>, EVEX_CD8<32, CD8VF>;

defm VPCMPQ : avx512_icmp_cc_rmb_vl<0x1F, "q", X86cmpm, avx512vl_i64_info,
                                    HasAVX512>, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPCMPUQ : avx512_icmp_cc_rmb_vl<0x1E, "uq", X86cmpmu, avx512vl_i64_info,
                                     HasAVX512>, VEX_W, EVEX_CD8<64, CD8VF>;

// avx512_cmp_packed - compare packed instructions
multiclass avx512_cmp_packed<RegisterClass KRC, RegisterClass RC,
                           X86MemOperand x86memop, ValueType vt,
                           string suffix, Domain d> {
  def rri : AVX512PIi8<0xC2, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2, AVXCC:$cc),
             !strconcat("vcmp${cc}", suffix,
                        "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (X86cmpm (vt RC:$src1), (vt RC:$src2), imm:$cc))], d>;
  let hasSideEffects = 0 in
  def rrib: AVX512PIi8<0xC2, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2, AVXCC:$cc),
     !strconcat("vcmp${cc}", suffix,
                "\t{{sae}, $src2, $src1, $dst|$dst, $src1, $src2, {sae}}"),
                [], d>, EVEX_B;
  def rmi : AVX512PIi8<0xC2, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, AVXCC:$cc),
              !strconcat("vcmp${cc}", suffix,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2, $cc}"),
             [(set KRC:$dst,
              (X86cmpm (vt RC:$src1), (load addr:$src2), imm:$cc))], d>;

  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512PIi8<0xC2, MRMSrcReg,
               (outs KRC:$dst), (ins RC:$src1, RC:$src2, u8imm:$cc),
              !strconcat("vcmp", suffix,
                        "\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}"), [], d>;
    def rrib_alt: AVX512PIi8<0xC2, MRMSrcReg,
               (outs KRC:$dst), (ins RC:$src1, RC:$src2, u8imm:$cc),
       !strconcat("vcmp", suffix,
                  "\t{{sae}, $cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc, {sae}}"),
                  [], d>, EVEX_B;
    let mayLoad = 1 in
    def rmi_alt : AVX512PIi8<0xC2, MRMSrcMem,
               (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, u8imm:$cc),
              !strconcat("vcmp", suffix,
                        "\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}"), [], d>;
  }
}

defm VCMPPSZ : avx512_cmp_packed<VK16, VR512, f512mem, v16f32,
               "ps", SSEPackedSingle>, PS, EVEX_4V, EVEX_V512,
               EVEX_CD8<32, CD8VF>;
defm VCMPPDZ : avx512_cmp_packed<VK8, VR512, f512mem, v8f64,
               "pd", SSEPackedDouble>, PD, EVEX_4V, VEX_W, EVEX_V512,
               EVEX_CD8<64, CD8VF>;

def : Pat<(v8i1 (X86cmpm (v8f32 VR256X:$src1), (v8f32 VR256X:$src2), imm:$cc)),
          (COPY_TO_REGCLASS (VCMPPSZrri
            (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            imm:$cc), VK8)>;
def : Pat<(v8i1 (X86cmpm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2), imm:$cc)),
          (COPY_TO_REGCLASS (VPCMPDZrri
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            imm:$cc), VK8)>;
def : Pat<(v8i1 (X86cmpmu (v8i32 VR256X:$src1), (v8i32 VR256X:$src2), imm:$cc)),
          (COPY_TO_REGCLASS (VPCMPUDZrri
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            imm:$cc), VK8)>;

def : Pat<(i16 (int_x86_avx512_mask_cmp_ps_512 (v16f32 VR512:$src1),
                (v16f32 VR512:$src2), i8immZExt5:$cc, (i16 -1),
                 FROUND_NO_EXC)),
          (COPY_TO_REGCLASS (VCMPPSZrrib VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR16)>;

def : Pat<(i8 (int_x86_avx512_mask_cmp_pd_512 (v8f64 VR512:$src1),
                (v8f64 VR512:$src2), i8immZExt5:$cc, (i8 -1),
                 FROUND_NO_EXC)),
          (COPY_TO_REGCLASS (VCMPPDZrrib VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR8)>;

def : Pat<(i16 (int_x86_avx512_mask_cmp_ps_512 (v16f32 VR512:$src1),
                (v16f32 VR512:$src2), i8immZExt5:$cc, (i16 -1),
                FROUND_CURRENT)),
          (COPY_TO_REGCLASS (VCMPPSZrri VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR16)>;

def : Pat<(i8 (int_x86_avx512_mask_cmp_pd_512 (v8f64 VR512:$src1),
                (v8f64 VR512:$src2), i8immZExt5:$cc, (i8 -1),
                 FROUND_CURRENT)),
          (COPY_TO_REGCLASS (VCMPPDZrri VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR8)>;

// Mask register copy, including
// - copy between mask registers
// - load/store mask registers
// - copy from GPR to mask register and vice versa
//
multiclass avx512_mask_mov<bits<8> opc_kk, bits<8> opc_km, bits<8> opc_mk,
                         string OpcodeStr, RegisterClass KRC,
                         ValueType vvt, X86MemOperand x86memop> {
  let hasSideEffects = 0 in {
    def kk : I<opc_kk, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
    let mayLoad = 1 in
    def km : I<opc_km, MRMSrcMem, (outs KRC:$dst), (ins x86memop:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (vvt (load addr:$src)))]>;
    let mayStore = 1 in
    def mk : I<opc_mk, MRMDestMem, (outs), (ins x86memop:$dst, KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               [(store KRC:$src, addr:$dst)]>;
  }
}

multiclass avx512_mask_mov_gpr<bits<8> opc_kr, bits<8> opc_rk,
                             string OpcodeStr,
                             RegisterClass KRC, RegisterClass GRC> {
  let hasSideEffects = 0 in {
    def kr : I<opc_kr, MRMSrcReg, (outs KRC:$dst), (ins GRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
    def rk : I<opc_rk, MRMSrcReg, (outs GRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
  }
}

let Predicates = [HasDQI] in
  defm KMOVB : avx512_mask_mov<0x90, 0x90, 0x91, "kmovb", VK8, v8i1, i8mem>,
               avx512_mask_mov_gpr<0x92, 0x93, "kmovb", VK8, GR32>,
               VEX, PD;

let Predicates = [HasAVX512] in
  defm KMOVW : avx512_mask_mov<0x90, 0x90, 0x91, "kmovw", VK16, v16i1, i16mem>,
               avx512_mask_mov_gpr<0x92, 0x93, "kmovw", VK16, GR32>,
               VEX, PS;

let Predicates = [HasBWI] in {
  defm KMOVD : avx512_mask_mov<0x90, 0x90, 0x91, "kmovd", VK32, v32i1,i32mem>,
               VEX, PD, VEX_W;
  defm KMOVD : avx512_mask_mov_gpr<0x92, 0x93, "kmovd", VK32, GR32>,
               VEX, XD;
}

let Predicates = [HasBWI] in {
  defm KMOVQ : avx512_mask_mov<0x90, 0x90, 0x91, "kmovq", VK64, v64i1, i64mem>,
               VEX, PS, VEX_W;
  defm KMOVQ : avx512_mask_mov_gpr<0x92, 0x93, "kmovq", VK64, GR64>,
               VEX, XD, VEX_W;
}

// GR from/to mask register
let Predicates = [HasDQI] in {
  def : Pat<(v8i1 (bitconvert (i8 GR8:$src))),
            (KMOVBkr (SUBREG_TO_REG (i32 0), GR8:$src, sub_8bit))>;
  def : Pat<(i8 (bitconvert (v8i1 VK8:$src))),
            (EXTRACT_SUBREG (KMOVBrk VK8:$src), sub_8bit)>;
}
let Predicates = [HasAVX512] in {
  def : Pat<(v16i1 (bitconvert (i16 GR16:$src))),
            (KMOVWkr (SUBREG_TO_REG (i32 0), GR16:$src, sub_16bit))>;
  def : Pat<(i16 (bitconvert (v16i1 VK16:$src))),
            (EXTRACT_SUBREG (KMOVWrk VK16:$src), sub_16bit)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(v32i1 (bitconvert (i32 GR32:$src))), (KMOVDkr GR32:$src)>;
  def : Pat<(i32 (bitconvert (v32i1 VK32:$src))), (KMOVDrk VK32:$src)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(v64i1 (bitconvert (i64 GR64:$src))), (KMOVQkr GR64:$src)>;
  def : Pat<(i64 (bitconvert (v64i1 VK64:$src))), (KMOVQrk VK64:$src)>;
}

// Load/store kreg
let Predicates = [HasDQI] in {
  def : Pat<(store (i8 (bitconvert (v8i1 VK8:$src))), addr:$dst),
            (KMOVBmk addr:$dst, VK8:$src)>;
  def : Pat<(v8i1 (bitconvert (i8 (load addr:$src)))),
            (KMOVBkm addr:$src)>;
}
let Predicates = [HasAVX512, NoDQI] in {
  def : Pat<(store (i8 (bitconvert (v8i1 VK8:$src))), addr:$dst),
            (KMOVWmk addr:$dst, (COPY_TO_REGCLASS VK8:$src, VK16))>;
  def : Pat<(v8i1 (bitconvert (i8 (load addr:$src)))),
            (COPY_TO_REGCLASS (KMOVWkm addr:$src), VK8)>;
}
let Predicates = [HasAVX512] in {
  def : Pat<(store (i16 (bitconvert (v16i1 VK16:$src))), addr:$dst),
            (KMOVWmk addr:$dst, VK16:$src)>;
  def : Pat<(i1 (load addr:$src)),
            (COPY_TO_REGCLASS (KMOVWkm addr:$src), VK1)>;
  def : Pat<(v16i1 (bitconvert (i16 (load addr:$src)))),
            (KMOVWkm addr:$src)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(store (i32 (bitconvert (v32i1 VK32:$src))), addr:$dst),
            (KMOVDmk addr:$dst, VK32:$src)>;
  def : Pat<(v32i1 (bitconvert (i32 (load addr:$src)))),
            (KMOVDkm addr:$src)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(store (i64 (bitconvert (v64i1 VK64:$src))), addr:$dst),
            (KMOVQmk addr:$dst, VK64:$src)>;
  def : Pat<(v64i1 (bitconvert (i64 (load addr:$src)))),
            (KMOVQkm addr:$src)>;
}

let Predicates = [HasAVX512] in {
  def : Pat<(i1 (trunc (i64 GR64:$src))),
            (COPY_TO_REGCLASS (KMOVWkr (AND32ri (EXTRACT_SUBREG $src, sub_32bit),
                                        (i32 1))), VK1)>;

  def : Pat<(i1 (trunc (i32 GR32:$src))),
            (COPY_TO_REGCLASS (KMOVWkr (AND32ri $src, (i32 1))), VK1)>;

  def : Pat<(i1 (trunc (i8 GR8:$src))),
       (COPY_TO_REGCLASS
        (KMOVWkr (AND32ri (SUBREG_TO_REG (i32 0), GR8:$src, sub_8bit), (i32 1))),
       VK1)>;
  def : Pat<(i1 (trunc (i16 GR16:$src))),
       (COPY_TO_REGCLASS
        (KMOVWkr (AND32ri (SUBREG_TO_REG (i32 0), $src, sub_16bit), (i32 1))),
       VK1)>;

  def : Pat<(i32 (zext VK1:$src)),
            (AND32ri (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)), (i32 1))>;
  def : Pat<(i8 (zext VK1:$src)),
            (EXTRACT_SUBREG
             (AND32ri (KMOVWrk
                       (COPY_TO_REGCLASS VK1:$src, VK16)), (i32 1)), sub_8bit)>;
  def : Pat<(i64 (zext VK1:$src)),
            (AND64ri8 (SUBREG_TO_REG (i64 0),
             (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)), sub_32bit), (i64 1))>;
  def : Pat<(i16 (zext VK1:$src)),
            (EXTRACT_SUBREG
             (AND32ri (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)), (i32 1)),
              sub_16bit)>;
  def : Pat<(v16i1 (scalar_to_vector VK1:$src)),
            (COPY_TO_REGCLASS VK1:$src, VK16)>;
  def : Pat<(v8i1 (scalar_to_vector VK1:$src)),
            (COPY_TO_REGCLASS VK1:$src, VK8)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(v32i1 (scalar_to_vector VK1:$src)),
            (COPY_TO_REGCLASS VK1:$src, VK32)>;
  def : Pat<(v64i1 (scalar_to_vector VK1:$src)),
            (COPY_TO_REGCLASS VK1:$src, VK64)>;
}


// With AVX-512 only, 8-bit mask is promoted to 16-bit mask.
let Predicates = [HasAVX512] in {
  // GR from/to 8-bit mask without native support
  def : Pat<(v8i1 (bitconvert (i8 GR8:$src))),
            (COPY_TO_REGCLASS
              (KMOVWkr (SUBREG_TO_REG (i32 0), GR8:$src, sub_8bit)),
              VK8)>;
  def : Pat<(i8 (bitconvert (v8i1 VK8:$src))),
            (EXTRACT_SUBREG
              (KMOVWrk (COPY_TO_REGCLASS VK8:$src, VK16)),
              sub_8bit)>;

  def : Pat<(i1 (X86Vextract VK16:$src, (iPTR 0))),
            (COPY_TO_REGCLASS VK16:$src, VK1)>;
  def : Pat<(i1 (X86Vextract VK8:$src, (iPTR 0))),
            (COPY_TO_REGCLASS VK8:$src, VK1)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(i1 (X86Vextract VK32:$src, (iPTR 0))),
            (COPY_TO_REGCLASS VK32:$src, VK1)>;
  def : Pat<(i1 (X86Vextract VK64:$src, (iPTR 0))),
            (COPY_TO_REGCLASS VK64:$src, VK1)>;
}

// Mask unary operation
// - KNOT
multiclass avx512_mask_unop<bits<8> opc, string OpcodeStr,
                            RegisterClass KRC, SDPatternOperator OpNode,
                            Predicate prd> {
  let Predicates = [prd] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (OpNode KRC:$src))]>;
}

multiclass avx512_mask_unop_all<bits<8> opc, string OpcodeStr,
                                SDPatternOperator OpNode> {
  defm B : avx512_mask_unop<opc, !strconcat(OpcodeStr, "b"), VK8, OpNode,
                            HasDQI>, VEX, PD;
  defm W : avx512_mask_unop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode,
                            HasAVX512>, VEX, PS;
  defm D : avx512_mask_unop<opc, !strconcat(OpcodeStr, "d"), VK32, OpNode,
                            HasBWI>, VEX, PD, VEX_W;
  defm Q : avx512_mask_unop<opc, !strconcat(OpcodeStr, "q"), VK64, OpNode,
                            HasBWI>, VEX, PS, VEX_W;
}

defm KNOT : avx512_mask_unop_all<0x44, "knot", not>;

multiclass avx512_mask_unop_int<string IntName, string InstName> {
  let Predicates = [HasAVX512] in
    def : Pat<(!cast<Intrinsic>("int_x86_avx512_"##IntName##"_w")
                (i16 GR16:$src)),
              (COPY_TO_REGCLASS (!cast<Instruction>(InstName##"Wrr")
              (v16i1 (COPY_TO_REGCLASS GR16:$src, VK16))), GR16)>;
}
defm : avx512_mask_unop_int<"knot", "KNOT">;

let Predicates = [HasDQI] in
def : Pat<(xor VK8:$src1, (v8i1 immAllOnesV)), (KNOTBrr VK8:$src1)>;
let Predicates = [HasAVX512] in
def : Pat<(xor VK16:$src1, (v16i1 immAllOnesV)), (KNOTWrr VK16:$src1)>;
let Predicates = [HasBWI] in
def : Pat<(xor VK32:$src1, (v32i1 immAllOnesV)), (KNOTDrr VK32:$src1)>;
let Predicates = [HasBWI] in
def : Pat<(xor VK64:$src1, (v64i1 immAllOnesV)), (KNOTQrr VK64:$src1)>;

// KNL does not support KMOVB, 8-bit mask is promoted to 16-bit
let Predicates = [HasAVX512, NoDQI] in {
def : Pat<(xor VK8:$src1,  (v8i1 immAllOnesV)),
          (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$src1, VK16)), VK8)>;

def : Pat<(not VK8:$src),
          (COPY_TO_REGCLASS
            (KNOTWrr (COPY_TO_REGCLASS VK8:$src, VK16)), VK8)>;
}

// Mask binary operation
// - KAND, KANDN, KOR, KXNOR, KXOR
multiclass avx512_mask_binop<bits<8> opc, string OpcodeStr,
                           RegisterClass KRC, SDPatternOperator OpNode,
                           Predicate prd> {
  let Predicates = [prd] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               [(set KRC:$dst, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_binop_all<bits<8> opc, string OpcodeStr,
                               SDPatternOperator OpNode> {
  defm B : avx512_mask_binop<opc, !strconcat(OpcodeStr, "b"), VK8, OpNode,
                             HasDQI>, VEX_4V, VEX_L, PD;
  defm W : avx512_mask_binop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode,
                             HasAVX512>, VEX_4V, VEX_L, PS;
  defm D : avx512_mask_binop<opc, !strconcat(OpcodeStr, "d"), VK32, OpNode,
                             HasBWI>, VEX_4V, VEX_L, VEX_W, PD;
  defm Q : avx512_mask_binop<opc, !strconcat(OpcodeStr, "q"), VK64, OpNode,
                             HasBWI>, VEX_4V, VEX_L, VEX_W, PS;
}

def andn : PatFrag<(ops node:$i0, node:$i1), (and (not node:$i0), node:$i1)>;
def xnor : PatFrag<(ops node:$i0, node:$i1), (not (xor node:$i0, node:$i1))>;

let isCommutable = 1 in {
  defm KAND  : avx512_mask_binop_all<0x41, "kand",  and>;
  defm KOR   : avx512_mask_binop_all<0x45, "kor",   or>;
  defm KXNOR : avx512_mask_binop_all<0x46, "kxnor", xnor>;
  defm KXOR  : avx512_mask_binop_all<0x47, "kxor",  xor>;
}
let isCommutable = 0 in
  defm KANDN : avx512_mask_binop_all<0x42, "kandn", andn>;

def : Pat<(xor VK1:$src1, VK1:$src2),
     (COPY_TO_REGCLASS (KXORWrr (COPY_TO_REGCLASS VK1:$src1, VK16),
                                (COPY_TO_REGCLASS VK1:$src2, VK16)), VK1)>;

def : Pat<(or VK1:$src1, VK1:$src2),
     (COPY_TO_REGCLASS (KORWrr (COPY_TO_REGCLASS VK1:$src1, VK16),
                               (COPY_TO_REGCLASS VK1:$src2, VK16)), VK1)>;

def : Pat<(and VK1:$src1, VK1:$src2),
     (COPY_TO_REGCLASS (KANDWrr (COPY_TO_REGCLASS VK1:$src1, VK16),
                                (COPY_TO_REGCLASS VK1:$src2, VK16)), VK1)>;

multiclass avx512_mask_binop_int<string IntName, string InstName> {
  let Predicates = [HasAVX512] in
    def : Pat<(!cast<Intrinsic>("int_x86_avx512_"##IntName##"_w")
                (i16 GR16:$src1), (i16 GR16:$src2)),
              (COPY_TO_REGCLASS (!cast<Instruction>(InstName##"Wrr")
              (v16i1 (COPY_TO_REGCLASS GR16:$src1, VK16)),
              (v16i1 (COPY_TO_REGCLASS GR16:$src2, VK16))), GR16)>;
}

defm : avx512_mask_binop_int<"kand",  "KAND">;
defm : avx512_mask_binop_int<"kandn", "KANDN">;
defm : avx512_mask_binop_int<"kor",   "KOR">;
defm : avx512_mask_binop_int<"kxnor", "KXNOR">;
defm : avx512_mask_binop_int<"kxor",  "KXOR">;

// With AVX-512, 8-bit mask is promoted to 16-bit mask.
multiclass avx512_binop_pat<SDPatternOperator OpNode, Instruction Inst> {
  let Predicates = [HasAVX512] in
    def : Pat<(OpNode VK8:$src1, VK8:$src2),
              (COPY_TO_REGCLASS
                (Inst (COPY_TO_REGCLASS VK8:$src1, VK16),
                      (COPY_TO_REGCLASS VK8:$src2, VK16)), VK8)>;
}

defm : avx512_binop_pat<and,  KANDWrr>;
defm : avx512_binop_pat<andn, KANDNWrr>;
defm : avx512_binop_pat<or,   KORWrr>;
defm : avx512_binop_pat<xnor, KXNORWrr>;
defm : avx512_binop_pat<xor,  KXORWrr>;

// Mask unpacking
multiclass avx512_mask_unpck<bits<8> opc, string OpcodeStr,
                           RegisterClass KRC> {
  let Predicates = [HasAVX512] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>;
}

multiclass avx512_mask_unpck_bw<bits<8> opc, string OpcodeStr> {
  defm BW : avx512_mask_unpck<opc, !strconcat(OpcodeStr, "bw"), VK16>,
                            VEX_4V, VEX_L, PD;
}

defm KUNPCK : avx512_mask_unpck_bw<0x4b, "kunpck">;
def : Pat<(v16i1 (concat_vectors (v8i1 VK8:$src1), (v8i1 VK8:$src2))),
          (KUNPCKBWrr (COPY_TO_REGCLASS VK8:$src2, VK16),
                  (COPY_TO_REGCLASS VK8:$src1, VK16))>;


multiclass avx512_mask_unpck_int<string IntName, string InstName> {
  let Predicates = [HasAVX512] in
    def : Pat<(!cast<Intrinsic>("int_x86_avx512_"##IntName##"_bw")
                (i16 GR16:$src1), (i16 GR16:$src2)),
              (COPY_TO_REGCLASS (!cast<Instruction>(InstName##"BWrr")
              (v16i1 (COPY_TO_REGCLASS GR16:$src1, VK16)),
              (v16i1 (COPY_TO_REGCLASS GR16:$src2, VK16))), GR16)>;
}
defm : avx512_mask_unpck_int<"kunpck",  "KUNPCK">;

// Mask bit testing
multiclass avx512_mask_testop<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                            SDNode OpNode> {
  let Predicates = [HasAVX512], Defs = [EFLAGS] in
    def rr : I<opc, MRMSrcReg, (outs), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
               [(set EFLAGS, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_testop_w<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm W : avx512_mask_testop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                            VEX, PS;
  let Predicates = [HasDQI] in
  defm B : avx512_mask_testop<opc, !strconcat(OpcodeStr, "b"), VK8, OpNode>,
                            VEX, PD;
  let Predicates = [HasBWI] in {
  defm Q : avx512_mask_testop<opc, !strconcat(OpcodeStr, "q"), VK64, OpNode>,
                            VEX, PS, VEX_W;
  defm D : avx512_mask_testop<opc, !strconcat(OpcodeStr, "d"), VK32, OpNode>,
                            VEX, PD, VEX_W;
  }
}

defm KORTEST : avx512_mask_testop_w<0x98, "kortest", X86kortest>;

// Mask shift
multiclass avx512_mask_shiftop<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                             SDNode OpNode> {
  let Predicates = [HasAVX512] in
    def ri : Ii8<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src, u8imm:$imm),
                 !strconcat(OpcodeStr,
                            "\t{$imm, $src, $dst|$dst, $src, $imm}"),
                            [(set KRC:$dst, (OpNode KRC:$src, (i8 imm:$imm)))]>;
}

multiclass avx512_mask_shiftop_w<bits<8> opc1, bits<8> opc2, string OpcodeStr,
                               SDNode OpNode> {
  defm W : avx512_mask_shiftop<opc1, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                               VEX, TAPD, VEX_W;
  let Predicates = [HasDQI] in
  defm B : avx512_mask_shiftop<opc1, !strconcat(OpcodeStr, "b"), VK8, OpNode>,
                               VEX, TAPD;
  let Predicates = [HasBWI] in {
  defm Q : avx512_mask_shiftop<opc2, !strconcat(OpcodeStr, "q"), VK64, OpNode>,
                               VEX, TAPD, VEX_W;
  let Predicates = [HasDQI] in
  defm D : avx512_mask_shiftop<opc2, !strconcat(OpcodeStr, "d"), VK32, OpNode>,
                               VEX, TAPD;
  }  
}

defm KSHIFTL : avx512_mask_shiftop_w<0x32, 0x33, "kshiftl", X86vshli>;
defm KSHIFTR : avx512_mask_shiftop_w<0x30, 0x31, "kshiftr", X86vsrli>;

// Mask setting all 0s or 1s
multiclass avx512_mask_setop<RegisterClass KRC, ValueType VT, PatFrag Val> {
  let Predicates = [HasAVX512] in
    let isReMaterializable = 1, isAsCheapAsAMove = 1, isPseudo = 1 in
      def #NAME# : I<0, Pseudo, (outs KRC:$dst), (ins), "",
                     [(set KRC:$dst, (VT Val))]>;
}

multiclass avx512_mask_setop_w<PatFrag Val> {
  defm B : avx512_mask_setop<VK8,   v8i1, Val>;
  defm W : avx512_mask_setop<VK16, v16i1, Val>;
}

defm KSET0 : avx512_mask_setop_w<immAllZerosV>;
defm KSET1 : avx512_mask_setop_w<immAllOnesV>;

// With AVX-512 only, 8-bit mask is promoted to 16-bit mask.
let Predicates = [HasAVX512] in {
  def : Pat<(v8i1 immAllZerosV), (COPY_TO_REGCLASS (KSET0W), VK8)>;
  def : Pat<(v8i1 immAllOnesV),  (COPY_TO_REGCLASS (KSET1W), VK8)>;
  def : Pat<(i1 0), (COPY_TO_REGCLASS (KSET0W), VK1)>;
  def : Pat<(i1 1), (COPY_TO_REGCLASS (KSET1W), VK1)>;
  def : Pat<(i1 -1), (COPY_TO_REGCLASS (KSET1W), VK1)>;
}
def : Pat<(v8i1 (extract_subvector (v16i1 VK16:$src), (iPTR 0))),
          (v8i1 (COPY_TO_REGCLASS VK16:$src, VK8))>;

def : Pat<(v16i1 (insert_subvector undef, (v8i1 VK8:$src), (iPTR 0))),
          (v16i1 (COPY_TO_REGCLASS VK8:$src, VK16))>;

def : Pat<(v8i1 (extract_subvector (v16i1 VK16:$src), (iPTR 8))),
          (v8i1 (COPY_TO_REGCLASS (KSHIFTRWri VK16:$src, (i8 8)), VK8))>;

let Predicates = [HasVLX] in {
  def : Pat<(v8i1 (insert_subvector undef, (v4i1 VK4:$src), (iPTR 0))),
            (v8i1 (COPY_TO_REGCLASS VK4:$src, VK8))>;
  def : Pat<(v8i1 (insert_subvector undef, (v2i1 VK2:$src), (iPTR 0))),
            (v8i1 (COPY_TO_REGCLASS VK2:$src, VK8))>;
  def : Pat<(v4i1 (insert_subvector undef, (v2i1 VK2:$src), (iPTR 0))),
            (v4i1 (COPY_TO_REGCLASS VK2:$src, VK4))>;
  def : Pat<(v4i1 (extract_subvector (v8i1 VK8:$src), (iPTR 0))),
            (v4i1 (COPY_TO_REGCLASS VK8:$src, VK4))>;
  def : Pat<(v2i1 (extract_subvector (v8i1 VK8:$src), (iPTR 0))),
            (v2i1 (COPY_TO_REGCLASS VK8:$src, VK2))>;
}

def : Pat<(v8i1 (X86vshli VK8:$src, (i8 imm:$imm))),
          (v8i1 (COPY_TO_REGCLASS
                 (KSHIFTLWri (COPY_TO_REGCLASS VK8:$src, VK16),
                  (I8Imm $imm)), VK8))>, Requires<[HasAVX512, NoDQI]>;

def : Pat<(v8i1 (X86vsrli VK8:$src, (i8 imm:$imm))),
          (v8i1 (COPY_TO_REGCLASS
                 (KSHIFTRWri (COPY_TO_REGCLASS VK8:$src, VK16),
                  (I8Imm $imm)), VK8))>, Requires<[HasAVX512, NoDQI]>;

def : Pat<(v4i1 (X86vshli VK4:$src, (i8 imm:$imm))),
          (v4i1 (COPY_TO_REGCLASS
                 (KSHIFTLWri (COPY_TO_REGCLASS VK4:$src, VK16),
                  (I8Imm $imm)), VK4))>, Requires<[HasAVX512]>;

def : Pat<(v4i1 (X86vsrli VK4:$src, (i8 imm:$imm))),
          (v4i1 (COPY_TO_REGCLASS
                 (KSHIFTRWri (COPY_TO_REGCLASS VK4:$src, VK16),
                  (I8Imm $imm)), VK4))>, Requires<[HasAVX512]>;

//===----------------------------------------------------------------------===//
// AVX-512 - Aligned and unaligned load and store
//


multiclass avx512_load<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         PatFrag ld_frag, PatFrag mload,
                         bit IsReMaterializable = 1> {
  let hasSideEffects = 0 in {
  def rr : AVX512PI<opc, MRMSrcReg, (outs _.RC:$dst), (ins _.RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), [],
                    _.ExeDomain>, EVEX;
  def rrkz : AVX512PI<opc, MRMSrcReg, (outs _.RC:$dst),
                      (ins _.KRCWM:$mask,  _.RC:$src),
                      !strconcat(OpcodeStr, "\t{$src, ${dst} {${mask}} {z}|",
                       "${dst} {${mask}} {z}, $src}"), [], _.ExeDomain>,
                       EVEX, EVEX_KZ;

  let canFoldAsLoad = 1, isReMaterializable = IsReMaterializable,
      SchedRW = [WriteLoad] in
  def rm : AVX512PI<opc, MRMSrcMem, (outs _.RC:$dst), (ins _.MemOp:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set _.RC:$dst, (_.VT (bitconvert (ld_frag addr:$src))))],
                    _.ExeDomain>, EVEX;

  let Constraints = "$src0 = $dst" in {
  def rrk : AVX512PI<opc, MRMSrcReg, (outs _.RC:$dst),
                    (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src1),
                    !strconcat(OpcodeStr, "\t{$src1, ${dst} {${mask}}|",
                    "${dst} {${mask}}, $src1}"),
                    [(set _.RC:$dst, (_.VT (vselect _.KRCWM:$mask,
                                        (_.VT _.RC:$src1),
                                        (_.VT _.RC:$src0))))], _.ExeDomain>,
                     EVEX, EVEX_K;
  let mayLoad = 1, SchedRW = [WriteLoad] in
    def rmk : AVX512PI<opc, MRMSrcMem, (outs _.RC:$dst),
                     (ins _.RC:$src0, _.KRCWM:$mask, _.MemOp:$src1),
                     !strconcat(OpcodeStr, "\t{$src1, ${dst} {${mask}}|",
                      "${dst} {${mask}}, $src1}"),
                     [(set _.RC:$dst, (_.VT
                         (vselect _.KRCWM:$mask,
                          (_.VT (bitconvert (ld_frag addr:$src1))),
                           (_.VT _.RC:$src0))))], _.ExeDomain>, EVEX, EVEX_K;
  }
  let mayLoad = 1, SchedRW = [WriteLoad] in
  def rmkz : AVX512PI<opc, MRMSrcMem, (outs _.RC:$dst),
                  (ins _.KRCWM:$mask, _.MemOp:$src),
                  OpcodeStr #"\t{$src, ${dst} {${mask}} {z}|"#
                                "${dst} {${mask}} {z}, $src}",
                  [(set _.RC:$dst, (_.VT (vselect _.KRCWM:$mask,
                    (_.VT (bitconvert (ld_frag addr:$src))), _.ImmAllZerosV)))],
                  _.ExeDomain>, EVEX, EVEX_KZ;
  }
  def : Pat<(_.VT (mload addr:$ptr, _.KRCWM:$mask, undef)),
            (!cast<Instruction>(NAME#_.ZSuffix##rmkz) _.KRCWM:$mask, addr:$ptr)>;

  def : Pat<(_.VT (mload addr:$ptr, _.KRCWM:$mask, _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#_.ZSuffix##rmkz) _.KRCWM:$mask, addr:$ptr)>;

  def : Pat<(_.VT (mload addr:$ptr, _.KRCWM:$mask, (_.VT _.RC:$src0))),
            (!cast<Instruction>(NAME#_.ZSuffix##rmk) _.RC:$src0,
             _.KRCWM:$mask, addr:$ptr)>;
}

multiclass avx512_alignedload_vl<bits<8> opc, string OpcodeStr,
                                  AVX512VLVectorVTInfo _,
                                  Predicate prd,
                                  bit IsReMaterializable = 1> {
  let Predicates = [prd] in
  defm Z : avx512_load<opc, OpcodeStr, _.info512, _.info512.AlignedLdFrag,
                       masked_load_aligned512, IsReMaterializable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
  defm Z256 : avx512_load<opc, OpcodeStr, _.info256, _.info256.AlignedLdFrag,
                          masked_load_aligned256, IsReMaterializable>, EVEX_V256;
  defm Z128 : avx512_load<opc, OpcodeStr, _.info128, _.info128.AlignedLdFrag,
                          masked_load_aligned128, IsReMaterializable>, EVEX_V128;
  }
}

multiclass avx512_load_vl<bits<8> opc, string OpcodeStr,
                                  AVX512VLVectorVTInfo _,
                                  Predicate prd,
                                  bit IsReMaterializable = 1> {
  let Predicates = [prd] in
  defm Z : avx512_load<opc, OpcodeStr, _.info512, _.info512.LdFrag,
                       masked_load_unaligned, IsReMaterializable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
  defm Z256 : avx512_load<opc, OpcodeStr, _.info256, _.info256.LdFrag,
                         masked_load_unaligned, IsReMaterializable>, EVEX_V256;
  defm Z128 : avx512_load<opc, OpcodeStr, _.info128, _.info128.LdFrag,
                         masked_load_unaligned, IsReMaterializable>, EVEX_V128;
  }
}

multiclass avx512_store<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                        PatFrag st_frag, PatFrag mstore> {
  let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in {
  def rr_alt : AVX512PI<opc, MRMDestReg, (outs _.RC:$dst), (ins _.RC:$src),
                        OpcodeStr # "\t{$src, $dst|$dst, $src}", [],
                        _.ExeDomain>, EVEX;
  let Constraints = "$src1 = $dst" in
  def rrk_alt : AVX512PI<opc, MRMDestReg, (outs  _.RC:$dst),
                         (ins _.RC:$src1, _.KRCWM:$mask, _.RC:$src2),
                         OpcodeStr #
                         "\t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}",
                         [], _.ExeDomain>,  EVEX, EVEX_K;
  def rrkz_alt : AVX512PI<opc, MRMDestReg, (outs  _.RC:$dst),
                          (ins _.KRCWM:$mask, _.RC:$src),
                          OpcodeStr #
                          "\t{$src, ${dst} {${mask}} {z}|" # 
                          "${dst} {${mask}} {z}, $src}",
                          [], _.ExeDomain>, EVEX, EVEX_KZ;
  }
  let mayStore = 1 in {
  def mr : AVX512PI<opc, MRMDestMem, (outs), (ins _.MemOp:$dst, _.RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(st_frag (_.VT _.RC:$src), addr:$dst)], _.ExeDomain>, EVEX;
  def mrk : AVX512PI<opc, MRMDestMem, (outs),
                     (ins _.MemOp:$dst, _.KRCWM:$mask, _.RC:$src),
              OpcodeStr # "\t{$src, ${dst} {${mask}}|${dst} {${mask}}, $src}",
               [], _.ExeDomain>, EVEX, EVEX_K;
  }

  def: Pat<(mstore addr:$ptr, _.KRCWM:$mask, (_.VT _.RC:$src)),
           (!cast<Instruction>(NAME#_.ZSuffix##mrk) addr:$ptr,
                                                    _.KRCWM:$mask, _.RC:$src)>;
}


multiclass avx512_store_vl< bits<8> opc, string OpcodeStr,
                            AVX512VLVectorVTInfo _, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_store<opc, OpcodeStr, _.info512, store,
                        masked_store_unaligned>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_store<opc, OpcodeStr, _.info256, store,
                             masked_store_unaligned>, EVEX_V256;
    defm Z128 : avx512_store<opc, OpcodeStr, _.info128, store,
                             masked_store_unaligned>, EVEX_V128;
  }
}

multiclass avx512_alignedstore_vl<bits<8> opc, string OpcodeStr,
                                  AVX512VLVectorVTInfo _,  Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_store<opc, OpcodeStr, _.info512, alignedstore512,
                        masked_store_aligned512>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_store<opc, OpcodeStr, _.info256, alignedstore256,
                             masked_store_aligned256>, EVEX_V256;
    defm Z128 : avx512_store<opc, OpcodeStr, _.info128, alignedstore,
                             masked_store_aligned128>, EVEX_V128;
  }
}

defm VMOVAPS : avx512_alignedload_vl<0x28, "vmovaps", avx512vl_f32_info,
                                     HasAVX512>,
               avx512_alignedstore_vl<0x29, "vmovaps", avx512vl_f32_info,
                                      HasAVX512>,  PS, EVEX_CD8<32, CD8VF>;

defm VMOVAPD : avx512_alignedload_vl<0x28, "vmovapd", avx512vl_f64_info,
                                     HasAVX512>,
               avx512_alignedstore_vl<0x29, "vmovapd", avx512vl_f64_info,
                                     HasAVX512>, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMOVUPS : avx512_load_vl<0x10, "vmovups", avx512vl_f32_info, HasAVX512>,
               avx512_store_vl<0x11, "vmovups", avx512vl_f32_info, HasAVX512>,
                              PS, EVEX_CD8<32, CD8VF>;

defm VMOVUPD : avx512_load_vl<0x10, "vmovupd", avx512vl_f64_info, HasAVX512, 0>,
               avx512_store_vl<0x11, "vmovupd", avx512vl_f64_info, HasAVX512>,
               PD, VEX_W, EVEX_CD8<64, CD8VF>;

def: Pat<(v8f64 (int_x86_avx512_mask_loadu_pd_512 addr:$ptr,
                (bc_v8f64 (v16i32 immAllZerosV)), GR8:$mask)),
       (VMOVUPDZrmkz (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)), addr:$ptr)>;

def: Pat<(v16f32 (int_x86_avx512_mask_loadu_ps_512 addr:$ptr,
                 (bc_v16f32 (v16i32 immAllZerosV)), GR16:$mask)),
       (VMOVUPSZrmkz (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)), addr:$ptr)>;

def: Pat<(v8f64 (int_x86_avx512_mask_load_pd_512 addr:$ptr,
                (bc_v8f64 (v16i32 immAllZerosV)), GR8:$mask)),
       (VMOVAPDZrmkz (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)), addr:$ptr)>;

def: Pat<(v16f32 (int_x86_avx512_mask_load_ps_512 addr:$ptr,
                 (bc_v16f32 (v16i32 immAllZerosV)), GR16:$mask)),
       (VMOVAPSZrmkz (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)), addr:$ptr)>;

def: Pat<(v8f64 (int_x86_avx512_mask_load_pd_512 addr:$ptr,
                (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1))),
       (VMOVAPDZrm addr:$ptr)>;

def: Pat<(v16f32 (int_x86_avx512_mask_load_ps_512 addr:$ptr,
                 (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1))),
       (VMOVAPSZrm addr:$ptr)>;

def: Pat<(int_x86_avx512_mask_storeu_ps_512 addr:$ptr, (v16f32 VR512:$src),
          GR16:$mask),
         (VMOVUPSZmrk addr:$ptr, (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)),
            VR512:$src)>;
def: Pat<(int_x86_avx512_mask_storeu_pd_512 addr:$ptr, (v8f64 VR512:$src),
          GR8:$mask),
         (VMOVUPDZmrk addr:$ptr, (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)),
            VR512:$src)>;

def: Pat<(int_x86_avx512_mask_store_ps_512 addr:$ptr, (v16f32 VR512:$src),
          GR16:$mask),
         (VMOVAPSZmrk addr:$ptr, (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)),
            VR512:$src)>;
def: Pat<(int_x86_avx512_mask_store_pd_512 addr:$ptr, (v8f64 VR512:$src),
          GR8:$mask),
         (VMOVAPDZmrk addr:$ptr, (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)),
            VR512:$src)>;

let Predicates = [HasAVX512, NoVLX] in {
def: Pat<(masked_store addr:$ptr, VK8WM:$mask, (v8f32 VR256:$src)),
         (VMOVUPSZmrk addr:$ptr,
         (v16i1 (COPY_TO_REGCLASS VK8WM:$mask, VK16WM)),
         (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR256:$src, sub_ymm))>;

def: Pat<(v8f32 (masked_load addr:$ptr, VK8WM:$mask, undef)),
         (v8f32 (EXTRACT_SUBREG (v16f32 (VMOVUPSZrmkz 
          (v16i1 (COPY_TO_REGCLASS VK8WM:$mask, VK16WM)), addr:$ptr)), sub_ymm))>;

def: Pat<(v8f32 (masked_load addr:$ptr, VK8WM:$mask, (v8f32 VR256:$src0))),
         (v8f32 (EXTRACT_SUBREG (v16f32 (VMOVUPSZrmk
         (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR256:$src0, sub_ymm),
          (v16i1 (COPY_TO_REGCLASS VK8WM:$mask, VK16WM)), addr:$ptr)), sub_ymm))>;
}

defm VMOVDQA32 : avx512_alignedload_vl<0x6F, "vmovdqa32", avx512vl_i32_info,
                                       HasAVX512>,
                 avx512_alignedstore_vl<0x7F, "vmovdqa32", avx512vl_i32_info,
                                       HasAVX512>, PD, EVEX_CD8<32, CD8VF>;

defm VMOVDQA64 : avx512_alignedload_vl<0x6F, "vmovdqa64", avx512vl_i64_info,
                                       HasAVX512>,
                 avx512_alignedstore_vl<0x7F, "vmovdqa64", avx512vl_i64_info,
                                    HasAVX512>, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMOVDQU8 : avx512_load_vl<0x6F, "vmovdqu8", avx512vl_i8_info, HasBWI>,
                 avx512_store_vl<0x7F, "vmovdqu8", avx512vl_i8_info,
                                 HasBWI>, XD, EVEX_CD8<8, CD8VF>;

defm VMOVDQU16 : avx512_load_vl<0x6F, "vmovdqu16", avx512vl_i16_info, HasBWI>,
                 avx512_store_vl<0x7F, "vmovdqu16", avx512vl_i16_info,
                                 HasBWI>, XD, VEX_W, EVEX_CD8<16, CD8VF>;

defm VMOVDQU32 : avx512_load_vl<0x6F, "vmovdqu32", avx512vl_i32_info, HasAVX512>,
                 avx512_store_vl<0x7F, "vmovdqu32", avx512vl_i32_info,
                                 HasAVX512>, XS, EVEX_CD8<32, CD8VF>;

defm VMOVDQU64 : avx512_load_vl<0x6F, "vmovdqu64", avx512vl_i64_info, HasAVX512>,
                 avx512_store_vl<0x7F, "vmovdqu64", avx512vl_i64_info,
                                 HasAVX512>, XS, VEX_W, EVEX_CD8<64, CD8VF>;

def: Pat<(v16i32 (int_x86_avx512_mask_loadu_d_512 addr:$ptr,
                 (v16i32 immAllZerosV), GR16:$mask)),
       (VMOVDQU32Zrmkz (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)), addr:$ptr)>;

def: Pat<(v8i64 (int_x86_avx512_mask_loadu_q_512 addr:$ptr,
                (bc_v8i64 (v16i32 immAllZerosV)), GR8:$mask)),
       (VMOVDQU64Zrmkz (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)), addr:$ptr)>;

def: Pat<(int_x86_avx512_mask_storeu_d_512 addr:$ptr, (v16i32 VR512:$src),
            GR16:$mask),
         (VMOVDQU32Zmrk addr:$ptr, (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)),
            VR512:$src)>;
def: Pat<(int_x86_avx512_mask_storeu_q_512 addr:$ptr, (v8i64 VR512:$src),
            GR8:$mask),
         (VMOVDQU64Zmrk addr:$ptr, (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)),
            VR512:$src)>;

let AddedComplexity = 20 in {
def : Pat<(v8i64 (vselect VK8WM:$mask, (v8i64 VR512:$src),
                          (bc_v8i64 (v16i32 immAllZerosV)))),
                  (VMOVDQU64Zrrkz VK8WM:$mask, VR512:$src)>;

def : Pat<(v8i64 (vselect VK8WM:$mask, (bc_v8i64 (v16i32 immAllZerosV)),
                          (v8i64 VR512:$src))),
   (VMOVDQU64Zrrkz (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$mask, VK16)),
                                              VK8), VR512:$src)>;

def : Pat<(v16i32 (vselect VK16WM:$mask, (v16i32 VR512:$src),
                           (v16i32 immAllZerosV))),
                  (VMOVDQU32Zrrkz VK16WM:$mask, VR512:$src)>;

def : Pat<(v16i32 (vselect VK16WM:$mask, (v16i32 immAllZerosV),
                           (v16i32 VR512:$src))),
                  (VMOVDQU32Zrrkz (KNOTWrr VK16WM:$mask), VR512:$src)>;
}
// NoVLX patterns
let Predicates = [HasAVX512, NoVLX] in {
def: Pat<(masked_store addr:$ptr, VK8WM:$mask, (v8i32 VR256:$src)),
         (VMOVDQU32Zmrk addr:$ptr,
         (v16i1 (COPY_TO_REGCLASS VK8WM:$mask, VK16WM)),
         (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR256:$src, sub_ymm))>;

def: Pat<(v8i32 (masked_load addr:$ptr, VK8WM:$mask, undef)),
         (v8i32 (EXTRACT_SUBREG (v16i32 (VMOVDQU32Zrmkz 
          (v16i1 (COPY_TO_REGCLASS VK8WM:$mask, VK16WM)), addr:$ptr)), sub_ymm))>;
}

// Move Int Doubleword to Packed Double Int
//
def VMOVDI2PDIZrr : AVX512BI<0x6E, MRMSrcReg, (outs VR128X:$dst), (ins GR32:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set VR128X:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))], IIC_SSE_MOVDQ>,
                        EVEX, VEX_LIG;
def VMOVDI2PDIZrm : AVX512BI<0x6E, MRMSrcMem, (outs VR128X:$dst), (ins i32mem:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set VR128X:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))],
                        IIC_SSE_MOVDQ>, EVEX, VEX_LIG, EVEX_CD8<32, CD8VT1>;
def VMOV64toPQIZrr : AVX512BI<0x6E, MRMSrcReg, (outs VR128X:$dst), (ins GR64:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                        [(set VR128X:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))],
                          IIC_SSE_MOVDQ>, EVEX, VEX_W, VEX_LIG;
let isCodeGenOnly = 1 in {
def VMOV64toSDZrr : AVX512BI<0x6E, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "vmovq\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))],
                       IIC_SSE_MOVDQ>, EVEX, VEX_W, Sched<[WriteMove]>;
def VMOVSDto64Zrr : AVX512BI<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64:$src),
                         "vmovq\t{$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (bitconvert FR64:$src))],
                         IIC_SSE_MOVDQ>, EVEX, VEX_W, Sched<[WriteMove]>;
}
def VMOVSDto64Zmr : AVX512BI<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64:$src),
                         "vmovq\t{$src, $dst|$dst, $src}",
                         [(store (i64 (bitconvert FR64:$src)), addr:$dst)],
                         IIC_SSE_MOVDQ>, EVEX, VEX_W, Sched<[WriteStore]>,
                         EVEX_CD8<64, CD8VT1>;

// Move Int Doubleword to Single Scalar
//
let isCodeGenOnly = 1 in {
def VMOVDI2SSZrr  : AVX512BI<0x6E, MRMSrcReg, (outs FR32X:$dst), (ins GR32:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set FR32X:$dst, (bitconvert GR32:$src))],
                      IIC_SSE_MOVDQ>, EVEX, VEX_LIG;

def VMOVDI2SSZrm  : AVX512BI<0x6E, MRMSrcMem, (outs FR32X:$dst), (ins i32mem:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set FR32X:$dst, (bitconvert (loadi32 addr:$src)))],
                      IIC_SSE_MOVDQ>, EVEX, VEX_LIG, EVEX_CD8<32, CD8VT1>;
}

// Move doubleword from xmm register to r/m32
//
def VMOVPDI2DIZrr  : AVX512BI<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128X:$src),
                       "vmovd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (vector_extract (v4i32 VR128X:$src),
                                        (iPTR 0)))], IIC_SSE_MOVD_ToGP>,
                       EVEX, VEX_LIG;
def VMOVPDI2DIZmr  : AVX512BI<0x7E, MRMDestMem, (outs),
                       (ins i32mem:$dst, VR128X:$src),
                       "vmovd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (vector_extract (v4i32 VR128X:$src),
                                     (iPTR 0))), addr:$dst)], IIC_SSE_MOVDQ>,
                       EVEX, VEX_LIG, EVEX_CD8<32, CD8VT1>;

// Move quadword from xmm1 register to r/m64
//
def VMOVPQIto64Zrr : I<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128X:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, (extractelt (v2i64 VR128X:$src),
                                                   (iPTR 0)))],
                      IIC_SSE_MOVD_ToGP>, PD, EVEX, VEX_LIG, VEX_W,
                      Requires<[HasAVX512, In64BitMode]>;

def VMOVPQIto64Zmr : I<0xD6, MRMDestMem, (outs),
                       (ins i64mem:$dst, VR128X:$src),
                       "vmovq\t{$src, $dst|$dst, $src}",
                       [(store (extractelt (v2i64 VR128X:$src), (iPTR 0)),
                               addr:$dst)], IIC_SSE_MOVDQ>,
                       EVEX, PD, VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>,
                       Sched<[WriteStore]>, Requires<[HasAVX512, In64BitMode]>;

// Move Scalar Single to Double Int
//
let isCodeGenOnly = 1 in {
def VMOVSS2DIZrr  : AVX512BI<0x7E, MRMDestReg, (outs GR32:$dst),
                      (ins FR32X:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (bitconvert FR32X:$src))],
                      IIC_SSE_MOVD_ToGP>, EVEX, VEX_LIG;
def VMOVSS2DIZmr  : AVX512BI<0x7E, MRMDestMem, (outs),
                      (ins i32mem:$dst, FR32X:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(store (i32 (bitconvert FR32X:$src)), addr:$dst)],
                      IIC_SSE_MOVDQ>, EVEX, VEX_LIG, EVEX_CD8<32, CD8VT1>;
}

// Move Quadword Int to Packed Quadword Int
//
def VMOVQI2PQIZrm : AVX512BI<0x6E, MRMSrcMem, (outs VR128X:$dst),
                      (ins i64mem:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [(set VR128X:$dst,
                        (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>,
                      EVEX, VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;

//===----------------------------------------------------------------------===//
// AVX-512  MOVSS, MOVSD
//===----------------------------------------------------------------------===//

multiclass avx512_move_scalar <string asm, RegisterClass RC,
                              SDNode OpNode, ValueType vt,
                              X86MemOperand x86memop, PatFrag mem_pat> {
  let hasSideEffects = 0 in {
  def rr : SI<0x10, MRMSrcReg, (outs VR128X:$dst), (ins VR128X:$src1, RC:$src2),
              !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set VR128X:$dst, (vt (OpNode VR128X:$src1,
                                      (scalar_to_vector RC:$src2))))],
              IIC_SSE_MOV_S_RR>, EVEX_4V, VEX_LIG;
  let Constraints = "$src1 = $dst" in
  def rrk : SI<0x10, MRMSrcReg, (outs VR128X:$dst),
              (ins VR128X:$src1, VK1WM:$mask, RC:$src2, RC:$src3),
              !strconcat(asm,
                "\t{$src3, $src2, $dst {${mask}}|$dst {${mask}}, $src2, $src3}"),
              [], IIC_SSE_MOV_S_RR>, EVEX_4V, VEX_LIG, EVEX_K;
  def rm : SI<0x10, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
              [(set RC:$dst, (mem_pat addr:$src))], IIC_SSE_MOV_S_RM>,
              EVEX, VEX_LIG;
  let mayStore = 1 in {
  def mr: SI<0x11, MRMDestMem, (outs), (ins x86memop:$dst, RC:$src),
             !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
             [(store RC:$src, addr:$dst)], IIC_SSE_MOV_S_MR>,
             EVEX, VEX_LIG;
  def mrk: SI<0x11, MRMDestMem, (outs), (ins x86memop:$dst, VK1WM:$mask, RC:$src),
             !strconcat(asm, "\t{$src, $dst {${mask}}|$dst {${mask}}, $src}"),
             [], IIC_SSE_MOV_S_MR>,
             EVEX, VEX_LIG, EVEX_K;
  } // mayStore
  } //hasSideEffects = 0
}

let ExeDomain = SSEPackedSingle in
defm VMOVSSZ : avx512_move_scalar<"movss", FR32X, X86Movss, v4f32, f32mem,
                                 loadf32>, XS, EVEX_CD8<32, CD8VT1>;

let ExeDomain = SSEPackedDouble in
defm VMOVSDZ : avx512_move_scalar<"movsd", FR64X, X86Movsd, v2f64, f64mem,
                                 loadf64>, XD, VEX_W, EVEX_CD8<64, CD8VT1>;

def : Pat<(f32 (X86select VK1WM:$mask, (f32 FR32X:$src1), (f32 FR32X:$src2))),
          (COPY_TO_REGCLASS (VMOVSSZrrk (COPY_TO_REGCLASS FR32X:$src2, VR128X),
           VK1WM:$mask, (f32 (IMPLICIT_DEF)), FR32X:$src1), FR32X)>;

def : Pat<(f64 (X86select VK1WM:$mask, (f64 FR64X:$src1), (f64 FR64X:$src2))),
          (COPY_TO_REGCLASS (VMOVSDZrrk (COPY_TO_REGCLASS FR64X:$src2, VR128X),
           VK1WM:$mask, (f64 (IMPLICIT_DEF)), FR64X:$src1), FR64X)>;

def : Pat<(int_x86_avx512_mask_store_ss addr:$dst, VR128X:$src, GR8:$mask),
          (VMOVSSZmrk addr:$dst, (i1 (COPY_TO_REGCLASS GR8:$mask, VK1WM)),
           (COPY_TO_REGCLASS VR128X:$src, FR32X))>;

// For the disassembler
let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in {
  def VMOVSSZrr_REV : SI<0x11, MRMDestReg, (outs VR128X:$dst),
                        (ins VR128X:$src1, FR32X:$src2),
                        "movss\t{$src2, $src1, $dst|$dst, $src1, $src2}", [],
                        IIC_SSE_MOV_S_RR>,
                        XS, EVEX_4V, VEX_LIG;
  def VMOVSDZrr_REV : SI<0x11, MRMDestReg, (outs VR128X:$dst),
                        (ins VR128X:$src1, FR64X:$src2),
                        "movsd\t{$src2, $src1, $dst|$dst, $src1, $src2}", [],
                        IIC_SSE_MOV_S_RR>,
                        XD, EVEX_4V, VEX_LIG, VEX_W;
}

let Predicates = [HasAVX512] in {
  let AddedComplexity = 15 in {
  // Move scalar to XMM zero-extended, zeroing a VR128X then do a
  // MOVS{S,D} to the lower bits.
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector FR32X:$src)))),
            (VMOVSSZrr (v4f32 (V_SET0)), FR32X:$src)>;
  def : Pat<(v4f32 (X86vzmovl (v4f32 VR128X:$src))),
            (VMOVSSZrr (v4f32 (V_SET0)), (COPY_TO_REGCLASS VR128X:$src, FR32X))>;
  def : Pat<(v4i32 (X86vzmovl (v4i32 VR128X:$src))),
            (VMOVSSZrr (v4i32 (V_SET0)), (COPY_TO_REGCLASS VR128X:$src, FR32X))>;
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector FR64X:$src)))),
            (VMOVSDZrr (v2f64 (V_SET0)), FR64X:$src)>;

  // Move low f32 and clear high bits.
  def : Pat<(v8f32 (X86vzmovl (v8f32 VR256X:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSZrr (v4f32 (V_SET0)),
              (EXTRACT_SUBREG (v8f32 VR256X:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v8i32 (X86vzmovl (v8i32 VR256X:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSZrr (v4i32 (V_SET0)),
                       (EXTRACT_SUBREG (v8i32 VR256X:$src), sub_xmm)), sub_xmm)>;
  }

  let AddedComplexity = 20 in {
  // MOVSSrm zeros the high parts of the register; represent this
  // with SUBREG_TO_REG. The AVX versions also write: DST[255:128] <- 0
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector (loadf32 addr:$src))))),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;
  def : Pat<(v4f32 (scalar_to_vector (loadf32 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;
  def : Pat<(v4f32 (X86vzmovl (loadv4f32 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;

  // MOVSDrm zeros the high parts of the register; represent this
  // with SUBREG_TO_REG. The AVX versions also write: DST[255:128] <- 0
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector (loadf64 addr:$src))))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (scalar_to_vector (loadf64 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (X86vzmovl (loadv2f64 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (X86vzmovl (bc_v2f64 (loadv4f32 addr:$src)))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (X86vzload addr:$src)),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;

  // Represent the same patterns above but in the form they appear for
  // 256-bit types
  def : Pat<(v8i32 (X86vzmovl (insert_subvector undef,
                   (v4i32 (scalar_to_vector (loadi32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrm addr:$src), sub_xmm)>;
  def : Pat<(v8f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector (loadf32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSSZrm addr:$src), sub_xmm)>;
  def : Pat<(v4f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector (loadf64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrm addr:$src), sub_xmm)>;
  }
  def : Pat<(v8f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector FR32X:$src)), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (v4f32 (VMOVSSZrr (v4f32 (V_SET0)),
                                            FR32X:$src)), sub_xmm)>;
  def : Pat<(v4f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector FR64X:$src)), (iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (v2f64 (VMOVSDZrr (v2f64 (V_SET0)),
                                     FR64X:$src)), sub_xmm)>;
  def : Pat<(v4i64 (X86vzmovl (insert_subvector undef,
                   (v2i64 (scalar_to_vector (loadi64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (VMOVQI2PQIZrm addr:$src), sub_xmm)>;

  // Move low f64 and clear high bits.
  def : Pat<(v4f64 (X86vzmovl (v4f64 VR256X:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSDZrr (v2f64 (V_SET0)),
                       (EXTRACT_SUBREG (v4f64 VR256X:$src), sub_xmm)), sub_xmm)>;

  def : Pat<(v4i64 (X86vzmovl (v4i64 VR256X:$src))),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrr (v2i64 (V_SET0)),
                       (EXTRACT_SUBREG (v4i64 VR256X:$src), sub_xmm)), sub_xmm)>;

  // Extract and store.
  def : Pat<(store (f32 (vector_extract (v4f32 VR128X:$src), (iPTR 0))),
                   addr:$dst),
            (VMOVSSZmr addr:$dst, (COPY_TO_REGCLASS (v4f32 VR128X:$src), FR32X))>;
  def : Pat<(store (f64 (vector_extract (v2f64 VR128X:$src), (iPTR 0))),
                   addr:$dst),
            (VMOVSDZmr addr:$dst, (COPY_TO_REGCLASS (v2f64 VR128X:$src), FR64X))>;

  // Shuffle with VMOVSS
  def : Pat<(v4i32 (X86Movss VR128X:$src1, VR128X:$src2)),
            (VMOVSSZrr (v4i32 VR128X:$src1),
                      (COPY_TO_REGCLASS (v4i32 VR128X:$src2), FR32X))>;
  def : Pat<(v4f32 (X86Movss VR128X:$src1, VR128X:$src2)),
            (VMOVSSZrr (v4f32 VR128X:$src1),
                      (COPY_TO_REGCLASS (v4f32 VR128X:$src2), FR32X))>;

  // 256-bit variants
  def : Pat<(v8i32 (X86Movss VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSSZrr (EXTRACT_SUBREG (v8i32 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v8i32 VR256X:$src2), sub_xmm)),
              sub_xmm)>;
  def : Pat<(v8f32 (X86Movss VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSSZrr (EXTRACT_SUBREG (v8f32 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v8f32 VR256X:$src2), sub_xmm)),
              sub_xmm)>;

  // Shuffle with VMOVSD
  def : Pat<(v2i64 (X86Movsd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v2f64 (X86Movsd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v4f32 (X86Movsd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v4i32 (X86Movsd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;

  // 256-bit variants
  def : Pat<(v4i64 (X86Movsd VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSDZrr (EXTRACT_SUBREG (v4i64 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v4i64 VR256X:$src2), sub_xmm)),
              sub_xmm)>;
  def : Pat<(v4f64 (X86Movsd VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSDZrr (EXTRACT_SUBREG (v4f64 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v4f64 VR256X:$src2), sub_xmm)),
              sub_xmm)>;

  def : Pat<(v2f64 (X86Movlpd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v2i64 (X86Movlpd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v4f32 (X86Movlps VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v4i32 (X86Movlps VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
}

let AddedComplexity = 15 in
def VMOVZPQILo2PQIZrr : AVX512XSI<0x7E, MRMSrcReg, (outs VR128X:$dst),
                                (ins VR128X:$src),
                                "vmovq\t{$src, $dst|$dst, $src}",
                                [(set VR128X:$dst, (v2i64 (X86vzmovl
                                                   (v2i64 VR128X:$src))))],
                                IIC_SSE_MOVQ_RR>, EVEX, VEX_W;

let AddedComplexity = 20 in
def VMOVZPQILo2PQIZrm : AVX512XSI<0x7E, MRMSrcMem, (outs VR128X:$dst),
                                 (ins i128mem:$src),
                                 "vmovq\t{$src, $dst|$dst, $src}",
                                 [(set VR128X:$dst, (v2i64 (X86vzmovl
                                                     (loadv2i64 addr:$src))))],
                                 IIC_SSE_MOVDQ>, EVEX, VEX_W,
                                 EVEX_CD8<8, CD8VT8>;

let Predicates = [HasAVX512] in {
  // AVX 128-bit movd/movq instruction write zeros in the high 128-bit part.
  let AddedComplexity = 20 in {
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector (loadi32 addr:$src))))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v2i64 (X86vzmovl (v2i64 (scalar_to_vector GR64:$src)))),
              (VMOV64toPQIZrr GR64:$src)>;
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector GR32:$src)))),
              (VMOVDI2PDIZrr GR32:$src)>;

    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv4f32 addr:$src)))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv2i64 addr:$src)))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v2i64 (X86vzmovl (loadv2i64 addr:$src))),
            (VMOVZPQILo2PQIZrm addr:$src)>;
    def : Pat<(v2f64 (X86vzmovl (v2f64 VR128X:$src))),
            (VMOVZPQILo2PQIZrr VR128X:$src)>;
    def : Pat<(v2i64 (X86vzload addr:$src)),
            (VMOVZPQILo2PQIZrm addr:$src)>;
  }

  // Use regular 128-bit instructions to match 256-bit scalar_to_vec+zext.
  def : Pat<(v8i32 (X86vzmovl (insert_subvector undef,
                               (v4i32 (scalar_to_vector GR32:$src)),(iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrr GR32:$src), sub_xmm)>;
  def : Pat<(v4i64 (X86vzmovl (insert_subvector undef,
                               (v2i64 (scalar_to_vector GR64:$src)),(iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (VMOV64toPQIZrr GR64:$src), sub_xmm)>;
}

def : Pat<(v16i32 (X86Vinsert (v16i32 immAllZerosV), GR32:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrr GR32:$src2), sub_xmm)>;

def : Pat<(v8i64 (X86Vinsert (bc_v8i64 (v16i32 immAllZerosV)), GR64:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOV64toPQIZrr GR64:$src2), sub_xmm)>;

def : Pat<(v16i32 (X86Vinsert undef, GR32:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrr GR32:$src2), sub_xmm)>;

def : Pat<(v8i64 (X86Vinsert undef, GR64:$src2, (iPTR 0))),
        (SUBREG_TO_REG (i32 0), (VMOV64toPQIZrr GR64:$src2), sub_xmm)>;

//===----------------------------------------------------------------------===//
// AVX-512 - Non-temporals
//===----------------------------------------------------------------------===//
let SchedRW = [WriteLoad] in {
  def VMOVNTDQAZrm : AVX512PI<0x2A, MRMSrcMem, (outs VR512:$dst),
                        (ins i512mem:$src), "vmovntdqa\t{$src, $dst|$dst, $src}",
                        [(set VR512:$dst, (int_x86_avx512_movntdqa addr:$src))],
                        SSEPackedInt>, EVEX, T8PD, EVEX_V512,
                        EVEX_CD8<64, CD8VF>;

  let Predicates = [HasAVX512, HasVLX] in {
    def VMOVNTDQAZ256rm : AVX512PI<0x2A, MRMSrcMem, (outs VR256X:$dst),
                             (ins i256mem:$src),
                             "vmovntdqa\t{$src, $dst|$dst, $src}", [],
                             SSEPackedInt>, EVEX, T8PD, EVEX_V256,
                             EVEX_CD8<64, CD8VF>;

    def VMOVNTDQAZ128rm : AVX512PI<0x2A, MRMSrcMem, (outs VR128X:$dst),
                             (ins i128mem:$src),
                             "vmovntdqa\t{$src, $dst|$dst, $src}", [],
                             SSEPackedInt>, EVEX, T8PD, EVEX_V128,
                             EVEX_CD8<64, CD8VF>;
  }
}

multiclass avx512_movnt<bits<8> opc, string OpcodeStr, PatFrag st_frag,
                        ValueType OpVT, RegisterClass RC, X86MemOperand memop,
                        Domain d, InstrItinClass itin = IIC_SSE_MOVNT> {
  let SchedRW = [WriteStore], mayStore = 1,
      AddedComplexity = 400 in
  def mr : AVX512PI<opc, MRMDestMem, (outs), (ins memop:$dst, RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(st_frag (OpVT RC:$src), addr:$dst)], d, itin>, EVEX;
}

multiclass avx512_movnt_vl<bits<8> opc, string OpcodeStr, PatFrag st_frag,
                           string elty, string elsz, string vsz512,
                           string vsz256, string vsz128, Domain d,
                           Predicate prd, InstrItinClass itin = IIC_SSE_MOVNT> {
  let Predicates = [prd] in
  defm Z : avx512_movnt<opc, OpcodeStr, st_frag,
                        !cast<ValueType>("v"##vsz512##elty##elsz), VR512,
                        !cast<X86MemOperand>(elty##"512mem"), d, itin>,
                        EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_movnt<opc, OpcodeStr, st_frag,
                             !cast<ValueType>("v"##vsz256##elty##elsz), VR256X,
                             !cast<X86MemOperand>(elty##"256mem"), d, itin>,
                             EVEX_V256;

    defm Z128 : avx512_movnt<opc, OpcodeStr, st_frag,
                             !cast<ValueType>("v"##vsz128##elty##elsz), VR128X,
                             !cast<X86MemOperand>(elty##"128mem"), d, itin>,
                             EVEX_V128;
  }
}

defm VMOVNTDQ : avx512_movnt_vl<0xE7, "vmovntdq", alignednontemporalstore,
                                "i", "64", "8", "4", "2", SSEPackedInt,
                                HasAVX512>, PD, EVEX_CD8<64, CD8VF>;

defm VMOVNTPD : avx512_movnt_vl<0x2B, "vmovntpd", alignednontemporalstore,
                                "f", "64", "8", "4", "2", SSEPackedDouble,
                                HasAVX512>, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMOVNTPS : avx512_movnt_vl<0x2B, "vmovntps", alignednontemporalstore,
                                "f", "32", "16", "8", "4", SSEPackedSingle,
                                HasAVX512>, PS, EVEX_CD8<32, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512 - Integer arithmetic
//
multiclass avx512_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           X86VectorVTInfo _, OpndItins itins,
                           bit IsCommutable = 0> {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.RC:$src2), OpcodeStr##_.Suffix,
                    "$src2, $src1", "$src1, $src2",
                    (_.VT (OpNode _.RC:$src1, _.RC:$src2)),
                    "", itins.rr, IsCommutable>,
            AVX512BIBase, EVEX_4V;

  let mayLoad = 1 in
    defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr##_.Suffix,
                    "$src2, $src1", "$src1, $src2",
                    (_.VT (OpNode _.RC:$src1,
                                  (bitconvert (_.LdFrag addr:$src2)))),
                    "", itins.rm>,
              AVX512BIBase, EVEX_4V;
}

multiclass avx512_binop_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _, OpndItins itins,
                            bit IsCommutable = 0> :
           avx512_binop_rm<opc, OpcodeStr, OpNode, _, itins, IsCommutable> {
  let mayLoad = 1 in
    defm rmb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr##_.Suffix,
                    "${src2}"##_.BroadcastStr##", $src1",
                    "$src1, ${src2}"##_.BroadcastStr,
                    (_.VT (OpNode _.RC:$src1,
                                  (X86VBroadcast
                                      (_.ScalarLdFrag addr:$src2)))),
                    "", itins.rm>,
               AVX512BIBase, EVEX_4V, EVEX_B;
}

multiclass avx512_binop_rm_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              AVX512VLVectorVTInfo VTInfo, OpndItins itins,
                              Predicate prd, bit IsCommutable = 0> {
  let Predicates = [prd] in
    defm Z : avx512_binop_rm<opc, OpcodeStr, OpNode, VTInfo.info512, itins,
                             IsCommutable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_binop_rm<opc, OpcodeStr, OpNode, VTInfo.info256, itins,
                             IsCommutable>, EVEX_V256;
    defm Z128 : avx512_binop_rm<opc, OpcodeStr, OpNode, VTInfo.info128, itins,
                             IsCommutable>, EVEX_V128;
  }
}

multiclass avx512_binop_rmb_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               AVX512VLVectorVTInfo VTInfo, OpndItins itins,
                               Predicate prd, bit IsCommutable = 0> {
  let Predicates = [prd] in
    defm Z : avx512_binop_rmb<opc, OpcodeStr, OpNode, VTInfo.info512, itins,
                             IsCommutable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_binop_rmb<opc, OpcodeStr, OpNode, VTInfo.info256, itins,
                             IsCommutable>, EVEX_V256;
    defm Z128 : avx512_binop_rmb<opc, OpcodeStr, OpNode, VTInfo.info128, itins,
                             IsCommutable>, EVEX_V128;
  }
}

multiclass avx512_binop_rm_vl_q<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rmb_vl<opc, OpcodeStr, OpNode, avx512vl_i64_info,
                               itins, prd, IsCommutable>,
                               VEX_W, EVEX_CD8<64, CD8VF>;
}

multiclass avx512_binop_rm_vl_d<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rmb_vl<opc, OpcodeStr, OpNode, avx512vl_i32_info,
                               itins, prd, IsCommutable>, EVEX_CD8<32, CD8VF>;
}

multiclass avx512_binop_rm_vl_w<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rm_vl<opc, OpcodeStr, OpNode, avx512vl_i16_info,
                              itins, prd, IsCommutable>, EVEX_CD8<16, CD8VF>;
}

multiclass avx512_binop_rm_vl_b<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rm_vl<opc, OpcodeStr, OpNode, avx512vl_i8_info,
                              itins, prd, IsCommutable>, EVEX_CD8<8, CD8VF>;
}

multiclass avx512_binop_rm_vl_dq<bits<8> opc_d, bits<8> opc_q, string OpcodeStr,
                                 SDNode OpNode, OpndItins itins, Predicate prd,
                                 bit IsCommutable = 0> {
  defm Q : avx512_binop_rm_vl_q<opc_q, OpcodeStr, OpNode, itins, prd,
                                   IsCommutable>;

  defm D : avx512_binop_rm_vl_d<opc_d, OpcodeStr, OpNode, itins, prd,
                                   IsCommutable>;
}

multiclass avx512_binop_rm_vl_bw<bits<8> opc_b, bits<8> opc_w, string OpcodeStr,
                                 SDNode OpNode, OpndItins itins, Predicate prd,
                                 bit IsCommutable = 0> {
  defm W : avx512_binop_rm_vl_w<opc_w, OpcodeStr, OpNode, itins, prd,
                                   IsCommutable>;

  defm B : avx512_binop_rm_vl_b<opc_b, OpcodeStr, OpNode, itins, prd,
                                   IsCommutable>;
}

multiclass avx512_binop_rm_vl_all<bits<8> opc_b, bits<8> opc_w,
                                  bits<8> opc_d, bits<8> opc_q,
                                  string OpcodeStr, SDNode OpNode,
                                  OpndItins itins, bit IsCommutable = 0> {
  defm NAME : avx512_binop_rm_vl_dq<opc_d, opc_q, OpcodeStr, OpNode,
                                    itins, HasAVX512, IsCommutable>,
              avx512_binop_rm_vl_bw<opc_b, opc_w, OpcodeStr, OpNode,
                                    itins, HasBWI, IsCommutable>;
}

multiclass avx512_binop_rm2<bits<8> opc, string OpcodeStr, ValueType DstVT,
                            ValueType SrcVT, RegisterClass KRC, RegisterClass RC,
                            PatFrag memop_frag, X86MemOperand x86memop,
                            PatFrag scalar_mfrag, X86MemOperand x86scalar_mop,
                            string BrdcstStr, OpndItins itins, bit IsCommutable = 0> {
  let isCommutable = IsCommutable in
  {
    def rr : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       []>, EVEX_4V;
    def rrk : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
               (ins KRC:$mask, RC:$src1, RC:$src2),
               !strconcat(OpcodeStr,
                  "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
               [], itins.rr>, EVEX_4V, EVEX_K;
    def rrkz : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, RC:$src2),
                !strconcat(OpcodeStr, "\t{$src2, $src1, $dst {${mask}} {z}" ,
                    "|$dst {${mask}} {z}, $src1, $src2}"),
                [], itins.rr>, EVEX_4V, EVEX_KZ;
  }
  let mayLoad = 1 in {
    def rm : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
              (ins RC:$src1, x86memop:$src2),
              !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              []>, EVEX_4V;
    def rmk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
               (ins KRC:$mask, RC:$src1, x86memop:$src2),
               !strconcat(OpcodeStr,
                   "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
               [], itins.rm>, EVEX_4V, EVEX_K;
    def rmkz : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, x86memop:$src2),
                !strconcat(OpcodeStr,
                    "\t{$src2, $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, $src2}"),
                [], itins.rm>, EVEX_4V, EVEX_KZ;
    def rmb : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, x86scalar_mop:$src2),
               !strconcat(OpcodeStr, "\t{${src2}", BrdcstStr,
                          ", $src1, $dst|$dst, $src1, ${src2}", BrdcstStr, "}"),
               [], itins.rm>, EVEX_4V, EVEX_B;
    def rmbk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2),
                !strconcat(OpcodeStr, "\t{${src2}", BrdcstStr,
                           ", $src1, $dst {${mask}}|$dst {${mask}}, $src1, ${src2}",
                           BrdcstStr, "}"),
                [], itins.rm>, EVEX_4V, EVEX_B, EVEX_K;
    def rmbkz : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                 (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2),
                 !strconcat(OpcodeStr, "\t{${src2}", BrdcstStr,
                            ", $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, ${src2}",
                            BrdcstStr, "}"),
                 [], itins.rm>, EVEX_4V, EVEX_B, EVEX_KZ;
  }
}

defm VPADD : avx512_binop_rm_vl_all<0xFC, 0xFD, 0xFE, 0xD4, "vpadd", add,
                                    SSE_INTALU_ITINS_P, 1>;
defm VPSUB : avx512_binop_rm_vl_all<0xF8, 0xF9, 0xFA, 0xFB, "vpsub", sub,
                                    SSE_INTALU_ITINS_P, 0>;
defm VPMULLD : avx512_binop_rm_vl_d<0x40, "vpmull", mul,
                                   SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;
defm VPMULLW : avx512_binop_rm_vl_w<0xD5, "vpmull", mul,
                                   SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMULLQ : avx512_binop_rm_vl_q<0x40, "vpmull", mul,
                                   SSE_INTALU_ITINS_P, HasDQI, 1>, T8PD;

defm VPMULDQZ : avx512_binop_rm2<0x28, "vpmuldq", v8i64, v16i32, VK8WM, VR512,
                   loadv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 1>, T8PD, EVEX_V512,
                   EVEX_CD8<64, CD8VF>, VEX_W;

defm VPMULUDQZ : avx512_binop_rm2<0xF4, "vpmuludq", v8i64, v16i32, VK8WM, VR512,
                   loadv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTMUL_ITINS_P, 1>, EVEX_V512, EVEX_CD8<64, CD8VF>, VEX_W;

def : Pat<(v8i64 (X86pmuludq (v16i32 VR512:$src1), (v16i32 VR512:$src2))),
          (VPMULUDQZrr VR512:$src1, VR512:$src2)>;

def : Pat<(v8i64 (int_x86_avx512_mask_pmulu_dq_512 (v16i32 VR512:$src1),
           (v16i32 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
          (VPMULUDQZrr VR512:$src1, VR512:$src2)>;
def : Pat<(v8i64 (int_x86_avx512_mask_pmul_dq_512 (v16i32 VR512:$src1),
           (v16i32 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
          (VPMULDQZrr VR512:$src1, VR512:$src2)>;

defm VPMAXSB : avx512_binop_rm_vl_b<0x3C, "vpmaxs", X86smax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMAXSW : avx512_binop_rm_vl_w<0xEE, "vpmaxs", X86smax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMAXS : avx512_binop_rm_vl_dq<0x3D, 0x3D, "vpmaxs", X86smax,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

defm VPMAXUB : avx512_binop_rm_vl_b<0xDE, "vpmaxu", X86umax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMAXUW : avx512_binop_rm_vl_w<0x3E, "vpmaxu", X86umax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMAXU : avx512_binop_rm_vl_dq<0x3F, 0x3F, "vpmaxu", X86umax,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

defm VPMINSB : avx512_binop_rm_vl_b<0x38, "vpmins", X86smin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMINSW : avx512_binop_rm_vl_w<0xEA, "vpmins", X86smin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMINS : avx512_binop_rm_vl_dq<0x39, 0x39, "vpmins", X86smin,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

defm VPMINUB : avx512_binop_rm_vl_b<0xDA, "vpminu", X86umin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMINUW : avx512_binop_rm_vl_w<0x3A, "vpminu", X86umin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMINU : avx512_binop_rm_vl_dq<0x3B, 0x3B, "vpminu", X86umin,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

def : Pat <(v16i32 (int_x86_avx512_mask_pmaxs_d_512 (v16i32 VR512:$src1),
                    (v16i32 VR512:$src2), (v16i32 immAllZerosV), (i16 -1))),
           (VPMAXSDZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v16i32 (int_x86_avx512_mask_pmaxu_d_512 (v16i32 VR512:$src1),
                    (v16i32 VR512:$src2), (v16i32 immAllZerosV), (i16 -1))),
           (VPMAXUDZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v8i64 (int_x86_avx512_mask_pmaxs_q_512 (v8i64 VR512:$src1),
                (v8i64 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
           (VPMAXSQZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v8i64 (int_x86_avx512_mask_pmaxu_q_512 (v8i64 VR512:$src1),
                (v8i64 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
           (VPMAXUQZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v16i32 (int_x86_avx512_mask_pmins_d_512 (v16i32 VR512:$src1),
                    (v16i32 VR512:$src2), (v16i32 immAllZerosV), (i16 -1))),
           (VPMINSDZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v16i32 (int_x86_avx512_mask_pminu_d_512 (v16i32 VR512:$src1),
                    (v16i32 VR512:$src2), (v16i32 immAllZerosV), (i16 -1))),
           (VPMINUDZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v8i64 (int_x86_avx512_mask_pmins_q_512 (v8i64 VR512:$src1),
                (v8i64 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
           (VPMINSQZrr VR512:$src1, VR512:$src2)>;
def : Pat <(v8i64 (int_x86_avx512_mask_pminu_q_512 (v8i64 VR512:$src1),
                (v8i64 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
           (VPMINUQZrr VR512:$src1, VR512:$src2)>;
//===----------------------------------------------------------------------===//
// AVX-512 - Unpack Instructions
//===----------------------------------------------------------------------===//

multiclass avx512_unpack_fp<bits<8> opc, SDNode OpNode, ValueType vt,
                                   PatFrag mem_frag, RegisterClass RC,
                                   X86MemOperand x86memop, string asm,
                                   Domain d> {
    def rr : AVX512PI<opc, MRMSrcReg,
                (outs RC:$dst), (ins RC:$src1, RC:$src2),
                asm, [(set RC:$dst,
                           (vt (OpNode RC:$src1, RC:$src2)))],
                           d>, EVEX_4V;
    def rm : AVX512PI<opc, MRMSrcMem,
                (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
                asm, [(set RC:$dst,
                       (vt (OpNode RC:$src1,
                            (bitconvert (mem_frag addr:$src2)))))],
                        d>, EVEX_4V;
}

defm VUNPCKHPSZ: avx512_unpack_fp<0x15, X86Unpckh, v16f32, loadv8f64,
      VR512, f512mem, "vunpckhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedSingle>, PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VUNPCKHPDZ: avx512_unpack_fp<0x15, X86Unpckh, v8f64, loadv8f64,
      VR512, f512mem, "vunpckhpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedDouble>, PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VUNPCKLPSZ: avx512_unpack_fp<0x14, X86Unpckl, v16f32, loadv8f64,
      VR512, f512mem, "vunpcklps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedSingle>, PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VUNPCKLPDZ: avx512_unpack_fp<0x14, X86Unpckl, v8f64, loadv8f64,
      VR512, f512mem, "vunpcklpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedDouble>, PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

multiclass avx512_unpack_int<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, RegisterClass RC, PatFrag memop_frag,
                        X86MemOperand x86memop> {
  def rr : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (OpVT (OpNode (OpVT RC:$src1), (OpVT RC:$src2))))],
       IIC_SSE_UNPCK>, EVEX_4V;
  def rm : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (OpVT (OpNode (OpVT RC:$src1),
                                     (bitconvert (memop_frag addr:$src2)))))],
                                     IIC_SSE_UNPCK>, EVEX_4V;
}
defm VPUNPCKLDQZ  : avx512_unpack_int<0x62, "vpunpckldq", X86Unpckl, v16i32,
                                VR512, loadv16i32, i512mem>, EVEX_V512,
                                EVEX_CD8<32, CD8VF>;
defm VPUNPCKLQDQZ : avx512_unpack_int<0x6C, "vpunpcklqdq", X86Unpckl, v8i64,
                                VR512, loadv8i64, i512mem>, EVEX_V512,
                                VEX_W, EVEX_CD8<64, CD8VF>;
defm VPUNPCKHDQZ  : avx512_unpack_int<0x6A, "vpunpckhdq", X86Unpckh, v16i32,
                                VR512, loadv16i32, i512mem>, EVEX_V512,
                                EVEX_CD8<32, CD8VF>;
defm VPUNPCKHQDQZ : avx512_unpack_int<0x6D, "vpunpckhqdq", X86Unpckh, v8i64,
                                VR512, loadv8i64, i512mem>, EVEX_V512,
                                VEX_W, EVEX_CD8<64, CD8VF>;
//===----------------------------------------------------------------------===//
// AVX-512 - PSHUFD
//

multiclass avx512_pshuf_imm<bits<8> opc, string OpcodeStr, RegisterClass RC,
                         SDNode OpNode, PatFrag mem_frag,
                         X86MemOperand x86memop, ValueType OpVT> {
  def ri : AVX512Ii8<opc, MRMSrcReg, (outs RC:$dst),
                     (ins RC:$src1, u8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode RC:$src1, (i8 imm:$src2))))]>,
                     EVEX;
  def mi : AVX512Ii8<opc, MRMSrcMem, (outs RC:$dst),
                     (ins x86memop:$src1, u8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode (mem_frag addr:$src1),
                              (i8 imm:$src2))))]>, EVEX;
}

defm VPSHUFDZ : avx512_pshuf_imm<0x70, "vpshufd", VR512, X86PShufd, loadv16i32,
                      i512mem, v16i32>, PD, EVEX_V512, EVEX_CD8<32, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512  Logical Instructions
//===----------------------------------------------------------------------===//

defm VPAND : avx512_binop_rm_vl_dq<0xDB, 0xDB, "vpand", and,
                                  SSE_INTALU_ITINS_P, HasAVX512, 1>;
defm VPOR : avx512_binop_rm_vl_dq<0xEB, 0xEB, "vpor", or,
                                  SSE_INTALU_ITINS_P, HasAVX512, 1>;
defm VPXOR : avx512_binop_rm_vl_dq<0xEF, 0xEF, "vpxor", xor,
                                  SSE_INTALU_ITINS_P, HasAVX512, 1>;
defm VPANDN : avx512_binop_rm_vl_dq<0xDF, 0xDF, "vpandn", X86andnp,
                                  SSE_INTALU_ITINS_P, HasAVX512, 1>;

//===----------------------------------------------------------------------===//
// AVX-512  FP arithmetic
//===----------------------------------------------------------------------===//
multiclass avx512_fp_scalar<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode OpNode, SDNode VecNode, OpndItins itins,
                         bit IsCommutable> {

  defm rr_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                           "$src2, $src1", "$src1, $src2",
                           (VecNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                           (i32 FROUND_CURRENT)),
                           "", itins.rr, IsCommutable>;

  defm rm_Int : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (VecNode (_.VT _.RC:$src1),
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))),
                           (i32 FROUND_CURRENT)),
                         "", itins.rm, IsCommutable>;
  let isCodeGenOnly = 1, isCommutable = IsCommutable,
      Predicates = [HasAVX512] in {
  def rr : I< opc, MRMSrcReg, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.FRC:$src2), 
                          OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                          [(set _.FRC:$dst, (OpNode _.FRC:$src1, _.FRC:$src2))],
                          itins.rr>;
  def rm : I< opc, MRMSrcMem, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.ScalarMemOp:$src2), 
                         OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         [(set _.FRC:$dst, (OpNode _.FRC:$src1,
                         (_.ScalarLdFrag addr:$src2)))], itins.rr>;
  }
}

multiclass avx512_fp_scalar_round<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode VecNode, OpndItins itins, bit IsCommutable> {

  defm rrb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                          (ins _.RC:$src1, _.RC:$src2, AVX512RC:$rc), OpcodeStr,
                          "$rc, $src2, $src1", "$src1, $src2, $rc",
                          (VecNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                          (i32 imm:$rc)), "", itins.rr, IsCommutable>,
                          EVEX_B, EVEX_RC;
}
multiclass avx512_fp_scalar_sae<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode VecNode, OpndItins itins, bit IsCommutable> {

  defm rrb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                            (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                            "$src2, $src1", "$src1, $src2",
                            (VecNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                            (i32 FROUND_NO_EXC)), "{sae}">, EVEX_B;
}

multiclass avx512_binop_s_round<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SDNode VecNode,
                                  SizeItins itins, bit IsCommutable> {
  defm SSZ : avx512_fp_scalar<opc, OpcodeStr#"ss", f32x_info, OpNode, VecNode,
                              itins.s, IsCommutable>,
             avx512_fp_scalar_round<opc, OpcodeStr#"ss", f32x_info, VecNode,
                              itins.s, IsCommutable>,
                              XS, EVEX_4V, VEX_LIG,  EVEX_CD8<32, CD8VT1>;
  defm SDZ : avx512_fp_scalar<opc, OpcodeStr#"sd", f64x_info, OpNode, VecNode,
                              itins.d,                  IsCommutable>,
             avx512_fp_scalar_round<opc, OpcodeStr#"sd", f64x_info, VecNode,
                              itins.d, IsCommutable>,
                              XD, VEX_W, EVEX_4V, VEX_LIG, EVEX_CD8<64, CD8VT1>;
}

multiclass avx512_binop_s_sae<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SDNode VecNode,
                                  SizeItins itins, bit IsCommutable> {
  defm SSZ : avx512_fp_scalar<opc, OpcodeStr#"ss", f32x_info, OpNode, VecNode,
                              itins.s, IsCommutable>,
             avx512_fp_scalar_sae<opc, OpcodeStr#"ss", f32x_info, VecNode,
                              itins.s, IsCommutable>,
                              XS, EVEX_4V, VEX_LIG,  EVEX_CD8<32, CD8VT1>;
  defm SDZ : avx512_fp_scalar<opc, OpcodeStr#"sd", f64x_info, OpNode, VecNode,
                              itins.d,                  IsCommutable>,
             avx512_fp_scalar_sae<opc, OpcodeStr#"sd", f64x_info, VecNode,
                              itins.d, IsCommutable>,
                              XD, VEX_W, EVEX_4V, VEX_LIG, EVEX_CD8<64, CD8VT1>;
}
defm VADD : avx512_binop_s_round<0x58, "vadd", fadd, X86faddRnd, SSE_ALU_ITINS_S, 1>;
defm VMUL : avx512_binop_s_round<0x59, "vmul", fmul, X86fmulRnd, SSE_ALU_ITINS_S, 1>;
defm VSUB : avx512_binop_s_round<0x5C, "vsub", fsub, X86fsubRnd, SSE_ALU_ITINS_S, 0>;
defm VDIV : avx512_binop_s_round<0x5E, "vdiv", fdiv, X86fdivRnd, SSE_ALU_ITINS_S, 0>;
defm VMIN : avx512_binop_s_sae  <0x5D, "vmin", X86fmin, X86fminRnd, SSE_ALU_ITINS_S, 1>;
defm VMAX : avx512_binop_s_sae  <0x5F, "vmax", X86fmax, X86fmaxRnd, SSE_ALU_ITINS_S, 1>;

multiclass avx512_fp_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _, bit IsCommutable> {
  defm rr: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2), OpcodeStr##_.Suffix,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode _.RC:$src1, _.RC:$src2))>, EVEX_4V;
  let mayLoad = 1 in {
    defm rm: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr##_.Suffix,
                    "$src2, $src1", "$src1, $src2",
                    (OpNode _.RC:$src1, (_.LdFrag addr:$src2))>, EVEX_4V;
    defm rmb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                     (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr##_.Suffix,
                     "${src2}"##_.BroadcastStr##", $src1",
                     "$src1, ${src2}"##_.BroadcastStr,
                     (OpNode  _.RC:$src1, (_.VT (X86VBroadcast
                                                (_.ScalarLdFrag addr:$src2))))>,
                     EVEX_4V, EVEX_B;
  }//let mayLoad = 1
}

multiclass avx512_fp_round_packed<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd,
                            X86VectorVTInfo _, bit IsCommutable> {
  defm rb: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2, AVX512RC:$rc), OpcodeStr##_.Suffix,
                  "$rc, $src2, $src1", "$src1, $src2, $rc",
                  (_.VT (OpNodeRnd _.RC:$src1, _.RC:$src2, (i32 imm:$rc)))>,
                  EVEX_4V, EVEX_B, EVEX_RC;
}

multiclass avx512_fp_binop_p<bits<8> opc, string OpcodeStr, SDNode OpNode, 
                             bit IsCommutable = 0> {
  defm PSZ : avx512_fp_packed<opc, OpcodeStr, OpNode, v16f32_info,
                              IsCommutable>, EVEX_V512, PS,
                              EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp_packed<opc, OpcodeStr, OpNode, v8f64_info,
                              IsCommutable>, EVEX_V512, PD, VEX_W,
                              EVEX_CD8<64, CD8VF>;

    // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_fp_packed<opc, OpcodeStr, OpNode, v4f32x_info,
                                   IsCommutable>, EVEX_V128, PS,
                                   EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_fp_packed<opc, OpcodeStr, OpNode, v8f32x_info,
                                   IsCommutable>, EVEX_V256, PS,
                                   EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_fp_packed<opc, OpcodeStr, OpNode, v2f64x_info,
                                   IsCommutable>, EVEX_V128, PD, VEX_W,
                                   EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_fp_packed<opc, OpcodeStr, OpNode, v4f64x_info,
                                   IsCommutable>, EVEX_V256, PD, VEX_W,
                                   EVEX_CD8<64, CD8VF>;
  }
}

multiclass avx512_fp_binop_p_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd> {
  defm PSZ : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, v16f32_info, 0>,
                              EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, v8f64_info, 0>,
                              EVEX_V512, PD, VEX_W,EVEX_CD8<64, CD8VF>;
}

defm VADD : avx512_fp_binop_p<0x58, "vadd", fadd, 1>,
            avx512_fp_binop_p_round<0x58, "vadd", X86faddRnd>;
defm VMUL : avx512_fp_binop_p<0x59, "vmul", fmul, 1>,
            avx512_fp_binop_p_round<0x59, "vmul", X86fmulRnd>;
defm VSUB : avx512_fp_binop_p<0x5C, "vsub", fsub>, 
            avx512_fp_binop_p_round<0x5C, "vsub", X86fsubRnd>;
defm VDIV : avx512_fp_binop_p<0x5E, "vdiv", fdiv>,
            avx512_fp_binop_p_round<0x5E, "vdiv", X86fdivRnd>;
defm VMIN : avx512_fp_binop_p<0x5D, "vmin", X86fmin, 1>;
defm VMAX : avx512_fp_binop_p<0x5F, "vmax", X86fmax, 1>;

def : Pat<(v16f32 (int_x86_avx512_mask_max_ps_512 (v16f32 VR512:$src1),
                   (v16f32 VR512:$src2), (bc_v16f32 (v16i32 immAllZerosV)),
                   (i16 -1), FROUND_CURRENT)),
          (VMAXPSZrr VR512:$src1, VR512:$src2)>;

def : Pat<(v8f64 (int_x86_avx512_mask_max_pd_512 (v8f64 VR512:$src1),
                   (v8f64 VR512:$src2), (bc_v8f64 (v16i32 immAllZerosV)),
                   (i8 -1), FROUND_CURRENT)),
          (VMAXPDZrr VR512:$src1, VR512:$src2)>;

def : Pat<(v16f32 (int_x86_avx512_mask_min_ps_512 (v16f32 VR512:$src1),
                   (v16f32 VR512:$src2), (bc_v16f32 (v16i32 immAllZerosV)),
                   (i16 -1), FROUND_CURRENT)),
          (VMINPSZrr VR512:$src1, VR512:$src2)>;

def : Pat<(v8f64 (int_x86_avx512_mask_min_pd_512 (v8f64 VR512:$src1),
                   (v8f64 VR512:$src2), (bc_v8f64 (v16i32 immAllZerosV)),
                   (i8 -1), FROUND_CURRENT)),
          (VMINPDZrr VR512:$src1, VR512:$src2)>;
//===----------------------------------------------------------------------===//
// AVX-512  VPTESTM instructions
//===----------------------------------------------------------------------===//

multiclass avx512_vptest<bits<8> opc, string OpcodeStr, RegisterClass KRC,
              RegisterClass RC, X86MemOperand x86memop, PatFrag memop_frag,
              SDNode OpNode, ValueType vt> {
  def rr : AVX512PI<opc, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), (vt RC:$src2)))],
             SSEPackedInt>, EVEX_4V;
  def rm : AVX512PI<opc, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1),
              (bitconvert (memop_frag addr:$src2))))], SSEPackedInt>, EVEX_4V;
}

defm VPTESTMDZ  : avx512_vptest<0x27, "vptestmd", VK16, VR512,  f512mem,
                              loadv16i32, X86testm, v16i32>, T8PD, EVEX_V512,
                              EVEX_CD8<32, CD8VF>;
defm VPTESTMQZ  : avx512_vptest<0x27, "vptestmq", VK8, VR512,  f512mem,
                              loadv8i64, X86testm, v8i64>, T8PD, EVEX_V512, VEX_W,
                              EVEX_CD8<64, CD8VF>;

let Predicates = [HasCDI] in {
defm VPTESTNMDZ  : avx512_vptest<0x27, "vptestnmd", VK16, VR512,  f512mem,
                              loadv16i32, X86testnm, v16i32>, T8XS, EVEX_V512,
                              EVEX_CD8<32, CD8VF>;
defm VPTESTNMQZ  : avx512_vptest<0x27, "vptestnmq", VK8, VR512,  f512mem,
                              loadv8i64, X86testnm, v8i64>, T8XS, EVEX_V512, VEX_W,
                              EVEX_CD8<64, CD8VF>;
}

def : Pat <(i16 (int_x86_avx512_mask_ptestm_d_512 (v16i32 VR512:$src1),
                 (v16i32 VR512:$src2), (i16 -1))),
                 (COPY_TO_REGCLASS (VPTESTMDZrr VR512:$src1, VR512:$src2), GR16)>;

def : Pat <(i8 (int_x86_avx512_mask_ptestm_q_512 (v8i64 VR512:$src1),
                 (v8i64 VR512:$src2), (i8 -1))),
                 (COPY_TO_REGCLASS (VPTESTMQZrr VR512:$src1, VR512:$src2), GR8)>;

//===----------------------------------------------------------------------===//
// AVX-512  Shift instructions
//===----------------------------------------------------------------------===//
multiclass avx512_shift_rmi<bits<8> opc, Format ImmFormR, Format ImmFormM,
                         string OpcodeStr, SDNode OpNode, X86VectorVTInfo _> {
  defm ri : AVX512_maskable<opc, ImmFormR, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, u8imm:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (i8 imm:$src2))),
                   " ",  SSE_INTSHIFT_ITINS_P.rr>, AVX512BIi8Base, EVEX_4V;
  defm mi : AVX512_maskable<opc, ImmFormM, _, (outs _.RC:$dst),
                   (ins _.MemOp:$src1, u8imm:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode (_.LdFrag addr:$src1), (i8 imm:$src2))),
                   " ",  SSE_INTSHIFT_ITINS_P.rm>, AVX512BIi8Base, EVEX_4V;
}

multiclass avx512_shift_rrm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            ValueType SrcVT, PatFrag bc_frag, X86VectorVTInfo _> {
   // src2 is always 128-bit
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, VR128X:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (SrcVT VR128X:$src2))),
                   " ",  SSE_INTSHIFT_ITINS_P.rr>, AVX512BIBase, EVEX_4V;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, i128mem:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (bc_frag (loadv2i64 addr:$src2)))),
                   " ",  SSE_INTSHIFT_ITINS_P.rm>, AVX512BIBase, EVEX_4V;
}

multiclass avx512_shift_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  ValueType SrcVT, PatFrag bc_frag, X86VectorVTInfo _> {
  defm Z : avx512_shift_rrm<opc, OpcodeStr, OpNode, SrcVT, bc_frag, _>, EVEX_V512;
}

multiclass avx512_shift_types<bits<8> opcd, bits<8> opcq, string OpcodeStr,
                                 SDNode OpNode> {
  defm D : avx512_shift_sizes<opcd, OpcodeStr#"d", OpNode, v4i32, bc_v4i32,
                                 v16i32_info>, EVEX_CD8<32, CD8VQ>;
  defm Q : avx512_shift_sizes<opcq, OpcodeStr#"q", OpNode, v2i64, bc_v2i64,
                                 v8i64_info>, EVEX_CD8<64, CD8VQ>, VEX_W;
}

defm VPSRLDZ : avx512_shift_rmi<0x72, MRM2r, MRM2m, "vpsrld", X86vsrli,
                           v16i32_info>,
                           EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPSRLQZ : avx512_shift_rmi<0x73, MRM2r, MRM2m, "vpsrlq", X86vsrli,
                           v8i64_info>, EVEX_V512,
                           EVEX_CD8<64, CD8VF>, VEX_W;

defm VPSLLDZ : avx512_shift_rmi<0x72, MRM6r, MRM6m, "vpslld", X86vshli,
                           v16i32_info>, EVEX_V512,
                           EVEX_CD8<32, CD8VF>;
defm VPSLLQZ : avx512_shift_rmi<0x73, MRM6r, MRM6m, "vpsllq", X86vshli,
                           v8i64_info>, EVEX_V512,
                           EVEX_CD8<64, CD8VF>, VEX_W;

defm VPSRADZ : avx512_shift_rmi<0x72, MRM4r, MRM4m, "vpsrad", X86vsrai,
                           v16i32_info>,
                           EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPSRAQZ : avx512_shift_rmi<0x72, MRM4r, MRM4m, "vpsraq", X86vsrai,
                           v8i64_info>, EVEX_V512,
                           EVEX_CD8<64, CD8VF>, VEX_W;

defm VPSLL : avx512_shift_types<0xF2, 0xF3, "vpsll", X86vshl>;
defm VPSRA : avx512_shift_types<0xE2, 0xE2, "vpsra", X86vsra>;
defm VPSRL : avx512_shift_types<0xD2, 0xD3, "vpsrl", X86vsrl>;

//===-------------------------------------------------------------------===//
// Variable Bit Shifts
//===-------------------------------------------------------------------===//
multiclass avx512_var_shift<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (_.VT _.RC:$src2))),
                   " ",  SSE_INTSHIFT_ITINS_P.rr>, AVX5128IBase, EVEX_4V;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (_.LdFrag addr:$src2))),
                   " ",  SSE_INTSHIFT_ITINS_P.rm>, AVX5128IBase, EVEX_4V;
}

multiclass avx512_var_shift_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  AVX512VLVectorVTInfo _> {
  defm Z : avx512_var_shift<opc, OpcodeStr, OpNode, _.info512>, EVEX_V512;
}

multiclass avx512_var_shift_types<bits<8> opc, string OpcodeStr,
                                 SDNode OpNode> {
  defm D : avx512_var_shift_sizes<opc, OpcodeStr#"d", OpNode,
                                 avx512vl_i32_info>, EVEX_CD8<32, CD8VQ>;
  defm Q : avx512_var_shift_sizes<opc, OpcodeStr#"q", OpNode,
                                 avx512vl_i64_info>, EVEX_CD8<64, CD8VQ>, VEX_W;
}

defm VPSLLV : avx512_var_shift_types<0x47, "vpsllv", shl>;
defm VPSRAV : avx512_var_shift_types<0x46, "vpsrav", sra>;
defm VPSRLV : avx512_var_shift_types<0x45, "vpsrlv", srl>;

//===----------------------------------------------------------------------===//
// AVX-512 - MOVDDUP
//===----------------------------------------------------------------------===//

multiclass avx512_movddup<string OpcodeStr, RegisterClass RC, ValueType VT,
                        X86MemOperand x86memop, PatFrag memop_frag> {
def rr  : AVX512PDI<0x12, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set RC:$dst, (VT (X86Movddup RC:$src)))]>, EVEX;
def rm  : AVX512PDI<0x12, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set RC:$dst,
                      (VT (X86Movddup (memop_frag addr:$src))))]>, EVEX;
}

defm VMOVDDUPZ : avx512_movddup<"vmovddup", VR512, v8f64, f512mem, loadv8f64>,
                 VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
def : Pat<(X86Movddup (v8f64 (scalar_to_vector (loadf64 addr:$src)))),
          (VMOVDDUPZrm addr:$src)>;

//===---------------------------------------------------------------------===//
// Replicate Single FP - MOVSHDUP and MOVSLDUP
//===---------------------------------------------------------------------===//
multiclass avx512_replicate_sfp<bits<8> op, SDNode OpNode, string OpcodeStr,
                              ValueType vt, RegisterClass RC, PatFrag mem_frag,
                              X86MemOperand x86memop> {
  def rr : AVX512XSI<op, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                      [(set RC:$dst, (vt (OpNode RC:$src)))]>, EVEX;
  let mayLoad = 1 in
  def rm : AVX512XSI<op, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                      [(set RC:$dst, (OpNode (mem_frag addr:$src)))]>, EVEX;
}

defm VMOVSHDUPZ  : avx512_replicate_sfp<0x16, X86Movshdup, "vmovshdup",
                       v16f32, VR512, loadv16f32, f512mem>, EVEX_V512,
                       EVEX_CD8<32, CD8VF>;
defm VMOVSLDUPZ  : avx512_replicate_sfp<0x12, X86Movsldup, "vmovsldup",
                       v16f32, VR512, loadv16f32, f512mem>, EVEX_V512,
                       EVEX_CD8<32, CD8VF>;

def : Pat<(v16i32 (X86Movshdup VR512:$src)), (VMOVSHDUPZrr VR512:$src)>;
def : Pat<(v16i32 (X86Movshdup (loadv16i32 addr:$src))),
           (VMOVSHDUPZrm addr:$src)>;
def : Pat<(v16i32 (X86Movsldup VR512:$src)), (VMOVSLDUPZrr VR512:$src)>;
def : Pat<(v16i32 (X86Movsldup (loadv16i32 addr:$src))),
           (VMOVSLDUPZrm addr:$src)>;

//===----------------------------------------------------------------------===//
// Move Low to High and High to Low packed FP Instructions
//===----------------------------------------------------------------------===//
def VMOVLHPSZrr : AVX512PSI<0x16, MRMSrcReg, (outs VR128X:$dst),
          (ins VR128X:$src1, VR128X:$src2),
          "vmovlhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          [(set VR128X:$dst, (v4f32 (X86Movlhps VR128X:$src1, VR128X:$src2)))],
           IIC_SSE_MOV_LH>, EVEX_4V;
def VMOVHLPSZrr : AVX512PSI<0x12, MRMSrcReg, (outs VR128X:$dst),
          (ins VR128X:$src1, VR128X:$src2),
          "vmovhlps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          [(set VR128X:$dst, (v4f32 (X86Movhlps VR128X:$src1, VR128X:$src2)))],
          IIC_SSE_MOV_LH>, EVEX_4V;

let Predicates = [HasAVX512] in {
  // MOVLHPS patterns
  def : Pat<(v4i32 (X86Movlhps VR128X:$src1, VR128X:$src2)),
            (VMOVLHPSZrr VR128X:$src1, VR128X:$src2)>;
  def : Pat<(v2i64 (X86Movlhps VR128X:$src1, VR128X:$src2)),
            (VMOVLHPSZrr (v2i64 VR128X:$src1), VR128X:$src2)>;

  // MOVHLPS patterns
  def : Pat<(v4i32 (X86Movhlps VR128X:$src1, VR128X:$src2)),
            (VMOVHLPSZrr VR128X:$src1, VR128X:$src2)>;
}

//===----------------------------------------------------------------------===//
// FMA - Fused Multiply Operations
//

let Constraints = "$src1 = $dst" in {
// Omitting the parameter OpNode (= null_frag) disables ISel pattern matching.
multiclass avx512_fma3p_rm<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                           SDPatternOperator OpNode = null_frag> {
  defm r: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src1, _.RC:$src2, _.RC:$src3))>,
         AVX512FMA3Base;

  let mayLoad = 1 in
  defm m: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
            (ins _.RC:$src2, _.MemOp:$src3),
            OpcodeStr, "$src3, $src2", "$src2, $src3",
            (_.VT (OpNode _.RC:$src1, _.RC:$src2, (_.LdFrag addr:$src3)))>,
            AVX512FMA3Base; 

  defm mb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
              (ins _.RC:$src2, _.ScalarMemOp:$src3),
              OpcodeStr,   !strconcat("${src3}", _.BroadcastStr,", $src2"), !strconcat("$src2, ${src3}", _.BroadcastStr ),
              (OpNode _.RC:$src1, _.RC:$src2,(_.VT (X86VBroadcast (_.ScalarLdFrag addr:$src3))))>,	
              AVX512FMA3Base, EVEX_B;
 }
} // Constraints = "$src1 = $dst"

let Constraints = "$src1 = $dst" in {
// Omitting the parameter OpNode (= null_frag) disables ISel pattern matching.
multiclass avx512_fma3_round_rrb<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                           SDPatternOperator OpNode> {
   defm rb: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3, AVX512RC:$rc),
          OpcodeStr, "$rc, $src3, $src2", "$src2, $src3, $rc",
          (_.VT ( OpNode _.RC:$src1, _.RC:$src2, _.RC:$src3, (i32 imm:$rc)))>,
          AVX512FMA3Base, EVEX_B, EVEX_RC;
 }
} // Constraints = "$src1 = $dst"

multiclass avx512_fma3_round_forms<bits<8> opc213, string OpcodeStr,
                              X86VectorVTInfo VTI, SDPatternOperator OpNode> {
  defm v213r : avx512_fma3_round_rrb<opc213, !strconcat(OpcodeStr, "213", VTI.Suffix),
                              VTI, OpNode>, EVEX_CD8<VTI.EltSize, CD8VF>;
}

multiclass avx512_fma3p_forms<bits<8> opc213, bits<8> opc231,
                              string OpcodeStr, X86VectorVTInfo VTI,
                              SDPatternOperator OpNode> {
  defm v213r : avx512_fma3p_rm<opc213, !strconcat(OpcodeStr, "213", VTI.Suffix),
                              VTI, OpNode>, EVEX_CD8<VTI.EltSize, CD8VF>;

  defm v231r : avx512_fma3p_rm<opc231, !strconcat(OpcodeStr, "231", VTI.Suffix),
                              VTI>, EVEX_CD8<VTI.EltSize, CD8VF>;
}

multiclass avx512_fma3p<bits<8> opc213, bits<8> opc231,
                              string OpcodeStr,
                              SDPatternOperator OpNode,
                              SDPatternOperator OpNodeRnd> {
let ExeDomain = SSEPackedSingle in {
    defm NAME##PSZ      : avx512_fma3p_forms<opc213, opc231, OpcodeStr,
                                             v16f32_info, OpNode>,
                          avx512_fma3_round_forms<opc213, OpcodeStr,
                                             v16f32_info, OpNodeRnd>, EVEX_V512;
    defm NAME##PSZ256   : avx512_fma3p_forms<opc213, opc231, OpcodeStr,
                                             v8f32x_info, OpNode>, EVEX_V256;
    defm NAME##PSZ128   : avx512_fma3p_forms<opc213, opc231, OpcodeStr,
                                             v4f32x_info, OpNode>, EVEX_V128;
  }
let ExeDomain = SSEPackedDouble in {
    defm  NAME##PDZ     : avx512_fma3p_forms<opc213, opc231, OpcodeStr,
                                             v8f64_info, OpNode>,
                          avx512_fma3_round_forms<opc213, OpcodeStr,
                                             v8f64_info, OpNodeRnd>, EVEX_V512, VEX_W;
    defm  NAME##PDZ256  : avx512_fma3p_forms<opc213, opc231, OpcodeStr,
                                             v4f64x_info, OpNode>, EVEX_V256, VEX_W;
    defm  NAME##PDZ128  : avx512_fma3p_forms<opc213, opc231, OpcodeStr,
                                             v2f64x_info, OpNode>, EVEX_V128, VEX_W;
  }
}

defm VFMADD    : avx512_fma3p<0xA8, 0xB8, "vfmadd", X86Fmadd, X86FmaddRnd>;
defm VFMSUB    : avx512_fma3p<0xAA, 0xBA, "vfmsub", X86Fmsub, X86FmsubRnd>;
defm VFMADDSUB : avx512_fma3p<0xA6, 0xB6, "vfmaddsub", X86Fmaddsub, X86FmaddsubRnd>;
defm VFMSUBADD : avx512_fma3p<0xA7, 0xB7, "vfmsubadd", X86Fmsubadd, X86FmsubaddRnd>;
defm VFNMADD   : avx512_fma3p<0xAC, 0xBC, "vfnmadd", X86Fnmadd, X86FnmaddRnd>;
defm VFNMSUB   : avx512_fma3p<0xAE, 0xBE, "vfnmsub", X86Fnmsub, X86FnmsubRnd>;

let Constraints = "$src1 = $dst" in {
multiclass avx512_fma3p_m132<bits<8> opc, string OpcodeStr, SDNode OpNode,
                             X86VectorVTInfo _> {
  let mayLoad = 1 in
  def m: AVX512FMA3<opc, MRMSrcMem, (outs _.RC:$dst),
          (ins _.RC:$src1, _.RC:$src3, _.MemOp:$src2),
          !strconcat(OpcodeStr, "\t{$src2, $src3, $dst|$dst, $src3, $src2}"),
          [(set _.RC:$dst, (_.VT (OpNode _.RC:$src1, (_.LdFrag addr:$src2),
                                                    _.RC:$src3)))]>;
   def mb: AVX512FMA3<opc, MRMSrcMem, (outs _.RC:$dst),
           (ins _.RC:$src1, _.RC:$src3, _.ScalarMemOp:$src2),
           !strconcat(OpcodeStr, "\t{${src2}", _.BroadcastStr,
            ", $src3, $dst|$dst, $src3, ${src2}", _.BroadcastStr, "}"),
           [(set _.RC:$dst,
               (OpNode _.RC:$src1, (_.VT (X86VBroadcast
                                            (_.ScalarLdFrag addr:$src2))),
                                   _.RC:$src3))]>, EVEX_B;
}
} // Constraints = "$src1 = $dst"


multiclass avx512_fma3p_m132_f<bits<8> opc,
                              string OpcodeStr,
                              SDNode OpNode> {

let ExeDomain = SSEPackedSingle in {
    defm NAME##PSZ      : avx512_fma3p_m132<opc, OpcodeStr##ps,
                                             OpNode,v16f32_info>, EVEX_V512, EVEX_CD8<32, CD8VF>;
    defm NAME##PSZ256   : avx512_fma3p_m132<opc, OpcodeStr##ps,
                                             OpNode, v8f32x_info>, EVEX_V256, EVEX_CD8<32, CD8VF>;
    defm NAME##PSZ128   : avx512_fma3p_m132<opc, OpcodeStr##ps,
                                             OpNode, v4f32x_info>, EVEX_V128, EVEX_CD8<32, CD8VF>;
  }
let ExeDomain = SSEPackedDouble in {
    defm  NAME##PDZ       : avx512_fma3p_m132<opc, OpcodeStr##pd,
                                           OpNode, v8f64_info>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VF>;
    defm  NAME##PDZ256    : avx512_fma3p_m132<opc, OpcodeStr##pd,
                                           OpNode, v4f64x_info>, EVEX_V256, VEX_W, EVEX_CD8<32, CD8VF>;
    defm  NAME##PDZ128    : avx512_fma3p_m132<opc, OpcodeStr##pd,
                                           OpNode, v2f64x_info>, EVEX_V128, VEX_W, EVEX_CD8<32, CD8VF>;
  }
}

defm VFMADD132    : avx512_fma3p_m132_f<0x98, "vfmadd132", X86Fmadd>;
defm VFMSUB132    : avx512_fma3p_m132_f<0x9A, "vfmsub132", X86Fmsub>;
defm VFMADDSUB132 : avx512_fma3p_m132_f<0x96, "vfmaddsub132", X86Fmaddsub>;
defm VFMSUBADD132 : avx512_fma3p_m132_f<0x97, "vfmsubadd132", X86Fmsubadd>;
defm VFNMADD132   : avx512_fma3p_m132_f<0x9C, "vfnmadd132", X86Fnmadd>;
defm VFNMSUB132   : avx512_fma3p_m132_f<0x9E, "vfnmsub132", X86Fnmsub>;


// Scalar FMA
let Constraints = "$src1 = $dst" in {
multiclass avx512_fma3s_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                 RegisterClass RC, ValueType OpVT,
                 X86MemOperand x86memop, Operand memop,
                 PatFrag mem_frag> {
  let isCommutable = 1 in
  def r     : AVX512FMA3<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1, RC:$src3)))]>;
  let mayLoad = 1 in
  def m     : AVX512FMA3<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, f128mem:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1,
                            (mem_frag addr:$src3))))]>;
}

} // Constraints = "$src1 = $dst"

defm VFMADDSSZ  : avx512_fma3s_rm<0xA9, "vfmadd213ss", X86Fmadd, FR32X,
                      f32, f32mem, ssmem, loadf32>, EVEX_CD8<32, CD8VT1>;
defm VFMADDSDZ  : avx512_fma3s_rm<0xA9, "vfmadd213sd", X86Fmadd, FR64X,
                      f64, f64mem, sdmem, loadf64>, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VFMSUBSSZ  : avx512_fma3s_rm<0xAB, "vfmsub213ss", X86Fmsub, FR32X,
                      f32, f32mem, ssmem, loadf32>, EVEX_CD8<32, CD8VT1>;
defm VFMSUBSDZ  : avx512_fma3s_rm<0xAB, "vfmsub213sd", X86Fmsub, FR64X,
                      f64, f64mem, sdmem, loadf64>, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VFNMADDSSZ  : avx512_fma3s_rm<0xAD, "vfnmadd213ss", X86Fnmadd, FR32X,
                      f32, f32mem, ssmem, loadf32>, EVEX_CD8<32, CD8VT1>;
defm VFNMADDSDZ  : avx512_fma3s_rm<0xAD, "vfnmadd213sd", X86Fnmadd, FR64X,
                      f64, f64mem, sdmem, loadf64>, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VFNMSUBSSZ  : avx512_fma3s_rm<0xAF, "vfnmsub213ss", X86Fnmsub, FR32X,
                      f32, f32mem, ssmem, loadf32>, EVEX_CD8<32, CD8VT1>;
defm VFNMSUBSDZ  : avx512_fma3s_rm<0xAF, "vfnmsub213sd", X86Fnmsub, FR64X,
                      f64, f64mem, sdmem, loadf64>, VEX_W, EVEX_CD8<64, CD8VT1>;

//===----------------------------------------------------------------------===//
// AVX-512  Scalar convert from sign integer to float/double
//===----------------------------------------------------------------------===//

multiclass avx512_vcvtsi<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                          X86MemOperand x86memop, string asm> {
let hasSideEffects = 0 in {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins DstRC:$src1, SrcRC:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
              EVEX_4V;
  let mayLoad = 1 in
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins DstRC:$src1, x86memop:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
              EVEX_4V;
} // hasSideEffects = 0
}
let Predicates = [HasAVX512] in {
defm VCVTSI2SSZ   : avx512_vcvtsi<0x2A, GR32, FR32X, i32mem, "cvtsi2ss{l}">,
                                  XS, VEX_LIG, EVEX_CD8<32, CD8VT1>;
defm VCVTSI642SSZ : avx512_vcvtsi<0x2A, GR64, FR32X, i64mem, "cvtsi2ss{q}">,
                                  XS, VEX_W, VEX_LIG, EVEX_CD8<64, CD8VT1>;
defm VCVTSI2SDZ   : avx512_vcvtsi<0x2A, GR32, FR64X, i32mem, "cvtsi2sd{l}">,
                                  XD, VEX_LIG, EVEX_CD8<32, CD8VT1>;
defm VCVTSI642SDZ : avx512_vcvtsi<0x2A, GR64, FR64X, i64mem, "cvtsi2sd{q}">,
                                  XD, VEX_W, VEX_LIG, EVEX_CD8<64, CD8VT1>;

def : Pat<(f32 (sint_to_fp (loadi32 addr:$src))),
          (VCVTSI2SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f32 (sint_to_fp (loadi64 addr:$src))),
          (VCVTSI642SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (sint_to_fp (loadi32 addr:$src))),
          (VCVTSI2SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (sint_to_fp (loadi64 addr:$src))),
          (VCVTSI642SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;

def : Pat<(f32 (sint_to_fp GR32:$src)),
          (VCVTSI2SSZrr (f32 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f32 (sint_to_fp GR64:$src)),
          (VCVTSI642SSZrr (f32 (IMPLICIT_DEF)), GR64:$src)>;
def : Pat<(f64 (sint_to_fp GR32:$src)),
          (VCVTSI2SDZrr (f64 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f64 (sint_to_fp GR64:$src)),
          (VCVTSI642SDZrr (f64 (IMPLICIT_DEF)), GR64:$src)>;

defm VCVTUSI2SSZ   : avx512_vcvtsi<0x7B, GR32, FR32X, i32mem, "cvtusi2ss{l}">,
                                  XS, VEX_LIG, EVEX_CD8<32, CD8VT1>;
defm VCVTUSI642SSZ : avx512_vcvtsi<0x7B, GR64, FR32X, i64mem, "cvtusi2ss{q}">,
                                  XS, VEX_W, VEX_LIG, EVEX_CD8<64, CD8VT1>;
defm VCVTUSI2SDZ   : avx512_vcvtsi<0x7B, GR32, FR64X, i32mem, "cvtusi2sd{l}">,
                                  XD, VEX_LIG, EVEX_CD8<32, CD8VT1>;
defm VCVTUSI642SDZ : avx512_vcvtsi<0x7B, GR64, FR64X, i64mem, "cvtusi2sd{q}">,
                                  XD, VEX_W, VEX_LIG, EVEX_CD8<64, CD8VT1>;

def : Pat<(f32 (uint_to_fp (loadi32 addr:$src))),
          (VCVTUSI2SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f32 (uint_to_fp (loadi64 addr:$src))),
          (VCVTUSI642SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (uint_to_fp (loadi32 addr:$src))),
          (VCVTUSI2SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (uint_to_fp (loadi64 addr:$src))),
          (VCVTUSI642SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;

def : Pat<(f32 (uint_to_fp GR32:$src)),
          (VCVTUSI2SSZrr (f32 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f32 (uint_to_fp GR64:$src)),
          (VCVTUSI642SSZrr (f32 (IMPLICIT_DEF)), GR64:$src)>;
def : Pat<(f64 (uint_to_fp GR32:$src)),
          (VCVTUSI2SDZrr (f64 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f64 (uint_to_fp GR64:$src)),
          (VCVTUSI642SDZrr (f64 (IMPLICIT_DEF)), GR64:$src)>;
}

//===----------------------------------------------------------------------===//
// AVX-512  Scalar convert from float/double to integer
//===----------------------------------------------------------------------===//
multiclass avx512_cvt_s_int<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                          Intrinsic Int, Operand memop, ComplexPattern mem_cpat,
                          string asm> {
let hasSideEffects = 0 in {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (Int SrcRC:$src))]>, EVEX, VEX_LIG,
              Requires<[HasAVX512]>;
  let mayLoad = 1 in
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins memop:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"), []>, EVEX, VEX_LIG,
              Requires<[HasAVX512]>;
} // hasSideEffects = 0
}
let Predicates = [HasAVX512] in {
// Convert float/double to signed/unsigned int 32/64
defm VCVTSS2SIZ:    avx512_cvt_s_int<0x2D, VR128X, GR32, int_x86_sse_cvtss2si,
                                   ssmem, sse_load_f32, "cvtss2si">,
                                   XS, EVEX_CD8<32, CD8VT1>;
defm VCVTSS2SI64Z:  avx512_cvt_s_int<0x2D, VR128X, GR64, int_x86_sse_cvtss2si64,
                                   ssmem, sse_load_f32, "cvtss2si">,
                                   XS, VEX_W, EVEX_CD8<32, CD8VT1>;
defm VCVTSS2USIZ:   avx512_cvt_s_int<0x79, VR128X, GR32, int_x86_avx512_cvtss2usi,
                                   ssmem, sse_load_f32, "cvtss2usi">,
                                   XS, EVEX_CD8<32, CD8VT1>;
defm VCVTSS2USI64Z: avx512_cvt_s_int<0x79, VR128X, GR64,
                                   int_x86_avx512_cvtss2usi64, ssmem,
                                   sse_load_f32, "cvtss2usi">, XS, VEX_W,
                                   EVEX_CD8<32, CD8VT1>;
defm VCVTSD2SIZ:    avx512_cvt_s_int<0x2D, VR128X, GR32, int_x86_sse2_cvtsd2si,
                                   sdmem, sse_load_f64, "cvtsd2si">,
                                   XD, EVEX_CD8<64, CD8VT1>;
defm VCVTSD2SI64Z:  avx512_cvt_s_int<0x2D, VR128X, GR64, int_x86_sse2_cvtsd2si64,
                                   sdmem, sse_load_f64, "cvtsd2si">,
                                   XD, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VCVTSD2USIZ:   avx512_cvt_s_int<0x79, VR128X, GR32, int_x86_avx512_cvtsd2usi,
                                   sdmem, sse_load_f64, "cvtsd2usi">,
                                   XD, EVEX_CD8<64, CD8VT1>;
defm VCVTSD2USI64Z: avx512_cvt_s_int<0x79, VR128X, GR64,
                                   int_x86_avx512_cvtsd2usi64, sdmem,
                                   sse_load_f64, "cvtsd2usi">, XD, VEX_W,
                                   EVEX_CD8<64, CD8VT1>;

let isCodeGenOnly = 1 in {
  defm Int_VCVTSI2SSZ : sse12_cvt_sint_3addr<0x2A, GR32, VR128X,
            int_x86_sse_cvtsi2ss, i32mem, loadi32, "cvtsi2ss{l}",
            SSE_CVT_Scalar, 0>, XS, EVEX_4V;
  defm Int_VCVTSI2SS64Z : sse12_cvt_sint_3addr<0x2A, GR64, VR128X,
            int_x86_sse_cvtsi642ss, i64mem, loadi64, "cvtsi2ss{q}",
            SSE_CVT_Scalar, 0>, XS, EVEX_4V, VEX_W;
  defm Int_VCVTSI2SDZ : sse12_cvt_sint_3addr<0x2A, GR32, VR128X,
            int_x86_sse2_cvtsi2sd, i32mem, loadi32, "cvtsi2sd{l}",
            SSE_CVT_Scalar, 0>, XD, EVEX_4V;
  defm Int_VCVTSI2SD64Z : sse12_cvt_sint_3addr<0x2A, GR64, VR128X,
            int_x86_sse2_cvtsi642sd, i64mem, loadi64, "cvtsi2sd{q}",
            SSE_CVT_Scalar, 0>, XD, EVEX_4V, VEX_W;

  defm Int_VCVTUSI2SSZ : sse12_cvt_sint_3addr<0x2A, GR32, VR128X,
            int_x86_avx512_cvtusi2ss, i32mem, loadi32, "cvtusi2ss{l}",
            SSE_CVT_Scalar, 0>, XS, EVEX_4V;
  defm Int_VCVTUSI2SS64Z : sse12_cvt_sint_3addr<0x2A, GR64, VR128X,
            int_x86_avx512_cvtusi642ss, i64mem, loadi64, "cvtusi2ss{q}",
            SSE_CVT_Scalar, 0>, XS, EVEX_4V, VEX_W;
  defm Int_VCVTUSI2SDZ : sse12_cvt_sint_3addr<0x2A, GR32, VR128X,
            int_x86_avx512_cvtusi2sd, i32mem, loadi32, "cvtusi2sd{l}",
            SSE_CVT_Scalar, 0>, XD, EVEX_4V;
  defm Int_VCVTUSI2SD64Z : sse12_cvt_sint_3addr<0x2A, GR64, VR128X,
            int_x86_avx512_cvtusi642sd, i64mem, loadi64, "cvtusi2sd{q}",
            SSE_CVT_Scalar, 0>, XD, EVEX_4V, VEX_W;
} // isCodeGenOnly = 1

// Convert float/double to signed/unsigned int 32/64 with truncation
let isCodeGenOnly = 1 in {
  defm Int_VCVTTSS2SIZ : avx512_cvt_s_int<0x2C, VR128X, GR32, int_x86_sse_cvttss2si,
                                     ssmem, sse_load_f32, "cvttss2si">,
                                     XS, EVEX_CD8<32, CD8VT1>;
  defm Int_VCVTTSS2SI64Z : avx512_cvt_s_int<0x2C, VR128X, GR64,
                                     int_x86_sse_cvttss2si64, ssmem, sse_load_f32,
                                     "cvttss2si">, XS, VEX_W,
                                     EVEX_CD8<32, CD8VT1>;
  defm Int_VCVTTSD2SIZ : avx512_cvt_s_int<0x2C, VR128X, GR32, int_x86_sse2_cvttsd2si,
                                     sdmem, sse_load_f64, "cvttsd2si">, XD,
                                     EVEX_CD8<64, CD8VT1>;
  defm Int_VCVTTSD2SI64Z : avx512_cvt_s_int<0x2C, VR128X, GR64,
                                     int_x86_sse2_cvttsd2si64, sdmem, sse_load_f64,
                                     "cvttsd2si">, XD, VEX_W,
                                     EVEX_CD8<64, CD8VT1>;
  defm Int_VCVTTSS2USIZ : avx512_cvt_s_int<0x78, VR128X, GR32,
                                     int_x86_avx512_cvttss2usi, ssmem, sse_load_f32,
                                     "cvttss2usi">, XS, EVEX_CD8<32, CD8VT1>;
  defm Int_VCVTTSS2USI64Z : avx512_cvt_s_int<0x78, VR128X, GR64,
                                     int_x86_avx512_cvttss2usi64, ssmem,
                                     sse_load_f32, "cvttss2usi">, XS, VEX_W,
                                     EVEX_CD8<32, CD8VT1>;
  defm Int_VCVTTSD2USIZ : avx512_cvt_s_int<0x78, VR128X, GR32,
                                     int_x86_avx512_cvttsd2usi,
                                     sdmem, sse_load_f64, "cvttsd2usi">, XD,
                                     EVEX_CD8<64, CD8VT1>;
  defm Int_VCVTTSD2USI64Z : avx512_cvt_s_int<0x78, VR128X, GR64,
                                     int_x86_avx512_cvttsd2usi64, sdmem,
                                     sse_load_f64, "cvttsd2usi">, XD, VEX_W,
                                     EVEX_CD8<64, CD8VT1>;
} // isCodeGenOnly = 1

multiclass avx512_cvt_s<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                         SDNode OpNode, X86MemOperand x86memop, PatFrag ld_frag,
                         string asm> {
  def rr : SI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (OpNode SrcRC:$src))]>, EVEX;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (OpNode (ld_frag addr:$src)))]>, EVEX;
}

defm VCVTTSS2SIZ    : avx512_cvt_s<0x2C, FR32X, GR32, fp_to_sint, f32mem,
                                  loadf32, "cvttss2si">, XS,
                                  EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2USIZ   : avx512_cvt_s<0x78, FR32X, GR32, fp_to_uint, f32mem,
                                  loadf32, "cvttss2usi">, XS,
                                  EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2SI64Z  : avx512_cvt_s<0x2C, FR32X, GR64, fp_to_sint, f32mem,
                                  loadf32, "cvttss2si">, XS, VEX_W,
                                  EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2USI64Z : avx512_cvt_s<0x78, FR32X, GR64, fp_to_uint, f32mem,
                                  loadf32, "cvttss2usi">, XS, VEX_W,
                                  EVEX_CD8<32, CD8VT1>;
defm VCVTTSD2SIZ    : avx512_cvt_s<0x2C, FR64X, GR32, fp_to_sint, f64mem,
                                  loadf64, "cvttsd2si">, XD,
                                  EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2USIZ   : avx512_cvt_s<0x78, FR64X, GR32, fp_to_uint, f64mem,
                                  loadf64, "cvttsd2usi">, XD,
                                  EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2SI64Z  : avx512_cvt_s<0x2C, FR64X, GR64, fp_to_sint, f64mem,
                                  loadf64, "cvttsd2si">, XD, VEX_W,
                                  EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2USI64Z : avx512_cvt_s<0x78, FR64X, GR64, fp_to_uint, f64mem,
                                  loadf64, "cvttsd2usi">, XD, VEX_W,
                                  EVEX_CD8<64, CD8VT1>;
} // HasAVX512
//===----------------------------------------------------------------------===//
// AVX-512  Convert form float to double and back
//===----------------------------------------------------------------------===//
let hasSideEffects = 0 in {
def VCVTSS2SDZrr : AVX512XSI<0x5A, MRMSrcReg, (outs FR64X:$dst),
                    (ins FR32X:$src1, FR32X:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    []>, EVEX_4V, VEX_LIG, Sched<[WriteCvtF2F]>;
let mayLoad = 1 in
def VCVTSS2SDZrm : AVX512XSI<0x5A, MRMSrcMem, (outs FR64X:$dst),
                    (ins FR32X:$src1, f32mem:$src2),
                    "vcvtss2sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    []>, EVEX_4V, VEX_LIG, Sched<[WriteCvtF2FLd, ReadAfterLd]>,
                    EVEX_CD8<32, CD8VT1>;

// Convert scalar double to scalar single
def VCVTSD2SSZrr  : AVX512XDI<0x5A, MRMSrcReg, (outs FR32X:$dst),
                      (ins FR64X:$src1, FR64X:$src2),
                      "vcvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      []>, EVEX_4V, VEX_LIG, VEX_W, Sched<[WriteCvtF2F]>;
let mayLoad = 1 in
def VCVTSD2SSZrm  : AVX512XDI<0x5A, MRMSrcMem, (outs FR32X:$dst),
                      (ins FR64X:$src1, f64mem:$src2),
                      "vcvtsd2ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      []>, EVEX_4V, VEX_LIG, VEX_W,
                      Sched<[WriteCvtF2FLd, ReadAfterLd]>, EVEX_CD8<64, CD8VT1>;
}

def : Pat<(f64 (fextend FR32X:$src)), (VCVTSS2SDZrr FR32X:$src, FR32X:$src)>,
      Requires<[HasAVX512]>;
def : Pat<(fextend (loadf32 addr:$src)),
    (VCVTSS2SDZrm (f32 (IMPLICIT_DEF)), addr:$src)>, Requires<[HasAVX512]>;

def : Pat<(extloadf32 addr:$src),
    (VCVTSS2SDZrm (f32 (IMPLICIT_DEF)), addr:$src)>,
      Requires<[HasAVX512, OptForSize]>;

def : Pat<(extloadf32 addr:$src),
    (VCVTSS2SDZrr (f32 (IMPLICIT_DEF)), (VMOVSSZrm addr:$src))>,
    Requires<[HasAVX512, OptForSpeed]>;

def : Pat<(f32 (fround FR64X:$src)), (VCVTSD2SSZrr FR64X:$src, FR64X:$src)>,
           Requires<[HasAVX512]>;

multiclass avx512_vcvt_fp_with_rc<bits<8> opc, string asm, RegisterClass SrcRC,
               RegisterClass DstRC, SDNode OpNode, PatFrag mem_frag,
               X86MemOperand x86memop, ValueType OpVT, ValueType InVT,
               Domain d> {
let hasSideEffects = 0 in {
  def rr : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT SrcRC:$src))))], d>, EVEX;
  def rrb : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src, AVX512RC:$rc),
              !strconcat(asm,"\t{$rc, $src, $dst|$dst, $src, $rc}"),
              [], d>, EVEX, EVEX_B, EVEX_RC;
  let mayLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT (bitconvert (mem_frag addr:$src))))))], d>, EVEX;
} // hasSideEffects = 0
}

multiclass avx512_vcvt_fp<bits<8> opc, string asm, RegisterClass SrcRC,
               RegisterClass DstRC, SDNode OpNode, PatFrag mem_frag,
               X86MemOperand x86memop, ValueType OpVT, ValueType InVT,
               Domain d> {
let hasSideEffects = 0 in {
  def rr : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT SrcRC:$src))))], d>, EVEX;
  let mayLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT (bitconvert (mem_frag addr:$src))))))], d>, EVEX;
} // hasSideEffects = 0
}

defm VCVTPD2PSZ : avx512_vcvt_fp_with_rc<0x5A, "vcvtpd2ps", VR512, VR256X, fround,
                                loadv8f64, f512mem, v8f32, v8f64,
                                SSEPackedSingle>, EVEX_V512, VEX_W, PD,
                                EVEX_CD8<64, CD8VF>;

defm VCVTPS2PDZ : avx512_vcvt_fp<0x5A, "vcvtps2pd", VR256X, VR512, fextend,
                                loadv4f64, f256mem, v8f64, v8f32,
                                SSEPackedDouble>, EVEX_V512, PS,
                                EVEX_CD8<32, CD8VH>;
def : Pat<(v8f64 (extloadv8f32 addr:$src)),
            (VCVTPS2PDZrm addr:$src)>;

def : Pat<(v8f32 (int_x86_avx512_mask_cvtpd2ps_512 (v8f64 VR512:$src),
                   (bc_v8f32(v8i32 immAllZerosV)), (i8 -1), (i32 FROUND_CURRENT))),
          (VCVTPD2PSZrr VR512:$src)>;

def : Pat<(v8f32 (int_x86_avx512_mask_cvtpd2ps_512 (v8f64 VR512:$src),
                   (bc_v8f32(v8i32 immAllZerosV)), (i8 -1), imm:$rc)),
          (VCVTPD2PSZrrb VR512:$src, imm:$rc)>;

//===----------------------------------------------------------------------===//
// AVX-512  Vector convert from sign integer to float/double
//===----------------------------------------------------------------------===//

defm VCVTDQ2PSZ : avx512_vcvt_fp_with_rc<0x5B, "vcvtdq2ps", VR512, VR512, sint_to_fp,
                                loadv8i64, i512mem, v16f32, v16i32,
                                SSEPackedSingle>, EVEX_V512, PS,
                                EVEX_CD8<32, CD8VF>;

defm VCVTDQ2PDZ : avx512_vcvt_fp<0xE6, "vcvtdq2pd", VR256X, VR512, sint_to_fp,
                                loadv4i64, i256mem, v8f64, v8i32,
                                SSEPackedDouble>, EVEX_V512, XS,
                                EVEX_CD8<32, CD8VH>;

defm VCVTTPS2DQZ : avx512_vcvt_fp<0x5B, "vcvttps2dq", VR512, VR512, fp_to_sint,
                                 loadv16f32, f512mem, v16i32, v16f32,
                                 SSEPackedSingle>, EVEX_V512, XS,
                                 EVEX_CD8<32, CD8VF>;

defm VCVTTPD2DQZ : avx512_vcvt_fp<0xE6, "vcvttpd2dq", VR512, VR256X, fp_to_sint,
                                 loadv8f64, f512mem, v8i32, v8f64,
                                 SSEPackedDouble>, EVEX_V512, PD, VEX_W,
                                 EVEX_CD8<64, CD8VF>;

defm VCVTTPS2UDQZ : avx512_vcvt_fp<0x78, "vcvttps2udq", VR512, VR512, fp_to_uint,
                                 loadv16f32, f512mem, v16i32, v16f32,
                                 SSEPackedSingle>, EVEX_V512, PS,
                                 EVEX_CD8<32, CD8VF>;

// cvttps2udq (src, 0, mask-all-ones, sae-current)
def : Pat<(v16i32 (int_x86_avx512_mask_cvttps2udq_512 (v16f32 VR512:$src),
                   (v16i32 immAllZerosV), (i16 -1), FROUND_CURRENT)),
          (VCVTTPS2UDQZrr VR512:$src)>;

defm VCVTTPD2UDQZ : avx512_vcvt_fp<0x78, "vcvttpd2udq", VR512, VR256X, fp_to_uint,
                                 loadv8f64, f512mem, v8i32, v8f64,
                                 SSEPackedDouble>, EVEX_V512, PS, VEX_W,
                                 EVEX_CD8<64, CD8VF>;

// cvttpd2udq (src, 0, mask-all-ones, sae-current)
def : Pat<(v8i32 (int_x86_avx512_mask_cvttpd2udq_512 (v8f64 VR512:$src),
                   (v8i32 immAllZerosV), (i8 -1), FROUND_CURRENT)),
          (VCVTTPD2UDQZrr VR512:$src)>;

defm VCVTUDQ2PDZ : avx512_vcvt_fp<0x7A, "vcvtudq2pd", VR256X, VR512, uint_to_fp,
                                 loadv4i64, f256mem, v8f64, v8i32,
                                 SSEPackedDouble>, EVEX_V512, XS,
                                 EVEX_CD8<32, CD8VH>;

defm VCVTUDQ2PSZ : avx512_vcvt_fp_with_rc<0x7A, "vcvtudq2ps", VR512, VR512, uint_to_fp,
                                 loadv16i32, f512mem, v16f32, v16i32,
                                 SSEPackedSingle>, EVEX_V512, XD,
                                 EVEX_CD8<32, CD8VF>;

def : Pat<(v8i32 (fp_to_uint (v8f32 VR256X:$src1))),
          (EXTRACT_SUBREG (v16i32 (VCVTTPS2UDQZrr
           (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v4i32 (fp_to_uint (v4f32 VR128X:$src1))),
          (EXTRACT_SUBREG (v16i32 (VCVTTPS2UDQZrr
           (v16f32 (SUBREG_TO_REG (i32 0), VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v8f32 (uint_to_fp (v8i32 VR256X:$src1))),
          (EXTRACT_SUBREG (v16f32 (VCVTUDQ2PSZrr
           (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v4f32 (uint_to_fp (v4i32 VR128X:$src1))),
          (EXTRACT_SUBREG (v16f32 (VCVTUDQ2PSZrr
           (v16i32 (SUBREG_TO_REG (i32 0), VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4f64 (uint_to_fp (v4i32 VR128X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTUDQ2PDZrr
           (v8i32 (SUBREG_TO_REG (i32 0), VR128X:$src1, sub_xmm)))), sub_ymm)>;

def : Pat<(v16f32 (int_x86_avx512_mask_cvtdq2ps_512 (v16i32 VR512:$src),
                   (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1), imm:$rc)),
          (VCVTDQ2PSZrrb VR512:$src, imm:$rc)>;
def : Pat<(v8f64 (int_x86_avx512_mask_cvtdq2pd_512 (v8i32 VR256X:$src),
                   (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1))),
          (VCVTDQ2PDZrr VR256X:$src)>;
def : Pat<(v16f32 (int_x86_avx512_mask_cvtudq2ps_512 (v16i32 VR512:$src),
                   (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1), imm:$rc)),
          (VCVTUDQ2PSZrrb VR512:$src, imm:$rc)>;
def : Pat<(v8f64 (int_x86_avx512_mask_cvtudq2pd_512 (v8i32 VR256X:$src),
                   (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1))),
          (VCVTUDQ2PDZrr VR256X:$src)>;

multiclass avx512_vcvt_fp2int<bits<8> opc, string asm, RegisterClass SrcRC,
               RegisterClass DstRC, PatFrag mem_frag,
               X86MemOperand x86memop, Domain d> {
let hasSideEffects = 0 in {
  def rr : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [], d>, EVEX;
  def rrb : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src, AVX512RC:$rc),
              !strconcat(asm,"\t{$rc, $src, $dst|$dst, $src, $rc}"),
              [], d>, EVEX, EVEX_B, EVEX_RC;
  let mayLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [], d>, EVEX;
} // hasSideEffects = 0
}

defm VCVTPS2DQZ : avx512_vcvt_fp2int<0x5B, "vcvtps2dq", VR512, VR512,
                                 loadv16f32, f512mem, SSEPackedSingle>, PD,
                                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VCVTPD2DQZ : avx512_vcvt_fp2int<0xE6, "vcvtpd2dq", VR512, VR256X,
                                 loadv8f64, f512mem, SSEPackedDouble>, XD, VEX_W,
                                 EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat <(v16i32 (int_x86_avx512_mask_cvtps2dq_512 (v16f32 VR512:$src),
                    (v16i32 immAllZerosV), (i16 -1), imm:$rc)),
           (VCVTPS2DQZrrb VR512:$src, imm:$rc)>;

def : Pat <(v8i32 (int_x86_avx512_mask_cvtpd2dq_512 (v8f64 VR512:$src),
                    (v8i32 immAllZerosV), (i8 -1), imm:$rc)),
           (VCVTPD2DQZrrb VR512:$src, imm:$rc)>;

defm VCVTPS2UDQZ : avx512_vcvt_fp2int<0x79, "vcvtps2udq", VR512, VR512,
                                 loadv16f32, f512mem, SSEPackedSingle>,
                                 PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VCVTPD2UDQZ : avx512_vcvt_fp2int<0x79, "vcvtpd2udq", VR512, VR256X,
                                 loadv8f64, f512mem, SSEPackedDouble>, VEX_W,
                                 PS, EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat <(v16i32 (int_x86_avx512_mask_cvtps2udq_512 (v16f32 VR512:$src),
                    (v16i32 immAllZerosV), (i16 -1), imm:$rc)),
           (VCVTPS2UDQZrrb VR512:$src, imm:$rc)>;

def : Pat <(v8i32 (int_x86_avx512_mask_cvtpd2udq_512 (v8f64 VR512:$src),
                    (v8i32 immAllZerosV), (i8 -1), imm:$rc)),
           (VCVTPD2UDQZrrb VR512:$src, imm:$rc)>;

let Predicates = [HasAVX512] in {
  def : Pat<(v8f32 (fround (loadv8f64 addr:$src))),
            (VCVTPD2PSZrm addr:$src)>;
  def : Pat<(v8f64 (extloadv8f32 addr:$src)),
            (VCVTPS2PDZrm addr:$src)>;
}

//===----------------------------------------------------------------------===//
// Half precision conversion instructions
//===----------------------------------------------------------------------===//
multiclass avx512_cvtph2ps<RegisterClass destRC, RegisterClass srcRC,
                             X86MemOperand x86memop> {
  def rr : AVX5128I<0x13, MRMSrcReg, (outs destRC:$dst), (ins srcRC:$src),
             "vcvtph2ps\t{$src, $dst|$dst, $src}",
             []>, EVEX;
  let hasSideEffects = 0, mayLoad = 1 in
  def rm : AVX5128I<0x13, MRMSrcMem, (outs destRC:$dst), (ins x86memop:$src),
             "vcvtph2ps\t{$src, $dst|$dst, $src}", []>, EVEX;
}

multiclass avx512_cvtps2ph<RegisterClass destRC, RegisterClass srcRC,
                             X86MemOperand x86memop> {
  def rr : AVX512AIi8<0x1D, MRMDestReg, (outs destRC:$dst),
               (ins srcRC:$src1, i32u8imm:$src2),
               "vcvtps2ph\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               []>, EVEX;
  let hasSideEffects = 0, mayStore = 1 in
  def mr : AVX512AIi8<0x1D, MRMDestMem, (outs),
               (ins x86memop:$dst, srcRC:$src1, i32u8imm:$src2),
               "vcvtps2ph\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>, EVEX;
}

defm VCVTPH2PSZ : avx512_cvtph2ps<VR512, VR256X, f256mem>, EVEX_V512,
                                    EVEX_CD8<32, CD8VH>;
defm VCVTPS2PHZ : avx512_cvtps2ph<VR256X, VR512, f256mem>, EVEX_V512,
                                    EVEX_CD8<32, CD8VH>;

def : Pat<(v16i16 (int_x86_avx512_mask_vcvtps2ph_512 (v16f32 VR512:$src),
           imm:$rc, (bc_v16i16(v8i32 immAllZerosV)), (i16 -1))),
           (VCVTPS2PHZrr VR512:$src, imm:$rc)>;

def : Pat<(v16f32 (int_x86_avx512_mask_vcvtph2ps_512 (v16i16 VR256X:$src),
           (bc_v16f32(v16i32 immAllZerosV)), (i16 -1), (i32 FROUND_CURRENT))),
           (VCVTPH2PSZrr VR256X:$src)>;

let Defs = [EFLAGS], Predicates = [HasAVX512] in {
  defm VUCOMISSZ : sse12_ord_cmp<0x2E, FR32X, X86cmp, f32, f32mem, loadf32,
                                 "ucomiss">, PS, EVEX, VEX_LIG,
                                 EVEX_CD8<32, CD8VT1>;
  defm VUCOMISDZ : sse12_ord_cmp<0x2E, FR64X, X86cmp, f64, f64mem, loadf64,
                                  "ucomisd">, PD, EVEX,
                                  VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;
  let Pattern = []<dag> in {
    defm VCOMISSZ  : sse12_ord_cmp<0x2F, VR128X, undef, v4f32, f128mem, load,
                                   "comiss">, PS, EVEX, VEX_LIG,
                                   EVEX_CD8<32, CD8VT1>;
    defm VCOMISDZ  : sse12_ord_cmp<0x2F, VR128X, undef, v2f64, f128mem, load,
                                   "comisd">, PD, EVEX,
                                    VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;
  }
  let isCodeGenOnly = 1 in {
    defm Int_VUCOMISSZ  : sse12_ord_cmp<0x2E, VR128X, X86ucomi, v4f32, f128mem,
                              load, "ucomiss">, PS, EVEX, VEX_LIG,
                              EVEX_CD8<32, CD8VT1>;
    defm Int_VUCOMISDZ  : sse12_ord_cmp<0x2E, VR128X, X86ucomi, v2f64, f128mem,
                              load, "ucomisd">, PD, EVEX,
                              VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;

    defm Int_VCOMISSZ  : sse12_ord_cmp<0x2F, VR128X, X86comi, v4f32, f128mem,
                              load, "comiss">, PS, EVEX, VEX_LIG,
                              EVEX_CD8<32, CD8VT1>;
    defm Int_VCOMISDZ  : sse12_ord_cmp<0x2F, VR128X, X86comi, v2f64, f128mem,
                              load, "comisd">, PD, EVEX,
                              VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;
  }
}

/// avx512_fp14_s rcp14ss, rcp14sd, rsqrt14ss, rsqrt14sd
multiclass avx512_fp14_s<bits<8> opc, string OpcodeStr, RegisterClass RC,
                            X86MemOperand x86memop> {
  let hasSideEffects = 0 in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
               (ins RC:$src1, RC:$src2),
               !strconcat(OpcodeStr,
               "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>, EVEX_4V;
  let mayLoad = 1 in {
  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, x86memop:$src2),
               !strconcat(OpcodeStr,
               "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>, EVEX_4V;
  }
}
}

defm VRCP14SS   : avx512_fp14_s<0x4D, "vrcp14ss", FR32X, f32mem>,
                  EVEX_CD8<32, CD8VT1>;
defm VRCP14SD   : avx512_fp14_s<0x4D, "vrcp14sd", FR64X, f64mem>,
                  VEX_W, EVEX_CD8<64, CD8VT1>;
defm VRSQRT14SS   : avx512_fp14_s<0x4F, "vrsqrt14ss", FR32X, f32mem>,
                  EVEX_CD8<32, CD8VT1>;
defm VRSQRT14SD   : avx512_fp14_s<0x4F, "vrsqrt14sd", FR64X, f64mem>,
                  VEX_W, EVEX_CD8<64, CD8VT1>;

def : Pat <(v4f32 (int_x86_avx512_rcp14_ss (v4f32 VR128X:$src1),
              (v4f32 VR128X:$src2), (bc_v4f32 (v4i32 immAllZerosV)), (i8 -1))),
           (COPY_TO_REGCLASS (VRCP14SSrr (COPY_TO_REGCLASS VR128X:$src1, FR32X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR32X)), VR128X)>;

def : Pat <(v2f64 (int_x86_avx512_rcp14_sd (v2f64 VR128X:$src1),
              (v2f64 VR128X:$src2), (bc_v2f64 (v4i32 immAllZerosV)), (i8 -1))),
           (COPY_TO_REGCLASS (VRCP14SDrr (COPY_TO_REGCLASS VR128X:$src1, FR64X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR64X)), VR128X)>;

def : Pat <(v4f32 (int_x86_avx512_rsqrt14_ss (v4f32 VR128X:$src1),
              (v4f32 VR128X:$src2), (bc_v4f32 (v4i32 immAllZerosV)), (i8 -1))),
           (COPY_TO_REGCLASS (VRSQRT14SSrr (COPY_TO_REGCLASS VR128X:$src1, FR32X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR32X)), VR128X)>;

def : Pat <(v2f64 (int_x86_avx512_rsqrt14_sd (v2f64 VR128X:$src1),
              (v2f64 VR128X:$src2), (bc_v2f64 (v4i32 immAllZerosV)), (i8 -1))),
           (COPY_TO_REGCLASS (VRSQRT14SDrr (COPY_TO_REGCLASS VR128X:$src1, FR64X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR64X)), VR128X)>;

/// avx512_fp14_p rcp14ps, rcp14pd, rsqrt14ps, rsqrt14pd
multiclass avx512_fp14_p<bits<8> opc, string OpcodeStr, SDNode OpNode,
                         X86VectorVTInfo _> {
  defm r: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src), OpcodeStr, "$src", "$src",
                         (_.FloatVT (OpNode _.RC:$src))>, EVEX, T8PD;
  let mayLoad = 1 in {
    defm m: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                           (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                           (OpNode (_.FloatVT
                             (bitconvert (_.LdFrag addr:$src))))>, EVEX, T8PD;
    defm mb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                            (ins _.ScalarMemOp:$src), OpcodeStr,
                            "${src}"##_.BroadcastStr, "${src}"##_.BroadcastStr,
                            (OpNode (_.FloatVT
                              (X86VBroadcast (_.ScalarLdFrag addr:$src))))>,
                            EVEX, T8PD, EVEX_B;
  }
}

multiclass avx512_fp14_p_vl_all<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm PSZ : avx512_fp14_p<opc, !strconcat(OpcodeStr, "ps"), OpNode, v16f32_info>,
                          EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pd"), OpNode, v8f64_info>,
                          EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

  // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "ps"),
                                OpNode, v4f32x_info>,
                               EVEX_V128, EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "ps"),
                                OpNode, v8f32x_info>,
                               EVEX_V256, EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pd"),
                                OpNode, v2f64x_info>,
                               EVEX_V128, VEX_W, EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pd"),
                                OpNode, v4f64x_info>,
                               EVEX_V256, VEX_W, EVEX_CD8<64, CD8VF>;
  }
}

defm VRSQRT14 : avx512_fp14_p_vl_all<0x4E, "vrsqrt14", X86frsqrt>;
defm VRCP14 : avx512_fp14_p_vl_all<0x4C, "vrcp14", X86frcp>;

def : Pat <(v16f32 (int_x86_avx512_rsqrt14_ps_512 (v16f32 VR512:$src),
              (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1))),
           (VRSQRT14PSZr VR512:$src)>;
def : Pat <(v8f64 (int_x86_avx512_rsqrt14_pd_512 (v8f64 VR512:$src),
              (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1))),
           (VRSQRT14PDZr VR512:$src)>;

def : Pat <(v16f32 (int_x86_avx512_rcp14_ps_512 (v16f32 VR512:$src),
              (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1))),
           (VRCP14PSZr VR512:$src)>;
def : Pat <(v8f64 (int_x86_avx512_rcp14_pd_512 (v8f64 VR512:$src),
              (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1))),
           (VRCP14PDZr VR512:$src)>;

/// avx512_fp28_s rcp28ss, rcp28sd, rsqrt28ss, rsqrt28sd
multiclass avx512_fp28_s<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode OpNode> {

  defm r : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                           "$src2, $src1", "$src1, $src2",
                           (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                           (i32 FROUND_CURRENT))>;

  defm rb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                            (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                            "$src2, $src1", "$src1, $src2",
                            (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                            (i32 FROUND_NO_EXC)), "{sae}">, EVEX_B;

  defm m : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (OpNode (_.VT _.RC:$src1),
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))),
                         (i32 FROUND_CURRENT))>;
}

multiclass avx512_eri_s<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm SS : avx512_fp28_s<opc, OpcodeStr#"ss", f32x_info, OpNode>,
              EVEX_CD8<32, CD8VT1>;
  defm SD : avx512_fp28_s<opc, OpcodeStr#"sd", f64x_info, OpNode>,
              EVEX_CD8<64, CD8VT1>, VEX_W;
}

let hasSideEffects = 0, Predicates = [HasERI] in {
  defm VRCP28   : avx512_eri_s<0xCB, "vrcp28",   X86rcp28s>,   T8PD, EVEX_4V;
  defm VRSQRT28 : avx512_eri_s<0xCD, "vrsqrt28", X86rsqrt28s>, T8PD, EVEX_4V;
}
/// avx512_fp28_p rcp28ps, rcp28pd, rsqrt28ps, rsqrt28pd

multiclass avx512_fp28_p<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         SDNode OpNode> {

  defm r : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.VT _.RC:$src), (i32 FROUND_CURRENT))>;

  defm rb : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _.RC:$src), OpcodeStr,
                        "$src", "$src",
                        (OpNode (_.VT _.RC:$src), (i32 FROUND_NO_EXC)),
                        "{sae}">, EVEX_B;

  defm m : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.FloatVT
                             (bitconvert (_.LdFrag addr:$src))),
                          (i32 FROUND_CURRENT))>;

  defm mb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.FloatVT
                                  (X86VBroadcast (_.ScalarLdFrag addr:$src))),
                                 (i32 FROUND_CURRENT))>, EVEX_B;
}

multiclass  avx512_eri<bits<8> opc, string OpcodeStr, SDNode OpNode> {
   defm PS : avx512_fp28_p<opc, OpcodeStr#"ps", v16f32_info, OpNode>,
                     EVEX_CD8<32, CD8VF>;
   defm PD : avx512_fp28_p<opc, OpcodeStr#"pd", v8f64_info, OpNode>,
                     VEX_W, EVEX_CD8<32, CD8VF>;
}

let Predicates = [HasERI], hasSideEffects = 0 in {

 defm VRSQRT28 : avx512_eri<0xCC, "vrsqrt28", X86rsqrt28>, EVEX, EVEX_V512, T8PD;
 defm VRCP28   : avx512_eri<0xCA, "vrcp28",   X86rcp28>,   EVEX, EVEX_V512, T8PD;
 defm VEXP2    : avx512_eri<0xC8, "vexp2",    X86exp2>,    EVEX, EVEX_V512, T8PD;
}

multiclass avx512_sqrt_packed<bits<8> opc, string OpcodeStr,
                              SDNode OpNode, X86VectorVTInfo _>{
  defm r: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src), OpcodeStr, "$src", "$src",
                         (_.FloatVT (OpNode _.RC:$src))>, EVEX;
  let mayLoad = 1 in {
    defm m: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                           (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                           (OpNode (_.FloatVT
                             (bitconvert (_.LdFrag addr:$src))))>, EVEX;

    defm mb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                            (ins _.ScalarMemOp:$src), OpcodeStr,
                            "${src}"##_.BroadcastStr, "${src}"##_.BroadcastStr,
                            (OpNode (_.FloatVT
                              (X86VBroadcast (_.ScalarLdFrag addr:$src))))>,
                            EVEX, EVEX_B;
  }
}

multiclass avx512_sqrt_scalar<bits<8> opc, string OpcodeStr,
                          Intrinsic F32Int, Intrinsic F64Int,
                          OpndItins itins_s, OpndItins itins_d> {
  def SSZr : SI<opc, MRMSrcReg, (outs FR32X:$dst),
               (ins FR32X:$src1, FR32X:$src2),
               !strconcat(OpcodeStr,
                          "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      [], itins_s.rr>, XS, EVEX_4V;
  let isCodeGenOnly = 1 in
  def SSZr_Int : SIi8<opc, MRMSrcReg, (outs VR128X:$dst),
               (ins VR128X:$src1, VR128X:$src2),
               !strconcat(OpcodeStr,
                "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               [(set VR128X:$dst,
                 (F32Int VR128X:$src1, VR128X:$src2))],
               itins_s.rr>, XS, EVEX_4V;
  let mayLoad = 1 in {
  def SSZm : SI<opc, MRMSrcMem, (outs FR32X:$dst),
               (ins FR32X:$src1, f32mem:$src2),
               !strconcat(OpcodeStr,
                          "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                      [], itins_s.rm>, XS, EVEX_4V, EVEX_CD8<32, CD8VT1>;
  let isCodeGenOnly = 1 in
  def SSZm_Int : SIi8<opc, MRMSrcMem, (outs VR128X:$dst),
                   (ins VR128X:$src1, ssmem:$src2),
                   !strconcat(OpcodeStr,
                 "ss\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set VR128X:$dst,
                     (F32Int VR128X:$src1, sse_load_f32:$src2))],
                   itins_s.rm>, XS, EVEX_4V, EVEX_CD8<32, CD8VT1>;
  }
  def SDZr : SI<opc, MRMSrcReg, (outs FR64X:$dst),
               (ins FR64X:$src1, FR64X:$src2),
               !strconcat(OpcodeStr,
                          "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>,
                      XD, EVEX_4V, VEX_W;
  let isCodeGenOnly = 1 in
  def SDZr_Int : SIi8<opc, MRMSrcReg, (outs VR128X:$dst),
               (ins VR128X:$src1, VR128X:$src2),
               !strconcat(OpcodeStr,
                "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               [(set VR128X:$dst,
                 (F64Int VR128X:$src1, VR128X:$src2))],
               itins_s.rr>, XD, EVEX_4V, VEX_W;
  let mayLoad = 1 in {
  def SDZm : SI<opc, MRMSrcMem, (outs FR64X:$dst),
               (ins FR64X:$src1, f64mem:$src2),
               !strconcat(OpcodeStr,
                  "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>,
               XD, EVEX_4V, VEX_W, EVEX_CD8<64, CD8VT1>;
  let isCodeGenOnly = 1 in
  def SDZm_Int : SIi8<opc, MRMSrcMem, (outs VR128X:$dst),
                  (ins VR128X:$src1, sdmem:$src2),
                   !strconcat(OpcodeStr,
                  "sd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                  [(set VR128X:$dst,
                    (F64Int VR128X:$src1, sse_load_f64:$src2))]>,
                  XD, EVEX_4V, VEX_W, EVEX_CD8<64, CD8VT1>;
  }
}

multiclass avx512_sqrt_packed_all<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode> {
  defm PSZ : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode,
                                v16f32_info>,
                                EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode,
                                v8f64_info>,
                                EVEX_V512, VEX_W, PD, EVEX_CD8<64, CD8VF>;
  // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "ps"),
                                     OpNode, v4f32x_info>,
                                     EVEX_V128, PS, EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "ps"),
                                     OpNode, v8f32x_info>,
                                     EVEX_V256, PS, EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pd"),
                                     OpNode, v2f64x_info>,
                                     EVEX_V128, VEX_W, PD, EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pd"),
                                     OpNode, v4f64x_info>,
                                     EVEX_V256, VEX_W, PD, EVEX_CD8<64, CD8VF>;
  }
}

defm VSQRT : avx512_sqrt_packed_all<0x51, "vsqrt", fsqrt>;

defm VSQRT  : avx512_sqrt_scalar<0x51, "sqrt",
                int_x86_avx512_sqrt_ss, int_x86_avx512_sqrt_sd,
                SSE_SQRTSS, SSE_SQRTSD>;

let Predicates = [HasAVX512] in {
  def : Pat<(v16f32 (int_x86_avx512_sqrt_ps_512 (v16f32 VR512:$src1),
                    (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1), FROUND_CURRENT)),
                   (VSQRTPSZr VR512:$src1)>;
  def : Pat<(v8f64 (int_x86_avx512_sqrt_pd_512 (v8f64 VR512:$src1),
                    (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1), FROUND_CURRENT)),
                   (VSQRTPDZr VR512:$src1)>;

  def : Pat<(f32 (fsqrt FR32X:$src)),
            (VSQRTSSZr (f32 (IMPLICIT_DEF)), FR32X:$src)>;
  def : Pat<(f32 (fsqrt (load addr:$src))),
            (VSQRTSSZm (f32 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[OptForSize]>;
  def : Pat<(f64 (fsqrt FR64X:$src)),
            (VSQRTSDZr (f64 (IMPLICIT_DEF)), FR64X:$src)>;
  def : Pat<(f64 (fsqrt (load addr:$src))),
            (VSQRTSDZm (f64 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[OptForSize]>;

  def : Pat<(f32 (X86frsqrt FR32X:$src)),
            (VRSQRT14SSrr (f32 (IMPLICIT_DEF)), FR32X:$src)>;
  def : Pat<(f32 (X86frsqrt (load addr:$src))),
            (VRSQRT14SSrm (f32 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[OptForSize]>;

  def : Pat<(f32 (X86frcp FR32X:$src)),
            (VRCP14SSrr (f32 (IMPLICIT_DEF)), FR32X:$src)>;
  def : Pat<(f32 (X86frcp (load addr:$src))),
            (VRCP14SSrm (f32 (IMPLICIT_DEF)), addr:$src)>,
            Requires<[OptForSize]>;

  def : Pat<(int_x86_sse_sqrt_ss VR128X:$src),
            (COPY_TO_REGCLASS (VSQRTSSZr (f32 (IMPLICIT_DEF)),
                                        (COPY_TO_REGCLASS VR128X:$src, FR32)),
                              VR128X)>;
  def : Pat<(int_x86_sse_sqrt_ss sse_load_f32:$src),
            (VSQRTSSZm_Int (v4f32 (IMPLICIT_DEF)), sse_load_f32:$src)>;

  def : Pat<(int_x86_sse2_sqrt_sd VR128X:$src),
            (COPY_TO_REGCLASS (VSQRTSDZr (f64 (IMPLICIT_DEF)),
                                        (COPY_TO_REGCLASS VR128X:$src, FR64)),
                              VR128X)>;
  def : Pat<(int_x86_sse2_sqrt_sd sse_load_f64:$src),
            (VSQRTSDZm_Int (v2f64 (IMPLICIT_DEF)), sse_load_f64:$src)>;
}


multiclass avx512_rndscale<bits<8> opc, string OpcodeStr,
                            X86MemOperand x86memop, RegisterClass RC,
                            PatFrag mem_frag, Domain d> {
let ExeDomain = d in {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def r : AVX512AIi8<opc, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32u8imm:$src2),
                    !strconcat(OpcodeStr,
                    "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX;

  // Vector intrinsic operation, mem
  def m : AVX512AIi8<opc, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32u8imm:$src2),
                    !strconcat(OpcodeStr,
                    "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX;
} // ExeDomain
}

defm VRNDSCALEPSZ : avx512_rndscale<0x08, "vrndscaleps", f512mem, VR512,
                                loadv16f32, SSEPackedSingle>, EVEX_V512,
                                EVEX_CD8<32, CD8VF>;

def : Pat<(v16f32 (int_x86_avx512_mask_rndscale_ps_512 (v16f32 VR512:$src1),
                   imm:$src2, (v16f32 VR512:$src1), (i16 -1),
                   FROUND_CURRENT)),
                   (VRNDSCALEPSZr VR512:$src1, imm:$src2)>;


defm VRNDSCALEPDZ : avx512_rndscale<0x09, "vrndscalepd", f512mem, VR512,
                                loadv8f64, SSEPackedDouble>, EVEX_V512,
                                VEX_W, EVEX_CD8<64, CD8VF>;

def : Pat<(v8f64 (int_x86_avx512_mask_rndscale_pd_512 (v8f64 VR512:$src1),
                  imm:$src2, (v8f64 VR512:$src1), (i8 -1),
                  FROUND_CURRENT)),
                   (VRNDSCALEPDZr VR512:$src1, imm:$src2)>;

multiclass
avx512_rndscale_scalar<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {

  let ExeDomain = _.ExeDomain in {
  defm r : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3), OpcodeStr,
                           "$src3, $src2, $src1", "$src1, $src2, $src3",
                           (_.VT (X86RndScale (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                            (i32 imm:$src3), (i32 FROUND_CURRENT)))>;

  defm rb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3), OpcodeStr,
                         "$src3, $src2, $src1", "$src1, $src2, $src3",
                         (_.VT (X86RndScale (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                         (i32 imm:$src3), (i32 FROUND_NO_EXC))), "{sae}">, EVEX_B;

  let mayLoad = 1 in
  defm m : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.MemOp:$src2, i32u8imm:$src3), OpcodeStr,
                         "$src3, $src2, $src1", "$src1, $src2, $src3",
                         (_.VT (X86RndScale (_.VT _.RC:$src1),
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))),
                          (i32 imm:$src3), (i32 FROUND_CURRENT)))>;
  }
  let Predicates = [HasAVX512] in {
  def : Pat<(ffloor _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x1))), _.FRC)>;
  def : Pat<(fceil _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x2))), _.FRC)>;
  def : Pat<(ftrunc _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x3))), _.FRC)>;
  def : Pat<(frint _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x4))), _.FRC)>;
  def : Pat<(fnearbyint _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0xc))), _.FRC)>;

  def : Pat<(ffloor (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x1))), _.FRC)>;
  def : Pat<(fceil (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x2))), _.FRC)>;
  def : Pat<(ftrunc (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x3))), _.FRC)>;
  def : Pat<(frint (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x4))), _.FRC)>;
  def : Pat<(fnearbyint (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0xc))), _.FRC)>;
  }
}

defm VRNDSCALESS : avx512_rndscale_scalar<0x0A, "vrndscaless", f32x_info>,
                                AVX512AIi8Base, EVEX_4V, EVEX_CD8<32, CD8VT1>;

defm VRNDSCALESD : avx512_rndscale_scalar<0x0B, "vrndscalesd", f64x_info>, VEX_W,
                                AVX512AIi8Base, EVEX_4V, EVEX_CD8<64, CD8VT1>;

let Predicates = [HasAVX512] in {
def : Pat<(v16f32 (ffloor VR512:$src)),
          (VRNDSCALEPSZr VR512:$src, (i32 0x1))>;
def : Pat<(v16f32 (fnearbyint VR512:$src)),
          (VRNDSCALEPSZr VR512:$src, (i32 0xC))>;
def : Pat<(v16f32 (fceil VR512:$src)),
          (VRNDSCALEPSZr VR512:$src, (i32 0x2))>;
def : Pat<(v16f32 (frint VR512:$src)),
          (VRNDSCALEPSZr VR512:$src, (i32 0x4))>;
def : Pat<(v16f32 (ftrunc VR512:$src)),
          (VRNDSCALEPSZr VR512:$src, (i32 0x3))>;

def : Pat<(v8f64 (ffloor VR512:$src)),
          (VRNDSCALEPDZr VR512:$src, (i32 0x1))>;
def : Pat<(v8f64 (fnearbyint VR512:$src)),
          (VRNDSCALEPDZr VR512:$src, (i32 0xC))>;
def : Pat<(v8f64 (fceil VR512:$src)),
          (VRNDSCALEPDZr VR512:$src, (i32 0x2))>;
def : Pat<(v8f64 (frint VR512:$src)),
          (VRNDSCALEPDZr VR512:$src, (i32 0x4))>;
def : Pat<(v8f64 (ftrunc VR512:$src)),
          (VRNDSCALEPDZr VR512:$src, (i32 0x3))>;
}
//-------------------------------------------------
// Integer truncate and extend operations
//-------------------------------------------------

multiclass avx512_trunc_sat<bits<8> opc, string OpcodeStr,
                          RegisterClass dstRC, RegisterClass srcRC,
                          RegisterClass KRC, X86MemOperand x86memop> {
  def rr : AVX512XS8I<opc, MRMDestReg, (outs dstRC:$dst),
               (ins srcRC:$src),
               !strconcat(OpcodeStr,"\t{$src, $dst|$dst, $src}"),
               []>, EVEX;

  def rrk : AVX512XS8I<opc, MRMDestReg, (outs dstRC:$dst),
               (ins KRC:$mask, srcRC:$src),
               !strconcat(OpcodeStr,
                 "\t{$src, ${dst} {${mask}}|${dst} {${mask}}, $src}"),
               []>, EVEX, EVEX_K;

  def rrkz : AVX512XS8I<opc, MRMDestReg, (outs dstRC:$dst),
               (ins KRC:$mask, srcRC:$src),
               !strconcat(OpcodeStr,
                 "\t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
               []>, EVEX, EVEX_KZ;

  def mr : AVX512XS8I<opc, MRMDestMem, (outs), (ins x86memop:$dst, srcRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               []>, EVEX;

  def mrk : AVX512XS8I<opc, MRMDestMem, (outs),
               (ins x86memop:$dst, KRC:$mask, srcRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst {${mask}}|${dst} {${mask}}, $src}"),
               []>, EVEX, EVEX_K;

}
defm VPMOVQB    : avx512_trunc_sat<0x32, "vpmovqb",   VR128X, VR512, VK8WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<8, CD8VO>;
defm VPMOVSQB   : avx512_trunc_sat<0x22, "vpmovsqb",  VR128X, VR512, VK8WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<8, CD8VO>;
defm VPMOVUSQB  : avx512_trunc_sat<0x12, "vpmovusqb", VR128X, VR512, VK8WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<8, CD8VO>;
defm VPMOVQW    : avx512_trunc_sat<0x34, "vpmovqw",   VR128X, VR512, VK8WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<16, CD8VQ>;
defm VPMOVSQW   : avx512_trunc_sat<0x24, "vpmovsqw",  VR128X, VR512, VK8WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<16, CD8VQ>;
defm VPMOVUSQW  : avx512_trunc_sat<0x14, "vpmovusqw", VR128X, VR512, VK8WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<16, CD8VQ>;
defm VPMOVQD    : avx512_trunc_sat<0x35, "vpmovqd",   VR256X, VR512, VK8WM,
                                 i256mem>, EVEX_V512, EVEX_CD8<32, CD8VH>;
defm VPMOVSQD   : avx512_trunc_sat<0x25, "vpmovsqd",  VR256X, VR512, VK8WM,
                                 i256mem>, EVEX_V512, EVEX_CD8<32, CD8VH>;
defm VPMOVUSQD  : avx512_trunc_sat<0x15, "vpmovusqd", VR256X, VR512, VK8WM,
                                 i256mem>, EVEX_V512, EVEX_CD8<32, CD8VH>;
defm VPMOVDW    : avx512_trunc_sat<0x33, "vpmovdw",   VR256X, VR512, VK16WM,
                                 i256mem>, EVEX_V512, EVEX_CD8<16, CD8VH>;
defm VPMOVSDW   : avx512_trunc_sat<0x23, "vpmovsdw",  VR256X, VR512, VK16WM,
                                 i256mem>, EVEX_V512, EVEX_CD8<16, CD8VH>;
defm VPMOVUSDW  : avx512_trunc_sat<0x13, "vpmovusdw", VR256X, VR512, VK16WM,
                                 i256mem>, EVEX_V512, EVEX_CD8<16, CD8VH>;
defm VPMOVDB    : avx512_trunc_sat<0x31, "vpmovdb",   VR128X, VR512, VK16WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<8, CD8VQ>;
defm VPMOVSDB   : avx512_trunc_sat<0x21, "vpmovsdb",  VR128X, VR512, VK16WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<8, CD8VQ>;
defm VPMOVUSDB  : avx512_trunc_sat<0x11, "vpmovusdb", VR128X, VR512, VK16WM,
                                 i128mem>, EVEX_V512, EVEX_CD8<8, CD8VQ>;

def : Pat<(v16i8  (X86vtrunc (v8i64  VR512:$src))), (VPMOVQBrr  VR512:$src)>;
def : Pat<(v8i16  (X86vtrunc (v8i64  VR512:$src))), (VPMOVQWrr  VR512:$src)>;
def : Pat<(v16i16 (X86vtrunc (v16i32 VR512:$src))), (VPMOVDWrr  VR512:$src)>;
def : Pat<(v16i8  (X86vtrunc (v16i32 VR512:$src))), (VPMOVDBrr  VR512:$src)>;
def : Pat<(v8i32  (X86vtrunc (v8i64  VR512:$src))), (VPMOVQDrr  VR512:$src)>;

def : Pat<(v16i8  (X86vtruncm VK16WM:$mask, (v16i32 VR512:$src))),
                  (VPMOVDBrrkz VK16WM:$mask, VR512:$src)>;
def : Pat<(v16i16 (X86vtruncm VK16WM:$mask, (v16i32 VR512:$src))),
                  (VPMOVDWrrkz VK16WM:$mask, VR512:$src)>;
def : Pat<(v8i16  (X86vtruncm VK8WM:$mask,  (v8i64 VR512:$src))),
                  (VPMOVQWrrkz  VK8WM:$mask, VR512:$src)>;
def : Pat<(v8i32  (X86vtruncm VK8WM:$mask,  (v8i64 VR512:$src))),
                  (VPMOVQDrrkz  VK8WM:$mask, VR512:$src)>;


multiclass avx512_extend<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                      RegisterClass DstRC, RegisterClass SrcRC, SDNode OpNode,
                      PatFrag mem_frag, X86MemOperand x86memop,
                      ValueType OpVT, ValueType InVT> {

  def rr : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst),
              (ins SrcRC:$src),
              !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (OpVT (OpNode (InVT SrcRC:$src))))]>, EVEX;

  def rrk : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst),
              (ins KRC:$mask, SrcRC:$src),
              !strconcat(OpcodeStr, "\t{$src, $dst {${mask}} |$dst {${mask}}, $src}"),
              []>, EVEX, EVEX_K;

  def rrkz : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst),
              (ins KRC:$mask, SrcRC:$src),
              !strconcat(OpcodeStr, "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
              []>, EVEX, EVEX_KZ;

  let mayLoad = 1 in {
    def rm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins x86memop:$src),
              !strconcat(OpcodeStr,"\t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT (bitconvert (mem_frag addr:$src))))))]>,
              EVEX;

    def rmk : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins KRC:$mask, x86memop:$src),
              !strconcat(OpcodeStr,"\t{$src, $dst {${mask}} |$dst {${mask}}, $src}"),
              []>,
              EVEX, EVEX_K;

    def rmkz : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins KRC:$mask, x86memop:$src),
              !strconcat(OpcodeStr,"\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
              []>,
              EVEX, EVEX_KZ;
  }
}

defm VPMOVZXBDZ: avx512_extend<0x31, "vpmovzxbd", VK16WM, VR512, VR128X, X86vzext,
                             loadv2i64, i128mem, v16i32, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VQ>;
defm VPMOVZXBQZ: avx512_extend<0x32, "vpmovzxbq", VK8WM, VR512, VR128X, X86vzext,
                             loadv2i64, i128mem, v8i64, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VO>;
defm VPMOVZXWDZ: avx512_extend<0x33, "vpmovzxwd", VK16WM, VR512, VR256X, X86vzext,
                             loadv4i64, i256mem, v16i32, v16i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VH>;
defm VPMOVZXWQZ: avx512_extend<0x34, "vpmovzxwq", VK8WM, VR512, VR128X, X86vzext,
                             loadv2i64, i128mem, v8i64, v8i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VQ>;
defm VPMOVZXDQZ: avx512_extend<0x35, "vpmovzxdq", VK8WM, VR512, VR256X, X86vzext,
                             loadv4i64, i256mem, v8i64, v8i32>, EVEX_V512,
                             EVEX_CD8<32, CD8VH>;

defm VPMOVSXBDZ: avx512_extend<0x21, "vpmovsxbd", VK16WM, VR512, VR128X, X86vsext,
                             loadv2i64, i128mem, v16i32, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VQ>;
defm VPMOVSXBQZ: avx512_extend<0x22, "vpmovsxbq", VK8WM, VR512, VR128X, X86vsext,
                             loadv2i64, i128mem, v8i64, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VO>;
defm VPMOVSXWDZ: avx512_extend<0x23, "vpmovsxwd", VK16WM, VR512, VR256X, X86vsext,
                             loadv4i64, i256mem, v16i32, v16i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VH>;
defm VPMOVSXWQZ: avx512_extend<0x24, "vpmovsxwq", VK8WM, VR512, VR128X, X86vsext,
                             loadv2i64, i128mem, v8i64, v8i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VQ>;
defm VPMOVSXDQZ: avx512_extend<0x25, "vpmovsxdq", VK8WM, VR512, VR256X, X86vsext,
                             loadv4i64, i256mem, v8i64, v8i32>, EVEX_V512,
                             EVEX_CD8<32, CD8VH>;

//===----------------------------------------------------------------------===//
// GATHER - SCATTER Operations

multiclass avx512_gather<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                       RegisterClass RC, X86MemOperand memop> {
let mayLoad = 1,
  Constraints = "@earlyclobber $dst, $src1 = $dst, $mask = $mask_wb" in
  def rm  : AVX5128I<opc, MRMSrcMem, (outs RC:$dst, KRC:$mask_wb),
            (ins RC:$src1, KRC:$mask, memop:$src2),
            !strconcat(OpcodeStr,
            "\t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
            []>, EVEX, EVEX_K;
}

let ExeDomain = SSEPackedDouble in {
defm VGATHERDPDZ : avx512_gather<0x92, "vgatherdpd", VK8WM, VR512, vy64xmem>,
                                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VGATHERQPDZ : avx512_gather<0x93, "vgatherqpd", VK8WM, VR512, vz64mem>,
                                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
}

let ExeDomain = SSEPackedSingle in {
defm VGATHERDPSZ : avx512_gather<0x92, "vgatherdps", VK16WM, VR512, vz32mem>,
                                 EVEX_V512, EVEX_CD8<32, CD8VT1>;
defm VGATHERQPSZ : avx512_gather<0x93, "vgatherqps", VK8WM, VR256X, vz64mem>,
                                 EVEX_V512, EVEX_CD8<32, CD8VT1>;
}

defm VPGATHERDQZ : avx512_gather<0x90, "vpgatherdq", VK8WM, VR512,  vy64xmem>,
                                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VPGATHERDDZ : avx512_gather<0x90, "vpgatherdd", VK16WM, VR512, vz32mem>,
                                 EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VPGATHERQQZ : avx512_gather<0x91, "vpgatherqq", VK8WM, VR512,  vz64mem>,
                                 EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VPGATHERQDZ : avx512_gather<0x91, "vpgatherqd", VK8WM, VR256X,  vz64mem>,
                                 EVEX_V512, EVEX_CD8<32, CD8VT1>;

multiclass avx512_scatter<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                       RegisterClass RC, X86MemOperand memop> {
let mayStore = 1, Constraints = "$mask = $mask_wb" in
  def mr  : AVX5128I<opc, MRMDestMem, (outs KRC:$mask_wb),
            (ins memop:$dst, KRC:$mask, RC:$src2),
            !strconcat(OpcodeStr,
            "\t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
            []>, EVEX, EVEX_K;
}

let ExeDomain = SSEPackedDouble in {
defm VSCATTERDPDZ : avx512_scatter<0xA2, "vscatterdpd", VK8WM, VR512, vy64xmem>,
                                   EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VSCATTERQPDZ : avx512_scatter<0xA3, "vscatterqpd", VK8WM, VR512, vz64mem>,
                                   EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
}

let ExeDomain = SSEPackedSingle in {
defm VSCATTERDPSZ : avx512_scatter<0xA2, "vscatterdps", VK16WM, VR512, vz32mem>,
                                   EVEX_V512, EVEX_CD8<32, CD8VT1>;
defm VSCATTERQPSZ : avx512_scatter<0xA3, "vscatterqps", VK8WM, VR256X, vz64mem>,
                                   EVEX_V512, EVEX_CD8<32, CD8VT1>;
}

defm VPSCATTERDQZ : avx512_scatter<0xA0, "vpscatterdq", VK8WM, VR512, vy64xmem>,
                                   EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VPSCATTERDDZ : avx512_scatter<0xA0, "vpscatterdd", VK16WM, VR512, vz32mem>,
                                   EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VPSCATTERQQZ : avx512_scatter<0xA1, "vpscatterqq", VK8WM, VR512, vz64mem>,
                                  EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VPSCATTERQDZ : avx512_scatter<0xA1, "vpscatterqd", VK8WM, VR256X, vz64mem>,
                                  EVEX_V512, EVEX_CD8<32, CD8VT1>;

// prefetch
multiclass avx512_gather_scatter_prefetch<bits<8> opc, Format F, string OpcodeStr,
                       RegisterClass KRC, X86MemOperand memop> {
  let Predicates = [HasPFI], hasSideEffects = 1 in
  def m  : AVX5128I<opc, F, (outs), (ins KRC:$mask, memop:$src),
            !strconcat(OpcodeStr, "\t{$src {${mask}}|{${mask}}, $src}"),
            []>, EVEX, EVEX_K;
}

defm VGATHERPF0DPS: avx512_gather_scatter_prefetch<0xC6, MRM1m, "vgatherpf0dps",
                     VK16WM, vz32mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF0QPS: avx512_gather_scatter_prefetch<0xC7, MRM1m, "vgatherpf0qps",
                     VK8WM, vz64mem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VGATHERPF0DPD: avx512_gather_scatter_prefetch<0xC6, MRM1m, "vgatherpf0dpd",
                     VK8WM, vy32mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF0QPD: avx512_gather_scatter_prefetch<0xC7, MRM1m, "vgatherpf0qpd",
                     VK8WM, vz64mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

defm VGATHERPF1DPS: avx512_gather_scatter_prefetch<0xC6, MRM2m, "vgatherpf1dps",
                     VK16WM, vz32mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF1QPS: avx512_gather_scatter_prefetch<0xC7, MRM2m, "vgatherpf1qps",
                     VK8WM, vz64mem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VGATHERPF1DPD: avx512_gather_scatter_prefetch<0xC6, MRM2m, "vgatherpf1dpd",
                     VK8WM, vy32mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF1QPD: avx512_gather_scatter_prefetch<0xC7, MRM2m, "vgatherpf1qpd",
                     VK8WM, vz64mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF0DPS: avx512_gather_scatter_prefetch<0xC6, MRM5m, "vscatterpf0dps",
                     VK16WM, vz32mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF0QPS: avx512_gather_scatter_prefetch<0xC7, MRM5m, "vscatterpf0qps",
                     VK8WM, vz64mem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF0DPD: avx512_gather_scatter_prefetch<0xC6, MRM5m, "vscatterpf0dpd",
                     VK8WM, vy32mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF0QPD: avx512_gather_scatter_prefetch<0xC7, MRM5m, "vscatterpf0qpd",
                     VK8WM, vz64mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF1DPS: avx512_gather_scatter_prefetch<0xC6, MRM6m, "vscatterpf1dps",
                     VK16WM, vz32mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF1QPS: avx512_gather_scatter_prefetch<0xC7, MRM6m, "vscatterpf1qps",
                     VK8WM, vz64mem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF1DPD: avx512_gather_scatter_prefetch<0xC6, MRM6m, "vscatterpf1dpd",
                     VK8WM, vy32mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF1QPD: avx512_gather_scatter_prefetch<0xC7, MRM6m, "vscatterpf1qpd",
                     VK8WM, vz64mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
//===----------------------------------------------------------------------===//
// VSHUFPS - VSHUFPD Operations

multiclass avx512_shufp<RegisterClass RC, X86MemOperand x86memop,
                      ValueType vt, string OpcodeStr, PatFrag mem_frag,
                      Domain d> {
  def rmi : AVX512PIi8<0xC6, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, x86memop:$src2, u8imm:$src3),
                   !strconcat(OpcodeStr,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                   [(set RC:$dst, (vt (X86Shufp RC:$src1, (mem_frag addr:$src2),
                                       (i8 imm:$src3))))], d, IIC_SSE_SHUFP>,
                   EVEX_4V, Sched<[WriteShuffleLd, ReadAfterLd]>;
  def rri : AVX512PIi8<0xC6, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, u8imm:$src3),
                   !strconcat(OpcodeStr,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                   [(set RC:$dst, (vt (X86Shufp RC:$src1, RC:$src2,
                                       (i8 imm:$src3))))], d, IIC_SSE_SHUFP>,
                   EVEX_4V, Sched<[WriteShuffle]>;
}

defm VSHUFPSZ  : avx512_shufp<VR512, f512mem, v16f32, "vshufps", loadv16f32,
                  SSEPackedSingle>, PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VSHUFPDZ  : avx512_shufp<VR512, f512mem, v8f64, "vshufpd", loadv8f64,
                  SSEPackedDouble>, PD, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat<(v16i32 (X86Shufp VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VSHUFPSZrri VR512:$src1, VR512:$src2, imm:$imm)>;
def : Pat<(v16i32 (X86Shufp VR512:$src1,
                    (loadv16i32 addr:$src2), (i8 imm:$imm))),
          (VSHUFPSZrmi VR512:$src1, addr:$src2, imm:$imm)>;

def : Pat<(v8i64 (X86Shufp VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VSHUFPDZrri VR512:$src1, VR512:$src2, imm:$imm)>;
def : Pat<(v8i64 (X86Shufp VR512:$src1,
                            (loadv8i64 addr:$src2), (i8 imm:$imm))),
          (VSHUFPDZrmi VR512:$src1, addr:$src2, imm:$imm)>;

multiclass avx512_valign<X86VectorVTInfo _> {
  defm rri : AVX512_maskable<0x03, MRMSrcReg, _, (outs _.RC:$dst),
                     (ins _.RC:$src1, _.RC:$src2, u8imm:$src3),
                     "valign"##_.Suffix,
                     "$src3, $src2, $src1", "$src1, $src2, $src3",
                     (_.VT (X86VAlign _.RC:$src2, _.RC:$src1,
                                      (i8 imm:$src3)))>,
             AVX512AIi8Base, EVEX_4V;

  // Also match valign of packed floats.
  def : Pat<(_.FloatVT (X86VAlign _.RC:$src1, _.RC:$src2, (i8 imm:$imm))),
            (!cast<Instruction>(NAME##rri) _.RC:$src2, _.RC:$src1, imm:$imm)>;

  let mayLoad = 1 in
  def rmi : AVX512AIi8<0x03, MRMSrcMem, (outs _.RC:$dst),
                     (ins _.RC:$src1, _.MemOp:$src2, u8imm:$src3),
                     !strconcat("valign"##_.Suffix,
                     "\t{$src3, $src2, $src1, $dst|"
                         "$dst, $src1, $src2, $src3}"),
                     []>, EVEX_4V;
}
defm VALIGND : avx512_valign<v16i32_info>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VALIGNQ : avx512_valign<v8i64_info>, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

// Helper fragments to match sext vXi1 to vXiY.
def v16i1sextv16i32  : PatLeaf<(v16i32 (X86vsrai VR512:$src, (i8 31)))>;
def v8i1sextv8i64  : PatLeaf<(v8i64 (X86vsrai VR512:$src, (i8 63)))>;

multiclass avx512_vpabs<bits<8> opc, string OpcodeStr, ValueType OpVT,
                        RegisterClass KRC, RegisterClass RC,
                        X86MemOperand x86memop, X86MemOperand x86scalar_mop,
                        string BrdcstStr> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
            !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
            []>, EVEX;
  def rrk : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src),
             !strconcat(OpcodeStr, "\t{$src, $dst {${mask}}|$dst {${mask}}, $src}"),
             []>, EVEX, EVEX_K;
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src),
              !strconcat(OpcodeStr,
                         "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
              []>, EVEX, EVEX_KZ;
  let mayLoad = 1 in {
    def rm : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
              (ins x86memop:$src),
              !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
              []>, EVEX;
    def rmk : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
               (ins KRC:$mask, x86memop:$src),
               !strconcat(OpcodeStr,
                          "\t{$src, $dst {${mask}}|$dst {${mask}}, $src}"),
               []>, EVEX, EVEX_K;
    def rmkz : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
                (ins KRC:$mask, x86memop:$src),
                !strconcat(OpcodeStr,
                           "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
                []>, EVEX, EVEX_KZ;
    def rmb : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
               (ins x86scalar_mop:$src),
               !strconcat(OpcodeStr, "\t{${src}", BrdcstStr,
                          ", $dst|$dst, ${src}", BrdcstStr, "}"),
               []>, EVEX, EVEX_B;
    def rmbk : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
                (ins KRC:$mask, x86scalar_mop:$src),
                !strconcat(OpcodeStr, "\t{${src}", BrdcstStr,
                           ", $dst {${mask}}|$dst {${mask}}, ${src}", BrdcstStr, "}"),
                []>, EVEX, EVEX_B, EVEX_K;
    def rmbkz : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
                 (ins KRC:$mask, x86scalar_mop:$src),
                 !strconcat(OpcodeStr, "\t{${src}", BrdcstStr,
                            ", $dst {${mask}} {z}|$dst {${mask}} {z}, ${src}",
                            BrdcstStr, "}"),
                 []>, EVEX, EVEX_B, EVEX_KZ;
  }
}

defm VPABSDZ : avx512_vpabs<0x1E, "vpabsd", v16i32, VK16WM, VR512,
                           i512mem, i32mem, "{1to16}">, EVEX_V512,
                           EVEX_CD8<32, CD8VF>;
defm VPABSQZ : avx512_vpabs<0x1F, "vpabsq", v8i64, VK8WM, VR512,
                           i512mem, i64mem, "{1to8}">, EVEX_V512, VEX_W,
                           EVEX_CD8<64, CD8VF>;

def : Pat<(xor
          (bc_v16i32 (v16i1sextv16i32)),
          (bc_v16i32 (add (v16i32 VR512:$src), (v16i1sextv16i32)))),
          (VPABSDZrr VR512:$src)>;
def : Pat<(xor
          (bc_v8i64 (v8i1sextv8i64)),
          (bc_v8i64 (add (v8i64 VR512:$src), (v8i1sextv8i64)))),
          (VPABSQZrr VR512:$src)>;

def : Pat<(v16i32 (int_x86_avx512_mask_pabs_d_512 (v16i32 VR512:$src),
                   (v16i32 immAllZerosV), (i16 -1))),
          (VPABSDZrr VR512:$src)>;
def : Pat<(v8i64 (int_x86_avx512_mask_pabs_q_512 (v8i64 VR512:$src),
                   (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
          (VPABSQZrr VR512:$src)>;

multiclass avx512_conflict<bits<8> opc, string OpcodeStr,
                        RegisterClass RC, RegisterClass KRC,
                        X86MemOperand x86memop,
                        X86MemOperand x86scalar_mop, string BrdcstStr> {
  let hasSideEffects = 0 in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src),
       !strconcat(OpcodeStr, "\t{$src, ${dst} |${dst}, $src}"),
       []>, EVEX;
  let mayLoad = 1 in
  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins x86memop:$src),
       !strconcat(OpcodeStr, "\t{$src, ${dst}|${dst}, $src}"),
       []>, EVEX;
  let mayLoad = 1 in
  def rmb : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins x86scalar_mop:$src),
       !strconcat(OpcodeStr, "\t{${src}", BrdcstStr,
                  ", ${dst}|${dst}, ${src}", BrdcstStr, "}"),
       []>, EVEX, EVEX_B;
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
       (ins KRC:$mask, RC:$src),
       !strconcat(OpcodeStr,
                  "\t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
       []>, EVEX, EVEX_KZ;
  let mayLoad = 1 in
  def rmkz : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, x86memop:$src),
       !strconcat(OpcodeStr,
                  "\t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
       []>, EVEX, EVEX_KZ;
  let mayLoad = 1 in
  def rmbkz : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, x86scalar_mop:$src),
       !strconcat(OpcodeStr, "\t{${src}", BrdcstStr,
                  ", ${dst} {${mask}} {z}|${dst} {${mask}} {z}, ${src}",
                  BrdcstStr, "}"),
       []>, EVEX, EVEX_KZ, EVEX_B;

  let Constraints = "$src1 = $dst" in {
  def rrk : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, KRC:$mask, RC:$src2),
       !strconcat(OpcodeStr,
                  "\t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
       []>, EVEX, EVEX_K;
  let mayLoad = 1 in
  def rmk : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, KRC:$mask, x86memop:$src2),
       !strconcat(OpcodeStr,
                  "\t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
       []>, EVEX, EVEX_K;
  let mayLoad = 1 in
  def rmbk : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, KRC:$mask, x86scalar_mop:$src2),
       !strconcat(OpcodeStr, "\t{${src2}", BrdcstStr,
                  ", ${dst} {${mask}}|${dst} {${mask}}, ${src2}", BrdcstStr, "}"),
       []>, EVEX, EVEX_K, EVEX_B;
  }
  }
}

let Predicates = [HasCDI] in {
defm VPCONFLICTD : avx512_conflict<0xC4, "vpconflictd", VR512, VK16WM,
                    i512mem, i32mem, "{1to16}">,
                    EVEX_V512, EVEX_CD8<32, CD8VF>;


defm VPCONFLICTQ : avx512_conflict<0xC4, "vpconflictq", VR512, VK8WM,
                    i512mem, i64mem, "{1to8}">,
                    EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

}

def : Pat<(int_x86_avx512_mask_conflict_d_512 VR512:$src2, VR512:$src1,
                                              GR16:$mask),
          (VPCONFLICTDrrk VR512:$src1,
           (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)), VR512:$src2)>;

def : Pat<(int_x86_avx512_mask_conflict_q_512 VR512:$src2, VR512:$src1,
                                              GR8:$mask),
          (VPCONFLICTQrrk VR512:$src1,
           (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)), VR512:$src2)>;

let Predicates = [HasCDI] in {
defm VPLZCNTD : avx512_conflict<0x44, "vplzcntd", VR512, VK16WM,
                    i512mem, i32mem, "{1to16}">,
                    EVEX_V512, EVEX_CD8<32, CD8VF>;


defm VPLZCNTQ : avx512_conflict<0x44, "vplzcntq", VR512, VK8WM,
                    i512mem, i64mem, "{1to8}">,
                    EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

}

def : Pat<(int_x86_avx512_mask_lzcnt_d_512 VR512:$src2, VR512:$src1,
                                              GR16:$mask),
          (VPLZCNTDrrk VR512:$src1,
           (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)), VR512:$src2)>;

def : Pat<(int_x86_avx512_mask_lzcnt_q_512 VR512:$src2, VR512:$src1,
                                              GR8:$mask),
          (VPLZCNTQrrk VR512:$src1,
           (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)), VR512:$src2)>;

def : Pat<(v16i32 (ctlz (loadv16i32 addr:$src))),
          (VPLZCNTDrm addr:$src)>;
def : Pat<(v16i32 (ctlz (v16i32 VR512:$src))),
          (VPLZCNTDrr VR512:$src)>;
def : Pat<(v8i64 (ctlz (loadv8i64 addr:$src))),
          (VPLZCNTQrm addr:$src)>;
def : Pat<(v8i64 (ctlz (v8i64 VR512:$src))),
          (VPLZCNTQrr VR512:$src)>;

def : Pat<(store (i1 -1), addr:$dst), (MOV8mi addr:$dst, (i8 1))>;
def : Pat<(store (i1  1), addr:$dst), (MOV8mi addr:$dst, (i8 1))>;
def : Pat<(store (i1  0), addr:$dst), (MOV8mi addr:$dst, (i8 0))>;

def : Pat<(store VK1:$src, addr:$dst),
          (MOV8mr addr:$dst,
           (EXTRACT_SUBREG (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)),
            sub_8bit))>, Requires<[HasAVX512, NoDQI]>;

def : Pat<(store VK8:$src, addr:$dst),
          (MOV8mr addr:$dst,
           (EXTRACT_SUBREG (KMOVWrk (COPY_TO_REGCLASS VK8:$src, VK16)),
            sub_8bit))>, Requires<[HasAVX512, NoDQI]>;

def truncstorei1 : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore node:$val, node:$ptr), [{
  return cast<StoreSDNode>(N)->getMemoryVT() == MVT::i1;
}]>;

def : Pat<(truncstorei1 GR8:$src, addr:$dst),
          (MOV8mr addr:$dst, GR8:$src)>;

multiclass cvt_by_vec_width<bits<8> opc, X86VectorVTInfo Vec, string OpcodeStr > {
def rr : AVX512XS8I<opc, MRMDestReg, (outs Vec.RC:$dst), (ins Vec.KRC:$src),
                  !strconcat(OpcodeStr##Vec.Suffix, "\t{$src, $dst|$dst, $src}"),
                  [(set Vec.RC:$dst, (Vec.VT (X86vsext Vec.KRC:$src)))]>, EVEX;
}

multiclass cvt_mask_by_elt_width<bits<8> opc, AVX512VLVectorVTInfo VTInfo,
                                 string OpcodeStr, Predicate prd> {
let Predicates = [prd] in
  defm Z : cvt_by_vec_width<opc, VTInfo.info512, OpcodeStr>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : cvt_by_vec_width<opc, VTInfo.info256, OpcodeStr>, EVEX_V256;
    defm Z128 : cvt_by_vec_width<opc, VTInfo.info128, OpcodeStr>, EVEX_V128;
  }
}

multiclass avx512_convert_mask_to_vector<string OpcodeStr> {
  defm NAME##B : cvt_mask_by_elt_width<0x28, avx512vl_i8_info,  OpcodeStr,
                                       HasBWI>;
  defm NAME##W : cvt_mask_by_elt_width<0x28, avx512vl_i16_info, OpcodeStr,
                                       HasBWI>, VEX_W;
  defm NAME##D : cvt_mask_by_elt_width<0x38, avx512vl_i32_info, OpcodeStr,
                                       HasDQI>;
  defm NAME##Q : cvt_mask_by_elt_width<0x38, avx512vl_i64_info, OpcodeStr,
                                       HasDQI>, VEX_W;
}

defm VPMOVM2 : avx512_convert_mask_to_vector<"vpmovm2">;

//===----------------------------------------------------------------------===//
// AVX-512 - COMPRESS and EXPAND
//
multiclass compress_by_vec_width<bits<8> opc, X86VectorVTInfo _,
                                 string OpcodeStr> {
  def rrkz : AVX5128I<opc, MRMDestReg, (outs _.RC:$dst),
              (ins _.KRCWM:$mask, _.RC:$src),
              OpcodeStr # "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}",
              [(set _.RC:$dst, (_.VT (X86compress _.KRCWM:$mask, _.RC:$src,
                                      _.ImmAllZerosV)))]>, EVEX_KZ;

  let Constraints = "$src0 = $dst" in
  def rrk : AVX5128I<opc, MRMDestReg, (outs _.RC:$dst),
                    (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src),
                    OpcodeStr # "\t{$src, $dst {${mask}} |$dst {${mask}}, $src}",
                    [(set _.RC:$dst, (_.VT (X86compress _.KRCWM:$mask, _.RC:$src,
                                            _.RC:$src0)))]>, EVEX_K;

  let mayStore = 1 in {
  def mrk : AVX5128I<opc, MRMDestMem, (outs),
              (ins _.MemOp:$dst, _.KRCWM:$mask, _.RC:$src),
              OpcodeStr # "\t{$src, $dst {${mask}} |$dst {${mask}}, $src}",
              [(store (_.VT (X86compress _.KRCWM:$mask, _.RC:$src, undef)),
                addr:$dst)]>,
              EVEX_K, EVEX_CD8<_.EltSize, CD8VT1>;
  }
}

multiclass compress_by_elt_width<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo> {
  defm Z : compress_by_vec_width<opc, VTInfo.info512, OpcodeStr>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256 : compress_by_vec_width<opc, VTInfo.info256, OpcodeStr>, EVEX_V256;
    defm Z128 : compress_by_vec_width<opc, VTInfo.info128, OpcodeStr>, EVEX_V128;
  }
}

defm VPCOMPRESSD : compress_by_elt_width <0x8B, "vpcompressd", avx512vl_i32_info>,
                                         EVEX;
defm VPCOMPRESSQ : compress_by_elt_width <0x8B, "vpcompressq", avx512vl_i64_info>,
                                         EVEX, VEX_W;
defm VCOMPRESSPS : compress_by_elt_width <0x8A, "vcompressps", avx512vl_f32_info>,
                                         EVEX;
defm VCOMPRESSPD : compress_by_elt_width <0x8A, "vcompresspd", avx512vl_f64_info>,
                                         EVEX, VEX_W;

// expand
multiclass expand_by_vec_width<bits<8> opc, X86VectorVTInfo _,
                                 string OpcodeStr> {
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
              (ins _.KRCWM:$mask, _.RC:$src),
              OpcodeStr # "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}",
              [(set _.RC:$dst, (_.VT (X86expand _.KRCWM:$mask, (_.VT _.RC:$src),
                                      _.ImmAllZerosV)))]>, EVEX_KZ;

  let Constraints = "$src0 = $dst" in
  def rrk : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
                    (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src),
                    OpcodeStr # "\t{$src, $dst {${mask}} |$dst {${mask}}, $src}",
                    [(set _.RC:$dst, (_.VT (X86expand _.KRCWM:$mask,
                                      (_.VT _.RC:$src), _.RC:$src0)))]>, EVEX_K;

  let mayLoad = 1, Constraints = "$src0 = $dst" in
  def rmk : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
              (ins _.RC:$src0, _.KRCWM:$mask, _.MemOp:$src),
              OpcodeStr # "\t{$src, $dst {${mask}} |$dst {${mask}}, $src}",
              [(set _.RC:$dst, (_.VT (X86expand _.KRCWM:$mask,
                                      (_.VT (bitconvert
                                              (_.LdFrag addr:$src))),
                                      _.RC:$src0)))]>,
              EVEX_K, EVEX_CD8<_.EltSize, CD8VT1>;
  
  let mayLoad = 1 in
  def rmkz : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
              (ins _.KRCWM:$mask, _.MemOp:$src),
              OpcodeStr # "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}",
              [(set _.RC:$dst, (_.VT (X86expand _.KRCWM:$mask,
                                      (_.VT (bitconvert (_.LdFrag addr:$src))),
                                     _.ImmAllZerosV)))]>,
              EVEX_KZ, EVEX_CD8<_.EltSize, CD8VT1>;
  
}

multiclass expand_by_elt_width<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo> {
  defm Z : expand_by_vec_width<opc, VTInfo.info512, OpcodeStr>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256 : expand_by_vec_width<opc, VTInfo.info256, OpcodeStr>, EVEX_V256;
    defm Z128 : expand_by_vec_width<opc, VTInfo.info128, OpcodeStr>, EVEX_V128;
  }
}

defm VPEXPANDD : expand_by_elt_width <0x89, "vpexpandd", avx512vl_i32_info>,
                                         EVEX;
defm VPEXPANDQ : expand_by_elt_width <0x89, "vpexpandq", avx512vl_i64_info>,
                                         EVEX, VEX_W;
defm VEXPANDPS : expand_by_elt_width <0x88, "vexpandps", avx512vl_f32_info>,
                                         EVEX;
defm VEXPANDPD : expand_by_elt_width <0x88, "vexpandpd", avx512vl_f64_info>,
                                         EVEX, VEX_W;
