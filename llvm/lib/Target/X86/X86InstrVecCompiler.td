//===- X86InstrVecCompiler.td - Vector Compiler Patterns ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the various vector pseudo instructions used by the
// compiler, as well as Pat patterns used during instruction selection.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Non-instruction patterns
//===----------------------------------------------------------------------===//

// Bitcasts between 128-bit vector types. Return the original type since
// no instruction is needed for the conversion
def : Pat<(v2i64 (bitconvert (v4i32 VR128:$src))), (v2i64 VR128:$src)>;
def : Pat<(v2i64 (bitconvert (v8i16 VR128:$src))), (v2i64 VR128:$src)>;
def : Pat<(v2i64 (bitconvert (v16i8 VR128:$src))), (v2i64 VR128:$src)>;
def : Pat<(v2i64 (bitconvert (v2f64 VR128:$src))), (v2i64 VR128:$src)>;
def : Pat<(v2i64 (bitconvert (v4f32 VR128:$src))), (v2i64 VR128:$src)>;
def : Pat<(v4i32 (bitconvert (v2i64 VR128:$src))), (v4i32 VR128:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16 VR128:$src))), (v4i32 VR128:$src)>;
def : Pat<(v4i32 (bitconvert (v16i8 VR128:$src))), (v4i32 VR128:$src)>;
def : Pat<(v4i32 (bitconvert (v2f64 VR128:$src))), (v4i32 VR128:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32 VR128:$src))), (v4i32 VR128:$src)>;
def : Pat<(v8i16 (bitconvert (v2i64 VR128:$src))), (v8i16 VR128:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32 VR128:$src))), (v8i16 VR128:$src)>;
def : Pat<(v8i16 (bitconvert (v16i8 VR128:$src))), (v8i16 VR128:$src)>;
def : Pat<(v8i16 (bitconvert (v2f64 VR128:$src))), (v8i16 VR128:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32 VR128:$src))), (v8i16 VR128:$src)>;
def : Pat<(v16i8 (bitconvert (v2i64 VR128:$src))), (v16i8 VR128:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32 VR128:$src))), (v16i8 VR128:$src)>;
def : Pat<(v16i8 (bitconvert (v8i16 VR128:$src))), (v16i8 VR128:$src)>;
def : Pat<(v16i8 (bitconvert (v2f64 VR128:$src))), (v16i8 VR128:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32 VR128:$src))), (v16i8 VR128:$src)>;
def : Pat<(v4f32 (bitconvert (v2i64 VR128:$src))), (v4f32 VR128:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32 VR128:$src))), (v4f32 VR128:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16 VR128:$src))), (v4f32 VR128:$src)>;
def : Pat<(v4f32 (bitconvert (v16i8 VR128:$src))), (v4f32 VR128:$src)>;
def : Pat<(v4f32 (bitconvert (v2f64 VR128:$src))), (v4f32 VR128:$src)>;
def : Pat<(v2f64 (bitconvert (v2i64 VR128:$src))), (v2f64 VR128:$src)>;
def : Pat<(v2f64 (bitconvert (v4i32 VR128:$src))), (v2f64 VR128:$src)>;
def : Pat<(v2f64 (bitconvert (v8i16 VR128:$src))), (v2f64 VR128:$src)>;
def : Pat<(v2f64 (bitconvert (v16i8 VR128:$src))), (v2f64 VR128:$src)>;
def : Pat<(v2f64 (bitconvert (v4f32 VR128:$src))), (v2f64 VR128:$src)>;
def : Pat<(f128  (bitconvert (i128  FR128:$src))), (f128  FR128:$src)>;
def : Pat<(i128  (bitconvert (f128  FR128:$src))), (i128  FR128:$src)>;

// Bitcasts between 256-bit vector types. Return the original type since
// no instruction is needed for the conversion
def : Pat<(v4i64  (bitconvert (v8i32  VR256:$src))), (v4i64  VR256:$src)>;
def : Pat<(v4i64  (bitconvert (v16i16 VR256:$src))), (v4i64  VR256:$src)>;
def : Pat<(v4i64  (bitconvert (v32i8  VR256:$src))), (v4i64  VR256:$src)>;
def : Pat<(v4i64  (bitconvert (v8f32  VR256:$src))), (v4i64  VR256:$src)>;
def : Pat<(v4i64  (bitconvert (v4f64  VR256:$src))), (v4i64  VR256:$src)>;
def : Pat<(v8i32  (bitconvert (v4i64  VR256:$src))), (v8i32  VR256:$src)>;
def : Pat<(v8i32  (bitconvert (v16i16 VR256:$src))), (v8i32  VR256:$src)>;
def : Pat<(v8i32  (bitconvert (v32i8  VR256:$src))), (v8i32  VR256:$src)>;
def : Pat<(v8i32  (bitconvert (v4f64  VR256:$src))), (v8i32  VR256:$src)>;
def : Pat<(v8i32  (bitconvert (v8f32  VR256:$src))), (v8i32  VR256:$src)>;
def : Pat<(v16i16 (bitconvert (v4i64  VR256:$src))), (v16i16 VR256:$src)>;
def : Pat<(v16i16 (bitconvert (v8i32  VR256:$src))), (v16i16 VR256:$src)>;
def : Pat<(v16i16 (bitconvert (v32i8  VR256:$src))), (v16i16 VR256:$src)>;
def : Pat<(v16i16 (bitconvert (v4f64  VR256:$src))), (v16i16 VR256:$src)>;
def : Pat<(v16i16 (bitconvert (v8f32  VR256:$src))), (v16i16 VR256:$src)>;
def : Pat<(v32i8  (bitconvert (v4i64  VR256:$src))), (v32i8  VR256:$src)>;
def : Pat<(v32i8  (bitconvert (v8i32  VR256:$src))), (v32i8  VR256:$src)>;
def : Pat<(v32i8  (bitconvert (v16i16 VR256:$src))), (v32i8  VR256:$src)>;
def : Pat<(v32i8  (bitconvert (v4f64  VR256:$src))), (v32i8  VR256:$src)>;
def : Pat<(v32i8  (bitconvert (v8f32  VR256:$src))), (v32i8  VR256:$src)>;
def : Pat<(v8f32  (bitconvert (v4i64  VR256:$src))), (v8f32  VR256:$src)>;
def : Pat<(v8f32  (bitconvert (v8i32  VR256:$src))), (v8f32  VR256:$src)>;
def : Pat<(v8f32  (bitconvert (v16i16 VR256:$src))), (v8f32  VR256:$src)>;
def : Pat<(v8f32  (bitconvert (v32i8  VR256:$src))), (v8f32  VR256:$src)>;
def : Pat<(v8f32  (bitconvert (v4f64  VR256:$src))), (v8f32  VR256:$src)>;
def : Pat<(v4f64  (bitconvert (v4i64  VR256:$src))), (v4f64  VR256:$src)>;
def : Pat<(v4f64  (bitconvert (v8i32  VR256:$src))), (v4f64  VR256:$src)>;
def : Pat<(v4f64  (bitconvert (v16i16 VR256:$src))), (v4f64  VR256:$src)>;
def : Pat<(v4f64  (bitconvert (v32i8  VR256:$src))), (v4f64  VR256:$src)>;
def : Pat<(v4f64  (bitconvert (v8f32  VR256:$src))), (v4f64  VR256:$src)>;

// Bitcasts between 512-bit vector types. Return the original type since
// no instruction is needed for the conversion.
def : Pat<(v8f64  (bitconvert (v8i64  VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v16i32 VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v32i16 VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v64i8  VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v16f32 VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v8i64  VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v16i32 VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v32i16 VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v64i8  VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v8f64  VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v16i32 VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v32i16 VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v64i8  VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v8f64  VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v16f32 VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v8i64  VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v16f32 VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v32i16 VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v64i8  VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v8f64  VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v8i64  VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v16i32 VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v64i8  VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v8f64  VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v16f32 VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v16f32 VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v8i64  VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v16i32 VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v32i16 VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v8f64  VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v16f32 VR512:$src))), (v64i8  VR512:$src)>;


// A vector extract of the first f32/f64 position is a subregister copy
def : Pat<(f32 (extractelt (v4f32 VR128:$src), (iPTR 0))),
          (COPY_TO_REGCLASS (v4f32 VR128:$src), FR32)>;
def : Pat<(f64 (extractelt (v2f64 VR128:$src), (iPTR 0))),
          (COPY_TO_REGCLASS (v2f64 VR128:$src), FR64)>;

// Implicitly promote a 32-bit scalar to a vector.
def : Pat<(v4f32 (scalar_to_vector FR32:$src)),
          (COPY_TO_REGCLASS FR32:$src, VR128)>;
// Implicitly promote a 64-bit scalar to a vector.
def : Pat<(v2f64 (scalar_to_vector FR64:$src)),
          (COPY_TO_REGCLASS FR64:$src, VR128)>;


// Patterns for insert_subvector/extract_subvector to/from index=0
multiclass subvector_subreg_lowering<RegisterClass subRC, ValueType subVT,
                                     RegisterClass RC, ValueType VT,
                                     SubRegIndex subIdx> {
  def : Pat<(subVT (extract_subvector (VT RC:$src), (iPTR 0))),
            (subVT (EXTRACT_SUBREG RC:$src, subIdx))>;

  let AddedComplexity = 25 in // to give priority over vinsertf128rm
  def : Pat<(VT (insert_subvector undef, subRC:$src, (iPTR 0))),
            (VT (INSERT_SUBREG (IMPLICIT_DEF), subRC:$src, subIdx))>;
}

// A 128-bit subvector extract from the first 256-bit vector position is a
// subregister copy that needs no instruction. Likewise, a 128-bit subvector
// insert to the first 256-bit vector position is a subregister copy that needs
// no instruction.
defm : subvector_subreg_lowering<VR128, v4i32, VR256, v8i32,  sub_xmm>;
defm : subvector_subreg_lowering<VR128, v4f32, VR256, v8f32,  sub_xmm>;
defm : subvector_subreg_lowering<VR128, v2i64, VR256, v4i64,  sub_xmm>;
defm : subvector_subreg_lowering<VR128, v2f64, VR256, v4f64,  sub_xmm>;
defm : subvector_subreg_lowering<VR128, v8i16, VR256, v16i16, sub_xmm>;
defm : subvector_subreg_lowering<VR128, v16i8, VR256, v32i8,  sub_xmm>;

// A 128-bit subvector extract from the first 512-bit vector position is a
// subregister copy that needs no instruction. Likewise, a 128-bit subvector
// insert to the first 512-bit vector position is a subregister copy that needs
// no instruction.
defm : subvector_subreg_lowering<VR128, v4i32, VR512, v16i32, sub_xmm>;
defm : subvector_subreg_lowering<VR128, v4f32, VR512, v16f32, sub_xmm>;
defm : subvector_subreg_lowering<VR128, v2i64, VR512, v8i64,  sub_xmm>;
defm : subvector_subreg_lowering<VR128, v2f64, VR512, v8f64,  sub_xmm>;
defm : subvector_subreg_lowering<VR128, v8i16, VR512, v32i16, sub_xmm>;
defm : subvector_subreg_lowering<VR128, v16i8, VR512, v64i8,  sub_xmm>;

// A 128-bit subvector extract from the first 512-bit vector position is a
// subregister copy that needs no instruction. Likewise, a 128-bit subvector
// insert to the first 512-bit vector position is a subregister copy that needs
// no instruction.
defm : subvector_subreg_lowering<VR256, v8i32,  VR512, v16i32, sub_ymm>;
defm : subvector_subreg_lowering<VR256, v8f32,  VR512, v16f32, sub_ymm>;
defm : subvector_subreg_lowering<VR256, v4i64,  VR512, v8i64,  sub_ymm>;
defm : subvector_subreg_lowering<VR256, v4f64,  VR512, v8f64,  sub_ymm>;
defm : subvector_subreg_lowering<VR256, v16i16, VR512, v32i16, sub_ymm>;
defm : subvector_subreg_lowering<VR256, v32i8,  VR512, v64i8,  sub_ymm>;
