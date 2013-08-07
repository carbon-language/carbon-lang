// Bitcasts between 512-bit vector types. Return the original type since
// no instruction is needed for the conversion
let Predicates = [HasAVX512] in {
  def : Pat<(v8f64  (bitconvert (v16f32 VR512:$src))), (v8f64 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v16i32 VR512:$src))), (v8f64 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v8i64 VR512:$src))),  (v8f64 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v16i32 VR512:$src))), (v16f32 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v8i64 VR512:$src))),  (v16f32 VR512:$src)>;
  def : Pat<(v16f32 (bitconvert (v8f64 VR512:$src))),  (v16f32 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v16f32 VR512:$src))), (v8i64 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v16i32 VR512:$src))), (v8i64 VR512:$src)>;
  def : Pat<(v8i64  (bitconvert (v8f64 VR512:$src))),  (v8i64 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v16f32 VR512:$src))), (v16i32 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v8i64 VR512:$src))),  (v16i32 VR512:$src)>;
  def : Pat<(v16i32 (bitconvert (v8f64 VR512:$src))),  (v16i32 VR512:$src)>;
  def : Pat<(v8f64  (bitconvert (v8i64 VR512:$src))),  (v8f64 VR512:$src)>;

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

//===----------------------------------------------------------------------===//
// AVX-512 - VECTOR INSERT
//
// -- 32x8 form --
let neverHasSideEffects = 1, ExeDomain = SSEPackedSingle in {
def VINSERTF32x4rr : AVX512AIi8<0x18, MRMSrcReg, (outs VR512:$dst),
          (ins VR512:$src1, VR128X:$src2, i8imm:$src3),
          "vinsertf32x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512;
let mayLoad = 1 in
def VINSERTF32x4rm : AVX512AIi8<0x18, MRMSrcMem, (outs VR512:$dst),
          (ins VR512:$src1, f128mem:$src2, i8imm:$src3),
          "vinsertf32x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512, EVEX_CD8<32, CD8VT4>;
}

// -- 64x4 fp form --
let neverHasSideEffects = 1, ExeDomain = SSEPackedDouble in {
def VINSERTF64x4rr : AVX512AIi8<0x1a, MRMSrcReg, (outs VR512:$dst),
          (ins VR512:$src1, VR256X:$src2, i8imm:$src3),
          "vinsertf64x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512, VEX_W;
let mayLoad = 1 in
def VINSERTF64x4rm : AVX512AIi8<0x1a, MRMSrcMem, (outs VR512:$dst),
          (ins VR512:$src1, i256mem:$src2, i8imm:$src3),
          "vinsertf64x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT4>;
}
// -- 32x4 integer form --
let neverHasSideEffects = 1 in {
def VINSERTI32x4rr : AVX512AIi8<0x38, MRMSrcReg, (outs VR512:$dst),
          (ins VR512:$src1, VR128X:$src2, i8imm:$src3),
          "vinserti32x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512;
let mayLoad = 1 in
def VINSERTI32x4rm : AVX512AIi8<0x38, MRMSrcMem, (outs VR512:$dst),
          (ins VR512:$src1, i128mem:$src2, i8imm:$src3),
          "vinserti32x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512, EVEX_CD8<32, CD8VT4>;

}

let neverHasSideEffects = 1 in {
// -- 64x4 form --
def VINSERTI64x4rr : AVX512AIi8<0x3a, MRMSrcReg, (outs VR512:$dst),
          (ins VR512:$src1, VR256X:$src2, i8imm:$src3),
          "vinserti64x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512, VEX_W;
let mayLoad = 1 in
def VINSERTI64x4rm : AVX512AIi8<0x3a, MRMSrcMem, (outs VR512:$dst),
          (ins VR512:$src1, i256mem:$src2, i8imm:$src3),
          "vinserti64x4\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
          []>, EVEX_4V, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT4>;
}

def : Pat<(vinsert128_insert:$ins (v16f32 VR512:$src1), (v4f32 VR128X:$src2),
           (iPTR imm)), (VINSERTF32x4rr VR512:$src1, VR128X:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8f64  VR512:$src1), (v2f64 VR128X:$src2),
           (iPTR imm)), (VINSERTF32x4rr VR512:$src1, VR128X:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i64  VR512:$src1), (v2i64 VR128X:$src2),
           (iPTR imm)), (VINSERTI32x4rr VR512:$src1, VR128X:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i32 VR512:$src1), (v4i32 VR128X:$src2),
           (iPTR imm)), (VINSERTI32x4rr VR512:$src1, VR128X:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
			
def : Pat<(vinsert128_insert:$ins (v16f32 VR512:$src1), (loadv4f32 addr:$src2),
           (iPTR imm)), (VINSERTF32x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i32 VR512:$src1),
	                (bc_v4i32 (loadv2i64 addr:$src2)),
           (iPTR imm)), (VINSERTI32x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8f64  VR512:$src1), (loadv2f64 addr:$src2),
           (iPTR imm)), (VINSERTF32x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i64  VR512:$src1), (loadv2i64 addr:$src2),
           (iPTR imm)), (VINSERTI32x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert128_imm VR512:$ins))>;

def : Pat<(vinsert256_insert:$ins (v16f32  VR512:$src1), (v8f32 VR256X:$src2),
           (iPTR imm)), (VINSERTF64x4rr VR512:$src1, VR256X:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;
def : Pat<(vinsert256_insert:$ins (v8f64  VR512:$src1), (v4f64 VR256X:$src2),
           (iPTR imm)), (VINSERTF64x4rr VR512:$src1, VR256X:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v8i64  VR512:$src1), (v4i64 VR256X:$src2),
           (iPTR imm)), (VINSERTI64x4rr VR512:$src1, VR256X:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;
def : Pat<(vinsert128_insert:$ins (v16i32 VR512:$src1), (v8i32 VR256X:$src2),
           (iPTR imm)), (VINSERTI64x4rr VR512:$src1, VR256X:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;

def : Pat<(vinsert256_insert:$ins (v16f32  VR512:$src1), (loadv8f32 addr:$src2),
           (iPTR imm)), (VINSERTF64x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;
def : Pat<(vinsert256_insert:$ins (v8f64  VR512:$src1), (loadv4f64 addr:$src2),
           (iPTR imm)), (VINSERTF64x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;
def : Pat<(vinsert256_insert:$ins (v8i64  VR512:$src1), (loadv4i64 addr:$src2),
           (iPTR imm)), (VINSERTI64x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;
def : Pat<(vinsert256_insert:$ins (v16i32 VR512:$src1),
	                (bc_v8i32 (loadv4i64 addr:$src2)),
           (iPTR imm)), (VINSERTI64x4rm VR512:$src1, addr:$src2,
                        (INSERT_get_vinsert256_imm VR512:$ins))>;

// vinsertps - insert f32 to XMM
def VINSERTPSzrr : AVX512AIi8<0x21, MRMSrcReg, (outs VR128X:$dst),
      (ins VR128X:$src1, VR128X:$src2, u32u8imm:$src3),
      "vinsertps{z}\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insrtps VR128X:$src1, VR128X:$src2, imm:$src3))]>,
      EVEX_4V;
def VINSERTPSzrm: AVX512AIi8<0x21, MRMSrcMem, (outs VR128X:$dst),
      (ins VR128X:$src1, f32mem:$src2, u32u8imm:$src3),
      "vinsertps{z}\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insrtps VR128X:$src1,
                          (v4f32 (scalar_to_vector (loadf32 addr:$src2))),
                          imm:$src3))]>, EVEX_4V, EVEX_CD8<32, CD8VT1>;

//===----------------------------------------------------------------------===//
// AVX-512 VECTOR EXTRACT
//---
let neverHasSideEffects = 1, ExeDomain = SSEPackedSingle in {
// -- 32x4 form --
def VEXTRACTF32x4rr : AVX512AIi8<0x19, MRMDestReg, (outs VR128X:$dst),
          (ins VR512:$src1, i8imm:$src2),
          "vextractf32x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512;
def VEXTRACTF32x4mr : AVX512AIi8<0x19, MRMDestMem, (outs),
          (ins f128mem:$dst, VR512:$src1, i8imm:$src2),
          "vextractf32x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512, EVEX_CD8<32, CD8VT4>;

// -- 64x4 form --
def VEXTRACTF64x4rr : AVX512AIi8<0x1b, MRMDestReg, (outs VR256X:$dst),
          (ins VR512:$src1, i8imm:$src2),
          "vextractf64x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512, VEX_W;
let mayStore = 1 in
def VEXTRACTF64x4mr : AVX512AIi8<0x1b, MRMDestMem, (outs),
          (ins f256mem:$dst, VR512:$src1, i8imm:$src2),
          "vextractf64x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT4>;
}

let neverHasSideEffects = 1 in {
// -- 32x4 form --
def VEXTRACTI32x4rr : AVX512AIi8<0x39, MRMDestReg, (outs VR128X:$dst),
          (ins VR512:$src1, i8imm:$src2),
          "vextracti32x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512;
def VEXTRACTI32x4mr : AVX512AIi8<0x39, MRMDestMem, (outs),
          (ins i128mem:$dst, VR512:$src1, i8imm:$src2),
          "vextracti32x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512, EVEX_CD8<32, CD8VT4>;

// -- 64x4 form --
def VEXTRACTI64x4rr : AVX512AIi8<0x3b, MRMDestReg, (outs VR256X:$dst),
          (ins VR512:$src1, i8imm:$src2),
          "vextracti64x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512, VEX_W;
let mayStore = 1 in
def VEXTRACTI64x4mr : AVX512AIi8<0x3b, MRMDestMem, (outs),
          (ins i256mem:$dst, VR512:$src1, i8imm:$src2),
          "vextracti64x4\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          []>, EVEX, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT4>;
}

def : Pat<(vextract128_extract:$ext (v16f32 VR512:$src1), (iPTR imm)),
          (v4f32 (VEXTRACTF32x4rr VR512:$src1,
                  (EXTRACT_get_vextract128_imm VR128X:$ext)))>;

def : Pat<(vextract128_extract:$ext VR512:$src1, (iPTR imm)),
          (v4i32 (VEXTRACTF32x4rr VR512:$src1,
                  (EXTRACT_get_vextract128_imm VR128X:$ext)))>;

def : Pat<(vextract128_extract:$ext (v8f64 VR512:$src1), (iPTR imm)),
          (v2f64 (VEXTRACTF32x4rr VR512:$src1,
                  (EXTRACT_get_vextract128_imm VR128X:$ext)))>;

def : Pat<(vextract128_extract:$ext (v8i64 VR512:$src1), (iPTR imm)),
          (v2i64 (VEXTRACTI32x4rr VR512:$src1,
                  (EXTRACT_get_vextract128_imm VR128X:$ext)))>;


def : Pat<(vextract256_extract:$ext (v16f32 VR512:$src1), (iPTR imm)),
          (v8f32 (VEXTRACTF64x4rr VR512:$src1,
                  (EXTRACT_get_vextract256_imm VR256X:$ext)))>;

def : Pat<(vextract256_extract:$ext (v16i32 VR512:$src1), (iPTR imm)),
          (v8i32 (VEXTRACTI64x4rr VR512:$src1,
                    (EXTRACT_get_vextract256_imm VR256X:$ext)))>;

def : Pat<(vextract256_extract:$ext (v8f64 VR512:$src1), (iPTR imm)),
          (v4f64 (VEXTRACTF64x4rr VR512:$src1,
                  (EXTRACT_get_vextract256_imm VR256X:$ext)))>;

def : Pat<(vextract256_extract:$ext (v8i64 VR512:$src1), (iPTR imm)),
          (v4i64 (VEXTRACTI64x4rr VR512:$src1,
                  (EXTRACT_get_vextract256_imm VR256X:$ext)))>;

// A 256-bit subvector extract from the first 512-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(v8i32 (extract_subvector (v16i32 VR512:$src), (iPTR 0))),
          (v8i32 (EXTRACT_SUBREG (v16i32 VR512:$src), sub_ymm))>;
def : Pat<(v8f32 (extract_subvector (v16f32 VR512:$src), (iPTR 0))),
          (v8f32 (EXTRACT_SUBREG (v16f32 VR512:$src), sub_ymm))>;
def : Pat<(v4i64 (extract_subvector (v8i64 VR512:$src), (iPTR 0))),
          (v4i64 (EXTRACT_SUBREG (v8i64 VR512:$src), sub_ymm))>;
def : Pat<(v4f64 (extract_subvector (v8f64 VR512:$src), (iPTR 0))),
          (v4f64 (EXTRACT_SUBREG (v8f64 VR512:$src), sub_ymm))>;

// zmm -> xmm
def : Pat<(v4i32 (extract_subvector (v16i32 VR512:$src), (iPTR 0))),
          (v4i32 (EXTRACT_SUBREG (v16i32 VR512:$src), sub_xmm))>;
def : Pat<(v2i64 (extract_subvector (v8i64 VR512:$src), (iPTR 0))),
          (v2i64 (EXTRACT_SUBREG (v8i64 VR512:$src), sub_xmm))>;
def : Pat<(v2f64 (extract_subvector (v8f64 VR512:$src), (iPTR 0))),
          (v2f64 (EXTRACT_SUBREG (v8f64 VR512:$src), sub_xmm))>;
def : Pat<(v4f32 (extract_subvector (v16f32 VR512:$src), (iPTR 0))),
          (v4f32 (EXTRACT_SUBREG (v16f32 VR512:$src), sub_xmm))>;


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
      (ins VR128X:$src1, u32u8imm:$src2),
      "vextractps{z}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set GR32:$dst, (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2))]>,
      EVEX;

def VEXTRACTPSzmr : AVX512AIi8<0x17, MRMDestMem, (outs),
      (ins f32mem:$dst, VR128X:$src1, u32u8imm:$src2),
      "vextractps{z}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(store (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2),
                          addr:$dst)]>, EVEX;

//===---------------------------------------------------------------------===//
// AVX-512 BROADCAST
//---
multiclass avx512_fp_broadcast<bits<8> opc, string OpcodeStr, 
                         RegisterClass DestRC,
                         RegisterClass SrcRC, X86MemOperand x86memop> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs DestRC:$dst), (ins SrcRC:$src),
         !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
         []>, EVEX;
  def rm : AVX5128I<opc, MRMSrcMem, (outs DestRC:$dst), (ins x86memop:$src),
        !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),[]>, EVEX;
}
let ExeDomain = SSEPackedSingle in {
  defm VBROADCASTSSZ  : avx512_fp_broadcast<0x18, "vbroadcastss{z}", VR512, 
                                       VR128X, f32mem>,
                                       EVEX_V512, EVEX_CD8<32, CD8VT1>;
}

let ExeDomain = SSEPackedDouble in {
  defm VBROADCASTSDZ  : avx512_fp_broadcast<0x19, "vbroadcastsd{z}", VR512,
                                       VR128X, f64mem>,
                                       EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
}

def : Pat<(v16f32 (X86VBroadcast (loadf32 addr:$src))),
          (VBROADCASTSSZrm addr:$src)>;
def : Pat<(v8f64 (X86VBroadcast (loadf64 addr:$src))),
          (VBROADCASTSDZrm addr:$src)>;

multiclass avx512_int_broadcast_reg<bits<8> opc, string OpcodeStr,
                          RegisterClass SrcRC, RegisterClass KRC> {
  def Zrr : AVX5128I<opc, MRMSrcReg, (outs VR512:$dst), (ins SrcRC:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   []>, EVEX, EVEX_V512;
  def Zkrr : AVX5128I<opc, MRMSrcReg, (outs VR512:$dst), 
                   (ins KRC:$mask, SrcRC:$src),
                   !strconcat(OpcodeStr, 
                        "\t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
                   []>, EVEX, EVEX_V512, EVEX_KZ;
}

defm VPBROADCASTDr  : avx512_int_broadcast_reg<0x7C, "vpbroadcastd", GR32, VK16WM>;
defm VPBROADCASTQr  : avx512_int_broadcast_reg<0x7C, "vpbroadcastq", GR64, VK8WM>,
                                            VEX_W;
                                            
def : Pat <(v16i32 (X86vzext VK16WM:$mask)),
           (VPBROADCASTDrZkrr VK16WM:$mask, (i32 (MOV32ri 0x1)))>;

def : Pat <(v8i64 (X86vzext VK8WM:$mask)),
           (VPBROADCASTQrZkrr VK8WM:$mask, (i64 (MOV64ri 0x1)))>;

def : Pat<(v16i32 (X86VBroadcast (i32 GR32:$src))),
        (VPBROADCASTDrZrr GR32:$src)>;
def : Pat<(v8i64 (X86VBroadcast (i64 GR64:$src))),
        (VPBROADCASTQrZrr GR64:$src)>;

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
                    "\t{$src, ${dst}{${mask}}{z}|${dst}{${mask}}{z}, $src}"),
                    [(set DstRC:$dst,
                      (OpVT (X86VBroadcastm KRC:$mask, (SrcVT VR128X:$src))))]>,
                    EVEX, EVEX_KZ;
  def rm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set DstRC:$dst, 
                    (OpVT (X86VBroadcast (ld_frag addr:$src))))]>, EVEX;
  def krm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst), (ins KRC:$mask,
                                                         x86memop:$src),
                  !strconcat(OpcodeStr, 
                      "\t{$src, ${dst}{${mask}}{z}|${dst}{${mask}}{z}, $src}"),
                  [(set DstRC:$dst, (OpVT (X86VBroadcastm KRC:$mask, 
                                     (ld_frag addr:$src))))]>, EVEX, EVEX_KZ;
}

defm VPBROADCASTDZ  : avx512_int_broadcast_rm<0x58, "vpbroadcastd", i32mem,
                      loadi32, VR512, v16i32, v4i32, VK16WM>,
                      EVEX_V512, EVEX_CD8<32, CD8VT1>;
defm VPBROADCASTQZ  : avx512_int_broadcast_rm<0x59, "vpbroadcastq", i64mem,
                      loadi64, VR512, v8i64, v2i64, VK8WM>,  EVEX_V512, VEX_W,
                      EVEX_CD8<64, CD8VT1>;

def : Pat<(v16f32 (X86VBroadcast (v4f32 VR128X:$src))),
          (VBROADCASTSSZrr VR128X:$src)>;
def : Pat<(v8f64 (X86VBroadcast (v2f64 VR128X:$src))),
          (VBROADCASTSDZrr VR128X:$src)>;
    
// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v16f32 (X86VBroadcast FR32X:$src)),
          (VBROADCASTSSZrr (COPY_TO_REGCLASS FR32X:$src, VR128X))>;
def : Pat<(v8f64 (X86VBroadcast FR64X:$src)),
          (VBROADCASTSDZrr (COPY_TO_REGCLASS FR64X:$src, VR128X))>;


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
                       RegisterClass DstRC, RegisterClass KRC,
                       ValueType OpVT, ValueType SrcVT> {
def rr : AVX512XS8I<opc, MRMDestReg, (outs DstRC:$dst), (ins KRC:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  []>, EVEX;
}

defm VPBROADCASTMW2D : avx512_mask_broadcast<0x3A, "vpbroadcastmw2d", VR512,
                                             VK16, v16i32, v16i1>, EVEX_V512;
defm VPBROADCASTMB2Q : avx512_mask_broadcast<0x2A, "vpbroadcastmb2q", VR512,
                                            VK8, v8i64, v8i1>, EVEX_V512, VEX_W;

// Mask register copy, including
// - copy between mask registers
// - load/store mask registers
// - copy from GPR to mask register and vice versa
//
multiclass avx512_mask_mov<bits<8> opc_kk, bits<8> opc_km, bits<8> opc_mk,
                         string OpcodeStr, RegisterClass KRC,
                         ValueType vt, X86MemOperand x86memop> {
  let neverHasSideEffects = 1 in {
    def kk : I<opc_kk, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
    let mayLoad = 1 in
    def km : I<opc_km, MRMSrcMem, (outs KRC:$dst), (ins x86memop:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (vt (load addr:$src)))]>;
    let mayStore = 1 in
    def mk : I<opc_mk, MRMDestMem, (outs), (ins x86memop:$dst, KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
  }
}

multiclass avx512_mask_mov_gpr<bits<8> opc_kr, bits<8> opc_rk,
                             string OpcodeStr,
                             RegisterClass KRC, RegisterClass GRC> {
  let neverHasSideEffects = 1 in {
    def kr : I<opc_kr, MRMSrcReg, (outs KRC:$dst), (ins GRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
    def rk : I<opc_rk, MRMSrcReg, (outs GRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
  }
}

let Predicates = [HasAVX512] in {
  defm KMOVW : avx512_mask_mov<0x90, 0x90, 0x91, "kmovw", VK16, v16i1, i16mem>,
               VEX, TB;
  defm KMOVW : avx512_mask_mov_gpr<0x92, 0x93, "kmovw", VK16, GR32>,
               VEX, TB;
}

let Predicates = [HasAVX512] in {
  // GR16 from/to 16-bit mask
  def : Pat<(v16i1 (bitconvert (i16 GR16:$src))),
            (KMOVWkr (SUBREG_TO_REG (i32 0), GR16:$src, sub_16bit))>;
  def : Pat<(i16 (bitconvert (v16i1 VK16:$src))),
            (EXTRACT_SUBREG (KMOVWrk VK16:$src), sub_16bit)>;

  // Store kreg in memory
  def : Pat<(store (v16i1 VK16:$src), addr:$dst),
            (KMOVWmk addr:$dst, VK16:$src)>;

  def : Pat<(store (v8i1 VK8:$src), addr:$dst),
            (KMOVWmk addr:$dst, (v16i1 (COPY_TO_REGCLASS VK8:$src, VK16)))>;
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
}

// Mask unary operation
// - KNOT
multiclass avx512_mask_unop<bits<8> opc, string OpcodeStr,
                         RegisterClass KRC, SDPatternOperator OpNode> {
  let Predicates = [HasAVX512] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (OpNode KRC:$src))]>;
}

multiclass avx512_mask_unop_w<bits<8> opc, string OpcodeStr,
                               SDPatternOperator OpNode> {
  defm W : avx512_mask_unop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                          VEX, TB;
}

defm KNOT : avx512_mask_unop_w<0x44, "knot", not>;

def : Pat<(xor VK16:$src1, (v16i1 immAllOnesV)), (KNOTWrr VK16:$src1)>;
def : Pat<(xor VK8:$src1,  (v8i1 immAllOnesV)),
          (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$src1, VK16)), VK8)>;

// With AVX-512, 8-bit mask is promoted to 16-bit mask.
def : Pat<(not VK8:$src),
          (COPY_TO_REGCLASS
            (KNOTWrr (COPY_TO_REGCLASS VK8:$src, VK16)), VK8)>;

// Mask binary operation
// - KADD, KAND, KANDN, KOR, KXNOR, KXOR
multiclass avx512_mask_binop<bits<8> opc, string OpcodeStr,
                           RegisterClass KRC, SDPatternOperator OpNode> {
  let Predicates = [HasAVX512] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               [(set KRC:$dst, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_binop_w<bits<8> opc, string OpcodeStr,
                             SDPatternOperator OpNode> {
  defm W : avx512_mask_binop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                           VEX_4V, VEX_L, TB;
}

def andn : PatFrag<(ops node:$i0, node:$i1), (and (not node:$i0), node:$i1)>;
def xnor : PatFrag<(ops node:$i0, node:$i1), (not (xor node:$i0, node:$i1))>;

let isCommutable = 1 in {
  defm KADD  : avx512_mask_binop_w<0x4a, "kadd",  add>;
  defm KAND  : avx512_mask_binop_w<0x41, "kand",  and>;
  let isCommutable = 0 in
  defm KANDN : avx512_mask_binop_w<0x42, "kandn", andn>;
  defm KOR   : avx512_mask_binop_w<0x45, "kor",   or>;
  defm KXNOR : avx512_mask_binop_w<0x46, "kxnor", xnor>;
  defm KXOR  : avx512_mask_binop_w<0x47, "kxor",  xor>;
}

multiclass avx512_mask_binop_int<string IntName, string InstName> {
  let Predicates = [HasAVX512] in
    def : Pat<(!cast<Intrinsic>("int_x86_"##IntName##"_v16i1")
                VK16:$src1, VK16:$src2),
              (!cast<Instruction>(InstName##"Wrr") VK16:$src1, VK16:$src2)>;
}

defm : avx512_mask_binop_int<"kadd",  "KADD">;
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
                           RegisterClass KRC1, RegisterClass KRC2> {
  let Predicates = [HasAVX512] in
    def rr : I<opc, MRMSrcReg, (outs KRC1:$dst), (ins KRC2:$src1, KRC2:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>;
}

multiclass avx512_mask_unpck_bw<bits<8> opc, string OpcodeStr> {
  defm BW : avx512_mask_unpck<opc, !strconcat(OpcodeStr, "bw"), VK16, VK8>,
                            VEX_4V, VEX_L, OpSize, TB;
}

defm KUNPCK : avx512_mask_unpck_bw<0x4b, "kunpck">;

multiclass avx512_mask_unpck_int<string IntName, string InstName> {
  let Predicates = [HasAVX512] in
    def : Pat<(!cast<Intrinsic>("int_x86_"##IntName##"_v16i1")
                VK8:$src1, VK8:$src2),
              (!cast<Instruction>(InstName##"BWrr") VK8:$src1, VK8:$src2)>;
}

defm : avx512_mask_unpck_int<"kunpck", "KUNPCK">;
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
                            VEX, TB;
}

defm KORTEST : avx512_mask_testop_w<0x98, "kortest", X86kortest>;
defm KTEST   : avx512_mask_testop_w<0x99, "ktest", X86ktest>;

// Mask shift
multiclass avx512_mask_shiftop<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                             SDNode OpNode> {
  let Predicates = [HasAVX512] in
    def ri : Ii8<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src, i8imm:$imm),
                 !strconcat(OpcodeStr,
                            "\t{$imm, $src, $dst|$dst, $src, $imm}"),
                            [(set KRC:$dst, (OpNode KRC:$src, (i8 imm:$imm)))]>;
}

multiclass avx512_mask_shiftop_w<bits<8> opc1, bits<8> opc2, string OpcodeStr,
                               SDNode OpNode> {
  defm W : avx512_mask_shiftop<opc1, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                             VEX, OpSize, TA, VEX_W;
}

defm KSHIFTL : avx512_mask_shiftop_w<0x32, 0x33, "kshiftl", shl>;
defm KSHIFTR : avx512_mask_shiftop_w<0x30, 0x31, "kshiftr", srl>;

// Mask setting all 0s or 1s
multiclass avx512_mask_setop<RegisterClass KRC, ValueType VT, PatFrag Val> {
  let Predicates = [HasAVX512] in
    let isReMaterializable = 1, isAsCheapAsAMove = 1, isPseudo = 1 in
      def #NAME# : I<0, Pseudo, (outs KRC:$dst), (ins), "",
                     [(set KRC:$dst, (VT Val))]>;
}

multiclass avx512_mask_setop_w<PatFrag Val> {
  defm B : avx512_mask_setop<VK8,  v8i1, Val>;
  defm W : avx512_mask_setop<VK16, v16i1, Val>;
}

defm KSET0 : avx512_mask_setop_w<immAllZerosV>;
defm KSET1 : avx512_mask_setop_w<immAllOnesV>;

// With AVX-512 only, 8-bit mask is promoted to 16-bit mask.
let Predicates = [HasAVX512] in {
  def : Pat<(v8i1 immAllZerosV), (COPY_TO_REGCLASS (KSET0W), VK8)>;
  def : Pat<(v8i1 immAllOnesV),  (COPY_TO_REGCLASS (KSET1W), VK8)>;
}
def : Pat<(v8i1 (extract_subvector (v16i1 VK16:$src), (iPTR 0))),
          (v8i1 (COPY_TO_REGCLASS VK16:$src, VK8))>;

def : Pat<(v16i1 (insert_subvector undef, (v8i1 VK8:$src), (iPTR 0))),
          (v16i1 (COPY_TO_REGCLASS VK8:$src, VK16))>;

def : Pat<(v8i1 (extract_subvector (v16i1 VK16:$src), (iPTR 8))),
          (v8i1 (COPY_TO_REGCLASS (KSHIFTRWri VK16:$src, (i8 8)), VK8))>;
