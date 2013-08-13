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

//===----------------------------------------------------------------------===//
// AVX-512 - VPERM
//
// -- immediate form --
multiclass avx512_perm_imm<bits<8> opc, string OpcodeStr, RegisterClass RC,
                         SDNode OpNode, PatFrag mem_frag, 
                         X86MemOperand x86memop, ValueType OpVT> {
  def ri : AVX512AIi8<opc, MRMSrcReg, (outs RC:$dst),
                     (ins RC:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode RC:$src1, (i8 imm:$src2))))]>,
                     EVEX;
  def mi : AVX512AIi8<opc, MRMSrcMem, (outs RC:$dst),
                     (ins x86memop:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode (mem_frag addr:$src1),
                              (i8 imm:$src2))))]>, EVEX;
}

defm VPERMQZ  : avx512_perm_imm<0x00, "vpermq", VR512, X86VPermi, memopv8i64,
                        i512mem, v8i64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
let ExeDomain = SSEPackedDouble in 
defm VPERMPDZ  : avx512_perm_imm<0x01, "vpermpd", VR512, X86VPermi, memopv8f64, 
                        f512mem, v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

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
                     (OpVT (X86VPermv RC:$src1,
                            (bitconvert (mem_frag addr:$src2)))))]>, EVEX_4V;
}

defm VPERMDZ   : avx512_perm<0x36, "vpermd",  VR512,  memopv8i64, i512mem, 
                           v16i32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMQZ   : avx512_perm<0x36, "vpermq",  VR512,  memopv8i64,  i512mem, 
                           v8i64>,  EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
let ExeDomain = SSEPackedSingle in
defm VPERMPSZ  : avx512_perm<0x16, "vpermps", VR512,  memopv8f64, f512mem, 
                           v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
let ExeDomain = SSEPackedDouble in
defm VPERMPDZ  : avx512_perm<0x16, "vpermpd", VR512,  memopv8f64, f512mem, 
                           v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

// -- VPERM2I - 3 source operands form --
multiclass avx512_perm_3src<bits<8> opc, string OpcodeStr, RegisterClass RC,
                          PatFrag mem_frag, X86MemOperand x86memop,
                          ValueType OpVT> {
let Constraints = "$src1 = $dst" in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                       "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (X86VPermv3 RC:$src1, RC:$src2, RC:$src3)))]>,
                    EVEX_4V;

  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (X86VPermv3 RC:$src1, RC:$src2, 
                      (bitconvert (mem_frag addr:$src3)))))]>, EVEX_4V;
  }
}
defm VPERMI2D  : avx512_perm_3src<0x76, "vpermi2d",  VR512, memopv16i32, i512mem, 
                               v16i32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMI2Q  : avx512_perm_3src<0x76, "vpermi2q",  VR512, memopv8i64, i512mem, 
                               v8i64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMI2PS : avx512_perm_3src<0x77, "vpermi2ps",  VR512, memopv16f32, i512mem, 
                               v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMI2PD : avx512_perm_3src<0x77, "vpermi2pd",  VR512, memopv8f64, i512mem, 
                               v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512 - BLEND using mask
//
multiclass avx512_blendmask<bits<8> opc, string OpcodeStr,
                          RegisterClass KRC, RegisterClass RC,
                          X86MemOperand x86memop, PatFrag mem_frag,
                          SDNode OpNode, ValueType vt> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
               (ins KRC:$mask, RC:$src1, RC:$src2),
               !strconcat(OpcodeStr,
                "\t{$src2, $src1, ${dst}{${mask}}|${dst}{${mask}}, $src1, $src2}"),
               [(set RC:$dst, (OpNode KRC:$mask, (vt RC:$src2), 
                 (vt RC:$src1)))]>, EVEX_4V, EVEX_K;

  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
               (ins KRC:$mask, RC:$src1, x86memop:$src2),
               !strconcat(OpcodeStr,
                "\t{$src2, $src1, $mask, $dst|$dst, $mask, $src1, $src2}"),
               []>, 
               EVEX_4V, EVEX_K;
}

let ExeDomain = SSEPackedSingle in
defm VBLENDMPSZ : avx512_blendmask<0x65, "vblendmps", VK16WM, VR512, f512mem,
                              memopv16f32, vselect, v16f32>, 
                              EVEX_CD8<32, CD8VF>, EVEX_V512;
let ExeDomain = SSEPackedDouble in
defm VBLENDMPDZ : avx512_blendmask<0x65, "vblendmpd", VK8WM, VR512, f512mem,
                              memopv8f64, vselect, v8f64>, 
                              VEX_W, EVEX_CD8<64, CD8VF>, EVEX_V512;

defm VPBLENDMDZ : avx512_blendmask<0x64, "vpblendmd", VK16WM, VR512, f512mem, 
                                memopv8i64, vselect, v16i32>, 
                                EVEX_CD8<32, CD8VF>, EVEX_V512;

defm VPBLENDMQZ : avx512_blendmask<0x64, "vpblendmq", VK8WM, VR512, f512mem, 
                                memopv8i64, vselect, v8i64>, VEX_W, 
                                EVEX_CD8<64, CD8VF>, EVEX_V512;


let Predicates = [HasAVX512] in {
def : Pat<(v8f32 (vselect (v8i1 VK8WM:$mask), (v8f32 VR256X:$src1),
                            (v8f32 VR256X:$src2))),
            (EXTRACT_SUBREG 
              (v16f32 (VBLENDMPSZrr (COPY_TO_REGCLASS VK8WM:$mask, VK16WM), 
            (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v8i32 (vselect (v8i1 VK8WM:$mask), (v8i32 VR256X:$src1),
                            (v8i32 VR256X:$src2))),
            (EXTRACT_SUBREG 
                (v16i32 (VPBLENDMDZrr (COPY_TO_REGCLASS VK8WM:$mask, VK16WM), 
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;
}

multiclass avx512_icmp_packed<bits<8> opc, string OpcodeStr, RegisterClass KRC, 
              RegisterClass RC, X86MemOperand x86memop, PatFrag memop_frag, 
              SDNode OpNode, ValueType vt> {
  def rr : AVX512BI<opc, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2), 
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), (vt RC:$src2)))], 
             IIC_SSE_CMPP_RR>, EVEX_4V;
  def rm : AVX512BI<opc, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2), 
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), 
              (bitconvert (memop_frag addr:$src2))))],
             IIC_SSE_CMPP_RM>, EVEX_4V;
}

defm VPCMPEQDZ : avx512_icmp_packed<0x76, "vpcmpeqd", VK16, VR512, i512mem, 
                           memopv8i64, X86pcmpeqm, v16i32>, EVEX_V512;
defm VPCMPEQQZ : avx512_icmp_packed<0x29, "vpcmpeqq", VK8, VR512, i512mem, 
                           memopv8i64, X86pcmpeqm, v8i64>, T8, EVEX_V512, VEX_W;

defm VPCMPGTDZ : avx512_icmp_packed<0x66, "vpcmpgtd", VK16, VR512, i512mem, 
                           memopv8i64, X86pcmpgtm, v16i32>, EVEX_V512;
defm VPCMPGTQZ : avx512_icmp_packed<0x37, "vpcmpgtq", VK8, VR512, i512mem, 
                           memopv8i64, X86pcmpgtm, v8i64>, T8, EVEX_V512, VEX_W;

def : Pat<(v8i1 (X86pcmpgtm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2))),
            (COPY_TO_REGCLASS (VPCMPGTDZrr 
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm))), VK8)>;

def : Pat<(v8i1 (X86pcmpeqm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2))),
            (COPY_TO_REGCLASS (VPCMPEQDZrr 
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)),
            (v16i32 (SUBREG_TO_REG (i32 0), VR256X:$src2, sub_ymm))), VK8)>;


multiclass avx512_icmp_cc<bits<8> opc, RegisterClass KRC,
              RegisterClass RC, X86MemOperand x86memop, PatFrag memop_frag, 
              SDNode OpNode, ValueType vt, Operand CC, string asm,
              string asm_alt> {
  def rri : AVX512AIi8<opc, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2, CC:$cc), asm,
             [(set KRC:$dst, (OpNode (vt RC:$src1), (vt RC:$src2), imm:$cc))], 
             IIC_SSE_CMPP_RR>, EVEX_4V;
  def rmi : AVX512AIi8<opc, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, CC:$cc), asm,
             [(set KRC:$dst, (OpNode (vt RC:$src1), 
              (bitconvert (memop_frag addr:$src2)), imm:$cc))],
             IIC_SSE_CMPP_RM>, EVEX_4V;
  // Accept explicit immediate argument form instead of comparison code.
  let neverHasSideEffects = 1 in {
    def rri_alt : AVX512AIi8<opc, MRMSrcReg,
               (outs RC:$dst), (ins RC:$src1, RC:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_CMPP_RR>, EVEX_4V;
    def rmi_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs RC:$dst), (ins RC:$src1, x86memop:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_CMPP_RM>, EVEX_4V;
  }
}

defm VPCMPDZ : avx512_icmp_cc<0x1F, VK16, VR512, i512mem, memopv8i64,
                              X86cmpm, v16i32, AVXCC,
              "vpcmp${cc}d\t{$src2, $src1, $dst|$dst, $src1, $src2}",
              "vpcmpd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
              EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPCMPUDZ : avx512_icmp_cc<0x1E, VK16, VR512, i512mem, memopv8i64,
                               X86cmpmu, v16i32, AVXCC,
              "vpcmp${cc}ud\t{$src2, $src1, $dst|$dst, $src1, $src2}",
              "vpcmpud\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
              EVEX_V512, EVEX_CD8<32, CD8VF>;

defm VPCMPQZ : avx512_icmp_cc<0x1F, VK8, VR512, i512mem, memopv8i64,
                              X86cmpm, v8i64, AVXCC,
              "vpcmp${cc}q\t{$src2, $src1, $dst|$dst, $src1, $src2}",
              "vpcmpq\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
              VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
defm VPCMPUQZ : avx512_icmp_cc<0x1E, VK8, VR512, i512mem, memopv8i64,
                               X86cmpmu, v8i64, AVXCC,
              "vpcmp${cc}uq\t{$src2, $src1, $dst|$dst, $src1, $src2}",
              "vpcmpuq\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
              VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

// avx512_cmp_packed - sse 1 & 2 compare packed instructions
multiclass avx512_cmp_packed<RegisterClass KRC, RegisterClass RC,
                           X86MemOperand x86memop, Operand CC,
                           SDNode OpNode, ValueType vt, string asm,
                           string asm_alt, Domain d> {
  def rri : AVX512PIi8<0xC2, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2, CC:$cc), asm,
             [(set KRC:$dst, (OpNode (vt RC:$src1), (vt RC:$src2), imm:$cc))], d>;
  def rmi : AVX512PIi8<0xC2, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, CC:$cc), asm,
             [(set KRC:$dst,
              (OpNode (vt RC:$src1), (memop addr:$src2), imm:$cc))], d>;

  // Accept explicit immediate argument form instead of comparison code.
  let neverHasSideEffects = 1 in {
    def rri_alt : PIi8<0xC2, MRMSrcReg,
               (outs RC:$dst), (ins RC:$src1, RC:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_CMPP_RR, d>;
    def rmi_alt : PIi8<0xC2, MRMSrcMem,
               (outs RC:$dst), (ins RC:$src1, x86memop:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_CMPP_RM, d>;
  }
}

defm VCMPPSZ : avx512_cmp_packed<VK16, VR512, f512mem, AVXCC, X86cmpm, v16f32,
               "vcmp${cc}ps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               "vcmpps\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
               SSEPackedSingle>, TB, EVEX_4V, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VCMPPDZ : avx512_cmp_packed<VK8, VR512, f512mem, AVXCC, X86cmpm, v8f64,
               "vcmp${cc}pd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
               "vcmppd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}",
               SSEPackedDouble>, TB, OpSize, EVEX_4V, VEX_W, EVEX_V512,
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

//===----------------------------------------------------------------------===//
// AVX-512 - Aligned and unaligned load and store
//

multiclass avx512_mov_packed<bits<8> opc, RegisterClass RC, RegisterClass KRC,
                            X86MemOperand x86memop, PatFrag ld_frag, 
                            string asm, Domain d> {
let neverHasSideEffects = 1 in
  def rr : AVX512PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"), [], d>,
              EVEX;
let canFoldAsLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
               [(set RC:$dst, (ld_frag addr:$src))], d>, EVEX;
let Constraints = "$src1 = $dst" in {
  def rrk : AVX512PI<opc, MRMSrcReg, (outs RC:$dst), 
                                     (ins RC:$src1, KRC:$mask, RC:$src2),
              !strconcat(asm, 
              "\t{$src2, ${dst}{${mask}}|${dst}{${mask}}, $src2}"), [], d>,
              EVEX, EVEX_K;
  def rmk : AVX512PI<opc, MRMSrcMem, (outs RC:$dst),
                                (ins RC:$src1, KRC:$mask, x86memop:$src2),
              !strconcat(asm, 
              "\t{$src2, ${dst}{${mask}}|${dst}{${mask}}, $src2}"),
               [], d>, EVEX, EVEX_K;
}
}

defm VMOVAPSZ : avx512_mov_packed<0x28, VR512, VK16WM, f512mem, alignedloadv16f32,
                              "vmovaps", SSEPackedSingle>,
                               EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVAPDZ : avx512_mov_packed<0x28, VR512, VK8WM, f512mem, alignedloadv8f64,
                              "vmovapd", SSEPackedDouble>,
                              OpSize, EVEX_V512, VEX_W,
                              EVEX_CD8<64, CD8VF>;
defm VMOVUPSZ : avx512_mov_packed<0x10, VR512, VK16WM, f512mem, loadv16f32,
                              "vmovups", SSEPackedSingle>,
                              TB, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVUPDZ : avx512_mov_packed<0x10, VR512, VK8WM, f512mem, loadv8f64,
                              "vmovupd", SSEPackedDouble>,
                               OpSize, EVEX_V512, VEX_W,
                               EVEX_CD8<64, CD8VF>;
def VMOVAPSZmr : AVX512PI<0x29, MRMDestMem, (outs), (ins f512mem:$dst, VR512:$src),
                    "vmovaps\t{$src, $dst|$dst, $src}",
                    [(alignedstore512 (v16f32 VR512:$src), addr:$dst)],
                    SSEPackedSingle>, EVEX, EVEX_V512, TB,
                    EVEX_CD8<32, CD8VF>;
def VMOVAPDZmr : AVX512PI<0x29, MRMDestMem, (outs), (ins f512mem:$dst, VR512:$src),
                    "vmovapd\t{$src, $dst|$dst, $src}",
                    [(alignedstore512 (v8f64 VR512:$src), addr:$dst)],
                    SSEPackedDouble>, EVEX, EVEX_V512,
                    OpSize, TB, VEX_W, EVEX_CD8<64, CD8VF>;
def VMOVUPSZmr : AVX512PI<0x11, MRMDestMem, (outs), (ins f512mem:$dst, VR512:$src),
                    "vmovups\t{$src, $dst|$dst, $src}",
                    [(store (v16f32 VR512:$src), addr:$dst)],
                    SSEPackedSingle>, EVEX, EVEX_V512, TB,
                    EVEX_CD8<32, CD8VF>;
def VMOVUPDZmr : AVX512PI<0x11, MRMDestMem, (outs), (ins f512mem:$dst, VR512:$src),
                    "vmovupd\t{$src, $dst|$dst, $src}",
                    [(store (v8f64 VR512:$src), addr:$dst)],
                    SSEPackedDouble>, EVEX, EVEX_V512,
                    OpSize, TB, VEX_W, EVEX_CD8<64, CD8VF>;

// Use vmovaps/vmovups for AVX-512 integer load/store.
// 512-bit load/store
def : Pat<(alignedloadv8i64 addr:$src),
          (VMOVAPSZrm addr:$src)>;
def : Pat<(loadv8i64 addr:$src),
          (VMOVUPSZrm addr:$src)>;

def : Pat<(alignedstore512 (v8i64  VR512:$src), addr:$dst), 
          (VMOVAPSZmr addr:$dst, VR512:$src)>;
def : Pat<(alignedstore512 (v16i32 VR512:$src), addr:$dst), 
          (VMOVAPSZmr addr:$dst, VR512:$src)>;

def : Pat<(store (v8i64  VR512:$src), addr:$dst), 
          (VMOVUPDZmr addr:$dst, VR512:$src)>;
def : Pat<(store (v16i32 VR512:$src), addr:$dst), 
          (VMOVUPSZmr addr:$dst, VR512:$src)>;

let neverHasSideEffects = 1 in {
  def VMOVDQA32rr  : AVX512BI<0x6F, MRMSrcReg, (outs VR512:$dst),
                             (ins VR512:$src),
                             "vmovdqa32\t{$src, $dst|$dst, $src}", []>,
                             EVEX, EVEX_V512;
  def VMOVDQA64rr  : AVX512BI<0x6F, MRMSrcReg, (outs VR512:$dst),
                             (ins VR512:$src),
                             "vmovdqa64\t{$src, $dst|$dst, $src}", []>,
                             EVEX, EVEX_V512, VEX_W;
let mayStore = 1 in {
  def VMOVDQA32mr  : AVX512BI<0x7F, MRMDestMem, (outs),
                     (ins i512mem:$dst, VR512:$src),
                     "vmovdqa32\t{$src, $dst|$dst, $src}", []>,
                     EVEX, EVEX_V512, EVEX_CD8<32, CD8VF>;
  def VMOVDQA64mr  : AVX512BI<0x7F, MRMDestMem, (outs),
                     (ins i512mem:$dst, VR512:$src),
                     "vmovdqa64\t{$src, $dst|$dst, $src}", []>,
                     EVEX, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
}
let mayLoad = 1 in {
def VMOVDQA32rm  : AVX512BI<0x6F, MRMSrcMem, (outs VR512:$dst), 
                           (ins i512mem:$src),
                           "vmovdqa32\t{$src, $dst|$dst, $src}", []>,
                           EVEX, EVEX_V512, EVEX_CD8<32, CD8VF>;
def VMOVDQA64rm  : AVX512BI<0x6F, MRMSrcMem, (outs VR512:$dst),
                           (ins i512mem:$src),
                           "vmovdqa64\t{$src, $dst|$dst, $src}", []>,
                           EVEX, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
}
}

multiclass avx512_mov_int<bits<8> opc, string asm, RegisterClass RC,
                          RegisterClass KRC,
                          PatFrag ld_frag, X86MemOperand x86memop> {
let neverHasSideEffects = 1 in
  def rr : AVX512XSI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"), []>,
              EVEX;
let canFoldAsLoad = 1 in
  def rm : AVX512XSI<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
               [(set RC:$dst, (ld_frag addr:$src))]>,
               EVEX;
let Constraints = "$src1 = $dst" in {
  def rrk : AVX512XSI<opc, MRMSrcReg, (outs RC:$dst),
                                      (ins RC:$src1, KRC:$mask, RC:$src2),
              !strconcat(asm, 
              "\t{$src2, ${dst}{${mask}}|${dst}{${mask}}, $src2}"), []>,
              EVEX, EVEX_K;
  def rmk : AVX512XSI<opc, MRMSrcMem, (outs RC:$dst),
                                  (ins RC:$src1, KRC:$mask, x86memop:$src2),
              !strconcat(asm, 
              "\t{$src2, ${dst}{${mask}}|${dst}{${mask}}, $src2}"),
               []>, EVEX, EVEX_K;
}
}

defm VMOVDQU32 : avx512_mov_int<0x6F, "vmovdqu32", VR512, VK16WM, memopv16i32, i512mem>,
                                EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVDQU64 : avx512_mov_int<0x6F, "vmovdqu64", VR512, VK8WM, memopv8i64, i512mem>,
                                EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

let AddedComplexity = 20 in {
def : Pat<(v16f32 (vselect VK16WM:$mask, (v16f32 VR512:$src1), 
                           (v16f32 VR512:$src2))),
                  (VMOVUPSZrrk VR512:$src2, VK16WM:$mask, VR512:$src1)>;
def : Pat<(v8f64 (vselect VK8WM:$mask, (v8f64 VR512:$src1), 
                           (v8f64 VR512:$src2))),
                  (VMOVUPDZrrk VR512:$src2, VK8WM:$mask, VR512:$src1)>;
def : Pat<(v16i32 (vselect VK16WM:$mask, (v16i32 VR512:$src1), 
                           (v16i32 VR512:$src2))),
                  (VMOVDQU32rrk VR512:$src2, VK16WM:$mask, VR512:$src1)>;
def : Pat<(v8i64 (vselect VK8WM:$mask, (v8i64 VR512:$src1), 
                           (v8i64 VR512:$src2))),
                  (VMOVDQU64rrk VR512:$src2, VK8WM:$mask, VR512:$src1)>;
}

