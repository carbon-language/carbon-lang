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
