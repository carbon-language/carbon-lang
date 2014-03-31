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
// -- 32x8 form --
let hasSideEffects = 0, ExeDomain = SSEPackedSingle in {
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
let hasSideEffects = 0, ExeDomain = SSEPackedDouble in {
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
let hasSideEffects = 0 in {
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

let hasSideEffects = 0 in {
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
      "vinsertps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insrtps VR128X:$src1, VR128X:$src2, imm:$src3))]>,
      EVEX_4V;
def VINSERTPSzrm: AVX512AIi8<0x21, MRMSrcMem, (outs VR128X:$dst),
      (ins VR128X:$src1, f32mem:$src2, u32u8imm:$src3),
      "vinsertps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insrtps VR128X:$src1,
                          (v4f32 (scalar_to_vector (loadf32 addr:$src2))),
                          imm:$src3))]>, EVEX_4V, EVEX_CD8<32, CD8VT1>;

//===----------------------------------------------------------------------===//
// AVX-512 VECTOR EXTRACT
//---
let hasSideEffects = 0, ExeDomain = SSEPackedSingle in {
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

let hasSideEffects = 0 in {
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
      "vextractps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set GR32:$dst, (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2))]>,
      EVEX;

def VEXTRACTPSzmr : AVX512AIi8<0x17, MRMDestMem, (outs),
      (ins f32mem:$dst, VR128X:$src1, u32u8imm:$src2),
      "vextractps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(store (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2),
                          addr:$dst)]>, EVEX, EVEX_CD8<32, CD8VT1>;

//===---------------------------------------------------------------------===//
// AVX-512 BROADCAST
//---
multiclass avx512_fp_broadcast<bits<8> opc, string OpcodeStr, 
                         RegisterClass DestRC,
                         RegisterClass SrcRC, X86MemOperand x86memop> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs DestRC:$dst), (ins SrcRC:$src),
         !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
         []>, EVEX;
  def rm : AVX5128I<opc, MRMSrcMem, (outs DestRC:$dst), (ins x86memop:$src),
        !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),[]>, EVEX;
}
let ExeDomain = SSEPackedSingle in {
  defm VBROADCASTSSZ  : avx512_fp_broadcast<0x18, "vbroadcastss", VR512,
                                       VR128X, f32mem>,
                                       EVEX_V512, EVEX_CD8<32, CD8VT1>;
}

let ExeDomain = SSEPackedDouble in {
  defm VBROADCASTSDZ  : avx512_fp_broadcast<0x19, "vbroadcastsd", VR512,
                                       VR128X, f64mem>,
                                       EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;
}

def : Pat<(v16f32 (X86VBroadcast (loadf32 addr:$src))),
          (VBROADCASTSSZrm addr:$src)>;
def : Pat<(v8f64 (X86VBroadcast (loadf64 addr:$src))),
          (VBROADCASTSDZrm addr:$src)>;

def : Pat<(int_x86_avx512_vbroadcast_ss_512 addr:$src),
          (VBROADCASTSSZrm addr:$src)>;
def : Pat<(int_x86_avx512_vbroadcast_sd_512 addr:$src),
          (VBROADCASTSDZrm addr:$src)>;

multiclass avx512_int_broadcast_reg<bits<8> opc, string OpcodeStr,
                          RegisterClass SrcRC, RegisterClass KRC> {
  def Zrr : AVX5128I<opc, MRMSrcReg, (outs VR512:$dst), (ins SrcRC:$src),
                   !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                   []>, EVEX, EVEX_V512;
  def Zkrr : AVX5128I<opc, MRMSrcReg, (outs VR512:$dst), 
                   (ins KRC:$mask, SrcRC:$src),
                   !strconcat(OpcodeStr, 
                        " \t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
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
def : Pat<(v16i32 (X86VBroadcastm VK16WM:$mask, (i32 GR32:$src))),
        (VPBROADCASTDrZkrr VK16WM:$mask, GR32:$src)>;
def : Pat<(v8i64 (X86VBroadcast (i64 GR64:$src))),
        (VPBROADCASTQrZrr GR64:$src)>;
def : Pat<(v8i64 (X86VBroadcastm VK8WM:$mask, (i64 GR64:$src))),
        (VPBROADCASTQrZkrr VK8WM:$mask, GR64:$src)>;

def : Pat<(v16i32 (int_x86_avx512_pbroadcastd_i32_512 (i32 GR32:$src))),
        (VPBROADCASTDrZrr GR32:$src)>;
def : Pat<(v8i64 (int_x86_avx512_pbroadcastq_i64_512 (i64 GR64:$src))),
        (VPBROADCASTQrZrr GR64:$src)>;

def : Pat<(v16i32 (int_x86_avx512_mask_pbroadcast_d_gpr_512 (i32 GR32:$src),
                   (v16i32 immAllZerosV), (i16 GR16:$mask))),
          (VPBROADCASTDrZkrr (COPY_TO_REGCLASS GR16:$mask, VK16WM), GR32:$src)>;
def : Pat<(v8i64 (int_x86_avx512_mask_pbroadcast_q_gpr_512 (i64 GR64:$src),
                   (bc_v8i64 (v16i32 immAllZerosV)), (i8 GR8:$mask))),
          (VPBROADCASTQrZkrr (COPY_TO_REGCLASS GR8:$mask, VK8WM), GR64:$src)>;

multiclass avx512_int_broadcast_rm<bits<8> opc, string OpcodeStr,
                          X86MemOperand x86memop, PatFrag ld_frag,
                          RegisterClass DstRC, ValueType OpVT, ValueType SrcVT,
                          RegisterClass KRC> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst), (ins VR128X:$src),
                  !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                  [(set DstRC:$dst,
                    (OpVT (X86VBroadcast (SrcVT VR128X:$src))))]>, EVEX;
  def krr : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst), (ins KRC:$mask,
                                                         VR128X:$src),
                    !strconcat(OpcodeStr, 
                    " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
                    [(set DstRC:$dst,
                      (OpVT (X86VBroadcastm KRC:$mask, (SrcVT VR128X:$src))))]>,
                    EVEX, EVEX_KZ;
  let mayLoad = 1 in {
  def rm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
                  !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                  [(set DstRC:$dst, 
                    (OpVT (X86VBroadcast (ld_frag addr:$src))))]>, EVEX;
  def krm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst), (ins KRC:$mask,
                                                         x86memop:$src),
                  !strconcat(OpcodeStr, 
                      " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
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

def : Pat<(v16i32 (int_x86_avx512_pbroadcastd_512 (v4i32 VR128X:$src))),
          (VPBROADCASTDZrr VR128X:$src)>;
def : Pat<(v8i64 (int_x86_avx512_pbroadcastq_512 (v2i64 VR128X:$src))),
          (VPBROADCASTQZrr VR128X:$src)>;

def : Pat<(v16f32 (X86VBroadcast (v4f32 VR128X:$src))),
          (VBROADCASTSSZrr VR128X:$src)>;
def : Pat<(v8f64 (X86VBroadcast (v2f64 VR128X:$src))),
          (VBROADCASTSDZrr VR128X:$src)>;

def : Pat<(v16f32 (int_x86_avx512_vbroadcast_ss_ps_512 (v4f32 VR128X:$src))),
          (VBROADCASTSSZrr VR128X:$src)>;
def : Pat<(v8f64 (int_x86_avx512_vbroadcast_sd_pd_512 (v2f64 VR128X:$src))),
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
                  !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
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
                         " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode RC:$src1, (i8 imm:$src2))))]>,
                     EVEX;
  def mi : AVX512AIi8<opc, MRMSrcMem, (outs RC:$dst),
                     (ins x86memop:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
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
                       " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set RC:$dst,
                     (OpVT (X86VPermv RC:$src1, RC:$src2)))]>, EVEX_4V;

  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, x86memop:$src2),
                   !strconcat(OpcodeStr,
                       " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                   [(set RC:$dst,
                     (OpVT (X86VPermv RC:$src1, (mem_frag addr:$src2))))]>,
                     EVEX_4V;
}

defm VPERMDZ   : avx512_perm<0x36, "vpermd",  VR512,  memopv16i32, i512mem,
                           v16i32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMQZ   : avx512_perm<0x36, "vpermq",  VR512,  memopv8i64,  i512mem, 
                           v8i64>,  EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
let ExeDomain = SSEPackedSingle in
defm VPERMPSZ  : avx512_perm<0x16, "vpermps", VR512,  memopv16f32, f512mem,
                           v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
let ExeDomain = SSEPackedDouble in
defm VPERMPDZ  : avx512_perm<0x16, "vpermpd", VR512,  memopv8f64, f512mem, 
                           v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

// -- VPERM2I - 3 source operands form --
multiclass avx512_perm_3src<bits<8> opc, string OpcodeStr, RegisterClass RC,
                          PatFrag mem_frag, X86MemOperand x86memop,
                          SDNode OpNode, ValueType OpVT> {
let Constraints = "$src1 = $dst" in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                       " \t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src1, RC:$src2, RC:$src3)))]>,
                    EVEX_4V;

  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                    " \t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src1, RC:$src2, 
                      (mem_frag addr:$src3))))]>, EVEX_4V;
  }
}
defm VPERMI2D  : avx512_perm_3src<0x76, "vpermi2d",  VR512, memopv16i32, i512mem, 
                               X86VPermiv3, v16i32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMI2Q  : avx512_perm_3src<0x76, "vpermi2q",  VR512, memopv8i64, i512mem, 
                               X86VPermiv3, v8i64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMI2PS : avx512_perm_3src<0x77, "vpermi2ps",  VR512, memopv16f32, i512mem, 
                               X86VPermiv3, v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMI2PD : avx512_perm_3src<0x77, "vpermi2pd",  VR512, memopv8f64, i512mem, 
                               X86VPermiv3, v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPERMT2D  : avx512_perm_3src<0x7E, "vpermt2d",  VR512, memopv16i32, i512mem, 
                               X86VPermv3, v16i32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMT2Q  : avx512_perm_3src<0x7E, "vpermt2q",  VR512, memopv8i64, i512mem, 
                               X86VPermv3, v8i64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMT2PS : avx512_perm_3src<0x7F, "vpermt2ps",  VR512, memopv16f32, i512mem, 
                               X86VPermv3, v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPERMT2PD : avx512_perm_3src<0x7F, "vpermt2pd",  VR512, memopv8f64, i512mem, 
                               X86VPermv3, v8f64>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
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
             " \t{$src2, $src1, ${dst} {${mask}}|${dst} {${mask}}, $src1, $src2}"),
             [(set RC:$dst, (OpNode KRC:$mask, (vt RC:$src2),
                 (vt RC:$src1)))]>, EVEX_4V, EVEX_K;
  let mayLoad = 1 in
  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
             (ins KRC:$mask, RC:$src1, x86memop:$src2),
             !strconcat(OpcodeStr,
             " \t{$src2, $src1, ${dst} {${mask}}|${dst} {${mask}}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_K;
}

let ExeDomain = SSEPackedSingle in
defm VBLENDMPSZ : avx512_blendmask<0x65, "vblendmps", 
                              VK16WM, VR512, f512mem,
                              memopv16f32, vselect, v16f32>, 
                              EVEX_CD8<32, CD8VF>, EVEX_V512;
let ExeDomain = SSEPackedDouble in
defm VBLENDMPDZ : avx512_blendmask<0x65, "vblendmpd", 
                              VK8WM, VR512, f512mem,
                              memopv8f64, vselect, v8f64>, 
                              VEX_W, EVEX_CD8<64, CD8VF>, EVEX_V512;

def : Pat<(v16f32 (int_x86_avx512_mask_blend_ps_512 (v16f32 VR512:$src1),
                 (v16f32 VR512:$src2), (i16 GR16:$mask))),
        (VBLENDMPSZrr (COPY_TO_REGCLASS GR16:$mask, VK16WM),
         VR512:$src1, VR512:$src2)>;

def : Pat<(v8f64 (int_x86_avx512_mask_blend_pd_512 (v8f64 VR512:$src1),
                 (v8f64 VR512:$src2), (i8 GR8:$mask))),
        (VBLENDMPDZrr (COPY_TO_REGCLASS GR8:$mask, VK8WM),
         VR512:$src1, VR512:$src2)>;

defm VPBLENDMDZ : avx512_blendmask<0x64, "vpblendmd", 
                              VK16WM, VR512, f512mem, 
                              memopv16i32, vselect, v16i32>, 
                              EVEX_CD8<32, CD8VF>, EVEX_V512;

defm VPBLENDMQZ : avx512_blendmask<0x64, "vpblendmq", 
                              VK8WM, VR512, f512mem, 
                              memopv8i64, vselect, v8i64>, 
                              VEX_W, EVEX_CD8<64, CD8VF>, EVEX_V512;

def : Pat<(v16i32 (int_x86_avx512_mask_blend_d_512 (v16i32 VR512:$src1),
                 (v16i32 VR512:$src2), (i16 GR16:$mask))),
        (VPBLENDMDZrr (COPY_TO_REGCLASS GR16:$mask, VK16),
         VR512:$src1, VR512:$src2)>;

def : Pat<(v8i64 (int_x86_avx512_mask_blend_q_512 (v8i64 VR512:$src1),
                 (v8i64 VR512:$src2), (i8 GR8:$mask))),
        (VPBLENDMQZrr (COPY_TO_REGCLASS GR8:$mask, VK8),
         VR512:$src1, VR512:$src2)>;

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
//===----------------------------------------------------------------------===//
// Compare Instructions
//===----------------------------------------------------------------------===//

// avx512_cmp_scalar - AVX512 CMPSS and CMPSD
multiclass avx512_cmp_scalar<RegisterClass RC, X86MemOperand x86memop,
                            Operand CC, SDNode OpNode, ValueType VT,
                            PatFrag ld_frag, string asm, string asm_alt> {
  def rr : AVX512Ii8<0xC2, MRMSrcReg,
                (outs VK1:$dst), (ins RC:$src1, RC:$src2, CC:$cc), asm,
                [(set VK1:$dst, (OpNode (VT RC:$src1), RC:$src2, imm:$cc))],
                IIC_SSE_ALU_F32S_RR>, EVEX_4V;
  def rm : AVX512Ii8<0xC2, MRMSrcMem,
                (outs VK1:$dst), (ins RC:$src1, x86memop:$src2, CC:$cc), asm,
                [(set VK1:$dst, (OpNode (VT RC:$src1),
                (ld_frag addr:$src2), imm:$cc))], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512Ii8<0xC2, MRMSrcReg,
               (outs VK1:$dst), (ins RC:$src1, RC:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_ALU_F32S_RR>, EVEX_4V;
    def rmi_alt : AVX512Ii8<0xC2, MRMSrcMem,
               (outs VK1:$dst), (ins RC:$src1, x86memop:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  }
}

let Predicates = [HasAVX512] in {
defm VCMPSSZ : avx512_cmp_scalar<FR32X, f32mem, AVXCC, X86cmpms, f32, loadf32,
                 "vcmp${cc}ss\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                 "vcmpss\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
                 XS;
defm VCMPSDZ : avx512_cmp_scalar<FR64X, f64mem, AVXCC, X86cmpms, f64, loadf64,
                 "vcmp${cc}sd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                 "vcmpsd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
                 XD, VEX_W;
}

multiclass avx512_icmp_packed<bits<8> opc, string OpcodeStr, RegisterClass KRC, 
              RegisterClass RC, X86MemOperand x86memop, PatFrag memop_frag, 
              SDNode OpNode, ValueType vt> {
  def rr : AVX512BI<opc, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2), 
             !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), (vt RC:$src2)))], 
             IIC_SSE_ALU_F32P_RR>, EVEX_4V;
  def rm : AVX512BI<opc, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2), 
             !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), (memop_frag addr:$src2)))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V;
}

defm VPCMPEQDZ : avx512_icmp_packed<0x76, "vpcmpeqd", VK16, VR512, i512mem, 
                           memopv16i32, X86pcmpeqm, v16i32>, EVEX_V512,
                           EVEX_CD8<32, CD8VF>;
defm VPCMPEQQZ : avx512_icmp_packed<0x29, "vpcmpeqq", VK8, VR512, i512mem, 
                           memopv8i64, X86pcmpeqm, v8i64>, T8PD, EVEX_V512,
                           VEX_W, EVEX_CD8<64, CD8VF>;

defm VPCMPGTDZ : avx512_icmp_packed<0x66, "vpcmpgtd", VK16, VR512, i512mem, 
                           memopv16i32, X86pcmpgtm, v16i32>, EVEX_V512,
                           EVEX_CD8<32, CD8VF>;
defm VPCMPGTQZ : avx512_icmp_packed<0x37, "vpcmpgtq", VK8, VR512, i512mem, 
                           memopv8i64, X86pcmpgtm, v8i64>, T8PD, EVEX_V512,
                           VEX_W, EVEX_CD8<64, CD8VF>;

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
             IIC_SSE_ALU_F32P_RR>, EVEX_4V;
  def rmi : AVX512AIi8<opc, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, CC:$cc), asm,
             [(set KRC:$dst, (OpNode (vt RC:$src1), (memop_frag addr:$src2),
                              imm:$cc))], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512AIi8<opc, MRMSrcReg,
               (outs KRC:$dst), (ins RC:$src1, RC:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_ALU_F32P_RR>, EVEX_4V;
    def rmi_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, i8imm:$cc),
               asm_alt, [], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  }
}

defm VPCMPDZ : avx512_icmp_cc<0x1F, VK16, VR512, i512mem, memopv16i32,
                              X86cmpm, v16i32, AVXCC,
              "vpcmp${cc}d\t{$src2, $src1, $dst|$dst, $src1, $src2}",
              "vpcmpd\t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}">,
              EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPCMPUDZ : avx512_icmp_cc<0x1E, VK16, VR512, i512mem, memopv16i32,
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
                           X86MemOperand x86memop, ValueType vt,
                           string suffix, Domain d> {
  def rri : AVX512PIi8<0xC2, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2, AVXCC:$cc),
             !strconcat("vcmp${cc}", suffix,
                        " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (X86cmpm (vt RC:$src1), (vt RC:$src2), imm:$cc))], d>;
  def rrib: AVX512PIi8<0xC2, MRMSrcReg,
             (outs KRC:$dst), (ins RC:$src1, RC:$src2, AVXCC:$cc),
     !strconcat("vcmp${cc}", suffix,
                " \t{{sae}, $src2, $src1, $dst|$dst, $src1, $src2, {sae}}"),
                [], d>, EVEX_B;
  def rmi : AVX512PIi8<0xC2, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2, AVXCC:$cc),
              !strconcat("vcmp${cc}", suffix,
                         " \t{$src2, $src1, $dst|$dst, $src1, $src2, $cc}"),
             [(set KRC:$dst,
              (X86cmpm (vt RC:$src1), (memop addr:$src2), imm:$cc))], d>;

  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512PIi8<0xC2, MRMSrcReg,
               (outs RC:$dst), (ins RC:$src1, RC:$src2, i8imm:$cc),
              !strconcat("vcmp", suffix,
                        " \t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}"), [], d>;
    def rmi_alt : AVX512PIi8<0xC2, MRMSrcMem,
               (outs RC:$dst), (ins RC:$src1, x86memop:$src2, i8imm:$cc),
              !strconcat("vcmp", suffix,
                        " \t{$cc, $src2, $src1, $dst|$dst, $src1, $src2, $cc}"), [], d>;
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
                (v16f32 VR512:$src2), imm:$cc, (i16 -1),
                 FROUND_NO_EXC)),
          (COPY_TO_REGCLASS (VCMPPSZrrib VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR16)>;
           
def : Pat<(i8 (int_x86_avx512_mask_cmp_pd_512 (v8f64 VR512:$src1),
                (v8f64 VR512:$src2), imm:$cc, (i8 -1),
                 FROUND_NO_EXC)),
          (COPY_TO_REGCLASS (VCMPPDZrrib VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR8)>;

def : Pat<(i16 (int_x86_avx512_mask_cmp_ps_512 (v16f32 VR512:$src1),
                (v16f32 VR512:$src2), imm:$cc, (i16 -1),
                FROUND_CURRENT)),
          (COPY_TO_REGCLASS (VCMPPSZrri VR512:$src1, VR512:$src2,
                             (I8Imm imm:$cc)), GR16)>;

def : Pat<(i8 (int_x86_avx512_mask_cmp_pd_512 (v8f64 VR512:$src1),
                (v8f64 VR512:$src2), imm:$cc, (i8 -1),
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
                         ValueType vt, X86MemOperand x86memop> {
  let hasSideEffects = 0 in {
    def kk : I<opc_kk, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"), []>;
    let mayLoad = 1 in
    def km : I<opc_km, MRMSrcMem, (outs KRC:$dst), (ins x86memop:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (vt (load addr:$src)))]>;
    let mayStore = 1 in
    def mk : I<opc_mk, MRMDestMem, (outs), (ins x86memop:$dst, KRC:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"), []>;
  }
}

multiclass avx512_mask_mov_gpr<bits<8> opc_kr, bits<8> opc_rk,
                             string OpcodeStr,
                             RegisterClass KRC, RegisterClass GRC> {
  let hasSideEffects = 0 in {
    def kr : I<opc_kr, MRMSrcReg, (outs KRC:$dst), (ins GRC:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"), []>;
    def rk : I<opc_rk, MRMSrcReg, (outs GRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"), []>;
  }
}

let Predicates = [HasAVX512] in {
  defm KMOVW : avx512_mask_mov<0x90, 0x90, 0x91, "kmovw", VK16, v16i1, i16mem>,
               VEX, PS;
  defm KMOVW : avx512_mask_mov_gpr<0x92, 0x93, "kmovw", VK16, GR32>,
               VEX, PS;
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

  def : Pat<(store VK8:$src, addr:$dst),
            (KMOVWmk addr:$dst, (COPY_TO_REGCLASS VK8:$src, VK16))>;

  def : Pat<(i1 (load addr:$src)),
            (COPY_TO_REGCLASS (KMOVWkm addr:$src), VK1)>;

  def : Pat<(v8i1 (load addr:$src)),
            (COPY_TO_REGCLASS (KMOVWkm addr:$src), VK8)>;

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

// Mask unary operation
// - KNOT
multiclass avx512_mask_unop<bits<8> opc, string OpcodeStr,
                         RegisterClass KRC, SDPatternOperator OpNode> {
  let Predicates = [HasAVX512] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (OpNode KRC:$src))]>;
}

multiclass avx512_mask_unop_w<bits<8> opc, string OpcodeStr,
                               SDPatternOperator OpNode> {
  defm W : avx512_mask_unop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                          VEX, PS;
}

defm KNOT : avx512_mask_unop_w<0x44, "knot", not>;

multiclass avx512_mask_unop_int<string IntName, string InstName> {
  let Predicates = [HasAVX512] in
    def : Pat<(!cast<Intrinsic>("int_x86_avx512_"##IntName##"_w")
                (i16 GR16:$src)),
              (COPY_TO_REGCLASS (!cast<Instruction>(InstName##"Wrr")
              (v16i1 (COPY_TO_REGCLASS GR16:$src, VK16))), GR16)>;
}
defm : avx512_mask_unop_int<"knot", "KNOT">;

def : Pat<(xor VK16:$src1, (v16i1 immAllOnesV)), (KNOTWrr VK16:$src1)>;
def : Pat<(xor VK8:$src1,  (v8i1 immAllOnesV)),
          (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$src1, VK16)), VK8)>;

// With AVX-512, 8-bit mask is promoted to 16-bit mask.
def : Pat<(not VK8:$src),
          (COPY_TO_REGCLASS
            (KNOTWrr (COPY_TO_REGCLASS VK8:$src, VK16)), VK8)>;

// Mask binary operation
// - KAND, KANDN, KOR, KXNOR, KXOR
multiclass avx512_mask_binop<bits<8> opc, string OpcodeStr,
                           RegisterClass KRC, SDPatternOperator OpNode> {
  let Predicates = [HasAVX512] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr,
                          " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               [(set KRC:$dst, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_binop_w<bits<8> opc, string OpcodeStr,
                             SDPatternOperator OpNode> {
  defm W : avx512_mask_binop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                           VEX_4V, VEX_L, PS;
}

def andn : PatFrag<(ops node:$i0, node:$i1), (and (not node:$i0), node:$i1)>;
def xnor : PatFrag<(ops node:$i0, node:$i1), (not (xor node:$i0, node:$i1))>;

let isCommutable = 1 in {
  defm KAND  : avx512_mask_binop_w<0x41, "kand",  and>;
  let isCommutable = 0 in
  defm KANDN : avx512_mask_binop_w<0x42, "kandn", andn>;
  defm KOR   : avx512_mask_binop_w<0x45, "kor",   or>;
  defm KXNOR : avx512_mask_binop_w<0x46, "kxnor", xnor>;
  defm KXOR  : avx512_mask_binop_w<0x47, "kxor",  xor>;
}

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
                          " \t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>;
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
               !strconcat(OpcodeStr, " \t{$src2, $src1|$src1, $src2}"),
               [(set EFLAGS, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_testop_w<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm W : avx512_mask_testop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                            VEX, PS;
}

defm KORTEST : avx512_mask_testop_w<0x98, "kortest", X86kortest>;

def : Pat<(X86cmp VK1:$src1, (i1 0)),
          (KORTESTWrr (COPY_TO_REGCLASS VK1:$src1, VK16),
           (COPY_TO_REGCLASS VK1:$src1, VK16))>;

// Mask shift
multiclass avx512_mask_shiftop<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                             SDNode OpNode> {
  let Predicates = [HasAVX512] in
    def ri : Ii8<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src, i8imm:$imm),
                 !strconcat(OpcodeStr,
                            " \t{$imm, $src, $dst|$dst, $src, $imm}"),
                            [(set KRC:$dst, (OpNode KRC:$src, (i8 imm:$imm)))]>;
}

multiclass avx512_mask_shiftop_w<bits<8> opc1, bits<8> opc2, string OpcodeStr,
                               SDNode OpNode> {
  defm W : avx512_mask_shiftop<opc1, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                             VEX, TAPD, VEX_W;
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

def : Pat<(v8i1 (X86vshli VK8:$src, (i8 imm:$imm))),
          (v8i1 (COPY_TO_REGCLASS (KSHIFTLWri (COPY_TO_REGCLASS VK8:$src, VK16), (I8Imm $imm)), VK8))>;

def : Pat<(v8i1 (X86vsrli VK8:$src, (i8 imm:$imm))),
          (v8i1 (COPY_TO_REGCLASS (KSHIFTRWri (COPY_TO_REGCLASS VK8:$src, VK16), (I8Imm $imm)), VK8))>;
//===----------------------------------------------------------------------===//
// AVX-512 - Aligned and unaligned load and store
//

multiclass avx512_load<bits<8> opc, RegisterClass RC, RegisterClass KRC,
                            X86MemOperand x86memop, PatFrag ld_frag, 
                            string asm, Domain d,
                            ValueType vt, bit IsReMaterializable = 1> {
let hasSideEffects = 0 in {
  def rr : AVX512PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
              !strconcat(asm, " \t{$src, $dst|$dst, $src}"), [], d>,
              EVEX;
  def rrkz : AVX512PI<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src),
               !strconcat(asm,
               " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
               [], d>, EVEX, EVEX_KZ;
  }
  let canFoldAsLoad = 1, isReMaterializable = IsReMaterializable in
  def rm : AVX512PI<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, " \t{$src, $dst|$dst, $src}"),
               [(set (vt RC:$dst), (ld_frag addr:$src))], d>, EVEX;
  let Constraints = "$src1 = $dst",  hasSideEffects = 0 in {
  def rrk : AVX512PI<opc, MRMSrcReg, (outs RC:$dst), 
                                     (ins RC:$src1, KRC:$mask, RC:$src2),
              !strconcat(asm, 
              " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"), [], d>,
              EVEX, EVEX_K;
  let mayLoad = 1 in
  def rmk : AVX512PI<opc, MRMSrcMem, (outs RC:$dst),
                                (ins RC:$src1, KRC:$mask, x86memop:$src2),
              !strconcat(asm, 
              " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
               [], d>, EVEX, EVEX_K;
  }
  let mayLoad = 1 in
  def rmkz : AVX512PI<opc, MRMSrcMem, (outs RC:$dst),
                      (ins KRC:$mask, x86memop:$src2),
              !strconcat(asm,
              " \t{$src2, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src2}"),
               [], d>, EVEX, EVEX_KZ;
}

multiclass avx512_store<bits<8> opc, RegisterClass RC, RegisterClass KRC,
                            X86MemOperand x86memop, PatFrag store_frag,
                            string asm, Domain d, ValueType vt> {
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
  def rr_alt : AVX512PI<opc, MRMDestReg, (outs RC:$dst), (ins RC:$src),
              !strconcat(asm, " \t{$src, $dst|$dst, $src}"), [], d>,
              EVEX;
  let Constraints = "$src1 = $dst" in
  def alt_rrk : AVX512PI<opc, MRMDestReg, (outs  RC:$dst),
                                     (ins RC:$src1, KRC:$mask, RC:$src2),
              !strconcat(asm,
              " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"), [], d>,
              EVEX, EVEX_K;
  def alt_rrkz : AVX512PI<opc, MRMDestReg, (outs  RC:$dst),
                                           (ins KRC:$mask, RC:$src),
              !strconcat(asm,
              " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
              [], d>, EVEX, EVEX_KZ;
  }
  let mayStore = 1 in {
  def mr : AVX512PI<opc, MRMDestMem, (outs), (ins x86memop:$dst, RC:$src),
              !strconcat(asm, " \t{$src, $dst|$dst, $src}"),
               [(store_frag (vt RC:$src), addr:$dst)], d>, EVEX;
  def mrk : AVX512PI<opc, MRMDestMem, (outs),
                                (ins x86memop:$dst, KRC:$mask, RC:$src),
              !strconcat(asm,
              " \t{$src, ${dst} {${mask}}|${dst} {${mask}}, $src}"),
               [], d>, EVEX, EVEX_K;
  def mrkz : AVX512PI<opc, MRMDestMem, (outs),
                      (ins x86memop:$dst, KRC:$mask, RC:$src),
              !strconcat(asm,
              " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
               [], d>, EVEX, EVEX_KZ;
  }
}

defm VMOVAPSZ : avx512_load<0x28, VR512, VK16WM, f512mem, alignedloadv16f32,
                              "vmovaps", SSEPackedSingle, v16f32>,
                avx512_store<0x29, VR512, VK16WM, f512mem, alignedstore512,
                              "vmovaps", SSEPackedSingle, v16f32>,
                               PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVAPDZ : avx512_load<0x28, VR512, VK8WM, f512mem, alignedloadv8f64,
                              "vmovapd", SSEPackedDouble, v8f64>,
                avx512_store<0x29, VR512, VK8WM, f512mem, alignedstore512,
                              "vmovapd", SSEPackedDouble, v8f64>,
                              PD, EVEX_V512, VEX_W,
                              EVEX_CD8<64, CD8VF>;
defm VMOVUPSZ : avx512_load<0x10, VR512, VK16WM, f512mem, loadv16f32,
                              "vmovups", SSEPackedSingle, v16f32>,
                avx512_store<0x11, VR512, VK16WM, f512mem, store,
                              "vmovups", SSEPackedSingle, v16f32>,
                              PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVUPDZ : avx512_load<0x10, VR512, VK8WM, f512mem, loadv8f64,
                              "vmovupd", SSEPackedDouble, v8f64, 0>,
                avx512_store<0x11, VR512, VK8WM, f512mem, store,
                              "vmovupd", SSEPackedDouble, v8f64>,
                               PD, EVEX_V512, VEX_W,
                               EVEX_CD8<64, CD8VF>;
def: Pat<(v8f64 (int_x86_avx512_mask_loadu_pd_512 addr:$ptr,
                 (bc_v8f64 (v16i32 immAllZerosV)), GR8:$mask)),
       (VMOVUPDZrmkz (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)), addr:$ptr)>;

def: Pat<(v16f32 (int_x86_avx512_mask_loadu_ps_512 addr:$ptr,
                 (bc_v16f32 (v16i32 immAllZerosV)), GR16:$mask)),
       (VMOVUPSZrmkz (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)), addr:$ptr)>;

def: Pat<(int_x86_avx512_mask_storeu_ps_512 addr:$ptr, (v16f32 VR512:$src),
          GR16:$mask),
         (VMOVUPSZmrk addr:$ptr, (v16i1 (COPY_TO_REGCLASS GR16:$mask, VK16WM)),
            VR512:$src)>;
def: Pat<(int_x86_avx512_mask_storeu_pd_512 addr:$ptr, (v8f64 VR512:$src),
          GR8:$mask),
         (VMOVUPDZmrk addr:$ptr, (v8i1 (COPY_TO_REGCLASS GR8:$mask, VK8WM)),
            VR512:$src)>;

defm VMOVDQA32: avx512_load<0x6F, VR512, VK16WM, i512mem, alignedloadv16i32,
                              "vmovdqa32", SSEPackedInt, v16i32>,
                avx512_store<0x7F, VR512, VK16WM, i512mem, alignedstore512,
                              "vmovdqa32", SSEPackedInt, v16i32>,
                               PD, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVDQA64: avx512_load<0x6F, VR512, VK8WM, i512mem, alignedloadv8i64,
                              "vmovdqa64", SSEPackedInt, v8i64>,
                avx512_store<0x7F, VR512, VK8WM, i512mem, alignedstore512,
                              "vmovdqa64", SSEPackedInt, v8i64>,
                               PD, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
defm VMOVDQU32: avx512_load<0x6F, VR512, VK16WM, i512mem, load,
                              "vmovdqu32", SSEPackedInt, v16i32>,
                avx512_store<0x7F, VR512, VK16WM, i512mem, store,
                              "vmovdqu32", SSEPackedInt, v16i32>,
                               XS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VMOVDQU64: avx512_load<0x6F, VR512, VK8WM, i512mem, load,
                              "vmovdqu64", SSEPackedInt, v8i64>,
                avx512_store<0x7F, VR512, VK8WM, i512mem, store,
                              "vmovdqu64", SSEPackedInt, v8i64>,
                               XS, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

let AddedComplexity = 20 in {
def : Pat<(v8i64 (vselect VK8WM:$mask, (v8i64 VR512:$src),
                           (bc_v8i64 (v16i32 immAllZerosV)))),
                  (VMOVDQU64rrkz VK8WM:$mask, VR512:$src)>;

def : Pat<(v8i64 (vselect VK8WM:$mask, (bc_v8i64 (v16i32 immAllZerosV)),
                  (v8i64 VR512:$src))),
   (VMOVDQU64rrkz (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$mask, VK16)),
                                              VK8), VR512:$src)>;

def : Pat<(v16i32 (vselect VK16WM:$mask, (v16i32 VR512:$src),
                           (v16i32 immAllZerosV))),
                  (VMOVDQU32rrkz VK16WM:$mask, VR512:$src)>;

def : Pat<(v16i32 (vselect VK16WM:$mask, (v16i32 immAllZerosV),
                   (v16i32 VR512:$src))),
   (VMOVDQU32rrkz (KNOTWrr VK16WM:$mask), VR512:$src)>;
                                              
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
              !strconcat(asm, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set VR128X:$dst, (vt (OpNode VR128X:$src1,
                                      (scalar_to_vector RC:$src2))))],
              IIC_SSE_MOV_S_RR>, EVEX_4V, VEX_LIG;
  let Constraints = "$src1 = $dst" in
  def rrk : SI<0x10, MRMSrcReg, (outs VR128X:$dst),
              (ins VR128X:$src1, VK1WM:$mask, RC:$src2, RC:$src3),
              !strconcat(asm,
                " \t{$src3, $src2, $dst {${mask}}|$dst {${mask}}, $src2, $src3}"),
              [], IIC_SSE_MOV_S_RR>, EVEX_4V, VEX_LIG, EVEX_K;
  def rm : SI<0x10, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
              !strconcat(asm, " \t{$src, $dst|$dst, $src}"),
              [(set RC:$dst, (mem_pat addr:$src))], IIC_SSE_MOV_S_RM>,
              EVEX, VEX_LIG;
  def mr: SI<0x11, MRMDestMem, (outs), (ins x86memop:$dst, RC:$src),
             !strconcat(asm, " \t{$src, $dst|$dst, $src}"),
             [(store RC:$src, addr:$dst)], IIC_SSE_MOV_S_MR>,
             EVEX, VEX_LIG;
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
// AVX-512 - Integer arithmetic
//
multiclass avx512_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, RegisterClass KRC,
                        RegisterClass RC, PatFrag memop_frag,
                        X86MemOperand x86memop, PatFrag scalar_mfrag,
                        X86MemOperand x86scalar_mop, string BrdcstStr,
                        OpndItins itins, bit IsCommutable = 0> {
  let isCommutable = IsCommutable in
    def rr : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
              (ins RC:$src1, RC:$src2),
              !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set RC:$dst, (OpVT (OpNode (OpVT RC:$src1), (OpVT RC:$src2))))],
              itins.rr>, EVEX_4V;
  let AddedComplexity = 30 in {
    let Constraints = "$src0 = $dst" in
      def rrk : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
                 (ins RC:$src0, KRC:$mask, RC:$src1, RC:$src2),
                 !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
                 [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                  (OpNode (OpVT RC:$src1), (OpVT RC:$src2)),
                                  RC:$src0)))],
                 itins.rr>, EVEX_4V, EVEX_K;
    def rrkz : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, RC:$src2),
                !strconcat(OpcodeStr, " \t{$src2, $src1, $dst {${mask}} {z}" ,
                    "|$dst {${mask}} {z}, $src1, $src2}"),
                [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                  (OpNode (OpVT RC:$src1), (OpVT RC:$src2)),
                                  (OpVT immAllZerosV))))],
                itins.rr>, EVEX_4V, EVEX_KZ;
  }

  let mayLoad = 1 in {
    def rm : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
              (ins RC:$src1, x86memop:$src2),
              !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set RC:$dst, (OpVT (OpNode (OpVT RC:$src1), (memop_frag addr:$src2))))],
              itins.rm>, EVEX_4V;
    let AddedComplexity = 30 in {
    let Constraints = "$src0 = $dst" in
      def rmk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                 (ins RC:$src0, KRC:$mask, RC:$src1, x86memop:$src2),
                 !strconcat(OpcodeStr,
                     " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
                 [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                    (OpNode (OpVT RC:$src1), (memop_frag addr:$src2)),
                                    RC:$src0)))],
                 itins.rm>, EVEX_4V, EVEX_K;
    def rmkz : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, x86memop:$src2),
                !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, $src2}"),
                [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                    (OpNode (OpVT RC:$src1), (memop_frag addr:$src2)),
                                    (OpVT immAllZerosV))))],
                itins.rm>, EVEX_4V, EVEX_KZ;
    }
    def rmb : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, x86scalar_mop:$src2),
               !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                          ", $src1, $dst|$dst, $src1, ${src2}", BrdcstStr, "}"),
               [(set RC:$dst, (OpNode RC:$src1,
                               (OpVT (X86VBroadcast (scalar_mfrag addr:$src2)))))],
               itins.rm>, EVEX_4V, EVEX_B;
    let AddedComplexity = 30 in {
    let Constraints = "$src0 = $dst" in
      def rmbk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                  (ins RC:$src0, KRC:$mask, RC:$src1, x86scalar_mop:$src2),
                  !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                             ", $src1, $dst {${mask}}|$dst {${mask}}, $src1, ${src2}",
                             BrdcstStr, "}"),
                  [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                    (OpNode (OpVT RC:$src1),
                                     (OpVT (X86VBroadcast (scalar_mfrag addr:$src2)))),
                                    RC:$src0)))],
                  itins.rm>, EVEX_4V, EVEX_B, EVEX_K;
    def rmbkz : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                 (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2),
                 !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                            ", $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, ${src2}",
                            BrdcstStr, "}"),
                 [(set RC:$dst, (OpVT (vselect KRC:$mask,
                                    (OpNode (OpVT RC:$src1),
                                     (OpVT (X86VBroadcast (scalar_mfrag addr:$src2)))),
                                    (OpVT immAllZerosV))))],
                 itins.rm>, EVEX_4V, EVEX_B, EVEX_KZ;
    }
  }
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
       !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       []>, EVEX_4V;
    def rrk : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
               (ins KRC:$mask, RC:$src1, RC:$src2),
               !strconcat(OpcodeStr,
                  " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
               [], itins.rr>, EVEX_4V, EVEX_K;
    def rrkz : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, RC:$src2),
                !strconcat(OpcodeStr, " \t{$src2, $src1, $dst {${mask}} {z}" ,
                    "|$dst {${mask}} {z}, $src1, $src2}"),
                [], itins.rr>, EVEX_4V, EVEX_KZ;
  }
  let mayLoad = 1 in {
    def rm : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
              (ins RC:$src1, x86memop:$src2),
              !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              []>, EVEX_4V;
    def rmk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
               (ins KRC:$mask, RC:$src1, x86memop:$src2),
               !strconcat(OpcodeStr,
                   " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
               [], itins.rm>, EVEX_4V, EVEX_K;
    def rmkz : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, x86memop:$src2),
                !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, $src2}"),
                [], itins.rm>, EVEX_4V, EVEX_KZ;
    def rmb : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, x86scalar_mop:$src2),
               !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                          ", $src1, $dst|$dst, $src1, ${src2}", BrdcstStr, "}"),
               [], itins.rm>, EVEX_4V, EVEX_B;
    def rmbk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2),
                !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                           ", $src1, $dst {${mask}}|$dst {${mask}}, $src1, ${src2}",
                           BrdcstStr, "}"),
                [], itins.rm>, EVEX_4V, EVEX_B, EVEX_K;
    def rmbkz : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
                 (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2),
                 !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                            ", $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, ${src2}",
                            BrdcstStr, "}"),
                 [], itins.rm>, EVEX_4V, EVEX_B, EVEX_KZ;
  }
}

defm VPADDDZ : avx512_binop_rm<0xFE, "vpaddd", add, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 1>, EVEX_V512, EVEX_CD8<32, CD8VF>;

defm VPSUBDZ : avx512_binop_rm<0xFA, "vpsubd", sub, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 0>, EVEX_V512, EVEX_CD8<32, CD8VF>;

defm VPMULLDZ : avx512_binop_rm<0x40, "vpmulld", mul, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 1>, T8PD, EVEX_V512, EVEX_CD8<32, CD8VF>;

defm VPADDQZ : avx512_binop_rm<0xD4, "vpaddq", add, v8i64, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 1>, EVEX_CD8<64, CD8VF>, EVEX_V512, VEX_W;

defm VPSUBQZ : avx512_binop_rm<0xFB, "vpsubq", sub, v8i64, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 0>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPMULDQZ : avx512_binop_rm2<0x28, "vpmuldq", v8i64, v16i32, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 1>, T8PD, EVEX_V512,
                   EVEX_CD8<64, CD8VF>, VEX_W;

defm VPMULUDQZ : avx512_binop_rm2<0xF4, "vpmuludq", v8i64, v16i32, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTMUL_ITINS_P, 1>, EVEX_V512, EVEX_CD8<64, CD8VF>, VEX_W;

def : Pat<(v8i64 (X86pmuludq (v16i32 VR512:$src1), (v16i32 VR512:$src2))),
          (VPMULUDQZrr VR512:$src1, VR512:$src2)>;

def : Pat<(v8i64 (int_x86_avx512_mask_pmulu_dq_512 (v16i32 VR512:$src1),
           (v16i32 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
          (VPMULUDQZrr VR512:$src1, VR512:$src2)>;
def : Pat<(v8i64 (int_x86_avx512_mask_pmul_dq_512 (v16i32 VR512:$src1),
           (v16i32 VR512:$src2), (bc_v8i64 (v16i32 immAllZerosV)), (i8 -1))),
          (VPMULDQZrr VR512:$src1, VR512:$src2)>;

defm VPMAXUDZ : avx512_binop_rm<0x3F, "vpmaxud", X86umax, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 1>,
                   T8PD, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPMAXUQZ : avx512_binop_rm<0x3F, "vpmaxuq", X86umax, v8i64, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 0>,
                   T8PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPMAXSDZ : avx512_binop_rm<0x3D, "vpmaxsd", X86smax, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 1>,
                   T8PD, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPMAXSQZ : avx512_binop_rm<0x3D, "vpmaxsq", X86smax, v8i64, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 0>,
                   T8PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPMINUDZ : avx512_binop_rm<0x3B, "vpminud", X86umin, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 1>,
                   T8PD, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPMINUQZ : avx512_binop_rm<0x3B, "vpminuq", X86umin, v8i64, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 0>,
                   T8PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPMINSDZ : avx512_binop_rm<0x39, "vpminsd", X86smin, v16i32, VK16WM, VR512,
                   memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                   SSE_INTALU_ITINS_P, 1>,
                   T8PD, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPMINSQZ : avx512_binop_rm<0x39, "vpminsq", X86smin, v8i64, VK8WM, VR512,
                   memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                   SSE_INTALU_ITINS_P, 0>,
                   T8PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

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

defm VUNPCKHPSZ: avx512_unpack_fp<0x15, X86Unpckh, v16f32, memopv8f64,
      VR512, f512mem, "vunpckhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedSingle>, PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VUNPCKHPDZ: avx512_unpack_fp<0x15, X86Unpckh, v8f64, memopv8f64,
      VR512, f512mem, "vunpckhpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedDouble>, PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VUNPCKLPSZ: avx512_unpack_fp<0x14, X86Unpckl, v16f32, memopv8f64,
      VR512, f512mem, "vunpcklps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedSingle>, PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VUNPCKLPDZ: avx512_unpack_fp<0x14, X86Unpckl, v8f64, memopv8f64,
      VR512, f512mem, "vunpcklpd\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      SSEPackedDouble>, PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

multiclass avx512_unpack_int<bits<8> opc, string OpcodeStr, SDNode OpNode,
                        ValueType OpVT, RegisterClass RC, PatFrag memop_frag,
                        X86MemOperand x86memop> {
  def rr : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, RC:$src2),
       !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (OpVT (OpNode (OpVT RC:$src1), (OpVT RC:$src2))))], 
       IIC_SSE_UNPCK>, EVEX_4V;
  def rm : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86memop:$src2),
       !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (OpVT (OpNode (OpVT RC:$src1),
                                     (bitconvert (memop_frag addr:$src2)))))],
                                     IIC_SSE_UNPCK>, EVEX_4V;
}
defm VPUNPCKLDQZ  : avx512_unpack_int<0x62, "vpunpckldq", X86Unpckl, v16i32,
                                VR512, memopv16i32, i512mem>, EVEX_V512,
                                EVEX_CD8<32, CD8VF>;
defm VPUNPCKLQDQZ : avx512_unpack_int<0x6C, "vpunpcklqdq", X86Unpckl, v8i64,
                                VR512, memopv8i64, i512mem>, EVEX_V512,
                                VEX_W, EVEX_CD8<64, CD8VF>;
defm VPUNPCKHDQZ  : avx512_unpack_int<0x6A, "vpunpckhdq", X86Unpckh, v16i32,
                                VR512, memopv16i32, i512mem>, EVEX_V512,
                                EVEX_CD8<32, CD8VF>;
defm VPUNPCKHQDQZ : avx512_unpack_int<0x6D, "vpunpckhqdq", X86Unpckh, v8i64,
                                VR512, memopv8i64, i512mem>, EVEX_V512,
                                VEX_W, EVEX_CD8<64, CD8VF>;
//===----------------------------------------------------------------------===//
// AVX-512 - PSHUFD
//

multiclass avx512_pshuf_imm<bits<8> opc, string OpcodeStr, RegisterClass RC,
                         SDNode OpNode, PatFrag mem_frag, 
                         X86MemOperand x86memop, ValueType OpVT> {
  def ri : AVX512Ii8<opc, MRMSrcReg, (outs RC:$dst),
                     (ins RC:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode RC:$src1, (i8 imm:$src2))))]>,
                     EVEX;
  def mi : AVX512Ii8<opc, MRMSrcMem, (outs RC:$dst),
                     (ins x86memop:$src1, i8imm:$src2),
                     !strconcat(OpcodeStr,
                         " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                       (OpVT (OpNode (mem_frag addr:$src1),
                              (i8 imm:$src2))))]>, EVEX;
}

defm VPSHUFDZ : avx512_pshuf_imm<0x70, "vpshufd", VR512, X86PShufd, memopv16i32,
                      i512mem, v16i32>, PD, EVEX_V512, EVEX_CD8<32, CD8VF>;

let ExeDomain = SSEPackedSingle in
defm VPERMILPSZ : avx512_pshuf_imm<0x04, "vpermilps", VR512, X86VPermilp,
                      memopv16f32, i512mem, v16f32>, TAPD, EVEX_V512,
                      EVEX_CD8<32, CD8VF>;
let ExeDomain = SSEPackedDouble in
defm VPERMILPDZ : avx512_pshuf_imm<0x05, "vpermilpd", VR512, X86VPermilp,
                      memopv8f64, i512mem, v8f64>, TAPD, EVEX_V512,
                      VEX_W, EVEX_CD8<32, CD8VF>;

def : Pat<(v16i32 (X86VPermilp VR512:$src1, (i8 imm:$imm))),
          (VPERMILPSZri VR512:$src1, imm:$imm)>;
def : Pat<(v8i64 (X86VPermilp VR512:$src1, (i8 imm:$imm))),
          (VPERMILPDZri VR512:$src1, imm:$imm)>;

//===----------------------------------------------------------------------===//
// AVX-512  Logical Instructions
//===----------------------------------------------------------------------===//

defm VPANDDZ : avx512_binop_rm<0xDB, "vpandd", and, v16i32, VK16WM, VR512, memopv16i32,
                      i512mem, loadi32, i32mem, "{1to16}", SSE_BIT_ITINS_P, 1>,
                      EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPANDQZ : avx512_binop_rm<0xDB, "vpandq", and, v8i64, VK8WM, VR512, memopv8i64,
                      i512mem, loadi64, i64mem, "{1to8}", SSE_BIT_ITINS_P, 1>,
                      EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPORDZ  : avx512_binop_rm<0xEB, "vpord", or, v16i32, VK16WM, VR512, memopv16i32,
                      i512mem, loadi32, i32mem, "{1to16}", SSE_BIT_ITINS_P, 1>,
                      EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPORQZ  : avx512_binop_rm<0xEB, "vporq", or, v8i64, VK8WM, VR512, memopv8i64,
                      i512mem, loadi64, i64mem, "{1to8}", SSE_BIT_ITINS_P, 1>,
                      EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPXORDZ : avx512_binop_rm<0xEF, "vpxord", xor, v16i32, VK16WM, VR512, memopv16i32,
                      i512mem, loadi32, i32mem, "{1to16}", SSE_BIT_ITINS_P, 1>,
                      EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPXORQZ : avx512_binop_rm<0xEF, "vpxorq", xor, v8i64, VK8WM, VR512, memopv8i64,
                      i512mem, loadi64, i64mem, "{1to8}", SSE_BIT_ITINS_P, 1>,
                      EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPANDNDZ : avx512_binop_rm<0xDF, "vpandnd", X86andnp, v16i32, VK16WM, VR512,
                      memopv16i32, i512mem, loadi32, i32mem, "{1to16}",
                      SSE_BIT_ITINS_P, 0>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPANDNQZ : avx512_binop_rm<0xDF, "vpandnq", X86andnp, v8i64, VK8WM, VR512,
                      memopv8i64, i512mem, loadi64, i64mem, "{1to8}",
                      SSE_BIT_ITINS_P, 0>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512  FP arithmetic
//===----------------------------------------------------------------------===//

multiclass avx512_binop_s<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SizeItins itins> {
  defm SSZ : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "ss"), OpNode, FR32X,
                             f32mem, itins.s, 0>, XS, EVEX_4V, VEX_LIG,
                             EVEX_CD8<32, CD8VT1>;
  defm SDZ : sse12_fp_scalar<opc, !strconcat(OpcodeStr, "sd"), OpNode, FR64X,
                             f64mem, itins.d, 0>, XD, VEX_W, EVEX_4V, VEX_LIG,
                             EVEX_CD8<64, CD8VT1>;
}

let isCommutable = 1 in {
defm VADD : avx512_binop_s<0x58, "add", fadd, SSE_ALU_ITINS_S>;
defm VMUL : avx512_binop_s<0x59, "mul", fmul, SSE_ALU_ITINS_S>;
defm VMIN : avx512_binop_s<0x5D, "min", X86fmin, SSE_ALU_ITINS_S>;
defm VMAX : avx512_binop_s<0x5F, "max", X86fmax, SSE_ALU_ITINS_S>;
}
let isCommutable = 0 in {
defm VSUB : avx512_binop_s<0x5C, "sub", fsub, SSE_ALU_ITINS_S>;
defm VDIV : avx512_binop_s<0x5E, "div", fdiv, SSE_ALU_ITINS_S>;
}

multiclass avx512_fp_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           RegisterClass KRC,
                           RegisterClass RC, ValueType vt,
                           X86MemOperand x86memop, PatFrag mem_frag,
                           X86MemOperand x86scalar_mop, PatFrag scalar_mfrag,
                           string BrdcstStr,
                           Domain d, OpndItins itins, bit commutable> {
  let isCommutable = commutable in {
    def rr : PI<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
       !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (vt (OpNode RC:$src1, RC:$src2)))], itins.rr, d>,
       EVEX_4V;

    def rrk: PI<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src1, RC:$src2),
       !strconcat(OpcodeStr,
           " \t{$src2, $src1, $dst {${mask}} |$dst {${mask}}, $src1, $src2}"),
       [], itins.rr, d>, EVEX_4V, EVEX_K;

    def rrkz: PI<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src1, RC:$src2),
       !strconcat(OpcodeStr,
           " \t{$src2, $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, $src2}"),
       [], itins.rr, d>, EVEX_4V, EVEX_KZ;
  }

  let mayLoad = 1 in {
    def rm : PI<opc, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
       !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (OpNode RC:$src1, (mem_frag addr:$src2)))],
          itins.rm, d>, EVEX_4V;

    def rmb : PI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, x86scalar_mop:$src2),
       !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
           ", $src1, $dst|$dst, $src1, ${src2}", BrdcstStr, "}"),
       [(set RC:$dst, (OpNode RC:$src1, 
                       (vt (X86VBroadcast (scalar_mfrag addr:$src2)))))],
       itins.rm, d>, EVEX_4V, EVEX_B;

    def rmk : PI<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, x86memop:$src2), !strconcat(OpcodeStr,
           "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
       [], itins.rm, d>, EVEX_4V, EVEX_K;

    def rmkz : PI<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, x86memop:$src2), !strconcat(OpcodeStr,
           "\t{$src2, $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, $src2}"),
       [], itins.rm, d>, EVEX_4V, EVEX_KZ;

    def rmbk : PI<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2), !strconcat(OpcodeStr,
           " \t{${src2}", BrdcstStr,
           ", $src1, $dst {${mask}}|$dst {${mask}}, $src1, ${src2}", BrdcstStr, "}"),
       [], itins.rm, d>, EVEX_4V, EVEX_B, EVEX_K;

    def rmbkz : PI<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, x86scalar_mop:$src2), !strconcat(OpcodeStr,
           " \t{${src2}", BrdcstStr,
           ", $src1, $dst {${mask}} {z}|$dst {${mask}} {z}, $src1, ${src2}",
           BrdcstStr, "}"),
       [], itins.rm, d>, EVEX_4V, EVEX_B, EVEX_KZ;
  }
}

defm VADDPSZ : avx512_fp_packed<0x58, "addps", fadd, VK16WM, VR512, v16f32, f512mem,
                   memopv16f32, f32mem, loadf32, "{1to16}", SSEPackedSingle, 
                   SSE_ALU_ITINS_P.s, 1>, EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
                   
defm VADDPDZ : avx512_fp_packed<0x58, "addpd", fadd, VK8WM, VR512, v8f64, f512mem,
                   memopv8f64, f64mem, loadf64, "{1to8}", SSEPackedDouble,
                   SSE_ALU_ITINS_P.d, 1>,
                   EVEX_V512, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMULPSZ : avx512_fp_packed<0x59, "mulps", fmul, VK16WM, VR512, v16f32, f512mem,
                   memopv16f32, f32mem, loadf32, "{1to16}", SSEPackedSingle,
                   SSE_ALU_ITINS_P.s, 1>, EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
defm VMULPDZ : avx512_fp_packed<0x59, "mulpd", fmul, VK8WM, VR512, v8f64, f512mem,
                   memopv8f64, f64mem, loadf64, "{1to8}", SSEPackedDouble,
                   SSE_ALU_ITINS_P.d, 1>,
                   EVEX_V512, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMINPSZ : avx512_fp_packed<0x5D, "minps", X86fmin, VK16WM, VR512, v16f32, f512mem,
                   memopv16f32, f32mem, loadf32, "{1to16}", SSEPackedSingle,
                   SSE_ALU_ITINS_P.s, 1>,
                   EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
defm VMAXPSZ : avx512_fp_packed<0x5F, "maxps", X86fmax, VK16WM, VR512, v16f32, f512mem,
                   memopv16f32, f32mem, loadf32, "{1to16}", SSEPackedSingle,
                   SSE_ALU_ITINS_P.s, 1>,
                   EVEX_V512, PS, EVEX_CD8<32, CD8VF>;

defm VMINPDZ : avx512_fp_packed<0x5D, "minpd", X86fmin, VK8WM, VR512, v8f64, f512mem,
                   memopv8f64, f64mem, loadf64, "{1to8}", SSEPackedDouble,
                   SSE_ALU_ITINS_P.d, 1>,
                   EVEX_V512, PD, VEX_W, EVEX_CD8<64, CD8VF>;
defm VMAXPDZ : avx512_fp_packed<0x5F, "maxpd", X86fmax, VK8WM, VR512, v8f64, f512mem,
                   memopv8f64, f64mem, loadf64, "{1to8}", SSEPackedDouble,
                   SSE_ALU_ITINS_P.d, 1>,
                   EVEX_V512, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VSUBPSZ : avx512_fp_packed<0x5C, "subps", fsub, VK16WM, VR512, v16f32, f512mem,
                   memopv16f32, f32mem, loadf32, "{1to16}", SSEPackedSingle,
                   SSE_ALU_ITINS_P.s, 0>, EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
defm VDIVPSZ : avx512_fp_packed<0x5E, "divps", fdiv, VK16WM, VR512, v16f32, f512mem,
                   memopv16f32, f32mem, loadf32, "{1to16}", SSEPackedSingle,
                   SSE_ALU_ITINS_P.s, 0>, EVEX_V512, PS, EVEX_CD8<32, CD8VF>;

defm VSUBPDZ : avx512_fp_packed<0x5C, "subpd", fsub, VK8WM, VR512, v8f64, f512mem,
                   memopv8f64, f64mem, loadf64, "{1to8}", SSEPackedDouble,
                   SSE_ALU_ITINS_P.d, 0>, 
                   EVEX_V512, PD, VEX_W, EVEX_CD8<64, CD8VF>;
defm VDIVPDZ : avx512_fp_packed<0x5E, "divpd", fdiv, VK8WM, VR512, v8f64, f512mem,
                   memopv8f64, f64mem, loadf64, "{1to8}", SSEPackedDouble,
                   SSE_ALU_ITINS_P.d, 0>, 
                   EVEX_V512, PD, VEX_W, EVEX_CD8<64, CD8VF>;

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
             !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), (vt RC:$src2)))],
             SSEPackedInt>, EVEX_4V;
  def rm : AVX512PI<opc, MRMSrcMem,
             (outs KRC:$dst), (ins RC:$src1, x86memop:$src2), 
             !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set KRC:$dst, (OpNode (vt RC:$src1), 
              (bitconvert (memop_frag addr:$src2))))], SSEPackedInt>, EVEX_4V;
}

defm VPTESTMDZ  : avx512_vptest<0x27, "vptestmd", VK16, VR512,  f512mem,
                              memopv16i32, X86testm, v16i32>, T8PD, EVEX_V512,
                              EVEX_CD8<32, CD8VF>;
defm VPTESTMQZ  : avx512_vptest<0x27, "vptestmq", VK8, VR512,  f512mem,
                              memopv8i64, X86testm, v8i64>, T8PD, EVEX_V512, VEX_W,
                              EVEX_CD8<64, CD8VF>;

let Predicates = [HasCDI] in {
defm VPTESTNMDZ  : avx512_vptest<0x27, "vptestnmd", VK16, VR512,  f512mem,
                              memopv16i32, X86testnm, v16i32>, T8XS, EVEX_V512,
                              EVEX_CD8<32, CD8VF>;
defm VPTESTNMQZ  : avx512_vptest<0x27, "vptestnmq", VK8, VR512,  f512mem,
                              memopv8i64, X86testnm, v8i64>, T8XS, EVEX_V512, VEX_W,
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
                         string OpcodeStr, SDNode OpNode, RegisterClass RC,
                         ValueType vt, X86MemOperand x86memop, PatFrag mem_frag,
                         RegisterClass KRC> {
  def ri : AVX512BIi8<opc, ImmFormR, (outs RC:$dst),
       (ins RC:$src1, i8imm:$src2),
           !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (vt (OpNode RC:$src1, (i8 imm:$src2))))],
        SSE_INTSHIFT_ITINS_P.rr>, EVEX_4V;
  def rik : AVX512BIi8<opc, ImmFormR, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, i8imm:$src2),
           !strconcat(OpcodeStr,
                " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
       [], SSE_INTSHIFT_ITINS_P.rr>, EVEX_4V, EVEX_K;
  def mi: AVX512BIi8<opc, ImmFormM, (outs RC:$dst),
       (ins x86memop:$src1, i8imm:$src2),
           !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (OpNode (mem_frag addr:$src1),
                     (i8 imm:$src2)))], SSE_INTSHIFT_ITINS_P.rm>, EVEX_4V;
  def mik: AVX512BIi8<opc, ImmFormM, (outs RC:$dst),
       (ins KRC:$mask, x86memop:$src1, i8imm:$src2),
           !strconcat(OpcodeStr,
                " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
       [], SSE_INTSHIFT_ITINS_P.rm>, EVEX_4V, EVEX_K;
}

multiclass avx512_shift_rrm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                          RegisterClass RC, ValueType vt, ValueType SrcVT,
                          PatFrag bc_frag, RegisterClass KRC> {
  // src2 is always 128-bit
  def rr : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, VR128X:$src2),
           !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (vt (OpNode RC:$src1, (SrcVT VR128X:$src2))))],
        SSE_INTSHIFT_ITINS_P.rr>, EVEX_4V;
  def rrk : AVX512BI<opc, MRMSrcReg, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, VR128X:$src2),
           !strconcat(OpcodeStr,
                " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
       [], SSE_INTSHIFT_ITINS_P.rr>, EVEX_4V, EVEX_K;
  def rm : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, i128mem:$src2),
           !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
       [(set RC:$dst, (vt (OpNode RC:$src1,
                       (bc_frag (memopv2i64 addr:$src2)))))],
                        SSE_INTSHIFT_ITINS_P.rm>, EVEX_4V;
  def rmk : AVX512BI<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, RC:$src1, i128mem:$src2),
           !strconcat(OpcodeStr,
                " \t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}"),
       [], SSE_INTSHIFT_ITINS_P.rm>, EVEX_4V, EVEX_K;
}

defm VPSRLDZ : avx512_shift_rmi<0x72, MRM2r, MRM2m, "vpsrld", X86vsrli,
                           VR512, v16i32, i512mem, memopv16i32, VK16WM>,
                           EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPSRLDZ : avx512_shift_rrm<0xD2, "vpsrld", X86vsrl,
                           VR512, v16i32, v4i32, bc_v4i32, VK16WM>, EVEX_V512,
                           EVEX_CD8<32, CD8VQ>;
                           
defm VPSRLQZ : avx512_shift_rmi<0x73, MRM2r, MRM2m, "vpsrlq", X86vsrli,
                           VR512, v8i64, i512mem, memopv8i64, VK8WM>, EVEX_V512,
                           EVEX_CD8<64, CD8VF>, VEX_W;
defm VPSRLQZ : avx512_shift_rrm<0xD3, "vpsrlq", X86vsrl,
                           VR512, v8i64, v2i64, bc_v2i64, VK8WM>, EVEX_V512,
                           EVEX_CD8<64, CD8VQ>, VEX_W;

defm VPSLLDZ : avx512_shift_rmi<0x72, MRM6r, MRM6m, "vpslld", X86vshli,
                           VR512, v16i32, i512mem, memopv16i32, VK16WM>, EVEX_V512,
                           EVEX_CD8<32, CD8VF>;
defm VPSLLDZ : avx512_shift_rrm<0xF2, "vpslld", X86vshl,
                           VR512, v16i32, v4i32, bc_v4i32, VK16WM>, EVEX_V512,
                           EVEX_CD8<32, CD8VQ>;
                           
defm VPSLLQZ : avx512_shift_rmi<0x73, MRM6r, MRM6m, "vpsllq", X86vshli,
                           VR512, v8i64, i512mem, memopv8i64, VK8WM>, EVEX_V512,
                           EVEX_CD8<64, CD8VF>, VEX_W;
defm VPSLLQZ : avx512_shift_rrm<0xF3, "vpsllq", X86vshl,
                           VR512, v8i64, v2i64, bc_v2i64, VK8WM>, EVEX_V512,
                           EVEX_CD8<64, CD8VQ>, VEX_W;

defm VPSRADZ : avx512_shift_rmi<0x72, MRM4r, MRM4m, "vpsrad", X86vsrai,
                           VR512, v16i32, i512mem, memopv16i32, VK16WM>,
                           EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VPSRADZ : avx512_shift_rrm<0xE2, "vpsrad", X86vsra,
                           VR512, v16i32, v4i32, bc_v4i32, VK16WM>, EVEX_V512,
                           EVEX_CD8<32, CD8VQ>;
                           
defm VPSRAQZ : avx512_shift_rmi<0x72, MRM4r, MRM4m, "vpsraq", X86vsrai,
                           VR512, v8i64, i512mem, memopv8i64, VK8WM>, EVEX_V512,
                           EVEX_CD8<64, CD8VF>, VEX_W;
defm VPSRAQZ : avx512_shift_rrm<0xE2, "vpsraq", X86vsra,
                           VR512, v8i64, v2i64, bc_v2i64, VK8WM>, EVEX_V512,
                           EVEX_CD8<64, CD8VQ>, VEX_W;

//===-------------------------------------------------------------------===//
// Variable Bit Shifts
//===-------------------------------------------------------------------===//
multiclass avx512_var_shift<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           RegisterClass RC, ValueType vt,
                           X86MemOperand x86memop, PatFrag mem_frag> {
  def rr  : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
             (ins RC:$src1, RC:$src2),
             !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst,
               (vt (OpNode RC:$src1, (vt RC:$src2))))]>,
             EVEX_4V;
  def rm  : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
             (ins RC:$src1, x86memop:$src2),
             !strconcat(OpcodeStr, " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst,
               (vt (OpNode RC:$src1, (mem_frag addr:$src2))))]>,
             EVEX_4V;
}

defm VPSLLVDZ : avx512_var_shift<0x47, "vpsllvd", shl, VR512, v16i32, 
                               i512mem, memopv16i32>, EVEX_V512,
                               EVEX_CD8<32, CD8VF>;
defm VPSLLVQZ : avx512_var_shift<0x47, "vpsllvq", shl, VR512, v8i64, 
                               i512mem, memopv8i64>, EVEX_V512, VEX_W,
                               EVEX_CD8<64, CD8VF>;
defm VPSRLVDZ : avx512_var_shift<0x45, "vpsrlvd", srl, VR512, v16i32, 
                               i512mem, memopv16i32>, EVEX_V512,
                               EVEX_CD8<32, CD8VF>;
defm VPSRLVQZ : avx512_var_shift<0x45, "vpsrlvq", srl, VR512, v8i64, 
                               i512mem, memopv8i64>, EVEX_V512, VEX_W,
                               EVEX_CD8<64, CD8VF>;
defm VPSRAVDZ : avx512_var_shift<0x46, "vpsravd", sra, VR512, v16i32, 
                               i512mem, memopv16i32>, EVEX_V512,
                               EVEX_CD8<32, CD8VF>;
defm VPSRAVQZ : avx512_var_shift<0x46, "vpsravq", sra, VR512, v8i64, 
                               i512mem, memopv8i64>, EVEX_V512, VEX_W,
                               EVEX_CD8<64, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512 - MOVDDUP
//===----------------------------------------------------------------------===//

multiclass avx512_movddup<string OpcodeStr, RegisterClass RC, ValueType VT, 
                        X86MemOperand x86memop, PatFrag memop_frag> {
def rr  : AVX512PDI<0x12, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                    !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                    [(set RC:$dst, (VT (X86Movddup RC:$src)))]>, EVEX;
def rm  : AVX512PDI<0x12, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                    !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                    [(set RC:$dst,
                      (VT (X86Movddup (memop_frag addr:$src))))]>, EVEX;
}

defm VMOVDDUPZ : avx512_movddup<"vmovddup", VR512, v8f64, f512mem, memopv8f64>,
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
                    !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                      [(set RC:$dst, (vt (OpNode RC:$src)))]>, EVEX;
  let mayLoad = 1 in
  def rm : AVX512XSI<op, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                    !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                      [(set RC:$dst, (OpNode (mem_frag addr:$src)))]>, EVEX;
}

defm VMOVSHDUPZ  : avx512_replicate_sfp<0x16, X86Movshdup, "vmovshdup",
                       v16f32, VR512, memopv16f32, f512mem>, EVEX_V512,
                       EVEX_CD8<32, CD8VF>;
defm VMOVSLDUPZ  : avx512_replicate_sfp<0x12, X86Movsldup, "vmovsldup",
                       v16f32, VR512, memopv16f32, f512mem>, EVEX_V512,
                       EVEX_CD8<32, CD8VF>;

def : Pat<(v16i32 (X86Movshdup VR512:$src)), (VMOVSHDUPZrr VR512:$src)>;
def : Pat<(v16i32 (X86Movshdup (memopv16i32 addr:$src))),
           (VMOVSHDUPZrm addr:$src)>;
def : Pat<(v16i32 (X86Movsldup VR512:$src)), (VMOVSLDUPZrr VR512:$src)>;
def : Pat<(v16i32 (X86Movsldup (memopv16i32 addr:$src))),
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
multiclass avx512_fma3p_rm<bits<8> opc, string OpcodeStr,
            RegisterClass RC, X86MemOperand x86memop,
            PatFrag mem_frag, X86MemOperand x86scalar_mop, PatFrag scalar_mfrag,
            string BrdcstStr, SDNode OpNode, ValueType OpVT> {
  def r: AVX512FMA3<opc, MRMSrcReg, (outs RC:$dst),
          (ins RC:$src1, RC:$src2, RC:$src3),
          !strconcat(OpcodeStr," \t{$src3, $src2, $dst|$dst, $src2, $src3}"),
          [(set RC:$dst, (OpVT(OpNode RC:$src1, RC:$src2, RC:$src3)))]>;

  let mayLoad = 1 in
  def m: AVX512FMA3<opc, MRMSrcMem, (outs RC:$dst),
          (ins RC:$src1, RC:$src2, x86memop:$src3),
          !strconcat(OpcodeStr, " \t{$src3, $src2, $dst|$dst, $src2, $src3}"),
          [(set RC:$dst, (OpVT (OpNode RC:$src1, RC:$src2,
                                               (mem_frag addr:$src3))))]>;
   def mb: AVX512FMA3<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, x86scalar_mop:$src3),
           !strconcat(OpcodeStr, " \t{${src3}", BrdcstStr,
            ", $src2, $dst|$dst, $src2, ${src3}", BrdcstStr, "}"),
           [(set RC:$dst, (OpNode RC:$src1, RC:$src2,
           (OpVT (X86VBroadcast (scalar_mfrag addr:$src3)))))]>, EVEX_B;
}
} // Constraints = "$src1 = $dst"

let ExeDomain = SSEPackedSingle in {
  defm VFMADD213PSZ    : avx512_fma3p_rm<0xA8, "vfmadd213ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmadd, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
  defm VFMSUB213PSZ    : avx512_fma3p_rm<0xAA, "vfmsub213ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmsub, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
  defm VFMADDSUB213PSZ : avx512_fma3p_rm<0xA6, "vfmaddsub213ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmaddsub, v16f32>,
                                    EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm VFMSUBADD213PSZ : avx512_fma3p_rm<0xA7, "vfmsubadd213ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmsubadd, v16f32>,
                                    EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm VFNMADD213PSZ   : avx512_fma3p_rm<0xAC, "vfnmadd213ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fnmadd, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
  defm VFNMSUB213PSZ   : avx512_fma3p_rm<0xAE, "vfnmsub213ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fnmsub, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFMADD213PDZ    : avx512_fma3p_rm<0xA8, "vfmadd213pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmadd, v8f64>, EVEX_V512,
                                    VEX_W, EVEX_CD8<64, CD8VF>;
  defm VFMSUB213PDZ    : avx512_fma3p_rm<0xAA, "vfmsub213pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmsub, v8f64>, EVEX_V512, VEX_W,
                                    EVEX_CD8<64, CD8VF>;
  defm VFMADDSUB213PDZ : avx512_fma3p_rm<0xA6, "vfmaddsub213pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmaddsub, v8f64>, EVEX_V512, VEX_W,
                                    EVEX_CD8<64, CD8VF>;
  defm VFMSUBADD213PDZ : avx512_fma3p_rm<0xA7, "vfmsubadd213pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmsubadd, v8f64>, EVEX_V512, VEX_W,
                                    EVEX_CD8<64, CD8VF>;
  defm VFNMADD213PDZ : avx512_fma3p_rm<0xAC, "vfnmadd213pd", VR512, f512mem,
                                  memopv8f64, f64mem, loadf64, "{1to8}",
                                  X86Fnmadd, v8f64>, EVEX_V512, VEX_W,
                                  EVEX_CD8<64, CD8VF>;
  defm VFNMSUB213PDZ : avx512_fma3p_rm<0xAE, "vfnmsub213pd", VR512, f512mem,
                                  memopv8f64, f64mem, loadf64, "{1to8}",
                                  X86Fnmsub, v8f64>, EVEX_V512, VEX_W,
                                  EVEX_CD8<64, CD8VF>;
}

let Constraints = "$src1 = $dst" in {
multiclass avx512_fma3p_m132<bits<8> opc, string OpcodeStr,
            RegisterClass RC, X86MemOperand x86memop,
            PatFrag mem_frag, X86MemOperand x86scalar_mop, PatFrag scalar_mfrag,
            string BrdcstStr, SDNode OpNode, ValueType OpVT> {
  let mayLoad = 1 in
  def m: AVX512FMA3<opc, MRMSrcMem, (outs RC:$dst),
          (ins RC:$src1, RC:$src3, x86memop:$src2),
          !strconcat(OpcodeStr, " \t{$src2, $src3, $dst|$dst, $src3, $src2}"),
          [(set RC:$dst, (OpVT (OpNode RC:$src1, (mem_frag addr:$src2), RC:$src3)))]>;
   def mb: AVX512FMA3<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, RC:$src3, x86scalar_mop:$src2),
           !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
            ", $src3, $dst|$dst, $src3, ${src2}", BrdcstStr, "}"),
           [(set RC:$dst, (OpNode RC:$src1, 
           (OpVT (X86VBroadcast (scalar_mfrag addr:$src2))), RC:$src3))]>, EVEX_B;
}
} // Constraints = "$src1 = $dst"


let ExeDomain = SSEPackedSingle in {
  defm VFMADD132PSZ    : avx512_fma3p_m132<0x98, "vfmadd132ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmadd, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
  defm VFMSUB132PSZ    : avx512_fma3p_m132<0x9A, "vfmsub132ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmsub, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
  defm VFMADDSUB132PSZ : avx512_fma3p_m132<0x96, "vfmaddsub132ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmaddsub, v16f32>,
                                    EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm VFMSUBADD132PSZ : avx512_fma3p_m132<0x97, "vfmsubadd132ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fmsubadd, v16f32>,
                                    EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm VFNMADD132PSZ   : avx512_fma3p_m132<0x9C, "vfnmadd132ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fnmadd, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
  defm VFNMSUB132PSZ   : avx512_fma3p_m132<0x9E, "vfnmsub132ps", VR512, f512mem,
                                    memopv16f32, f32mem, loadf32, "{1to16}",
                                    X86Fnmsub, v16f32>, EVEX_V512,
                                    EVEX_CD8<32, CD8VF>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFMADD132PDZ    : avx512_fma3p_m132<0x98, "vfmadd132pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmadd, v8f64>, EVEX_V512,
                                    VEX_W, EVEX_CD8<64, CD8VF>;
  defm VFMSUB132PDZ    : avx512_fma3p_m132<0x9A, "vfmsub132pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmsub, v8f64>, EVEX_V512, VEX_W,
                                    EVEX_CD8<64, CD8VF>;
  defm VFMADDSUB132PDZ : avx512_fma3p_m132<0x96, "vfmaddsub132pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmaddsub, v8f64>, EVEX_V512, VEX_W,
                                    EVEX_CD8<64, CD8VF>;
  defm VFMSUBADD132PDZ : avx512_fma3p_m132<0x97, "vfmsubadd132pd", VR512, f512mem,
                                    memopv8f64, f64mem, loadf64, "{1to8}",
                                    X86Fmsubadd, v8f64>, EVEX_V512, VEX_W,
                                    EVEX_CD8<64, CD8VF>;
  defm VFNMADD132PDZ : avx512_fma3p_m132<0x9C, "vfnmadd132pd", VR512, f512mem,
                                  memopv8f64, f64mem, loadf64, "{1to8}",
                                  X86Fnmadd, v8f64>, EVEX_V512, VEX_W,
                                  EVEX_CD8<64, CD8VF>;
  defm VFNMSUB132PDZ : avx512_fma3p_m132<0x9E, "vfnmsub132pd", VR512, f512mem,
                                  memopv8f64, f64mem, loadf64, "{1to8}",
                                  X86Fnmsub, v8f64>, EVEX_V512, VEX_W,
                                  EVEX_CD8<64, CD8VF>;
}

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
                              " \t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1, RC:$src3)))]>;
  let mayLoad = 1 in
  def m     : AVX512FMA3<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, f128mem:$src3),
                   !strconcat(OpcodeStr,
                              " \t{$src3, $src2, $dst|$dst, $src2, $src3}"),
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
              !strconcat(asm," \t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
              EVEX_4V;
  let mayLoad = 1 in
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins DstRC:$src1, x86memop:$src),
              !strconcat(asm," \t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
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
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (Int SrcRC:$src))]>, EVEX, VEX_LIG,
              Requires<[HasAVX512]>;
  let mayLoad = 1 in
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins memop:$src),
              !strconcat(asm," \t{$src, $dst|$dst, $src}"), []>, EVEX, VEX_LIG,
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
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (OpNode SrcRC:$src))]>, EVEX;
  def rm : SI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
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
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT SrcRC:$src))))], d>, EVEX;
  def rrb : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src, AVX512RC:$rc),
              !strconcat(asm," \t{$rc, $src, $dst|$dst, $src, $rc}"),
              [], d>, EVEX, EVEX_B, EVEX_RC;
  let mayLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
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
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT SrcRC:$src))))], d>, EVEX;
  let mayLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT (bitconvert (mem_frag addr:$src))))))], d>, EVEX;
} // hasSideEffects = 0
}

defm VCVTPD2PSZ : avx512_vcvt_fp_with_rc<0x5A, "vcvtpd2ps", VR512, VR256X, fround,
                                memopv8f64, f512mem, v8f32, v8f64,
                                SSEPackedSingle>, EVEX_V512, VEX_W, PD,
                                EVEX_CD8<64, CD8VF>;

defm VCVTPS2PDZ : avx512_vcvt_fp<0x5A, "vcvtps2pd", VR256X, VR512, fextend,
                                memopv4f64, f256mem, v8f64, v8f32,
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
                                memopv8i64, i512mem, v16f32, v16i32,
                                SSEPackedSingle>, EVEX_V512, PS,
                                EVEX_CD8<32, CD8VF>;

defm VCVTDQ2PDZ : avx512_vcvt_fp<0xE6, "vcvtdq2pd", VR256X, VR512, sint_to_fp,
                                memopv4i64, i256mem, v8f64, v8i32,
                                SSEPackedDouble>, EVEX_V512, XS,
                                EVEX_CD8<32, CD8VH>;

defm VCVTTPS2DQZ : avx512_vcvt_fp<0x5B, "vcvttps2dq", VR512, VR512, fp_to_sint,
                                 memopv16f32, f512mem, v16i32, v16f32,
                                 SSEPackedSingle>, EVEX_V512, XS,
                                 EVEX_CD8<32, CD8VF>;

defm VCVTTPD2DQZ : avx512_vcvt_fp<0xE6, "vcvttpd2dq", VR512, VR256X, fp_to_sint,
                                 memopv8f64, f512mem, v8i32, v8f64, 
                                 SSEPackedDouble>, EVEX_V512, PD, VEX_W,
                                 EVEX_CD8<64, CD8VF>;

defm VCVTTPS2UDQZ : avx512_vcvt_fp<0x78, "vcvttps2udq", VR512, VR512, fp_to_uint,
                                 memopv16f32, f512mem, v16i32, v16f32,
                                 SSEPackedSingle>, EVEX_V512, PS,
                                 EVEX_CD8<32, CD8VF>;

// cvttps2udq (src, 0, mask-all-ones, sae-current)
def : Pat<(v16i32 (int_x86_avx512_mask_cvttps2udq_512 (v16f32 VR512:$src),
                   (v16i32 immAllZerosV), (i16 -1), FROUND_CURRENT)),
          (VCVTTPS2UDQZrr VR512:$src)>;

defm VCVTTPD2UDQZ : avx512_vcvt_fp<0x78, "vcvttpd2udq", VR512, VR256X, fp_to_uint,
                                 memopv8f64, f512mem, v8i32, v8f64,
                                 SSEPackedDouble>, EVEX_V512, PS, VEX_W,
                                 EVEX_CD8<64, CD8VF>;
                                 
// cvttpd2udq (src, 0, mask-all-ones, sae-current)
def : Pat<(v8i32 (int_x86_avx512_mask_cvttpd2udq_512 (v8f64 VR512:$src),
                   (v8i32 immAllZerosV), (i8 -1), FROUND_CURRENT)),
          (VCVTTPD2UDQZrr VR512:$src)>;

defm VCVTUDQ2PDZ : avx512_vcvt_fp<0x7A, "vcvtudq2pd", VR256X, VR512, uint_to_fp,
                                 memopv4i64, f256mem, v8f64, v8i32,
                                 SSEPackedDouble>, EVEX_V512, XS,
                                 EVEX_CD8<32, CD8VH>;
                                 
defm VCVTUDQ2PSZ : avx512_vcvt_fp_with_rc<0x7A, "vcvtudq2ps", VR512, VR512, uint_to_fp,
                                 memopv16i32, f512mem, v16f32, v16i32,
                                 SSEPackedSingle>, EVEX_V512, XD,
                                 EVEX_CD8<32, CD8VF>;

def : Pat<(v8i32 (fp_to_uint (v8f32 VR256X:$src1))),
          (EXTRACT_SUBREG (v16i32 (VCVTTPS2UDQZrr 
           (v16f32 (SUBREG_TO_REG (i32 0), VR256X:$src1, sub_ymm)))), sub_ymm)>;
                                 

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
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [], d>, EVEX;
  def rrb : AVX512PI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src, AVX512RC:$rc),
              !strconcat(asm," \t{$rc, $src, $dst|$dst, $src, $rc}"),
              [], d>, EVEX, EVEX_B, EVEX_RC;
  let mayLoad = 1 in
  def rm : AVX512PI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src),
              !strconcat(asm," \t{$src, $dst|$dst, $src}"),
              [], d>, EVEX;
} // hasSideEffects = 0
}

defm VCVTPS2DQZ : avx512_vcvt_fp2int<0x5B, "vcvtps2dq", VR512, VR512,
                                 memopv16f32, f512mem, SSEPackedSingle>, PD,
                                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VCVTPD2DQZ : avx512_vcvt_fp2int<0xE6, "vcvtpd2dq", VR512, VR256X,
                                 memopv8f64, f512mem, SSEPackedDouble>, XD, VEX_W,
                                 EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat <(v16i32 (int_x86_avx512_mask_cvtps2dq_512 (v16f32 VR512:$src),
                    (v16i32 immAllZerosV), (i16 -1), imm:$rc)),
           (VCVTPS2DQZrrb VR512:$src, imm:$rc)>;

def : Pat <(v8i32 (int_x86_avx512_mask_cvtpd2dq_512 (v8f64 VR512:$src),
                    (v8i32 immAllZerosV), (i8 -1), imm:$rc)),
           (VCVTPD2DQZrrb VR512:$src, imm:$rc)>;

defm VCVTPS2UDQZ : avx512_vcvt_fp2int<0x79, "vcvtps2udq", VR512, VR512,
                                 memopv16f32, f512mem, SSEPackedSingle>,
                                 PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VCVTPD2UDQZ : avx512_vcvt_fp2int<0x79, "vcvtpd2udq", VR512, VR256X,
                                 memopv8f64, f512mem, SSEPackedDouble>, VEX_W,
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
               (ins srcRC:$src1, i32i8imm:$src2),
               "vcvtps2ph \t{$src2, $src1, $dst|$dst, $src1, $src2}",
               []>, EVEX;
  let hasSideEffects = 0, mayStore = 1 in
  def mr : AVX512AIi8<0x1D, MRMDestMem, (outs),
               (ins x86memop:$dst, srcRC:$src1, i32i8imm:$src2),
               "vcvtps2ph \t{$src2, $src1, $dst|$dst, $src1, $src2}", []>, EVEX;
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
               " \t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>, EVEX_4V;
  let mayLoad = 1 in {
  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, x86memop:$src2),
               !strconcat(OpcodeStr,
               " \t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>, EVEX_4V;
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
                         RegisterClass RC, X86MemOperand x86memop,
                         PatFrag mem_frag, ValueType OpVt> {
  def r : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                        !strconcat(OpcodeStr,
                                   " \t{$src, $dst|$dst, $src}"),
                        [(set RC:$dst, (OpVt (OpNode RC:$src)))]>,
                        EVEX;
  def m : AVX5128I<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                        !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                        [(set RC:$dst, (OpVt (OpNode (mem_frag addr:$src))))]>,
                        EVEX;
}
defm VRSQRT14PSZ : avx512_fp14_p<0x4E, "vrsqrt14ps", X86frsqrt, VR512, f512mem,
                        memopv16f32, v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VRSQRT14PDZ : avx512_fp14_p<0x4E, "vrsqrt14pd", X86frsqrt, VR512, f512mem,
                        memopv8f64, v8f64>, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
defm VRCP14PSZ : avx512_fp14_p<0x4C, "vrcp14ps", X86frcp, VR512, f512mem,
                        memopv16f32, v16f32>, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VRCP14PDZ : avx512_fp14_p<0x4C, "vrcp14pd", X86frcp, VR512, f512mem,
                        memopv8f64, v8f64>, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

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
multiclass avx512_fp28_s<bits<8> opc, string OpcodeStr, RegisterClass RC,
                            X86MemOperand x86memop> {
  let hasSideEffects = 0, Predicates = [HasERI] in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
               (ins RC:$src1, RC:$src2),
               !strconcat(OpcodeStr,
               " \t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>, EVEX_4V;
  def rrb : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
               (ins RC:$src1, RC:$src2),
               !strconcat(OpcodeStr,
               " \t{{sae}, $src2, $src1, $dst|$dst, $src1, $src2, {sae}}"),
               []>, EVEX_4V, EVEX_B;
  let mayLoad = 1 in {
  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
               (ins RC:$src1, x86memop:$src2),
               !strconcat(OpcodeStr,
               " \t{$src2, $src1, $dst|$dst, $src1, $src2}"), []>, EVEX_4V;
  }
}
}

defm VRCP28SS   : avx512_fp28_s<0xCB, "vrcp28ss", FR32X, f32mem>,
                  EVEX_CD8<32, CD8VT1>;
defm VRCP28SD   : avx512_fp28_s<0xCB, "vrcp28sd", FR64X, f64mem>,
                  VEX_W, EVEX_CD8<64, CD8VT1>;
defm VRSQRT28SS   : avx512_fp28_s<0xCD, "vrsqrt28ss", FR32X, f32mem>,
                  EVEX_CD8<32, CD8VT1>;
defm VRSQRT28SD   : avx512_fp28_s<0xCD, "vrsqrt28sd", FR64X, f64mem>,
                  VEX_W, EVEX_CD8<64, CD8VT1>;

def : Pat <(v4f32 (int_x86_avx512_rcp28_ss (v4f32 VR128X:$src1),
              (v4f32 VR128X:$src2), (bc_v4f32 (v4i32 immAllZerosV)), (i8 -1),
                   FROUND_NO_EXC)),
           (COPY_TO_REGCLASS (VRCP28SSrrb (COPY_TO_REGCLASS VR128X:$src1, FR32X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR32X)), VR128X)>;

def : Pat <(v2f64 (int_x86_avx512_rcp28_sd (v2f64 VR128X:$src1),
              (v2f64 VR128X:$src2), (bc_v2f64 (v4i32 immAllZerosV)), (i8 -1),
                   FROUND_NO_EXC)),
           (COPY_TO_REGCLASS (VRCP28SDrrb (COPY_TO_REGCLASS VR128X:$src1, FR64X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR64X)), VR128X)>;

def : Pat <(v4f32 (int_x86_avx512_rsqrt28_ss (v4f32 VR128X:$src1),
              (v4f32 VR128X:$src2), (bc_v4f32 (v4i32 immAllZerosV)), (i8 -1),
                   FROUND_NO_EXC)),
           (COPY_TO_REGCLASS (VRSQRT28SSrrb (COPY_TO_REGCLASS VR128X:$src1, FR32X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR32X)), VR128X)>;

def : Pat <(v2f64 (int_x86_avx512_rsqrt28_sd (v2f64 VR128X:$src1),
              (v2f64 VR128X:$src2), (bc_v2f64 (v4i32 immAllZerosV)), (i8 -1),
                   FROUND_NO_EXC)),
           (COPY_TO_REGCLASS (VRSQRT28SDrrb (COPY_TO_REGCLASS VR128X:$src1, FR64X),
                       (COPY_TO_REGCLASS VR128X:$src2, FR64X)), VR128X)>;

/// avx512_fp28_p rcp28ps, rcp28pd, rsqrt28ps, rsqrt28pd
multiclass avx512_fp28_p<bits<8> opc, string OpcodeStr,
                         RegisterClass RC, X86MemOperand x86memop> {
  let hasSideEffects = 0, Predicates = [HasERI] in {
  def r : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                        !strconcat(OpcodeStr,
                                   " \t{$src, $dst|$dst, $src}"),
                        []>, EVEX;
  def rb : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
                        !strconcat(OpcodeStr,
                                   " \t{{sae}, $src, $dst|$dst, $src, {sae}}"),
                        []>, EVEX, EVEX_B;
  def m : AVX5128I<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src),
                        !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
                        []>, EVEX;
  }
}
defm VRSQRT28PSZ : avx512_fp28_p<0xCC, "vrsqrt28ps", VR512, f512mem>,
                        EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VRSQRT28PDZ : avx512_fp28_p<0xCC, "vrsqrt28pd", VR512, f512mem>,
                        VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
defm VRCP28PSZ : avx512_fp28_p<0xCA, "vrcp28ps", VR512, f512mem>,
                        EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VRCP28PDZ : avx512_fp28_p<0xCA, "vrcp28pd", VR512, f512mem>,
                        VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat <(v16f32 (int_x86_avx512_rsqrt28_ps (v16f32 VR512:$src),
              (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1), FROUND_NO_EXC)),
           (VRSQRT28PSZrb VR512:$src)>;
def : Pat <(v8f64 (int_x86_avx512_rsqrt28_pd (v8f64 VR512:$src),
              (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1), FROUND_NO_EXC)),
           (VRSQRT28PDZrb VR512:$src)>;

def : Pat <(v16f32 (int_x86_avx512_rcp28_ps (v16f32 VR512:$src),
              (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1), FROUND_NO_EXC)),
           (VRCP28PSZrb VR512:$src)>;
def : Pat <(v8f64 (int_x86_avx512_rcp28_pd (v8f64 VR512:$src),
              (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1), FROUND_NO_EXC)),
           (VRCP28PDZrb VR512:$src)>;

multiclass avx512_sqrt_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              Intrinsic V16F32Int, Intrinsic V8F64Int,
                              OpndItins itins_s, OpndItins itins_d> {
  def PSZrr :AVX512PSI<opc, MRMSrcReg, (outs VR512:$dst), (ins VR512:$src),
             !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
             [(set VR512:$dst, (v16f32 (OpNode VR512:$src)))], itins_s.rr>,
             EVEX, EVEX_V512;

  let mayLoad = 1 in
  def PSZrm : AVX512PSI<opc, MRMSrcMem, (outs VR512:$dst), (ins f512mem:$src),
              !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
              [(set VR512:$dst, 
                (OpNode (v16f32 (bitconvert (memopv16f32 addr:$src)))))],
              itins_s.rm>, EVEX, EVEX_V512, EVEX_CD8<32, CD8VF>;

  def PDZrr : AVX512PDI<opc, MRMSrcReg, (outs VR512:$dst), (ins VR512:$src),
              !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
              [(set VR512:$dst, (v8f64 (OpNode VR512:$src)))], itins_d.rr>,
              EVEX, EVEX_V512;

  let mayLoad = 1 in
    def PDZrm : AVX512PDI<opc, MRMSrcMem, (outs VR512:$dst), (ins f512mem:$src),
                !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                [(set VR512:$dst, (OpNode
                  (v8f64 (bitconvert (memopv16f32 addr:$src)))))],
                itins_d.rm>, EVEX, EVEX_V512, EVEX_CD8<64, CD8VF>;

let isCodeGenOnly = 1 in {
  def PSZr_Int : AVX512PSI<opc, MRMSrcReg, (outs VR512:$dst), (ins VR512:$src),
                           !strconcat(OpcodeStr,
                                      "ps\t{$src, $dst|$dst, $src}"),
                           [(set VR512:$dst, (V16F32Int VR512:$src))]>, 
                           EVEX, EVEX_V512;
  def PSZm_Int : AVX512PSI<opc, MRMSrcMem, (outs VR512:$dst), (ins f512mem:$src),
                          !strconcat(OpcodeStr, "ps\t{$src, $dst|$dst, $src}"),
                          [(set VR512:$dst, 
                           (V16F32Int (memopv16f32 addr:$src)))]>, EVEX,
                          EVEX_V512, EVEX_CD8<32, CD8VF>;
  def PDZr_Int : AVX512PDI<opc, MRMSrcReg, (outs VR512:$dst), (ins VR512:$src),
                           !strconcat(OpcodeStr, "pd\t{$src, $dst|$dst, $src}"),
                           [(set VR512:$dst, (V8F64Int VR512:$src))]>, 
                           EVEX, EVEX_V512, VEX_W;
  def PDZm_Int : AVX512PDI<opc, MRMSrcMem, (outs VR512:$dst), (ins f512mem:$src),
                         !strconcat(OpcodeStr,
                         "pd\t{$src, $dst|$dst, $src}"),
                         [(set VR512:$dst, (V8F64Int (memopv8f64 addr:$src)))]>,
                         EVEX, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>; 
} // isCodeGenOnly = 1
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


defm VSQRT  : avx512_sqrt_scalar<0x51, "sqrt", 
                int_x86_avx512_sqrt_ss, int_x86_avx512_sqrt_sd, 
                SSE_SQRTSS, SSE_SQRTSD>,
              avx512_sqrt_packed<0x51, "vsqrt", fsqrt,
                int_x86_avx512_sqrt_ps_512, int_x86_avx512_sqrt_pd_512,
                SSE_SQRTPS, SSE_SQRTPD>;

let Predicates = [HasAVX512] in {
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


multiclass avx512_fp_unop_rm<bits<8> opcps, bits<8> opcpd, string OpcodeStr,
                            X86MemOperand x86memop, RegisterClass RC,
                            PatFrag mem_frag32, PatFrag mem_frag64,
                            Intrinsic V4F32Int, Intrinsic V2F64Int,
                            CD8VForm VForm> {
let ExeDomain = SSEPackedSingle in {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def PSr : AVX512AIi8<opcps, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst, (V4F32Int RC:$src1, imm:$src2))]>;

  // Vector intrinsic operation, mem
  def PSm : AVX512AIi8<opcps, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    "ps\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    [(set RC:$dst,
                          (V4F32Int (mem_frag32 addr:$src1),imm:$src2))]>,
                    EVEX_CD8<32, VForm>;
} // ExeDomain = SSEPackedSingle

let ExeDomain = SSEPackedDouble in {
  // Vector intrinsic operation, reg
  def PDr : AVX512AIi8<opcpd, MRMSrcReg,
                     (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                     !strconcat(OpcodeStr,
                     "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst, (V2F64Int RC:$src1, imm:$src2))]>;

  // Vector intrinsic operation, mem
  def PDm : AVX512AIi8<opcpd, MRMSrcMem,
                     (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                     !strconcat(OpcodeStr,
                     "pd\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                     [(set RC:$dst,
                          (V2F64Int (mem_frag64 addr:$src1),imm:$src2))]>,
                     EVEX_CD8<64, VForm>;
} // ExeDomain = SSEPackedDouble
}

multiclass avx512_fp_binop_rm<bits<8> opcss, bits<8> opcsd,
                            string OpcodeStr,
                            Intrinsic F32Int,
                            Intrinsic F64Int> {
let ExeDomain = GenericDomain in {
  // Operation, reg.
  let hasSideEffects = 0 in
  def SSr : AVX512AIi8<opcss, MRMSrcReg,
      (outs FR32X:$dst), (ins FR32X:$src1, FR32X:$src2, i32i8imm:$src3),
      !strconcat(OpcodeStr,
              "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
      []>;

  // Intrinsic operation, reg.
  let isCodeGenOnly = 1 in
  def SSr_Int : AVX512AIi8<opcss, MRMSrcReg,
        (outs VR128X:$dst), (ins VR128X:$src1, VR128X:$src2, i32i8imm:$src3),
        !strconcat(OpcodeStr,
                "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        [(set VR128X:$dst, (F32Int VR128X:$src1, VR128X:$src2, imm:$src3))]>;

  // Intrinsic operation, mem.
  def SSm : AVX512AIi8<opcss, MRMSrcMem, (outs VR128X:$dst),
                     (ins VR128X:$src1, ssmem:$src2, i32i8imm:$src3),
                     !strconcat(OpcodeStr,
                   "ss\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                     [(set VR128X:$dst, (F32Int VR128X:$src1, 
                                         sse_load_f32:$src2, imm:$src3))]>,
                     EVEX_CD8<32, CD8VT1>;

  // Operation, reg.
  let hasSideEffects = 0 in
  def SDr : AVX512AIi8<opcsd, MRMSrcReg,
        (outs FR64X:$dst), (ins FR64X:$src1, FR64X:$src2, i32i8imm:$src3),
        !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        []>, VEX_W;

  // Intrinsic operation, reg.
  let isCodeGenOnly = 1 in
  def SDr_Int : AVX512AIi8<opcsd, MRMSrcReg,
        (outs VR128X:$dst), (ins VR128X:$src1, VR128X:$src2, i32i8imm:$src3),
        !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        [(set VR128X:$dst, (F64Int VR128X:$src1, VR128X:$src2, imm:$src3))]>,
        VEX_W;

  // Intrinsic operation, mem.
  def SDm : AVX512AIi8<opcsd, MRMSrcMem,
        (outs VR128X:$dst), (ins VR128X:$src1, sdmem:$src2, i32i8imm:$src3),
        !strconcat(OpcodeStr,
                "sd\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
        [(set VR128X:$dst,
              (F64Int VR128X:$src1, sse_load_f64:$src2, imm:$src3))]>,
        VEX_W, EVEX_CD8<64, CD8VT1>;
} // ExeDomain = GenericDomain
}

multiclass avx512_rndscale<bits<8> opc, string OpcodeStr,
                            X86MemOperand x86memop, RegisterClass RC,
                            PatFrag mem_frag, Domain d> {
let ExeDomain = d in {
  // Intrinsic operation, reg.
  // Vector intrinsic operation, reg
  def r : AVX512AIi8<opc, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX;

  // Vector intrinsic operation, mem
  def m : AVX512AIi8<opc, MRMSrcMem,
                    (outs RC:$dst), (ins x86memop:$src1, i32i8imm:$src2),
                    !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX;
} // ExeDomain
}


defm VRNDSCALEPSZ : avx512_rndscale<0x08, "vrndscaleps", f512mem, VR512,
                                memopv16f32, SSEPackedSingle>, EVEX_V512,
                                EVEX_CD8<32, CD8VF>;

def : Pat<(v16f32 (int_x86_avx512_mask_rndscale_ps_512 (v16f32 VR512:$src1),
                   imm:$src2, (bc_v16f32 (v16i32 immAllZerosV)), (i16 -1),
                   FROUND_CURRENT)),
                   (VRNDSCALEPSZr VR512:$src1, imm:$src2)>;


defm VRNDSCALEPDZ : avx512_rndscale<0x09, "vrndscalepd", f512mem, VR512,
                                memopv8f64, SSEPackedDouble>, EVEX_V512,
                                VEX_W, EVEX_CD8<64, CD8VF>;

def : Pat<(v8f64 (int_x86_avx512_mask_rndscale_pd_512 (v8f64 VR512:$src1),
                  imm:$src2, (bc_v8f64 (v16i32 immAllZerosV)), (i8 -1),
                  FROUND_CURRENT)),
                   (VRNDSCALEPDZr VR512:$src1, imm:$src2)>;

multiclass avx512_rndscale_scalar<bits<8> opc, string OpcodeStr,
                     Operand x86memop, RegisterClass RC, Domain d> {
let ExeDomain = d in {
  def r : AVX512AIi8<opc, MRMSrcReg,
                    (outs RC:$dst), (ins RC:$src1, RC:$src2, i32i8imm:$src3),
                    !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX_4V;

  def m : AVX512AIi8<opc, MRMSrcMem,
                    (outs RC:$dst), (ins RC:$src1, x86memop:$src2,  i32i8imm:$src3),
                    !strconcat(OpcodeStr,
                    " \t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                    []>, EVEX_4V;
} // ExeDomain
}

defm VRNDSCALESS : avx512_rndscale_scalar<0x0A, "vrndscaless", ssmem, FR32X,
                                SSEPackedSingle>, EVEX_CD8<32, CD8VT1>;
                                
defm VRNDSCALESD : avx512_rndscale_scalar<0x0B, "vrndscalesd", sdmem, FR64X,
                                SSEPackedDouble>, EVEX_CD8<64, CD8VT1>;

def : Pat<(ffloor FR32X:$src),
          (VRNDSCALESSr (f32 (IMPLICIT_DEF)), FR32X:$src, (i32 0x1))>;
def : Pat<(f64 (ffloor FR64X:$src)),
          (VRNDSCALESDr (f64 (IMPLICIT_DEF)), FR64X:$src, (i32 0x1))>;
def : Pat<(f32 (fnearbyint FR32X:$src)),
          (VRNDSCALESSr (f32 (IMPLICIT_DEF)), FR32X:$src, (i32 0xC))>;
def : Pat<(f64 (fnearbyint FR64X:$src)),
          (VRNDSCALESDr (f64 (IMPLICIT_DEF)), FR64X:$src, (i32 0xC))>;
def : Pat<(f32 (fceil FR32X:$src)),
          (VRNDSCALESSr (f32 (IMPLICIT_DEF)), FR32X:$src, (i32 0x2))>;
def : Pat<(f64 (fceil FR64X:$src)),
          (VRNDSCALESDr (f64 (IMPLICIT_DEF)), FR64X:$src, (i32 0x2))>;
def : Pat<(f32 (frint FR32X:$src)),
          (VRNDSCALESSr (f32 (IMPLICIT_DEF)), FR32X:$src, (i32 0x4))>;
def : Pat<(f64 (frint FR64X:$src)),
          (VRNDSCALESDr (f64 (IMPLICIT_DEF)), FR64X:$src, (i32 0x4))>;
def : Pat<(f32 (ftrunc FR32X:$src)),
          (VRNDSCALESSr (f32 (IMPLICIT_DEF)), FR32X:$src, (i32 0x3))>;
def : Pat<(f64 (ftrunc FR64X:$src)),
          (VRNDSCALESDr (f64 (IMPLICIT_DEF)), FR64X:$src, (i32 0x3))>;

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

//-------------------------------------------------
// Integer truncate and extend operations
//-------------------------------------------------

multiclass avx512_trunc_sat<bits<8> opc, string OpcodeStr,
                          RegisterClass dstRC, RegisterClass srcRC,
                          RegisterClass KRC, X86MemOperand x86memop> {
  def rr : AVX512XS8I<opc, MRMDestReg, (outs dstRC:$dst),
               (ins srcRC:$src),
               !strconcat(OpcodeStr," \t{$src, $dst|$dst, $src}"),
               []>, EVEX;

  def krr : AVX512XS8I<opc, MRMDestReg, (outs dstRC:$dst),
               (ins KRC:$mask, srcRC:$src),
               !strconcat(OpcodeStr,
                 " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
               []>, EVEX, EVEX_KZ;

  def mr : AVX512XS8I<opc, MRMDestMem, (outs), (ins x86memop:$dst, srcRC:$src),
               !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
               []>, EVEX;
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
                  (VPMOVDBkrr VK16WM:$mask, VR512:$src)>;
def : Pat<(v16i16 (X86vtruncm VK16WM:$mask, (v16i32 VR512:$src))),
                  (VPMOVDWkrr VK16WM:$mask, VR512:$src)>;
def : Pat<(v8i16  (X86vtruncm VK8WM:$mask,  (v8i64 VR512:$src))),
                  (VPMOVQWkrr  VK8WM:$mask, VR512:$src)>;
def : Pat<(v8i32  (X86vtruncm VK8WM:$mask,  (v8i64 VR512:$src))),
                  (VPMOVQDkrr  VK8WM:$mask, VR512:$src)>;


multiclass avx512_extend<bits<8> opc, string OpcodeStr, RegisterClass DstRC,
                      RegisterClass SrcRC, SDNode OpNode, PatFrag mem_frag, 
                      X86MemOperand x86memop, ValueType OpVT, ValueType InVT> {

  def rr : AVX5128I<opc, MRMSrcReg, (outs DstRC:$dst),
              (ins SrcRC:$src),
              !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst, (OpVT (OpNode (InVT SrcRC:$src))))]>, EVEX;
  def rm : AVX5128I<opc, MRMSrcMem, (outs DstRC:$dst),
              (ins x86memop:$src),
              !strconcat(OpcodeStr," \t{$src, $dst|$dst, $src}"),
              [(set DstRC:$dst,
                (OpVT (OpNode (InVT (bitconvert (mem_frag addr:$src))))))]>,
              EVEX;
}

defm VPMOVZXBDZ: avx512_extend<0x31, "vpmovzxbd", VR512, VR128X, X86vzext, 
                             memopv2i64, i128mem, v16i32, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VQ>;
defm VPMOVZXBQZ: avx512_extend<0x32, "vpmovzxbq", VR512, VR128X, X86vzext, 
                             memopv2i64, i128mem, v8i64, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VO>;
defm VPMOVZXWDZ: avx512_extend<0x33, "vpmovzxwd", VR512, VR256X, X86vzext, 
                             memopv4i64, i256mem, v16i32, v16i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VH>;
defm VPMOVZXWQZ: avx512_extend<0x34, "vpmovzxwq", VR512, VR128X, X86vzext, 
                             memopv2i64, i128mem, v8i64, v8i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VQ>;
defm VPMOVZXDQZ: avx512_extend<0x35, "vpmovzxdq", VR512, VR256X, X86vzext, 
                             memopv4i64, i256mem, v8i64, v8i32>, EVEX_V512,
                             EVEX_CD8<32, CD8VH>;
                             
defm VPMOVSXBDZ: avx512_extend<0x21, "vpmovsxbd", VR512, VR128X, X86vsext, 
                             memopv2i64, i128mem, v16i32, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VQ>;
defm VPMOVSXBQZ: avx512_extend<0x22, "vpmovsxbq", VR512, VR128X, X86vsext, 
                             memopv2i64, i128mem, v8i64, v16i8>, EVEX_V512,
                             EVEX_CD8<8, CD8VO>;
defm VPMOVSXWDZ: avx512_extend<0x23, "vpmovsxwd", VR512, VR256X, X86vsext, 
                             memopv4i64, i256mem, v16i32, v16i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VH>;
defm VPMOVSXWQZ: avx512_extend<0x24, "vpmovsxwq", VR512, VR128X, X86vsext, 
                             memopv2i64, i128mem, v8i64, v8i16>, EVEX_V512,
                             EVEX_CD8<16, CD8VQ>;
defm VPMOVSXDQZ: avx512_extend<0x25, "vpmovsxdq", VR512, VR256X, X86vsext, 
                             memopv4i64, i256mem, v8i64, v8i32>, EVEX_V512,
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
            " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
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
            " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
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

//===----------------------------------------------------------------------===//
// VSHUFPS - VSHUFPD Operations

multiclass avx512_shufp<RegisterClass RC, X86MemOperand x86memop,
                      ValueType vt, string OpcodeStr, PatFrag mem_frag,
                      Domain d> {
  def rmi : AVX512PIi8<0xC6, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, x86memop:$src2, i8imm:$src3),
                   !strconcat(OpcodeStr,
                   " \t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                   [(set RC:$dst, (vt (X86Shufp RC:$src1, (mem_frag addr:$src2),
                                       (i8 imm:$src3))))], d, IIC_SSE_SHUFP>,
                   EVEX_4V, Sched<[WriteShuffleLd, ReadAfterLd]>;
  def rri : AVX512PIi8<0xC6, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, i8imm:$src3),
                   !strconcat(OpcodeStr,
                   " \t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                   [(set RC:$dst, (vt (X86Shufp RC:$src1, RC:$src2,
                                       (i8 imm:$src3))))], d, IIC_SSE_SHUFP>,
                   EVEX_4V, Sched<[WriteShuffle]>;
}

defm VSHUFPSZ  : avx512_shufp<VR512, f512mem, v16f32, "vshufps", memopv16f32,
                  SSEPackedSingle>, PS, EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VSHUFPDZ  : avx512_shufp<VR512, f512mem, v8f64, "vshufpd", memopv8f64,
                  SSEPackedDouble>, PD, VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat<(v16i32 (X86Shufp VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VSHUFPSZrri VR512:$src1, VR512:$src2, imm:$imm)>;
def : Pat<(v16i32 (X86Shufp VR512:$src1,
                    (memopv16i32 addr:$src2), (i8 imm:$imm))),
          (VSHUFPSZrmi VR512:$src1, addr:$src2, imm:$imm)>;

def : Pat<(v8i64 (X86Shufp VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VSHUFPDZrri VR512:$src1, VR512:$src2, imm:$imm)>;
def : Pat<(v8i64 (X86Shufp VR512:$src1,
                            (memopv8i64 addr:$src2), (i8 imm:$imm))),
          (VSHUFPDZrmi VR512:$src1, addr:$src2, imm:$imm)>;

multiclass avx512_alignr<string OpcodeStr, RegisterClass RC,
                       X86MemOperand x86memop> {
  def rri : AVX512AIi8<0x03, MRMSrcReg, (outs RC:$dst),
                     (ins RC:$src1, RC:$src2, i8imm:$src3),
                     !strconcat(OpcodeStr,
                     " \t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                     []>, EVEX_4V;
  let mayLoad = 1 in
  def rmi : AVX512AIi8<0x03, MRMSrcMem, (outs RC:$dst),
                     (ins RC:$src1, x86memop:$src2, i8imm:$src3),
                     !strconcat(OpcodeStr,
                     " \t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                     []>, EVEX_4V;
}
defm VALIGND : avx512_alignr<"valignd", VR512, i512mem>, 
                 EVEX_V512, EVEX_CD8<32, CD8VF>;
defm VALIGNQ : avx512_alignr<"valignq", VR512, i512mem>, 
                 VEX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;

def : Pat<(v16f32 (X86PAlignr VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VALIGNDrri VR512:$src2, VR512:$src1, imm:$imm)>;
def : Pat<(v8f64 (X86PAlignr VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VALIGNQrri VR512:$src2, VR512:$src1, imm:$imm)>;
def : Pat<(v16i32 (X86PAlignr VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VALIGNDrri VR512:$src2, VR512:$src1, imm:$imm)>;
def : Pat<(v8i64 (X86PAlignr VR512:$src1, VR512:$src2, (i8 imm:$imm))),
          (VALIGNQrri VR512:$src2, VR512:$src1, imm:$imm)>;

// Helper fragments to match sext vXi1 to vXiY.
def v16i1sextv16i32  : PatLeaf<(v16i32 (X86vsrai VR512:$src, (i8 31)))>;
def v8i1sextv8i64  : PatLeaf<(v8i64 (X86vsrai VR512:$src, (i8 63)))>;

multiclass avx512_vpabs<bits<8> opc, string OpcodeStr, ValueType OpVT,
                        RegisterClass KRC, RegisterClass RC,
                        X86MemOperand x86memop, X86MemOperand x86scalar_mop,
                        string BrdcstStr> {
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src),
            !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
            []>, EVEX;
  def rrk : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src),
             !strconcat(OpcodeStr, " \t{$src, $dst {${mask}}|$dst {${mask}}, $src}"),
             []>, EVEX, EVEX_K;
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs RC:$dst), (ins KRC:$mask, RC:$src),
              !strconcat(OpcodeStr,
                         " \t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
              []>, EVEX, EVEX_KZ;
  let mayLoad = 1 in {
    def rm : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
              (ins x86memop:$src),
              !strconcat(OpcodeStr, " \t{$src, $dst|$dst, $src}"),
              []>, EVEX;
    def rmk : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
               (ins KRC:$mask, x86memop:$src),
               !strconcat(OpcodeStr,
                          " \t{$src, $dst {${mask}}|$dst {${mask}}, $src}"),
               []>, EVEX, EVEX_K;
    def rmkz : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
                (ins KRC:$mask, x86memop:$src),
                !strconcat(OpcodeStr,
                           " \t{$src, $dst {${mask}} {z}|$dst {${mask}} {z}, $src}"),
                []>, EVEX, EVEX_KZ;
    def rmb : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
               (ins x86scalar_mop:$src),
               !strconcat(OpcodeStr, " \t{${src}", BrdcstStr,
                          ", $dst|$dst, ${src}", BrdcstStr, "}"),
               []>, EVEX, EVEX_B;
    def rmbk : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
                (ins KRC:$mask, x86scalar_mop:$src),
                !strconcat(OpcodeStr, " \t{${src}", BrdcstStr,
                           ", $dst {${mask}}|$dst {${mask}}, ${src}", BrdcstStr, "}"),
                []>, EVEX, EVEX_B, EVEX_K;
    def rmbkz : AVX5128I<opc, MRMSrcMem, (outs VR512:$dst),
                 (ins KRC:$mask, x86scalar_mop:$src),
                 !strconcat(OpcodeStr, " \t{${src}", BrdcstStr,
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
  def rr : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src),
       !strconcat(OpcodeStr, " \t{$src, ${dst} |${dst}, $src}"),
       []>, EVEX;
  def rm : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins x86memop:$src),
       !strconcat(OpcodeStr, " \t{$src, ${dst}|${dst}, $src}"),
       []>, EVEX;
  def rmb : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins x86scalar_mop:$src),
       !strconcat(OpcodeStr, " \t{${src}", BrdcstStr,
                  ", ${dst}|${dst}, ${src}", BrdcstStr, "}"),
       []>, EVEX, EVEX_B;
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
       (ins KRC:$mask, RC:$src),
       !strconcat(OpcodeStr,
                  " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
       []>, EVEX, EVEX_KZ;
  def rmkz : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, x86memop:$src),
       !strconcat(OpcodeStr,
                  " \t{$src, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src}"),
       []>, EVEX, EVEX_KZ;
  def rmbkz : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins KRC:$mask, x86scalar_mop:$src),
       !strconcat(OpcodeStr, " \t{${src}", BrdcstStr,
                  ", ${dst} {${mask}} {z}|${dst} {${mask}} {z}, ${src}",
                  BrdcstStr, "}"),
       []>, EVEX, EVEX_KZ, EVEX_B;
       
  let Constraints = "$src1 = $dst" in {
  def rrk : AVX5128I<opc, MRMSrcReg, (outs RC:$dst),
       (ins RC:$src1, KRC:$mask, RC:$src2),
       !strconcat(OpcodeStr,
                  " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
       []>, EVEX, EVEX_K;
  def rmk : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, KRC:$mask, x86memop:$src2),
       !strconcat(OpcodeStr,
                  " \t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
       []>, EVEX, EVEX_K;
  def rmbk : AVX5128I<opc, MRMSrcMem, (outs RC:$dst),
       (ins RC:$src1, KRC:$mask, x86scalar_mop:$src2),
       !strconcat(OpcodeStr, " \t{${src2}", BrdcstStr,
                  ", ${dst} {${mask}}|${dst} {${mask}}, ${src2}", BrdcstStr, "}"),
       []>, EVEX, EVEX_K, EVEX_B;
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
