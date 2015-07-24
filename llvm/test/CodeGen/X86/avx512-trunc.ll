; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=SKX 
 
 attributes #0 = { nounwind }
 
; KNL-LABEL: trunc_16x32_to_16x8
; KNL: vpmovdb
; KNL: ret
define <16 x i8> @trunc_16x32_to_16x8(<16 x i32> %i) #0 {
  %x = trunc <16 x i32> %i to <16 x i8>
  ret <16 x i8> %x
}

; KNL-LABEL: trunc_8x64_to_8x16
; KNL: vpmovqw
; KNL: ret
define <8 x i16> @trunc_8x64_to_8x16(<8 x i64> %i) #0 {
  %x = trunc <8 x i64> %i to <8 x i16>
  ret <8 x i16> %x
}

; KNL-LABEL: trunc_v16i32_to_v16i16
; KNL: vpmovdw
; KNL: ret
define <16 x i16> @trunc_v16i32_to_v16i16(<16 x i32> %x) #0 {
  %1 = trunc <16 x i32> %x to <16 x i16>
  ret <16 x i16> %1
}

define <8 x i8> @trunc_qb_512(<8 x i64> %i) #0 {
; SKX-LABEL: trunc_qb_512:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqw %zmm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <8 x i64> %i to <8 x i8>
  ret <8 x i8> %x
}

define void @trunc_qb_512_mem(<8 x i64> %i, <8 x i8>* %res) #0 {
; SKX-LABEL: trunc_qb_512_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqb %zmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <8 x i64> %i to <8 x i8>
    store <8 x i8> %x, <8 x i8>* %res
    ret void
}

define <4 x i8> @trunc_qb_256(<4 x i64> %i) #0 {
; SKX-LABEL: trunc_qb_256:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %ymm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <4 x i64> %i to <4 x i8>
  ret <4 x i8> %x
}

define void @trunc_qb_256_mem(<4 x i64> %i, <4 x i8>* %res) #0 {
; SKX-LABEL: trunc_qb_256_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqb %ymm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <4 x i64> %i to <4 x i8>
    store <4 x i8> %x, <4 x i8>* %res
    ret void
}

define <2 x i8> @trunc_qb_128(<2 x i64> %i) #0 {
; SKX-LABEL: trunc_qb_128:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %x = trunc <2 x i64> %i to <2 x i8>
  ret <2 x i8> %x
}

define void @trunc_qb_128_mem(<2 x i64> %i, <2 x i8>* %res) #0 {
; SKX-LABEL: trunc_qb_128_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqb %xmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <2 x i64> %i to <2 x i8>
    store <2 x i8> %x, <2 x i8>* %res
    ret void
}

define <8 x i16> @trunc_qw_512(<8 x i64> %i) #0 {
; SKX-LABEL: trunc_qw_512:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqw %zmm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <8 x i64> %i to <8 x i16>
  ret <8 x i16> %x
}

define void @trunc_qw_512_mem(<8 x i64> %i, <8 x i16>* %res) #0 {
; SKX-LABEL: trunc_qw_512_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqw %zmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <8 x i64> %i to <8 x i16>
    store <8 x i16> %x, <8 x i16>* %res
    ret void
}

define <4 x i16> @trunc_qw_256(<4 x i64> %i) #0 {
; SKX-LABEL: trunc_qw_256:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %ymm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <4 x i64> %i to <4 x i16>
  ret <4 x i16> %x
}

define void @trunc_qw_256_mem(<4 x i64> %i, <4 x i16>* %res) #0 {
; SKX-LABEL: trunc_qw_256_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqw %ymm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <4 x i64> %i to <4 x i16>
    store <4 x i16> %x, <4 x i16>* %res
    ret void
}

define <2 x i16> @trunc_qw_128(<2 x i64> %i) #0 {
; SKX-LABEL: trunc_qw_128:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %x = trunc <2 x i64> %i to <2 x i16>
  ret <2 x i16> %x
}

define void @trunc_qw_128_mem(<2 x i64> %i, <2 x i16>* %res) #0 {
; SKX-LABEL: trunc_qw_128_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqw %xmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <2 x i64> %i to <2 x i16>
    store <2 x i16> %x, <2 x i16>* %res
    ret void
}

define <8 x i32> @trunc_qd_512(<8 x i64> %i) #0 {
; SKX-LABEL: trunc_qd_512:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %zmm0, %ymm0
; SKX-NEXT:    retq
  %x = trunc <8 x i64> %i to <8 x i32>
  ret <8 x i32> %x
}

define void @trunc_qd_512_mem(<8 x i64> %i, <8 x i32>* %res) #0 {
; SKX-LABEL: trunc_qd_512_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %zmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <8 x i64> %i to <8 x i32>
    store <8 x i32> %x, <8 x i32>* %res
    ret void
}

define <4 x i32> @trunc_qd_256(<4 x i64> %i) #0 {
; SKX-LABEL: trunc_qd_256:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %ymm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <4 x i64> %i to <4 x i32>
  ret <4 x i32> %x
}

define void @trunc_qd_256_mem(<4 x i64> %i, <4 x i32>* %res) #0 {
; SKX-LABEL: trunc_qd_256_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %ymm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <4 x i64> %i to <4 x i32>
    store <4 x i32> %x, <4 x i32>* %res
    ret void
}

define <2 x i32> @trunc_qd_128(<2 x i64> %i) #0 {
; SKX-LABEL: trunc_qd_128:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %x = trunc <2 x i64> %i to <2 x i32>
  ret <2 x i32> %x
}

define void @trunc_qd_128_mem(<2 x i64> %i, <2 x i32>* %res) #0 {
; SKX-LABEL: trunc_qd_128_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovqd %xmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <2 x i64> %i to <2 x i32>
    store <2 x i32> %x, <2 x i32>* %res
    ret void
}

define <16 x i8> @trunc_db_512(<16 x i32> %i) #0 {
; SKX-LABEL: trunc_db_512:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdb %zmm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <16 x i32> %i to <16 x i8>
  ret <16 x i8> %x
}

define void @trunc_db_512_mem(<16 x i32> %i, <16 x i8>* %res) #0 {
; SKX-LABEL: trunc_db_512_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdb %zmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <16 x i32> %i to <16 x i8>
    store <16 x i8> %x, <16 x i8>* %res
    ret void
}

define <8 x i8> @trunc_db_256(<8 x i32> %i) #0 {
; SKX-LABEL: trunc_db_256:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdw %ymm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <8 x i32> %i to <8 x i8>
  ret <8 x i8> %x
}

define void @trunc_db_256_mem(<8 x i32> %i, <8 x i8>* %res) #0 {
; SKX-LABEL: trunc_db_256_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdb %ymm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <8 x i32> %i to <8 x i8>
    store <8 x i8> %x, <8 x i8>* %res
    ret void
}

define <4 x i8> @trunc_db_128(<4 x i32> %i) #0 {
; SKX-LABEL: trunc_db_128:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %x = trunc <4 x i32> %i to <4 x i8>
  ret <4 x i8> %x
}

define void @trunc_db_128_mem(<4 x i32> %i, <4 x i8>* %res) #0 {
; SKX-LABEL: trunc_db_128_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdb %xmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <4 x i32> %i to <4 x i8>
    store <4 x i8> %x, <4 x i8>* %res
    ret void
}

define <16 x i16> @trunc_dw_512(<16 x i32> %i) #0 {
; SKX-LABEL: trunc_dw_512:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdw %zmm0, %ymm0
; SKX-NEXT:    retq
  %x = trunc <16 x i32> %i to <16 x i16>
  ret <16 x i16> %x
}

define void @trunc_dw_512_mem(<16 x i32> %i, <16 x i16>* %res) #0 {
; SKX-LABEL: trunc_dw_512_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdw %zmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <16 x i32> %i to <16 x i16>
    store <16 x i16> %x, <16 x i16>* %res
    ret void
}

define <8 x i16> @trunc_dw_256(<8 x i32> %i) #0 {
; SKX-LABEL: trunc_dw_256:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdw %ymm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <8 x i32> %i to <8 x i16>
  ret <8 x i16> %x
}

define void @trunc_dw_256_mem(<8 x i32> %i, <8 x i16>* %res) #0 {
; SKX-LABEL: trunc_dw_256_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdw %ymm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <8 x i32> %i to <8 x i16>
    store <8 x i16> %x, <8 x i16>* %res
    ret void
}

define <4 x i16> @trunc_dw_128(<4 x i32> %i) #0 {
; SKX-LABEL: trunc_dw_128:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %x = trunc <4 x i32> %i to <4 x i16>
  ret <4 x i16> %x
}

define void @trunc_dw_128_mem(<4 x i32> %i, <4 x i16>* %res) #0 {
; SKX-LABEL: trunc_dw_128_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovdw %xmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <4 x i32> %i to <4 x i16>
    store <4 x i16> %x, <4 x i16>* %res
    ret void
}

define <32 x i8> @trunc_wb_512(<32 x i16> %i) #0 {
; SKX-LABEL: trunc_wb_512:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovwb %zmm0, %ymm0
; SKX-NEXT:    retq
  %x = trunc <32 x i16> %i to <32 x i8>
  ret <32 x i8> %x
}

define void @trunc_wb_512_mem(<32 x i16> %i, <32 x i8>* %res) #0 {
; SKX-LABEL: trunc_wb_512_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovwb %zmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <32 x i16> %i to <32 x i8>
    store <32 x i8> %x, <32 x i8>* %res
    ret void
}

define <16 x i8> @trunc_wb_256(<16 x i16> %i) #0 {
; SKX-LABEL: trunc_wb_256:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovwb %ymm0, %xmm0
; SKX-NEXT:    retq
  %x = trunc <16 x i16> %i to <16 x i8>
  ret <16 x i8> %x
}

define void @trunc_wb_256_mem(<16 x i16> %i, <16 x i8>* %res) #0 {
; SKX-LABEL: trunc_wb_256_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovwb %ymm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <16 x i16> %i to <16 x i8>
    store <16 x i8> %x, <16 x i8>* %res
    ret void
}

define <8 x i8> @trunc_wb_128(<8 x i16> %i) #0 {
; SKX-LABEL: trunc_wb_128:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %x = trunc <8 x i16> %i to <8 x i8>
  ret <8 x i8> %x
}

define void @trunc_wb_128_mem(<8 x i16> %i, <8 x i8>* %res) #0 {
; SKX-LABEL: trunc_wb_128_mem:
; SKX:       ## BB#0:
; SKX-NEXT:    vpmovwb %xmm0, (%rdi)
; SKX-NEXT:    retq
    %x = trunc <8 x i16> %i to <8 x i8>
    store <8 x i8> %x, <8 x i8>* %res
    ret void
}
