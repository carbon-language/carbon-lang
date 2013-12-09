; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32-AE -check-prefix=MIPS32-BE %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32-AE -check-prefix=MIPS32-LE %s

define void @loadstore_v16i8_near() nounwind {
  ; MIPS32-AE: loadstore_v16i8_near:

  %1 = alloca <16 x i8>
  %2 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0($sp)
  store volatile <16 x i8> %2, <16 x i8>* %1
  ; MIPS32-AE: st.b [[R1]], 0($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_near
}

define void @loadstore_v16i8_just_under_simm10() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_under_simm10:

  %1 = alloca <16 x i8>
  %2 = alloca [496 x i8] ; Push the frame right up to 512 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 496($sp)
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: st.b [[R1]], 496($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_under_simm10
}

define void @loadstore_v16i8_just_over_simm10() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_over_simm10:

  %1 = alloca <16 x i8>
  %2 = alloca [497 x i8] ; Push the frame just over 512 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 512
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 512
  ; MIPS32-AE: st.b [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_over_simm10
}

define void @loadstore_v16i8_just_under_simm16() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_under_simm16:

  %1 = alloca <16 x i8>
  %2 = alloca [32752 x i8] ; Push the frame right up to 32768 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.b [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_under_simm16
}

define void @loadstore_v16i8_just_over_simm16() nounwind {
  ; MIPS32-AE: loadstore_v16i8_just_over_simm16:

  %1 = alloca <16 x i8>
  %2 = alloca [32753 x i8] ; Push the frame just over 32768 bytes

  %3 = load volatile <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.b [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v16i8_just_over_simm16
}

define void @loadstore_v8i16_near() nounwind {
  ; MIPS32-AE: loadstore_v8i16_near:

  %1 = alloca <8 x i16>
  %2 = load volatile <8 x i16>* %1
  ; MIPS32-AE: ld.h [[R1:\$w[0-9]+]], 0($sp)
  store volatile <8 x i16> %2, <8 x i16>* %1
  ; MIPS32-AE: st.h [[R1]], 0($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v8i16_near
}

define void @loadstore_v8i16_just_under_simm10() nounwind {
  ; MIPS32-AE: loadstore_v8i16_just_under_simm10:

  %1 = alloca <8 x i16>
  %2 = alloca [1008 x i8] ; Push the frame right up to 1024 bytes

  %3 = load volatile <8 x i16>* %1
  ; MIPS32-AE: ld.h [[R1:\$w[0-9]+]], 1008($sp)
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; MIPS32-AE: st.h [[R1]], 1008($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v8i16_just_under_simm10
}

define void @loadstore_v8i16_just_over_simm10() nounwind {
  ; MIPS32-AE: loadstore_v8i16_just_over_simm10:

  %1 = alloca <8 x i16>
  %2 = alloca [1009 x i8] ; Push the frame just over 1024 bytes

  %3 = load volatile <8 x i16>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 1024
  ; MIPS32-AE: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 1024
  ; MIPS32-AE: st.h [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v8i16_just_over_simm10
}

define void @loadstore_v8i16_just_under_simm16() nounwind {
  ; MIPS32-AE: loadstore_v8i16_just_under_simm16:

  %1 = alloca <8 x i16>
  %2 = alloca [32752 x i8] ; Push the frame right up to 32768 bytes

  %3 = load volatile <8 x i16>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.h [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v8i16_just_under_simm16
}

define void @loadstore_v8i16_just_over_simm16() nounwind {
  ; MIPS32-AE: loadstore_v8i16_just_over_simm16:

  %1 = alloca <8 x i16>
  %2 = alloca [32753 x i8] ; Push the frame just over 32768 bytes

  %3 = load volatile <8 x i16>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.h [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v8i16_just_over_simm16
}

define void @loadstore_v4i32_near() nounwind {
  ; MIPS32-AE: loadstore_v4i32_near:

  %1 = alloca <4 x i32>
  %2 = load volatile <4 x i32>* %1
  ; MIPS32-AE: ld.w [[R1:\$w[0-9]+]], 0($sp)
  store volatile <4 x i32> %2, <4 x i32>* %1
  ; MIPS32-AE: st.w [[R1]], 0($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v4i32_near
}

define void @loadstore_v4i32_just_under_simm10() nounwind {
  ; MIPS32-AE: loadstore_v4i32_just_under_simm10:

  %1 = alloca <4 x i32>
  %2 = alloca [2032 x i8] ; Push the frame right up to 2048 bytes

  %3 = load volatile <4 x i32>* %1
  ; MIPS32-AE: ld.w [[R1:\$w[0-9]+]], 2032($sp)
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; MIPS32-AE: st.w [[R1]], 2032($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v4i32_just_under_simm10
}

define void @loadstore_v4i32_just_over_simm10() nounwind {
  ; MIPS32-AE: loadstore_v4i32_just_over_simm10:

  %1 = alloca <4 x i32>
  %2 = alloca [2033 x i8] ; Push the frame just over 2048 bytes

  %3 = load volatile <4 x i32>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 2048
  ; MIPS32-AE: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 2048
  ; MIPS32-AE: st.w [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v4i32_just_over_simm10
}

define void @loadstore_v4i32_just_under_simm16() nounwind {
  ; MIPS32-AE: loadstore_v4i32_just_under_simm16:

  %1 = alloca <4 x i32>
  %2 = alloca [32752 x i8] ; Push the frame right up to 32768 bytes

  %3 = load volatile <4 x i32>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.w [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v4i32_just_under_simm16
}

define void @loadstore_v4i32_just_over_simm16() nounwind {
  ; MIPS32-AE: loadstore_v4i32_just_over_simm16:

  %1 = alloca <4 x i32>
  %2 = alloca [32753 x i8] ; Push the frame just over 32768 bytes

  %3 = load volatile <4 x i32>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.w [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v4i32_just_over_simm16
}

define void @loadstore_v2i64_near() nounwind {
  ; MIPS32-AE: loadstore_v2i64_near:

  %1 = alloca <2 x i64>
  %2 = load volatile <2 x i64>* %1
  ; MIPS32-AE: ld.d [[R1:\$w[0-9]+]], 0($sp)
  store volatile <2 x i64> %2, <2 x i64>* %1
  ; MIPS32-AE: st.d [[R1]], 0($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v2i64_near
}

define void @loadstore_v2i64_just_under_simm10() nounwind {
  ; MIPS32-AE: loadstore_v2i64_just_under_simm10:

  %1 = alloca <2 x i64>
  %2 = alloca [4080 x i8] ; Push the frame right up to 4096 bytes

  %3 = load volatile <2 x i64>* %1
  ; MIPS32-AE: ld.d [[R1:\$w[0-9]+]], 4080($sp)
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; MIPS32-AE: st.d [[R1]], 4080($sp)

  ret void
  ; MIPS32-AE: .size loadstore_v2i64_just_under_simm10
}

define void @loadstore_v2i64_just_over_simm10() nounwind {
  ; MIPS32-AE: loadstore_v2i64_just_over_simm10:

  %1 = alloca <2 x i64>
  %2 = alloca [4081 x i8] ; Push the frame just over 4096 bytes

  %3 = load volatile <2 x i64>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 4096
  ; MIPS32-AE: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; MIPS32-AE: addiu [[BASE:\$[0-9]+]], $sp, 4096
  ; MIPS32-AE: st.d [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v2i64_just_over_simm10
}

define void @loadstore_v2i64_just_under_simm16() nounwind {
  ; MIPS32-AE: loadstore_v2i64_just_under_simm16:

  %1 = alloca <2 x i64>
  %2 = alloca [32752 x i8] ; Push the frame right up to 32768 bytes

  %3 = load volatile <2 x i64>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.d [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v2i64_just_under_simm16
}

define void @loadstore_v2i64_just_over_simm16() nounwind {
  ; MIPS32-AE: loadstore_v2i64_just_over_simm16:

  %1 = alloca <2 x i64>
  %2 = alloca [32753 x i8] ; Push the frame just over 32768 bytes

  %3 = load volatile <2 x i64>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; MIPS32-AE: ori [[R2:\$[0-9]+]], $zero, 32768
  ; MIPS32-AE: addu [[BASE:\$[0-9]+]], $sp, [[R2]]
  ; MIPS32-AE: st.d [[R1]], 0([[BASE]])

  ret void
  ; MIPS32-AE: .size loadstore_v2i64_just_over_simm16
}
