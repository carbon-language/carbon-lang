; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r5 < %s | FileCheck %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r5 < %s | FileCheck %s

define void @loadstore_v16i8_near() nounwind {
  ; CHECK: loadstore_v16i8_near:

  %1 = alloca <16 x i8>
  %2 = load volatile <16 x i8>, <16 x i8>* %1
  ; CHECK: ld.b [[R1:\$w[0-9]+]], 0($sp)
  store volatile <16 x i8> %2, <16 x i8>* %1
  ; CHECK: st.b [[R1]], 0($sp)

  ret void
  ; CHECK: .size loadstore_v16i8_near
}

define void @loadstore_v16i8_just_under_simm10() nounwind {
  ; CHECK: loadstore_v16i8_just_under_simm10:

  %1 = alloca <16 x i8>
  %2 = alloca [492 x i8] ; Push the frame--acounting for the emergency spill
                         ; slot--right up to 512 bytes

  %3 = load volatile <16 x i8>, <16 x i8>* %1
  ; CHECK: ld.b [[R1:\$w[0-9]+]], 496($sp)
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; CHECK: st.b [[R1]], 496($sp)

  ret void
  ; CHECK: .size loadstore_v16i8_just_under_simm10
}

define void @loadstore_v16i8_just_over_simm10() nounwind {
  ; CHECK: loadstore_v16i8_just_over_simm10:

  %1 = alloca <16 x i8>
  %2 = alloca [497 x i8] ; Push the frame--acounting for the emergency spill
                         ; slot--right up to 512 bytes

  %3 = load volatile <16 x i8>, <16 x i8>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 512
  ; CHECK: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 512
  ; CHECK: st.b [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v16i8_just_over_simm10
}

define void @loadstore_v16i8_just_under_simm16() nounwind {
  ; CHECK: loadstore_v16i8_just_under_simm16:

  %1 = alloca <16 x i8>
  %2 = alloca [32752 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--right up to 32768 bytes

  %3 = load volatile <16 x i8>, <16 x i8>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.b [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v16i8_just_under_simm16
}

define void @loadstore_v16i8_just_over_simm16() nounwind {
  ; CHECK: loadstore_v16i8_just_over_simm16:

  %1 = alloca <16 x i8>
  %2 = alloca [32753 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--just over 32768 bytes

  %3 = load volatile <16 x i8>, <16 x i8>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.b [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <16 x i8> %3, <16 x i8>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.b [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v16i8_just_over_simm16
}

define void @loadstore_v8i16_near() nounwind {
  ; CHECK: loadstore_v8i16_near:

  %1 = alloca <8 x i16>
  %2 = load volatile <8 x i16>, <8 x i16>* %1
  ; CHECK: ld.h [[R1:\$w[0-9]+]], 0($sp)
  store volatile <8 x i16> %2, <8 x i16>* %1
  ; CHECK: st.h [[R1]], 0($sp)

  ret void
  ; CHECK: .size loadstore_v8i16_near
}

define void @loadstore_v8i16_unaligned() nounwind {
  ; CHECK: loadstore_v8i16_unaligned:

  %1 = alloca [2 x <8 x i16>]
  %2 = bitcast [2 x <8 x i16>]* %1 to i8*
  %3 = getelementptr i8, i8* %2, i32 1
  %4 = bitcast i8* %3 to [2 x <8 x i16>]*
  %5 = getelementptr [2 x <8 x i16>], [2 x <8 x i16>]* %4, i32 0, i32 0

  %6 = load volatile <8 x i16>, <8 x i16>* %5
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1
  ; CHECK: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %6, <8 x i16>* %5
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1
  ; CHECK: st.h [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v8i16_unaligned
}

define void @loadstore_v8i16_just_under_simm10() nounwind {
  ; CHECK: loadstore_v8i16_just_under_simm10:

  %1 = alloca <8 x i16>
  %2 = alloca [1004 x i8] ; Push the frame--acounting for the emergency spill
                          ; slot--right up to 1024 bytes

  %3 = load volatile <8 x i16>, <8 x i16>* %1
  ; CHECK: ld.h [[R1:\$w[0-9]+]], 1008($sp)
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; CHECK: st.h [[R1]], 1008($sp)

  ret void
  ; CHECK: .size loadstore_v8i16_just_under_simm10
}

define void @loadstore_v8i16_just_over_simm10() nounwind {
  ; CHECK: loadstore_v8i16_just_over_simm10:

  %1 = alloca <8 x i16>
  %2 = alloca [1009 x i8] ; Push the frame--acounting for the emergency spill
                          ; slot--just over 1024 bytes

  %3 = load volatile <8 x i16>, <8 x i16>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1024
  ; CHECK: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1024
  ; CHECK: st.h [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v8i16_just_over_simm10
}

define void @loadstore_v8i16_just_under_simm16() nounwind {
  ; CHECK: loadstore_v8i16_just_under_simm16:

  %1 = alloca <8 x i16>
  %2 = alloca [32752 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--right up to 32768 bytes

  %3 = load volatile <8 x i16>, <8 x i16>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.h [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v8i16_just_under_simm16
}

define void @loadstore_v8i16_just_over_simm16() nounwind {
  ; CHECK: loadstore_v8i16_just_over_simm16:

  %1 = alloca <8 x i16>
  %2 = alloca [32753 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--just over 32768 bytes

  %3 = load volatile <8 x i16>, <8 x i16>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.h [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <8 x i16> %3, <8 x i16>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.h [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v8i16_just_over_simm16
}

define void @loadstore_v4i32_near() nounwind {
  ; CHECK: loadstore_v4i32_near:

  %1 = alloca <4 x i32>
  %2 = load volatile <4 x i32>, <4 x i32>* %1
  ; CHECK: ld.w [[R1:\$w[0-9]+]], 0($sp)
  store volatile <4 x i32> %2, <4 x i32>* %1
  ; CHECK: st.w [[R1]], 0($sp)

  ret void
  ; CHECK: .size loadstore_v4i32_near
}

define void @loadstore_v4i32_unaligned() nounwind {
  ; CHECK: loadstore_v4i32_unaligned:

  %1 = alloca [2 x <4 x i32>]
  %2 = bitcast [2 x <4 x i32>]* %1 to i8*
  %3 = getelementptr i8, i8* %2, i32 1
  %4 = bitcast i8* %3 to [2 x <4 x i32>]*
  %5 = getelementptr [2 x <4 x i32>], [2 x <4 x i32>]* %4, i32 0, i32 0

  %6 = load volatile <4 x i32>, <4 x i32>* %5
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1
  ; CHECK: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %6, <4 x i32>* %5
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1
  ; CHECK: st.w [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v4i32_unaligned
}

define void @loadstore_v4i32_just_under_simm10() nounwind {
  ; CHECK: loadstore_v4i32_just_under_simm10:

  %1 = alloca <4 x i32>
  %2 = alloca [2028 x i8] ; Push the frame--acounting for the emergency spill
                          ; slot--right up to 2048 bytes

  %3 = load volatile <4 x i32>, <4 x i32>* %1
  ; CHECK: ld.w [[R1:\$w[0-9]+]], 2032($sp)
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; CHECK: st.w [[R1]], 2032($sp)

  ret void
  ; CHECK: .size loadstore_v4i32_just_under_simm10
}

define void @loadstore_v4i32_just_over_simm10() nounwind {
  ; CHECK: loadstore_v4i32_just_over_simm10:

  %1 = alloca <4 x i32>
  %2 = alloca [2033 x i8] ; Push the frame--acounting for the emergency spill
                          ; slot--just over 2048 bytes

  %3 = load volatile <4 x i32>, <4 x i32>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 2048
  ; CHECK: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 2048
  ; CHECK: st.w [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v4i32_just_over_simm10
}

define void @loadstore_v4i32_just_under_simm16() nounwind {
  ; CHECK: loadstore_v4i32_just_under_simm16:

  %1 = alloca <4 x i32>
  %2 = alloca [32752 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot-- right up to 32768 bytes

  %3 = load volatile <4 x i32>, <4 x i32>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.w [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v4i32_just_under_simm16
}

define void @loadstore_v4i32_just_over_simm16() nounwind {
  ; CHECK: loadstore_v4i32_just_over_simm16:

  %1 = alloca <4 x i32>
  %2 = alloca [32753 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--just over 32768 bytes

  %3 = load volatile <4 x i32>, <4 x i32>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.w [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <4 x i32> %3, <4 x i32>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.w [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v4i32_just_over_simm16
}

define void @loadstore_v2i64_near() nounwind {
  ; CHECK: loadstore_v2i64_near:

  %1 = alloca <2 x i64>
  %2 = load volatile <2 x i64>, <2 x i64>* %1
  ; CHECK: ld.d [[R1:\$w[0-9]+]], 0($sp)
  store volatile <2 x i64> %2, <2 x i64>* %1
  ; CHECK: st.d [[R1]], 0($sp)

  ret void
  ; CHECK: .size loadstore_v2i64_near
}

define void @loadstore_v2i64_unaligned() nounwind {
  ; CHECK: loadstore_v2i64_unaligned:

  %1 = alloca [2 x <2 x i64>]
  %2 = bitcast [2 x <2 x i64>]* %1 to i8*
  %3 = getelementptr i8, i8* %2, i32 1
  %4 = bitcast i8* %3 to [2 x <2 x i64>]*
  %5 = getelementptr [2 x <2 x i64>], [2 x <2 x i64>]* %4, i32 0, i32 0

  %6 = load volatile <2 x i64>, <2 x i64>* %5
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1
  ; CHECK: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %6, <2 x i64>* %5
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 1
  ; CHECK: st.d [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v2i64_unaligned
}

define void @loadstore_v2i64_just_under_simm10() nounwind {
  ; CHECK: loadstore_v2i64_just_under_simm10:

  %1 = alloca <2 x i64>
  %2 = alloca [4076 x i8] ; Push the frame--acounting for the emergency spill
                          ; slot--right up to 4096 bytes
  %3 = load volatile <2 x i64>, <2 x i64>* %1
  ; CHECK: ld.d [[R1:\$w[0-9]+]], 4080($sp)
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; CHECK: st.d [[R1]], 4080($sp)

  ret void
  ; CHECK: .size loadstore_v2i64_just_under_simm10
}

define void @loadstore_v2i64_just_over_simm10() nounwind {
  ; CHECK: loadstore_v2i64_just_over_simm10:

  %1 = alloca <2 x i64>
  %2 = alloca [4081 x i8] ; Push the frame--acounting for the emergency spill
                          ; slot--just over 4096 bytes

  %3 = load volatile <2 x i64>, <2 x i64>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 4096
  ; CHECK: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; CHECK: addiu [[BASE:\$([0-9]+|gp)]], $sp, 4096
  ; CHECK: st.d [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v2i64_just_over_simm10
}

define void @loadstore_v2i64_just_under_simm16() nounwind {
  ; CHECK: loadstore_v2i64_just_under_simm16:

  %1 = alloca <2 x i64>
  %2 = alloca [32752 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--right up to 32768 bytes

  %3 = load volatile <2 x i64>, <2 x i64>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.d [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v2i64_just_under_simm16
}

define void @loadstore_v2i64_just_over_simm16() nounwind {
  ; CHECK: loadstore_v2i64_just_over_simm16:

  %1 = alloca <2 x i64>
  %2 = alloca [32753 x i8] ; Push the frame--acounting for the emergency spill
                           ; slot--just over 32768 bytes

  %3 = load volatile <2 x i64>, <2 x i64>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: ld.d [[R1:\$w[0-9]+]], 0([[BASE]])
  store volatile <2 x i64> %3, <2 x i64>* %1
  ; CHECK: ori [[R2:\$([0-9]+|gp)]], $zero, 32768
  ; CHECK: addu [[BASE:\$([0-9]+|gp)]], $sp, [[R2]]
  ; CHECK: st.d [[R1]], 0([[BASE]])

  ret void
  ; CHECK: .size loadstore_v2i64_just_over_simm16
}
