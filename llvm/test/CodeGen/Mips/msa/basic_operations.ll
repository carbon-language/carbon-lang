; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32-AE -check-prefix=MIPS32-BE %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32-AE -check-prefix=MIPS32-LE %s

@v4i8 = global <4 x i8> <i8 0, i8 0, i8 0, i8 0>
@v16i8 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
@v8i16 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
@v4i32 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>
@v2i64 = global <2 x i64> <i64 0, i64 0>
@i64 = global i64 0

define void @const_v16i8() nounwind {
  ; MIPS32-AE: const_v16i8:

  store volatile <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8>*@v16i8
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8>*@v16i8
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 31>, <16 x i8>*@v16i8
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.b  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6>, <16 x i8>*@v16i8
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.b  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <16 x i8> <i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0>, <16 x i8>*@v16i8
  ; MIPS32-BE: ldi.h [[R1:\$w[0-9]+]], 256
  ; MIPS32-LE: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4>, <16 x i8>*@v16i8
  ; MIPS32-BE-DAG: lui [[R2:\$[0-9]+]], 258
  ; MIPS32-LE-DAG: lui [[R2:\$[0-9]+]], 1027
  ; MIPS32-BE-DAG: ori [[R2]], [[R2]], 772
  ; MIPS32-LE-DAG: ori [[R2]], [[R2]], 513
  ; MIPS32-AE-DAG: fill.w [[R1:\$w[0-9]+]], [[R2]]

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, <16 x i8>*@v16i8
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.b  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
  ; MIPS32-AE: .size const_v16i8
}

define void @const_v8i16() nounwind {
  ; MIPS32-AE: const_v8i16:

  store volatile <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <8 x i16>*@v8i16
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>*@v8i16
  ; MIPS32-AE: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <8 x i16> <i16 1, i16 1, i16 1, i16 2, i16 1, i16 1, i16 1, i16 31>, <8 x i16>*@v8i16
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.h  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <8 x i16> <i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028>, <8 x i16>*@v8i16
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 4

  store volatile <8 x i16> <i16 1, i16 2, i16 1, i16 2, i16 1, i16 2, i16 1, i16 2>, <8 x i16>*@v8i16
  ; MIPS32-BE-DAG: lui [[R2:\$[0-9]+]], 1
  ; MIPS32-LE-DAG: lui [[R2:\$[0-9]+]], 2
  ; MIPS32-BE-DAG: ori [[R2]], [[R2]], 2
  ; MIPS32-LE-DAG: ori [[R2]], [[R2]], 1
  ; MIPS32-AE-DAG: fill.w [[R1:\$w[0-9]+]], [[R2]]

  store volatile <8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 1, i16 2, i16 3, i16 4>, <8 x i16>*@v8i16
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.h  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
  ; MIPS32-AE: .size const_v8i16
}

define void @const_v4i32() nounwind {
  ; MIPS32-AE: const_v4i32:

  store volatile <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32>*@v4i32
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32>*@v4i32
  ; MIPS32-AE: ldi.w [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 1, i32 1, i32 1, i32 31>, <4 x i32>*@v4i32
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x i32> <i32 16843009, i32 16843009, i32 16843009, i32 16843009>, <4 x i32>*@v4i32
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 65537, i32 65537, i32 65537, i32 65537>, <4 x i32>*@v4i32
  ; MIPS32-AE: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 1, i32 2, i32 1, i32 2>, <4 x i32>*@v4i32
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x i32> <i32 3, i32 4, i32 5, i32 6>, <4 x i32>*@v4i32
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
  ; MIPS32-AE: .size const_v4i32
}

define void @const_v2i64() nounwind {
  ; MIPS32-AE: const_v2i64:

  store volatile <2 x i64> <i64 0, i64 0>, <2 x i64>*@v2i64
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <2 x i64> <i64 72340172838076673, i64 72340172838076673>, <2 x i64>*@v2i64
  ; MIPS32-AE: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 281479271743489, i64 281479271743489>, <2 x i64>*@v2i64
  ; MIPS32-AE: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 4294967297, i64 4294967297>, <2 x i64>*@v2i64
  ; MIPS32-AE: ldi.w [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 1, i64 1>, <2 x i64>*@v2i64
  ; MIPS32-AE: ldi.d [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 1, i64 31>, <2 x i64>*@v2i64
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x i64> <i64 3, i64 4>, <2 x i64>*@v2i64
  ; MIPS32-AE: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32-AE: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
  ; MIPS32-AE: .size const_v2i64
}

define void @nonconst_v16i8(i8 %a, i8 %b, i8 %c, i8 %d, i8 %e, i8 %f, i8 %g, i8 %h) nounwind {
  ; MIPS32-AE: nonconst_v16i8:

  %1 = insertelement <16 x i8> undef, i8 %a, i32 0
  %2 = insertelement <16 x i8> %1, i8 %b, i32 1
  %3 = insertelement <16 x i8> %2, i8 %c, i32 2
  %4 = insertelement <16 x i8> %3, i8 %d, i32 3
  %5 = insertelement <16 x i8> %4, i8 %e, i32 4
  %6 = insertelement <16 x i8> %5, i8 %f, i32 5
  %7 = insertelement <16 x i8> %6, i8 %g, i32 6
  %8 = insertelement <16 x i8> %7, i8 %h, i32 7
  %9 = insertelement <16 x i8> %8, i8 %h, i32 8
  %10 = insertelement <16 x i8> %9, i8 %h, i32 9
  %11 = insertelement <16 x i8> %10, i8 %h, i32 10
  %12 = insertelement <16 x i8> %11, i8 %h, i32 11
  %13 = insertelement <16 x i8> %12, i8 %h, i32 12
  %14 = insertelement <16 x i8> %13, i8 %h, i32 13
  %15 = insertelement <16 x i8> %14, i8 %h, i32 14
  %16 = insertelement <16 x i8> %15, i8 %h, i32 15
  ; MIPS32-AE-DAG: insert.b [[R1:\$w[0-9]+]][0], $4
  ; MIPS32-AE-DAG: insert.b [[R1]][1], $5
  ; MIPS32-AE-DAG: insert.b [[R1]][2], $6
  ; MIPS32-AE-DAG: insert.b [[R1]][3], $7
  ; MIPS32-BE-DAG: lbu [[R2:\$[0-9]+]], 19($sp)
  ; MIPS32-LE-DAG: lbu [[R2:\$[0-9]+]], 16($sp)
  ; MIPS32-AE-DAG: insert.b [[R1]][4], [[R2]]
  ; MIPS32-BE-DAG: lbu [[R3:\$[0-9]+]], 23($sp)
  ; MIPS32-LE-DAG: lbu [[R3:\$[0-9]+]], 20($sp)
  ; MIPS32-AE-DAG: insert.b [[R1]][5], [[R3]]
  ; MIPS32-BE-DAG: lbu [[R4:\$[0-9]+]], 27($sp)
  ; MIPS32-LE-DAG: lbu [[R4:\$[0-9]+]], 24($sp)
  ; MIPS32-AE-DAG: insert.b [[R1]][6], [[R4]]
  ; MIPS32-BE-DAG: lbu [[R5:\$[0-9]+]], 31($sp)
  ; MIPS32-LE-DAG: lbu [[R5:\$[0-9]+]], 28($sp)
  ; MIPS32-AE-DAG: insert.b [[R1]][7], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][8], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][9], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][10], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][11], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][12], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][13], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][14], [[R5]]
  ; MIPS32-AE-DAG: insert.b [[R1]][15], [[R5]]

  store volatile <16 x i8> %16, <16 x i8>*@v16i8

  ret void
  ; MIPS32-AE: .size nonconst_v16i8
}

define void @nonconst_v8i16(i16 %a, i16 %b, i16 %c, i16 %d, i16 %e, i16 %f, i16 %g, i16 %h) nounwind {
  ; MIPS32-AE: nonconst_v8i16:

  %1 = insertelement <8 x i16> undef, i16 %a, i32 0
  %2 = insertelement <8 x i16> %1, i16 %b, i32 1
  %3 = insertelement <8 x i16> %2, i16 %c, i32 2
  %4 = insertelement <8 x i16> %3, i16 %d, i32 3
  %5 = insertelement <8 x i16> %4, i16 %e, i32 4
  %6 = insertelement <8 x i16> %5, i16 %f, i32 5
  %7 = insertelement <8 x i16> %6, i16 %g, i32 6
  %8 = insertelement <8 x i16> %7, i16 %h, i32 7
  ; MIPS32-AE-DAG: insert.h [[R1:\$w[0-9]+]][0], $4
  ; MIPS32-AE-DAG: insert.h [[R1]][1], $5
  ; MIPS32-AE-DAG: insert.h [[R1]][2], $6
  ; MIPS32-AE-DAG: insert.h [[R1]][3], $7
  ; MIPS32-BE-DAG: lhu [[R2:\$[0-9]+]], 18($sp)
  ; MIPS32-LE-DAG: lhu [[R2:\$[0-9]+]], 16($sp)
  ; MIPS32-AE-DAG: insert.h [[R1]][4], [[R2]]
  ; MIPS32-BE-DAG: lhu [[R2:\$[0-9]+]], 22($sp)
  ; MIPS32-LE-DAG: lhu [[R2:\$[0-9]+]], 20($sp)
  ; MIPS32-AE-DAG: insert.h [[R1]][5], [[R2]]
  ; MIPS32-BE-DAG: lhu [[R2:\$[0-9]+]], 26($sp)
  ; MIPS32-LE-DAG: lhu [[R2:\$[0-9]+]], 24($sp)
  ; MIPS32-AE-DAG: insert.h [[R1]][6], [[R2]]
  ; MIPS32-BE-DAG: lhu [[R2:\$[0-9]+]], 30($sp)
  ; MIPS32-LE-DAG: lhu [[R2:\$[0-9]+]], 28($sp)
  ; MIPS32-AE-DAG: insert.h [[R1]][7], [[R2]]

  store volatile <8 x i16> %8, <8 x i16>*@v8i16

  ret void
  ; MIPS32-AE: .size nonconst_v8i16
}

define void @nonconst_v4i32(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
  ; MIPS32-AE: nonconst_v4i32:

  %1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %2 = insertelement <4 x i32> %1, i32 %b, i32 1
  %3 = insertelement <4 x i32> %2, i32 %c, i32 2
  %4 = insertelement <4 x i32> %3, i32 %d, i32 3
  ; MIPS32-AE: insert.w [[R1:\$w[0-9]+]][0], $4
  ; MIPS32-AE: insert.w [[R1]][1], $5
  ; MIPS32-AE: insert.w [[R1]][2], $6
  ; MIPS32-AE: insert.w [[R1]][3], $7

  store volatile <4 x i32> %4, <4 x i32>*@v4i32

  ret void
  ; MIPS32-AE: .size nonconst_v4i32
}

define void @nonconst_v2i64(i64 %a, i64 %b) nounwind {
  ; MIPS32-AE: nonconst_v2i64:

  %1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %2 = insertelement <2 x i64> %1, i64 %b, i32 1
  ; MIPS32-AE: insert.w [[R1:\$w[0-9]+]][0], $4
  ; MIPS32-AE: insert.w [[R1]][1], $5
  ; MIPS32-AE: insert.w [[R1]][2], $6
  ; MIPS32-AE: insert.w [[R1]][3], $7

  store volatile <2 x i64> %2, <2 x i64>*@v2i64

  ret void
  ; MIPS32-AE: .size nonconst_v2i64
}

define i32 @extract_sext_v16i8() nounwind {
  ; MIPS32-AE: extract_sext_v16i8:

  %1 = load <16 x i8>* @v16i8
  ; MIPS32-AE-DAG: ld.b [[R1:\$w[0-9]+]],

  %2 = add <16 x i8> %1, %1
  ; MIPS32-AE-DAG: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <16 x i8> %2, i32 1
  %4 = sext i8 %3 to i32
  ; MIPS32-AE-DAG: copy_s.b [[R3:\$[0-9]+]], [[R1]][1]
  ; MIPS32-AE-NOT: sll
  ; MIPS32-AE-NOT: sra

  ret i32 %4
  ; MIPS32-AE: .size extract_sext_v16i8
}

define i32 @extract_sext_v8i16() nounwind {
  ; MIPS32-AE: extract_sext_v8i16:

  %1 = load <8 x i16>* @v8i16
  ; MIPS32-AE-DAG: ld.h [[R1:\$w[0-9]+]],

  %2 = add <8 x i16> %1, %1
  ; MIPS32-AE-DAG: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <8 x i16> %2, i32 1
  %4 = sext i16 %3 to i32
  ; MIPS32-AE-DAG: copy_s.h [[R3:\$[0-9]+]], [[R1]][1]
  ; MIPS32-AE-NOT: sll
  ; MIPS32-AE-NOT: sra

  ret i32 %4
  ; MIPS32-AE: .size extract_sext_v8i16
}

define i32 @extract_sext_v4i32() nounwind {
  ; MIPS32-AE: extract_sext_v4i32:

  %1 = load <4 x i32>* @v4i32
  ; MIPS32-AE-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = add <4 x i32> %1, %1
  ; MIPS32-AE-DAG: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x i32> %2, i32 1
  ; MIPS32-AE-DAG: copy_s.w [[R3:\$[0-9]+]], [[R1]][1]

  ret i32 %3
  ; MIPS32-AE: .size extract_sext_v4i32
}

define i64 @extract_sext_v2i64() nounwind {
  ; MIPS32-AE: extract_sext_v2i64:

  %1 = load <2 x i64>* @v2i64
  ; MIPS32-AE-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = add <2 x i64> %1, %1
  ; MIPS32-AE-DAG: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <2 x i64> %2, i32 1
  ; MIPS32-AE-DAG: copy_s.w [[R3:\$[0-9]+]], [[R1]][2]
  ; MIPS32-AE-DAG: copy_s.w [[R4:\$[0-9]+]], [[R1]][3]
  ; MIPS32-AE-NOT: sll
  ; MIPS32-AE-NOT: sra

  ret i64 %3
  ; MIPS32-AE: .size extract_sext_v2i64
}

define i32 @extract_zext_v16i8() nounwind {
  ; MIPS32-AE: extract_zext_v16i8:

  %1 = load <16 x i8>* @v16i8
  ; MIPS32-AE-DAG: ld.b [[R1:\$w[0-9]+]],

  %2 = add <16 x i8> %1, %1
  ; MIPS32-AE-DAG: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <16 x i8> %2, i32 1
  %4 = zext i8 %3 to i32
  ; MIPS32-AE-DAG: copy_u.b [[R3:\$[0-9]+]], [[R1]][1]
  ; MIPS32-AE-NOT: andi

  ret i32 %4
  ; MIPS32-AE: .size extract_zext_v16i8
}

define i32 @extract_zext_v8i16() nounwind {
  ; MIPS32-AE: extract_zext_v8i16:

  %1 = load <8 x i16>* @v8i16
  ; MIPS32-AE-DAG: ld.h [[R1:\$w[0-9]+]],

  %2 = add <8 x i16> %1, %1
  ; MIPS32-AE-DAG: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <8 x i16> %2, i32 1
  %4 = zext i16 %3 to i32
  ; MIPS32-AE-DAG: copy_u.h [[R3:\$[0-9]+]], [[R1]][1]
  ; MIPS32-AE-NOT: andi

  ret i32 %4
  ; MIPS32-AE: .size extract_zext_v8i16
}

define i32 @extract_zext_v4i32() nounwind {
  ; MIPS32-AE: extract_zext_v4i32:

  %1 = load <4 x i32>* @v4i32
  ; MIPS32-AE-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = add <4 x i32> %1, %1
  ; MIPS32-AE-DAG: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x i32> %2, i32 1
  ; MIPS32-AE-DAG: copy_{{[su]}}.w [[R3:\$[0-9]+]], [[R1]][1]

  ret i32 %3
  ; MIPS32-AE: .size extract_zext_v4i32
}

define i64 @extract_zext_v2i64() nounwind {
  ; MIPS32-AE: extract_zext_v2i64:

  %1 = load <2 x i64>* @v2i64
  ; MIPS32-AE-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = add <2 x i64> %1, %1
  ; MIPS32-AE-DAG: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <2 x i64> %2, i32 1
  ; MIPS32-AE-DAG: copy_{{[su]}}.w [[R3:\$[0-9]+]], [[R1]][2]
  ; MIPS32-AE-DAG: copy_{{[su]}}.w [[R4:\$[0-9]+]], [[R1]][3]
  ; MIPS32-AE-NOT: andi

  ret i64 %3
  ; MIPS32-AE: .size extract_zext_v2i64
}

define void @insert_v16i8(i32 %a) nounwind {
  ; MIPS32-AE: insert_v16i8:

  %1 = load <16 x i8>* @v16i8
  ; MIPS32-AE-DAG: ld.b [[R1:\$w[0-9]+]],

  %a2 = trunc i32 %a to i8
  %a3 = sext i8 %a2 to i32
  %a4 = trunc i32 %a3 to i8
  ; MIPS32-AE-NOT: andi
  ; MIPS32-AE-NOT: sra

  %2 = insertelement <16 x i8> %1, i8 %a4, i32 1
  ; MIPS32-AE-DAG: insert.b [[R1]][1], $4

  store <16 x i8> %2, <16 x i8>* @v16i8
  ; MIPS32-AE-DAG: st.b [[R1]]

  ret void
  ; MIPS32-AE: .size insert_v16i8
}

define void @insert_v8i16(i32 %a) nounwind {
  ; MIPS32-AE: insert_v8i16:

  %1 = load <8 x i16>* @v8i16
  ; MIPS32-AE-DAG: ld.h [[R1:\$w[0-9]+]],

  %a2 = trunc i32 %a to i16
  %a3 = sext i16 %a2 to i32
  %a4 = trunc i32 %a3 to i16
  ; MIPS32-AE-NOT: andi
  ; MIPS32-AE-NOT: sra

  %2 = insertelement <8 x i16> %1, i16 %a4, i32 1
  ; MIPS32-AE-DAG: insert.h [[R1]][1], $4

  store <8 x i16> %2, <8 x i16>* @v8i16
  ; MIPS32-AE-DAG: st.h [[R1]]

  ret void
  ; MIPS32-AE: .size insert_v8i16
}

define void @insert_v4i32(i32 %a) nounwind {
  ; MIPS32-AE: insert_v4i32:

  %1 = load <4 x i32>* @v4i32
  ; MIPS32-AE-DAG: ld.w [[R1:\$w[0-9]+]],

  ; MIPS32-AE-NOT: andi
  ; MIPS32-AE-NOT: sra

  %2 = insertelement <4 x i32> %1, i32 %a, i32 1
  ; MIPS32-AE-DAG: insert.w [[R1]][1], $4

  store <4 x i32> %2, <4 x i32>* @v4i32
  ; MIPS32-AE-DAG: st.w [[R1]]

  ret void
  ; MIPS32-AE: .size insert_v4i32
}

define void @insert_v2i64(i64 %a) nounwind {
  ; MIPS32-AE: insert_v2i64:

  %1 = load <2 x i64>* @v2i64
  ; MIPS32-AE-DAG: ld.w [[R1:\$w[0-9]+]],

  ; MIPS32-AE-NOT: andi
  ; MIPS32-AE-NOT: sra

  %2 = insertelement <2 x i64> %1, i64 %a, i32 1
  ; MIPS32-AE-DAG: insert.w [[R1]][2], $4
  ; MIPS32-AE-DAG: insert.w [[R1]][3], $5

  store <2 x i64> %2, <2 x i64>* @v2i64
  ; MIPS32-AE-DAG: st.w [[R1]]

  ret void
  ; MIPS32-AE: .size insert_v2i64
}

define void @truncstore() nounwind {
  ; MIPS32-AE: truncstore:

  store volatile <4 x i8> <i8 -1, i8 -1, i8 -1, i8 -1>, <4 x i8>*@v4i8
  ; TODO: What code should be emitted?

  ret void
  ; MIPS32-AE: .size truncstore
}
