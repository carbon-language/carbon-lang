; RUN: llc -march=mips -mattr=+msa < %s | FileCheck -check-prefix=MIPS32 %s

@v16i8 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
@v8i16 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
@v4i32 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>
@v2i64 = global <2 x i64> <i64 0, i64 0>
@i64 = global i64 0

define void @const_v16i8() nounwind {
  ; MIPS32: const_v16i8:

  store volatile <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8>*@v16i8
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8>*@v16i8
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 31>, <16 x i8>*@v16i8
  ; MIPS32: ld.b  [[R1:\$w[0-9]+]], %lo(

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6>, <16 x i8>*@v16i8
  ; MIPS32: ld.b  [[R1:\$w[0-9]+]], %lo(

  store volatile <16 x i8> <i8 1, i8 2, i8 1, i8 2, i8 1, i8 2, i8 1, i8 2, i8 1, i8 2, i8 1, i8 2, i8 1, i8 2, i8 1, i8 2>, <16 x i8>*@v16i8
  ; MIPS32: ldi.h [[R1:\$w[0-9]+]], 258

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4>, <16 x i8>*@v16i8
  ; MIPS32-DAG: lui [[R2:\$[0-9]+]], 258
  ; MIPS32-DAG: ori [[R2]], [[R2]], 772
  ; MIPS32-DAG: fill.w [[R1:\$w[0-9]+]], [[R2]]

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, <16 x i8>*@v16i8
  ; MIPS32: ld.b  [[R1:\$w[0-9]+]], %lo(

  ret void
  ; MIPS32: .size const_v16i8
}

define void @const_v8i16() nounwind {
  ; MIPS32: const_v8i16:

  store volatile <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <8 x i16>*@v8i16
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>*@v8i16
  ; MIPS32: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <8 x i16> <i16 1, i16 1, i16 1, i16 2, i16 1, i16 1, i16 1, i16 31>, <8 x i16>*@v8i16
  ; MIPS32: ld.h  [[R1:\$w[0-9]+]], %lo(

  store volatile <8 x i16> <i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028>, <8 x i16>*@v8i16
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 4

  store volatile <8 x i16> <i16 1, i16 2, i16 1, i16 2, i16 1, i16 2, i16 1, i16 2>, <8 x i16>*@v8i16
  ; MIPS32-DAG: lui [[R2:\$[0-9]+]], 1
  ; MIPS32-DAG: ori [[R2]], [[R2]], 2
  ; MIPS32-DAG: fill.w [[R1:\$w[0-9]+]], [[R2]]

  store volatile <8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 1, i16 2, i16 3, i16 4>, <8 x i16>*@v8i16
  ; MIPS32: ld.h  [[R1:\$w[0-9]+]], %lo(

  ret void
  ; MIPS32: .size const_v8i16
}

define void @const_v4i32() nounwind {
  ; MIPS32: const_v4i32:

  store volatile <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32>*@v4i32
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32>*@v4i32
  ; MIPS32: ldi.w [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 1, i32 1, i32 1, i32 31>, <4 x i32>*@v4i32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x i32> <i32 16843009, i32 16843009, i32 16843009, i32 16843009>, <4 x i32>*@v4i32
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 65537, i32 65537, i32 65537, i32 65537>, <4 x i32>*@v4i32
  ; MIPS32: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 1, i32 2, i32 1, i32 2>, <4 x i32>*@v4i32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x i32> <i32 3, i32 4, i32 5, i32 6>, <4 x i32>*@v4i32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  ret void
  ; MIPS32: .size const_v4i32
}

define void @const_v2i64() nounwind {
  ; MIPS32: const_v2i64:

  store volatile <2 x i64> <i64 0, i64 0>, <2 x i64>*@v2i64
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <2 x i64> <i64 72340172838076673, i64 72340172838076673>, <2 x i64>*@v2i64
  ; MIPS32: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 281479271743489, i64 281479271743489>, <2 x i64>*@v2i64
  ; MIPS32: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 4294967297, i64 4294967297>, <2 x i64>*@v2i64
  ; MIPS32: ldi.w [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 1, i64 1>, <2 x i64>*@v2i64
  ; MIPS32: ldi.d [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 1, i64 31>, <2 x i64>*@v2i64
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x i64> <i64 3, i64 4>, <2 x i64>*@v2i64
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  ret void
  ; MIPS32: .size const_v2i64
}
