; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32 %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32 %s

@v4f32 = global <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>
@v2f64 = global <2 x double> <double 0.0, double 0.0>
@f32 = global float 0.0
@f64 = global double 0.0

define void @const_v4f32() nounwind {
  ; MIPS32: const_v4f32:

  store volatile <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>, <4 x float>*@v4f32
  ; MIPS32: ldi.b  [[R1:\$w[0-9]+]], 0

  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, <4 x float>*@v4f32
  ; MIPS32: lui     [[R1:\$[0-9]+]], 16256
  ; MIPS32: fill.w  [[R2:\$w[0-9]+]], [[R1]]

  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 31.0>, <4 x float>*@v4f32
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x float> <float 65537.0, float 65537.0, float 65537.0, float 65537.0>, <4 x float>*@v4f32
  ; MIPS32: lui     [[R1:\$[0-9]+]], 18304
  ; MIPS32: ori     [[R2:\$[0-9]+]], [[R1]], 128
  ; MIPS32: fill.w  [[R3:\$w[0-9]+]], [[R2]]

  store volatile <4 x float> <float 1.0, float 2.0, float 1.0, float 2.0>, <4 x float>*@v4f32
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x float> <float 3.0, float 4.0, float 5.0, float 6.0>, <4 x float>*@v4f32
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
  ; MIPS32: .size const_v4f32
}

define void @const_v2f64() nounwind {
  ; MIPS32: const_v2f64:

  store volatile <2 x double> <double 0.0, double 0.0>, <2 x double>*@v2f64
  ; MIPS32: ldi.b  [[R1:\$w[0-9]+]], 0

  store volatile <2 x double> <double 72340172838076673.0, double 72340172838076673.0>, <2 x double>*@v2f64
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 281479271743489.0, double 281479271743489.0>, <2 x double>*@v2f64
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 4294967297.0, double 4294967297.0>, <2 x double>*@v2f64
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 1.0, double 1.0>, <2 x double>*@v2f64
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 1.0, double 31.0>, <2 x double>*@v2f64
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x double> <double 3.0, double 4.0>, <2 x double>*@v2f64
  ; MIPS32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
  ; MIPS32: .size const_v2f64
}

define void @nonconst_v4f32() nounwind {
  ; MIPS32: nonconst_v4f32:

  %1 = load float *@f32
  %2 = insertelement <4 x float> undef, float %1, i32 0
  %3 = insertelement <4 x float> %2, float %1, i32 1
  %4 = insertelement <4 x float> %3, float %1, i32 2
  %5 = insertelement <4 x float> %4, float %1, i32 3
  store volatile <4 x float> %5, <4 x float>*@v4f32
  ; MIPS32: lwc1 $f[[R1:[0-9]+]], 0(
  ; MIPS32: splati.w [[R2:\$w[0-9]+]], $w[[R1]]

  ret void
  ; MIPS32: .size nonconst_v4f32
}

define void @nonconst_v2f64() nounwind {
  ; MIPS32: nonconst_v2f64:

  %1 = load double *@f64
  %2 = insertelement <2 x double> undef, double %1, i32 0
  %3 = insertelement <2 x double> %2, double %1, i32 1
  store volatile <2 x double> %3, <2 x double>*@v2f64
  ; MIPS32: ldc1 $f[[R1:[0-9]+]], 0(
  ; MIPS32: splati.d [[R2:\$w[0-9]+]], $w[[R1]]

  ret void
  ; MIPS32: .size nonconst_v2f64
}

define float @extract_v4f32() nounwind {
  ; MIPS32: extract_v4f32:

  %1 = load <4 x float>* @v4f32
  ; MIPS32-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = fadd <4 x float> %1, %1
  ; MIPS32-DAG: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x float> %2, i32 1
  ; Element 1 can be obtained by splatting it across the vector and extracting
  ; $w0:sub_lo
  ; MIPS32-DAG: splati.w $w0, [[R1]][1]

  ret float %3
  ; MIPS32: .size extract_v4f32
}

define float @extract_v4f32_elt0() nounwind {
  ; MIPS32: extract_v4f32_elt0:

  %1 = load <4 x float>* @v4f32
  ; MIPS32-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = fadd <4 x float> %1, %1
  ; MIPS32-DAG: fadd.w $w0, [[R1]], [[R1]]

  %3 = extractelement <4 x float> %2, i32 0
  ; Element 0 can be obtained by extracting $w0:sub_lo ($f0)
  ; MIPS32-NOT: copy_u.w
  ; MIPS32-NOT: mtc1

  ret float %3
  ; MIPS32: .size extract_v4f32_elt0
}

define double @extract_v2f64() nounwind {
  ; MIPS32: extract_v2f64:

  %1 = load <2 x double>* @v2f64
  ; MIPS32-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = fadd <2 x double> %1, %1
  ; MIPS32-DAG: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <2 x double> %2, i32 1
  ; Element 1 can be obtained by splatting it across the vector and extracting
  ; $w0:sub_64
  ; MIPS32-DAG: splati.d $w0, [[R1]][1]
  ; MIPS32-NOT: copy_u.w
  ; MIPS32-NOT: mtc1
  ; MIPS32-NOT: mthc1
  ; MIPS32-NOT: sll
  ; MIPS32-NOT: sra

  ret double %3
  ; MIPS32: .size extract_v2f64
}

define double @extract_v2f64_elt0() nounwind {
  ; MIPS32: extract_v2f64_elt0:

  %1 = load <2 x double>* @v2f64
  ; MIPS32-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = fadd <2 x double> %1, %1
  ; MIPS32-DAG: fadd.d $w0, [[R1]], [[R1]]

  %3 = extractelement <2 x double> %2, i32 0
  ; Element 0 can be obtained by extracting $w0:sub_64 ($f0)
  ; MIPS32-NOT: copy_u.w
  ; MIPS32-NOT: mtc1
  ; MIPS32-NOT: mthc1
  ; MIPS32-NOT: sll
  ; MIPS32-NOT: sra

  ret double %3
  ; MIPS32: .size extract_v2f64_elt0
}

define void @insert_v4f32(float %a) nounwind {
  ; MIPS32: insert_v4f32:

  %1 = load <4 x float>* @v4f32
  ; MIPS32-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = insertelement <4 x float> %1, float %a, i32 1
  ; float argument passed in $f12
  ; MIPS32-DAG: insve.w [[R1]][1], $w12[0]

  store <4 x float> %2, <4 x float>* @v4f32
  ; MIPS32-DAG: st.w [[R1]]

  ret void
  ; MIPS32: .size insert_v4f32
}

define void @insert_v2f64(double %a) nounwind {
  ; MIPS32: insert_v2f64:

  %1 = load <2 x double>* @v2f64
  ; MIPS32-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = insertelement <2 x double> %1, double %a, i32 1
  ; double argument passed in $f12
  ; MIPS32-DAG: insve.d [[R1]][1], $w12[0]

  store <2 x double> %2, <2 x double>* @v2f64
  ; MIPS32-DAG: st.d [[R1]]

  ret void
  ; MIPS32: .size insert_v2f64
}
