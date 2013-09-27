; RUN: llc -march=mips -mattr=+msa,+fp64 < %s | FileCheck -check-prefix=MIPS32 %s

@v4f32 = global <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>
@v2f64 = global <2 x double> <double 0.0, double 0.0>

define void @const_v4f32() nounwind {
  ; MIPS32: const_v4f32:

  store volatile <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>, <4 x float>*@v4f32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, <4 x float>*@v4f32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 31.0>, <4 x float>*@v4f32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x float> <float 65537.0, float 65537.0, float 65537.0, float 65537.0>, <4 x float>*@v4f32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x float> <float 1.0, float 2.0, float 1.0, float 2.0>, <4 x float>*@v4f32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  store volatile <4 x float> <float 3.0, float 4.0, float 5.0, float 6.0>, <4 x float>*@v4f32
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], %lo(

  ret void
  ; MIPS32: .size const_v4f32
}

define void @const_v2f64() nounwind {
  ; MIPS32: const_v2f64:

  store volatile <2 x double> <double 0.0, double 0.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x double> <double 72340172838076673.0, double 72340172838076673.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x double> <double 281479271743489.0, double 281479271743489.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x double> <double 4294967297.0, double 4294967297.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x double> <double 1.0, double 1.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x double> <double 1.0, double 31.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  store volatile <2 x double> <double 3.0, double 4.0>, <2 x double>*@v2f64
  ; MIPS32: ld.d  [[R1:\$w[0-9]+]], %lo(

  ret void
  ; MIPS32: .size const_v2f64
}
