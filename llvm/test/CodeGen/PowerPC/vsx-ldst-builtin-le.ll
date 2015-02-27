; RUN: llc -mcpu=pwr8 -mattr=+vsx -O2 -mtriple=powerpc64le-unknown-linux-gnu < %s > %t
; RUN: grep lxvd2x < %t | count 18
; RUN: grep stxvd2x < %t | count 18
; RUN: grep xxpermdi < %t | count 36

@vf = global <4 x float> <float -1.500000e+00, float 2.500000e+00, float -3.500000e+00, float 4.500000e+00>, align 16
@vd = global <2 x double> <double 3.500000e+00, double -7.500000e+00>, align 16
@vsi = global <4 x i32> <i32 -1, i32 2, i32 -3, i32 4>, align 16
@vui = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>, align 16
@vsll = global <2 x i64> <i64 255, i64 -937>, align 16
@vull = global <2 x i64> <i64 1447, i64 2894>, align 16
@res_vsi = common global <4 x i32> zeroinitializer, align 16
@res_vui = common global <4 x i32> zeroinitializer, align 16
@res_vf = common global <4 x float> zeroinitializer, align 16
@res_vsll = common global <2 x i64> zeroinitializer, align 16
@res_vull = common global <2 x i64> zeroinitializer, align 16
@res_vd = common global <2 x double> zeroinitializer, align 16

define void @test1() {
entry:
; CHECK-LABEL: test1
  %__a.addr.i31 = alloca i32, align 4
  %__b.addr.i32 = alloca <4 x i32>*, align 8
  %__a.addr.i29 = alloca i32, align 4
  %__b.addr.i30 = alloca <4 x float>*, align 8
  %__a.addr.i27 = alloca i32, align 4
  %__b.addr.i28 = alloca <2 x i64>*, align 8
  %__a.addr.i25 = alloca i32, align 4
  %__b.addr.i26 = alloca <2 x i64>*, align 8
  %__a.addr.i23 = alloca i32, align 4
  %__b.addr.i24 = alloca <2 x double>*, align 8
  %__a.addr.i20 = alloca <4 x i32>, align 16
  %__b.addr.i21 = alloca i32, align 4
  %__c.addr.i22 = alloca <4 x i32>*, align 8
  %__a.addr.i17 = alloca <4 x i32>, align 16
  %__b.addr.i18 = alloca i32, align 4
  %__c.addr.i19 = alloca <4 x i32>*, align 8
  %__a.addr.i14 = alloca <4 x float>, align 16
  %__b.addr.i15 = alloca i32, align 4
  %__c.addr.i16 = alloca <4 x float>*, align 8
  %__a.addr.i11 = alloca <2 x i64>, align 16
  %__b.addr.i12 = alloca i32, align 4
  %__c.addr.i13 = alloca <2 x i64>*, align 8
  %__a.addr.i8 = alloca <2 x i64>, align 16
  %__b.addr.i9 = alloca i32, align 4
  %__c.addr.i10 = alloca <2 x i64>*, align 8
  %__a.addr.i6 = alloca <2 x double>, align 16
  %__b.addr.i7 = alloca i32, align 4
  %__c.addr.i = alloca <2 x double>*, align 8
  %__a.addr.i = alloca i32, align 4
  %__b.addr.i = alloca <4 x i32>*, align 8
  store i32 0, i32* %__a.addr.i, align 4
  store <4 x i32>* @vsi, <4 x i32>** %__b.addr.i, align 8
  %0 = load i32* %__a.addr.i, align 4
  %1 = load <4 x i32>** %__b.addr.i, align 8
  %2 = bitcast <4 x i32>* %1 to i8*
  %3 = getelementptr i8, i8* %2, i32 %0
  %4 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* %3)
  store <4 x i32> %4, <4 x i32>* @res_vsi, align 16
  store i32 0, i32* %__a.addr.i31, align 4
  store <4 x i32>* @vui, <4 x i32>** %__b.addr.i32, align 8
  %5 = load i32* %__a.addr.i31, align 4
  %6 = load <4 x i32>** %__b.addr.i32, align 8
  %7 = bitcast <4 x i32>* %6 to i8*
  %8 = getelementptr i8, i8* %7, i32 %5
  %9 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* %8)
  store <4 x i32> %9, <4 x i32>* @res_vui, align 16
  store i32 0, i32* %__a.addr.i29, align 4
  store <4 x float>* @vf, <4 x float>** %__b.addr.i30, align 8
  %10 = load i32* %__a.addr.i29, align 4
  %11 = load <4 x float>** %__b.addr.i30, align 8
  %12 = bitcast <4 x float>* %11 to i8*
  %13 = getelementptr i8, i8* %12, i32 %10
  %14 = call <4 x i32> @llvm.ppc.vsx.lxvw4x(i8* %13)
  %15 = bitcast <4 x i32> %14 to <4 x float>
  store <4 x float> %15, <4 x float>* @res_vf, align 16
  store i32 0, i32* %__a.addr.i27, align 4
  store <2 x i64>* @vsll, <2 x i64>** %__b.addr.i28, align 8
  %16 = load i32* %__a.addr.i27, align 4
  %17 = load <2 x i64>** %__b.addr.i28, align 8
  %18 = bitcast <2 x i64>* %17 to i8*
  %19 = getelementptr i8, i8* %18, i32 %16
  %20 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* %19)
  %21 = bitcast <2 x double> %20 to <2 x i64>
  store <2 x i64> %21, <2 x i64>* @res_vsll, align 16
  store i32 0, i32* %__a.addr.i25, align 4
  store <2 x i64>* @vull, <2 x i64>** %__b.addr.i26, align 8
  %22 = load i32* %__a.addr.i25, align 4
  %23 = load <2 x i64>** %__b.addr.i26, align 8
  %24 = bitcast <2 x i64>* %23 to i8*
  %25 = getelementptr i8, i8* %24, i32 %22
  %26 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* %25)
  %27 = bitcast <2 x double> %26 to <2 x i64>
  store <2 x i64> %27, <2 x i64>* @res_vull, align 16
  store i32 0, i32* %__a.addr.i23, align 4
  store <2 x double>* @vd, <2 x double>** %__b.addr.i24, align 8
  %28 = load i32* %__a.addr.i23, align 4
  %29 = load <2 x double>** %__b.addr.i24, align 8
  %30 = bitcast <2 x double>* %29 to i8*
  %31 = getelementptr i8, i8* %30, i32 %28
  %32 = call <2 x double> @llvm.ppc.vsx.lxvd2x(i8* %31)
  store <2 x double> %32, <2 x double>* @res_vd, align 16
  %33 = load <4 x i32>* @vsi, align 16
  store <4 x i32> %33, <4 x i32>* %__a.addr.i20, align 16
  store i32 0, i32* %__b.addr.i21, align 4
  store <4 x i32>* @res_vsi, <4 x i32>** %__c.addr.i22, align 8
  %34 = load <4 x i32>* %__a.addr.i20, align 16
  %35 = load i32* %__b.addr.i21, align 4
  %36 = load <4 x i32>** %__c.addr.i22, align 8
  %37 = bitcast <4 x i32>* %36 to i8*
  %38 = getelementptr i8, i8* %37, i32 %35
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %34, i8* %38)
  %39 = load <4 x i32>* @vui, align 16
  store <4 x i32> %39, <4 x i32>* %__a.addr.i17, align 16
  store i32 0, i32* %__b.addr.i18, align 4
  store <4 x i32>* @res_vui, <4 x i32>** %__c.addr.i19, align 8
  %40 = load <4 x i32>* %__a.addr.i17, align 16
  %41 = load i32* %__b.addr.i18, align 4
  %42 = load <4 x i32>** %__c.addr.i19, align 8
  %43 = bitcast <4 x i32>* %42 to i8*
  %44 = getelementptr i8, i8* %43, i32 %41
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %40, i8* %44)
  %45 = load <4 x float>* @vf, align 16
  store <4 x float> %45, <4 x float>* %__a.addr.i14, align 16
  store i32 0, i32* %__b.addr.i15, align 4
  store <4 x float>* @res_vf, <4 x float>** %__c.addr.i16, align 8
  %46 = load <4 x float>* %__a.addr.i14, align 16
  %47 = bitcast <4 x float> %46 to <4 x i32>
  %48 = load i32* %__b.addr.i15, align 4
  %49 = load <4 x float>** %__c.addr.i16, align 8
  %50 = bitcast <4 x float>* %49 to i8*
  %51 = getelementptr i8, i8* %50, i32 %48
  call void @llvm.ppc.vsx.stxvw4x(<4 x i32> %47, i8* %51) #1
  %52 = load <2 x i64>* @vsll, align 16
  store <2 x i64> %52, <2 x i64>* %__a.addr.i11, align 16
  store i32 0, i32* %__b.addr.i12, align 4
  store <2 x i64>* @res_vsll, <2 x i64>** %__c.addr.i13, align 8
  %53 = load <2 x i64>* %__a.addr.i11, align 16
  %54 = bitcast <2 x i64> %53 to <2 x double>
  %55 = load i32* %__b.addr.i12, align 4
  %56 = load <2 x i64>** %__c.addr.i13, align 8
  %57 = bitcast <2 x i64>* %56 to i8*
  %58 = getelementptr i8, i8* %57, i32 %55
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %54, i8* %58)
  %59 = load <2 x i64>* @vull, align 16
  store <2 x i64> %59, <2 x i64>* %__a.addr.i8, align 16
  store i32 0, i32* %__b.addr.i9, align 4
  store <2 x i64>* @res_vull, <2 x i64>** %__c.addr.i10, align 8
  %60 = load <2 x i64>* %__a.addr.i8, align 16
  %61 = bitcast <2 x i64> %60 to <2 x double>
  %62 = load i32* %__b.addr.i9, align 4
  %63 = load <2 x i64>** %__c.addr.i10, align 8
  %64 = bitcast <2 x i64>* %63 to i8*
  %65 = getelementptr i8, i8* %64, i32 %62
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %61, i8* %65)
  %66 = load <2 x double>* @vd, align 16
  store <2 x double> %66, <2 x double>* %__a.addr.i6, align 16
  store i32 0, i32* %__b.addr.i7, align 4
  store <2 x double>* @res_vd, <2 x double>** %__c.addr.i, align 8
  %67 = load <2 x double>* %__a.addr.i6, align 16
  %68 = load i32* %__b.addr.i7, align 4
  %69 = load <2 x double>** %__c.addr.i, align 8
  %70 = bitcast <2 x double>* %69 to i8*
  %71 = getelementptr i8, i8* %70, i32 %68
  call void @llvm.ppc.vsx.stxvd2x(<2 x double> %67, i8* %71)
  ret void
}

declare void @llvm.ppc.vsx.stxvd2x(<2 x double>, i8*)
declare void @llvm.ppc.vsx.stxvw4x(<4 x i32>, i8*)
declare <2 x double> @llvm.ppc.vsx.lxvd2x(i8*)
declare <4 x i32> @llvm.ppc.vsx.lxvw4x(i8*)
