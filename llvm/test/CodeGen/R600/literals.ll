; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; Test using an integer literal constant.
; Generated ASM should be:
; ADD_INT REG literal.x, 5
; or
; ADD_INT literal.x REG, 5

; CHECK: @i32_literal
; CHECK: ADD_INT * {{[A-Z0-9,. ]*}}literal.x
; CHECK-NEXT: 5
define void @i32_literal(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 5, %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Test using a float literal constant.
; Generated ASM should be:
; ADD REG literal.x, 5.0
; or
; ADD literal.x REG, 5.0

; CHECK: @float_literal
; CHECK: ADD * {{[A-Z0-9,. ]*}}literal.x
; CHECK-NEXT: 1084227584(5.0
define void @float_literal(float addrspace(1)* %out, float %in) {
entry:
  %0 = fadd float 5.0, %in
  store float %0, float addrspace(1)* %out
  ret void
}

; CHECK: @main
; CHECK: -2147483648
; CHECK-NEXT-NOT: -2147483648

define void @main() #0 {
main_body:
  %0 = call float @llvm.R600.load.input(i32 4)
  %1 = call float @llvm.R600.load.input(i32 5)
  %2 = call float @llvm.R600.load.input(i32 6)
  %3 = call float @llvm.R600.load.input(i32 7)
  %4 = call float @llvm.R600.load.input(i32 8)
  %5 = call float @llvm.R600.load.input(i32 9)
  %6 = call float @llvm.R600.load.input(i32 10)
  %7 = call float @llvm.R600.load.input(i32 11)
  %8 = call float @llvm.R600.load.input(i32 12)
  %9 = call float @llvm.R600.load.input(i32 13)
  %10 = call float @llvm.R600.load.input(i32 14)
  %11 = call float @llvm.R600.load.input(i32 15)
  %12 = load <4 x float> addrspace(8)* null
  %13 = extractelement <4 x float> %12, i32 0
  %14 = fsub float -0.000000e+00, %13
  %15 = fadd float %0, %14
  %16 = load <4 x float> addrspace(8)* null
  %17 = extractelement <4 x float> %16, i32 1
  %18 = fsub float -0.000000e+00, %17
  %19 = fadd float %1, %18
  %20 = load <4 x float> addrspace(8)* null
  %21 = extractelement <4 x float> %20, i32 2
  %22 = fsub float -0.000000e+00, %21
  %23 = fadd float %2, %22
  %24 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %25 = extractelement <4 x float> %24, i32 0
  %26 = fmul float %25, %0
  %27 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %28 = extractelement <4 x float> %27, i32 1
  %29 = fmul float %28, %0
  %30 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %31 = extractelement <4 x float> %30, i32 2
  %32 = fmul float %31, %0
  %33 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %34 = extractelement <4 x float> %33, i32 3
  %35 = fmul float %34, %0
  %36 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %37 = extractelement <4 x float> %36, i32 0
  %38 = fmul float %37, %1
  %39 = fadd float %38, %26
  %40 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %41 = extractelement <4 x float> %40, i32 1
  %42 = fmul float %41, %1
  %43 = fadd float %42, %29
  %44 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %45 = extractelement <4 x float> %44, i32 2
  %46 = fmul float %45, %1
  %47 = fadd float %46, %32
  %48 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %49 = extractelement <4 x float> %48, i32 3
  %50 = fmul float %49, %1
  %51 = fadd float %50, %35
  %52 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %53 = extractelement <4 x float> %52, i32 0
  %54 = fmul float %53, %2
  %55 = fadd float %54, %39
  %56 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %57 = extractelement <4 x float> %56, i32 1
  %58 = fmul float %57, %2
  %59 = fadd float %58, %43
  %60 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %61 = extractelement <4 x float> %60, i32 2
  %62 = fmul float %61, %2
  %63 = fadd float %62, %47
  %64 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %65 = extractelement <4 x float> %64, i32 3
  %66 = fmul float %65, %2
  %67 = fadd float %66, %51
  %68 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %69 = extractelement <4 x float> %68, i32 0
  %70 = fmul float %69, %3
  %71 = fadd float %70, %55
  %72 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %73 = extractelement <4 x float> %72, i32 1
  %74 = fmul float %73, %3
  %75 = fadd float %74, %59
  %76 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %77 = extractelement <4 x float> %76, i32 2
  %78 = fmul float %77, %3
  %79 = fadd float %78, %63
  %80 = load <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %81 = extractelement <4 x float> %80, i32 3
  %82 = fmul float %81, %3
  %83 = fadd float %82, %67
  %84 = insertelement <4 x float> undef, float %15, i32 0
  %85 = insertelement <4 x float> %84, float %19, i32 1
  %86 = insertelement <4 x float> %85, float %23, i32 2
  %87 = insertelement <4 x float> %86, float 0.000000e+00, i32 3
  %88 = insertelement <4 x float> undef, float %15, i32 0
  %89 = insertelement <4 x float> %88, float %19, i32 1
  %90 = insertelement <4 x float> %89, float %23, i32 2
  %91 = insertelement <4 x float> %90, float 0.000000e+00, i32 3
  %92 = call float @llvm.AMDGPU.dp4(<4 x float> %87, <4 x float> %91)
  %93 = call float @fabs(float %92)
  %94 = call float @llvm.AMDGPU.rsq(float %93)
  %95 = fmul float %15, %94
  %96 = fmul float %19, %94
  %97 = fmul float %23, %94
  %98 = insertelement <4 x float> undef, float %4, i32 0
  %99 = insertelement <4 x float> %98, float %5, i32 1
  %100 = insertelement <4 x float> %99, float %6, i32 2
  %101 = insertelement <4 x float> %100, float 0.000000e+00, i32 3
  %102 = insertelement <4 x float> undef, float %4, i32 0
  %103 = insertelement <4 x float> %102, float %5, i32 1
  %104 = insertelement <4 x float> %103, float %6, i32 2
  %105 = insertelement <4 x float> %104, float 0.000000e+00, i32 3
  %106 = call float @llvm.AMDGPU.dp4(<4 x float> %101, <4 x float> %105)
  %107 = call float @fabs(float %106)
  %108 = call float @llvm.AMDGPU.rsq(float %107)
  %109 = fmul float %4, %108
  %110 = fmul float %5, %108
  %111 = fmul float %6, %108
  %112 = insertelement <4 x float> undef, float %95, i32 0
  %113 = insertelement <4 x float> %112, float %96, i32 1
  %114 = insertelement <4 x float> %113, float %97, i32 2
  %115 = insertelement <4 x float> %114, float 0.000000e+00, i32 3
  %116 = insertelement <4 x float> undef, float %109, i32 0
  %117 = insertelement <4 x float> %116, float %110, i32 1
  %118 = insertelement <4 x float> %117, float %111, i32 2
  %119 = insertelement <4 x float> %118, float 0.000000e+00, i32 3
  %120 = call float @llvm.AMDGPU.dp4(<4 x float> %115, <4 x float> %119)
  %121 = fsub float -0.000000e+00, %120
  %122 = fcmp uge float 0.000000e+00, %121
  %123 = select i1 %122, float 0.000000e+00, float %121
  %124 = insertelement <4 x float> undef, float %8, i32 0
  %125 = insertelement <4 x float> %124, float %9, i32 1
  %126 = insertelement <4 x float> %125, float 5.000000e-01, i32 2
  %127 = insertelement <4 x float> %126, float 1.000000e+00, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %127, i32 60, i32 1)
  %128 = insertelement <4 x float> undef, float %71, i32 0
  %129 = insertelement <4 x float> %128, float %75, i32 1
  %130 = insertelement <4 x float> %129, float %79, i32 2
  %131 = insertelement <4 x float> %130, float %83, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %131, i32 0, i32 2)
  %132 = insertelement <4 x float> undef, float %123, i32 0
  %133 = insertelement <4 x float> %132, float %96, i32 1
  %134 = insertelement <4 x float> %133, float %97, i32 2
  %135 = insertelement <4 x float> %134, float 0.000000e+00, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %135, i32 1, i32 2)
  ret void
}

; Function Attrs: readnone
declare float @llvm.R600.load.input(i32) #1

; Function Attrs: readnone
declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1

; Function Attrs: readonly
declare float @fabs(float) #2

; Function Attrs: readnone
declare float @llvm.AMDGPU.rsq(float) #1

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { readnone }
attributes #2 = { readonly }
