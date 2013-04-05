; RUN: %lli -force-interpreter=true %s | FileCheck %s
; XFAIL: mips
; CHECK: 1
; CHECK: 2
; CHECK: 3
; CHECK: 4
; CHECK: 5.{{[0]+}}e+{{[0]+}}
; CHECK: 6.{{[0]+}}e+{{[0]+}}
; CHECK: 7.{{[0]+}}e+{{[0]+}}
; CHECK: 8.{{[0]+}}e+{{[0]+}}
; CHECK: 9.{{[0]+}}e+{{[0]+}}
; CHECK: 1.{{[0]+}}e+{{[0]+}}1
; CHECK: 1.1{{[0]+}}e+{{[0]+}}1
; CHECK: 1.2{{[0]+}}e+{{[0]+}}1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

@format_i32 = internal global [4 x i8] c"%d\0A\00"
@format_float = internal global [4 x i8] c"%e\0A\00"

declare i32 @printf(i8*, ...)

define i32 @main() {
  %a = alloca <4 x i32>, align 16
  %b = alloca <4 x double>, align 16
  %c = alloca <4 x float>, align 16
  
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32>* %a, align 16

  %val0 = load <4 x i32> *%a, align 16

  %res_i32_0 = extractelement <4 x i32> %val0, i32 0
  %res_i32_1 = extractelement <4 x i32> %val0, i32 1
  %res_i32_2 = extractelement <4 x i32> %val0, i32 2
  %res_i32_3 = extractelement <4 x i32> %val0, i32 3
  
  %ptr0 = getelementptr [4 x i8]* @format_i32, i32 0, i32 0
  call i32 (i8*,...)* @printf(i8* %ptr0, i32 %res_i32_0)
  call i32 (i8*,...)* @printf(i8* %ptr0, i32 %res_i32_1)
  call i32 (i8*,...)* @printf(i8* %ptr0, i32 %res_i32_2)
  call i32 (i8*,...)* @printf(i8* %ptr0, i32 %res_i32_3)

  store <4 x double> <double 5.0, double 6.0, double 7.0, double 8.0>, <4 x double>* %b, align 16

  %val1 = load <4 x double> *%b, align 16

  %res_double_0 = extractelement <4 x double> %val1, i32 0
  %res_double_1 = extractelement <4 x double> %val1, i32 1
  %res_double_2 = extractelement <4 x double> %val1, i32 2
  %res_double_3 = extractelement <4 x double> %val1, i32 3
  
  %ptr1 = getelementptr [4 x i8]* @format_float, i32 0, i32 0
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_double_0)
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_double_1)
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_double_2)
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_double_3)


  store <4 x float> <float 9.0, float 10.0, float 11.0, float 12.0>, <4 x float>* %c, align 16
  
  %val2 = load <4 x float> *%c, align 16
  
  %ptr2 = getelementptr [4 x i8]* @format_float, i32 0, i32 0

  ; by some reason printf doesn't print float correctly, so
  ; floats are casted to doubles and are printed as doubles
  
  %res_serv_0 = extractelement <4 x float> %val2, i32 0
  %res_float_0 = fpext float %res_serv_0 to double
  %res_serv_1 = extractelement <4 x float> %val2, i32 1
  %res_float_1 = fpext float %res_serv_1 to double
  %res_serv_2 = extractelement <4 x float> %val2, i32 2
  %res_float_2 = fpext float %res_serv_2 to double
  %res_serv_3 = extractelement <4 x float> %val2, i32 3
  %res_float_3 = fpext float %res_serv_3 to double

 
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_float_0)
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_float_1)
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_float_2)
  call i32 (i8*,...)* @printf(i8* %ptr1, double %res_float_3)
 
  
  ret i32 0
}
