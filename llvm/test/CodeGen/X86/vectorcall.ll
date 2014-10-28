; RUN: llc -mtriple=i686-pc-win32 -mattr=+sse2 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=X86
; RUN: llc -mtriple=x86_64-pc-win32 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=X64

; Test integer arguments.

define x86_vectorcallcc i32 @test_int_1() {
  ret i32 0
}

; CHECK-LABEL: {{^}}test_int_1@@0:
; CHECK: xorl %eax, %eax

define x86_vectorcallcc i32 @test_int_2(i32 inreg %a) {
  ret i32 %a
}

; X86-LABEL: {{^}}test_int_2@@4:
; X64-LABEL: {{^}}test_int_2@@8:
; CHECK: movl %ecx, %eax

define x86_vectorcallcc i32 @test_int_3(i64 inreg %a) {
  %at = trunc i64 %a to i32
  ret i32 %at
}

; X86-LABEL: {{^}}test_int_3@@8:
; X64-LABEL: {{^}}test_int_3@@8:
; CHECK: movl %ecx, %eax

define x86_vectorcallcc i32 @test_int_4(i32 inreg %a, i32 inreg %b) {
  %s = add i32 %a, %b
  ret i32 %s
}

; X86-LABEL: {{^}}test_int_4@@8:
; X86: leal (%ecx,%edx), %eax

; X64-LABEL: {{^}}test_int_4@@16:
; X64: leal (%rcx,%rdx), %eax

define x86_vectorcallcc i32 @"\01test_int_5"(i32, i32) {
  ret i32 0
}
; CHECK-LABEL: {{^}}test_int_5:

define x86_vectorcallcc double @test_fp_1(double %a, double %b) {
  ret double %b
}
; CHECK-LABEL: {{^}}test_fp_1@@16:
; CHECK: movaps %xmm1, %xmm0

define x86_vectorcallcc double @test_fp_2(
    double, double, double, double, double, double, double %r) {
  ret double %r
}
; CHECK-LABEL: {{^}}test_fp_2@@56:
; CHECK: movsd {{[0-9]+\(%[re]sp\)}}, %xmm0

define x86_vectorcallcc {double, double, double, double} @test_fp_3() {
  ret {double, double, double, double}
        { double 0.0, double 0.0, double 0.0, double 0.0 }
}
; CHECK-LABEL: {{^}}test_fp_3@@0:
; CHECK: xorps %xmm0
; CHECK: xorps %xmm1
; CHECK: xorps %xmm2
; CHECK: xorps %xmm3

; FIXME: Returning via x87 isn't compatible, but its hard to structure the
; tablegen any other way.
define x86_vectorcallcc {double, double, double, double, double} @test_fp_4() {
  ret {double, double, double, double, double}
        { double 0.0, double 0.0, double 0.0, double 0.0, double 0.0 }
}
; CHECK-LABEL: {{^}}test_fp_4@@0:
; CHECK: fldz
; CHECK: xorps %xmm0
; CHECK: xorps %xmm1
; CHECK: xorps %xmm2
; CHECK: xorps %xmm3

define x86_vectorcallcc <16 x i8> @test_vec_1(<16 x i8> %a, <16 x i8> %b) {
  ret <16 x i8> %b
}
; CHECK-LABEL: {{^}}test_vec_1@@32:
; CHECK: movaps %xmm1, %xmm0

define x86_vectorcallcc <16 x i8> @test_vec_2(
    double, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> %r) {
  ret <16 x i8> %r
}
; CHECK-LABEL: {{^}}test_vec_2@@104:
; CHECK: movaps (%{{[re]}}cx), %xmm0
