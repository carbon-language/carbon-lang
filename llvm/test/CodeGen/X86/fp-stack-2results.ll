; RUN: llvm-as < %s | llc -march=x86 | grep fldz
; RUN: llvm-as < %s | llc -march=x86-64 | grep fld1

; This is basically this code on x86-64:
; _Complex long double test() { return 1.0; }
define {x86_fp80, x86_fp80} @test() {
  %A = fpext double 1.0 to x86_fp80
  %B = fpext double 0.0 to x86_fp80
  ret x86_fp80 %A, x86_fp80 %B
}


;_test2:
;	fld1
;	fld	%st(0)
;	ret
define {x86_fp80, x86_fp80} @test2() {
  %A = fpext double 1.0 to x86_fp80
  ret x86_fp80 %A, x86_fp80 %A
}

