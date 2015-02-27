; RUN: llc -O0 -mtriple=x86_64-apple-darwin12 -filetype=obj -o - %s | macho-dump | FileCheck %s
; Test that we emit weak_odr thread_locals correctly into the thread_bss section
; PR15972

; CHECK: __thread_bss
; CHECK: 'size', 8
; CHECK: 'alignment', 3
; CHECK: __thread_vars

; Generated from this C++ source
; template<class T>
; struct Tls {
;   static __thread void* val;
; };

; template<class T> __thread void* Tls<T>::val;

; void* f(int x) {
;         return Tls<long>::val;
; }

@_ZN3TlsIlE3valE = weak_odr thread_local global i8* null, align 8

; Function Attrs: nounwind ssp uwtable
define i8* @_Z1fi(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i8*, i8** @_ZN3TlsIlE3valE, align 8
  ret i8* %0
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
