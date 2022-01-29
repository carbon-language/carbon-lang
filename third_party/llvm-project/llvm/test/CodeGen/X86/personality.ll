; RUN: llc < %s -mtriple=x86_64-apple-darwin9 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=i386-apple-darwin9 | FileCheck %s -check-prefix=X32
; PR1632

define void @_Z1fv() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @_Z1gv()
          to label %return unwind label %unwind

unwind:                                           ; preds = %entry
  %exn = landingpad {i8*, i32}
            cleanup
  br i1 false, label %eh_then, label %cleanup20

eh_then:                                          ; preds = %unwind
  invoke void @__cxa_end_catch()
          to label %return unwind label %unwind10

unwind10:                                         ; preds = %eh_then
  %exn10 = landingpad {i8*, i32}
            cleanup
  %upgraded.eh_select13 = extractvalue { i8*, i32 } %exn10, 1
  %upgraded.eh_select131 = sext i32 %upgraded.eh_select13 to i64
  %tmp18 = icmp slt i64 %upgraded.eh_select131, 0
  br i1 %tmp18, label %filter, label %cleanup20

filter:                                           ; preds = %unwind10
  unreachable

cleanup20:                                        ; preds = %unwind10, %unwind
  %eh_selector.0 = phi i64 [ 0, %unwind ], [ %upgraded.eh_select131, %unwind10 ]
  ret void

return:                                           ; preds = %eh_then, %entry
  ret void
}

declare void @_Z1gv()

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)

; X64-NOT: .quad ___gxx_personality_v0
; X64: .cfi_personality 155, ___gxx_personality_v0

; X32-NOT: .long ___gxx_personality_v0
; X32: .cfi_personality 155, L___gxx_personality_v0$non_lazy_ptr

; X32:        .section	__IMPORT,__pointers,non_lazy_symbol_pointers
; X32-NEXT: L___gxx_personality_v0$non_lazy_ptr:
; X32-NEXT:   .indirect_symbol ___gxx_personality_v0
