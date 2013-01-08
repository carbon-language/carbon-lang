; This tests .eh_frame CIE descriptor for the.
; Data alignment factor

; RUN: llc -filetype=obj -mcpu=mips64r2 -mattr=n64 -march=mips64el %s -o - \
; RUN: | llvm-objdump -s - | FileCheck %s

; N64
; CHECK: Contents of section .eh_frame:
; CHECK-NEXT:  0000 1c000000 00000000 017a504c 52000178  .........zPLR..x
; CHECK-NEXT:  0010 1f0b0000 00000000 00000000 000c1d00  ................
; CHECK-NEXT:  0020 2c000000 24000000 00000000 00000000  ,...$...........
; CHECK-NEXT:  0030 7c000000 00000000 08000000 00000000  |...............
; CHECK-NEXT:  0040 00440e10 489f019c 02000000 00000000  .D..H...........

; ModuleID = 'simple_throw.cpp'

@_ZTIi = external constant i8*
@str = private unnamed_addr constant [7 x i8] c"All ok\00"

define i32 @main() {
entry:
  %exception.i = tail call i8* @__cxa_allocate_exception(i64 4) nounwind
  %0 = bitcast i8* %exception.i to i32*
  store i32 5, i32* %0, align 4
  invoke void @__cxa_throw(i8* %exception.i, i8* bitcast (i8** @_ZTIi to i8*), i8* null) noreturn
          to label %.noexc unwind label %return

.noexc:                                           ; preds = %entry
  unreachable

return:                                           ; preds = %entry
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2) nounwind
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([7 x i8]* @str, i64 0, i64 0))
  tail call void @__cxa_end_catch()
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @puts(i8* nocapture) nounwind

