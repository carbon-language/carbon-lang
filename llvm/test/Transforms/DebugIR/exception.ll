; ModuleID = 'exception.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTIi = external constant i8*

; Function Attrs: uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  %cleanup.dest.slot = alloca i32
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !13), !dbg !14
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !15), !dbg !14
  %exception = call i8* @__cxa_allocate_exception(i64 4) #2, !dbg !16
  %0 = bitcast i8* %exception to i32*, !dbg !16
  %1 = load i32* %argc.addr, align 4, !dbg !16
  store i32 %1, i32* %0, !dbg !16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
          to label %unreachable unwind label %lpad, !dbg !16

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*), !dbg !16
  %3 = extractvalue { i8*, i32 } %2, 0, !dbg !16
  store i8* %3, i8** %exn.slot, !dbg !16
  %4 = extractvalue { i8*, i32 } %2, 1, !dbg !16
  store i32 %4, i32* %ehselector.slot, !dbg !16
  br label %catch.dispatch, !dbg !16

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32* %ehselector.slot, !dbg !18
  %5 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2, !dbg !18
  %matches = icmp eq i32 %sel, %5, !dbg !18
  br i1 %matches, label %catch, label %eh.resume, !dbg !18

catch:                                            ; preds = %catch.dispatch
  call void @llvm.dbg.declare(metadata !{i32* %e}, metadata !19), !dbg !20
  %exn = load i8** %exn.slot, !dbg !18
  %6 = call i8* @__cxa_begin_catch(i8* %exn) #2, !dbg !18
  %7 = bitcast i8* %6 to i32*, !dbg !18
  %8 = load i32* %7, align 4, !dbg !18
  store i32 %8, i32* %e, align 4, !dbg !18
  %9 = load i32* %e, align 4, !dbg !21
  store i32 %9, i32* %retval, !dbg !21
  store i32 1, i32* %cleanup.dest.slot
  call void @__cxa_end_catch() #2, !dbg !23
  br label %try.cont

try.cont:                                         ; preds = %catch
  %10 = load i32* %retval, !dbg !24
  ret i32 %10, !dbg !24

eh.resume:                                        ; preds = %catch.dispatch
  %exn1 = load i8** %exn.slot, !dbg !18
  %sel2 = load i32* %ehselector.slot, !dbg !18
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn1, 0, !dbg !18
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel2, 1, !dbg !18
  resume { i8*, i32 } %lpad.val3, !dbg !18

unreachable:                                      ; preds = %entry
  unreachable
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare i8* @__cxa_allocate_exception(i64)

declare void @__cxa_throw(i8*, i8*, i8*)

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/exception.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"exception.cpp", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 10, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32, i8**)* @main, null, null, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 10] [def] [scope 11] [main]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/exception.cpp]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8, metadata !9}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!11 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!12 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!13 = metadata !{i32 786689, metadata !4, metadata !"argc", metadata !5, i32 16777226, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [argc] [line 10]
!14 = metadata !{i32 10, i32 0, metadata !4, null}
!15 = metadata !{i32 786689, metadata !4, metadata !"argv", metadata !5, i32 33554442, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [argv] [line 10]
!16 = metadata !{i32 13, i32 0, metadata !17, null}
!17 = metadata !{i32 786443, metadata !1, metadata !4, i32 12, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/exception.cpp]
!18 = metadata !{i32 14, i32 0, metadata !17, null}
!19 = metadata !{i32 786688, metadata !4, metadata !"e", metadata !5, i32 14, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [e] [line 14]
!20 = metadata !{i32 14, i32 0, metadata !4, null}
!21 = metadata !{i32 15, i32 0, metadata !22, null}
!22 = metadata !{i32 786443, metadata !1, metadata !4, i32 14, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/exception.cpp]
!23 = metadata !{i32 16, i32 0, metadata !22, null}
!24 = metadata !{i32 17, i32 0, metadata !4, null}
; RUN: opt < %s -debug-ir -S | FileCheck %s.check
