; RUN: opt -inline -S < %s | FileCheck %s

; Created from source
;  1 extern int global_var;
;  2 extern int test_ext(int k);
;  3 __inline__ __attribute__((always_inline)) int test (int k) {
;  4   int k2 = test_ext(k);
;  5   if (k2 > 100)
;  6     return k2;
;  7   return 0;
;  8 }
;  9 
; 10 int test2() {
; 11   try
; 12   {
; 13     test(global_var);
; 14   }
; 15   catch (int e) {
; 16     global_var = 0;
; 17   }
; 18   global_var = 1;
; 19   return 0;
; 20 }

target triple = "x86_64-apple-darwin"

@_ZTIi = external constant i8*
@global_var = external global i32

define i32 @_Z5test2v() #0 {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %e = alloca i32, align 4
  %0 = load i32* @global_var, align 4, !dbg !11
  %call = invoke i32 @_Z4testi(i32 %0)
          to label %invoke.cont unwind label %lpad, !dbg !11

; CHECK: _Z4testi.exit:
; Make sure the branch instruction created during inlining has a debug location
; CHECK: br label %invoke.cont, !dbg !20
; The branch instruction has source location of line 8 and its inlined location
; has source location of line 13.
; CHECK: !11 = metadata !{i32 13, i32 0, metadata !12, null}
; CHECK: !20 = metadata !{i32 8, i32 0, metadata !8, metadata !11}

invoke.cont:
  br label %try.cont, !dbg !13

lpad:
  %1 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*), !dbg !11
  %2 = extractvalue { i8*, i32 } %1, 0, !dbg !11
  store i8* %2, i8** %exn.slot, !dbg !11
  %3 = extractvalue { i8*, i32 } %1, 1, !dbg !11
  store i32 %3, i32* %ehselector.slot, !dbg !11
  br label %catch.dispatch, !dbg !11

catch.dispatch:
  %sel = load i32* %ehselector.slot, !dbg !13
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #4, !dbg !13
  %matches = icmp eq i32 %sel, %4, !dbg !13
  br i1 %matches, label %catch, label %eh.resume, !dbg !13

catch:
  call void @llvm.dbg.declare(metadata !{i32* %e}, metadata !14), !dbg !15
  %exn = load i8** %exn.slot, !dbg !13
  %5 = call i8* @__cxa_begin_catch(i8* %exn) #4, !dbg !13
  %6 = bitcast i8* %5 to i32*, !dbg !13
  %exn.scalar = load i32* %6, !dbg !13
  store i32 %exn.scalar, i32* %e, align 4, !dbg !13
  store i32 0, i32* @global_var, align 4, !dbg !16
  call void @__cxa_end_catch() #4, !dbg !18
  br label %try.cont, !dbg !18

try.cont:
  store i32 1, i32* @global_var, align 4, !dbg !19
  ret i32 0, !dbg !20

eh.resume:
  %exn1 = load i8** %exn.slot, !dbg !13
  %sel2 = load i32* %ehselector.slot, !dbg !13
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn1, 0, !dbg !13
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel2, 1, !dbg !13
  resume { i8*, i32 } %lpad.val3, !dbg !13
}

define linkonce_odr i32 @_Z4testi(i32 %k) #1 {
entry:
  %retval = alloca i32, align 4
  %k.addr = alloca i32, align 4
  %k2 = alloca i32, align 4
  store i32 %k, i32* %k.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %k.addr}, metadata !21), !dbg !22
  call void @llvm.dbg.declare(metadata !{i32* %k2}, metadata !23), !dbg !24
  %0 = load i32* %k.addr, align 4, !dbg !24
  %call = call i32 @_Z8test_exti(i32 %0), !dbg !24
  store i32 %call, i32* %k2, align 4, !dbg !24
  %1 = load i32* %k2, align 4, !dbg !25
  %cmp = icmp sgt i32 %1, 100, !dbg !25
  br i1 %cmp, label %if.then, label %if.end, !dbg !25

if.then:
  %2 = load i32* %k2, align 4, !dbg !26
  store i32 %2, i32* %retval, !dbg !26
  br label %return, !dbg !26

if.end:
  store i32 0, i32* %retval, !dbg !27
  br label %return, !dbg !27

return:
  %3 = load i32* %retval, !dbg !28
  ret i32 %3, !dbg !28
}

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*) #2

declare void @llvm.dbg.declare(metadata, metadata) #2

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare i32 @_Z8test_exti(i32) #3

attributes #0 = { ssp uwtable "target-cpu"="core2" "target-features"="-sse4a,-avx2,-xop,-fma4,-bmi2,-3dnow,-3dnowa,-pclmul,+sse,-avx,-sse41,+ssse3,+mmx,-rtm,-sse42,-lzcnt,-f16c,-popcnt,-bmi,-aes,-fma,-rdrand,+sse2,+sse3" }
attributes #1 = { alwaysinline inlinehint ssp uwtable "target-cpu"="core2" "target-features"="-sse4a,-avx2,-xop,-fma4,-bmi2,-3dnow,-3dnowa,-pclmul,+sse,-avx,-sse41,+ssse3,+mmx,-rtm,-sse42,-lzcnt,-f16c,-popcnt,-bmi,-aes,-fma,-rdrand,+sse2,+sse3" }
attributes #2 = { nounwind readnone }
attributes #3 = { "target-cpu"="core2" "target-features"="-sse4a,-avx2,-xop,-fma4,-bmi2,-3dnow,-3dnowa,-pclmul,+sse,-avx,-sse41,+ssse3,+mmx,-rtm,-sse42,-lzcnt,-f16c,-popcnt,-bmi,-aes,-fma,-rdrand,+sse2,+sse3" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !"test3.cpp", metadata !"/Users/manmanren/test-Nov/rdar_12415623/unit_test3", metadata !"clang version 3.3 (trunk 176115) (llvm/trunk 176114:176220M)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !2, metadata !1, metadata !""} ; [ DW_TAG_compile_unit ] [/Users/manmanren/test-Nov/rdar_12415623/unit_test3/test3.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!2 = metadata !{metadata !3, metadata !8}
!3 = metadata !{i32 786478, i32 0, metadata !4, metadata !"test2", metadata !"test2", metadata !"_Z5test2v", metadata !4, i32 10, metadata !5, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z5test2v, null, null, metadata !1, i32 10} ; [ DW_TAG_subprogram ] [line 10] [def] [test2]
!4 = metadata !{i32 786473, metadata !"test3.cpp", metadata !"/Users/manmanren/test-Nov/rdar_12415623/unit_test3", null} ; [ DW_TAG_file_type ]
!5 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, i32 0, metadata !4, metadata !"test", metadata !"test", metadata !"_Z4testi", metadata !4, i32 3, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z4testi, null, null, metadata !1, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [test]
!9 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !7, metadata !7}
!11 = metadata !{i32 13, i32 0, metadata !12, null}
!12 = metadata !{i32 786443, metadata !3, i32 12, i32 0, metadata !4, i32 0} ; [ DW_TAG_lexical_block ] [/Users/manmanren/test-Nov/rdar_12415623/unit_test3/test3.cpp]
!13 = metadata !{i32 14, i32 0, metadata !12, null}
!14 = metadata !{i32 786688, metadata !3, metadata !"e", metadata !4, i32 15, metadata !7, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [e] [line 15]
!15 = metadata !{i32 15, i32 0, metadata !3, null}
!16 = metadata !{i32 16, i32 0, metadata !17, null}
!17 = metadata !{i32 786443, metadata !3, i32 15, i32 0, metadata !4, i32 1} ; [ DW_TAG_lexical_block ] [/Users/manmanren/test-Nov/rdar_12415623/unit_test3/test3.cpp]
!18 = metadata !{i32 17, i32 0, metadata !17, null}
!19 = metadata !{i32 18, i32 0, metadata !3, null}
!20 = metadata !{i32 19, i32 0, metadata !3, null}
!21 = metadata !{i32 786689, metadata !8, metadata !"k", metadata !4, i32 16777219, metadata !7, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [k] [line 3]
!22 = metadata !{i32 3, i32 0, metadata !8, null}
!23 = metadata !{i32 786688, metadata !8, metadata !"k2", metadata !4, i32 4, metadata !7, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [k2] [line 4]
!24 = metadata !{i32 4, i32 0, metadata !8, null}
!25 = metadata !{i32 5, i32 0, metadata !8, null}
!26 = metadata !{i32 6, i32 0, metadata !8, null}
!27 = metadata !{i32 7, i32 0, metadata !8, null}
!28 = metadata !{i32 8, i32 0, metadata !8, null}
