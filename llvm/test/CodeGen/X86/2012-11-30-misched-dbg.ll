; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test MachineScheduler handling of DBG_VALUE.
; rdar://12776937.
;
; CHECK: %if.else581
; CHECK: DEBUG_VALUE: num1
; CHECK: call

%union.rec = type {}

@.str15 = external hidden unnamed_addr constant [6 x i8], align 1

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @AttachGalley(%union.rec** nocapture %suspend_pt) nounwind uwtable ssp {
entry:
  %num14075 = alloca [20 x i8], align 16
  br label %if.end33

if.end33:                                         ; preds = %entry
  %cmp1733 = icmp eq i32 undef, 0
  br label %if.else581

if.else581:                                       ; preds = %if.end33
  %cmp586 = icmp eq i8 undef, -123
  br i1 %cmp586, label %if.then588, label %if.else594

if.then588:                                       ; preds = %if.else581
  br label %for.cond1710.preheader

if.else594:                                       ; preds = %if.else581
  unreachable

for.cond1710.preheader:                           ; preds = %if.then588
  br label %for.cond1710

for.cond1710:                                     ; preds = %for.cond1710, %for.cond1710.preheader
  br i1 undef, label %for.cond1710, label %if.then3344

if.then3344:
  br label %if.then4073

if.then4073:                                      ; preds = %if.then3344
  call void @llvm.dbg.declare(metadata [20 x i8]* %num14075, metadata !4, metadata !{!"0x102"})
  %arraydecay4078 = getelementptr inbounds [20 x i8], [20 x i8]* %num14075, i64 0, i64 0
  %0 = load i32* undef, align 4
  %add4093 = add nsw i32 %0, 0
  %conv4094 = sitofp i32 %add4093 to float
  %div4095 = fdiv float %conv4094, 5.670000e+02
  %conv4096 = fpext float %div4095 to double
  %call4097 = call i32 (i8*, i32, i64, i8*, ...)* @__sprintf_chk(i8* %arraydecay4078, i32 0, i64 20, i8* getelementptr inbounds ([6 x i8]* @.str15, i64 0, i64 0), double %conv4096) nounwind
  br i1 %cmp1733, label %if.then4107, label %if.else4114

if.then4107:                                      ; preds = %if.then4073
  unreachable

if.else4114:                                      ; preds = %if.then4073
  unreachable
}

declare i32 @__sprintf_chk(i8*, i32, i64, i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35}

!0 = !{!"0x11\0012\00clang version 3.3 (trunk 168918) (llvm/trunk 168920)\001\00\000\00\000", !19, !2, !2, !20, !2, null} ; [ DW_TAG_compile_unit ] [MultiSource/Benchmarks/MiBench/consumer-typeset/MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c] [DW_LANG_C99]
!1 = !{!2}
!2 = !{}
!4 = !{!"0x100\00num1\00815\000", !5, !14, !15} ; [ DW_TAG_auto_variable ] [num1] [line 815]
!5 = !{!"0xb\00815\000\00177", !14, !6} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!6 = !{!"0xb\00812\000\00176", !14, !7} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!7 = !{!"0xb\00807\000\00175", !14, !8} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!8 = !{!"0xb\00440\000\0094", !14, !9} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!9 = !{!"0xb\00435\000\0091", !14, !10} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!10 = !{!"0xb\00434\000\0090", !14, !11} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!11 = !{!"0xb\00250\000\0024", !14, !12} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!12 = !{!"0xb\00249\000\0023", !14, !13} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!13 = !{!"0xb\00221\000\0019", !14, !2} ; [ DW_TAG_lexical_block ] [MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c]
!14 = !{!"0x29", !19} ; [ DW_TAG_file_type ]
!15 = !{!"0x1\00\000\00160\008\000\000", null, null, !16, !17, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 160, align 8, offset 0] [from char]
!16 = !{!"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!17 = !{!18}
!18 = !{!"0x21\000\0020"}       ; [ DW_TAG_subrange_type ] [0, 19]
!19 = !{!"MultiSource/Benchmarks/MiBench/consumer-typeset/z19.c", !"MultiSource/Benchmarks/MiBench/consumer-typeset"}

!20 = !{!21}
!21 = !{!"0x2e\00AttachGalley\00AttachGalley\00\000\000\001\000\006\00256\001\001", !19, !14, !22, null, i32 (%union.rec**)* @AttachGalley, null, null, null} ; [ DW_TAG_subprogram ] [def] [AttachGalley]
!22 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = !{null}

; Test DebugValue uses visited by RegisterPressureTracker findUseBetween().
;
; CHECK: @main
; CHECK: DEBUG_VALUE: X
; CHECK: call

%"class.__gnu_cxx::hash_map" = type { %"class.__gnu_cxx::hashtable" }
%"class.__gnu_cxx::hashtable" = type { i64, i64, i64, i64, i64, i64 }

define void @main() uwtable ssp {
entry:
  %X = alloca %"class.__gnu_cxx::hash_map", align 8
  br i1 undef, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  unreachable

cond.end:                                         ; preds = %entry
  call void @llvm.dbg.declare(metadata %"class.__gnu_cxx::hash_map"* %X, metadata !31, metadata !{!"0x102"})
  %_M_num_elements.i.i.i.i = getelementptr inbounds %"class.__gnu_cxx::hash_map", %"class.__gnu_cxx::hash_map"* %X, i64 0, i32 0, i32 5
  invoke void @_Znwm()
          to label %exit.i unwind label %lpad2.i.i.i.i

exit.i:                                           ; preds = %cond.end
  unreachable

lpad2.i.i.i.i:                                    ; preds = %cond.end
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  br i1 undef, label %lpad.body.i.i, label %if.then.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %lpad2.i.i.i.i
  unreachable

lpad.body.i.i:                                    ; preds = %lpad2.i.i.i.i
  resume { i8*, i32 } %0
}

declare i32 @__gxx_personality_v0(...)

declare void @_Znwm()

!llvm.dbg.cu = !{!30}

!30 = !{!"0x11\004\00clang version 3.3 (trunk 169129) (llvm/trunk 169135)\001\00\000\00\000", !34, !2, !2, !36, null, null} ; [ DW_TAG_compile_unit ] [SingleSource/Benchmarks/Shootout-C++/hash.cpp] [DW_LANG_C_plus_plus]
!31 = !{!"0x100\00X\0029\000", null, null, !32} ; [ DW_TAG_auto_variable ] [X] [line 29]
!32 = !{!"0x16\00HM\0028\000\000\000\000", !34, null, null} ; [ DW_TAG_typedef ] [HM] [line 28, size 0, align 0, offset 0] [from ]
!33 = !{!"0x29", !34} ; [ DW_TAG_file_type ]
!34 = !{!"SingleSource/Benchmarks/Shootout-C++/hash.cpp", !"SingleSource/Benchmarks/Shootout-C++"}
!35 = !{i32 1, !"Debug Info Version", i32 2}
!36 = !{!37}
!37 = !{!"0x2e\00main\00main\00\000\000\001\000\006\00256\001\001", !19, !14, !22, null, void ()* @main, null, null, null} ; [ DW_TAG_subprogram ] [def] [main]
