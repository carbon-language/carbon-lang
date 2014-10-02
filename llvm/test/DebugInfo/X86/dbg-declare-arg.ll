; RUN: llc -O0 -fast-isel=false < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"
;Radar 9321650

;CHECK: ##DEBUG_VALUE: my_a 

%class.A = type { i32, i32, i32, i32 }

define void @_Z3fooi(%class.A* sret %agg.result, i32 %i) ssp {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  %nrvo = alloca i1
  %cleanup.dest.slot = alloca i32
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !26, metadata !{metadata !"0x102"}), !dbg !27
  call void @llvm.dbg.declare(metadata !{i32* %j}, metadata !28, metadata !{metadata !"0x102"}), !dbg !30
  store i32 0, i32* %j, align 4, !dbg !31
  %tmp = load i32* %i.addr, align 4, !dbg !32
  %cmp = icmp eq i32 %tmp, 42, !dbg !32
  br i1 %cmp, label %if.then, label %if.end, !dbg !32

if.then:                                          ; preds = %entry
  %tmp1 = load i32* %i.addr, align 4, !dbg !33
  %add = add nsw i32 %tmp1, 1, !dbg !33
  store i32 %add, i32* %j, align 4, !dbg !33
  br label %if.end, !dbg !35

if.end:                                           ; preds = %if.then, %entry
  store i1 false, i1* %nrvo, !dbg !36
  call void @llvm.dbg.declare(metadata !{%class.A* %agg.result}, metadata !37, metadata !{metadata !"0x102"}), !dbg !39
  %tmp2 = load i32* %j, align 4, !dbg !40
  %x = getelementptr inbounds %class.A* %agg.result, i32 0, i32 0, !dbg !40
  store i32 %tmp2, i32* %x, align 4, !dbg !40
  store i1 true, i1* %nrvo, !dbg !41
  store i32 1, i32* %cleanup.dest.slot
  %nrvo.val = load i1* %nrvo, !dbg !42
  br i1 %nrvo.val, label %nrvo.skipdtor, label %nrvo.unused, !dbg !42

nrvo.unused:                                      ; preds = %if.end
  call void @_ZN1AD1Ev(%class.A* %agg.result), !dbg !42
  br label %nrvo.skipdtor, !dbg !42

nrvo.skipdtor:                                    ; preds = %nrvo.unused, %if.end
  ret void, !dbg !42
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AD1Ev(%class.A* %this) unnamed_addr ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !43, metadata !{metadata !"0x102"}), !dbg !44
  %this1 = load %class.A** %this.addr
  call void @_ZN1AD2Ev(%class.A* %this1)
  ret void, !dbg !45
}

define linkonce_odr void @_ZN1AD2Ev(%class.A* %this) unnamed_addr nounwind ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !46, metadata !{metadata !"0x102"}), !dbg !47
  %this1 = load %class.A** %this.addr
  %x = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !48
  store i32 1, i32* %x, align 4, !dbg !48
  ret void, !dbg !48
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!52}

!0 = metadata !{metadata !"0x2e\00~A\00~A\00\002\000\000\000\006\00256\000\000", metadata !51, metadata !1, metadata !11, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x2\00A\002\00128\0032\000\000\000", metadata !51, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_class_type ] [A] [line 2, size 128, align 32, offset 0] [def] [from ]
!2 = metadata !{metadata !"0x11\004\00clang version 3.0 (trunk 130127)\000\00\000\00\001", metadata !51, metadata !24, metadata !24, metadata !50, null, null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x29", metadata !51} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !5, metadata !7, metadata !8, metadata !9, metadata !0, metadata !10, metadata !14}
!5 = metadata !{metadata !"0xd\00x\002\0032\0032\000\000", metadata !51, metadata !3, metadata !6} ; [ DW_TAG_member ]
!6 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0xd\00y\002\0032\0032\0032\000", metadata !51, metadata !3, metadata !6} ; [ DW_TAG_member ]
!8 = metadata !{metadata !"0xd\00z\002\0032\0032\0064\000", metadata !51, metadata !3, metadata !6} ; [ DW_TAG_member ]
!9 = metadata !{metadata !"0xd\00o\002\0032\0032\0096\000", metadata !51, metadata !3, metadata !6} ; [ DW_TAG_member ]
!10 = metadata !{metadata !"0x2e\00A\00A\00\002\000\000\000\006\00320\000\000", metadata !51, metadata !1, metadata !11, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !51, metadata !3, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", metadata !2, null, metadata !1} ; [ DW_TAG_pointer_type ]
!14 = metadata !{metadata !"0x2e\00A\00A\00\002\000\000\000\006\00320\000\000", metadata !51, metadata !1, metadata !15, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !51, metadata !3, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !13, metadata !17}
!17 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, metadata !2, metadata !18} ; [ DW_TAG_reference_type ]
!18 = metadata !{metadata !"0x26\00\000\000\000\000\000", metadata !2, null, metadata !1} ; [ DW_TAG_const_type ]
!19 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3fooi\004\000\001\000\006\00256\000\000", metadata !51, metadata !3, metadata !20, null, void (%class.A*, i32)* @_Z3fooi, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [foo]
!20 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !51, metadata !3, null, metadata !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !1}
!22 = metadata !{metadata !"0x2e\00~A\00~A\00_ZN1AD1Ev\002\000\001\000\006\00256\000\000", metadata !51, metadata !3, metadata !23, null, void (%class.A*)* @_ZN1AD1Ev, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [~A]
!23 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !51, metadata !3, null, metadata !24, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{null}
!25 = metadata !{metadata !"0x2e\00~A\00~A\00_ZN1AD2Ev\002\000\001\000\006\00256\000\000", metadata !51, metadata !3, metadata !23, null, void (%class.A*)* @_ZN1AD2Ev, null, null, null} ; [ DW_TAG_subprogram ] [line 2] [def] [scope 0] [~A]
!26 = metadata !{metadata !"0x101\00i\0016777220\000", metadata !19, metadata !3, metadata !6} ; [ DW_TAG_arg_variable ]
!27 = metadata !{i32 4, i32 11, metadata !19, null}
!28 = metadata !{metadata !"0x100\00j\005\000", metadata !29, metadata !3, metadata !6} ; [ DW_TAG_auto_variable ]
!29 = metadata !{metadata !"0xb\004\0014\000", metadata !51, metadata !19} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 5, i32 7, metadata !29, null}
!31 = metadata !{i32 5, i32 12, metadata !29, null}
!32 = metadata !{i32 6, i32 3, metadata !29, null}
!33 = metadata !{i32 7, i32 5, metadata !34, null}
!34 = metadata !{metadata !"0xb\006\0016\001", metadata !51, metadata !29} ; [ DW_TAG_lexical_block ]
!35 = metadata !{i32 8, i32 3, metadata !34, null}
!36 = metadata !{i32 9, i32 9, metadata !29, null}
!37 = metadata !{metadata !"0x100\00my_a\009\000", metadata !29, metadata !3, metadata !38} ; [ DW_TAG_auto_variable ]
!38 = metadata !{metadata !"0x10\00\000\000\000\000\000", metadata !2, null, metadata !1} ; [ DW_TAG_reference_type ]
!39 = metadata !{i32 9, i32 5, metadata !29, null}
!40 = metadata !{i32 10, i32 3, metadata !29, null}
!41 = metadata !{i32 11, i32 3, metadata !29, null}
!42 = metadata !{i32 12, i32 1, metadata !29, null}
!43 = metadata !{metadata !"0x101\00this\0016777218\0064", metadata !22, metadata !3, metadata !13} ; [ DW_TAG_arg_variable ]
!44 = metadata !{i32 2, i32 47, metadata !22, null}
!45 = metadata !{i32 2, i32 61, metadata !22, null}
!46 = metadata !{metadata !"0x101\00this\0016777218\0064", metadata !25, metadata !3, metadata !13} ; [ DW_TAG_arg_variable ]
!47 = metadata !{i32 2, i32 47, metadata !25, null}
!48 = metadata !{i32 2, i32 54, metadata !49, null}
!49 = metadata !{metadata !"0xb\002\0052\002", metadata !51, metadata !25} ; [ DW_TAG_lexical_block ]
!50 = metadata !{metadata !19, metadata !22, metadata !25}
!51 = metadata !{metadata !"a.cc", metadata !"/private/tmp"}
!52 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
