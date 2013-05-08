; RUN: llc -O0 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Check that the linkage name is present for constructors/destructors.

; CHECK: DW_AT_MIPS_linkage_name {{.*}}"_ZN3FooD1Ev"
; CHECK: DW_AT_MIPS_linkage_name {{.*}}"_ZN3FooC1Ev"
; END

; $ cat test.cpp
; class Foo {
; public:
;   Foo() {}
;   ~Foo() {}
; };
;
; void bar()
; {
;     Foo F;
; }

; $ clang++ -g -O0 -c test.cpp -S -emit-llvm -o test.ll
; $ cat test.ll

; ModuleID = 'test.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Foo = type { i8 }

; Function Attrs: 
define void @_Z3barv() #0 {
entry:
  %F = alloca %class.Foo, align 1
  call void @llvm.dbg.declare(metadata !{%class.Foo* %F}, metadata !21), !dbg !22
  call void @_ZN3FooC1Ev(%class.Foo* %F), !dbg !22
  call void @_ZN3FooD1Ev(%class.Foo* %F), !dbg !23
  ret void, !dbg !23
}

declare void @llvm.dbg.declare(metadata, metadata) #1

define linkonce_odr void @_ZN3FooC1Ev(%class.Foo* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.Foo*, align 8
  store %class.Foo* %this, %class.Foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.Foo** %this.addr}, metadata !24), !dbg !26
  %this1 = load %class.Foo** %this.addr
  call void @_ZN3FooC2Ev(%class.Foo* %this1), !dbg !26
  ret void, !dbg !26
}

define linkonce_odr void @_ZN3FooD1Ev(%class.Foo* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.Foo*, align 8
  store %class.Foo* %this, %class.Foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.Foo** %this.addr}, metadata !27), !dbg !28
  %this1 = load %class.Foo** %this.addr
  call void @_ZN3FooD2Ev(%class.Foo* %this1), !dbg !28
  ret void, !dbg !28
}

define linkonce_odr void @_ZN3FooD2Ev(%class.Foo* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.Foo*, align 8
  store %class.Foo* %this, %class.Foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.Foo** %this.addr}, metadata !29), !dbg !30
  %this1 = load %class.Foo** %this.addr
  ret void, !dbg !30
}

define linkonce_odr void @_ZN3FooC2Ev(%class.Foo* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.Foo*, align 8
  store %class.Foo* %this, %class.Foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.Foo** %this.addr}, metadata !31), !dbg !32
  %this1 = load %class.Foo** %this.addr
  ret void, !dbg !32
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/test.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"test.cpp", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8, metadata !18, metadata !19, metadata !20}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"_Z3barv", i32 7, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3barv, null, null, metadata !2, i32 8} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/test.cpp]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, null, metadata !"~Foo", metadata !"~Foo", metadata !"_ZN3FooD1Ev", i32 4, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.Foo*)* @_ZN3FooD1Ev, null, metadata !16, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [~Foo]
!9 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Foo]
!12 = metadata !{i32 786434, metadata !1, null, metadata !"Foo", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !13, i32 0, null, null} ; [ DW_TAG_class_type ] [Foo] [line 1, size 8, align 8, offset 0] [from ]
!13 = metadata !{metadata !14, metadata !16}
!14 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"Foo", metadata !"Foo", metadata !"", i32 3, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !15, i32 3} ; [ DW_TAG_subprogram ] [line 3] [Foo]
!15 = metadata !{i32 786468}
!16 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"~Foo", metadata !"~Foo", metadata !"", i32 4, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !17, i32 4} ; [ DW_TAG_subprogram ] [line 4] [~Foo]
!17 = metadata !{i32 786468}
!18 = metadata !{i32 786478, metadata !1, null, metadata !"~Foo", metadata !"~Foo", metadata !"_ZN3FooD2Ev", i32 4, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.Foo*)* @_ZN3FooD2Ev, null, metadata !16, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [~Foo]
!19 = metadata !{i32 786478, metadata !1, null, metadata !"Foo", metadata !"Foo", metadata !"_ZN3FooC1Ev", i32 3, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.Foo*)* @_ZN3FooC1Ev, null, metadata !14, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [Foo]
!20 = metadata !{i32 786478, metadata !1, null, metadata !"Foo", metadata !"Foo", metadata !"_ZN3FooC2Ev", i32 3, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.Foo*)* @_ZN3FooC2Ev, null, metadata !14, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [Foo]
!21 = metadata !{i32 786688, metadata !4, metadata !"F", metadata !5, i32 9, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [F] [line 9]
!22 = metadata !{i32 9, i32 0, metadata !4, null}
!23 = metadata !{i32 10, i32 0, metadata !4, null}
!24 = metadata !{i32 786689, metadata !19, metadata !"this", metadata !5, i32 16777219, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 3]
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from Foo]
!26 = metadata !{i32 3, i32 0, metadata !19, null}
!27 = metadata !{i32 786689, metadata !8, metadata !"this", metadata !5, i32 16777220, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 4]
!28 = metadata !{i32 4, i32 0, metadata !8, null}
!29 = metadata !{i32 786689, metadata !18, metadata !"this", metadata !5, i32 16777220, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 4]
!30 = metadata !{i32 4, i32 0, metadata !18, null}
!31 = metadata !{i32 786689, metadata !20, metadata !"this", metadata !5, i32 16777219, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 3]
!32 = metadata !{i32 3, i32 0, metadata !20, null}
