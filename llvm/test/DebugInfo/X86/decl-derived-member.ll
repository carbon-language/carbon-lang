; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; XFAIL: *

; Testcase from:
; struct base {
;  virtual ~base();
; };
; struct foo {
;  base b;
; };
; foo f;

; Where member b should be seen as a field at an offset and not a bitfield.

; CHECK: DW_AT_member
; CHECK: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_AT_bit_offset

@f = global %struct.foo zeroinitializer, align 8
@__dso_handle = external global i8
@_ZTV4base = external unnamed_addr constant [4 x i8*]
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  call void @_ZN3fooC2Ev(%struct.foo* @f) #2, !dbg !35
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.foo*)* @_ZN3fooD2Ev to void (i8*)*), i8* bitcast (%struct.foo* @f to i8*), i8* @__dso_handle) #2, !dbg !35
  ret void, !dbg !35
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN3fooC2Ev(%struct.foo* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.foo** %this.addr}, metadata !36), !dbg !38
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !39
  call void @_ZN4baseC2Ev(%struct.base* %b) #2, !dbg !39
  ret void, !dbg !39
}

; Function Attrs: inlinehint uwtable
define linkonce_odr void @_ZN3fooD2Ev(%struct.foo* %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca %struct.foo*, align 8
  store %struct.foo* %this, %struct.foo** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.foo** %this.addr}, metadata !40), !dbg !41
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !42
  call void @_ZN4baseD1Ev(%struct.base* %b), !dbg !42
  ret void, !dbg !44
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #3

declare void @_ZN4baseD1Ev(%struct.base*) #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN4baseC2Ev(%struct.base* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.base*, align 8
  store %struct.base* %this, %struct.base** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.base** %this.addr}, metadata !45), !dbg !47
  %this1 = load %struct.base** %this.addr
  %0 = bitcast %struct.base* %this1 to i8***, !dbg !48
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV4base, i64 0, i64 2), i8*** %0, !dbg !48
  ret void, !dbg !48
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !49
  ret void, !dbg !49
}

attributes #0 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!32, !33}
!llvm.ident = !{!34}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 203673) (llvm/trunk 203681)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !8, metadata !30, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/foo.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cc", metadata !"/usr/local/google/home/echristo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !7}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"foo", i32 5, i64 64, i64 64, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 5, size 64, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS3foo", metadata !"b", i32 6, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS4base"} ; [ DW_TAG_member ] [b] [line 6, size 64, align 64, offset 0] [from _ZTS4base]
!7 = metadata !{i32 786451, metadata !1, null, metadata !"base", i32 1, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, metadata !"_ZTS4base"} ; [ DW_TAG_structure_type ] [base] [line 1, size 0, align 0, offset 0] [decl] [from ]
!8 = metadata !{metadata !9, metadata !13, metadata !19, metadata !22, metadata !28}
!9 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 9, metadata !11, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [local] [def] [__cxx_global_var_init]
!10 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/foo.cc]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null}
!13 = metadata !{i32 786478, metadata !1, metadata !"_ZTS3foo", metadata !"~foo", metadata !"~foo", metadata !"_ZN3fooD2Ev", i32 5, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 320, i1 false, void (%struct.foo*)* @_ZN3fooD2Ev, null, metadata !17, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [~foo]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{null, metadata !16}
!16 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
!17 = metadata !{i32 786478, null, metadata !"_ZTS3foo", metadata !"~foo", metadata !"~foo", metadata !"", i32 0, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !18, i32 0} ; [ DW_TAG_subprogram ] [line 0] [~foo]
!18 = metadata !{i32 786468}
!19 = metadata !{i32 786478, metadata !1, metadata !"_ZTS3foo", metadata !"foo", metadata !"foo", metadata !"_ZN3fooC2Ev", i32 5, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 320, i1 false, void (%struct.foo*)* @_ZN3fooC2Ev, null, metadata !20, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!20 = metadata !{i32 786478, null, metadata !"_ZTS3foo", metadata !"foo", metadata !"foo", metadata !"", i32 0, metadata !14, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !21, i32 0} ; [ DW_TAG_subprogram ] [line 0] [foo]
!21 = metadata !{i32 786468}
!22 = metadata !{i32 786478, metadata !1, metadata !"_ZTS4base", metadata !"base", metadata !"base", metadata !"_ZN4baseC2Ev", i32 1, metadata !23, i1 false, i1 true, i32 0, i32 0, null, i32 320, i1 false, void (%struct.base*)* @_ZN4baseC2Ev, null, metadata !26, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [base]
!23 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !24, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{null, metadata !25}
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS4base]
!26 = metadata !{i32 786478, null, metadata !"_ZTS4base", metadata !"base", metadata !"base", metadata !"", i32 0, metadata !23, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !27, i32 0} ; [ DW_TAG_subprogram ] [line 0] [base]
!27 = metadata !{i32 786468}
!28 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"", metadata !"", metadata !"_GLOBAL__I_a", i32 1, metadata !29, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__I_a, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [local] [def]
!29 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{metadata !31}
!31 = metadata !{i32 786484, i32 0, null, metadata !"f", metadata !"f", metadata !"", metadata !10, i32 9, metadata !4, i32 0, i32 1, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 9] [def]
!32 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!34 = metadata !{metadata !"clang version 3.5.0 (trunk 203673) (llvm/trunk 203681)"}
!35 = metadata !{i32 9, i32 0, metadata !9, null}
!36 = metadata !{i32 786689, metadata !19, metadata !"this", null, i32 16777216, metadata !37, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3foo]
!38 = metadata !{i32 0, i32 0, metadata !19, null}
!39 = metadata !{i32 5, i32 0, metadata !19, null}
!40 = metadata !{i32 786689, metadata !13, metadata !"this", null, i32 16777216, metadata !37, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!41 = metadata !{i32 0, i32 0, metadata !13, null}
!42 = metadata !{i32 5, i32 0, metadata !43, null}
!43 = metadata !{i32 786443, metadata !1, metadata !13, i32 5, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/foo.cc]
!44 = metadata !{i32 5, i32 0, metadata !13, null}
!45 = metadata !{i32 786689, metadata !22, metadata !"this", null, i32 16777216, metadata !46, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!46 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS4base]
!47 = metadata !{i32 0, i32 0, metadata !22, null}
!48 = metadata !{i32 1, i32 0, metadata !22, null}
!49 = metadata !{i32 1, i32 0, metadata !28, null}
