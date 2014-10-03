; REQUIRES: object-emission

; RUN: llc -mtriple x86_64-pc-linux -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Testcase from:
; struct base {
;  virtual ~base();
; };
; struct foo {
;  base b;
; };
; foo f;

; Where member b should be seen as a field at an offset and not a bitfield.

; CHECK: DW_TAG_member
; CHECK: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_AT_bit_offset

%struct.foo = type { %struct.base }
%struct.base = type { i32 (...)** }
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
  call void @llvm.dbg.declare(metadata !{%struct.foo** %this.addr}, metadata !36, metadata !{metadata !"0x102"}), !dbg !38
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
  call void @llvm.dbg.declare(metadata !{%struct.foo** %this.addr}, metadata !40, metadata !{metadata !"0x102"}), !dbg !41
  %this1 = load %struct.foo** %this.addr
  %b = getelementptr inbounds %struct.foo* %this1, i32 0, i32 0, !dbg !42
  call void @_ZN4baseD1Ev(%struct.base* %b), !dbg !42
  ret void, !dbg !44
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

declare void @_ZN4baseD1Ev(%struct.base*) #4

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZN4baseC2Ev(%struct.base* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.base*, align 8
  store %struct.base* %this, %struct.base** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.base** %this.addr}, metadata !45, metadata !{metadata !"0x102"}), !dbg !47
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

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 (trunk 203673) (llvm/trunk 203681)\000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !8, metadata !30, metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/foo.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cc", metadata !"/usr/local/google/home/echristo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !7}
!4 = metadata !{metadata !"0x13\00foo\005\0064\0064\000\000\000", metadata !1, null, null, metadata !5, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 5, size 64, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{metadata !"0xd\00b\006\0064\0064\000\000", metadata !1, metadata !"_ZTS3foo", metadata !"_ZTS4base"} ; [ DW_TAG_member ] [b] [line 6, size 64, align 64, offset 0] [from _ZTS4base]
!7 = metadata !{metadata !"0x13\00base\001\000\000\000\004\000", metadata !1, null, null, null, null, null, metadata !"_ZTS4base"} ; [ DW_TAG_structure_type ] [base] [line 1, size 0, align 0, offset 0] [decl] [from ]
!8 = metadata !{metadata !9, metadata !13, metadata !19, metadata !22, metadata !28}
!9 = metadata !{metadata !"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\009\001\001\000\006\00256\000\009", metadata !1, metadata !10, metadata !11, null, void ()* @__cxx_global_var_init, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 9] [local] [def] [__cxx_global_var_init]
!10 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/foo.cc]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null}
!13 = metadata !{metadata !"0x2e\00~foo\00~foo\00_ZN3fooD2Ev\005\000\001\000\006\00320\000\005", metadata !1, metadata !"_ZTS3foo", metadata !14, null, void (%struct.foo*)* @_ZN3fooD2Ev, null, metadata !17, metadata !2} ; [ DW_TAG_subprogram ] [line 5] [def] [~foo]
!14 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{null, metadata !16}
!16 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
!17 = metadata !{metadata !"0x2e\00~foo\00~foo\00\000\000\000\000\006\00320\000\000", null, metadata !"_ZTS3foo", metadata !14, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ] [line 0] [~foo]
!18 = metadata !{i32 786468}
!19 = metadata !{metadata !"0x2e\00foo\00foo\00_ZN3fooC2Ev\005\000\001\000\006\00320\000\005", metadata !1, metadata !"_ZTS3foo", metadata !14, null, void (%struct.foo*)* @_ZN3fooC2Ev, null, metadata !20, metadata !2} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!20 = metadata !{metadata !"0x2e\00foo\00foo\00\000\000\000\000\006\00320\000\000", null, metadata !"_ZTS3foo", metadata !14, null, null, null, i32 0, metadata !21} ; [ DW_TAG_subprogram ] [line 0] [foo]
!21 = metadata !{i32 786468}
!22 = metadata !{metadata !"0x2e\00base\00base\00_ZN4baseC2Ev\001\000\001\000\006\00320\000\001", metadata !1, metadata !"_ZTS4base", metadata !23, null, void (%struct.base*)* @_ZN4baseC2Ev, null, metadata !26, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [base]
!23 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !24, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{null, metadata !25}
!25 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS4base]
!26 = metadata !{metadata !"0x2e\00base\00base\00\000\000\000\000\006\00320\000\000", null, metadata !"_ZTS4base", metadata !23, null, null, null, i32 0, metadata !27} ; [ DW_TAG_subprogram ] [line 0] [base]
!27 = metadata !{i32 786468}
!28 = metadata !{metadata !"0x2e\00\00\00_GLOBAL__I_a\001\001\001\000\006\0064\000\001", metadata !1, metadata !10, metadata !29, null, void ()* @_GLOBAL__I_a, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [local] [def]
!29 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{metadata !31}
!31 = metadata !{metadata !"0x34\00f\00f\00\009\000\001", null, metadata !10, metadata !4, %struct.foo* @f, null} ; [ DW_TAG_variable ] [f] [line 9] [def]
!32 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!34 = metadata !{metadata !"clang version 3.5.0 (trunk 203673) (llvm/trunk 203681)"}
!35 = metadata !{i32 9, i32 0, metadata !9, null}
!36 = metadata !{metadata !"0x101\00this\0016777216\001088", metadata !19, null, metadata !37} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3foo]
!38 = metadata !{i32 0, i32 0, metadata !19, null}
!39 = metadata !{i32 5, i32 0, metadata !19, null}
!40 = metadata !{metadata !"0x101\00this\0016777216\001088", metadata !13, null, metadata !37} ; [ DW_TAG_arg_variable ] [this] [line 0]
!41 = metadata !{i32 0, i32 0, metadata !13, null}
!42 = metadata !{i32 5, i32 0, metadata !43, null}
!43 = metadata !{metadata !"0xb\005\000\000", metadata !1, metadata !13} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/foo.cc]
!44 = metadata !{i32 5, i32 0, metadata !13, null}
!45 = metadata !{metadata !"0x101\00this\0016777216\001088", metadata !22, null, metadata !46} ; [ DW_TAG_arg_variable ] [this] [line 0]
!46 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !"_ZTS4base"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS4base]
!47 = metadata !{i32 0, i32 0, metadata !22, null}
!48 = metadata !{i32 1, i32 0, metadata !22, null}
!49 = metadata !{i32 1, i32 0, metadata !28, null}
