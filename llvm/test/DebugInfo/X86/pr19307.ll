; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Generated from the source file pr19307.cc:
; #include <string>
; void parse_range(unsigned long long &offset, unsigned long long &limit,
;                  std::string range) {
;   if (range.compare(0, 6, "items=") != 0 || range[6] == '-')
;     offset = 1;
;   range.erase(0, 6);
;   limit = 2;
; }
; with "clang++ -S -emit-llvm -O0 -g pr19307.cc"

; Location of "range" string is spilled from %rdx to stack and is
; addressed via %rbp.
; CHECK: movq %rdx, {{[-0-9]+}}(%rbp)
; CHECK-NEXT: [[START_LABEL:.Ltmp[0-9]+]]
; This location should be valid until the end of the function.

; Verify that we have proper range in debug_loc section:
; CHECK: .Ldebug_loc{{[0-9]+}}:
; CHECK: DW_OP_breg1
; CHECK:      .quad [[START_LABEL]]-.Lfunc_begin0
; CHECK-NEXT: .quad .Lfunc_end0-.Lfunc_begin0
; CHECK: DW_OP_breg6
; CHECK: DW_OP_deref

; ModuleID = 'pr19307.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }

@.str = private unnamed_addr constant [7 x i8] c"items=\00", align 1

; Function Attrs: uwtable
define void @_Z11parse_rangeRyS_Ss(i64* %offset, i64* %limit, %"class.std::basic_string"* %range) #0 {
entry:
  %offset.addr = alloca i64*, align 8
  %limit.addr = alloca i64*, align 8
  store i64* %offset, i64** %offset.addr, align 8
  call void @llvm.dbg.declare(metadata !{i64** %offset.addr}, metadata !45, metadata !{metadata !"0x102"}), !dbg !46
  store i64* %limit, i64** %limit.addr, align 8
  call void @llvm.dbg.declare(metadata !{i64** %limit.addr}, metadata !47, metadata !{metadata !"0x102"}), !dbg !46
  call void @llvm.dbg.declare(metadata !{%"class.std::basic_string"* %range}, metadata !48, metadata !{metadata !"0x102"}), !dbg !49
  %call = call i32 @_ZNKSs7compareEmmPKc(%"class.std::basic_string"* %range, i64 0, i64 6, i8* getelementptr inbounds ([7 x i8]* @.str, i32 0, i32 0)), !dbg !50
  %cmp = icmp ne i32 %call, 0, !dbg !50
  br i1 %cmp, label %if.then, label %lor.lhs.false, !dbg !50

lor.lhs.false:                                    ; preds = %entry
  %call1 = call i8* @_ZNSsixEm(%"class.std::basic_string"* %range, i64 6), !dbg !52
  %0 = load i8* %call1, !dbg !52
  %conv = sext i8 %0 to i32, !dbg !52
  %cmp2 = icmp eq i32 %conv, 45, !dbg !52
  br i1 %cmp2, label %if.then, label %if.end, !dbg !52

if.then:                                          ; preds = %lor.lhs.false, %entry
  %1 = load i64** %offset.addr, align 8, !dbg !54
  store i64 1, i64* %1, align 8, !dbg !54
  br label %if.end, !dbg !54

if.end:                                           ; preds = %if.then, %lor.lhs.false
  %call3 = call %"class.std::basic_string"* @_ZNSs5eraseEmm(%"class.std::basic_string"* %range, i64 0, i64 6), !dbg !55
  %2 = load i64** %limit.addr, align 8, !dbg !56
  store i64 2, i64* %2, align 8, !dbg !56
  ret void, !dbg !57
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @_ZNKSs7compareEmmPKc(%"class.std::basic_string"*, i64, i64, i8*) #2

declare i8* @_ZNSsixEm(%"class.std::basic_string"*, i64) #2

declare %"class.std::basic_string"* @_ZNSs5eraseEmm(%"class.std::basic_string"*, i64, i64) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!42, !43}
!llvm.ident = !{!44}

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 (209308)\000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !12, metadata !2, metadata !21} ; [ DW_TAG_compile_unit ] [/llvm_cmake_gcc/pr19307.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"pr19307.cc", metadata !"/llvm_cmake_gcc"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !6, metadata !8}
!4 = metadata !{metadata !"0x13\00\0083\000\000\000\004\000", metadata !5, null, null, null, null, null, metadata !"_ZTS11__mbstate_t"} ; [ DW_TAG_structure_type ] [line 83, size 0, align 0, offset 0] [decl] [from ]
!5 = metadata !{metadata !"/usr/include/wchar.h", metadata !"/llvm_cmake_gcc"}
!6 = metadata !{metadata !"0x13\00lconv\0054\000\000\000\004\000", metadata !7, null, null, null, null, null, metadata !"_ZTS5lconv"} ; [ DW_TAG_structure_type ] [lconv] [line 54, size 0, align 0, offset 0] [decl] [from ]
!7 = metadata !{metadata !"/usr/include/locale.h", metadata !"/llvm_cmake_gcc"}
!8 = metadata !{metadata !"0x2\00basic_string<char, std::char_traits<char>, std::allocator<char> >\001134\000\000\000\004\000", metadata !9, metadata !10, null, null, null, null, metadata !"_ZTSSs"} ; [ DW_TAG_class_type ] [basic_string<char, std::char_traits<char>, std::allocator<char> >] [line 1134, size 0, align 0, offset 0] [decl] [from ]
!9 = metadata !{metadata !"/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/bits/basic_string.tcc", metadata !"/llvm_cmake_gcc"}
!10 = metadata !{metadata !"0x39\00std\00153", metadata !11, null} ; [ DW_TAG_namespace ] [std] [line 153]
!11 = metadata !{metadata !"/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/x86_64-linux-gnu/bits/c++config.h", metadata !"/llvm_cmake_gcc"}
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x2e\00parse_range\00parse_range\00_Z11parse_rangeRyS_Ss\003\000\001\000\006\00256\000\004", metadata !1, metadata !14, metadata !15, null, void (i64*, i64*, %"class.std::basic_string"*)* @_Z11parse_rangeRyS_Ss, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [parse_range]
!14 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/llvm_cmake_gcc/pr19307.cc]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17, metadata !17, metadata !19}
!17 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !18} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from long long unsigned int]
!18 = metadata !{metadata !"0x24\00long long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!19 = metadata !{metadata !"0x16\00string\0065\000\000\000\000", metadata !20, metadata !10, metadata !"_ZTSSs"} ; [ DW_TAG_typedef ] [string] [line 65, size 0, align 0, offset 0] [from _ZTSSs]
!20 = metadata !{metadata !"/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/bits/stringfwd.h", metadata !"/llvm_cmake_gcc"}
!21 = metadata !{metadata !22, metadata !26, metadata !29, metadata !33, metadata !38, metadata !41}
!22 = metadata !{metadata !"0x3a\0057\00", metadata !23, metadata !25} ; [ DW_TAG_imported_module ]
!23 = metadata !{metadata !"0x39\00__gnu_debug\0055", metadata !24, null} ; [ DW_TAG_namespace ] [__gnu_debug] [line 55]
!24 = metadata !{metadata !"/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/debug/debug.h", metadata !"/llvm_cmake_gcc"}
!25 = metadata !{metadata !"0x39\00__debug\0049", metadata !24, metadata !10} ; [ DW_TAG_namespace ] [__debug] [line 49]
!26 = metadata !{metadata !"0x8\0066\00", metadata !10, metadata !27} ; [ DW_TAG_imported_declaration ]
!27 = metadata !{metadata !"0x16\00mbstate_t\00106\000\000\000\000", metadata !5, null, metadata !28} ; [ DW_TAG_typedef ] [mbstate_t] [line 106, size 0, align 0, offset 0] [from __mbstate_t]
!28 = metadata !{metadata !"0x16\00__mbstate_t\0095\000\000\000\000", metadata !5, null, metadata !"_ZTS11__mbstate_t"} ; [ DW_TAG_typedef ] [__mbstate_t] [line 95, size 0, align 0, offset 0] [from _ZTS11__mbstate_t]
!29 = metadata !{metadata !"0x8\00141\00", metadata !10, metadata !30} ; [ DW_TAG_imported_declaration ]
!30 = metadata !{metadata !"0x16\00wint_t\00141\000\000\000\000", metadata !31, null, metadata !32} ; [ DW_TAG_typedef ] [wint_t] [line 141, size 0, align 0, offset 0] [from unsigned int]
!31 = metadata !{metadata !"/llvm_cmake_gcc/bin/../lib/clang/3.5.0/include/stddef.h", metadata !"/llvm_cmake_gcc"}
!32 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!33 = metadata !{metadata !"0x8\0042\00", metadata !34, metadata !36} ; [ DW_TAG_imported_declaration ]
!34 = metadata !{metadata !"0x39\00__gnu_cxx\0069", metadata !35, null} ; [ DW_TAG_namespace ] [__gnu_cxx] [line 69]
!35 = metadata !{metadata !"/usr/lib/gcc/x86_64-linux-gnu/4.6/../../../../include/c++/4.6/bits/cpp_type_traits.h", metadata !"/llvm_cmake_gcc"}
!36 = metadata !{metadata !"0x16\00size_t\00155\000\000\000\000", metadata !11, metadata !10, metadata !37} ; [ DW_TAG_typedef ] [size_t] [line 155, size 0, align 0, offset 0] [from long unsigned int]
!37 = metadata !{metadata !"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!38 = metadata !{metadata !"0x8\0043\00", metadata !34, metadata !39} ; [ DW_TAG_imported_declaration ]
!39 = metadata !{metadata !"0x16\00ptrdiff_t\00156\000\000\000\000", metadata !11, metadata !10, metadata !40} ; [ DW_TAG_typedef ] [ptrdiff_t] [line 156, size 0, align 0, offset 0] [from long int]
!40 = metadata !{metadata !"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!41 = metadata !{metadata !"0x8\0055\00", metadata !10, metadata !"_ZTS5lconv"} ; [ DW_TAG_imported_declaration ]
!42 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!43 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!44 = metadata !{metadata !"clang version 3.5.0 (209308)"}
!45 = metadata !{metadata !"0x101\00offset\0016777219\000", metadata !13, metadata !14, metadata !17} ; [ DW_TAG_arg_variable ] [offset] [line 3]
!46 = metadata !{i32 3, i32 0, metadata !13, null}
!47 = metadata !{metadata !"0x101\00limit\0033554435\000", metadata !13, metadata !14, metadata !17} ; [ DW_TAG_arg_variable ] [limit] [line 3]
!48 = metadata !{metadata !"0x101\00range\0050331652\008192", metadata !13, metadata !14, metadata !19} ; [ DW_TAG_arg_variable ] [range] [line 4]
!49 = metadata !{i32 4, i32 0, metadata !13, null}
!50 = metadata !{i32 5, i32 0, metadata !51, null}
!51 = metadata !{metadata !"0xb\005\000\000", metadata !1, metadata !13} ; [ DW_TAG_lexical_block ] [/llvm_cmake_gcc/pr19307.cc]
!52 = metadata !{i32 5, i32 0, metadata !53, null}
!53 = metadata !{metadata !"0xb\005\000\001", metadata !1, metadata !51} ; [ DW_TAG_lexical_block ] [/llvm_cmake_gcc/pr19307.cc]
!54 = metadata !{i32 6, i32 0, metadata !51, null}
!55 = metadata !{i32 7, i32 0, metadata !13, null}
!56 = metadata !{i32 8, i32 0, metadata !13, null}
!57 = metadata !{i32 9, i32 0, metadata !13, null}

