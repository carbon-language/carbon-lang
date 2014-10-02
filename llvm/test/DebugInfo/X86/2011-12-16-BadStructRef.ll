; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: b_ref
; CHECK-NOT: AT_bit_size

%struct.bar = type { %struct.baz, %struct.baz* }
%struct.baz = type { i32 }

define i32 @main(i32 %argc, i8** %argv) uwtable ssp {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %myBar = alloca %struct.bar, align 8
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %argc.addr}, metadata !49, metadata !{metadata !"0x102"}), !dbg !50
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8*** %argv.addr}, metadata !51, metadata !{metadata !"0x102"}), !dbg !52
  call void @llvm.dbg.declare(metadata !{%struct.bar* %myBar}, metadata !53, metadata !{metadata !"0x102"}), !dbg !55
  call void @_ZN3barC1Ei(%struct.bar* %myBar, i32 1), !dbg !56
  ret i32 0, !dbg !57
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN3barC1Ei(%struct.bar* %this, i32 %x) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.bar*, align 8
  %x.addr = alloca i32, align 4
  store %struct.bar* %this, %struct.bar** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.bar** %this.addr}, metadata !58, metadata !{metadata !"0x102"}), !dbg !59
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !60, metadata !{metadata !"0x102"}), !dbg !61
  %this1 = load %struct.bar** %this.addr
  %0 = load i32* %x.addr, align 4, !dbg !62
  call void @_ZN3barC2Ei(%struct.bar* %this1, i32 %0), !dbg !62
  ret void, !dbg !62
}

define linkonce_odr void @_ZN3barC2Ei(%struct.bar* %this, i32 %x) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.bar*, align 8
  %x.addr = alloca i32, align 4
  store %struct.bar* %this, %struct.bar** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.bar** %this.addr}, metadata !63, metadata !{metadata !"0x102"}), !dbg !64
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !65, metadata !{metadata !"0x102"}), !dbg !66
  %this1 = load %struct.bar** %this.addr
  %b = getelementptr inbounds %struct.bar* %this1, i32 0, i32 0, !dbg !67
  %0 = load i32* %x.addr, align 4, !dbg !67
  call void @_ZN3bazC1Ei(%struct.baz* %b, i32 %0), !dbg !67
  %1 = getelementptr inbounds %struct.bar* %this1, i32 0, i32 1, !dbg !67
  %b2 = getelementptr inbounds %struct.bar* %this1, i32 0, i32 0, !dbg !67
  store %struct.baz* %b2, %struct.baz** %1, align 8, !dbg !67
  ret void, !dbg !68
}

define linkonce_odr void @_ZN3bazC1Ei(%struct.baz* %this, i32 %a) unnamed_addr uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.baz*, align 8
  %a.addr = alloca i32, align 4
  store %struct.baz* %this, %struct.baz** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.baz** %this.addr}, metadata !70, metadata !{metadata !"0x102"}), !dbg !71
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !72, metadata !{metadata !"0x102"}), !dbg !73
  %this1 = load %struct.baz** %this.addr
  %0 = load i32* %a.addr, align 4, !dbg !74
  call void @_ZN3bazC2Ei(%struct.baz* %this1, i32 %0), !dbg !74
  ret void, !dbg !74
}

define linkonce_odr void @_ZN3bazC2Ei(%struct.baz* %this, i32 %a) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %struct.baz*, align 8
  %a.addr = alloca i32, align 4
  store %struct.baz* %this, %struct.baz** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.baz** %this.addr}, metadata !75, metadata !{metadata !"0x102"}), !dbg !76
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !77, metadata !{metadata !"0x102"}), !dbg !78
  %this1 = load %struct.baz** %this.addr
  %h = getelementptr inbounds %struct.baz* %this1, i32 0, i32 0, !dbg !79
  %0 = load i32* %a.addr, align 4, !dbg !79
  store i32 %0, i32* %h, align 4, !dbg !79
  ret void, !dbg !80
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83}

!0 = metadata !{metadata !"0x11\004\00clang version 3.1 (trunk 146596)\000\00\000\00\000", metadata !82, metadata !1, metadata !3, metadata !27, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5, metadata !9}
!5 = metadata !{metadata !"0x2\00bar\009\00128\0064\000\000\000", metadata !82, null, null, metadata !7, null, null, null} ; [ DW_TAG_class_type ] [bar] [line 9, size 128, align 64, offset 0] [def] [from ]
!6 = metadata !{metadata !"0x29", metadata !82} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !8, metadata !19, metadata !21}
!8 = metadata !{metadata !"0xd\00b\0011\0032\0032\000\000", metadata !82, metadata !5, metadata !9} ; [ DW_TAG_member ]
!9 = metadata !{metadata !"0x2\00baz\003\0032\0032\000\000\000", metadata !82, null, null, metadata !10, null, null, null} ; [ DW_TAG_class_type ] [baz] [line 3, size 32, align 32, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !13}
!11 = metadata !{metadata !"0xd\00h\005\0032\0032\000\000", metadata !82, metadata !9, metadata !12} ; [ DW_TAG_member ]
!12 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!13 = metadata !{metadata !"0x2e\00baz\00baz\00\006\000\000\000\006\00256\000\000", metadata !82, metadata !9, metadata !14, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!14 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{null, metadata !16, metadata !12}
!16 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !9} ; [ DW_TAG_pointer_type ]
!19 = metadata !{metadata !"0xd\00b_ref\0012\0064\0064\0064\000", metadata !82, metadata !5, metadata !20} ; [ DW_TAG_member ]
!20 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !9} ; [ DW_TAG_reference_type ]
!21 = metadata !{metadata !"0x2e\00bar\00bar\00\0013\000\000\000\006\00256\000\000", metadata !82, metadata !5, metadata !22, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!22 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{null, metadata !24, metadata !12}
!24 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !5} ; [ DW_TAG_pointer_type ]
!27 = metadata !{metadata !29, metadata !37, metadata !40, metadata !43, metadata !46}
!29 = metadata !{metadata !"0x2e\00main\00main\00\0017\000\001\000\006\00256\000\000", metadata !82, metadata !6, metadata !30, null, i32 (i32, i8**)* @main, null, null, null} ; [ DW_TAG_subprogram ] [line 17] [def] [scope 0] [main]
!30 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !31, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = metadata !{metadata !12, metadata !12, metadata !32}
!32 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !33} ; [ DW_TAG_pointer_type ]
!33 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !34} ; [ DW_TAG_pointer_type ]
!34 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!35 = metadata !{metadata !36}
!36 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!37 = metadata !{metadata !"0x2e\00bar\00bar\00_ZN3barC1Ei\0013\000\001\000\006\00256\000\000", metadata !82, null, metadata !22, null, void (%struct.bar*, i32)* @_ZN3barC1Ei, null, metadata !21, null} ; [ DW_TAG_subprogram ] [line 13] [def] [scope 0] [bar]
!38 = metadata !{metadata !39}
!39 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!40 = metadata !{metadata !"0x2e\00bar\00bar\00_ZN3barC2Ei\0013\000\001\000\006\00256\000\000", metadata !82, null, metadata !22, null, void (%struct.bar*, i32)* @_ZN3barC2Ei, null, metadata !21, null} ; [ DW_TAG_subprogram ] [line 13] [def] [scope 0] [bar]
!41 = metadata !{metadata !42}
!42 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!43 = metadata !{metadata !"0x2e\00baz\00baz\00_ZN3bazC1Ei\006\000\001\000\006\00256\000\000", metadata !82, null, metadata !14, null, void (%struct.baz*, i32)* @_ZN3bazC1Ei, null, metadata !13, null} ; [ DW_TAG_subprogram ] [line 6] [def] [scope 0] [baz]
!44 = metadata !{metadata !45}
!45 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ]
!46 = metadata !{metadata !"0x2e\00baz\00baz\00_ZN3bazC2Ei\006\000\001\000\006\00256\000\000", metadata !82, null, metadata !14, null, void (%struct.baz*, i32)* @_ZN3bazC2Ei, null, metadata !13, null} ; [ DW_TAG_subprogram ] [line 6] [def] [scope 0] [baz]
!49 = metadata !{metadata !"0x101\00argc\0016777232\000", metadata !29, metadata !6, metadata !12} ; [ DW_TAG_arg_variable ]
!50 = metadata !{i32 16, i32 14, metadata !29, null}
!51 = metadata !{metadata !"0x101\00argv\0033554448\000", metadata !29, metadata !6, metadata !32} ; [ DW_TAG_arg_variable ]
!52 = metadata !{i32 16, i32 27, metadata !29, null}
!53 = metadata !{metadata !"0x100\00myBar\0018\000", metadata !54, metadata !6, metadata !5} ; [ DW_TAG_auto_variable ]
!54 = metadata !{metadata !"0xb\0017\001\000", metadata !82, metadata !29} ; [ DW_TAG_lexical_block ]
!55 = metadata !{i32 18, i32 9, metadata !54, null}
!56 = metadata !{i32 18, i32 17, metadata !54, null}
!57 = metadata !{i32 19, i32 5, metadata !54, null}
!58 = metadata !{metadata !"0x101\00this\0016777229\0064", metadata !37, metadata !6, metadata !24} ; [ DW_TAG_arg_variable ]
!59 = metadata !{i32 13, i32 5, metadata !37, null}
!60 = metadata !{metadata !"0x101\00x\0033554445\000", metadata !37, metadata !6, metadata !12} ; [ DW_TAG_arg_variable ]
!61 = metadata !{i32 13, i32 13, metadata !37, null}
!62 = metadata !{i32 13, i32 34, metadata !37, null}
!63 = metadata !{metadata !"0x101\00this\0016777229\0064", metadata !40, metadata !6, metadata !24} ; [ DW_TAG_arg_variable ]
!64 = metadata !{i32 13, i32 5, metadata !40, null}
!65 = metadata !{metadata !"0x101\00x\0033554445\000", metadata !40, metadata !6, metadata !12} ; [ DW_TAG_arg_variable ]
!66 = metadata !{i32 13, i32 13, metadata !40, null}
!67 = metadata !{i32 13, i32 33, metadata !40, null}
!68 = metadata !{i32 13, i32 34, metadata !69, null}
!69 = metadata !{metadata !"0xb\0013\0033\001", metadata !82, metadata !40} ; [ DW_TAG_lexical_block ]
!70 = metadata !{metadata !"0x101\00this\0016777222\0064", metadata !43, metadata !6, metadata !16} ; [ DW_TAG_arg_variable ]
!71 = metadata !{i32 6, i32 5, metadata !43, null}
!72 = metadata !{metadata !"0x101\00a\0033554438\000", metadata !43, metadata !6, metadata !12} ; [ DW_TAG_arg_variable ]
!73 = metadata !{i32 6, i32 13, metadata !43, null}
!74 = metadata !{i32 6, i32 24, metadata !43, null}
!75 = metadata !{metadata !"0x101\00this\0016777222\0064", metadata !46, metadata !6, metadata !16} ; [ DW_TAG_arg_variable ]
!76 = metadata !{i32 6, i32 5, metadata !46, null}
!77 = metadata !{metadata !"0x101\00a\0033554438\000", metadata !46, metadata !6, metadata !12} ; [ DW_TAG_arg_variable ]
!78 = metadata !{i32 6, i32 13, metadata !46, null}
!79 = metadata !{i32 6, i32 23, metadata !46, null}
!80 = metadata !{i32 6, i32 24, metadata !81, null}
!81 = metadata !{metadata !"0xb\006\0023\002", metadata !82, metadata !46} ; [ DW_TAG_lexical_block ]
!82 = metadata !{metadata !"main.cpp", metadata !"/Users/echristo/tmp/bad-struct-ref"}
!83 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
