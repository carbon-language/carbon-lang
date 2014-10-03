; RUN: llc -O0 < %s | FileCheck %s
; CHECK: @DEBUG_VALUE: foobar_func_block_invoke_0:mydata <- [SP+{{[0-9]+}}]
; Radar 9331779
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%0 = type opaque
%1 = type { [4 x i32] }
%2 = type <{ i8*, i32, i32, i8*, %struct.Re*, i8*, %3*, %struct.my_struct* }>
%3 = type opaque
%struct.CP = type { float, float }
%struct.CR = type { %struct.CP, %struct.CP }
%struct.Re = type { i32, i32 }
%struct.__block_byref_mydata = type { i8*, %struct.__block_byref_mydata*, i32, i32, i8*, i8*, %0* }
%struct.my_struct = type opaque

@"\01L_OBJC_SELECTOR_REFERENCES_13" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"OBJC_IVAR_$_MyWork._bounds" = external hidden global i32, section "__DATA, __objc_const", align 4
@"OBJC_IVAR_$_MyWork._data" = external hidden global i32, section "__DATA, __objc_const", align 4
@"\01L_OBJC_SELECTOR_REFERENCES_222" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @objc_msgSend(i8*, i8*, ...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define hidden void @foobar_func_block_invoke_0(i8* %.block_descriptor, %0* %loadedMydata, [4 x i32] %bounds.coerce0, [4 x i32] %data.coerce0) ssp {
  %1 = alloca %0*, align 4
  %bounds = alloca %struct.CR, align 4
  %data = alloca %struct.CR, align 4
  call void @llvm.dbg.value(metadata !{i8* %.block_descriptor}, i64 0, metadata !27, metadata !{metadata !"0x102"}), !dbg !129
  store %0* %loadedMydata, %0** %1, align 4
  call void @llvm.dbg.declare(metadata !{%0** %1}, metadata !130, metadata !{metadata !"0x102"}), !dbg !131
  %2 = bitcast %struct.CR* %bounds to %1*
  %3 = getelementptr %1* %2, i32 0, i32 0
  store [4 x i32] %bounds.coerce0, [4 x i32]* %3
  call void @llvm.dbg.declare(metadata !{%struct.CR* %bounds}, metadata !132, metadata !{metadata !"0x102"}), !dbg !133
  %4 = bitcast %struct.CR* %data to %1*
  %5 = getelementptr %1* %4, i32 0, i32 0
  store [4 x i32] %data.coerce0, [4 x i32]* %5
  call void @llvm.dbg.declare(metadata !{%struct.CR* %data}, metadata !134, metadata !{metadata !"0x102"}), !dbg !135
  %6 = bitcast i8* %.block_descriptor to %2*
  %7 = getelementptr inbounds %2* %6, i32 0, i32 6
  call void @llvm.dbg.declare(metadata !{%2* %6}, metadata !136, metadata !163), !dbg !137
  call void @llvm.dbg.declare(metadata !{%2* %6}, metadata !138, metadata !164), !dbg !137
  call void @llvm.dbg.declare(metadata !{%2* %6}, metadata !139, metadata !165), !dbg !140
  %8 = load %0** %1, align 4, !dbg !141
  %9 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_13", !dbg !141
  %10 = bitcast %0* %8 to i8*, !dbg !141
  %11 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %10, i8* %9), !dbg !141
  %12 = bitcast i8* %11 to %0*, !dbg !141
  %13 = getelementptr inbounds %2* %6, i32 0, i32 5, !dbg !141
  %14 = load i8** %13, !dbg !141
  %15 = bitcast i8* %14 to %struct.__block_byref_mydata*, !dbg !141
  %16 = getelementptr inbounds %struct.__block_byref_mydata* %15, i32 0, i32 1, !dbg !141
  %17 = load %struct.__block_byref_mydata** %16, !dbg !141
  %18 = getelementptr inbounds %struct.__block_byref_mydata* %17, i32 0, i32 6, !dbg !141
  store %0* %12, %0** %18, align 4, !dbg !141
  %19 = getelementptr inbounds %2* %6, i32 0, i32 6, !dbg !143
  %20 = load %3** %19, align 4, !dbg !143
  %21 = load i32* @"OBJC_IVAR_$_MyWork._data", !dbg !143
  %22 = bitcast %3* %20 to i8*, !dbg !143
  %23 = getelementptr inbounds i8* %22, i32 %21, !dbg !143
  %24 = bitcast i8* %23 to %struct.CR*, !dbg !143
  %25 = bitcast %struct.CR* %24 to i8*, !dbg !143
  %26 = bitcast %struct.CR* %data to i8*, !dbg !143
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %25, i8* %26, i32 16, i32 4, i1 false), !dbg !143
  %27 = getelementptr inbounds %2* %6, i32 0, i32 6, !dbg !144
  %28 = load %3** %27, align 4, !dbg !144
  %29 = load i32* @"OBJC_IVAR_$_MyWork._bounds", !dbg !144
  %30 = bitcast %3* %28 to i8*, !dbg !144
  %31 = getelementptr inbounds i8* %30, i32 %29, !dbg !144
  %32 = bitcast i8* %31 to %struct.CR*, !dbg !144
  %33 = bitcast %struct.CR* %32 to i8*, !dbg !144
  %34 = bitcast %struct.CR* %bounds to i8*, !dbg !144
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %33, i8* %34, i32 16, i32 4, i1 false), !dbg !144
  %35 = getelementptr inbounds %2* %6, i32 0, i32 6, !dbg !145
  %36 = load %3** %35, align 4, !dbg !145
  %37 = getelementptr inbounds %2* %6, i32 0, i32 5, !dbg !145
  %38 = load i8** %37, !dbg !145
  %39 = bitcast i8* %38 to %struct.__block_byref_mydata*, !dbg !145
  %40 = getelementptr inbounds %struct.__block_byref_mydata* %39, i32 0, i32 1, !dbg !145
  %41 = load %struct.__block_byref_mydata** %40, !dbg !145
  %42 = getelementptr inbounds %struct.__block_byref_mydata* %41, i32 0, i32 6, !dbg !145
  %43 = load %0** %42, align 4, !dbg !145
  %44 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_222", !dbg !145
  %45 = bitcast %3* %36 to i8*, !dbg !145
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %0*)*)(i8* %45, i8* %44, %0* %43), !dbg !145
  ret void, !dbg !146
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!162}

!0 = metadata !{metadata !"0x11\0016\00Apple clang version 2.1\000\00\002\00\001", metadata !153, metadata !147, metadata !26, metadata !148, null, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x4\00\00248\0032\0032\000\000\000", metadata !160, metadata !0, null, metadata !3, null, null, null} ; [ DW_TAG_enumeration_type ] [line 248, size 32, align 32, offset 0] [def] [from ]
!2 = metadata !{metadata !"0x29", metadata !160} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x28\00Ver1\000"} ; [ DW_TAG_enumerator ]
!5 = metadata !{metadata !"0x4\00Mode\0079\0032\0032\000\000\000", metadata !160, metadata !0, null, metadata !7, null, null, null} ; [ DW_TAG_enumeration_type ] [Mode] [line 79, size 32, align 32, offset 0] [def] [from ]
!6 = metadata !{metadata !"0x29", metadata !161} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x28\00One\000"} ; [ DW_TAG_enumerator ]
!9 = metadata !{metadata !"0x4\00\0015\0032\0032\000\000\000", metadata !149, metadata !0, null, metadata !11, null, null, null} ; [ DW_TAG_enumeration_type ] [line 15, size 32, align 32, offset 0] [def] [from ]
!10 = metadata !{metadata !"0x29", metadata !149} ; [ DW_TAG_file_type ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{metadata !"0x28\00Unknown\000"} ; [ DW_TAG_enumerator ]
!13 = metadata !{metadata !"0x28\00Known\001"} ; [ DW_TAG_enumerator ]
!14 = metadata !{metadata !"0x4\00\0020\0032\0032\000\000\000", metadata !150, metadata !0, null, metadata !16, null, null, null} ; [ DW_TAG_enumeration_type ] [line 20, size 32, align 32, offset 0] [def] [from ]
!15 = metadata !{metadata !"0x29", metadata !150} ; [ DW_TAG_file_type ]
!16 = metadata !{metadata !17, metadata !18}
!17 = metadata !{metadata !"0x28\00Single\000"} ; [ DW_TAG_enumerator ]
!18 = metadata !{metadata !"0x28\00Double\001"} ; [ DW_TAG_enumerator ]
!19 = metadata !{metadata !"0x4\00\0014\0032\0032\000\000\000", metadata !151, metadata !0, null, metadata !21, null, null, null} ; [ DW_TAG_enumeration_type ] [line 14, size 32, align 32, offset 0] [def] [from ]
!20 = metadata !{metadata !"0x29", metadata !151} ; [ DW_TAG_file_type ]
!21 = metadata !{metadata !22}
!22 = metadata !{metadata !"0x28\00Eleven\000"} ; [ DW_TAG_enumerator ]
!23 = metadata !{metadata !"0x2e\00foobar_func_block_invoke_0\00foobar_func_block_invoke_0\00\00609\001\001\000\006\00256\000\00609", metadata !152, metadata !24, metadata !25, null, void (i8*, %0*, [4 x i32], [4 x i32])* @foobar_func_block_invoke_0, null, null, null} ; [ DW_TAG_subprogram ] [line 609] [local] [def] [foobar_func_block_invoke_0]
!24 = metadata !{metadata !"0x29", metadata !152} ; [ DW_TAG_file_type ]
!25 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !152, metadata !24, null, metadata !26, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!26 = metadata !{null}
!27 = metadata !{metadata !"0x101\00.block_descriptor\0016777825\0064", metadata !23, metadata !24, metadata !28} ; [ DW_TAG_arg_variable ]
!28 = metadata !{metadata !"0xf\00\000\0032\000\000\000", null, metadata !0, metadata !29} ; [ DW_TAG_pointer_type ]
!29 = metadata !{metadata !"0x13\00__block_literal_14\00609\00256\0032\000\000\000", metadata !152, metadata !24, null, metadata !30, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_14] [line 609, size 256, align 32, offset 0] [def] [from ]
!30 = metadata !{metadata !31, metadata !33, metadata !35, metadata !36, metadata !37, metadata !48, metadata !89, metadata !124}
!31 = metadata !{metadata !"0xd\00__isa\00609\0032\0032\000\000", metadata !152, metadata !24, metadata !32} ; [ DW_TAG_member ]
!32 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, null} ; [ DW_TAG_pointer_type ]
!33 = metadata !{metadata !"0xd\00__flags\00609\0032\0032\0032\000", metadata !152, metadata !24, metadata !34} ; [ DW_TAG_member ]
!34 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!35 = metadata !{metadata !"0xd\00__reserved\00609\0032\0032\0064\000", metadata !152, metadata !24, metadata !34} ; [ DW_TAG_member ]
!36 = metadata !{metadata !"0xd\00__FuncPtr\00609\0032\0032\0096\000", metadata !152, metadata !24, metadata !32} ; [ DW_TAG_member ]
!37 = metadata !{metadata !"0xd\00__descriptor\00609\0032\0032\00128\000", metadata !152, metadata !24, metadata !38} ; [ DW_TAG_member ]
!38 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !39} ; [ DW_TAG_pointer_type ]
!39 = metadata !{metadata !"0x13\00__block_descriptor_withcopydispose\00307\00128\0032\000\000\000", metadata !153, metadata !0, null, metadata !41, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 307, size 128, align 32, offset 0] [def] [from ]
!40 = metadata !{metadata !"0x29", metadata !153} ; [ DW_TAG_file_type ]
!41 = metadata !{metadata !42, metadata !44, metadata !45, metadata !47}
!42 = metadata !{metadata !"0xd\00reserved\00307\0032\0032\000\000", metadata !153, metadata !40, metadata !43} ; [ DW_TAG_member ]
!43 = metadata !{metadata !"0x24\00long unsigned int\000\0032\0032\000\000\007", null, metadata !0} ; [ DW_TAG_base_type ]
!44 = metadata !{metadata !"0xd\00Size\00307\0032\0032\0032\000", metadata !153, metadata !40, metadata !43} ; [ DW_TAG_member ]
!45 = metadata !{metadata !"0xd\00CopyFuncPtr\00307\0032\0032\0064\000", metadata !153, metadata !40, metadata !46} ; [ DW_TAG_member ]
!46 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !32} ; [ DW_TAG_pointer_type ]
!47 = metadata !{metadata !"0xd\00DestroyFuncPtr\00307\0032\0032\0096\000", metadata !153, metadata !40, metadata !46} ; [ DW_TAG_member ]
!48 = metadata !{metadata !"0xd\00mydata\00609\0032\0032\00160\000", metadata !152, metadata !24, metadata !49} ; [ DW_TAG_member ]
!49 = metadata !{metadata !"0xf\00\000\0032\000\000\000", null, metadata !0, metadata !50} ; [ DW_TAG_pointer_type ]
!50 = metadata !{metadata !"0x13\00\000\00224\000\000\0016\000", metadata !152, metadata !24, null, metadata !51, null, null, null} ; [ DW_TAG_structure_type ] [line 0, size 224, align 0, offset 0] [def] [from ]
!51 = metadata !{metadata !52, metadata !53, metadata !54, metadata !55, metadata !56, metadata !57, metadata !58}
!52 = metadata !{metadata !"0xd\00__isa\000\0032\0032\000\000", metadata !152, metadata !24, metadata !32} ; [ DW_TAG_member ]
!53 = metadata !{metadata !"0xd\00__forwarding\000\0032\0032\0032\000", metadata !152, metadata !24, metadata !32} ; [ DW_TAG_member ]
!54 = metadata !{metadata !"0xd\00__flags\000\0032\0032\0064\000", metadata !152, metadata !24, metadata !34} ; [ DW_TAG_member ]
!55 = metadata !{metadata !"0xd\00__size\000\0032\0032\0096\000", metadata !152, metadata !24, metadata !34} ; [ DW_TAG_member ]
!56 = metadata !{metadata !"0xd\00__copy_helper\000\0032\0032\00128\000", metadata !152, metadata !24, metadata !32} ; [ DW_TAG_member ]
!57 = metadata !{metadata !"0xd\00__destroy_helper\000\0032\0032\00160\000", metadata !152, metadata !24, metadata !32} ; [ DW_TAG_member ]
!58 = metadata !{metadata !"0xd\00mydata\000\0032\0032\00192\000", metadata !152, metadata !24, metadata !59} ; [ DW_TAG_member ]
!59 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !60} ; [ DW_TAG_pointer_type ]
!60 = metadata !{metadata !"0x13\00UIMydata\0026\00128\0032\000\000\0016", metadata !154, metadata !24, null, metadata !62, null, null, null} ; [ DW_TAG_structure_type ] [UIMydata] [line 26, size 128, align 32, offset 0] [def] [from ]
!61 = metadata !{metadata !"0x29", metadata !154} ; [ DW_TAG_file_type ]
!62 = metadata !{metadata !63, metadata !71, metadata !75, metadata !79}
!63 = metadata !{metadata !"0x1c\00\000\000\000\000\000", metadata !60, null, metadata !64} ; [ DW_TAG_inheritance ]
!64 = metadata !{metadata !"0x13\00NSO\0066\0032\0032\000\000\0016", metadata !155, metadata !40, null, metadata !66, null, null, null} ; [ DW_TAG_structure_type ] [NSO] [line 66, size 32, align 32, offset 0] [def] [from ]
!65 = metadata !{metadata !"0x29", metadata !155} ; [ DW_TAG_file_type ]
!66 = metadata !{metadata !67}
!67 = metadata !{metadata !"0xd\00isa\0067\0032\0032\000\002", metadata !155, metadata !65, metadata !68, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!68 = metadata !{metadata !"0x16\00Class\00197\000\000\000\000", metadata !153, metadata !0, metadata !69} ; [ DW_TAG_typedef ]
!69 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !70} ; [ DW_TAG_pointer_type ]
!70 = metadata !{metadata !"0x13\00objc_class\000\000\000\000\004\000", metadata !153, metadata !0, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!71 = metadata !{metadata !"0xd\00_mydataRef\0028\0032\0032\0032\000", metadata !154, metadata !61, metadata !72, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!72 = metadata !{metadata !"0x16\00CFTypeRef\00313\000\000\000\000", metadata !152, metadata !0, metadata !73} ; [ DW_TAG_typedef ]
!73 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !74} ; [ DW_TAG_pointer_type ]
!74 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, metadata !0, null} ; [ DW_TAG_const_type ]
!75 = metadata !{metadata !"0xd\00_scale\0029\0032\0032\0064\000", metadata !154, metadata !61, metadata !76, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!76 = metadata !{metadata !"0x16\00Float\0089\000\000\000\000", metadata !156, metadata !0, metadata !78} ; [ DW_TAG_typedef ]
!77 = metadata !{metadata !"0x29", metadata !156} ; [ DW_TAG_file_type ]
!78 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, metadata !0} ; [ DW_TAG_base_type ]
!79 = metadata !{metadata !"0xd\00_mydataFlags\0037\008\008\0096\000", metadata !154, metadata !61, metadata !80, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!80 = metadata !{metadata !"0x13\00\0030\008\008\000\000\000", metadata !154, metadata !0, null, metadata !81, null, null, null} ; [ DW_TAG_structure_type ] [line 30, size 8, align 8, offset 0] [def] [from ]
!81 = metadata !{metadata !82, metadata !84, metadata !85, metadata !86, metadata !87, metadata !88}
!82 = metadata !{metadata !"0xd\00named\0031\001\0032\000\000", metadata !154, metadata !61, metadata !83} ; [ DW_TAG_member ]
!83 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", null, metadata !0} ; [ DW_TAG_base_type ]
!84 = metadata !{metadata !"0xd\00mydataO\0032\003\0032\001\000", metadata !154, metadata !61, metadata !83} ; [ DW_TAG_member ]
!85 = metadata !{metadata !"0xd\00cached\0033\001\0032\004\000", metadata !154, metadata !61, metadata !83} ; [ DW_TAG_member ]
!86 = metadata !{metadata !"0xd\00hasBeenCached\0034\001\0032\005\000", metadata !154, metadata !61, metadata !83} ; [ DW_TAG_member ]
!87 = metadata !{metadata !"0xd\00hasPattern\0035\001\0032\006\000", metadata !154, metadata !61, metadata !83} ; [ DW_TAG_member ]
!88 = metadata !{metadata !"0xd\00isCIMydata\0036\001\0032\007\000", metadata !154, metadata !61, metadata !83} ; [ DW_TAG_member ]
!89 = metadata !{metadata !"0xd\00self\00609\0032\0032\00192\000", metadata !152, metadata !24, metadata !90} ; [ DW_TAG_member ]
!90 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !91} ; [ DW_TAG_pointer_type ]
!91 = metadata !{metadata !"0x13\00MyWork\0036\00384\0032\000\000\0016", metadata !152, metadata !40, null, metadata !92, null, null, null} ; [ DW_TAG_structure_type ] [MyWork] [line 36, size 384, align 32, offset 0] [def] [from ]
!92 = metadata !{metadata !93, metadata !98, metadata !101, metadata !107, metadata !123}
!93 = metadata !{metadata !"0x1c\00\000\000\000\000\000", metadata !152, metadata !91, metadata !94} ; [ DW_TAG_inheritance ]
!94 = metadata !{metadata !"0x13\00twork\0043\0032\0032\000\000\0016", metadata !157, metadata !40, null, metadata !96, null, null, null} ; [ DW_TAG_structure_type ] [twork] [line 43, size 32, align 32, offset 0] [def] [from ]
!95 = metadata !{metadata !"0x29", metadata !157} ; [ DW_TAG_file_type ]
!96 = metadata !{metadata !97}
!97 = metadata !{metadata !"0x1c\00\000\000\000\000\000", metadata !94, null, metadata !64} ; [ DW_TAG_inheritance ]
!98 = metadata !{metadata !"0xd\00_itemID\0038\0064\0032\0032\001", metadata !152, metadata !24, metadata !99, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!99 = metadata !{metadata !"0x16\00uint64_t\0055\000\000\000\000", metadata !153, metadata !0, metadata !100} ; [ DW_TAG_typedef ]
!100 = metadata !{metadata !"0x24\00long long unsigned int\000\0064\0032\000\000\007", null, metadata !0} ; [ DW_TAG_base_type ]
!101 = metadata !{metadata !"0xd\00_library\0039\0032\0032\0096\001", metadata !152, metadata !24, metadata !102, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!102 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !103} ; [ DW_TAG_pointer_type ]
!103 = metadata !{metadata !"0x13\00MyLibrary2\0022\0032\0032\000\000\0016", metadata !158, metadata !40, null, metadata !105, null, null, null} ; [ DW_TAG_structure_type ] [MyLibrary2] [line 22, size 32, align 32, offset 0] [def] [from ]
!104 = metadata !{metadata !"0x29", metadata !158} ; [ DW_TAG_file_type ]
!105 = metadata !{metadata !106}
!106 = metadata !{metadata !"0x1c\00\000\000\000\000\000", metadata !103, null, metadata !64} ; [ DW_TAG_inheritance ]
!107 = metadata !{metadata !"0xd\00_bounds\0040\00128\0032\00128\001", metadata !152, metadata !24, metadata !108, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!108 = metadata !{metadata !"0x16\00CR\0033\000\000\000\000", metadata !153, metadata !0, metadata !109} ; [ DW_TAG_typedef ]
!109 = metadata !{metadata !"0x13\00CR\0029\00128\0032\000\000\000", metadata !156, metadata !0, null, metadata !110, null, null, null} ; [ DW_TAG_structure_type ] [CR] [line 29, size 128, align 32, offset 0] [def] [from ]
!110 = metadata !{metadata !111, metadata !117}
!111 = metadata !{metadata !"0xd\00origin\0030\0064\0032\000\000", metadata !156, metadata !77, metadata !112} ; [ DW_TAG_member ]
!112 = metadata !{metadata !"0x16\00CP\0017\000\000\000\000", metadata !156, metadata !0, metadata !113} ; [ DW_TAG_typedef ]
!113 = metadata !{metadata !"0x13\00CP\0013\0064\0032\000\000\000", metadata !156, metadata !0, null, metadata !114, null, null, null} ; [ DW_TAG_structure_type ] [CP] [line 13, size 64, align 32, offset 0] [def] [from ]
!114 = metadata !{metadata !115, metadata !116}
!115 = metadata !{metadata !"0xd\00x\0014\0032\0032\000\000", metadata !156, metadata !77, metadata !76} ; [ DW_TAG_member ]
!116 = metadata !{metadata !"0xd\00y\0015\0032\0032\0032\000", metadata !156, metadata !77, metadata !76} ; [ DW_TAG_member ]
!117 = metadata !{metadata !"0xd\00size\0031\0064\0032\0064\000", metadata !156, metadata !77, metadata !118} ; [ DW_TAG_member ]
!118 = metadata !{metadata !"0x16\00Size\0025\000\000\000\000", metadata !156, metadata !0, metadata !119} ; [ DW_TAG_typedef ]
!119 = metadata !{metadata !"0x13\00Size\0021\0064\0032\000\000\000", metadata !156, metadata !0, null, metadata !120, null, null, null} ; [ DW_TAG_structure_type ] [Size] [line 21, size 64, align 32, offset 0] [def] [from ]
!120 = metadata !{metadata !121, metadata !122}
!121 = metadata !{metadata !"0xd\00width\0022\0032\0032\000\000", metadata !156, metadata !77, metadata !76} ; [ DW_TAG_member ]
!122 = metadata !{metadata !"0xd\00height\0023\0032\0032\0032\000", metadata !156, metadata !77, metadata !76} ; [ DW_TAG_member ]
!123 = metadata !{metadata !"0xd\00_data\0040\00128\0032\00256\001", metadata !152, metadata !24, metadata !108, metadata !"", metadata !"", metadata !"", i32 0} ; [ DW_TAG_member ]
!124 = metadata !{metadata !"0xd\00semi\00609\0032\0032\00224\000", metadata !152, metadata !24, metadata !125} ; [ DW_TAG_member ]
!125 = metadata !{metadata !"0x16\00d_t\0035\000\000\000\000", metadata !152, metadata !0, metadata !126} ; [ DW_TAG_typedef ]
!126 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !0, metadata !127} ; [ DW_TAG_pointer_type ]
!127 = metadata !{metadata !"0x13\00my_struct\0049\000\000\000\004\000", metadata !159, metadata !0, null, null, null, null, null} ; [ DW_TAG_structure_type ] [my_struct] [line 49, size 0, align 0, offset 0] [decl] [from ]
!128 = metadata !{metadata !"0x29", metadata !159} ; [ DW_TAG_file_type ]
!129 = metadata !{i32 609, i32 144, metadata !23, null}
!130 = metadata !{metadata !"0x101\00loadedMydata\0033555041\000", metadata !23, metadata !24, metadata !59} ; [ DW_TAG_arg_variable ]
!131 = metadata !{i32 609, i32 155, metadata !23, null}
!132 = metadata !{metadata !"0x101\00bounds\0050332257\000", metadata !23, metadata !24, metadata !108} ; [ DW_TAG_arg_variable ]
!133 = metadata !{i32 609, i32 175, metadata !23, null}
!134 = metadata !{metadata !"0x101\00data\0067109473\000", metadata !23, metadata !24, metadata !108} ; [ DW_TAG_arg_variable ]
!135 = metadata !{i32 609, i32 190, metadata !23, null}
!136 = metadata !{metadata !"0x100\00mydata\00604\000", metadata !23, metadata !24, metadata !50} ; [ DW_TAG_auto_variable ]
!137 = metadata !{i32 604, i32 49, metadata !23, null}
!138 = metadata !{metadata !"0x100\00self\00604\000", metadata !23, metadata !40, metadata !90} ; [ DW_TAG_auto_variable ]
!139 = metadata !{metadata !"0x100\00semi\00607\000", metadata !23, metadata !24, metadata !125} ; [ DW_TAG_auto_variable ]
!140 = metadata !{i32 607, i32 30, metadata !23, null}
!141 = metadata !{i32 610, i32 17, metadata !142, null}
!142 = metadata !{metadata !"0xb\00609\00200\0094", metadata !152, metadata !23} ; [ DW_TAG_lexical_block ]
!143 = metadata !{i32 611, i32 17, metadata !142, null}
!144 = metadata !{i32 612, i32 17, metadata !142, null}
!145 = metadata !{i32 613, i32 17, metadata !142, null}
!146 = metadata !{i32 615, i32 13, metadata !142, null}
!147 = metadata !{metadata !1, metadata !1, metadata !5, metadata !5, metadata !9, metadata !14, metadata !19, metadata !19, metadata !14, metadata !14, metadata !14, metadata !19, metadata !19, metadata !19}
!148 = metadata !{metadata !23}
!149 = metadata !{metadata !"header3.h", metadata !"/Volumes/Sandbox/llvm"}
!150 = metadata !{metadata !"Private.h", metadata !"/Volumes/Sandbox/llvm"}
!151 = metadata !{metadata !"header4.h", metadata !"/Volumes/Sandbox/llvm"}
!152 = metadata !{metadata !"MyLibrary.m", metadata !"/Volumes/Sandbox/llvm"}
!153 = metadata !{metadata !"MyLibrary.i", metadata !"/Volumes/Sandbox/llvm"}
!154 = metadata !{metadata !"header11.h", metadata !"/Volumes/Sandbox/llvm"}
!155 = metadata !{metadata !"NSO.h", metadata !"/Volumes/Sandbox/llvm"}
!156 = metadata !{metadata !"header12.h", metadata !"/Volumes/Sandbox/llvm"}
!157 = metadata !{metadata !"header13.h", metadata !"/Volumes/Sandbox/llvm"}
!158 = metadata !{metadata !"header14.h", metadata !"/Volumes/Sandbox/llvm"}
!159 = metadata !{metadata !"header15.h", metadata !"/Volumes/Sandbox/llvm"}
!160 = metadata !{metadata !"header.h", metadata !"/Volumes/Sandbox/llvm"}
!161 = metadata !{metadata !"header2.h", metadata !"/Volumes/Sandbox/llvm"}
!162 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!163 = metadata !{metadata !"0x102\0034\0020\006\0034\004\006\0034\0024"} ; [ DW_TAG_expression ] [DW_OP_plus 20 DW_OP_deref DW_OP_plus 4 DW_OP_deref DW_OP_plus 24]
!164 = metadata !{metadata !"0x102\0034\0024"} ; [ DW_TAG_expression ] [DW_OP_plus 24]
!165 = metadata !{metadata !"0x102\0034\0028"} ; [ DW_TAG_expression ] [DW_OP_plus 28]
