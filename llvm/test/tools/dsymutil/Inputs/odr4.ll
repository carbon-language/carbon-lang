; Generated from odr4.cpp and odr-types.h by running:
; clang -emit-llvm -g -S -std=c++11 odr4.cpp
; ModuleID = 'odr4.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%"class.(anonymous namespace)::AnonC" = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_Z3bazv() #0 {
entry:
  %ac = alloca %"class.(anonymous namespace)::AnonC", align 1
  call void @llvm.dbg.declare(metadata %"class.(anonymous namespace)::AnonC"* %ac, metadata !11, metadata !14), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 242534)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "odr4.cpp", directory: "/Inputs")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 6, type: !5, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, function: void ()* @_Z3bazv, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 3.8.0 (trunk 242534)"}
!11 = !DILocalVariable(name: "ac", scope: !4, file: !1, line: 7, type: !12)
!12 = !DICompositeType(tag: DW_TAG_class_type, name: "AnonC", scope: !13, file: !1, line: 2, size: 8, align: 8, elements: !2)
!13 = !DINamespace(scope: null, file: !1, line: 1)
!14 = !DIExpression()
!15 = !DILocation(line: 7, column: 8, scope: !4)
!16 = !DILocation(line: 8, column: 1, scope: !4)
