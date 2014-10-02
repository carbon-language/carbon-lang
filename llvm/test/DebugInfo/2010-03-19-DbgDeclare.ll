; RUN: opt < %s -verify -S | FileCheck %s

; CHECK: [DW_LANG_Mips_Assembler]

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.declare(metadata !{i32* null}, metadata !1, metadata !{metadata !"0x102"})
  ret void
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5}
!2 = metadata !{metadata !"0x11\0032769\00clang version 3.3 \000\00\000\00\001", metadata !4, metadata !3, metadata !3, metadata !3, metadata !3,  metadata !3} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp] [lang 0x8001]
!3 = metadata !{}
!0 = metadata !{i32 662302, i32 26, metadata !1, null}
!1 = metadata !{i32 4, metadata !"foo"}
!4 = metadata !{metadata !"scratch.cpp", metadata !"/usr/local/google/home/blaikie/dev/scratch"}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
!5 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
