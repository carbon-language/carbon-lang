; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/2009-09-03-mdnode2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc

declare void @f() nounwind

define i32 @main(...) nounwind {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  call void @llvm.dbg.func.start(metadata !0)
  store i32 0, i32* %retval
  call void @llvm.dbg.stoppoint(i32 4, i32 5, metadata !1)
  call void @f()
  br label %return

return:                                           ; preds = %entry
  %0 = load i32, i32* %retval                          ; <i32> [#uses=1]
  call void @llvm.dbg.stoppoint(i32 5, i32 1, metadata !1)
  call void @llvm.dbg.region.end(metadata !0)
  ret i32 %0
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.stoppoint(i32, i32, metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!0 = !{!"0x2e\00main\00main\00main\002\000\001\000\006\000\000\000", i32 0, !1, null, null, null, null, null} ; [ DW_TAG_subprogram ]
!1 = !{!"0x11\0012\00ellcc 0.1.0\001\00\000\00\000", !2, null, null, null, null, null} ; [ DW_TAG_compile_unit ]
!2 = !{!"a.c", !"/home/rich/ellcc/test/source"}
