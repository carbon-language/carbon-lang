; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; Stackmap Header: no constants - 6 callsites
; CHECK-LABEL: .section	__LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .long   0
; Num Constants
; CHECK-NEXT:   .long   0
; Num Callsites
; CHECK-NEXT:   .long   6

; test
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   L{{.*}}-_test
; CHECK-NEXT:   .short  0
; 3 locations
; CHECK-NEXT:   .short  3
; Loc 0: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 2: Constant 3
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long 3
define i64 @test() nounwind ssp uwtable {
entry:
  call anyregcc void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 0, i32 15, i8* null, i32 2, i32 1, i32 2, i64 3)
  ret i64 0
}

; property access 1 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-NEXT:   .long   1
; CHECK-NEXT:   .long   L{{.*}}-_property_access1
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @property_access1(i8* %obj) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 1, i32 15, i8* %f, i32 1, i8* %obj)
  ret i64 %ret
}

; property access 2 - %obj is an anyreg call argument and should therefore be in a register
; CHECK-NEXT:   .long   2
; CHECK-NEXT:   .long   L{{.*}}-_property_access2
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @property_access2() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 2, i32 15, i8* %f, i32 1, i64* %obj)
  ret i64 %ret
}

; property access 3 - %obj is a frame index
; CHECK-NEXT:   .long   3
; CHECK-NEXT:   .long   L{{.*}}-_property_access3
; CHECK-NEXT:   .short  0
; 2 locations
; CHECK-NEXT:   .short  2
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register <-- this will be folded once folding for FI is implemented
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @property_access3() nounwind ssp uwtable {
entry:
  %obj = alloca i64, align 8
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 3, i32 15, i8* %f, i32 0, i64* %obj)
  ret i64 %ret
}

; anyreg_test1
; CHECK-NEXT:   .long   4
; CHECK-NEXT:   .long   L{{.*}}-_anyreg_test1
; CHECK-NEXT:   .short  0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @anyreg_test1(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 4, i32 15, i8* %f, i32 13, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

; anyreg_test2
; CHECK-NEXT:   .long   5
; CHECK-NEXT:   .long   L{{.*}}-_anyreg_test2
; CHECK-NEXT:   .short  0
; 14 locations
; CHECK-NEXT:   .short  14
; Loc 0: Register <-- this is the return register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 1: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 2: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 3: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 4: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 5: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 6: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 7: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 8: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 9: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 10: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 11: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 12: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
; Loc 13: Register
; CHECK-NEXT:   .byte 1
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short {{[0-9]+}}
; CHECK-NEXT:   .long 0
define i64 @anyreg_test2(i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13) nounwind ssp uwtable {
entry:
  %f = inttoptr i64 12297829382473034410 to i8*
  %ret = call anyregcc i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 5, i32 15, i8* %f, i32 8, i8* %a1, i8* %a2, i8* %a3, i8* %a4, i8* %a5, i8* %a6, i8* %a7, i8* %a8, i8* %a9, i8* %a10, i8* %a11, i8* %a12, i8* %a13)
  ret i64 %ret
}

declare void @llvm.experimental.patchpoint.void(i32, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i32, i32, i8*, i32, ...)

