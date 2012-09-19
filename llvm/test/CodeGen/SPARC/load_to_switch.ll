; RUN: llc -march=sparc < %s | FileCheck %s

; Check that all the switches turned into lookup tables by SimplifyCFG are
; turned back into switches for targets that don't like lookup tables.

@.str = private unnamed_addr constant [4 x i8] c"foo\00", align 1
@.str1 = private unnamed_addr constant [4 x i8] c"bar\00", align 1
@.str2 = private unnamed_addr constant [4 x i8] c"baz\00", align 1
@.str3 = private unnamed_addr constant [4 x i8] c"qux\00", align 1
@.str4 = private unnamed_addr constant [6 x i8] c"error\00", align 1
@switch.table = private unnamed_addr constant [7 x i32] [i32 55, i32 123, i32 0, i32 -1, i32 27, i32 62, i32 1]
@switch.table1 = private unnamed_addr constant [4 x i8] c"*\09X\05"
@switch.table2 = private unnamed_addr constant [4 x float] [float 0x40091EB860000000, float 0x3FF3BE76C0000000, float 0x4012449BA0000000, float 0x4001AE1480000000]
@switch.table3 = private unnamed_addr constant [4 x i8*] [i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8]* @.str1, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8]* @.str2, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8]* @.str3, i64 0, i64 0)]

define i32 @f(i32 %c)  {
entry:
  %switch.tableidx = sub i32 %c, 42
  %0 = icmp ult i32 %switch.tableidx, 7
  br i1 %0, label %switch.lookup, label %return

switch.lookup:
  %switch.gep = getelementptr inbounds [7 x i32]* @switch.table, i32 0, i32 %switch.tableidx
  %switch.load = load i32* %switch.gep
  ret i32 %switch.load

return:
  ret i32 15

; CHECK: f:
; CHECK: %switch.lookup
; CHECK-NOT: sethi %hi(.Lswitch.table)
}

declare void @dummy(i8 signext, float)

define void @h(i32 %x) {
entry:
  %switch.tableidx = sub i32 %x, 0
  %0 = icmp ult i32 %switch.tableidx, 4
  br i1 %0, label %switch.lookup, label %sw.epilog

switch.lookup:
  %switch.gep = getelementptr inbounds [4 x i8]* @switch.table1, i32 0, i32 %switch.tableidx
  %switch.load = load i8* %switch.gep
  %switch.gep1 = getelementptr inbounds [4 x float]* @switch.table2, i32 0, i32 %switch.tableidx
  %switch.load2 = load float* %switch.gep1
  br label %sw.epilog

sw.epilog:
  %a.0 = phi i8 [ %switch.load, %switch.lookup ], [ 7, %entry ]
  %b.0 = phi float [ %switch.load2, %switch.lookup ], [ 0x4023FAE140000000, %entry ]
  call void @dummy(i8 signext %a.0, float %b.0)
  ret void

; CHECK: h:
; CHECK: %switch.lookup
; CHECK-NOT: sethi %hi(.Lswitch.table{{[0-9]}})
; CHECK-NOT: sethi %hi(.Lswitch.table{{[0-9]}})
}

define i8* @foostring(i32 %x) {
entry:
  %switch.tableidx = sub i32 %x, 0
  %0 = icmp ult i32 %switch.tableidx, 4
  br i1 %0, label %switch.lookup, label %return

switch.lookup:
  %switch.gep = getelementptr inbounds [4 x i8*]* @switch.table3, i32 0, i32 %switch.tableidx
  %switch.load = load i8** %switch.gep
  ret i8* %switch.load

return:
  ret i8* getelementptr inbounds ([6 x i8]* @.str4, i64 0, i64 0)

; CHECK: foostring:
; CHECK: %switch.lookup
; CHECK-NOT: sethi %hi(.Lswitch.table3)
}

; CHECK-NOT: .Lswitch.table
; CHECK-NOT: .Lswitch.table1
; CHECK-NOT: .Lswitch.table2
; CHECK-NOT: .Lswitch.table3
