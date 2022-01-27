; RUN: llc -mtriple=armv6m-linux-gnueabi -verify-machineinstrs -o - %s -disable-constant-hoisting | FileCheck %s

%struct_type = type { [10000 x i32], i32, i32 }

define void @test1(%struct_type** %s, i32 %n) {
; CHECK-LABEL: test1
entry:
  %struct = load %struct_type*, %struct_type** %s
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
; CHECK:      str      r{{[0-9]+}}, [r{{[0-9]+}}]
; CHECK-NEXT: str      r{{[0-9]+}}, [r{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
; CHECK: .LCPI0_0:
; CHECK-NEXT: .long   40000
; CHECK-NOT: LCPI0
}

define void @test2(%struct_type* %struct, i32 %n) {
; CHECK-LABEL: test2
entry:
  %cmp = icmp eq %struct_type* %struct, null
  br i1 %cmp, label %while_end, label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp1 = icmp slt i32 %phi, %n
  br i1 %cmp1, label %while_body, label %while_end

while_body:
; CHECK:      str      r{{[0-9]+}}, [r{{[0-9]+}}]
; CHECK-NEXT: str      r{{[0-9]+}}, [r{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
; CHECK: .LCPI1_0:
; CHECK-NEXT: .long   40000
; CHECK-NOT: LCPI1
}

define void @test3(%struct_type* %s1, %struct_type* %s2, i1 %cond, i32 %n) {
; CHECK-LABEL: test3
entry:
  br i1 %cond, label %if_true, label %if_end

if_true:
  br label %if_end

if_end:
  %struct = phi %struct_type* [ %s1, %entry ], [ %s2, %if_true ]
  %cmp = icmp eq %struct_type* %struct, null
  br i1 %cmp, label %while_end, label %while_cond

while_cond:
  %phi = phi i32 [ 0, %if_end ], [ %i, %while_body ]
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp1 = icmp slt i32 %phi, %n
  br i1 %cmp1, label %while_body, label %while_end

while_body:
; CHECK:      str      r{{[0-9]+}}, [r{{[0-9]+}}]
; CHECK-NEXT: str      r{{[0-9]+}}, [r{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
; CHECK: .LCPI2_0:
; CHECK-NEXT: .long   40000
; CHECK-NOT: LCPI2
}

declare %struct_type* @foo()

define void @test4(i32 %n) personality i32 (...)* @__FrameHandler {
; CHECK-LABEL: test4
entry:
  %struct = invoke %struct_type* @foo() to label %while_cond unwind label %cleanup

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
; CHECK:      str      r{{[0-9]+}}, [r{{[0-9]+}}]
; CHECK-NEXT: str      r{{[0-9]+}}, [r{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void

cleanup:
  landingpad { i8*, i32 } cleanup
  unreachable
; CHECK: .LCPI3_0:
; CHECK-NEXT: .long   40000
; CHECK-NOT: LCPI3
}

declare i32 @__FrameHandler(...)

define void @test5([65536 x i32]** %s, i32 %n) {
; CHECK-LABEL: test5
entry:
  %struct = load [65536 x i32]*, [65536 x i32]** %s
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %gep0 = getelementptr [65536 x i32], [65536 x i32]* %struct, i64 0, i32 20000
  %gep1 = getelementptr [65536 x i32], [65536 x i32]* %struct, i64 0, i32 20001
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
; CHECK:      str      r{{[0-9]+}}, [r{{[0-9]+}}]
; CHECK-NEXT: str      r{{[0-9]+}}, [r{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
; CHECK: .LCPI4_0:
; CHECK-NEXT: .long   80000
; CHECK-NOT: LCPI4
}
