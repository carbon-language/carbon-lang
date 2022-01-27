; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 <%s | FileCheck %s

%struct.Record = type { %struct.Record*, i32 }

@n = local_unnamed_addr global i32 500000000, align 4
@m = common global %struct.Record zeroinitializer, align 8
@a = hidden local_unnamed_addr global %struct.Record* @m, align 8
@o = common global %struct.Record zeroinitializer, align 8
@b = hidden local_unnamed_addr global %struct.Record* @o, align 8

define signext i32 @foo() local_unnamed_addr {
entry:
  %0 = load i64, i64* bitcast (%struct.Record** @b to i64*), align 8
  %1 = load i64*, i64** bitcast (%struct.Record** @a to i64**), align 8
  store i64 %0, i64* %1, align 8
  %2 = load i32, i32* @n, align 4
  %cmp9 = icmp eq i32 %2, 0
  br i1 %cmp9, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %3 = load %struct.Record*, %struct.Record** @a, align 8
  %IntComp = getelementptr inbounds %struct.Record, %struct.Record* %3, i64 0, i32 1
  store i32 5, i32* %IntComp, align 8
  %PtrComp2 = getelementptr inbounds %struct.Record, %struct.Record* %3, i64 0, i32 0
  %4 = load %struct.Record*, %struct.Record** %PtrComp2, align 8
  %IntComp3 = getelementptr inbounds %struct.Record, %struct.Record* %4, i64 0, i32 1
  store i32 5, i32* %IntComp3, align 8
  %PtrComp6 = getelementptr inbounds %struct.Record, %struct.Record* %4, i64 0, i32 0
  store %struct.Record* %4, %struct.Record** %PtrComp6, align 8
  %inc = add nuw i32 %i.010, 1
  %cmp = icmp ult i32 %inc, %2
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret i32 0

; CHECK-LABEL: foo
; CHECK: addis [[REG1:[0-9]+]], 2, a@toc@ha
; CHECK: li [[REG4:[0-9]+]], 5
; CHECK: [[LAB:[a-z0-9A-Z_.]+]]:
; CHECK: ld [[REG2:[0-9]+]], a@toc@l([[REG1]])
; CHECK: stw [[REG4]], 8([[REG2]])
; CHECK: ld [[REG3:[0-9]+]], 0([[REG2]])
; CHECK: stw [[REG4]], 8([[REG3]]) 
; CHECK: std [[REG3]], 0([[REG3]])
; CHECK: bdnz [[LAB]]
}

