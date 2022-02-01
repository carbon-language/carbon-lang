; NOTE: This test case generates a jump table on PowerPC big and little endian
; NOTE: then verifies that the command line option to enable absolute jump
; NOTE: table works correctly.
; RUN:  llc -mtriple=powerpc64le-unknown-linux-gnu -o - \
; RUN:      -ppc-use-absolute-jumptables -ppc-asm-full-reg-names \
; RUN:      -verify-machineinstrs %s | FileCheck %s -check-prefix=CHECK-LE
; RUN:  llc -mtriple=powerpc64-unknown-linux-gnu -o - \
; RUN:      -ppc-use-absolute-jumptables -ppc-asm-full-reg-names \
; RUN:      -verify-machineinstrs %s | FileCheck %s -check-prefix=CHECK-BE

%struct.node = type { i8, %struct.node* }

; Function Attrs: norecurse nounwind readonly
define zeroext i32 @jumpTableTest(%struct.node* readonly %list) {
; CHECK-LE-LABEL: jumpTableTest:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE:       rldic r[[REG:[0-9]+]], r[[REG]], 3, 29
; CHECK-LE:       ldx r[[REG]], r[[REG]], r[[REG1:[0-9]+]]
; CHECK-LE:       mtctr r[[REG]]
; CHECK-LE:       bctr
; CHECK-LE:       blr
;
; CHECK-BE-LABEL: jumpTableTest:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE:       rldic r[[REG:[0-9]+]], r[[REG]], 2, 30
; CHECK-BE:       lwax r[[REG]], r[[REG]], r[[REG1:[0-9]+]]
; CHECK-BE:       mtctr r[[REG]]
; CHECK-BE:       bctr
; CHECK-BE:       blr
entry:
  %cmp36 = icmp eq %struct.node* %list, null
  br i1 %cmp36, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %sw.epilog
  %result.038 = phi i32 [ %result.1, %sw.epilog ], [ 0, %entry ]
  %current.037 = phi %struct.node* [ %spec.store.select, %sw.epilog ], [ %list, %entry ]
  %next1 = getelementptr inbounds %struct.node, %struct.node* %current.037, i64 0, i32 1
  %0 = load %struct.node*, %struct.node** %next1, align 8
  %cmp2 = icmp eq %struct.node* %0, %current.037
  %spec.store.select = select i1 %cmp2, %struct.node* null, %struct.node* %0
  %type = getelementptr inbounds %struct.node, %struct.node* %current.037, i64 0, i32 0
  %1 = load i8, i8* %type, align 8
  switch i8 %1, label %sw.epilog [
    i8 1, label %sw.bb
    i8 2, label %sw.bb3
    i8 3, label %sw.bb5
    i8 4, label %sw.bb7
    i8 5, label %sw.bb9
    i8 6, label %sw.bb11
    i8 7, label %sw.bb13
    i8 8, label %sw.bb15
    i8 9, label %sw.bb17
  ]

sw.bb:                                            ; preds = %while.body
  %add = add nsw i32 %result.038, 13
  br label %sw.epilog

sw.bb3:                                           ; preds = %while.body
  %add4 = add nsw i32 %result.038, 5
  br label %sw.epilog

sw.bb5:                                           ; preds = %while.body
  %add6 = add nsw i32 %result.038, 2
  br label %sw.epilog

sw.bb7:                                           ; preds = %while.body
  %add8 = add nsw i32 %result.038, 7
  br label %sw.epilog

sw.bb9:                                           ; preds = %while.body
  %add10 = add nsw i32 %result.038, 11
  br label %sw.epilog

sw.bb11:                                          ; preds = %while.body
  %add12 = add nsw i32 %result.038, 17
  br label %sw.epilog

sw.bb13:                                          ; preds = %while.body
  %add14 = add nsw i32 %result.038, 16
  br label %sw.epilog

sw.bb15:                                          ; preds = %while.body
  %add16 = add nsw i32 %result.038, 81
  br label %sw.epilog

sw.bb17:                                          ; preds = %while.body
  %add18 = add nsw i32 %result.038, 72
  br label %sw.epilog

sw.epilog:                                        ; preds = %while.body, %sw.bb17, %sw.bb15, %sw.bb13, %sw.bb11, %sw.bb9, %sw.bb7, %sw.bb5, %sw.bb3, %sw.bb
  %result.1 = phi i32 [ %result.038, %while.body ], [ %add18, %sw.bb17 ], [ %add16, %sw.bb15 ], [ %add14, %sw.bb13 ], [ %add12, %sw.bb11 ], [ %add10, %sw.bb9 ], [ %add8, %sw.bb7 ], [ %add6, %sw.bb5 ], [ %add4, %sw.bb3 ], [ %add, %sw.bb ]
  %cmp = icmp eq %struct.node* %spec.store.select, null
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %sw.epilog, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %result.1, %sw.epilog ]
  ret i32 %result.0.lcssa
}

