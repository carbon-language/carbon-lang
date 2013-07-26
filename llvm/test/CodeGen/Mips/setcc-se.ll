; RUN: llc -march=mipsel < %s | FileCheck %s

@g1 = external global i32

; CHECK-LABEL: seteq0:
; CHECK: sltiu ${{[0-9]+}}, $4, 1

define i32 @seteq0(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: setne0:
; CHECK: sltu ${{[0-9]+}}, $zero, $4

define i32 @setne0(i32 %a) {
entry:
  %cmp = icmp ne i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: slti_beq0:
; CHECK: slti $[[R0:[0-9]+]], $4, -32768
; CHECK: beqz $[[R0]]

define void @slti_beq0(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, -32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: slti_beq1:
; CHECK: slt ${{[0-9]+}}

define void @slti_beq1(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, -32769
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: slti_beq2:
; CHECK: slti $[[R0:[0-9]+]], $4, 32767
; CHECK: beqz $[[R0]]

define void @slti_beq2(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 32767
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: slti_beq3:
; CHECK: slt ${{[0-9]+}}

define void @slti_beq3(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq0:
; CHECK: sltiu $[[R0:[0-9]+]], $4, 32767
; CHECK: beqz $[[R0]]

define void @sltiu_beq0(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, 32767
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq1:
; CHECK: sltu ${{[0-9]+}}

define void @sltiu_beq1(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, 32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq2:
; CHECK: sltiu $[[R0:[0-9]+]], $4, -32768
; CHECK: beqz $[[R0]]

define void @sltiu_beq2(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, -32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq3:
; CHECK: sltu ${{[0-9]+}}

define void @sltiu_beq3(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, -32769
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, i32* @g1, align 4
  br label %if.end

if.end:
  ret void
}
