; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu %s -o - -enable-shrink-wrap=false |  FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
;
; Note: Lots of tests use inline asm instead of regular calls.
; This allows to have a better control on what the allocation will do.
; Otherwise, we may have spill right in the entry block, defeating
; shrink-wrapping. Moreover, some of the inline asm statement (nop)
; are here to ensure that the related paths do not end up as critical
; edges.


; Initial motivating example: Simple diamond with a call just on one side.
; CHECK-LABEL: foo:
;
; Compare the arguments and return
; No prologue needed.
; ENABLE: cmpw 0, 3, 4
; ENABLE-NEXT: bgelr 0
;
; Prologue code.
;  At a minimum, we save/restore the link register. Other registers may be saved
;  as well. 
; CHECK: mflr 
;
; Compare the arguments and jump to exit.
; After the prologue is set.
; DISABLE: cmpw 0, 3, 4
; DISABLE-NEXT: bge 0, .[[EXIT_LABEL:LBB[0-9_]+]]
;
; Store %a on the stack
; CHECK: stw 3, {{[0-9]+([0-9]+)}}
; Set the alloca address in the second argument.
; CHECK-NEXT: addi 4, 1, {{[0-9]+}}
; Set the first argument to zero.
; CHECK-NEXT: li 3, 0
; CHECK-NEXT: bl doSomething
;
; With shrink-wrapping, epilogue is just after the call.
; Restore the link register and return.
; Note that there could be other epilog code before the link register is 
; restored but we will not check for it here.
; ENABLE: mtlr
; ENABLE-NEXT: blr
;
; DISABLE: [[EXIT_LABEL]]:
;
; Without shrink-wrapping, epilogue is in the exit block.
; Epilogue code. (What we pop does not matter.)
; DISABLE: mtlr 0
; DISABLE-NEXT: blr
;

define i32 @foo(i32 %a, i32 %b) {
  %tmp = alloca i32, align 4
  %tmp2 = icmp slt i32 %a, %b
  br i1 %tmp2, label %true, label %false

true:
  store i32 %a, i32* %tmp, align 4
  %tmp4 = call i32 @doSomething(i32 0, i32* %tmp)
  br label %false

false:
  %tmp.0 = phi i32 [ %tmp4, %true ], [ %a, %0 ]
  ret i32 %tmp.0
}

; Function Attrs: optsize
declare i32 @doSomething(i32, i32*)



; Check that we do not perform the restore inside the loop whereas the save
; is outside.
; CHECK-LABEL: freqSaveAndRestoreOutsideLoop:
;
; Shrink-wrapping allows to skip the prologue in the else case.
; ENABLE: cmplwi 0, 3, 0
; ENABLE: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the link register
; CHECK: mflr 0
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
;
; Next BB.
; CHECK: slwi 3, [[SUM]], 3
;
; Jump to epilogue.
; DISABLE: b .[[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; DISABLE: slwi 3, 4, 1
; DISABLE: .[[EPILOG_BB]]: # %if.end
;
; Epilogue code.
; CHECK: mtlr 0
; CHECK-NEXT: blr
;
; ENABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; ENABLE: slwi 3, 4, 1
; ENABLE-NEXT: blr
define i32 @freqSaveAndRestoreOutsideLoop(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  %sum.04 = phi i32 [ %add, %for.body ], [ 0, %for.preheader ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

declare i32 @something(...)

; Check that we do not perform the shrink-wrapping inside the loop even
; though that would be legal. The cost model must prevent that.
; CHECK-LABEL: freqSaveAndRestoreOutsideLoop2:
; Prologue code.
; Make sure we save the link register before the call
; CHECK: mflr 0
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
;
; Next BB
; CHECK: %for.exit
; CHECK: mtlr 0
; CHECK-NEXT: blr
define i32 @freqSaveAndRestoreOutsideLoop2(i32 %cond) {
entry:
  br label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %for.preheader ], [ %inc, %for.body ]
  %sum.03 = phi i32 [ 0, %for.preheader ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.03
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.exit, label %for.body

for.exit:
  tail call void asm "nop", ""()
  br label %for.end

for.end:                                          ; preds = %for.body
  ret i32 %add
}


; Check with a more complex case that we do not have save within the loop and
; restore outside.
; CHECK-LABEL: loopInfoSaveOutsideLoop:
;
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the link register 
; CHECK: mflr 0
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
; 
; Next BB
; CHECK: bl somethingElse 
; CHECK: slwi 3, [[SUM]], 3
;
; Jump to epilogue
; DISABLE: b .[[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; DISABLE: slwi 3, 4, 1
;
; DISABLE: .[[EPILOG_BB]]: # %if.end
; Epilog code
; CHECK: mtlr 0
; CHECK-NEXT: blr
; 
; ENABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; ENABLE: slwi 3, 4, 1
; ENABLE-NEXT: blr
define i32 @loopInfoSaveOutsideLoop(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  %sum.04 = phi i32 [ %add, %for.body ], [ 0, %for.preheader ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  tail call void bitcast (void (...)* @somethingElse to void ()*)()
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

declare void @somethingElse(...)

; Check with a more complex case that we do not have restore within the loop and
; save outside.
; CHECK-LABEL: loopInfoRestoreOutsideLoop:
;
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the link register
; CHECK: mflr 0
;
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; CHECK: bl somethingElse
;
; Loop preheader
; CHECK-DAG: li [[SUM:[0-9]+]], 0
; CHECK-DAG: li [[IV:[0-9]+]], 10
; 
; Loop body
; CHECK: .[[LOOP:LBB[0-9_]+]]: # %for.body
; CHECK: bl something
; CHECK-DAG: addi [[IV]], [[IV]], -1
; CHECK-DAG: add [[SUM]], 3, [[SUM]] 
; CHECK-NEXT: cmplwi [[IV]], 0
; CHECK-NEXT: bne 0, .[[LOOP]]
;
; Next BB. 
; slwi 3, [[SUM]], 3
;
; DISABLE: b .[[EPILOG_BB:LBB[0-9_]+]]
;
; DISABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; DISABLE: slwi 3, 4, 1
; DISABLE: .[[EPILOG_BB]]: # %if.end
;
; Epilogue code.
; CHECK: mtlr 0
; CHECK-NEXT: blr
;
; ENABLE: .[[ELSE_LABEL]]: # %if.else
; Shift second argument by one and store into returned register.
; ENABLE: slwi 3, 4, 1
; ENABLE-NEXT: blr
define i32 @loopInfoRestoreOutsideLoop(i32 %cond, i32 %N) #0 {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @somethingElse to void ()*)()
  br label %for.body

for.body:                                         ; preds = %for.body, %if.then
  %i.05 = phi i32 [ 0, %if.then ], [ %inc, %for.body ]
  %sum.04 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.04
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %shl = shl i32 %add, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %for.end
  %sum.1 = phi i32 [ %shl, %for.end ], [ %mul, %if.else ]
  ret i32 %sum.1
}

; Check that we handle function with no frame information correctly.
; CHECK-LABEL: emptyFrame:
; CHECK: # %entry
; CHECK-NEXT: li 3, 0
; CHECK-NEXT: blr
define i32 @emptyFrame() {
entry:
  ret i32 0
}


; Check that we handle inline asm correctly.
; CHECK-LABEL: inlineAsm:
;
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; Make sure we save the CSR used in the inline asm: r14
; ENABLE-DAG: li [[IV:[0-9]+]], 10
; ENABLE-DAG: std 14, -[[STACK_OFFSET:[0-9]+]](1) # 8-byte Folded Spill
;
; DISABLE: std 14, -[[STACK_OFFSET:[0-9]+]](1) # 8-byte Folded Spill
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
; DISABLE: li [[IV:[0-9]+]], 10
;
; CHECK: nop
; CHECK: mtctr [[IV]]
;
; CHECK: .[[LOOP_LABEL:LBB[0-9_]+]]: # %for.body
; Inline asm statement.
; CHECK: addi 14, 14, 1
; CHECK: bdnz .[[LOOP_LABEL]]
;
; Epilogue code.
; CHECK: li 3, 0
; CHECK-DAG: ld 14, -[[STACK_OFFSET]](1) # 8-byte Folded Reload
; CHECK: nop
; CHECK: blr
;
; CHECK: [[ELSE_LABEL]]
; CHECK-NEXT: slwi 3, 4, 1
; DISABLE: ld 14, -[[STACK_OFFSET]](1) # 8-byte Folded Reload
; CHECK-NEXT blr
; 
define i32 @inlineAsm(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %for.preheader

for.preheader:
  tail call void asm "nop", ""()
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %for.preheader ]
  tail call void asm "addi 14, 14, 1", "~{r14}"()
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10
  br i1 %exitcond, label %for.exit, label %for.body

for.exit:
  tail call void asm "nop", ""()
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %for.body, %if.else
  %sum.0 = phi i32 [ %mul, %if.else ], [ 0, %for.exit ]
  ret i32 %sum.0
}


; Check that we handle calls to variadic functions correctly.
; CHECK-LABEL: callVariadicFunc:
;
; ENABLE: cmplwi 0, 3, 0
; ENABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Prologue code.
; CHECK: mflr 0
; 
; DISABLE: cmplwi 0, 3, 0
; DISABLE-NEXT: beq 0, .[[ELSE_LABEL:LBB[0-9_]+]]
;
; Setup of the varags.
; CHECK: mr 4, 3
; CHECK-NEXT: mr 5, 3
; CHECK-NEXT: mr 6, 3
; CHECK-NEXT: mr 7, 3
; CHECK-NEXT: mr 8, 3
; CHECK-NEXT: mr 9, 3
; CHECK-NEXT: bl someVariadicFunc
; CHECK: slwi 3, 3, 3
; DISABLE: b .[[EPILOGUE_BB:LBB[0-9_]+]]
;
; ENABLE: mtlr 0
; ENABLE-NEXT: blr
;
; CHECK: .[[ELSE_LABEL]]: # %if.else
; CHECK-NEXT: slwi 3, 4, 1
; 
; DISABLE: .[[EPILOGUE_BB]]: # %if.end
; DISABLE: mtlr
; CHECK: blr
define i32 @callVariadicFunc(i32 %cond, i32 %N) {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i32 (i32, ...) @someVariadicFunc(i32 %N, i32 %N, i32 %N, i32 %N, i32 %N, i32 %N, i32 %N)
  %shl = shl i32 %call, 3
  br label %if.end

if.else:                                          ; preds = %entry
  %mul = shl nsw i32 %N, 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %sum.0 = phi i32 [ %shl, %if.then ], [ %mul, %if.else ]
  ret i32 %sum.0
}

declare i32 @someVariadicFunc(i32, ...)



; Make sure we do not insert unreachable code after noreturn function.
; Although this is not incorrect to insert such code, it is useless
; and it hurts the binary size.
;
; CHECK-LABEL: noreturn:
; DISABLE: mflr 0
;
; CHECK: cmplwi 3, 0
; CHECK-NEXT: bne 0, .[[ABORT:LBB[0-9_]+]]
;
; CHECK: li 3, 42
;
; DISABLE: mtlr 0
;
; CHECK-NEXT: blr
;
; CHECK: .[[ABORT]]: # %if.abort
;
; ENABLE: mflr 0
;
; CHECK: bl abort
; ENABLE-NOT: mtlr 0
define i32 @noreturn(i8 signext %bad_thing) {
entry:
  %tobool = icmp eq i8 %bad_thing, 0
  br i1 %tobool, label %if.end, label %if.abort

if.abort:
  tail call void @abort() #0
  unreachable

if.end:
  ret i32 42
}

declare void @abort() #0

attributes #0 = { noreturn nounwind }


; Make sure that we handle infinite loops properly When checking that the Save
; and Restore blocks are control flow equivalent, the loop searches for the
; immediate (post) dominator for the (restore) save blocks. When either the Save
; or Restore block is located in an infinite loop the only immediate (post)
; dominator is itself. In this case, we cannot perform shrink wrapping, but we
; should return gracefully and continue compilation.
; The only condition for this test is the compilation finishes correctly.
; 
; CHECK-LABEL: infiniteloop
; CHECK: blr
define void @infiniteloop() {
entry:
  br i1 undef, label %if.then, label %if.end

if.then:
  %ptr = alloca i32, i32 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %sum.03 = phi i32 [ 0, %if.then ], [ %add, %for.body ]
  %call = tail call i32 bitcast (i32 (...)* @something to i32 ()*)()
  %add = add nsw i32 %call, %sum.03
  store i32 %add, i32* %ptr
  br label %for.body

if.end:
  ret void
}
