; RUN: sed -e s/.T1:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s
; RUN: sed -e s/.T3:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK3 %s
; RUN: sed -e s/.T4:// %s | not llvm-as -disable-output 2>&1 | FileCheck --check-prefix=CHECK4 %s
; RUN: sed -e s/.T5:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK5 %s
; RUN: sed -e s/.T6:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK6 %s
; RUN: sed -e s/.T7:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK7 %s
; RUN: sed -e s/.T8:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK8 %s
; RUN: sed -e s/.T9:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK9 %s
; RUN: sed -e s/.T10:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK10 %s
; RUN: sed -e s/.T11:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK11 %s
; RUN: sed -e s/.T12:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK12 %s
; RUN: sed -e s/.T13:// %s | not opt -verify -disable-output 2>&1 | FileCheck --check-prefix=CHECK13 %s

declare void @g()

;T1: define void @f() {
;T1:   entry:
;T1:     catchret from undef to label %next
;T1:     ; CHECK1: CatchReturnInst needs to be provided a CatchPad
;T1:   next:
;T1:     unreachable
;T1: }

;T2: define void @f() {
;T2:   entry:
;T2:     %x = cleanuppad within none []
;T2:     ; catchret's first operand's operator must be catchpad
;T2:     catchret from %x to label %entry
;T2:     ; CHECK2: CatchReturnInst needs to be provided a CatchPad
;T2: }

;T3: define void @f() {
;T3:   entry:
;T3:     cleanupret from undef unwind label %next
;T3:     ; CHECK3: CleanupReturnInst needs to be provided a CleanupPad
;T3:   next:
;T3:     unreachable
;T3: }

;T4: define void @f() {
;T4:   entry:
;T4:     %cs = catchswitch within none [label %next] unwind to caller
;T4:   next:
;T4:     %x = catchpad within %cs []
;T4:     ; cleanupret first operand's operator must be cleanuppad
;T4:     cleanupret from %x unwind to caller
;T4:     ; CHECK4: CleanupReturnInst needs to be provided a CleanupPad
;T4: }

;T5: define void @f() personality void ()* @g {
;T5:   entry:
;T5:     ret void
;T5:   switch:
;T5:     %cs = catchswitch within none [label %catch] unwind to caller
;T5:   catch:
;T5:     catchpad within %cs []
;T5:     unreachable
;T5:   bogus:
;T5:     cleanuppad within %cs []
;T5:     ; CHECK5: CleanupPadInst has an invalid parent
;T5:     unreachable
;T5: }

;T6: define void @f() personality void ()* @g {
;T6:   entry:
;T6:     ret void
;T6:   switch1:
;T6:     %cs1 = catchswitch within none [label %catch1] unwind label %catch2
;T6:     ; CHECK6: Block containg CatchPadInst must be jumped to only by its catchswitch
;T6:   catch1:
;T6:     catchpad within %cs1 []
;T6:     unreachable
;T6:   switch2:
;T6:     %cs2 = catchswitch within none [label %catch2] unwind to caller
;T6:   catch2:
;T6:     catchpad within %cs2 []
;T6:     unreachable
;T6: }

;T7: define void @f() personality void ()* @g {
;T7:   entry:
;T7:     ret void
;T7:   switch1:
;T7:     %cs1 = catchswitch within none [label %catch1] unwind to caller
;T7:   catch1:
;T7:     catchpad within %cs1 []
;T7:     unreachable
;T7:   switch2:
;T7:     %cs2 = catchswitch within %cs1 [label %catch2] unwind to caller
;T7:     ; CHECK7: CatchSwitchInst has an invalid parent
;T7:   catch2:
;T7:     catchpad within %cs2 []
;T7:     unreachable
;T7: }

;T8: define void @f() personality void ()* @g {
;T8:   entry:
;T8:     ret void
;T8:   switch1:
;T8:     %cs1 = catchswitch within none [ label %switch1 ] unwind to caller
;T8:     ; CHECK8: CatchSwitchInst handlers must be catchpads
;T8: }

;T9: define void @f() personality void ()* @g {
;T9:   entry:
;T9:     ret void
;T9:   cleanup:
;T9:     %cp = cleanuppad within none []
;T9:     invoke void @g() [ "funclet"(token %cp) ]
;T9:       to label %exit unwind label %cleanup
;T9:       ; CHECK9: EH pad cannot handle exceptions raised within it
;T9:       ; CHECK9-NEXT: %cp = cleanuppad within none []
;T9:       ; CHECK9-NEXT: invoke void @g() [ "funclet"(token %cp) ]
;T9:   exit:
;T9:     ret void
;T9: }

;T10: define void @f() personality void ()* @g {
;T10:   entry:
;T10:     ret void
;T10:   cleanup1:
;T10:     %cp1 = cleanuppad within none []
;T10:     unreachable
;T10:   switch:
;T10:     %cs = catchswitch within %cp1 [label %catch] unwind to caller
;T10:   catch:
;T10:     %catchp1 = catchpad within %cs [i32 1]
;T10:     unreachable
;T10:   cleanup2:
;T10:     %cp2 = cleanuppad within %catchp1 []
;T10:     unreachable
;T10:   cleanup3:
;T10:     %cp3 = cleanuppad within %cp2 []
;T10:     cleanupret from %cp3 unwind label %switch
;T10:       ; CHECK10: EH pad cannot handle exceptions raised within it
;T10:       ; CHECK10-NEXT: %cs = catchswitch within %cp1 [label %catch] unwind to caller
;T10:       ; CHECK10-NEXT: cleanupret from %cp3 unwind label %switch
;T10: }

;T11: define void @f() personality void ()* @g {
;T11:   entry:
;T11:     ret void
;T11:   cleanup1:
;T11:     %cp1 = cleanuppad within none []
;T11:     unreachable
;T11:   cleanup2:
;T11:     %cp2 = cleanuppad within %cp1 []
;T11:     unreachable
;T11:   switch:
;T11:     %cs = catchswitch within none [label %catch] unwind label %cleanup2
;T11:     ; CHECK11: A single unwind edge may only enter one EH pad
;T11:     ; CHECK11-NEXT: %cs = catchswitch within none [label %catch] unwind label %cleanup2
;T11:   catch:
;T11:     catchpad within %cs [i32 1]
;T11:     unreachable
;T11: }

;T12: define void @f() personality void ()* @g {
;T12:   entry:
;T12:     ret void
;T12:   cleanup:
;T12:     %cp = cleanuppad within none []
;T12:     cleanupret from %cp unwind label %switch
;T12:     ; CHECK12: A cleanupret must exit its cleanup
;T12:     ; CHECK12-NEXT: cleanupret from %cp unwind label %switch
;T12:   switch:
;T12:     %cs = catchswitch within %cp [label %catch] unwind to caller
;T12:   catch:
;T12:     catchpad within %cs [i32 1]
;T12:     unreachable
;T12: }

;T13: define void @f() personality void ()* @g {
;T13:   entry:
;T13:     ret void
;T13:   switch:
;T13:     %cs = catchswitch within none [label %catch] unwind label %switch
;T13:     ; CHECK13: EH pad cannot handle exceptions raised within it
;T13:     ; CHECK13-NEXT:  %cs = catchswitch within none [label %catch] unwind label %switch
;T13:   catch:
;T13:     catchpad within %cs [i32 0]
;T13:     unreachable
;T13: }

