; RUN: opt -S -objc-arc < %s | FileCheck %s

declare i8* @objc_retain(i8*) nonlazybind
declare void @objc_release(i8*) nonlazybind
declare i8* @objc_retainBlock(i8*)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Use by an instruction which copies the value is an escape if the             ;
; result is an escape. The current instructions with this property are:        ;
;                                                                              ;
; 1. BitCast.                                                                  ;
; 2. GEP.                                                                      ;
; 3. PhiNode.                                                                  ;
; 4. SelectInst.                                                               ;
;                                                                              ;
; Make sure that such instructions do not confuse the optimizer into removing  ;
; an objc_retainBlock that is needed.                                          ;
;                                                                              ;
; rdar://13273675. (With extra test cases to handle bitcast, phi, and select.  ;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define void @bitcasttest(i8* %storage, void (...)* %block)  {
; CHECK: define void @bitcasttest
entry:
  %t1 = bitcast void (...)* %block to i8*
; CHECK-NOT: tail call i8* @objc_retain
  %t2 = tail call i8* @objc_retain(i8* %t1)
; CHECK: tail call i8* @objc_retainBlock
  %t3 = tail call i8* @objc_retainBlock(i8* %t1), !clang.arc.copy_on_escape !0
  %t4 = bitcast i8* %storage to void (...)**
  %t5 = bitcast i8* %t3 to void (...)*
  store void (...)* %t5, void (...)** %t4, align 8
; CHECK-NOT: call void @objc_release
  call void @objc_release(i8* %t1)
  ret void
}

define void @geptest(void (...)** %storage_array, void (...)* %block)  {
; CHECK: define void @geptest
entry:
  %t1 = bitcast void (...)* %block to i8*
; CHECK-NOT: tail call i8* @objc_retain
  %t2 = tail call i8* @objc_retain(i8* %t1)
; CHECK: tail call i8* @objc_retainBlock
  %t3 = tail call i8* @objc_retainBlock(i8* %t1), !clang.arc.copy_on_escape !0
  %t4 = bitcast i8* %t3 to void (...)*
  
  %storage = getelementptr inbounds void (...)** %storage_array, i64 0
  
  store void (...)* %t4, void (...)** %storage, align 8
; CHECK-NOT: call void @objc_release
  call void @objc_release(i8* %t1)
  ret void
}

define void @selecttest(void (...)** %store1, void (...)** %store2,
                        void (...)* %block) {
; CHECK: define void @selecttest
entry:
  %t1 = bitcast void (...)* %block to i8*
; CHECK-NOT: tail call i8* @objc_retain
  %t2 = tail call i8* @objc_retain(i8* %t1)
; CHECK: tail call i8* @objc_retainBlock
  %t3 = tail call i8* @objc_retainBlock(i8* %t1), !clang.arc.copy_on_escape !0
  %t4 = bitcast i8* %t3 to void (...)*
  %store = select i1 undef, void (...)** %store1, void (...)** %store2
  store void (...)* %t4, void (...)** %store, align 8
; CHECK-NOT: call void @objc_release
  call void @objc_release(i8* %t1)
  ret void
}

define void @phinodetest(void (...)** %storage1,
                         void (...)** %storage2,
                         void (...)* %block) {
; CHECK: define void @phinodetest
entry:
  %t1 = bitcast void (...)* %block to i8*
; CHECK-NOT: tail call i8* @objc_retain
  %t2 = tail call i8* @objc_retain(i8* %t1)
; CHECK: tail call i8* @objc_retainBlock
  %t3 = tail call i8* @objc_retainBlock(i8* %t1), !clang.arc.copy_on_escape !0
  %t4 = bitcast i8* %t3 to void (...)*
  br i1 undef, label %store1_set, label %store2_set

store1_set:
  br label %end

store2_set:
  br label %end

end:
  %storage = phi void (...)** [ %storage1, %store1_set ], [ %storage2, %store2_set]
  store void (...)* %t4, void (...)** %storage, align 8
; CHECK-NOT: call void @objc_release
  call void @objc_release(i8* %t1)
  ret void
}

!0 = metadata !{}
