; RUN: opt -functionattrs --disable-nofree-inference=false -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for the "nofree" function attribute.
; We use FIXME's to indicate problems and missing attributes.

; Free functions
declare void @free(i8* nocapture) local_unnamed_addr #1
declare noalias i8* @realloc(i8* nocapture, i64) local_unnamed_addr #0
declare void @_ZdaPv(i8*) local_unnamed_addr #2


; TEST 1 (positive case)
; FIXME: missing "nofree"
; CHECK: Function Attrs: noinline norecurse nounwind readnone uwtable
; CHECK: define void @only_return() 
define void @only_return() #0 {
    ret void
}


; TEST 2 (nagative case)
; Only free
; void only_free(char* p) {
;    free(p);
; }

; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define void @only_free(i8* nocapture) local_unnamed_addr 
define void @only_free(i8* nocapture) local_unnamed_addr #0 {
    tail call void @free(i8* %0) #1
    ret void
}


; TEST 3 (nagative case)
; Free occurs in same scc.
; void free_in_scc1(char*p){
;    free_in_scc2(p);
; }
; void free_in_scc2(char*p){
;    free_in_scc1(p);
;    free(p);
; }


; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define void @free_in_scc1(i8* nocapture) local_unnamed_addr 
define void @free_in_scc1(i8* nocapture) local_unnamed_addr #0 {
  tail call void @free_in_scc2(i8* %0) #1
  ret void
}


; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define void @free_in_scc2(i8* nocapture) local_unnamed_addr 
define void @free_in_scc2(i8*) local_unnamed_addr #0 {
  tail call void @free_in_scc1(i8* %0)
  tail call void @free(i8* %0) #1
  ret void
}


; TEST 4 (positive case)
; Free doesn't occur.
; void mutual_recursion1(){
;    mutual_recursion2();
; }
; void mutual_recursion2(){
;     mutual_recursion1();
; }


; FIXME: missing "nofree"
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define void @mutual_recursion1() 
define void @mutual_recursion1() #0 {
  call void @mutual_recursion2()
  ret void
}

; FIXME: missing "nofree"
; CHECK: Function Attrs: noinline nounwind readnone uwtable
; CHECK: define void @mutual_recursion2() 
define void @mutual_recursion2() #0 {
  call void @mutual_recursion1()
  ret void
}


; TEST 5
; C++ delete operation (nagative case)
; void delete_op (char p[]){
;     delete [] p;
; }

; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define void @_Z9delete_opPc(i8*) local_unnamed_addr 
define void @_Z9delete_opPc(i8*) local_unnamed_addr #0 {
  %2 = icmp eq i8* %0, null
  br i1 %2, label %4, label %3

; <label>:3:                                      ; preds = %1
  tail call void @_ZdaPv(i8* nonnull %0) #2
  br label %4

; <label>:4:                                      ; preds = %3, %1
  ret void
}


; TEST 6 (negative case)
; Call realloc
; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define noalias i8* @call_realloc(i8* nocapture, i64) local_unnamed_addr 
define noalias i8* @call_realloc(i8*nocapture, i64) local_unnamed_addr #0 {
    %ret = tail call i8* @realloc(i8* %0, i64 %1) #2
    ret i8* %ret
}


; TEST 7 (positive case)
; Call function declaration with "nofree"

declare void @nofree_function() nofree readnone #0

; FIXME: missing "nofree"
; Function Attrs: noinline nounwind readnone uwtable
; CHECK: define void @call_nofree_function() 
define void @call_nofree_function() #0 {
    tail call void @nofree_function()
    ret void
}

; TEST 8 (nagative case)
; Call function declaration without "nofree"

declare void @maybe_free() #0


; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define void @call_maybe_free() 
define void @call_maybe_free() #0 {
    tail call void @maybe_free()
    ret void
}


; TEST 9 (nagative case)
; Call both of above functions

; CHECK: Function Attrs: noinline nounwind uwtable
; CHECK: define void @call_both() 
define void @call_both() #0 {
    tail call void @maybe_free()
    tail call void @nofree_function()
    ret void
}


attributes #0 = { nounwind uwtable noinline }
attributes #1 = { nounwind }
attributes #2 = { nobuiltin nounwind }
