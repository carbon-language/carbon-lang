; RUN: opt -basicaa -aa-eval -print-all-alias-modref-info -disable-output < %s |& FileCheck  %s

@global = internal global i32 0

declare void @should_not_be_called()
declare i32 @f()

; CHECK: Function: g: 2 pointers, 0 call sites
define void @g(i32* nocapture %p) {
  store i32 0, i32* @global

  ; @global is internal, is only used in this function, and never has its
  ; address taken so it can't alias p.
  ; CHECK: NoAlias:	i32* %p, i32* @global
  store i32 1, i32* %p
  %1 = load i32* @global
  ret void
}

