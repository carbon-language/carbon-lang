; This test checks to make sure that constant exprs don't fold in some simple
; situations

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Even give it a datalayout, to tempt folding as much as possible.
target datalayout = "p:32:32"

@A = global i64 0
@B = global i64 0

; Don't fold this. @A might really be allocated next to @B, in which case the
; icmp should return true. It's not valid to *dereference* in @B from a pointer
; based on @A, but icmp isn't a dereference.

; CHECK: @C = global i1 icmp eq (i64* getelementptr inbounds (i64* @A, i64 1), i64* @B)
@C = global i1 icmp eq (i64* getelementptr inbounds (i64* @A, i64 1), i64* @B)

; Don't fold this completely away either. In theory this could be simplified
; to only use a gep on one side of the icmp though.

; CHECK: @D = global i1 icmp eq (i64* getelementptr inbounds (i64* @A, i64 1), i64* getelementptr inbounds (i64* @B, i64 2))
@D = global i1 icmp eq (i64* getelementptr inbounds (i64* @A, i64 1), i64* getelementptr inbounds (i64* @B, i64 2))

; CHECK: @E = global i64 addrspace(1)* addrspacecast (i64* @A to i64 addrspace(1)*)
@E = global i64 addrspace(1)* addrspacecast(i64* @A to i64 addrspace(1)*)
