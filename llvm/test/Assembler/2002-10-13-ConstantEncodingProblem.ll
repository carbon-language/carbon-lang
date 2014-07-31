; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s -preserve-bc-use-list-order -num-shuffles=5

%Domain = type { %Domain**, %Domain* }
@D = global %Domain zeroinitializer             ; <%Domain*> [#uses=0]

