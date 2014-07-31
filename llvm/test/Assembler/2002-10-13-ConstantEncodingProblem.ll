; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s -preserve-bc-use-list-order

%Domain = type { %Domain**, %Domain* }
@D = global %Domain zeroinitializer             ; <%Domain*> [#uses=0]

