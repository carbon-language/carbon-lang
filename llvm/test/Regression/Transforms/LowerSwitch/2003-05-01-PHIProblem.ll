; RUN: as < %s | opt -lowerswitch

void %child(int %ct.1) {
entry:          ; No predecessors!
        switch uint 0, label %return [
                 uint 2, label %UnifiedExitNode
                 uint 3, label %UnifiedExitNode
                 uint 0, label %return
                 uint 1, label %UnifiedExitNode
        ]

return:         ; preds = %entry, %entry
        %result.0 = phi %struct.quad_struct* [ null, %entry ], [ null, %entry ]         ; <%struct.quad_struct*> [#uses=0]
        br label %UnifiedExitNode

UnifiedExitNode:                ; preds = %entry, %return, %entry, %entry
        ret void
}

