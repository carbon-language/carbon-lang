; RUN: as < %s | opt -lowerswitch

void %child(int %ct.1) {
entry:          ; No predecessors!
        switch uint 0, label %return [
                 uint 3, label %UnifiedExitNode
                 uint 0, label %return
        ]

return:         ; preds = %entry, %entry
        %result.0 = phi int* [ null, %entry ], [ null, %entry ]         ; <%struct.quad_struct*> [#uses=0]
        br label %UnifiedExitNode

UnifiedExitNode:                ; preds = %entry, %return, %entry, %entry
        ret void
}

