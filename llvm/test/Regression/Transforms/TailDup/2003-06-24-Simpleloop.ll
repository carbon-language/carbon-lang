; RUN: as < %s | opt -tailduplicate -disable-output

void %motion_result7() {
entry:
        br label %endif

endif:
        %i.1 = phi int [ %inc, %no_exit ], [ 0, %entry ]
        %inc = add int %i.1, 1
        br bool false, label %no_exit, label %UnifiedExitNode

no_exit:
        br bool false, label %UnifiedExitNode, label %endif

UnifiedExitNode:
        ret void
}

