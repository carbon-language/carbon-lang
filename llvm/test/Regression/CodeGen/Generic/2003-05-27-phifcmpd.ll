; RUN: llvm-as -f %s -o - | llc

void %QRiterate(int %p.1, double %tmp.212) { 
entry:          ; No predecessors!
        %tmp.184 = setgt int %p.1, 0            ; <bool> [#uses=1]
        br bool %tmp.184, label %shortcirc_next.1, label %shortcirc_done.1

shortcirc_next.1:               ; preds = %entry
        %tmp.213 = setne double %tmp.212, 0.000000e+00
        br label %shortcirc_done.1

shortcirc_done.1:               ; preds = %entry, %shortcirc_next.1
        %val.1 = phi bool [ false, %entry ], [ %tmp.213, %shortcirc_next.1 ]
        br bool %val.1, label %shortcirc_next.1, label %exit.1

exit.1:
	ret void
}
