; %inc2 uses it's own value, but that's ok, as it's unreachable!

void %test() {
entry:
        ret void

no_exit.2:              ; preds = %endif.6
        %tmp.103 = setlt double 0x0, 0x0                ; <bool> [#uses=1]
        br bool %tmp.103, label %endif.6, label %else.0

else.0:         ; preds = %no_exit.2
        store ushort 0, ushort* null
        br label %endif.6

endif.6:                ; preds = %no_exit.2, %else.0
        %inc.2 = add int %inc.2, 1              ; <int> [#uses=2]
        %tmp.96 = setlt int %inc.2, 0           ; <bool> [#uses=1]
        br bool %tmp.96, label %no_exit.2, label %UnifiedReturnBlock1

UnifiedReturnBlock1:            ; preds = %endif.6
        ret void
}

