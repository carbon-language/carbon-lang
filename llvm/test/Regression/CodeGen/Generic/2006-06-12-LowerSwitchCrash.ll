; RUN: llvm-as < %s | llc -fast

float %test(uint %tmp12771278) {
        switch uint %tmp12771278, label %bb1279 [
        ]

bb1279:         ; preds = %cond_next1272
        ret float 1.0
}

