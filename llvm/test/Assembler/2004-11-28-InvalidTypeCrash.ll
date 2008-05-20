; Test for PR463.  This program is erroneous, but should not crash llvm-as.
; RUN: not llvm-as %s -o /dev/null -f |& \
; RUN:   grep {Cannot create a null initialized value of this type}

@.FOO  = internal global %struct.none zeroinitializer
