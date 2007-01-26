; RUN: llvm-as 2>&1 < %s -o /dev/null -f | \
; RUN:   grep 'Cannot create a null initialized value of this type!'
; Test for PR463.  This program is erroneous, but should not crash llvm-as.
@.FOO  = internal global %struct.none zeroinitializer
