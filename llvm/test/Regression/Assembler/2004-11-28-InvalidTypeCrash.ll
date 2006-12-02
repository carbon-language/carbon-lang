; RUN: (llvm-upgrade < %s | llvm-as -o /dev/null -f) 2>&1 | grep 'Cannot create a'
; Test for PR463.  This program is erroneous, but should not crash llvm-as.
%.FOO  = internal global %struct.none zeroinitializer
