; RUN: sh -c 'opt --reject-this-option 2>&-; echo $?; opt -o /dev/null /dev/null 2>&-; echo $?;' \
; RUN:   | FileCheck %s
; CHECK: {{^70$}}
; CHECK: {{^0$}}
; XFAIL: vg_leak
; REQUIRES: shell

; On MSYS, 70 is not seen, but 1. r127726 should be reworked.
; XFAIL: mingw32

; Test that the error handling when writing to stderr fails exits the
; program cleanly rather than aborting.
