; RUN: sh -c 'opt --reject-this-option 2>&-; echo $?; opt -o /dev/null /dev/null 2>&-; echo $?;' \
; RUN:   | FileCheck %s

; CHECK: {{^1$}}
; On valgrind, we got 127 here.
; XFAIL: valgrind

; CHECK: {{^0$}}
; XFAIL: vg_leak
; REQUIRES: shell

; FIXME: sys::fs::status() doesn't work on pipes, since r186560.
; XFAIL: mingw32

; opt will fail to open /dev/null on native win32.
; XFAIL: win32

; Test that the error handling when writing to stderr fails exits the
; program cleanly rather than aborting.
