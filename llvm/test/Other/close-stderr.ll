; RUN: sh -c "\
; RUN:        opt --reject-this-option 2>&-; echo \$?; \
; RUN:        opt -o /dev/null /dev/null 2>&-; echo \$?; \
; RUN:       " | FileCheck %s
; CHECK: {{^1$}}
; CHECK: {{^0$}}

; Test that the error handling when writing to stderr fails exits the
; program cleanly rather than aborting.
