; RUN: llvm-as < /dev/null | not opt --foo >& /dev/null

; there is no --foo
