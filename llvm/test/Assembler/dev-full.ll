; RUN: not llvm-as < %s > /dev/full 2>/dev/null

; raw_ostream should detect write errors.
