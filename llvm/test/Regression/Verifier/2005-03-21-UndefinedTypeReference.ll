; RUN: not llvm-as -f %s -o /dev/null

void %test() {
        malloc %InvalidType
        ret void
}

