; RUN: llvm-upgrade %s | not llvm-as -o /dev/null -f &&
; RUN: llvm-upgrade %s | llvm-as -o /dev/null -f 2>&1 | grep 'Cannot form'

int %main() {
         %foo  = call sbyte* %llvm.stacksave()
         %foop = cast sbyte* %foo to label*
         %nret = load label* %foop
         br label %nret;
}
