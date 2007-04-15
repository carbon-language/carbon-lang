; RUN: ignore llvm-as < %s > /dev/null |& \
; RUN:    grep {Cannot form a pointer to a basic block}

define i32 @main() {
         %foo  = call i8* %llvm.stacksave()
         %foop = bitcast i8* %foo to label*
         %nret = load label* %foop
         br label %nret;
}
