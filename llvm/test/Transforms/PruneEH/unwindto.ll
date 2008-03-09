; RUN: llvm-as < %s | opt -prune-eh | llvm-dis | not grep unwind_to

define i8 @test7(i1 %b) {
entry: unwind_to %cleanup
  br i1 %b, label %cond_true, label %cond_false
cond_true: unwind_to %cleanup
  br label %cleanup
cond_false: unwind_to %cleanup
  br label %cleanup
cleanup:
  %x = phi i8 [0, %entry], [1, %cond_true], [1, %cond_true],
                           [2, %cond_false], [2, %cond_false]
  ret i8 %x
}

