; Test fastcc works. Test from bug 2770.
; RUN: llc < %s -relocation-model=pic


%struct.__gcov_var = type {  i32 }
@__gcov_var = external global %struct.__gcov_var

define fastcc void @gcov_read_words(i32 %words) {
entry:
        store i32 %words, i32* getelementptr (%struct.__gcov_var, %struct.__gcov_var* 
@__gcov_var,
i32 0, i32 0)
        ret void
}
