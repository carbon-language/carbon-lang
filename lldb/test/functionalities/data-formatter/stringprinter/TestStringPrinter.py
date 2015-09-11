import lldbinline

lldbinline.MakeInlineTest(__file__, globals(), [lldbtest.expectedFailureWindows("llvm.org/pr24772")])
