import lldbinline
import lldbtest

lldbinline.MakeInlineTest(__file__, globals(), [lldbtest.expectedFailureWindows("llvm.org/pr24772")])
