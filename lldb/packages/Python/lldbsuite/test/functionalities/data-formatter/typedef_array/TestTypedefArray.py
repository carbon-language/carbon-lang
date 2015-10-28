import lldbinline
import lldbtest

lldbinline.MakeInlineTest(__file__, globals(), [lldbtest.expectedFailureGcc])
