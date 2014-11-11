import lldbinline
import lldbtest

lldbinline.MakeInlineTest(__file__, globals(), [lldbtest.skipIfFreeBSD,lldbtest.skipIfLinux,lldbtest.skipIfWindows])
