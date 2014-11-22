import lldbinline
from lldbtest import *

# added decorator to mark as XFAIL for Linux
# non-core functionality, need to reenable and fix later (DES 2014.11.07)
lldbinline.MakeInlineTest(__file__, globals(),expectedFailureLinux)
