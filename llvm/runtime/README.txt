Date: Wed, 6 Nov 2002 14:05:32 -0600 (CST)
From: Chris Lattner <sabre@nondot.org>
To: LLVMdev List <llvmdev@cs.uiuc.edu>
Subject: What is test/Libraries?

Hey everyone,

There has recently been some confusion over what test/Libraries is and
what it is used for.  The short answer is: it's used when building GCC,
not for tests, so you all shouldn't need to use it.

test/Libraries contains the LLVM "system libraries", which are linked to
programs when the linker is run with the appropriate -l switch (for
example -lm links in the "math" library).  In general, these libraries are
just stubbed out libraries, because noone has had the time to do a full
glibc port to LLVM.

Problems arise because the makefiles have a number of hardcoded paths in
it that are used to copy files around and install the libraries, which
cause problems if anyone (except for me) uses them.  I'm sorry a better
system isn't in place yet for these libraries, but if you just ignore
them, they won't cause you any harm.  :)

-Chris

