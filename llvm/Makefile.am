#===-- Makefile.am - Common make rules for LLVM ------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

SUBDIRS = lib/System lib/Support utils lib tools runtime

EXTRA_DIST = CREDITS.txt LICENSE.txt llvm.spec README.txt include docs autoconf

# Remove CVS directories from the distribution
dist-hook:
	rm -rf `find $(top_distdir) -name CVS`
