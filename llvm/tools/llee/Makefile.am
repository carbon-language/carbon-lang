#===-- tools/llee/Makefile.am ------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LTLIBRARIES = libexecve.la

libexecve_la_SOURCES = ExecveHandler.c StorageProxy.c SysUtils.c

all: llee

llee: Makefile
	echo exec env LD_PRELOAD=$(DESTLIBCURRENT)/libexecve$(SHLIBEXT) $$\* > $@
	chmod u+x $@

clean:
	rm -f /llee
