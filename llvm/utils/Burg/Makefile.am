##===- utils/Burg/Makefile.am ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = burg

BUILT_SOURCES= gram.tab.c
burg_SOURCES = be.c burs.c closure.c delta.c fe.c gram.yc item.c lex.c list.c \
               main.c map.c nonterminal.c operator.c pattern.c plank.c queue.c \
               rule.c sample.gr string.c symtab.c table.c trim.c zalloc.c \
               $(BUILT_SOURCES)

EXTRA_DIST   = LICENSE.TXT

lex.o: gram.tab.h

gram.tab.c gram.tab.h: gram.yc
	$(BISON) -o gram.tab.c -d $<

clean::
	$(VERB) $(RM) -rf gram.tab.h gram.tab.c core* *.aux *.log *.dvi sample sample.c tmp

doc.dvi: doc.tex
	$(VERB) latex doc; latex doc

test:: $(TOOLEXENAME_G) sample.gr
	$(TOOLEXENAME_G) -I     <sample.gr   >sample.c && $(CC) $(CFLAGS) -o sample sample.c && ./sample
	$(TOOLEXENAME_G) -I      sample.gr   >tmp && cmp tmp sample.c
	$(TOOLEXENAME_G) -I     <sample.gr -o tmp && cmp tmp sample.c
	$(TOOLEXENAME_G) -I      sample.gr -o tmp && cmp tmp sample.c
	$(TOOLEXENAME_G) -I -O0 <sample.gr   >tmp && cmp tmp sample.c
	$(TOOLEXENAME_G) -I -=  <sample.gr   >tmp && cmp tmp sample.c
	$(RM) -f tmp sample.c
