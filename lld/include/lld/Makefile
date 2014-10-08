LLD_LEVEL := ../..
DIRS := Config

include $(LLD_LEVEL)/Makefile

install-local::
	$(Echo) Installing lld include files
	$(Verb) $(MKDIR) $(DESTDIR)$(PROJ_includedir)
	$(Verb) if test -d "$(PROJ_SRC_DIR)" ; then \
	  cd $(PROJ_SRC_DIR)/.. && \
	  for  hdr in `find lld -type f \
	      '(' -name LICENSE.TXT \
	       -o -name '*.def' \
	       -o -name '*.h' \
	       -o -name '*.inc' \
	      ')' -print \
              | grep -v CVS | grep -v .svn | grep -v .dir` ; do \
	    instdir=$(DESTDIR)`dirname "$(PROJ_includedir)/$$hdr"` ; \
	    if test \! -d "$$instdir" ; then \
	      $(EchoCmd) Making install directory $$instdir ; \
	      $(MKDIR) $$instdir ;\
	    fi ; \
	    $(DataInstall) $$hdr $(DESTDIR)$(PROJ_includedir)/$$hdr ; \
	  done ; \
	fi
ifneq ($(PROJ_SRC_ROOT),$(PROJ_OBJ_ROOT))
	$(Verb) if test -d "$(PROJ_OBJ_ROOT)/tools/lld/include/lld" ; then \
	  cd $(PROJ_OBJ_ROOT)/tools/lld/include && \
	  for hdr in `find lld -type f \
	      '(' -name LICENSE.TXT \
	       -o -name '*.def' \
	       -o -name '*.h' \
	       -o -name '*.inc' \
	      ')' -print \
            | grep -v CVS | grep -v .tmp | grep -v .dir` ; do \
	    instdir=$(DESTDIR)`dirname "$(PROJ_includedir)/$$hdr"` ; \
	    if test \! -d "$$instdir" ; then \
	      $(EchoCmd) Making install directory $$instdir ; \
	      $(MKDIR) $$instdir ;\
	    fi ; \
	    $(DataInstall) $$hdr $(DESTDIR)$(PROJ_includedir)/$$hdr ; \
	  done ; \
	fi
endif
