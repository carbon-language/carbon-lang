LEVEL = ../..
DIRS := include lib tools docs

include $(LEVEL)/Makefile.common

ifneq ($(PROJ_SRC_ROOT),$(PROJ_OBJ_ROOT))
test::
	$(Verb) if [ ! -f test/Makefile ]; then \
	  $(MKDIR) test; \
	  $(CP) $(PROJ_SRC_DIR)/test/Makefile test/Makefile; \
	fi
endif

test::
	@ $(MAKE) -C test 

report::
	@ $(MAKE) -C test report

clean::
	@ $(MAKE) -C test clean

tags::
	$(Verb) etags `find . -type f -name \*.h | grep -v /lib/Headers | grep -v /test/` `find . -type f -name \*.cpp | grep -v /lib/Headers | grep -v /test/`

cscope.files:
	find tools lib include -name '*.cpp' \
	                    -or -name '*.def' \
	                    -or -name '*.td' \
	                    -or -name '*.h' > cscope.files

.PHONY: test report clean cscope.files
