LLDB_LEVEL := ..
include $(LLDB_LEVEL)/Makefile

.PHONY: programs

all:: check-local

#----------------------------------------------------------------------
# Make all of the test programs
#----------------------------------------------------------------------
programs:
	find . -type d -depth 1 | xargs -J % find % \
		-name Makefile \
		-exec echo \; \
		-exec echo make -f '{}' \; \
		-execdir make \;

#----------------------------------------------------------------------
# Clean all of the test programs
#----------------------------------------------------------------------
clean::
	find . -type d -depth 1 | xargs -J % find % \
		-name Makefile \
		-exec echo \; \
		-exec echo make -f '{}' clean \; \
		-execdir make clean \;

#----------------------------------------------------------------------
# Run the tests
#----------------------------------------------------------------------
check-local::
	rm -rf ../test-rdir
	env PATH="$(ToolDir):$(PATH)" \
	    PYTHONPATH=$(ToolDir) \
	    $(SHLIBPATH_VAR)=$(LibDir) \
	     python $(PROJ_SRC_DIR)/dotest.py -i -v -r ../test-rdir
