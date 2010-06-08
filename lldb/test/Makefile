.PHONY: clean all

#----------------------------------------------------------------------
# Make all of the test programs
#----------------------------------------------------------------------
all:
	find . -type d -depth 1 | xargs -J % find % \
		-name Makefile \
		-exec echo \; \
		-exec echo make -f '{}' \; \
		-execdir make \;

#----------------------------------------------------------------------
# Make all of the test programs
#----------------------------------------------------------------------
clean:
	find . -type d -depth 1 | xargs -J % find % \
		-name Makefile \
		-exec echo \; \
		-exec echo make -f '{}' clean \; \
		-execdir make clean \;
