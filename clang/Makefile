LEVEL = ../..
DIRS := lib Driver 

include $(LEVEL)/Makefile.common

test::
	@ $(MAKE) -C test -f Makefile.parallel

report::
	@ $(MAKE) -C test -f Makefile.parallel report

clean::
	@ $(MAKE) -C test -f Makefile.parallel clean

.PHONY: test report clean
