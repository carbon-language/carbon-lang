LEVEL = ../..
DIRS := lib Driver 

include $(LEVEL)/Makefile.common

test::
	cd test && $(MAKE) -f Makefile.parallel

report::
	cd test && $(MAKE) -f Makefile.parallel report

clean::
	@rm -rf build
	@rm -rf `find test -name Output`
