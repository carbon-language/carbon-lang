LEVEL = ../..
include $(LEVEL)/Makefile.common

all:: testPow2

clean::
	rm -f testPow2

testPow2 : $(ObjectsG)
	$(LinkG) -o testPow2 $(ObjectsG) -lsupport

run: testPow2
	testPow2 25
