LLC  := ../tools/Debug/llc
LAS  := ../tools/Debug/as
LDIS := ../tools/Debug/dis 
LINK := ../tools/Debug/link
LLCLIBS := runtime.o
LLCOPTS := 

ifeq ($(TRACE), yes)
    LLCOPTS := -trace
endif

CC      = /opt/SUNWspro/bin/cc
AS	= /opt/SUNWspro/bin/cc
DIS     = /usr/ccs/bin/dis
CFLAGS  = -g -xarch=v9
CCFLAGS = $(CFLAGS)
LDFLAGS = $(CFLAGS)
ASFLAGS = -c $(CFLAGS)


TESTS := $(wildcard *.ll)

LLCTESTS := $(shell /bin/ls *.ll | grep -v testswitch | grep -v opttest | grep -v xx | grep -v calltest | grep -v alloca | grep -v memory )


test all : testasmdis testopt testcodegen
	@echo "All tests successfully completed!"

testasmdis  : $(TESTS:%.ll=%.ll.asmdis)
	@echo "All assembler/disassembler test succeeded!"

testopt     : $(TESTS:%.ll=%.ll.opt)

testselect  : $(LLCTESTS:%.ll=%.mc)

testsched   : $(LLCTESTS:%.ll=%.mc)

testcodegen : $(LLCTESTS:%.ll=%.mc)

testsparc   : $(LLCTESTS:%.ll=%.s)

clean :
	rm -f *.[123] *.bc *.mc *.s *.o a.out core 

%.asmdis: %
	@echo "Running assembler/disassembler test on $<"
	@./TestAsmDisasm.sh $<

%.opt: %
	@echo "Running optimizier test on $<"
	@./TestOptimizer.sh $<

%.bc: %.ll $(LAS)
	$(LAS) -f $<

%.mc: %.bc $(LLC) $(AS)
	@echo "Generating machine instructions for $<"
	$(LLC) -dsched y $(LLCOPTS) $< > $@

%.trace.bc: %.bc $(LLC)
	$(LLC) -f -trace $(LLCOPTS) $<

%.s: %.bc $(LLC)
	$(LLC) -f $(LLCOPTS) $<

%: %.o $(LLCLIBS)
	$(CC) -o $@ $(LDFLAGS) $< $(LLCLIBS)


## Cancel built-in implicit rule that overrides the above rule
%: %.s

## The next two rules are for disassembling an executable or an object file
%.dis: %
	$(DIS) $< > $@

%.dis: %.o
	$(DIS) $< > $@

