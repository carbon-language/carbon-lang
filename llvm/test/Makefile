LLC  := ../tools/Debug/llc
AS   := ../tools/Debug/as
LLCOPTS := -dsched t

TESTS := $(wildcard *.ll)

LLCTESTS := $(shell /bin/ls *.ll | grep -v testmemory | grep -v testswitch | grep -v sumarray | grep -v globalvars)


test all : testasmdis testopt testcodegen
	@echo "All tests successfully completed!"

testasmdis  : $(TESTS:%.ll=%.ll.asmdis)
	@echo "All assembler/disassembler test succeeded!"

testopt     : $(TESTS:%.ll=%.ll.opt)

testselect  : $(LLCTESTS:%.ll=%.mc)

testsched   : $(LLCTESTS:%.ll=%.mc)

testcodegen : $(LLCTESTS:%.ll=%.mc)

clean :
	rm -f *.[123] *.bc *.mc core

%.asmdis: %
	@echo "Running assembler/disassembler test on $<"
	@./TestAsmDisasm.sh $<

%.opt: %
	@echo "Running optimizier test on $<"
	@./TestOptimizer.sh $<

%.bc: %.ll
	$(AS) $< -f

%.mc: %.ll $(LLC) $(AS)
	@echo "Generating machine instructions for $<"
	$(AS) < $< | $(LLC) -dsched=t > $@

