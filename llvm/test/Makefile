LLC  := ../tools/Debug/llc
AS   := ../tools/Debug/as
LLCOPTS := -dsched y

TESTS := $(wildcard *.ll)

LLCTESTS := $(shell /bin/ls *.ll | grep -v testmemory | grep -v testswitch | grep -v sumarray | grep -v opttest | grep -v xx.ll | grep -v select.ll )


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
	$(AS) < $< | $(LLC) $(LLCOPTS) > $@

%.s: %.ll $(LLC) $(AS)
	$(AS) < $<  | $(LLC) > $@

%.o: %.s %.ll
	/usr/ccs/bin/as -xarch=v9 $<

