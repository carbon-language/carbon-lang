TESTS := $(wildcard *.ll)

SELECTTESTS := $(shell /bin/ls *.ll | grep -v testmemory | grep -v testswitch")

test all : testasmdis testopt
	@echo "All tests successfully completed!"

testasmdis : $(TESTS:%.ll=%.ll.asmdis)
	@echo "All assembler/disassembler test succeeded!"

testopt : $(TESTS:%.ll=%.ll.opt)

testselect : $(SELECTTESTS:%.ll=%.mc)

clean :
	rm -f *.[123] *.bc *.mc core

%.asmdis: %
	@echo "Running assembler/disassembler test on $<"
	@./TestAsmDisasm.sh $<

%.opt: %
	@echo "Running optimizier test on $<"
	@./TestOptimizer.sh $<

%.bc: %.ll
	rm -f $@
	as $<

%.mc: %.bc
	@echo "Generating machine instructions for $<"
	@llc -dselect i $< > $@
