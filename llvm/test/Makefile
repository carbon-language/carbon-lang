TESTS := $(wildcard *.ll)

test all : testasmdis testopt
	@echo "All tests successfully completed!"

testasmdis : $(TESTS:%.ll=%.ll.asmdis)
	@echo "All assembler/disassembler test succeeded!"

testopt : $(TESTS:%.ll=%.ll.opt)

clean :
	rm -f *.[123] *.bc core

%.asmdis: %
	@echo "Running assembler/disassembler test on $<"
	@./TestAsmDisasm.sh $<

%.opt: %
	@echo "Running optimizier test on $<"
	@./TestOptimizer.sh $<
