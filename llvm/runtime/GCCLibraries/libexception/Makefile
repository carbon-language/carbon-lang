LEVEL = ../../..
BYTECODE_LIBRARY=1
DONT_BUILD_RELINKED=1
LIBRARYNAME=exception

#Source = crtend.c listend.ll

###EXPORTED_SYMBOL_LIST = __main,llvm.global_ctors,llvm.global_dtors

include $(LEVEL)/Makefile.common


$(LLVMGCCDIR)/bytecode-libs/crtend.o: $(LIBNAME_BC)
	@cp $< $@

install:: $(LLVMGCCDIR)/bytecode-libs/crtend.o
	@rm $(LLVMGCCDIR)/bytecode-libs/libcrtend.bc
