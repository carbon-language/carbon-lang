LEVEL = ../../..
LIBNAME = crtend
Source = crtend.c listend.ll

EXPORTED_SYMBOL_LIST = __main,llvm.global_ctors,llvm.global_dtors

include ../Makefile.libs

install:: $(DESTLIBNAME)
	cp $(DESTLIBNAME) $(INSTALL_DIR)/crtend.o
	rm $(INSTALL_DIR)/libcrtend.bc
