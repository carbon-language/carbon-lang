LEVEL = ../..
TOOLNAME = jello
USEDLIBS = bcreader vmcore codegen x86 support.a target.a

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = -ldl $(ExportSymbols)

include $(LEVEL)/Makefile.common
