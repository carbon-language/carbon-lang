LEVEL = ../..
TOOLNAME = lli
USEDLIBS = bcreader vmcore analysis.a support.a target.a transforms.a

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = -ldl $(ExportSymbols)

include $(LEVEL)/Makefile.common
