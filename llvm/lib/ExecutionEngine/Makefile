LEVEL = ../..
TOOLNAME = lli
DIRS = Interpreter JIT

JITLIBS  = lli-jit codegen x86
USEDLIBS = lli-interpreter $(JITLIBS) bcreader vmcore analysis.a support.a target.a 
#transforms.a

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common
