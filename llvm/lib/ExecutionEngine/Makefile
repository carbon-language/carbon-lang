LEVEL = ../..
TOOLNAME = lli
PARALLEL_DIRS = Interpreter JIT

JITLIBS  = lli-jit codegen x86
USEDLIBS = lli-interpreter $(JITLIBS) bcreader vmcore scalaropts.a \
           analysis.a support.a target.a 

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common
