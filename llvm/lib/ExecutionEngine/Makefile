LEVEL = ../..
TOOLNAME = lli
PARALLEL_DIRS = Interpreter JIT

# Get the config name...
include $(LEVEL)/Makefile.config

# Generic JIT libraries
JITLIBS   = lli-jit codegen
ARCHLIBS  =

# What the X86 JIT requires
JITLIBS  += x86
# X86 doesn't require any ARCHLIBS



# What the Sparc JIT requires
ifeq ($(ARCH),Sparc)
JITLIBS  += sparc 
ARCHLIBS  = sched livevar instrument.a profpaths transformutils.a \
           bcwriter transforms.a ipo.a ipa.a datastructure.a regalloc \
           mapping select postopts.a preopts

endif

USEDLIBS = lli-interpreter $(JITLIBS) $(ARCHLIBS) bcreader vmcore scalaropts.a \
           analysis.a support.a target.a

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common
