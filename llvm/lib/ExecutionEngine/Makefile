LEVEL = ../..
TOOLNAME = lli
PARALLEL_DIRS = Interpreter JIT

# FIXME: This enables testing the Sparc JIT on x86.
# Perhaps in the future this should be a ./configure option.

# Generic JIT libraries
JITLIBS   = lli-jit codegen
ARCHLIBS  =

# What the X86 JIT requires
JITLIBS  += x86
ARCHLIBS += 

# What the Sparc JIT requires
JITLIBS  += sparc 
ARCHLIBS  = sched livevar instrument.a profpaths transformutils.a \
           bcwriter transforms.a ipo.a ipa.a datastructure.a regalloc \
           mapping select postopts.a preopts

USEDLIBS = lli-interpreter $(JITLIBS) bcreader vmcore scalaropts \
           analysis.a support.a target.a $(ARCHLIBS)

# Have gcc tell the linker to export symbols from the program so that
# dynamically loaded modules can be linked against them.
#
TOOLLINKOPTS = -ldl

include $(LEVEL)/Makefile.common
