LEVEL = ../../..

LIBNAME = instr

# We use assert and memset here... and we are linked into the program AFTER final
# link.  Because of this, we have to play funny games here. :(
LDFLAGS = /home/vadve/lattner/cvs/gcc_install/lib/gcc-lib/llvm/3.1/libgcc.bc \
          /home/vadve/lattner/cvs/gcc_install/lib/gcc-lib/llvm/3.1/libc.bc

include ../Makefile.libs

