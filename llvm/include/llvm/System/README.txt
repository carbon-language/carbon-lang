System Utilities Interface
==========================

The design of this library has several key constraints aimed at shielding LLVM
from the vagaries of operating system differences. The goal here is to provide
interfaces to operating system concepts (files, memory maps, sockets, signals,
locking, etc) efficiently and in such a way that the remainder of LLVM is
completely operating system agnostic. 

PLEASE READ AND COMPREHEND FULLY THE DOCUMENTATION in 

llvm/docs/SystemLibrary.html 

before making changes to this library.
