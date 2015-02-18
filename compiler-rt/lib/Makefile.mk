#===- lib/Makefile.mk --------------------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

SubDirs :=

# Add submodules.
SubDirs += asan
SubDirs += builtins
SubDirs += interception
SubDirs += lsan
SubDirs += profile
SubDirs += sanitizer_common
SubDirs += ubsan
