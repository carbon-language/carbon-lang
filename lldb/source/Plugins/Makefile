##===- source/Plugins/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..

include $(LLDB_LEVEL)/../../Makefile.config


DIRS := ABI/MacOSX-arm ABI/MacOSX-i386 ABI/SysV-x86_64 Disassembler/llvm \
	ObjectContainer/BSD-Archive ObjectFile/ELF SymbolFile/DWARF \
	SymbolFile/Symtab Process/Utility DynamicLoader/Static \
	Platform Process/gdb-remote Instruction/ARM \
	UnwindAssembly/InstEmulation UnwindAssembly/x86 \
	LanguageRuntime/CPlusPlus/ItaniumABI \
	LanguageRuntime/ObjC/AppleObjCRuntime

ifeq ($(HOST_OS),Darwin)
DIRS += DynamicLoader/MacOSX-DYLD ObjectContainer/Universal-Mach-O \
	ObjectFile/Mach-O SymbolVendor/MacOSX
#DIRS += Process/MacOSX-User
endif

ifeq ($(HOST_OS),Linux)
DIRS += Process/Linux DynamicLoader/Linux-DYLD
endif

include $(LLDB_LEVEL)/Makefile
