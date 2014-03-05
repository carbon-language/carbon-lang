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
	ObjectContainer/BSD-Archive ObjectFile/ELF ObjectFile/PECOFF \
	SymbolFile/DWARF SymbolFile/Symtab Process/Utility \
	DynamicLoader/Static Platform Process/gdb-remote Instruction/ARM \
	UnwindAssembly/InstEmulation UnwindAssembly/x86 \
	LanguageRuntime/CPlusPlus/ItaniumABI \
	LanguageRuntime/ObjC/AppleObjCRuntime \
	DynamicLoader/POSIX-DYLD \
	OperatingSystem/Python \
	SymbolVendor/ELF

ifeq ($(HOST_OS),Darwin)
DIRS += Process/MacOSX-Kernel
DIRS += DynamicLoader/MacOSX-DYLD DynamicLoader/Darwin-Kernel
DIRS +=	ObjectContainer/Universal-Mach-O ObjectFile/Mach-O
DIRS += SymbolVendor/MacOSX
#DIRS += Process/MacOSX-User
DIRS += Process/mach-core
DIRS += SystemRuntime/MacOSX
endif

ifeq ($(HOST_OS),Linux)
DIRS += DynamicLoader/MacOSX-DYLD
DIRS += Process/Linux Process/POSIX
DIRS += Process/elf-core
DIRS += JITLoader/GDB
endif

ifneq (,$(filter $(HOST_OS), FreeBSD GNU/kFreeBSD))
DIRS += Process/FreeBSD Process/POSIX
DIRS += Process/elf-core
DIRS += JITLoader/GDB
endif

include $(LLDB_LEVEL)/Makefile
