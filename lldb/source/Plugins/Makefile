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


PARALLEL_DIRS := ABI/MacOSX-arm ABI/MacOSX-arm64 ABI/MacOSX-i386 ABI/SysV-x86_64 \
	ABI/SysV-hexagon ABI/SysV-ppc ABI/SysV-ppc64 \
	Disassembler/llvm \
	ObjectContainer/BSD-Archive ObjectFile/ELF ObjectFile/PECOFF \
	ObjectFile/JIT SymbolFile/DWARF SymbolFile/Symtab Process/Utility \
	DynamicLoader/Static Platform Process/gdb-remote \
	Instruction/ARM Instruction/ARM64 \
	UnwindAssembly/InstEmulation UnwindAssembly/x86 \
	LanguageRuntime/CPlusPlus/ItaniumABI \
	LanguageRuntime/ObjC/AppleObjCRuntime \
	DynamicLoader/POSIX-DYLD \
	DynamicLoader/Hexagon-DYLD \
	OperatingSystem/Python \
	SymbolVendor/ELF \
	MemoryHistory/asan \
	InstrumentationRuntime/AddressSanitizer

ifeq ($(HOST_OS),Darwin)
PARALLEL_DIRS += Process/MacOSX-Kernel
PARALLEL_DIRS += DynamicLoader/MacOSX-DYLD DynamicLoader/Darwin-Kernel
PARALLEL_DIRS +=	ObjectContainer/Universal-Mach-O ObjectFile/Mach-O
PARALLEL_DIRS += SymbolVendor/MacOSX
#PARALLEL_DIRS += Process/MacOSX-User
PARALLEL_DIRS += Process/mach-core
PARALLEL_DIRS += SystemRuntime/MacOSX
PARALLEL_DIRS += Process/elf-core
PARALLEL_DIRS += JITLoader/GDB
endif

ifeq ($(HOST_OS),Linux)
PARALLEL_DIRS += DynamicLoader/MacOSX-DYLD
PARALLEL_DIRS += Process/Linux Process/POSIX
PARALLEL_DIRS += Process/elf-core
PARALLEL_DIRS += JITLoader/GDB
endif

ifeq ($(HOST_OS),MingW)
PARALLEL_DIRS += Process/elf-core
PARALLEL_DIRS += JITLoader/GDB
endif

ifneq (,$(filter $(HOST_OS), FreeBSD GNU/kFreeBSD))
PARALLEL_DIRS += Process/FreeBSD Process/POSIX
PARALLEL_DIRS += Process/elf-core
PARALLEL_DIRS += JITLoader/GDB
endif

include $(LLDB_LEVEL)/Makefile
