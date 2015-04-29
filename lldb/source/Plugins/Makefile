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
	ABI/SysV-arm ABI/SysV-hexagon ABI/SysV-ppc ABI/SysV-ppc64 \
	Disassembler/llvm \
	ObjectContainer/BSD-Archive ObjectFile/ELF ObjectFile/PECOFF \
	ObjectContainer/Universal-Mach-O ObjectFile/Mach-O \
	ObjectFile/JIT SymbolFile/DWARF SymbolFile/Symtab Process/Utility \
	DynamicLoader/Static Platform Process/gdb-remote \
	Instruction/ARM Instruction/ARM64 Instruction/MIPS64 \
	UnwindAssembly/InstEmulation UnwindAssembly/x86 \
	LanguageRuntime/CPlusPlus/ItaniumABI \
	LanguageRuntime/ObjC/AppleObjCRuntime \
	LanguageRuntime/RenderScript/RenderScriptRuntime \
	DynamicLoader/POSIX-DYLD \
	DynamicLoader/Hexagon-DYLD \
	DynamicLoader/MacOSX-DYLD \
	OperatingSystem/Python \
	SystemRuntime/MacOSX \
	SymbolVendor/ELF \
	MemoryHistory/asan \
	InstrumentationRuntime/AddressSanitizer

ifeq ($(HOST_OS),Darwin)
PARALLEL_DIRS += Process/MacOSX-Kernel
PARALLEL_DIRS += DynamicLoader/Darwin-Kernel
PARALLEL_DIRS += SymbolVendor/MacOSX
#PARALLEL_DIRS += Process/MacOSX-User
PARALLEL_DIRS += Process/mach-core
PARALLEL_DIRS += Process/elf-core
PARALLEL_DIRS += JITLoader/GDB
endif

ifeq ($(HOST_OS),Linux)
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
