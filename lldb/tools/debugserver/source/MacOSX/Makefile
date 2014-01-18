##===- tools/debugserver/source/MacOSX/Makefile ------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LLDB_LEVEL := ../../../..

DIRS := i386 x86_64

TOOLNAME = debugserver

CODESIGN_TOOLS := 1

TOOL_CODESIGN_IDENTITY := lldb_codesign

LLVMLibsOptions += -llldbDebugserverCommon -llldbUtility -llldbDebugserverMacOSX_I386 -llldbDebugserverMacOSX_X86_64 \
					-framework Foundation -framework CoreFoundation

GENERATED_MACH_SOURCES = $(PROJ_OBJ_DIR)/mach_excServer.c $(PROJ_OBJ_DIR)/mach_excUser.c

SOURCES := CFBundle.cpp \
  CFData.cpp			\
  CFString.cpp			\
  MachException.cpp		\
  MachProcess.cpp		\
  MachTask.cpp			\
  MachThread.cpp		\
  MachThreadList.cpp	\
  MachVMMemory.cpp		\
  MachVMRegion.cpp

BUILT_SOURCES = $(GENERATED_MACH_SOURCES) $(PROJ_OBJ_DIR)/HasAVX.o

CPP.Flags += -I$(PROJ_OBJ_DIR)/../.. -I$(PROJ_SRC_DIR)/..

LD.Flags += -Wl,-sectcreate,__TEXT,__info_plist,$(PROJ_SRC_DIR)/../../resources/lldb-debugserver-Info.plist

include $(LLDB_LEVEL)/Makefile

ObjectsO += $(PROJ_OBJ_DIR)/HasAVX.o

$(PROJ_OBJ_DIR)/HasAVX.o: $(PROJ_SRC_DIR)/HasAVX.s
	$(Echo) "Compiling HasAVX.s for $(BuildMode) build" $(PIC_FLAG)
	$(CC) $(TargetCommonOpts) $(CompileCommonOpts) -c $< -o $@

ifeq ($(HOST_OS),Darwin)
LLVMLibsOptions += -Wl,-rpath,@loader_path/../lib/
endif

$(GENERATED_MACH_SOURCES):
	mig -I$(PROJ_OBJ_DIR)/../.. $(PROJ_SRC_DIR)/dbgnub-mig.defs