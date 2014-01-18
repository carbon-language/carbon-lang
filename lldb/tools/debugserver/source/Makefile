##===- tools/debugserver/source/Makefile -------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LLDB_LEVEL := ../../..

LIBRARYNAME := lldbDebugserverCommon
BUILD_ARCHIVE = 1

SOURCES := debugserver.cpp 		\
	DNBArch.cpp					\
  	DNBBreakpoint.cpp			\
  	DNB.cpp						\
  	DNBDataRef.cpp				\
  	DNBError.cpp				\
  	DNBLog.cpp					\
  	DNBRegisterInfo.cpp			\
  	DNBThreadResumeActions.cpp	\
  	libdebugserver.cpp			\
  	PseudoTerminal.cpp			\
  	PThreadEvent.cpp			\
  	PThreadMutex.cpp			\
  	RNBContext.cpp				\
  	RNBRemote.cpp				\
  	RNBServices.cpp				\
  	RNBSocket.cpp				\
  	SysSignal.cpp				\
  	TTYState.cpp

include $(LLDB_LEVEL)/Makefile

ifeq ($(HOST_OS),Darwin)
DIRS := MacOSX/i386 MacOSX/x86_64 MacOSX
CPP.Flags += -I$(PROJ_SRC_DIR)/MacOSX
CPP.Flags += -I$(PROJ_OBJ_DIR)/..
BUILT_SOURCES = debugserver_vers.c
endif

ifeq ($(HOST_OS),Darwin)
debugserver_vers.c: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/generate-vers.pl $(PROJ_SRC_DIR)/../debugserver.xcodeproj/project.pbxproj
	"$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/generate-vers.pl" "$(PROJ_SRC_DIR)/../debugserver.xcodeproj/project.pbxproj" debugserver > debugserver_vers.c
endif
