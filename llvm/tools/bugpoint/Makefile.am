#===-- tools/bugpoint/Makefile.am --------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = bugpoint

bugpoint_SOURCES = \
  BugDriver.cpp \
  bugpoint.cpp \
  CrashDebugger.cpp \
  ExecutionDriver.cpp \
  ExtractFunction.cpp \
  Miscompilation.cpp \
  OptimizerDriver.cpp \
  TestPasses.cpp

bugpoint_LDADD = \
  $(call GETOBJS,IPO,ScalarOpts,Analysis,Transforms,Instrument,ProfilePaths) \
  $(call GETOBJS,DataStructure,IPA) \
  $(call GETLIBS,Target) \
  $(call GETOBJS,TransformUtils,AsmParser,BCReader,BCWriter,Core) \
  $(call GETLIBS,Support,System)
