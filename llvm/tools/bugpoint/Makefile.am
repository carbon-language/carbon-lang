#===-- tools/bugpoint/Makefile.am --------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

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
  $(call GETOBJS,AsmParser,BCReader,BCWriter,Core,ProfilePaths) \
  $(call GETLIBS,IPO,ScalarOpts,Analysis,Transforms,Instrument) \
  $(call GETLIBS,DataStructure,IPA,Target,TransformUtils,Support,System)
