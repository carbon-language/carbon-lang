set(LLVM_NO_RTTI 1)

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DEXPORT_LIBLLDB )
endif()

# An OBJECT library is a special type of CMake library that produces
# no archive, has no link interface, and no link inputs.  It is like
# a regular archive, just without the physical output.  To link against
# an OBJECT library, you reference it in the *source* file list of a
# library using the special syntax $<TARGET_OBJECTS:lldbAPI>.  This will
# cause every object file to be passed to the linker independently, as
# opposed to a single archive being passed to the linker.
#
# This is *extremely* important on Windows.  lldbAPI exports all of the
# SB classes using __declspec(dllexport).  Unfortunately for technical
# reasons it is not possible (well, extremely difficult) to get the linker
# to propagate a __declspec(dllexport) attribute from a symbol in an
# object file in an archive to a DLL that links against that archive.
# The solution to this is for the DLL to link the object file directly.
# So lldbAPI must be an OBJECT library.
add_lldb_library(lldbAPI OBJECT
  SBAddress.cpp
  SBAttachInfo.cpp
  SBBlock.cpp
  SBBreakpoint.cpp
  SBBreakpointLocation.cpp
  SBBroadcaster.cpp
  SBCommandInterpreter.cpp
  SBCommandReturnObject.cpp
  SBCommunication.cpp
  SBCompileUnit.cpp
  SBData.cpp
  SBDebugger.cpp
  SBDeclaration.cpp
  SBError.cpp
  SBEvent.cpp
  SBExecutionContext.cpp
  SBExpressionOptions.cpp
  SBFileSpec.cpp
  SBFileSpecList.cpp
  SBFrame.cpp
  SBFunction.cpp
  SBHostOS.cpp
  SBInstruction.cpp
  SBInstructionList.cpp
  SBLaunchInfo.cpp
  SBLineEntry.cpp
  SBListener.cpp
  SBModule.cpp
  SBModuleSpec.cpp
  SBPlatform.cpp
  SBProcess.cpp
  SBQueue.cpp
  SBQueueItem.cpp
  SBSection.cpp
  SBSourceManager.cpp
  SBStream.cpp
  SBStringList.cpp
  SBSymbol.cpp
  SBSymbolContext.cpp
  SBSymbolContextList.cpp
  SBTarget.cpp
  SBThread.cpp
  SBThreadCollection.cpp
  SBThreadPlan.cpp
  SBType.cpp
  SBTypeCategory.cpp
  SBTypeEnumMember.cpp
  SBTypeFilter.cpp
  SBTypeFormat.cpp
  SBTypeNameSpecifier.cpp
  SBTypeSummary.cpp
  SBTypeSynthetic.cpp
  SBValue.cpp
  SBValueList.cpp
  SBVariablesOptions.cpp
  SBWatchpoint.cpp
  SBUnixSignals.cpp
  )
