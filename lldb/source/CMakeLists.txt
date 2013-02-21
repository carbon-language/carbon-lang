include_directories(.)

# FIXME: these includes should be enabled only for Linux builds
include_directories(
  Plugins/Process/Linux
  Plugins/Process/POSIX
  )

add_subdirectory(API)
add_subdirectory(Breakpoint)
add_subdirectory(Commands)
add_subdirectory(Core)
add_subdirectory(DataFormatters)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(Plugins)
add_subdirectory(Symbol)
add_subdirectory(Target)
add_subdirectory(Utility)

set( LLDB_USED_LIBS
  lldbAPI
  lldbBreakpoint
  lldbCommands
  lldbDataFormatters
  lldbHostCommon
  lldbCore
  lldbExpression
  #lldbInitAndLog
  lldbInterpreter
  lldbSymbol
  lldbTarget  
  lldbUtility
  
  # Plugins
  lldbPluginDisassemblerLLVM
  lldbPluginSymbolFileDWARF
  lldbPluginSymbolFileSymtab
  lldbPluginDynamicLoaderStatic
  
  lldbPluginObjectFileMachO
  lldbPluginObjectFileELF
  lldbPluginObjectContainerBSDArchive
  lldbPluginObjectContainerMachOArchive
  lldbPluginProcessGDBRemote  
  lldbPluginProcessUtility
  lldbPluginPlatformGDB
  lldbPluginPlatformFreeBSD
  lldbPluginPlatformLinux
  lldbPluginObjectFileMachO
  lldbPluginObjectContainerMachOArchive
  lldbPluginObjectContainerBSDArchive
  lldbPluginPlatformMacOSX
  lldbPluginDynamicLoaderMacOSXDYLD
  lldbPluginDynamicLoaderPosixDYLD
  lldbPluginUnwindAssemblyInstEmulation
  lldbPluginUnwindAssemblyX86
  lldbPluginDynamicLoaderDarwinKernel
  lldbPluginAppleObjCRuntime
  lldbPluginCXXItaniumABI
  lldbPluginABIMacOSX_arm
  lldbPluginABIMacOSX_i386
  lldbPluginABISysV_x86_64
  lldbPluginInstructionARM
  lldbPluginOSPython

  
  # Windows (FIXME: detect host OS and enable below if Windows)
  #lldbHostWindows
  #lldbPluginPlatformWindows
  lldbPluginObjectFilePECOFF
  #Ws2_32
  
  # Linux (FIXME: detect host OS and enable below if Linux)
  lldbHostLinux
  lldbPluginProcessLinux
  lldbPluginProcessPOSIX
  )
  
set( CLANG_USED_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangRewriteCore
  clangRewriteFrontend
  clangSema
  clangSerialization
  )  
  
set( LLDB_SYSTEM_LIBS
  edit
  python2.7
  )

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  jit
  interpreter
  nativecodegen
  asmparser
  bitreader
  bitwriter
  codegen
  ipo
  selectiondag
  bitreader
  mc
  mcjit
  core
  mcdisassembler
  executionengine
  )

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/../scripts/LLDBWrapPython.cpp PROPERTIES GENERATED 1)
set(SHARED_LIBRARY 1)

add_lldb_library(liblldb
  lldb.cpp
  lldb-log.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/../scripts/LLDBWrapPython.cpp
  )
set_target_properties(liblldb PROPERTIES OUTPUT_NAME lldb)
target_link_libraries(liblldb ${LLDB_SYSTEM_LIBS})
add_dependencies(liblldb swig_wrapper)

install(TARGETS liblldb
  LIBRARY DESTINATION lib)
