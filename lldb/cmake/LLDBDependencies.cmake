set( LLDB_USED_LIBS
  lldbBase
  lldbBreakpoint
  lldbCommands
  lldbDataFormatters
  lldbHost
  lldbCore
  lldbExpression
  lldbInitialization
  lldbInterpreter
  lldbSymbol
  lldbTarget
  lldbUtility

  # Plugins
  lldbPluginDisassemblerLLVM
  lldbPluginSymbolFileDWARF
  lldbPluginSymbolFileSymtab
  lldbPluginDynamicLoaderStatic
  lldbPluginDynamicLoaderPosixDYLD
  lldbPluginDynamicLoaderHexagonDYLD

  lldbPluginObjectFileELF
  lldbPluginObjectFileJIT
  lldbPluginSymbolVendorELF
  lldbPluginObjectContainerBSDArchive
  lldbPluginObjectContainerMachOArchive
  lldbPluginProcessGDBRemote
  lldbPluginProcessUtility
  lldbPluginPlatformAndroid
  lldbPluginPlatformGDB
  lldbPluginPlatformFreeBSD
  lldbPluginPlatformKalimba
  lldbPluginPlatformLinux
  lldbPluginPlatformPOSIX
  lldbPluginPlatformWindows
  lldbPluginObjectContainerMachOArchive
  lldbPluginObjectContainerBSDArchive
  lldbPluginPlatformMacOSX
  lldbPluginDynamicLoaderMacOSXDYLD
  lldbPluginUnwindAssemblyInstEmulation
  lldbPluginUnwindAssemblyX86
  lldbPluginAppleObjCRuntime
  lldbPluginRenderScriptRuntime
  lldbPluginCXXItaniumABI
  lldbPluginABIMacOSX_arm
  lldbPluginABIMacOSX_arm64
  lldbPluginABIMacOSX_i386
  lldbPluginABISysV_arm
  lldbPluginABISysV_arm64
  lldbPluginABISysV_i386
  lldbPluginABISysV_x86_64
  lldbPluginABISysV_hexagon
  lldbPluginABISysV_ppc
  lldbPluginABISysV_ppc64
  lldbPluginABISysV_mips
  lldbPluginABISysV_mips64
  lldbPluginInstructionARM
  lldbPluginInstructionARM64
  lldbPluginInstructionMIPS
  lldbPluginInstructionMIPS64
  lldbPluginObjectFilePECOFF
  lldbPluginOSPython
  lldbPluginMemoryHistoryASan
  lldbPluginInstrumentationRuntimeAddressSanitizer
  lldbPluginSystemRuntimeMacOSX
  lldbPluginProcessElfCore
  lldbPluginJITLoaderGDB
  )

# Windows-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessWindows
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
    Ws2_32
    Rpcrt4
    )
endif ()

# Linux-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Linux" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessLinux
    lldbPluginProcessPOSIX
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
   )
endif ()

# FreeBSD-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "FreeBSD" )
  list(APPEND LLDB_USED_LIBS
    lldbPluginProcessFreeBSD
    lldbPluginProcessPOSIX
    lldbPluginProcessElfCore
    lldbPluginJITLoaderGDB
    )
endif ()

# Darwin-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Darwin" )
  set(LLDB_VERS_GENERATED_FILE ${LLDB_BINARY_DIR}/source/LLDB_vers.c)
  add_custom_command(OUTPUT ${LLDB_VERS_GENERATED_FILE}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/generate-vers.pl
            ${LLDB_SOURCE_DIR}/lldb.xcodeproj/project.pbxproj liblldb_core
            > ${LLDB_VERS_GENERATED_FILE})

  set_source_files_properties(${LLDB_VERS_GENERATED_FILE} PROPERTIES GENERATED 1)
  list(APPEND LLDB_USED_LIBS
    lldbPluginDynamicLoaderDarwinKernel
    lldbPluginObjectFileMachO
    lldbPluginProcessMachCore
    lldbPluginProcessMacOSXKernel
    lldbPluginSymbolVendorMacOSX
    )
endif()

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
  clangRewrite
  clangRewriteFrontend
  clangSema
  clangSerialization
  )

set(LLDB_SYSTEM_LIBS)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT __ANDROID_NDK__)
  if (NOT LLDB_DISABLE_LIBEDIT)
    list(APPEND LLDB_SYSTEM_LIBS edit)
  endif()
  if (NOT LLDB_DISABLE_CURSES)
    list(APPEND LLDB_SYSTEM_LIBS panel ncurses)
  endif()
endif()
# On FreeBSD backtrace() is provided by libexecinfo, not libc.
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  list(APPEND LLDB_SYSTEM_LIBS execinfo)
endif()

if (NOT LLDB_DISABLE_PYTHON AND NOT LLVM_BUILD_STATIC)
  list(APPEND LLDB_SYSTEM_LIBS ${PYTHON_LIBRARIES})
endif()

list(APPEND LLDB_SYSTEM_LIBS ${system_libs})

if (LLVM_BUILD_STATIC)
  list(APPEND LLDB_SYSTEM_LIBS python2.7 z util termcap gpm ssl crypto bsd)
endif()

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  interpreter
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
  runtimedyld
  option
  support
  )

if ( NOT LLDB_DISABLE_PYTHON )
  set(LLDB_WRAP_PYTHON ${LLDB_BINARY_DIR}/scripts/LLDBWrapPython.cpp)

  set_source_files_properties(${LLDB_WRAP_PYTHON} PROPERTIES GENERATED 1)
  if (CLANG_CL)
    set_source_files_properties(${LLDB_WRAP_PYTHON} PROPERTIES COMPILE_FLAGS -Wno-unused-function)
  endif()
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${LLDB_WRAP_PYTHON}
                 APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-sequence-point")
  endif ()
endif()
