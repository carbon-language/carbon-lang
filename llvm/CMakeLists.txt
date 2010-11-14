# See docs/CMake.html for instructions about how to build LLVM with CMake.

project(LLVM)
cmake_minimum_required(VERSION 2.8)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

set(PACKAGE_VERSION "2.9")
include(VersionFromVCS)
add_version_info_from_vcs(PACKAGE_VERSION)

set(PACKAGE_NAME llvm)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed.
CMake would overwrite the makefiles distributed with LLVM.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

set(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVM_MAIN_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)
set(LLVM_EXAMPLES_BINARY_DIR ${LLVM_BINARY_DIR}/examples)
set(LLVM_LIBDIR_SUFFIX "" CACHE STRING "Define suffix of library directory name (32/64)" )

if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
  file(GLOB_RECURSE
    tablegenned_files_on_include_dir
    "${LLVM_MAIN_SRC_DIR}/include/llvm/*.gen")
  file(GLOB_RECURSE
    tablegenned_files_on_lib_dir
    "${LLVM_MAIN_SRC_DIR}/lib/Target/*.inc")
  if( tablegenned_files_on_include_dir OR tablegenned_files_on_lib_dir)
    message(FATAL_ERROR "Apparently there is a previous in-source build,
probably as the result of running `configure' and `make' on
${LLVM_MAIN_SRC_DIR}.
This may cause problems. The suspicious files are:
${tablegenned_files_on_lib_dir}
${tablegenned_files_on_include_dir}
Please clean the source directory.")
  endif()
endif()

set(LLVM_ALL_TARGETS
  Alpha
  ARM
  Blackfin
  CBackend
  CellSPU
  CppBackend
  Mips
  MBlaze
  MSP430
  PowerPC
  PTX
  Sparc
  SystemZ
  X86
  XCore
  )

if( MSVC )
  set(LLVM_TARGETS_TO_BUILD X86
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
else( MSVC )
  set(LLVM_TARGETS_TO_BUILD ${LLVM_ALL_TARGETS}
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
endif( MSVC )

set(CLANG_RESOURCE_DIR "" CACHE STRING
  "Relative directory from the Clang binary to its resource files.")

set(C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories clang will search for headers.")

set(LLVM_TARGET_ARCH "host"
  CACHE STRING "Set target to use for LLVM JIT or use \"host\" for automatic detection.")

set(LIT_ARGS_DEFAULT "-sv")
if (MSVC OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}"
    CACHE STRING "Default options for lit")

option(LLVM_ENABLE_THREADS "Use threads if available." ON)

if( uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
  option(LLVM_ENABLE_ASSERTIONS "Enable assertions" OFF)
else()
  option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
endif()

if( LLVM_ENABLE_ASSERTIONS )
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if( NOT MSVC )
    add_definitions( -D_DEBUG )
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
    add_definitions( -UNDEBUG )
  endif()
else()
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
    add_definitions( -DNDEBUG )
  endif()
endif()

if( LLVM_TARGETS_TO_BUILD STREQUAL "all" )
  set( LLVM_TARGETS_TO_BUILD ${LLVM_ALL_TARGETS} )
endif()

set(LLVM_ENUM_TARGETS "")
foreach(c ${LLVM_TARGETS_TO_BUILD})
  list(FIND LLVM_ALL_TARGETS ${c} idx)
  if( idx LESS 0 )
    message(FATAL_ERROR "The target `${c}' does not exist.
    It should be one of\n${LLVM_ALL_TARGETS}")
  else()
    set(LLVM_ENUM_TARGETS "${LLVM_ENUM_TARGETS}LLVM_TARGET(${c})\n")
  endif()
endforeach(c)

# Produce llvm/Config/Targets.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/Targets.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/Targets.def
  )

set(llvm_builded_incs_dir ${LLVM_BINARY_DIR}/include/llvm)

include(AddLLVMDefinitions)

if(WIN32)
  if(CYGWIN)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
  else(CYGWIN)
    set(LLVM_ON_WIN32 1)
    set(LLVM_ON_UNIX 0)
  endif(CYGWIN)
  set(LTDL_SHLIB_EXT ".dll")
  set(EXEEXT ".exe")
  # Maximum path length is 160 for non-unicode paths
  set(MAXPATHLEN 160)
else(WIN32)
  if(UNIX)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
    if(APPLE)
      set(LTDL_SHLIB_EXT ".dylib")
    else(APPLE)
      set(LTDL_SHLIB_EXT ".so")
    endif(APPLE)
    set(EXEEXT "")
    # FIXME: Maximum path length is currently set to 'safe' fixed value
    set(MAXPATHLEN 2024)
  else(UNIX)
    MESSAGE(SEND_ERROR "Unable to determine platform")
  endif(UNIX)
endif(WIN32)

include(config-ix)

option(LLVM_ENABLE_PIC "Build Position-Independent Code" ON)

set(ENABLE_PIC 0)
if( LLVM_ENABLE_PIC )
 if( XCODE )
   # Xcode has -mdynamic-no-pic on by default, which overrides -fPIC. I don't
   # know how to disable this, so just force ENABLE_PIC off for now.
   message(STATUS "Warning: -fPIC not supported with Xcode.")
 else( XCODE )
   if( SUPPORTS_FPIC_FLAG )
      message(STATUS "Building with -fPIC")
      add_llvm_definitions(-fPIC)
      set(ENABLE_PIC 1)
   else( SUPPORTS_FPIC_FLAG )
      message(STATUS "Warning: -fPIC not supported.")
   endif()
 endif()
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )

# set(CMAKE_VERBOSE_MAKEFILE true)

add_llvm_definitions( -D__STDC_LIMIT_MACROS )
add_llvm_definitions( -D__STDC_CONSTANT_MACROS )

# MSVC has a gazillion warnings with this.
if( MSVC )
  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." OFF)
else( MSVC )
  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)
endif()

option(LLVM_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LLVM_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)

if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )
  # TODO: support other platforms and toolchains.
  option(LLVM_BUILD_32_BITS "Build 32 bits executables and libraries." OFF)
  if( LLVM_BUILD_32_BITS )
    message(STATUS "Building 32 bits executables and libraries.")
    add_llvm_definitions( -m32 )
    list(APPEND CMAKE_EXE_LINKER_FLAGS -m32)
    list(APPEND CMAKE_SHARED_LINKER_FLAGS -m32)
  endif( LLVM_BUILD_32_BITS )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )

if( MSVC )
  include(ChooseMSVCCRT)

  add_llvm_definitions( -D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS )
  add_llvm_definitions( -D_SCL_SECURE_NO_WARNINGS -DCRT_NONSTDC_NO_WARNINGS )
  add_llvm_definitions( -D_SCL_SECURE_NO_DEPRECATE )
  add_llvm_definitions( -wd4146 -wd4503 -wd4996 -wd4800 -wd4244 -wd4624 )
  add_llvm_definitions( -wd4355 -wd4715 -wd4180 -wd4345 -wd4224 -wd4267 )

  # Suppress 'new behavior: elements of array 'array' will be default initialized'
  add_llvm_definitions( -wd4351 )

  # Enable warnings
  if (LLVM_ENABLE_WARNINGS)
    add_llvm_definitions( /W4 /Wall )
    if (LLVM_ENABLE_PEDANTIC)
      # No MSVC equivalent available
    endif (LLVM_ENABLE_PEDANTIC)
  endif (LLVM_ENABLE_WARNINGS)
  if (LLVM_ENABLE_WERROR)
    add_llvm_definitions( /WX )
  endif (LLVM_ENABLE_WERROR)
elseif( CMAKE_COMPILER_IS_GNUCXX )
  if (LLVM_ENABLE_WARNINGS)
    add_llvm_definitions( -Wall -W -Wno-unused-parameter -Wwrite-strings )
    if (LLVM_ENABLE_PEDANTIC)
      add_llvm_definitions( -pedantic -Wno-long-long )
    endif (LLVM_ENABLE_PEDANTIC)
  endif (LLVM_ENABLE_WARNINGS)
  if (LLVM_ENABLE_WERROR)
    add_llvm_definitions( -Werror )
  endif (LLVM_ENABLE_WERROR)
endif( MSVC )

include_directories( ${LLVM_BINARY_DIR}/include ${LLVM_MAIN_INCLUDE_DIR})

if( ${CMAKE_SYSTEM_NAME} MATCHES SunOS )
   SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-include llvm/System/Solaris.h")
endif( ${CMAKE_SYSTEM_NAME} MATCHES SunOS )

include(AddLLVM)
include(TableGen)

if( MINGW )
  get_system_libs(LLVM_SYSTEM_LIBS_LIST)
  foreach(l ${LLVM_SYSTEM_LIBS_LIST})
    set(LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS} -l${l}")
  endforeach()
  set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES}${LLVM_SYSTEM_LIBS}")
  set(CMAKE_C_STANDARD_LIBRARIES "${CMAKE_C_STANDARD_LIBRARIES}${LLVM_SYSTEM_LIBS}")
endif()

add_subdirectory(lib/Support)
add_subdirectory(lib/System)

# Everything else depends on Support and System:
set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${LLVM_LIBS} )

set(LLVM_TABLEGEN "tblgen" CACHE
  STRING "Native TableGen executable. Saves building one when cross-compiling.")
# Effective tblgen executable to be used:
set(LLVM_TABLEGEN_EXE ${LLVM_TABLEGEN})

add_subdirectory(utils/TableGen)

if( CMAKE_CROSSCOMPILING )
  # This adds a dependency on target `tblgen', so must go after utils/TableGen
  include( CrossCompileLLVM )
endif( CMAKE_CROSSCOMPILING )

add_subdirectory(include/llvm)

add_subdirectory(lib/VMCore)
add_subdirectory(lib/CodeGen)
add_subdirectory(lib/CodeGen/SelectionDAG)
add_subdirectory(lib/CodeGen/AsmPrinter)
add_subdirectory(lib/Bitcode/Reader)
add_subdirectory(lib/Bitcode/Writer)
add_subdirectory(lib/Transforms/Utils)
add_subdirectory(lib/Transforms/Instrumentation)
add_subdirectory(lib/Transforms/InstCombine)
add_subdirectory(lib/Transforms/Scalar)
add_subdirectory(lib/Transforms/IPO)
add_subdirectory(lib/Transforms/Hello)
add_subdirectory(lib/Linker)
add_subdirectory(lib/Analysis)
add_subdirectory(lib/Analysis/IPA)
add_subdirectory(lib/MC)
add_subdirectory(lib/MC/MCParser)
add_subdirectory(lib/MC/MCDisassembler)

add_subdirectory(utils/FileCheck)
add_subdirectory(utils/count)
add_subdirectory(utils/not)
add_subdirectory(utils/llvm-lit)

set(LLVM_ENUM_ASM_PRINTERS "")
set(LLVM_ENUM_ASM_PARSERS "")
set(LLVM_ENUM_DISASSEMBLERS "")
foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(lib/Target/${t})
  add_subdirectory(lib/Target/${t}/TargetInfo)
  if( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/InstPrinter/CMakeLists.txt )
    add_subdirectory(lib/Target/${t}/InstPrinter)
    set(LLVM_ENUM_ASM_PRINTERS
      "${LLVM_ENUM_ASM_PRINTERS}LLVM_ASM_PRINTER(${t})\n")
  endif( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/InstPrinter/CMakeLists.txt )
  if( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/AsmParser/CMakeLists.txt )
    add_subdirectory(lib/Target/${t}/AsmParser)
    set(LLVM_ENUM_ASM_PARSERS 
      "${LLVM_ENUM_ASM_PARSERS}LLVM_ASM_PARSER(${t})\n")
  endif( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/AsmParser/CMakeLists.txt )
  if( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/Disassembler/CMakeLists.txt )
    add_subdirectory(lib/Target/${t}/Disassembler)
    set(LLVM_ENUM_DISASSEMBLERS
      "${LLVM_ENUM_DISASSEMBLERS}LLVM_DISASSEMBLER(${t})\n")
  endif( EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Target/${t}/Disassembler/CMakeLists.txt )
  set(CURRENT_LLVM_TARGET)
endforeach(t)

# Produce llvm/Config/AsmPrinters.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/AsmPrinters.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/AsmPrinters.def
  )

# Produce llvm/Config/AsmParsers.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/AsmParsers.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/AsmParsers.def
  )

# Produce llvm/Config/Disassemblers.def
configure_file(
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Config/Disassemblers.def.in
  ${LLVM_BINARY_DIR}/include/llvm/Config/Disassemblers.def
  )

add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/ExecutionEngine/Interpreter)
add_subdirectory(lib/ExecutionEngine/JIT)
add_subdirectory(lib/Target)
add_subdirectory(lib/AsmParser)
add_subdirectory(lib/Archive)

add_subdirectory(projects)

option(LLVM_BUILD_TOOLS
  "Build the LLVM tools. If OFF, just generate build targets." ON)
option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
if( LLVM_INCLUDE_TOOLS )
  add_subdirectory(tools)
endif()

option(LLVM_BUILD_EXAMPLES
  "Build the LLVM example programs. If OFF, just generate build targets." OFF)
option(LLVM_INCLUDE_EXAMPLES "Generate build targets for the LLVM examples" ON)
if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()

option(LLVM_BUILD_TESTS
  "Build LLVM unit tests. If OFF, just generate build targes." OFF)
option(LLVM_INCLUDE_TESTS "Generate build targets for the LLVM unit tests." ON)
if( LLVM_INCLUDE_TESTS )
  add_subdirectory(test)
  add_subdirectory(utils/unittest)
  add_subdirectory(unittests)
  if (MSVC)
    # This utility is used to prevent chrashing tests from calling Dr. Watson on
    # Windows.
    add_subdirectory(utils/KillTheDoctor)
  endif()
endif()

add_subdirectory(cmake/modules)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.td"
  PATTERN "*.inc"
  PATTERN ".svn" EXCLUDE
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  # Exclude include/llvm/CMakeFiles/intrinsics_gen.dir, matched by "*.def"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN ".svn" EXCLUDE
  )

# TODO: make and install documentation.

set(CPACK_PACKAGE_VENDOR "LLVM")
set(CPACK_PACKAGE_VERSION_MAJOR 2)
set(CPACK_PACKAGE_VERSION_MINOR 9)
add_version_info_from_vcs(CPACK_PACKAGE_VERSION_PATCH)
include(CPack)
