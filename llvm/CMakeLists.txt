cmake_minimum_required(VERSION 2.6.1)

set(PACKAGE_NAME llvm)
set(PACKAGE_VERSION svn)
set(PACKAGE_BUGREPORT "ofv@wanadoo.es")
#set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

include(FindPerl)

set(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVM_MAIN_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include/llvm)
set(LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)
set(LLVM_EXAMPLES_BINARY_DIR ${LLVM_BINARY_DIR}/examples)

if( MSVC )
  set(LLVM_TARGETS_TO_BUILD X86
    CACHE STRING "Semicolon-separated list of targets to build")
else( MSVC )
  set(LLVM_TARGETS_TO_BUILD
    Alpha ARM CBackend CellSPU CppBackend IA64 Mips MSIL PIC16 PowerPC Sparc X86
    CACHE STRING "Semicolon-separated list of targets to build")
endif( MSVC )

if( NOT MSVC )
  set(CMAKE_CXX_LINK_EXECUTABLE "sh -c \"${CMAKE_CXX_LINK_EXECUTABLE}\"")
endif( NOT MSVC )

# TODO: Eliminate?
get_filename_component(llvm_include_path ${LLVM_MAIN_SRC_DIR}/include ABSOLUTE)

#get_filename_component(llvm_builded_incs_dir ${LLVM_BINARY_DIR}/include/llvm ABSOLUTE)
set(llvm_builded_incs_dir ${LLVM_BINARY_DIR}/include/llvm)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${LLVM_MAIN_SRC_DIR}/cmake"
  "${LLVM_MAIN_SRC_DIR}/cmake/modules"
  )

if(WIN32)
  set(LLVM_ON_WIN32 1)
  set(LLVM_ON_UNIX 0)
  set(LTDL_SHLIB_EXT ".dll")
  set(EXEEXT ".exe")
  # Maximum path length is 160 for non-unicode paths
  set(MAXPATHLEN 160)
else(WIN32)
  if(UNIX)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
    set(LTDL_SHLIB_EXT ".so")
    set(EXEEXT "")
    # FIXME: Maximum path length is currently set to 'safe' fixed value
    set(MAXPATHLEN 2024)
  else(UNIX)
    MESSAGE(SEND_ERROR "Unable to determine platform")
  endif(UNIX)
endif(WIN32)

if( EXISTS ${LLVM_TOOLS_BINARY_DIR}/llvm-config )
  set(HAVE_LLVM_CONFIG 1)
endif( EXISTS ${LLVM_TOOLS_BINARY_DIR}/llvm-config )
# find_file(HAVE_LLVM_CONFIG llvm-config ${LLVM_TOOLS_BINARY_DIR})
# message(STATUS ${HAVE_LLVM_CONFIG})

if( MSVC )
  message(STATUS "Configuring using MSVC hack.")
  include(config-w32)
else( MSVC )
  message(STATUS "Configuring traditional style.")
  include(config-ix)
endif( MSVC )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LLVM_TOOLS_BINARY_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/lib )

# set(CMAKE_VERBOSE_MAKEFILE true)

add_definitions( -D__STDC_LIMIT_MACROS )
add_definitions( -D__STDC_CONSTANT_MACROS )

if( LLVM_ON_UNIX )
  add_definitions( -DLLVM_ON_UNIX )
else( LLVM_ON_UNIX )
  add_definitions( -DLLVM_ON_WIN32 )
endif( LLVM_ON_UNIX )

if( MSVC )
  add_definitions( -D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS )
  add_definitions( -D_SCL_SECURE_NO_WARNINGS -DCRT_NONSTDC_NO_WARNINGS )
  add_definitions( -D_SCL_SECURE_NO_DEPRECATE )
  add_definitions( -wd4146 -wd4503 -wd4996 -wd4800 -wd4244 -wd4624 )
  add_definitions( -wd4355 -wd4715 )
endif( MSVC )

include_directories( ${LLVM_BINARY_DIR}/include ${llvm_include_path})
#link_directories( d:/dev/lib )

#add_subdirectory(lib)

include(AddLLVM)
include(AddPartiallyLinkedObject)
include(TableGen)

add_subdirectory(lib/Support)
add_subdirectory(lib/System)

# Everything else depends on Support and System:
set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${LLVM_LIBS} )

add_subdirectory(utils/TableGen)

add_custom_command(OUTPUT ${llvm_builded_incs_dir}/Intrinsics.gen
  COMMAND tblgen -gen-intrinsic -I ${llvm_include_path} ${llvm_include_path}/llvm/Intrinsics.td -o ${llvm_builded_incs_dir}/Intrinsics.gen
  DEPENDS tblgen
  COMMENT "Building intrinsics.gen...")

add_custom_target(intrinsics_gen ALL
  DEPENDS ${llvm_builded_incs_dir}/Intrinsics.gen)

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} intrinsics_gen )

add_subdirectory(lib/VMCore)
add_subdirectory(lib/CodeGen)
add_subdirectory(lib/CodeGen/SelectionDAG)
add_subdirectory(lib/CodeGen/AsmPrinter)
add_subdirectory(lib/Bitcode/Reader)
add_subdirectory(lib/Bitcode/Writer)
add_subdirectory(lib/Transforms/Utils)
add_subdirectory(lib/Transforms/Instrumentation)
add_subdirectory(lib/Transforms/Scalar)
add_subdirectory(lib/Transforms/IPO)
add_subdirectory(lib/Transforms/Hello)
add_subdirectory(lib/Linker)
add_subdirectory(lib/Analysis)
add_subdirectory(lib/Analysis/IPA)

foreach(t ${LLVM_TARGETS_TO_BUILD})
  message(STATUS "Targeting ${t}")
  add_subdirectory(lib/Target/${t})
  if( EXISTS ${CMAKE_SOURCE_DIR}/lib/Target/${t}/AsmPrinter/CMakeLists.txt )
    add_subdirectory(lib/Target/${t}/AsmPrinter)
  endif( EXISTS ${CMAKE_SOURCE_DIR}/lib/Target/${t}/AsmPrinter/CMakeLists.txt )
endforeach(t)

add_subdirectory(lib/ExecutionEngine)
add_subdirectory(lib/ExecutionEngine/Interpreter)
add_subdirectory(lib/ExecutionEngine/JIT)
add_subdirectory(lib/Target)
add_subdirectory(lib/AsmParser)
add_subdirectory(lib/Debugger)
add_subdirectory(lib/Archive)

add_subdirectory(tools)

add_subdirectory(examples)

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  PATTERN "*.cmake" EXCLUDE
  PATTERN "*.in" EXCLUDE
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include
  DESTINATION .
  )

# TODO: make and install documentation.
