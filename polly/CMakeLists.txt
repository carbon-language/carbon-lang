# Check if this is a in tree build.
if (NOT DEFINED LLVM_MAIN_SRC_DIR)
  project(Polly)
  cmake_minimum_required(VERSION 2.8)

  # Where is LLVM installed?
  set(LLVM_INSTALL_ROOT "" CACHE PATH "Root of LLVM install.")
  # Check if the LLVM_INSTALL_ROOT valid.
  if( NOT EXISTS ${LLVM_INSTALL_ROOT}/include/llvm )
    message(FATAL_ERROR "LLVM_INSTALL_ROOT (${LLVM_INSTALL_ROOT}) is not a valid LLVM installation.")
  endif(NOT EXISTS ${LLVM_INSTALL_ROOT}/include/llvm)
  # Add the llvm header path.
  include_directories(${LLVM_INSTALL_ROOT}/include/)
  
  # Get the system librarys that will link into LLVM.
  function(get_system_libs return_var)
    # Returns in `return_var' a list of system libraries used by LLVM.
    if( NOT MSVC )
      if( MINGW )
        set(system_libs ${system_libs} imagehlp psapi)
      elseif( CMAKE_HOST_UNIX )
        if( HAVE_LIBDL )
          set(system_libs ${system_libs} ${CMAKE_DL_LIBS})
        endif()
        if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
          set(system_libs ${system_libs} pthread)
        endif()
      endif( MINGW )
    endif( NOT MSVC )
    set(${return_var} ${system_libs} PARENT_SCOPE)
  endfunction(get_system_libs)

  # Now set the header paths.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --includedir
                  OUTPUT_VARIABLE LLVM_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  include_directories( ${LLVM_INCLUDE_DIR} )

  # And then set the cxx flags.
  execute_process(COMMAND "${LLVM_INSTALL_ROOT}/bin/llvm-config" --cxxflags
                  OUTPUT_VARIABLE LLVM_CXX_FLAGS
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${LLVM_CXX_FLAGS})
endif(NOT DEFINED LLVM_MAIN_SRC_DIR)

set(POLLY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POLLY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Add appropriate flags for GCC
if (CMAKE_COMPILER_IS_GNUCXX)
  # FIXME: Turn off exceptions, RTTI:
  # -fno-exceptions -fno-rtti
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-exceptions -fno-rtti")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

# Add path for custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${POLLY_SOURCE_DIR}/cmake")

FIND_PACKAGE(Cloog REQUIRED)
FIND_PACKAGE(Isl REQUIRED)
FIND_PACKAGE(Gmp REQUIRED)

option(POLLY_ENABLE_OPENSCOP "Enable Openscop library for scop import/export" ON)
if (POLLY_ENABLE_OPENSCOP)
  FIND_PACKAGE(OpenScop)
endif(POLLY_ENABLE_OPENSCOP)

option(POLLY_ENABLE_SCOPLIB "Enable SCoPLib library for scop import/export" ON)
if (POLLY_ENABLE_SCOPLIB)
  FIND_PACKAGE(SCoPLib)
endif(POLLY_ENABLE_SCOPLIB)

INCLUDE_DIRECTORIES( ${CLOOG_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${ISL_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${GMP_INCLUDE_DIR} )

# Support OpenScop export/import if the library is available.
if (OPENSCOP_FOUND)
  INCLUDE_DIRECTORIES( ${OPENSCOP_INCLUDE_DIR} )
endif(OPENSCOP_FOUND)
if (SCOPLIB_FOUND)
  INCLUDE_DIRECTORIES( ${SCOPLIB_INCLUDE_DIR} )
endif(SCOPLIB_FOUND)

macro(add_polly_library name)
  set(srcs ${ARGN})
  if(MSVC_IDE OR XCODE)
    file( GLOB_RECURSE headers *.h *.td *.def)
    set(srcs ${srcs} ${headers})
    string( REGEX MATCHALL "/[^/]+" split_path ${CMAKE_CURRENT_SOURCE_DIR})
    list( GET split_path -1 dir)
    file( GLOB_RECURSE headers 
      ../../include/polly${dir}/*.h)
    set(srcs ${srcs} ${headers})
  endif(MSVC_IDE OR XCODE)
  if (MODULE)
    set(libkind MODULE)
  elseif (SHARED_LIBRARY)
    set(libkind SHARED)
  else()
    set(libkind)
  endif()
  add_library( ${name} ${libkind} ${srcs} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )
  if( LLVM_USED_LIBS )
    foreach(lib ${LLVM_USED_LIBS})
      target_link_libraries( ${name} ${lib} )
    endforeach(lib)
  endif( LLVM_USED_LIBS )

  target_link_libraries( ${name} ${CLOOG_LIBRARY} ${ISL_LIBRARY} ${GMP_LIBRARY})
  if (OPENSCOP_FOUND)
    target_link_libraries( ${name} ${OPENSCOP_LIBRARY})
  endif(OPENSCOP_FOUND)
  if (SCOPLIB_FOUND)
    target_link_libraries( ${name} ${SCOPLIB_LIBRARY})
  endif(SCOPLIB_FOUND)

  if( LLVM_LINK_COMPONENTS )
    llvm_config(${name} ${LLVM_LINK_COMPONENTS})
  endif( LLVM_LINK_COMPONENTS )
  get_system_libs(llvm_system_libs)
  if( llvm_system_libs )
    target_link_libraries(${name} ${llvm_system_libs})
  endif( llvm_system_libs )
  
  if(MSVC)
    get_target_property(cflag ${name} COMPILE_FLAGS)
    if(NOT cflag)
      set(cflag "")
    endif(NOT cflag)
    set(cflag "${cflag} /Za")
    set_target_properties(${name} PROPERTIES COMPILE_FLAGS ${cflag})
  endif(MSVC)
  install(TARGETS ${name}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
endmacro(add_polly_library)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/JSON/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )

install(DIRECTORY include
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  )

add_definitions( -D_GNU_SOURCE )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
# TODO: docs.


configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/include/polly/Config/config.h.cmake
                ${POLLY_BINARY_DIR}/include/polly/Config/config.h )

