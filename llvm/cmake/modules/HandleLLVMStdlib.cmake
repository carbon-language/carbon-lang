# This CMake module is responsible for setting the standard library to libc++
# if the user has requested it.

if(NOT DEFINED LLVM_STDLIB_HANDLED)
  set(LLVM_STDLIB_HANDLED ON)

  if(CMAKE_COMPILER_IS_GNUCXX)
    set(LLVM_COMPILER_IS_GCC_COMPATIBLE ON)
  elseif( MSVC )
    set(LLVM_COMPILER_IS_GCC_COMPATIBLE OFF)
  elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
    set(LLVM_COMPILER_IS_GCC_COMPATIBLE ON)
  endif()

  function(append value)
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endfunction()

  include(CheckCXXCompilerFlag)
  if(LLVM_ENABLE_LIBCXX)
    if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
      check_cxx_compiler_flag("-stdlib=libc++" CXX_SUPPORTS_STDLIB)
      if(CXX_SUPPORTS_STDLIB)
        append("-stdlib=libc++"
          CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
          CMAKE_MODULE_LINKER_FLAGS)
        if(LLVM_ENABLE_LIBCXXABI)
          append("-lc++abi"
            CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
            CMAKE_MODULE_LINKER_FLAGS)
        endif()
      else()
        message(WARNING "Can't specify libc++ with '-stdlib='")
      endif()
    else()
      message(WARNING "Not sure how to specify libc++ for this compiler")
    endif()
  endif()
endif()
