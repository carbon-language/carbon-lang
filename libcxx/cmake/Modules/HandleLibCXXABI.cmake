
#===============================================================================
# Add an ABI library if appropriate
#===============================================================================

#
# _setup_abi: Set up the build to use an ABI library
#
# Parameters:
#   abidefines: A list of defines needed to compile libc++ with the ABI library
#   abilibs   : A list of libraries to link against
#   abifiles  : A list of files (which may be relative paths) to copy into the
#               libc++ build tree for the build.  These files will also be
#               installed alongside the libc++ headers.
#   abidirs   : A list of relative paths to create under an include directory
#               in the libc++ build directory.
#
macro(setup_abi_lib abipathvar abidefines abilibs abifiles abidirs)
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS ${abidefines})
  set(${abipathvar} "${${abipathvar}}"
    CACHE PATH
    "Paths to C++ ABI header directories separated by ';'." FORCE
    )

  # To allow for libraries installed along non-default paths we use find_library
  # to locate the ABI libraries we want. Making sure to clean the cache before
  # each run of find_library.
  set(LIBCXX_CXX_ABI_LIBRARIES "")
  foreach(alib ${abilibs})
    unset(_Res CACHE)
    find_library(_Res ${alib})
    if (${_Res} STREQUAL "_Res-NOTFOUND")
      message(FATAL_ERROR "Failed to find ABI library: ${alib}")
    else()
      message(STATUS "Adding ABI library: ${_Res}")
      list(APPEND LIBCXX_CXX_ABI_LIBRARIES ${_Res})
    endif()
  endforeach()

  set(LIBCXX_ABILIB_FILES ${abifiles})

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
  foreach(_d ${abidirs})
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/${_d}")
  endforeach()

  foreach(fpath ${LIBCXX_ABILIB_FILES})
    set(found FALSE)
    foreach(incpath ${${abipathvar}})
      if (EXISTS "${incpath}/${fpath}")
        set(found TRUE)
        get_filename_component(dstdir ${fpath} PATH)
        get_filename_component(ifile ${fpath} NAME)
        file(COPY "${incpath}/${fpath}"
          DESTINATION "${CMAKE_BINARY_DIR}/include/${dstdir}"
          )
        install(FILES "${CMAKE_BINARY_DIR}/include/${fpath}"
          DESTINATION include/c++/v1/${dstdir}
          PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
          )
        list(APPEND abilib_headers "${CMAKE_BINARY_DIR}/include/${fpath}")
      endif()
    endforeach()
    if (NOT found)
      message(FATAL_ERROR "Failed to find ${fpath}")
    endif()
  endforeach()

  add_custom_target(LIBCXX_CXX_ABI_DEPS DEPENDS ${abilib_headers})
  include_directories("${CMAKE_BINARY_DIR}/include")

endmacro()

if ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libstdc++" OR
    "${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libsupc++")
  set(_LIBSUPCXX_INCLUDE_FILES
    cxxabi.h bits/c++config.h bits/os_defines.h bits/cpu_defines.h
    bits/cxxabi_tweaks.h bits/cxxabi_forced.h
    )
  if ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libstdc++")
    set(_LIBSUPCXX_DEFINES "-DLIBSTDCXX")
    set(_LIBSUPCXX_LIBNAME stdc++)
  else()
    set(_LIBSUPCXX_DEFINES "")
    set(_LIBSUPCXX_LIBNAME supc++)
  endif()
  setup_abi_lib("LIBCXX_LIBSUPCXX_INCLUDE_PATHS"
    "-D__GLIBCXX__ ${_LIBSUPCXX_DEFINES}"
    "${_LIBSUPCXX_LIBNAME}" "${_LIBSUPCXX_INCLUDE_FILES}" "bits"
    )
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libcxxabi")
  if (LIBCXX_CXX_ABI_INTREE)
    # Link against just-built "cxxabi" target.
    set(CXXABI_LIBNAME cxxabi)
  else()
    # Assume c++abi is installed in the system, rely on -lc++abi link flag.
    set(CXXABI_LIBNAME "c++abi")
  endif()
  setup_abi_lib("LIBCXX_LIBCXXABI_INCLUDE_PATHS" ""
    ${CXXABI_LIBNAME} "cxxabi.h" ""
    )
elseif ("${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "libcxxrt")
  setup_abi_lib("LIBCXX_LIBCXXRT_INCLUDE_PATHS" "-DLIBCXXRT"
    "cxxrt" "cxxabi.h;unwind.h;unwind-arm.h;unwind-itanium.h" ""
    )
elseif (NOT "${LIBCXX_CXX_ABI_LIBNAME}" STREQUAL "none")
  message(FATAL_ERROR
    "Currently libstdc++, libsupc++, libcxxabi, libcxxrt and none are "
    "supported for c++ abi."
    )
endif ()