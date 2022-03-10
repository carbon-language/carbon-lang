# Check if compile-rt library file path exists.
# If found, cache the path in:
#    COMPILER_RT_LIBRARY-<name>-<target>
# If err_flag is true OR path not found, emit a message and set:
#    COMPILER_RT_LIBRARY-<name>-<target> to NOTFOUND
function(cache_compiler_rt_library err_flag name target library_file)
  if(err_flag OR NOT EXISTS "${library_file}")
    message(STATUS "Failed to find compiler-rt ${name} library for ${target}")
    set(COMPILER_RT_LIBRARY_${name}_${target} "NOTFOUND" CACHE INTERNAL
        "compiler-rt ${name} library for ${target}")
  else()
    message(STATUS "Found compiler-rt ${name} library: ${library_file}")
    set(COMPILER_RT_LIBRARY_${name}_${target} "${library_file}" CACHE INTERNAL
        "compiler-rt ${name} library for ${target}")
  endif()
endfunction()

function(get_component_name name variable)
  if(APPLE)
    if(NOT name MATCHES "builtins.*")
      set(component_name "${name}_")
    endif()
    # TODO: Support ios, tvos and watchos as well.
    set(component_name "${component_name}osx")
  else()
    set(component_name "${name}")
  endif()
  set(${variable} "${component_name}" PARENT_SCOPE)
endfunction()

# Find the path to compiler-rt library `name` (e.g. "builtins") for the
# specified `TARGET` (e.g. "x86_64-linux-gnu") and return it in `variable`.
# This calls cache_compiler_rt_library that caches the path to speed up
# repeated invocations with the same `name` and `target`.
function(find_compiler_rt_library name variable)
  cmake_parse_arguments(ARG "" "TARGET;FLAGS" "" ${ARGN})
  # While we can use compiler-rt runtimes with other compilers, we need to
  # query the compiler for runtime location and thus we require Clang.
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(${variable} "NOTFOUND" PARENT_SCOPE)
    return()
  endif()
  set(target "${ARG_TARGET}")
  if(NOT target AND CMAKE_CXX_COMPILER_TARGET)
    set(target "${CMAKE_CXX_COMPILER_TARGET}")
  endif()
  if(NOT DEFINED COMPILER_RT_LIBRARY_builtins_${target})
    # If the cache variable is not defined, invoke Clang and then
    # set it with cache_compiler_rt_library.
    set(clang_command ${CMAKE_CXX_COMPILER} "${ARG_FLAGS}")
    if(target)
      list(APPEND clang_command "--target=${target}")
    endif()
    get_property(cxx_flags CACHE CMAKE_CXX_FLAGS PROPERTY VALUE)
    string(REPLACE " " ";" cxx_flags "${cxx_flags}")
    list(APPEND clang_command ${cxx_flags})
    execute_process(
      COMMAND ${clang_command} "--rtlib=compiler-rt" "-print-libgcc-file-name"
      RESULT_VARIABLE had_error
      OUTPUT_VARIABLE library_file
    )
    string(STRIP "${library_file}" library_file)
    file(TO_CMAKE_PATH "${library_file}" library_file)
    get_filename_component(dirname ${library_file} DIRECTORY)
    if(APPLE)
      execute_process(
        COMMAND ${clang_command} "--print-resource-dir"
        RESULT_VARIABLE had_error
        OUTPUT_VARIABLE resource_dir
      )
      string(STRIP "${resource_dir}" resource_dir)
      set(dirname "${resource_dir}/lib/darwin")
    endif()
    get_filename_component(basename ${library_file} NAME)
    if(basename MATCHES "libclang_rt\.([a-z0-9_\-]+)\.a")
      set(from_name ${CMAKE_MATCH_1})
      get_component_name(${CMAKE_MATCH_1} to_name)
      string(REPLACE "${from_name}" "${to_name}" basename "${basename}")
      set(library_file "${dirname}/${basename}")
      cache_compiler_rt_library(${had_error} builtins "${target}" "${library_file}")
    endif()
  endif()
  if(NOT COMPILER_RT_LIBRARY_builtins_${target})
    set(${variable} "NOTFOUND" PARENT_SCOPE)
    return()
  endif()
  if(NOT DEFINED COMPILER_RT_LIBRARY_${name}_${target})
    # Clang gives only the builtins library path. Other library paths are
    # obtained by substituting "builtins" with ${name} in the builtins
    # path and then checking if the resultant path exists. The result of
    # this check is also cached by cache_compiler_rt_library.
    set(library_file "${COMPILER_RT_LIBRARY_builtins_${target}}")
    if(library_file MATCHES ".*libclang_rt\.([a-z0-9_\-]+)\.a")
      set(from_name ${CMAKE_MATCH_0})
      get_component_name(${name} to_name)
      string(REPLACE "${from_name}" "${to_name}" library_file "${library_file}")
      cache_compiler_rt_library(FALSE "${name}" "${target}" "${library_file}")
    endif()
  endif()
  set(${variable} "${COMPILER_RT_LIBRARY_${name}_${target}}" PARENT_SCOPE)
endfunction()
