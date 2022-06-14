# This is a helper function and not a build rule. It is to be used by the
# various test rules to generate the full list of object files
# recursively produced by "add_entrypoint_object" and "add_object_library"
# targets.
# Usage:
#   get_object_files_for_test(<result var>
#                             <skipped_entrypoints_var>
#                             <target0> [<target1> ...])
#
#   The list of object files is collected in <result_var>.
#   If skipped entrypoints were found, then <skipped_entrypoints_var> is
#   set to a true value.
#   targetN is either an "add_entrypoint_target" target or an
#   "add_object_library" target.
function(get_object_files_for_test result skipped_entrypoints_list)
  set(object_files "")
  set(skipped_list "")
  foreach(dep IN LISTS ARGN)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT dep_type)
      # Target for which TARGET_TYPE property is not set do not
      # provide any object files.
      continue()
    endif()

    if(${dep_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
      get_target_property(dep_object_files ${dep} "OBJECT_FILES")
      if(dep_object_files)
        list(APPEND object_files ${dep_object_files})
      endif()
    elseif(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
      get_target_property(is_skipped ${dep} "SKIPPED")
      if(is_skipped)
        list(APPEND skipped_list ${dep})
        continue()
      endif()
      get_target_property(object_file_raw ${dep} "OBJECT_FILE_RAW")
      if(object_file_raw)
        list(APPEND object_files ${object_file_raw})
      endif()
    endif()

    get_target_property(indirect_deps ${dep} "DEPS")
    get_object_files_for_test(
        indirect_objfiles indirect_skipped_list ${indirect_deps})
    list(APPEND object_files ${indirect_objfiles})
    if(indirect_skipped_list)
      list(APPEND skipped_list ${indirect_skipped_list})
    endif()
  endforeach(dep)
  list(REMOVE_DUPLICATES object_files)
  set(${result} ${object_files} PARENT_SCOPE)
  list(REMOVE_DUPLICATES skipped_list)
  set(${skipped_entrypoints_list} ${skipped_list} PARENT_SCOPE)
endfunction(get_object_files_for_test)

# Rule to add a libc unittest.
# Usage
#    add_libc_unittest(
#      <target name>
#      SUITE <name of the suite this test belongs to>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#      COMPILE_OPTIONS <list of special compile options for this target>
#      LINK_LIBRARIES <list of linking libraries for this target>
#    )
function(create_libc_unittest fq_target_name)
  if(NOT LLVM_INCLUDE_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    "LIBC_UNITTEST"
    "NO_RUN_POSTBUILD;NO_LIBC_UNITTEST_TEST_MAIN" # Optional arguments
    "SUITE;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;LINK_LIBRARIES;FLAGS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_UNITTEST_SRCS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT LIBC_UNITTEST_DEPENDS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_deps_list(fq_deps_list ${LIBC_UNITTEST_DEPENDS})
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    # If a test is OS/target machine independent, it has to be skipped if the
    # OS/target machine combination does not provide any dependent entrypoints.
    # If a test is OS/target machine specific, then such a test will live is a
    # OS/target machine specific directory and will be skipped at the directory
    # level if required.
    #
    # There can potentially be a setup like this: A unittest is setup for a
    # OS/target machine independent object library, which in turn depends on a
    # machine specific object library. Such a test would be testing internals of
    # the libc and it is assumed that they will be rare in practice. So, they
    # can be skipped in the corresponding CMake files using platform specific
    # logic. This pattern is followed in the loader tests for example.
    #
    # Another pattern that is present currently is to detect machine
    # capabilities and add entrypoints and tests accordingly. That approach is
    # much lower level approach and is independent of the kind of skipping that
    # is happening here at the entrypoint level.

    set(msg "Skipping unittest ${fq_target_name} as it has missing deps: "
            "${skipped_entrypoints_list}.")
    message(STATUS ${msg})
    return()
  endif()

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding unit test ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_OBJECT_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_UNITTEST_SRCS}
    ${LIBC_UNITTEST_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )
  target_compile_options(
    ${fq_target_name}
    PRIVATE ${LIBC_COMPILE_OPTIONS_DEFAULT}
  )
  if(LIBC_UNITTEST_COMPILE_OPTIONS)
    target_compile_options(
      ${fq_target_name}
      PRIVATE ${LIBC_UNITTEST_COMPILE_OPTIONS}
    )
  endif()
  if(NOT LIBC_UNITTEST_CXX_STANDARD)
    set(LIBC_UNITTEST_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      CXX_STANDARD ${LIBC_UNITTEST_CXX_STANDARD}
  )

  # Test object files will depend on LINK_LIBRARIES passed down from `add_fp_unittest`
  set(link_libraries ${link_object_files} ${LIBC_UNITTEST_LINK_LIBRARIES})

  set_target_properties(${fq_target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_target_name}
    ${fq_deps_list}
  )

  # LibcUnitTest and libc_test_utils should not depend on anything in LINK_LIBRARIES.
  if(NO_LIBC_UNITTEST_TEST_MAIN)
    list(APPEND link_libraries LibcUnitTest libc_test_utils)
  else()
    list(APPEND link_libraries LibcUnitTest LibcUnitTestMain libc_test_utils)
  endif()

  target_link_libraries(${fq_target_name} PRIVATE ${link_libraries})

  if(NOT LIBC_UNITTEST_NO_RUN_POSTBUILD)
    add_custom_command(
      TARGET ${fq_target_name}
      POST_BUILD
      COMMAND $<TARGET_FILE:${fq_target_name}>
    )
  endif()

  if(LIBC_UNITTEST_SUITE)
    add_dependencies(
      ${LIBC_UNITTEST_SUITE}
      ${fq_target_name}
    )
  endif()
endfunction(create_libc_unittest)

# Internal function, used by `add_libc_unittest`.
function(expand_flags_for_libc_unittest target_name flags)
  cmake_parse_arguments(
    "EXPAND_FLAGS"
    "IGNORE_MARKER" # No Optional arguments
    "" # No Single-value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  list(LENGTH flags nflags)
  if(NOT ${nflags})
    create_libc_unittest(
      ${target_name}
      DEPENDS "${EXPAND_FLAGS_DEPENDS}"
      FLAGS "${EXPAND_FLAGS_FLAGS}"
      "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
    )
    return()
  endif()

  list(GET flags 0 flag)
  list(REMOVE_AT flags 0)
  extract_flag_modifier(${flag} real_flag modifier)

  if(NOT "${modifier}" STREQUAL "NO")
    expand_flags_for_libc_unittest(
      ${target_name}
      "${flags}"
      DEPENDS "${EXPAND_FLAGS_DEPENDS}" IGNORE_MARKER
      FLAGS "${EXPAND_FLAGS_FLAGS}" IGNORE_MARKER
      "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
    )
  endif()

  if("${real_flag}" STREQUAL "" OR "${modifier}" STREQUAL "ONLY")
    return()
  endif()

  set(NEW_FLAGS ${EXPAND_FLAGS_FLAGS})
  list(REMOVE_ITEM NEW_FLAGS ${flag})
  get_fq_dep_list_without_flag(NEW_DEPS ${real_flag} ${EXPAND_FLAGS_DEPENDS})

  # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
  # `flag__ONLY` do not.
  if(NOT "${modifier}")
    set(TARGET_NAME "${target_name}.__NO_${flag}")
  else()
    set(TARGET_NAME "${target_name}")
  endif()

  expand_flags_for_libc_unittest(
    ${TARGET_NAME}
    "${flags}"
    DEPENDS "${NEW_DEPS}" IGNORE_MARKER
    FLAGS "${NEW_FLAGS}" IGNORE_MARKER
    "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
  )
endfunction(expand_flags_for_libc_unittest)

function(add_libc_unittest target_name)
  cmake_parse_arguments(
    "ADD_TO_EXPAND"
    "" # Optional arguments
    "" # Single value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_TO_EXPAND_DEPENDS AND ("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS"))
    message(STATUS "Gathering FLAGS from dependencies for ${fq_target_name}")
  endif()

  get_fq_deps_list(fq_deps_list ${ADD_TO_EXPAND_DEPENDS})
  get_flags_from_dep_list(deps_flag_list ${fq_deps_list})
  
  list(APPEND ADD_TO_EXPAND_FLAGS ${deps_flag_list})
  remove_duplicated_flags("${ADD_TO_EXPAND_FLAGS}" flags)
  list(SORT flags)

  if(SHOW_INTERMEDIATE_OBJECTS AND flags)
    message(STATUS "Unit test ${fq_target_name} has FLAGS: ${flags}")
  endif()

  expand_flags_for_libc_unittest(
    ${fq_target_name}
    "${flags}"
    DEPENDS ${fq_deps_list} IGNORE_MARKER
    FLAGS ${flags} IGNORE_MARKER
    ${ADD_TO_EXPAND_UNPARSED_ARGUMENTS}
  )
endfunction(add_libc_unittest)

function(add_libc_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(check-llvmlibc ${suite_name})
endfunction(add_libc_testsuite)

function(add_libc_exhaustive_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(exhaustive-check-libc ${suite_name})
endfunction(add_libc_exhaustive_testsuite)

function(add_libc_long_running_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(libc-long-running-tests ${suite_name})
endfunction(add_libc_long_running_testsuite)

# Rule to add a fuzzer test.
# Usage
#    add_libc_fuzzer(
#      <target name>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#    )
function(add_libc_fuzzer target_name)
  cmake_parse_arguments(
    "LIBC_FUZZER"
    "" # No optional arguments
    "" # Single value arguments
    "SRCS;HDRS;DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_FUZZER_SRCS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT LIBC_FUZZER_DEPENDS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${LIBC_FUZZER_DEPENDS})
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    set(msg "Skipping fuzzer target ${fq_target_name} as it has missing deps: "
            "${skipped_entrypoints_list}.")
    message(STATUS ${msg})
    add_custom_target(${fq_target_name})

    # A post build custom command is used to avoid running the command always.
    add_custom_command(
      TARGET ${fq_target_name}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo ${msg}
    )
    return()
  endif()

  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_FUZZER_SRCS}
    ${LIBC_FUZZER_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )

  target_link_libraries(${fq_target_name} PRIVATE ${link_object_files})

  set_target_properties(${fq_target_name}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_target_name}
    ${fq_deps_list}
  )
  add_dependencies(libc-fuzzer ${fq_target_name})
endfunction(add_libc_fuzzer)

# Rule to add an integration test. An integration test is like a unit test
# but does not use the system libc. Not even the loader from the system libc
# is linked to the final executable. The final exe is fully statically linked.
# The libc that the final exe links to consists of only the object files of
# the DEPENDS targets.
# 
# Usage:
#   add_integration_test(
#     <target name>
#     SUITE <the suite to which the test should belong>
#     SRCS <src1.cpp> [src2.cpp ...]
#     HDRS [hdr1.cpp ...]
#     LOADER <fully qualified loader target name>
#     DEPENDS <list of entrypoint or other object targets>
#     ARGS <list of command line arguments to be passed to the test>
#     ENV <list of environment variables to set before running the test>
#   )
#
# The loader target should provide a property named LOADER_OBJECT which is
# the full path to the object file produces when the loader is built.
#
# The DEPENDS list can be empty. If not empty, it should be a list of
# targets added with add_entrypoint_object or add_object_library.
function(add_integration_test test_name)
  get_fq_target_name(${test_name} fq_target_name)
  if(NOT (${LIBC_TARGET_OS} STREQUAL "linux"))
    message(STATUS "Skipping ${fq_target_name} as it is not available on ${LIBC_TARGET_OS}.")
    return()
  endif()
  cmake_parse_arguments(
    "INTEGRATION_TEST"
    "" # No optional arguments
    "SUITE;LOADER" # Single value arguments
    "SRCS;HDRS;DEPENDS;ARGS;ENV" # Multi-value arguments
    ${ARGN}
  )

  if(NOT INTEGRATION_TEST_SUITE)
    message(FATAL_ERROR "SUITE not specified for ${fq_target_name}")
  endif()
  if(NOT INTEGRATION_TEST_LOADER)
    message(FATAL_ERROR "The LOADER to link to the integration test is missing.")
  endif()
  if(NOT INTEGRATION_TEST_SRCS)
    message(FATAL_ERROR "The SRCS list for add_integration_test is missing.")
  endif()

  get_fq_target_name(${test_name}.libc fq_libc_target_name)

  get_fq_deps_list(fq_deps_list ${INTEGRATION_TEST_DEPENDS})
  # Add memory functions to which compilers can emit calls.
  list(APPEND fq_deps_list
          libc.src.string.bcmp
          libc.src.string.bzero
          libc.src.string.memcmp
          libc.src.string.memcpy
          libc.src.string.memset)
  list(REMOVE_DUPLICATES fq_deps_list)
  # TODO: Instead of gathering internal object files from entrypoints,
  # collect the object files with public names of entrypoints.
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    message(STATUS "Skipping ${fq_target_name} as it has skipped deps.")
    return()
  endif()

  # Create a sysroot structure
  set(sysroot ${CMAKE_CURRENT_BINARY_DIR}/${test_name}/sysroot)
  file(MAKE_DIRECTORY ${sysroot})
  file(MAKE_DIRECTORY ${sysroot}/include)
  set(sysroot_lib ${sysroot}/lib)
  file(MAKE_DIRECTORY ${sysroot_lib})
  get_target_property(loader_object_file ${INTEGRATION_TEST_LOADER} LOADER_OBJECT)
  get_target_property(crti_object_file libc.loader.linux.crti LOADER_OBJECT)
  get_target_property(crtn_object_file libc.loader.linux.crtn LOADER_OBJECT)
  set(dummy_archive $<TARGET_PROPERTY:libc_integration_test_dummy,ARCHIVE_OUTPUT_DIRECTORY>/lib$<TARGET_PROPERTY:libc_integration_test_dummy,ARCHIVE_OUTPUT_NAME>.a)
  if(NOT loader_object_file)
    message(FATAL_ERROR "Missing LOADER_OBJECT property of ${INTEGRATION_TEST_LOADER}.")
  endif()
  set(loader_dst ${sysroot_lib}/${LIBC_TARGET_ARCHITECTURE}-linux-gnu/crt1.o)
  add_custom_command(
    OUTPUT ${loader_dst} ${sysroot}/lib/crti.o ${sysroot}/lib/crtn.o ${sysroot}/lib/libm.a ${sysroot}/lib/libc++.a
    COMMAND cmake -E copy ${loader_object_file} ${loader_dst}
    COMMAND cmake -E copy ${crti_object_file} ${sysroot}/lib
    COMMAND cmake -E copy ${crtn_object_file} ${sysroot}/lib
    # We copy the dummy archive as libm.a and libc++.a as the compiler drivers expect them.
    COMMAND cmake -E copy ${dummy_archive} ${sysroot}/lib/libm.a
    COMMAND cmake -E copy ${dummy_archive} ${sysroot}/lib/libc++.a
    DEPENDS ${INTEGRATION_TEST_LOADER} libc.loader.linux.crti libc.loader.linux.crtn libc_integration_test_dummy
  )
  add_custom_target(
    ${fq_target_name}.__copy_loader__
    DEPENDS ${loader_dst}
  )

  add_library(
    ${fq_libc_target_name}
    STATIC
    ${link_object_files}
  )
  set_target_properties(${fq_libc_target_name} PROPERTIES ARCHIVE_OUTPUT_NAME c)
  set_target_properties(${fq_libc_target_name} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${sysroot_lib})

  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${INTEGRATION_TEST_SRCS}
    ${INTEGRATION_TEST_HDRS}
  )
  set_target_properties(${fq_target_name}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )
  # We set a number of link options to prevent picking up system libc binaries.
  # Also, we restrict the integration tests to fully static executables. The
  # rtlib is set to compiler-rt to make the compiler drivers pick up the compiler
  # runtime binaries using full paths. Otherwise, files like crtbegin.o are passed
  # as is (and not as paths like /usr/lib/.../crtbegin.o).
  target_link_options(${fq_target_name} PRIVATE --sysroot=${sysroot} -static -stdlib=libc++ --rtlib=compiler-rt)
  add_dependencies(${fq_target_name}
                   ${fq_target_name}.__copy_loader__
                   ${fq_libc_target_name}
                   libc.utils.IntegrationTest.test)

  add_custom_command(
    TARGET ${fq_target_name}
    POST_BUILD
    COMMAND ${INTEGRATION_TEST_ENV} $<TARGET_FILE:${fq_target_name}> ${INTEGRATION_TEST_ARGS}
  )

  add_dependencies(${INTEGRATION_TEST_SUITE} ${fq_target_name})
endfunction(add_integration_test)
