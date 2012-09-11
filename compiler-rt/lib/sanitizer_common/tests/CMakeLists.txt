set(SANITIZER_UNITTESTS
  sanitizer_allocator_test.cc
  sanitizer_common_test.cc
  sanitizer_flags_test.cc
  sanitizer_list_test.cc
  sanitizer_stackdepot_test.cc
  sanitizer_test_main.cc
  )
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT LLVM_BUILD_32_BITS)
  list(APPEND SANITIZER_UNITTESTS sanitizer_allocator64_test.cc)
endif()

include_directories(..)
include_directories(../..)

# Unittest target.
add_custom_target(SanitizerUnitTests)
set_target_properties(SanitizerUnitTests PROPERTIES
  FOLDER "Sanitizer unittests")
add_unittest(SanitizerUnitTests SanitizerUnitTest ${SANITIZER_UNITTESTS})
# Link with sanitizer runtime.
target_link_libraries(SanitizerUnitTest RTSanitizerCommon.test)
# Build unit tests with debug info.
set_property(TARGET SanitizerUnitTest APPEND_STRING
  PROPERTY COMPILE_FLAGS " -g -Werror")

# Run unittests as a part of lit testsuite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

add_lit_testsuite(check-sanitizer "Running sanitizer library unittests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS SanitizerUnitTests
  )
set_target_properties(check-sanitizer PROPERTIES FOLDER "Sanitizer unittests")
