add_custom_target(LLDBUnitTests)
set_target_properties(LLDBUnitTests PROPERTIES FOLDER "LLDB tests")

include_directories(${LLDB_SOURCE_ROOT})

set(LLDB_GTEST_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/gtest_common.h)
if (MSVC)
  list(APPEND LLVM_COMPILE_FLAGS /FI ${LLDB_GTEST_COMMON_INCLUDE})
else ()
  list(APPEND LLVM_COMPILE_FLAGS -include ${LLDB_GTEST_COMMON_INCLUDE})
endif ()

# add_lldb_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against
function(add_lldb_unittest test_name)
  add_unittest(LLDBUnitTests ${test_name} ${ARGN})
endfunction()

add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(Plugins)
add_subdirectory(Utility)
