function(add_clang_unittest test_dirname link_components used_libs)
  separate_arguments(link_components)
  set(LLVM_LINK_COMPONENTS ${link_components})
  separate_arguments(used_libs)
  set(LLVM_USED_LIBS ${used_libs})
  string(REGEX MATCH "([^/]+)$" test_name ${test_dirname})
  if (CMAKE_BUILD_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${CLANG_BINARY_DIR}/unittests/${test_dirname}/${CMAKE_BUILD_TYPE})
  else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      ${CLANG_BINARY_DIR}/unittests/${test_dirname})
  endif()
  if( NOT LLVM_BUILD_TESTS )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_clang_executable(${test_name}Tests ${ARGN})
  add_dependencies(ClangUnitTests ${test_name}Tests)
endfunction()

add_custom_target(ClangUnitTests)

include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
add_definitions(-DGTEST_HAS_RTTI=0)
if( CMAKE_COMPILER_IS_GNUCXX )
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "-frtti" "-fno-rtti")
elseif( MSVC )
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS "/GR" "/GR-")
endif()

if (NOT LLVM_ENABLE_THREADS)
  add_definitions(-DGTEST_HAS_PTHREAD=0)
endif()

if(SUPPORTS_NO_VARIADIC_MACROS_FLAG)
  add_definitions("-Wno-variadic-macros")
endif()

add_clang_unittest(Basic
  ""
  "gtest gtest_main clangBasic"
  Basic/FileManagerTest.cpp
 )

add_clang_unittest(Frontend
  ""
  "gtest gtest_main clangFrontend"
  Frontend/FrontendActionTest.cpp
 )
