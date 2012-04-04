include(LLVMParseArguments)

# add_clang_unittest(test_dirname file1.cpp file2.cpp ...
#                    [USED_LIBS lib1 lib2]
#                    [LINK_COMPONENTS component1 component2])
#
# Will compile the list of files together and link against the clang
# libraries in the USED_LIBS list and the llvm-config components in
# the LINK_COMPONENTS list.  Produces a binary named
# 'basename(test_dirname)Tests'.
function(add_clang_unittest)
  PARSE_ARGUMENTS(CLANG_UNITTEST "USED_LIBS;LINK_COMPONENTS" "" ${ARGN})
  set(LLVM_LINK_COMPONENTS ${CLANG_UNITTEST_LINK_COMPONENTS})
  set(LLVM_USED_LIBS ${CLANG_UNITTEST_USED_LIBS})
  list(GET CLANG_UNITTEST_DEFAULT_ARGS 0 test_dirname)
  list(REMOVE_AT CLANG_UNITTEST_DEFAULT_ARGS 0)

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
  add_clang_executable(${test_name}Tests ${CLANG_UNITTEST_DEFAULT_ARGS})
  add_dependencies(ClangUnitTests ${test_name}Tests)
  set_target_properties(${test_name}Tests PROPERTIES FOLDER "Clang tests")
endfunction()

add_custom_target(ClangUnitTests)
set_target_properties(ClangUnitTests PROPERTIES FOLDER "Clang tests")

include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
add_definitions(-DGTEST_HAS_RTTI=0)
if( LLVM_COMPILER_IS_GCC_COMPATIBLE )
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
  Basic/FileManagerTest.cpp
  Basic/SourceManagerTest.cpp
  USED_LIBS gtest gtest_main clangLex
 )

add_clang_unittest(Lex
  Lex/LexerTest.cpp
  USED_LIBS gtest gtest_main clangLex
 )

add_clang_unittest(Frontend
  Frontend/FrontendActionTest.cpp
  USED_LIBS gtest gtest_main clangFrontend
 )

add_clang_unittest(Tooling
  Tooling/CompilationDatabaseTest.cpp
  Tooling/ToolingTest.cpp
  USED_LIBS gtest gtest_main clangTooling
 )
