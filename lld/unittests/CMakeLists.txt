add_custom_target(LLDUnitTests)
set_target_properties(LLDUnitTests PROPERTIES FOLDER "lld tests")

# add_lld_unittest(test_dirname file1.cpp file2.cpp)
#
# Will compile the list of files together and link against lld
# Produces a binary named 'basename(test_dirname)'.
function(add_lld_unittest test_dirname)
  add_unittest(LLDUnitTests ${test_dirname} ${ARGN})
endfunction()

set(LLVM_LINK_COMPONENTS
    support
    )

add_lld_unittest(CoreTests ErrorOrTest.cpp)
