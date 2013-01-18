include_directories(../rtl)

add_custom_target(TsanUnitTests)
set_target_properties(TsanUnitTests PROPERTIES
  FOLDER "TSan unittests")
function(add_tsan_unittest testname)
  # Build unit tests only on 64-bit Linux.
  if(UNIX AND NOT APPLE
      AND CAN_TARGET_x86_64
      AND CMAKE_SIZEOF_VOID_P EQUAL 8
      AND NOT LLVM_BUILD_32_BITS)
    add_unittest(TsanUnitTests ${testname} ${ARGN})
    # Link with TSan runtime.
    target_link_libraries(${testname} clang_rt.tsan-x86_64)
    # Build tests with PIE and debug info.
    set_property(TARGET ${testname} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -fPIE -g")
    set_property(TARGET ${testname} APPEND_STRING
      PROPERTY LINK_FLAGS " -pie")
  endif()
endfunction()

add_subdirectory(rtl)
add_subdirectory(unit)
