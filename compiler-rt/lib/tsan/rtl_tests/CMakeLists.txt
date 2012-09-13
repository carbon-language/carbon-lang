set(TSAN_RTL_TESTS
  tsan_bench.cc
  tsan_mop.cc
  tsan_mutex.cc
  tsan_posix.cc
  tsan_string.cc
  tsan_test.cc
  tsan_thread.cc
  )

if(UNIX AND NOT APPLE)
  list(APPEND TSAN_RTL_TESTS tsan_test_util_linux.cc)
endif()

add_tsan_unittest(TsanRtlTest ${TSAN_RTL_TESTS})
