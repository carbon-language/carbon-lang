set(TSAN_UNIT_TESTS
  tsan_clock_test.cc
  tsan_flags_test.cc
  tsan_mman_test.cc
  tsan_mutex_test.cc
  tsan_platform_test.cc
  tsan_printf_test.cc
  tsan_shadow_test.cc
  tsan_stack_test.cc
  tsan_suppressions_test.cc
  tsan_sync_test.cc
  tsan_vector_test.cc
  )

add_tsan_unittest(TsanUnitTest ${TSAN_UNIT_TESTS})
