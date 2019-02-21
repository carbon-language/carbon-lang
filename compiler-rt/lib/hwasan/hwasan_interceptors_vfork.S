#define COMMON_INTERCEPTOR_SPILL_AREA __hwasan_extra_spill_area
#define COMMON_INTERCEPTOR_HANDLE_VFORK __hwasan_handle_vfork
#include "sanitizer_common/sanitizer_common_interceptors_vfork_aarch64.inc.S"

#if defined(__linux__)
.section        .note.GNU-stack,"",@progbits
#endif
