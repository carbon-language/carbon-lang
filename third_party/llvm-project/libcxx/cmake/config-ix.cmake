include(CMakePushCheckState)
include(CheckLibraryExists)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckCSourceCompiles)

if(WIN32 AND NOT MINGW)
  # NOTE(compnerd) this is technically a lie, there is msvcrt, but for now, lets
  # let the default linking take care of that.
  set(LIBCXX_HAS_C_LIB NO)
else()
  check_library_exists(c fopen "" LIBCXX_HAS_C_LIB)
endif()

if (NOT LIBCXX_USE_COMPILER_RT)
  if(WIN32 AND NOT MINGW)
    set(LIBCXX_HAS_GCC_S_LIB NO)
  else()
    if(ANDROID)
      check_library_exists(gcc __gcc_personality_v0 "" LIBCXX_HAS_GCC_LIB)
    else()
      check_library_exists(gcc_s __gcc_personality_v0 "" LIBCXX_HAS_GCC_S_LIB)
    endif()
  endif()
endif()

# libc++ is using -nostdlib++ at the link step when available,
# otherwise -nodefaultlibs is used. We want all our checks to also
# use one of these options, otherwise we may end up with an inconsistency between
# the flags we think we require during configuration (if the checks are
# performed without one of those options) and the flags that are actually
# required during compilation (which has the -nostdlib++ or -nodefaultlibs). libc is
# required for the link to go through. We remove sanitizers from the
# configuration checks to avoid spurious link errors.

check_c_compiler_flag(-nostdlib++ LIBCXX_SUPPORTS_NOSTDLIBXX_FLAG)
if (LIBCXX_SUPPORTS_NOSTDLIBXX_FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nostdlib++")
else()
  check_c_compiler_flag(-nodefaultlibs LIBCXX_SUPPORTS_NODEFAULTLIBS_FLAG)
  if (LIBCXX_SUPPORTS_NODEFAULTLIBS_FLAG)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nodefaultlibs")
  endif()
endif()

if (LIBCXX_SUPPORTS_NOSTDLIBXX_FLAG OR LIBCXX_SUPPORTS_NODEFAULTLIBS_FLAG)
  if (LIBCXX_HAS_C_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES c)
  endif ()
  if (LIBCXX_USE_COMPILER_RT)
    list(APPEND CMAKE_REQUIRED_FLAGS -rtlib=compiler-rt)
    find_compiler_rt_library(builtins LIBCXX_BUILTINS_LIBRARY)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "${LIBCXX_BUILTINS_LIBRARY}")
  elseif (LIBCXX_HAS_GCC_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES gcc)
  elseif (LIBCXX_HAS_GCC_S_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES gcc_s)
  endif ()
  if (MINGW)
    # Mingw64 requires quite a few "C" runtime libraries in order for basic
    # programs to link successfully with -nodefaultlibs.
    if (LIBCXX_USE_COMPILER_RT)
      set(MINGW_RUNTIME ${LIBCXX_BUILTINS_LIBRARY})
    else ()
      set(MINGW_RUNTIME gcc_s gcc)
    endif()
    set(MINGW_LIBRARIES mingw32 ${MINGW_RUNTIME} moldname mingwex msvcrt advapi32
                        shell32 user32 kernel32 mingw32 ${MINGW_RUNTIME}
                        moldname mingwex msvcrt)
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${MINGW_LIBRARIES})
  endif()
  if (CMAKE_C_FLAGS MATCHES -fsanitize OR CMAKE_CXX_FLAGS MATCHES -fsanitize)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-sanitize=all")
  endif ()
  if (CMAKE_C_FLAGS MATCHES -fsanitize-coverage OR CMAKE_CXX_FLAGS MATCHES -fsanitize-coverage)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters")
  endif ()
endif ()

# Check compiler pragmas
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  cmake_push_check_state()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=unknown-pragmas")
  check_c_source_compiles("
#pragma comment(lib, \"c\")
int main() { return 0; }
" LIBCXX_HAS_COMMENT_LIB_PRAGMA)
  cmake_pop_check_state()
endif()

# Check libraries
if(WIN32 AND NOT MINGW)
  # TODO(compnerd) do we want to support an emulation layer that allows for the
  # use of pthread-win32 or similar libraries to emulate pthreads on Windows?
  set(LIBCXX_HAS_PTHREAD_LIB NO)
  set(LIBCXX_HAS_M_LIB NO)
  set(LIBCXX_HAS_RT_LIB NO)
  set(LIBCXX_HAS_SYSTEM_LIB NO)
  set(LIBCXX_HAS_ATOMIC_LIB NO)
elseif(APPLE)
  check_library_exists(System write "" LIBCXX_HAS_SYSTEM_LIB)
  set(LIBCXX_HAS_PTHREAD_LIB NO)
  set(LIBCXX_HAS_M_LIB NO)
  set(LIBCXX_HAS_RT_LIB NO)
  set(LIBCXX_HAS_ATOMIC_LIB NO)
elseif(FUCHSIA)
  set(LIBCXX_HAS_M_LIB NO)
  set(LIBCXX_HAS_PTHREAD_LIB NO)
  set(LIBCXX_HAS_RT_LIB NO)
  set(LIBCXX_HAS_SYSTEM_LIB NO)
  check_library_exists(atomic __atomic_fetch_add_8 "" LIBCXX_HAS_ATOMIC_LIB)
else()
  check_library_exists(pthread pthread_create "" LIBCXX_HAS_PTHREAD_LIB)
  check_library_exists(m ccos "" LIBCXX_HAS_M_LIB)
  check_library_exists(rt clock_gettime "" LIBCXX_HAS_RT_LIB)
  set(LIBCXX_HAS_SYSTEM_LIB NO)
  check_library_exists(atomic __atomic_fetch_add_8 "" LIBCXX_HAS_ATOMIC_LIB)
endif()
