/*===- InstrProfiling.h- Support library for PGO instrumentation ----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILING_H_
#define PROFILE_INSTRPROFILING_H_

#if defined(__FreeBSD__) && defined(__i386__)

/* System headers define 'size_t' incorrectly on x64 FreeBSD (prior to
 * FreeBSD 10, r232261) when compiled in 32-bit mode.
 */
#define PRIu64 "llu"
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef uint32_t uintptr_t;

#else /* defined(__FreeBSD__) && defined(__i386__) */

#include <inttypes.h>
#include <stdint.h>

#endif /* defined(__FreeBSD__) && defined(__i386__) */

#define PROFILE_HEADER_SIZE 7

typedef struct __llvm_profile_data {
  const uint32_t NameSize;
  const uint32_t NumCounters;
  const uint64_t FuncHash;
  const char *const Name;
  uint64_t *const Counters;
} __llvm_profile_data;

/*!
 * \brief Get required size for profile buffer.
 */
uint64_t __llvm_profile_get_size_for_buffer(void);

/*!
 * \brief Write instrumentation data to the given buffer.
 *
 * \pre \c Buffer is the start of a buffer at least as big as \a
 * __llvm_profile_get_size_for_buffer().
 */
int __llvm_profile_write_buffer(char *Buffer);

const __llvm_profile_data *__llvm_profile_data_begin(void);
const __llvm_profile_data *__llvm_profile_data_end(void);
const char *__llvm_profile_names_begin(void);
const char *__llvm_profile_names_end(void);
uint64_t *__llvm_profile_counters_begin(void);
uint64_t *__llvm_profile_counters_end(void);

#define PROFILE_RANGE_SIZE(Range) \
  (__llvm_profile_ ## Range ## _end() - __llvm_profile_ ## Range ## _begin())

/*!
 * \brief Write instrumentation data to the current file.
 *
 * Writes to the file with the last name given to \a __llvm_profile_set_filename(),
 * or if it hasn't been called, the \c LLVM_PROFILE_FILE environment variable,
 * or if that's not set, \c "default.profdata".
 */
int __llvm_profile_write_file(void);

/*!
 * \brief Set the filename for writing instrumentation data.
 *
 * Sets the filename to be used for subsequent calls to
 * \a __llvm_profile_write_file().
 *
 * \c Name is not copied, so it must remain valid.  Passing NULL resets the
 * filename logic to the default behaviour.
 */
void __llvm_profile_set_filename(const char *Name);

/*! \brief Register to write instrumentation data to file at exit. */
int __llvm_profile_register_write_file_atexit(void);

/*! \brief Initialize file handling. */
void __llvm_profile_initialize_file(void);

/*! \brief Get the magic token for the file format. */
uint64_t __llvm_profile_get_magic(void);

/*! \brief Get the version of the file format. */
uint64_t __llvm_profile_get_version(void);

#endif /* PROFILE_INSTRPROFILING_H_ */
