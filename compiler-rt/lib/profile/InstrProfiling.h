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


typedef struct __llvm_profile_data {
  const uint32_t NameSize;
  const uint32_t NumCounters;
  const uint64_t FuncHash;
  const char *const NamePtr;
  uint64_t *const CounterPtr;
} __llvm_profile_data;

typedef struct __llvm_profile_header {
  uint64_t Magic;
  uint64_t Version;
  uint64_t DataSize;
  uint64_t CountersSize;
  uint64_t NamesSize;
  uint64_t CountersDelta;
  uint64_t NamesDelta;
} __llvm_profile_header;


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

const __llvm_profile_data *__llvm_profile_begin_data(void);
const __llvm_profile_data *__llvm_profile_end_data(void);
const char *__llvm_profile_begin_names(void);
const char *__llvm_profile_end_names(void);
uint64_t *__llvm_profile_begin_counters(void);
uint64_t *__llvm_profile_end_counters(void);

/*!
 * \brief Write instrumentation data to the current file.
 *
 * Writes to the file with the last name given to \a __llvm_profile_set_filename(),
 * or if it hasn't been called, the \c LLVM_PROFILE_FILE environment variable,
 * or if that's not set, the last name given to
 * \a __llvm_profile_override_default_filename(), or if that's not set,
 * \c "default.profraw".
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

/*!
 * \brief Set the filename for writing instrumentation data, unless the
 * \c LLVM_PROFILE_FILE environment variable was set.
 *
 * Unless overridden, sets the filename to be used for subsequent calls to
 * \a __llvm_profile_write_file().
 *
 * \c Name is not copied, so it must remain valid.  Passing NULL resets the
 * filename logic to the default behaviour (unless the \c LLVM_PROFILE_FILE
 * was set in which case it has no effect).
 */
void __llvm_profile_override_default_filename(const char *Name);

/*! \brief Register to write instrumentation data to file at exit. */
int __llvm_profile_register_write_file_atexit(void);

/*! \brief Initialize file handling. */
void __llvm_profile_initialize_file(void);

/*! \brief Get the magic token for the file format. */
uint64_t __llvm_profile_get_magic(void);

/*! \brief Get the version of the file format. */
uint64_t __llvm_profile_get_version(void);

#endif /* PROFILE_INSTRPROFILING_H_ */
