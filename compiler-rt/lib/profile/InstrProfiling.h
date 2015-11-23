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

#ifdef _MSC_VER
# define LLVM_ALIGNAS(x) __declspec(align(x))
#elif __GNUC__
#define LLVM_ALIGNAS(x) __attribute__((aligned(x)))
#endif

#define LLVM_LIBRARY_VISIBILITY __attribute__((visibility("hidden")))
#define LLVM_SECTION(Sect) __attribute__((section(Sect)))

#if defined(__FreeBSD__) && defined(__i386__)

/* System headers define 'size_t' incorrectly on x64 FreeBSD (prior to
 * FreeBSD 10, r232261) when compiled in 32-bit mode.
 */
#define PRIu64 "llu"
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef uint32_t uintptr_t;
#elif defined(__FreeBSD__) && defined(__x86_64__)
#define PRIu64 "lu"
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long int uintptr_t;

#else /* defined(__FreeBSD__) && defined(__i386__) */

#include <inttypes.h>
#include <stdint.h>

#endif /* defined(__FreeBSD__) && defined(__i386__) */

enum ValueKind {
  IPVK_IndirectCallTarget = 0,
  IPVK_First = IPVK_IndirectCallTarget,
  IPVK_Last = IPVK_IndirectCallTarget
};

typedef struct __llvm_profile_value_data {
  uint64_t TargetValue;
  uint64_t NumTaken;
} __llvm_profile_value_data;

typedef void *IntPtrT;
typedef struct LLVM_ALIGNAS(8) __llvm_profile_data {
  const uint32_t NameSize;
  const uint32_t NumCounters;
  const uint64_t FuncHash;
  const IntPtrT NamePtr;
  const IntPtrT CounterPtr;
  const IntPtrT FunctionPointer;
  IntPtrT Values;
  const uint16_t NumValueSites[IPVK_Last + 1];
} __llvm_profile_data;

typedef struct __llvm_profile_header {
  uint64_t Magic;
  uint64_t Version;
  uint64_t DataSize;
  uint64_t CountersSize;
  uint64_t NamesSize;
  uint64_t CountersDelta;
  uint64_t NamesDelta;
  uint64_t ValueKindLast;
  uint64_t ValueDataSize;
  uint64_t ValueDataDelta;
} __llvm_profile_header;

/*!
 * \brief Get number of bytes necessary to pad the argument to eight
 * byte boundary.
 */
uint8_t __llvm_profile_get_num_padding_bytes(uint64_t SizeInBytes);

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
 * \brief Clear profile counters to zero.
 *
 */
void __llvm_profile_reset_counters(void);

/*!
 * \brief Counts the number of times a target value is seen.
 *
 * Records the target value for the CounterIndex if not seen before. Otherwise,
 * increments the counter associated w/ the target value.
 */
void __llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
                                      uint32_t CounterIndex);

/*!
 * \brief Prepares the value profiling data for output.
 *
 * Prepares a single __llvm_profile_value_data array out of the many
 * ValueProfNode trees (one per instrumented function).
 */
uint64_t __llvm_profile_gather_value_data(uint8_t **DataArray);

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
