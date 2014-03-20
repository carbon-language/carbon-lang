/*===- InstrProfilingExtras.h - Support library for PGO instrumentation ---===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

/*!
 * \brief Write instrumentation data to the current file.
 *
 * Writes to the file with the last name given to \a __llvm_profile_set_filename(),
 * or if it hasn't been called, the \c LLVM_PROFILE_FILE environment variable,
 * or if that's not set, \c "default.profdata".
 */
void __llvm_profile_write_file();

/*!
 * \brief Set the filename for writing instrumentation data.
 *
 * Sets the filename to be used for subsequent calls to
 * \a __llvm_profile_write_file().
 */
void __llvm_profile_set_filename(const char *Name);

/*! \brief Register to write instrumentation data to file at exit. */
void __llvm_profile_register_write_file_atexit();
