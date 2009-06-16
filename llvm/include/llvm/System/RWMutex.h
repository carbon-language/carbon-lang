//===- RWMutex.h - Reader/Writer Mutual Exclusion Lock ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::RWMutex class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_RWMUTEX_H
#define LLVM_SYSTEM_RWMUTEX_H

namespace llvm
{
  namespace sys
  {
    /// @brief Platform agnostic Mutex class.
    class RWMutex
    {
    /// @name Constructors
    /// @{
    public:

      /// Initializes the lock but doesn't acquire it.
      /// @brief Default Constructor.
      explicit RWMutex();

      /// Releases and removes the lock
      /// @brief Destructor
      ~RWMutex();

    /// @}
    /// @name Methods
    /// @{
    public:

      /// Attempts to unconditionally acquire the lock in reader mode. If the
      /// lock is held by a writer, this method will wait until it can acquire
      /// the lock. 
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally acquire the lock in reader mode.
      bool reader_acquire();

      /// Attempts to release the lock in reader mode.
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally release the lock in reader mode.
      bool reader_release();

      /// Attempts to unconditionally acquire the lock in reader mode. If the
      /// lock is held by any readers, this method will wait until it can
      /// acquire the lock. 
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally acquire the lock in writer mode.
      bool writer_acquire();

      /// Attempts to release the lock in writer mode.
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally release the lock in write mode.
      bool writer_release();

    //@}
    /// @name Platform Dependent Data
    /// @{
    private:
#ifdef ENABLE_THREADS
      void* data_; ///< We don't know what the data will be
#endif

    /// @}
    /// @name Do Not Implement
    /// @{
    private:
      RWMutex(const RWMutex & original);
      void operator=(const RWMutex &);
    /// @}
    };
  }
}

#endif
