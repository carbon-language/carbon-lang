//===- llvm/System/Mutex.h - Mutex Operating System Concept -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Mutex class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_MUTEX_H
#define LLVM_SYSTEM_MUTEX_H

#include "llvm/System/Threading.h"

namespace llvm
{
  namespace sys
  {
    /// @brief Platform agnostic Mutex class.
    class MutexImpl
    {
    /// @name Constructors
    /// @{
    public:

      /// Initializes the lock but doesn't acquire it. if \p recursive is set
      /// to false, the lock will not be recursive which makes it cheaper but
      /// also more likely to deadlock (same thread can't acquire more than
      /// once).
      /// @brief Default Constructor.
      explicit MutexImpl(bool recursive = true);

      /// Releases and removes the lock
      /// @brief Destructor
      ~MutexImpl();

    /// @}
    /// @name Methods
    /// @{
    public:

      /// Attempts to unconditionally acquire the lock. If the lock is held by
      /// another thread, this method will wait until it can acquire the lock.
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally acquire the lock.
      bool acquire();

      /// Attempts to release the lock. If the lock is held by the current
      /// thread, the lock is released allowing other threads to acquire the
      /// lock.
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally release the lock.
      bool release();

      /// Attempts to acquire the lock without blocking. If the lock is not
      /// available, this function returns false quickly (without blocking). If
      /// the lock is available, it is acquired.
      /// @returns false if any kind of error occurs or the lock is not
      /// available, true otherwise.
      /// @brief Try to acquire the lock.
      bool tryacquire();

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
      MutexImpl(const MutexImpl & original);
      void operator=(const MutexImpl &);
    /// @}
    };
    
    
    /// SmartMutex - A mutex with a compile time constant parameter that 
    /// indicates whether this mutex should become a no-op when we're not
    /// running in multithreaded mode.
    template<bool mt_only>
    class SmartMutex {
      MutexImpl mtx;
    public:
      explicit SmartMutex(bool recursive = true) : mtx(recursive) { }
      
      bool acquire() {
        if (!mt_only || (mt_only && llvm_is_multithreaded()))
          return mtx.acquire();
        return true;
      }

      bool release() {
        if (!mt_only || (mt_only && llvm_is_multithreaded()))
          return mtx.release();
        return true;
      }

      bool tryacquire() {
        if (!mt_only || (mt_only && llvm_is_multithreaded()))
          return mtx.tryacquire();
        return true;
      }
      
      private:
        SmartMutex<mt_only>(const SmartMutex<mt_only> & original);
        void operator=(const SmartMutex<mt_only> &);
    };
    
    /// Mutex - A standard, always enforced mutex.
    typedef SmartMutex<false> Mutex;
  }
}

#endif
