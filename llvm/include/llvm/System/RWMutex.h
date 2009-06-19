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

#include "llvm/System/Threading.h"

namespace llvm
{
  namespace sys
  {
    /// @brief Platform agnostic RWMutex class.
    class RWMutexImpl
    {
    /// @name Constructors
    /// @{
    public:

      /// Initializes the lock but doesn't acquire it.
      /// @brief Default Constructor.
      explicit RWMutexImpl();

      /// Releases and removes the lock
      /// @brief Destructor
      ~RWMutexImpl();

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
      void* data_; ///< We don't know what the data will be

    /// @}
    /// @name Do Not Implement
    /// @{
    private:
      RWMutexImpl(const RWMutexImpl & original);
      void operator=(const RWMutexImpl &);
    /// @}
    };
    
    /// SmartMutex - An R/W mutex with a compile time constant parameter that 
    /// indicates whether this mutex should become a no-op when we're not
    /// running in multithreaded mode.
    template<bool mt_only>
    class SmartRWMutex : public RWMutexImpl {
    public:
      explicit SmartRWMutex() : RWMutexImpl() { }
      
      bool reader_acquire() {
        if (!mt_only || llvm_is_multithreaded())
          return RWMutexImpl::reader_acquire();
        return true;
      }
      
      bool reader_release() {
        if (!mt_only || llvm_is_multithreaded())
          return RWMutexImpl::reader_release();
        return true;
      }
      
      bool writer_acquire() {
        if (!mt_only || llvm_is_multithreaded())
          return RWMutexImpl::writer_acquire();
        return true;
      }
      
      bool writer_release() {
        if (!mt_only || llvm_is_multithreaded())
          return RWMutexImpl::writer_release();
        return true;
      }
      
    private:
      SmartRWMutex(const SmartRWMutex<mt_only> & original);
      void operator=(const SmartRWMutex<mt_only> &);
    };
    typedef SmartRWMutex<false> RWMutex;
    
    /// ScopedReader - RAII acquisition of a reader lock
    template<bool mt_only>
    struct SmartScopedReader {
      SmartRWMutex<mt_only>* mutex;
      
      explicit SmartScopedReader(SmartRWMutex<mt_only>* m) {
        mutex = m;
        mutex->reader_acquire();
      }
      
      ~SmartScopedReader() {
        mutex->reader_release();
      }
    };
    typedef SmartScopedReader<false> ScopedReader;
    
    /// ScopedWriter - RAII acquisition of a writer lock
    template<bool mt_only>
    struct SmartScopedWriter {
      SmartRWMutex<mt_only>* mutex;
      
      explicit SmartScopedWriter(SmartRWMutex<mt_only>* m) {
        mutex = m;
        mutex->writer_acquire();
      }
      
      ~SmartScopedWriter() {
        mutex->writer_release();
      }
    };
    typedef SmartScopedWriter<false> ScopedWriter;
  }
}

#endif
