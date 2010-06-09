//===-- Mutex.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Mutex_h_
#define liblldb_Mutex_h_
#if defined(__cplusplus)

#include <pthread.h>
#include <assert.h>

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Mutex Mutex.h "lldb/Host/Mutex.h"
/// @brief A C++ wrapper class for pthread mutexes.
//----------------------------------------------------------------------
class Mutex
{
public:
    enum Type
    {
        eMutexTypeNormal,       ///< Mutex that can't recursively entered by the same thread
        eMutexTypeRecursive     ///< Mutex can be recursively entered by the same thread
    };

    //------------------------------------------------------------------
    /// @class Mutex::Locker
    ///
    /// A scoped locking class that allows a variety of pthread mutex
    /// objects to have a mutex locked when an Mutex::Locker
    /// object is created, and unlocked when it goes out of scope or
    /// when the Mutex::Locker::Reset(pthread_mutex_t *)
    /// is called. This provides an exception safe way to lock a mutex
    /// in a scope.
    //------------------------------------------------------------------
    class Locker
    {
    public:
        //--------------------------------------------------------------
        /// Default constructor.
        ///
        /// This will create a scoped mutex locking object that doesn't
        /// have a mutex to lock. One will need to be provided using the
        /// Mutex::Locker::Reset(pthread_mutex_t *) method.
        ///
        /// @see Mutex::Locker::Reset(pthread_mutex_t *)
        //--------------------------------------------------------------
        Locker();

        //--------------------------------------------------------------
        /// Constructor with a Mutex object.
        ///
        /// This will create a scoped mutex locking object that extracts
        /// the mutex owned by \a m and locks it.
        ///
        /// @param[in] m
        ///     An instance of a Mutex object that contains a
        ///     valid mutex object.
        //--------------------------------------------------------------
        Locker(Mutex& m);

        //--------------------------------------------------------------
        /// Constructor with a Mutex object pointer.
        ///
        /// This will create a scoped mutex locking object that extracts
        /// the mutex owned by a m and locks it.
        ///
        /// @param[in] m
        ///     A pointer to instance of a Mutex object that
        ///     contains a valid mutex object.
        //--------------------------------------------------------------
        Locker(Mutex* m);

        //--------------------------------------------------------------
        /// Constructor with a raw pthread mutex object pointer.
        ///
        /// This will create a scoped mutex locking object that locks
        /// \a mutex.
        ///
        /// @param[in] mutex
        ///     A pointer to a pthread_mutex_t that will get locked if
        ///     non-NULL.
        //--------------------------------------------------------------
        Locker(pthread_mutex_t *mutex);

        //--------------------------------------------------------------
        /// Desstructor
        ///
        /// Unlocks any valid pthread_mutex_t that this object may
        /// contain.
        //--------------------------------------------------------------
        ~Locker();

        //--------------------------------------------------------------
        /// Change the contained mutex.
        ///
        /// Unlock the current mutex in this object (if it contains a
        /// valid mutex) and lock the new \a mutex object if it is
        /// non-NULL.
        //--------------------------------------------------------------
        void
        Reset(pthread_mutex_t *mutex = NULL);

        //--------------------------------------------------------------
        /// Change the contained mutex only if the mutex can be locked.
        ///
        /// Unlock the current mutex in this object (if it contains a
        /// valid mutex) and try to lock \a mutex. If \a mutex can be 
        /// locked this object will take ownership of the lock and will
        /// unlock it when it goes out of scope or Reset or TryLock are
        /// called again. If the mutex is already locked, this object
        /// will not take ownership of the mutex.
        ///
        /// @return
        ///     Returns \b true if the lock was aquired and the this 
        ///     object will unlock the mutex when it goes out of scope,
        ///     returns \b false otherwise.
        //--------------------------------------------------------------
        bool
        TryLock (pthread_mutex_t *mutex);

    protected:
        //--------------------------------------------------------------
        /// Member variables
        //--------------------------------------------------------------
        pthread_mutex_t *m_mutex_ptr;   ///< A pthread mutex that is locked when
                                        ///< acquired and unlocked when destroyed
                                        ///< or reset.

    private:
        Locker(const Locker&);
        const Locker& operator=(const Locker&);
    };


    //------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Creates a pthread mutex with no attributes.
    //------------------------------------------------------------------
    Mutex();

    //------------------------------------------------------------------
    /// Default constructor.
    ///
    /// Creates a pthread mutex with \a type as the mutex type.
    /// Valid values for \a type include:
    ///     @li Mutex::Type::eMutexTypeNormal
    ///     @li Mutex::Type::eMutexTypeRecursive
    ///
    /// @param[in] type
    ///     The type of the mutex.
    ///
    /// @see ::pthread_mutexattr_settype()
    //------------------------------------------------------------------
    Mutex(Mutex::Type type);

    //------------------------------------------------------------------
    /// Destructor.
    ///
    /// Destroys the mutex owned by this object.
    //------------------------------------------------------------------
    ~Mutex();

    //------------------------------------------------------------------
    /// Mutex get accessor.
    ///
    /// @return
    ///     A pointer to the pthread mutex object owned by this object.
    //------------------------------------------------------------------
    pthread_mutex_t *
    GetMutex();

    //------------------------------------------------------------------
    /// Lock the mutex.
    ///
    /// Locks the mutex owned by this object. If the mutex is already
    /// locked, the calling thread will block until the mutex becomes
    /// available.
    ///
    /// @return
    ///     The error code from \c pthread_mutex_lock().
    //------------------------------------------------------------------
    int
    Lock();

    //------------------------------------------------------------------
    /// Try to lock the mutex.
    ///
    /// Attempts to lock the mutex owned by this object without blocking.
    /// If the mutex is already locked, TryLock() will not block waiting
    /// for the mutex, but will return an error condition.
    ///
    /// @return
    ///     The error code from \c pthread_mutex_trylock().
    //------------------------------------------------------------------
    int
    TryLock();

    //------------------------------------------------------------------
    /// Unlock the mutex.
    ///
    /// If the current thread holds the lock on the owned mutex, then
    /// Unlock() will unlock the mutex. Calling Unlock() on this object
    /// when the calling thread does not hold the lock will result in
    /// undefined behavior.
    ///
    /// @return
    ///     The error code from \c pthread_mutex_unlock().
    //------------------------------------------------------------------
    int
    Unlock();

    static
    int Lock (pthread_mutex_t *mutex_ptr);

    static
    int TryLock (pthread_mutex_t *mutex_ptr);

    static
    int Unlock (pthread_mutex_t *mutex_ptr);

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    pthread_mutex_t m_mutex; ///< The pthread mutex object.
private:
    Mutex(const Mutex&);
    const Mutex& operator=(const Mutex&);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif
