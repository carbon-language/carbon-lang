#if USE_ITT_BUILD
/*
 * kmp_itt.inl -- Inline functions of ITT Notify.
 * $Revision: 42866 $
 * $Date: 2013-12-10 15:15:58 -0600 (Tue, 10 Dec 2013) $
 */


//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


// Inline function definitions. This file should be included into kmp_itt.h file for prodiction
// build (to let compliler inline functions) or into kmp_itt.c file for debug build (to reduce
// the number of files to recompile and save build time).


#include "kmp.h"
#include "kmp_str.h"

#if KMP_ITT_DEBUG
    extern kmp_bootstrap_lock_t __kmp_itt_debug_lock;
    #define KMP_ITT_DEBUG_LOCK() {                                   \
        __kmp_acquire_bootstrap_lock( & __kmp_itt_debug_lock );      \
    }
    #define KMP_ITT_DEBUG_PRINT( ... ) {                             \
        fprintf( stderr, "#%02d: ", __kmp_get_gtid() );              \
        fprintf( stderr, __VA_ARGS__ );                              \
        fflush( stderr );                                            \
        __kmp_release_bootstrap_lock( & __kmp_itt_debug_lock );      \
    }
#else
    #define KMP_ITT_DEBUG_LOCK()
    #define KMP_ITT_DEBUG_PRINT( ... )
#endif // KMP_ITT_DEBUG

// Ensure that the functions are static if they're supposed to be
// being inlined. Otherwise they cannot be used in more than one file,
// since there will be multiple definitions.
#if KMP_DEBUG
# define LINKAGE
#else
# define LINKAGE static inline
#endif

// ZCA interface used by Intel(R) Inspector. Intel(R) Parallel Amplifier uses this
// API to support user-defined synchronization primitives, but does not use ZCA;
// it would be safe to turn this off until wider support becomes available.
#if USE_ITT_ZCA
#ifdef __INTEL_COMPILER
#   if __INTEL_COMPILER >= 1200
#       undef __itt_sync_acquired
#       undef __itt_sync_releasing
#       define __itt_sync_acquired(addr)    __notify_zc_intrinsic((char *)"sync_acquired", addr)
#       define __itt_sync_releasing(addr)   __notify_intrinsic((char *)"sync_releasing", addr)
#   endif
#endif
#endif

/*
    ------------------------------------------------------------------------------------------------
    Parallel region reporting.

        * __kmp_itt_region_forking should be called by master thread of a team. Exact moment of
          call does not matter, but it should be completed before any thread of this team calls
          __kmp_itt_region_starting.
        * __kmp_itt_region_starting should be called by each thread of a team just before entering
          parallel region body.
        * __kmp_itt_region_finished should be called by each thread of a team right after returning
          from parallel region body.
        * __kmp_itt_region_joined should be called by master thread of a team, after all threads
          called __kmp_itt_region_finished.

    Note: Thread waiting at join barrier (after __kmp_itt_region_finished) can execute some more
    user code -- such a thread can execute tasks.

    Note: The overhead of logging region_starting and region_finished in each thread is too large,
    so these calls are not used.

    ------------------------------------------------------------------------------------------------
*/

// -------------------------------------------------------------------------------------------------

LINKAGE void
__kmp_itt_region_forking( int gtid, int serialized ) {
#if USE_ITT_NOTIFY
    kmp_team_t *      team = __kmp_team_from_gtid( gtid );
#if OMP_30_ENABLED
    if (team->t.t_active_level + serialized > 1)
#endif
    {
        // The frame notifications are only supported for the outermost teams.
        return;
    }
    ident_t *         loc  = __kmp_thread_from_gtid( gtid )->th.th_ident;
    if (loc) {
        // Use the reserved_2 field to store the index to the region domain.
        // Assume that reserved_2 contains zero initially.  Since zero is special
        // value here, store the index into domain array increased by 1.
        if (loc->reserved_2 == 0) {
            if (__kmp_frame_domain_count < KMP_MAX_FRAME_DOMAINS) {
                int frm = KMP_TEST_THEN_INC32( & __kmp_frame_domain_count ); // get "old" value
                if (frm >= KMP_MAX_FRAME_DOMAINS) {
                    KMP_TEST_THEN_DEC32( & __kmp_frame_domain_count );       // revert the count
                    return;                      // loc->reserved_2 is still 0
                }
                //if (!KMP_COMPARE_AND_STORE_ACQ32( &loc->reserved_2, 0, frm + 1 )) {
                //    frm = loc->reserved_2 - 1;   // get value saved by other thread for same loc
                //} // AC: this block is to replace next unsynchronized line
                loc->reserved_2 = frm + 1;                                   // save "new" value

                // Transform compiler-generated region location into the format
                // that the tools more or less standardized on:
                //                               "<func>$omp$parallel@[file:]<line>[:<col>]"
                const char * buff = NULL;
                kmp_str_loc_t str_loc = __kmp_str_loc_init( loc->psource, 1 );
                buff = __kmp_str_format("%s$omp$parallel@%s:%d:%d",
                                        str_loc.func, str_loc.file,
                                        str_loc.line, str_loc.col);
                __kmp_str_loc_free( &str_loc );

                __itt_suppress_push(__itt_suppress_memory_errors);
                __kmp_itt_domains[ frm ] = __itt_domain_create( buff );
                __itt_suppress_pop();

                __kmp_str_free( &buff );
                __itt_frame_begin_v3(__kmp_itt_domains[ frm ], NULL);
            }
        } else { // if it is not 0 then it should be <= KMP_MAX_FRAME_DOMAINS
            __itt_frame_begin_v3(__kmp_itt_domains[loc->reserved_2 - 1], NULL);
        }
        KMP_ITT_DEBUG_LOCK();
        KMP_ITT_DEBUG_PRINT( "[frm beg] gtid=%d, idx=%d, serialized:%d, loc:%p\n",
                         gtid, loc->reserved_2 - 1, serialized, loc );
    }
#endif
} // __kmp_itt_region_forking

// -------------------------------------------------------------------------------------------------

LINKAGE void
__kmp_itt_frame_submit( int gtid, __itt_timestamp begin, __itt_timestamp end, int imbalance, ident_t * loc ) {
#if USE_ITT_NOTIFY
        if (loc) {
            if (loc->reserved_2 == 0) {
                if (__kmp_frame_domain_count < KMP_MAX_FRAME_DOMAINS) {
                    int frm = KMP_TEST_THEN_INC32( & __kmp_frame_domain_count ); // get "old" value
                    if (frm >= KMP_MAX_FRAME_DOMAINS) {
                        KMP_TEST_THEN_DEC32( & __kmp_frame_domain_count );       // revert the count
                        return;                      // loc->reserved_2 is still 0
                    }
                    // Should it be synchronized? See the comment in __kmp_itt_region_forking
                    loc->reserved_2 = frm + 1;                                   // save "new" value

                    // Transform compiler-generated region location into the format
                    // that the tools more or less standardized on:
                    //                               "<func>$omp$frame@[file:]<line>[:<col>]"
                    const char * buff = NULL;
                    kmp_str_loc_t str_loc = __kmp_str_loc_init( loc->psource, 1 );
                    if( imbalance ) {
                        buff = __kmp_str_format("%s$omp$barrier-imbalance@%s:%d",
                                                str_loc.func, str_loc.file, str_loc.col);
                    } else {
                        buff = __kmp_str_format("%s$omp$barrier@%s:%d",
                                                str_loc.func, str_loc.file, str_loc.col);
                    }
                    __kmp_str_loc_free( &str_loc );

                    __itt_suppress_push(__itt_suppress_memory_errors);
                    __kmp_itt_domains[ frm ] = __itt_domain_create( buff );
                    __itt_suppress_pop();

                    __kmp_str_free( &buff );
                    __itt_frame_submit_v3(__kmp_itt_domains[ frm ], NULL, begin, end );
                }
            } else { // if it is not 0 then it should be <= KMP_MAX_FRAME_DOMAINS
                __itt_frame_submit_v3(__kmp_itt_domains[loc->reserved_2 - 1], NULL, begin, end );
            }
    }

#endif
} // __kmp_itt_frame_submit

// -------------------------------------------------------------------------------------------------

LINKAGE void
__kmp_itt_region_starting( int gtid ) {
#if USE_ITT_NOTIFY
#endif
} // __kmp_itt_region_starting

// -------------------------------------------------------------------------------------------------

LINKAGE void
__kmp_itt_region_finished( int gtid ) {
#if USE_ITT_NOTIFY
#endif
} // __kmp_itt_region_finished

// -------------------------------------------------------------------------------------------------

LINKAGE void
__kmp_itt_region_joined( int gtid, int serialized ) {
#if USE_ITT_NOTIFY
    kmp_team_t *      team = __kmp_team_from_gtid( gtid );
#if OMP_30_ENABLED
    if (team->t.t_active_level + serialized > 1)
#endif
    {
        // The frame notifications are only supported for the outermost teams.
        return;
    }
    ident_t *         loc  = __kmp_thread_from_gtid( gtid )->th.th_ident;
    if (loc && loc->reserved_2 && loc->reserved_2 <= KMP_MAX_FRAME_DOMAINS) {
        KMP_ITT_DEBUG_LOCK();
        __itt_frame_end_v3(__kmp_itt_domains[loc->reserved_2 - 1], NULL);
        KMP_ITT_DEBUG_PRINT( "[frm end] gtid=%d, idx=%d, serialized:%d, loc:%p\n",
                         gtid, loc->reserved_2 - 1, serialized, loc );
    }
#endif
} // __kmp_itt_region_joined

/*
    ------------------------------------------------------------------------------------------------
    Barriers reporting.

    A barrier consists of two phases:

        1. Gather -- master waits for arriving of all the worker threads; each worker thread
           registers arrival and goes further.
        2. Release -- each worker threads waits until master lets it go; master lets worker threads
           go.

    Function should be called by each thread:

        * __kmp_itt_barrier_starting() -- before arriving to the gather phase.
        * __kmp_itt_barrier_middle()   -- between gather and release phases.
        * __kmp_itt_barrier_finished() -- after release phase.

    Note: Call __kmp_itt_barrier_object() before call to __kmp_itt_barrier_starting() and save
    result in local variable. __kmp_itt_barrier_object(), being called too late (e. g. after gather
    phase) would return itt sync object for the next barrier!

    ITT need an address (void *) to be specified as a sync object. OpenMP RTL does not have
    barrier object or barrier data structure. Barrier is just a counter in team and thread
    structures. We could use an address of team structure as an barrier sync object, but ITT wants
    different objects for different barriers (even whithin the same team). So let us use
    team address as barrier sync object for the first barrier, then increase it by one for the next
    barrier, and so on (but wrap it not to use addresses outside of team structure).

    ------------------------------------------------------------------------------------------------
*/

void *
__kmp_itt_barrier_object(
    int  gtid,
    int  bt,
    int  set_name,
    int  delta    // 0 (current barrier) is default value; specify -1 to get previous barrier.
) {
    void * object = NULL;
#if USE_ITT_NOTIFY
    kmp_info_t *    thr  = __kmp_thread_from_gtid( gtid );
    kmp_team_t *    team = thr->th.th_team;

    // NOTE:
    // If the function is called from __kmp_fork_barrier, team pointer can be NULL. This "if"
    // helps to avoid crash. However, this is not complete solution, and reporting fork/join
    // barriers to ITT should be revisited.

    if ( team != NULL ) {

        // Master thread increases b_arrived by KMP_BARRIER_STATE_BUMP each time. Divide b_arrived
        // by KMP_BARRIER_STATE_BUMP to get plain barrier counter.
        kmp_uint counter = team->t.t_bar[ bt ].b_arrived / KMP_BARRIER_STATE_BUMP + delta;
        // Now form the barrier id. Encode barrier type (bt) in barrier id too, so barriers of
        // different types do not have the same ids.
        KMP_BUILD_ASSERT( sizeof( kmp_team_t ) >= bs_last_barrier );
            // This conditon is a must (we would have zero divide otherwise).
        KMP_BUILD_ASSERT( sizeof( kmp_team_t ) >= 2 * bs_last_barrier );
            // More strong condition: make sure we have room at least for for two differtent ids
            // (for each barrier type).
        object =
            reinterpret_cast< void * >(
                kmp_uintptr_t( team )
                    + counter % ( sizeof( kmp_team_t ) / bs_last_barrier ) * bs_last_barrier
                    + bt
            );
        KMP_ITT_DEBUG_LOCK();
        KMP_ITT_DEBUG_PRINT( "[bar obj] type=%d, counter=%d, object=%p\n", bt, counter, object );

        if ( set_name ) {
            ident_t const * loc  = NULL;
            char const *    src  = NULL;
            char const *    type = "OMP Barrier";
            switch ( bt ) {
                case bs_plain_barrier : {
                    // For plain barrier compiler calls __kmpc_barrier() function, which saves
                    // location in thr->th.th_ident.
                    loc = thr->th.th_ident;
                    // Get the barrier type from flags provided by compiler.
                    kmp_int32   expl = 0;
                    kmp_uint32  impl = 0;
                    if ( loc != NULL ) {
                        src  = loc->psource;
                        expl = ( loc->flags & KMP_IDENT_BARRIER_EXPL ) != 0;
                        impl = ( loc->flags & KMP_IDENT_BARRIER_IMPL ) != 0;
                    }; // if
                    if ( impl ) {
                        switch ( loc->flags & KMP_IDENT_BARRIER_IMPL_MASK ) {
                            case KMP_IDENT_BARRIER_IMPL_FOR : {
                                type = "OMP For Barrier";
                            } break;
                            case KMP_IDENT_BARRIER_IMPL_SECTIONS : {
                                type = "OMP Sections Barrier";
                            } break;
                            case KMP_IDENT_BARRIER_IMPL_SINGLE : {
                                type = "OMP Single Barrier";
                            } break;
                            case KMP_IDENT_BARRIER_IMPL_WORKSHARE : {
                                type = "OMP Workshare Barrier";
                            } break;
                            default : {
                                type = "OMP Implicit Barrier";
                                KMP_DEBUG_ASSERT( 0 );
                            };
                        }; /* switch */
                    } else if ( expl ) {
                        type = "OMP Explicit Barrier";
                    }; /* if */
                } break;
                case bs_forkjoin_barrier : {
                    // In case of fork/join barrier we can read thr->th.th_ident, because it
                    // contains location of last passed construct (while join barrier is not
                    // such one). Use th_ident of master thread instead -- __kmp_join_call()
                    // called by the master thread saves location.
                    //
                    // AC: cannot read from master because __kmp_join_call may be not called
                    //    yet, so we read the location from team. This is the same location.
                    //    And team is valid at the enter to join barrier where this happens.
                    loc  = team->t.t_ident;
                    if ( loc != NULL ) {
                        src  = loc->psource;
                    }; // if
                    type = "OMP Join Barrier";
                } break;
            }; // switch
            KMP_ITT_DEBUG_LOCK();
            __itt_sync_create( object, type, src, __itt_attr_barrier );
            KMP_ITT_DEBUG_PRINT( "[bar sta] scre( %p, \"%s\", \"%s\", __itt_attr_barrier )\n", object, type, src );
        }; // if

    }; // if
#endif
    return object;
} // __kmp_itt_barrier_object

// -------------------------------------------------------------------------------------------------

void
__kmp_itt_barrier_starting( int gtid, void * object ) {
#if USE_ITT_NOTIFY
    if ( !KMP_MASTER_GTID( gtid ) ) {
        KMP_ITT_DEBUG_LOCK();
        __itt_sync_releasing( object );
        KMP_ITT_DEBUG_PRINT( "[bar sta] srel( %p )\n", object );
    }; // if
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_prepare( object );
    KMP_ITT_DEBUG_PRINT( "[bar sta] spre( %p )\n", object );
#endif
} // __kmp_itt_barrier_starting

// -------------------------------------------------------------------------------------------------

void
__kmp_itt_barrier_middle( int gtid, void * object ) {
#if USE_ITT_NOTIFY
    if ( KMP_MASTER_GTID( gtid ) ) {
        KMP_ITT_DEBUG_LOCK();
        __itt_sync_acquired( object );
        KMP_ITT_DEBUG_PRINT( "[bar mid] sacq( %p )\n", object );
        KMP_ITT_DEBUG_LOCK();
        __itt_sync_releasing( object );
        KMP_ITT_DEBUG_PRINT( "[bar mid] srel( %p )\n", object );
    } else {
    }; // if
#endif
} // __kmp_itt_barrier_middle

// -------------------------------------------------------------------------------------------------

void
__kmp_itt_barrier_finished( int gtid, void * object ) {
#if USE_ITT_NOTIFY
    if ( KMP_MASTER_GTID( gtid ) ) {
    } else {
        KMP_ITT_DEBUG_LOCK();
        __itt_sync_acquired( object );
        KMP_ITT_DEBUG_PRINT( "[bar end] sacq( %p )\n", object );
    }; // if
#endif
} // __kmp_itt_barrier_finished

#if OMP_30_ENABLED

/*
    ------------------------------------------------------------------------------------------------
    Taskwait reporting.

    ITT need an address (void *) to be specified as a sync object. OpenMP RTL does not have taskwait
    structure, so we need to construct something.

*/

void *
__kmp_itt_taskwait_object( int gtid ) {
    void * object = NULL;
#if USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr ) {
        kmp_info_t *     thread   = __kmp_thread_from_gtid( gtid );
        kmp_taskdata_t * taskdata = thread -> th.th_current_task;
        object =
            reinterpret_cast< void * >(
                kmp_uintptr_t( taskdata ) + taskdata->td_taskwait_counter % sizeof( kmp_taskdata_t )
            );
    }; // if
#endif
    return object;
} // __kmp_itt_taskwait_object

void
__kmp_itt_taskwait_starting(
    int     gtid,
    void *  object
) {
#if USE_ITT_NOTIFY
    kmp_info_t *     thread   = __kmp_thread_from_gtid( gtid );
    kmp_taskdata_t * taskdata = thread -> th.th_current_task;
    ident_t const *  loc      = taskdata->td_taskwait_ident;
    char const *     src      = ( loc == NULL? NULL : loc->psource );
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_create( object, "OMP Taskwait", src, 0 );
    KMP_ITT_DEBUG_PRINT( "[twa sta] scre( %p, \"OMP Taskwait\", \"%s\", 0 )\n", object, src );
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_prepare( object );
    KMP_ITT_DEBUG_PRINT( "[twa sta] spre( %p )\n", object );
#endif
} // __kmp_itt_taskwait_starting

void
__kmp_itt_taskwait_finished(
    int     gtid,
    void *  object
) {
#if USE_ITT_NOTIFY
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_acquired( object );
    KMP_ITT_DEBUG_PRINT( "[twa end] sacq( %p )\n", object );
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_destroy( object );
    KMP_ITT_DEBUG_PRINT( "[twa end] sdes( %p )\n", object );
#endif
} // __kmp_itt_taskwait_finished

/*
    ------------------------------------------------------------------------------------------------
    Task reporting.

    Only those tasks are reported which are executed by a thread spinning at barrier (or taskwait).
    Synch object passed to the function must be barrier of taskwait the threads waiting at.
    ------------------------------------------------------------------------------------------------
*/

void
__kmp_itt_task_starting(
    void * object     // ITT sync object: barrier or taskwait.
) {
#if USE_ITT_NOTIFY
    if ( object != NULL ) {
        KMP_ITT_DEBUG_LOCK();
        __itt_sync_cancel( object );
        KMP_ITT_DEBUG_PRINT( "[tsk sta] scan( %p )\n", object );
    }; // if
#endif
} // __kmp_itt_task_starting

// -------------------------------------------------------------------------------------------------

void
__kmp_itt_task_finished(
    void * object     // ITT sync object: barrier or taskwait.
) {
#if USE_ITT_NOTIFY
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_prepare( object );
    KMP_ITT_DEBUG_PRINT( "[tsk end] spre( %p )\n", object );
#endif
} // __kmp_itt_task_finished

// -------------------------------------------------------------------------------------------------

#endif /* OMP_30_ENABLED */

/*
    ------------------------------------------------------------------------------------------------
    Lock reporting.

        * __kmp_itt_lock_creating( lock ) should be called *before* the first lock operation
          (set/unset). It is not a real event shown to the user but just setting a name for
          syncronization object. `lock' is an address of sync object, the same address should be
          used in all subsequent calls.

        * __kmp_itt_lock_acquiring() should be called before setting the lock.

        * __kmp_itt_lock_acquired() should be called after setting the lock.

        * __kmp_itt_lock_realeasing() should be called before unsetting the lock.

        * __kmp_itt_lock_cancelled() should be called after thread cancelled waiting for the lock.

        * __kmp_itt_lock_destroyed( lock ) should be called after the last lock operation. After
          __kmp_itt_lock_destroyed() all the references to the same address will be considered
          as another sync object, not related with the original one.
    ------------------------------------------------------------------------------------------------
*/

// -------------------------------------------------------------------------------------------------

// Internal guts -- common code for locks and critical sections, do not call directly.
__kmp_inline
void
___kmp_itt_lock_init( kmp_user_lock_p lock, char const * type ) {
#if USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr ) {
        ident_t const * loc = NULL;
        if ( __kmp_get_user_lock_location_ != NULL )
            loc = __kmp_get_user_lock_location_( (lock) );
        char const *    src = ( loc == NULL ? NULL : loc->psource );
        KMP_ITT_DEBUG_LOCK();
        __itt_sync_create( lock, type, src, 0 );
        KMP_ITT_DEBUG_PRINT( "[lck ini] scre( %p, \"%s\", \"%s\", 0 )\n", lock, type, src );
    }; // if
#endif
} // ___kmp_itt_lock_init

// Internal guts -- common code for locks and critical sections, do not call directly.
__kmp_inline
void
___kmp_itt_lock_fini( kmp_user_lock_p lock, char const * type ) {
#if USE_ITT_NOTIFY
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_destroy( lock );
    KMP_ITT_DEBUG_PRINT( "[lck dst] sdes( %p )\n", lock );
#endif
} // ___kmp_itt_lock_fini


// -------------------------------------------------------------------------------------------------

void
__kmp_itt_lock_creating( kmp_user_lock_p lock ) {
    ___kmp_itt_lock_init( lock, "OMP Lock" );
} // __kmp_itt_lock_creating

void
__kmp_itt_lock_acquiring( kmp_user_lock_p lock ) {
    __itt_sync_prepare( lock );
} // __kmp_itt_lock_acquiring

void
__kmp_itt_lock_acquired( kmp_user_lock_p lock ) {
    __itt_sync_acquired( lock );
} // __kmp_itt_lock_acquired

void
__kmp_itt_lock_releasing( kmp_user_lock_p lock ) {
    __itt_sync_releasing( lock );
} // __kmp_itt_lock_releasing

void
__kmp_itt_lock_cancelled( kmp_user_lock_p lock ) {
    __itt_sync_cancel( lock );
} // __kmp_itt_lock_cancelled

void
__kmp_itt_lock_destroyed( kmp_user_lock_p lock ) {
    ___kmp_itt_lock_fini( lock, "OMP Lock" );
} // __kmp_itt_lock_destroyed

/*
    ------------------------------------------------------------------------------------------------
    Critical reporting.

    Critical sections are treated exactly as locks (but have different object type).
    ------------------------------------------------------------------------------------------------
*/

void
__kmp_itt_critical_creating( kmp_user_lock_p lock ) {
    ___kmp_itt_lock_init( lock, "OMP Critical" );
} // __kmp_itt_critical_creating

void
__kmp_itt_critical_acquiring( kmp_user_lock_p lock ) {
    __itt_sync_prepare( lock );
} // __kmp_itt_critical_acquiring

void
__kmp_itt_critical_acquired( kmp_user_lock_p lock ) {
    __itt_sync_acquired( lock );
} // __kmp_itt_critical_acquired

void
__kmp_itt_critical_releasing( kmp_user_lock_p lock ) {
    __itt_sync_releasing( lock );
} // __kmp_itt_critical_releasing

void
__kmp_itt_critical_destroyed( kmp_user_lock_p lock ) {
    ___kmp_itt_lock_fini( lock, "OMP Critical" );
} // __kmp_itt_critical_destroyed

/*
    ------------------------------------------------------------------------------------------------
    Single reporting.
    ------------------------------------------------------------------------------------------------
*/

void
__kmp_itt_single_start( int gtid ) {
#if USE_ITT_NOTIFY
    if ( __itt_mark_create_ptr || KMP_ITT_DEBUG ) {
        kmp_info_t *   thr = __kmp_thread_from_gtid( (gtid) );
        ident_t *      loc = thr->th.th_ident;
        char const *   src = ( loc == NULL ? NULL : loc->psource );
        kmp_str_buf_t  name;
        __kmp_str_buf_init( & name );
        __kmp_str_buf_print( & name, "OMP Single-%s", src );
        KMP_ITT_DEBUG_LOCK();
        thr->th.th_itt_mark_single = __itt_mark_create( name.str );
        KMP_ITT_DEBUG_PRINT( "[sin sta] mcre( \"%s\") -> %d\n", name.str, thr->th.th_itt_mark_single );
        __kmp_str_buf_free( & name );
        KMP_ITT_DEBUG_LOCK();
        __itt_mark( thr->th.th_itt_mark_single, NULL );
        KMP_ITT_DEBUG_PRINT( "[sin sta] mark( %d, NULL )\n", thr->th.th_itt_mark_single );
    }; // if
#endif
} // __kmp_itt_single_start

void
__kmp_itt_single_end( int gtid ) {
#if USE_ITT_NOTIFY
    __itt_mark_type  mark = __kmp_thread_from_gtid( gtid )->th.th_itt_mark_single;
    KMP_ITT_DEBUG_LOCK();
    __itt_mark_off( mark );
    KMP_ITT_DEBUG_PRINT( "[sin end] moff( %d )\n", mark );
#endif
} // __kmp_itt_single_end

/*
    ------------------------------------------------------------------------------------------------
    Ordered reporting.

    __kmp_itt_ordered_init is called by each thread *before* first using sync
    object. ITT team would like it to be called once, but it requires extra synchronization.

    __kmp_itt_ordered_prep is called when thread is going to enter ordered section
    (before synchronization).

    __kmp_itt_ordered_start is called just before entering user code (after
    synchronization).

    __kmp_itt_ordered_end is called after returning from user code.

    Sync object is th->th.th_dispatch->th_dispatch_sh_current.

    Events are not generated in case of serialized team.
    ------------------------------------------------------------------------------------------------
*/

void
__kmp_itt_ordered_init( int gtid ) {
#if USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr ) {
        kmp_info_t *   thr   = __kmp_thread_from_gtid( gtid );
        ident_t const * loc  = thr->th.th_ident;
        char const *    src  = ( loc == NULL ? NULL : loc->psource );
        __itt_sync_create(
            thr->th.th_dispatch->th_dispatch_sh_current, "OMP Ordered", src, 0
        );
    }; // if
#endif
} // __kmp_itt_ordered_init

void
__kmp_itt_ordered_prep( int gtid ) {
#if USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr ) {
        kmp_team_t * t = __kmp_team_from_gtid( gtid );
        if ( ! t->t.t_serialized ) {
            kmp_info_t * th = __kmp_thread_from_gtid( gtid );
            __itt_sync_prepare( th->th.th_dispatch->th_dispatch_sh_current );
        }; // if
    }; // if
#endif
} // __kmp_itt_ordered_prep

void
__kmp_itt_ordered_start( int gtid ) {
#if USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr ) {
        kmp_team_t * t = __kmp_team_from_gtid( gtid );
        if ( ! t->t.t_serialized ) {
            kmp_info_t * th = __kmp_thread_from_gtid( gtid );
            __itt_sync_acquired( th->th.th_dispatch->th_dispatch_sh_current );
        }; // if
    }; // if
#endif
} // __kmp_itt_ordered_start

void
__kmp_itt_ordered_end( int gtid ) {
#if USE_ITT_NOTIFY
    if ( __itt_sync_create_ptr ) {
        kmp_team_t * t = __kmp_team_from_gtid( gtid );
        if ( ! t->t.t_serialized ) {
            kmp_info_t * th = __kmp_thread_from_gtid( gtid );
            __itt_sync_releasing( th->th.th_dispatch->th_dispatch_sh_current );
        }; // if
    }; // if
#endif
} // __kmp_itt_ordered_end


/*
    ------------------------------------------------------------------------------------------------
    Threads reporting.
    ------------------------------------------------------------------------------------------------
*/

void
__kmp_itt_thread_ignore() {
    __itt_thr_ignore();
} // __kmp_itt_thread_ignore

void
__kmp_itt_thread_name( int gtid ) {
#if USE_ITT_NOTIFY
    if ( __itt_thr_name_set_ptr ) {
        kmp_str_buf_t name;
        __kmp_str_buf_init( & name );
        __kmp_str_buf_print( & name, "OMP Worker Thread #%d", gtid );
        KMP_ITT_DEBUG_LOCK();
        __itt_thr_name_set( name.str, name.used );
        KMP_ITT_DEBUG_PRINT( "[thr nam] name( \"%s\")\n", name.str );
        __kmp_str_buf_free( & name );
    }; // if
#endif
} // __kmp_itt_thread_name


/*
    --------------------------------------------------------------------------
    System object reporting.

    ITT catches operations with system sync objects (like Windows* OS on IA-32
    architecture API critical sections and events). We only need to specify
    name ("OMP Scheduler") for the object to let ITT know it is an object used
    by OpenMP RTL for internal purposes.
    --------------------------------------------------------------------------
*/

void
__kmp_itt_system_object_created( void * object, char const * name ) {
#if USE_ITT_NOTIFY
    KMP_ITT_DEBUG_LOCK();
    __itt_sync_create( object, "OMP Scheduler", name, 0 );
   KMP_ITT_DEBUG_PRINT( "[sys obj] scre( %p, \"OMP Scheduler\", \"%s\", 0 )\n", object, name );
#endif
} // __kmp_itt_system_object_created


/*
    ------------------------------------------------------------------------------------------------
    Stack stitching api.

    Master calls "create" and put the stitching id into team structure.
    Workers read the stitching id and call "enter" / "leave" api.
    Master calls "destroy" at the end of the parallel region.
    ------------------------------------------------------------------------------------------------
*/

__itt_caller
__kmp_itt_stack_caller_create()
{
#if USE_ITT_NOTIFY
    if ( !__itt_stack_caller_create_ptr )
        return NULL;
    KMP_ITT_DEBUG_LOCK();
    __itt_caller id = __itt_stack_caller_create();
    KMP_ITT_DEBUG_PRINT( "[stk cre] %p\n", id );
    return id;
#endif
    return NULL;
}

void
__kmp_itt_stack_caller_destroy( __itt_caller id )
{
#if USE_ITT_NOTIFY
    if ( __itt_stack_caller_destroy_ptr ) {
        KMP_ITT_DEBUG_LOCK();
        __itt_stack_caller_destroy( id );
        KMP_ITT_DEBUG_PRINT( "[stk des] %p\n", id );
    }
#endif
}

void
__kmp_itt_stack_callee_enter( __itt_caller id )
{
#if USE_ITT_NOTIFY
    if ( __itt_stack_callee_enter_ptr ) {
        KMP_ITT_DEBUG_LOCK();
        __itt_stack_callee_enter( id );
        KMP_ITT_DEBUG_PRINT( "[stk ent] %p\n", id );
    }
#endif
}

void
__kmp_itt_stack_callee_leave( __itt_caller id )
{
#if USE_ITT_NOTIFY
    if ( __itt_stack_callee_leave_ptr ) {
        KMP_ITT_DEBUG_LOCK();
        __itt_stack_callee_leave( id );
        KMP_ITT_DEBUG_PRINT( "[stk lea] %p\n", id );
    }
#endif
}

#endif /* USE_ITT_BUILD */
