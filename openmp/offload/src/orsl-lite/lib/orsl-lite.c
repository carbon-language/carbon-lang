//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include <errno.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#include "orsl-lite/include/orsl-lite.h"

#define DISABLE_SYMBOL_VERSIONING

#if defined(__linux__) && !defined(DISABLE_SYMBOL_VERSIONING)
#define symver(src, tgt, verstr) __asm__(".symver " #src "," #tgt verstr)
symver(ORSLReserve0, ORSLReserve, "@@ORSL_0.0");
symver(ORSLTryReserve0, ORSLTryReserve, "@@ORSL_0.0");
symver(ORSLReservePartial0, ORSLReservePartial, "@@ORSL_0.0");
symver(ORSLRelease0, ORSLRelease, "@@ORSL_0.0");
#else
#define ORSLReserve0 ORSLReserve
#define ORSLTryReserve0 ORSLTryReserve
#define ORSLReservePartial0 ORSLReservePartial
#define ORSLRelease0 ORSLRelease
#endif

#ifdef __linux__
#include <pthread.h>
static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t release_cond = PTHREAD_COND_INITIALIZER;
#endif

#ifdef _WIN32
#include <windows.h>
#pragma intrinsic(_ReadWriteBarrier)
static SRWLOCK global_mutex = SRWLOCK_INIT;
static volatile int release_cond_initialized = 0;
static CONDITION_VARIABLE release_cond;

static void state_lazy_init_sync()
{
    if (!release_cond_initialized) {
        AcquireSRWLockExclusive(&global_mutex);
        _ReadWriteBarrier();
        if (!release_cond_initialized) {
            InitializeConditionVariable(&release_cond);
            release_cond_initialized = 1;
        }
        ReleaseSRWLockExclusive(&global_mutex);
    }
}
#endif

static int state_lock()
{
#ifdef __linux__
    return pthread_mutex_lock(&global_mutex);
#endif

#ifdef _WIN32
    AcquireSRWLockExclusive(&global_mutex);
    return 0;
#endif
}

static int state_unlock()
{
#ifdef __linux__
    return pthread_mutex_unlock(&global_mutex);
#endif

#ifdef _WIN32
    ReleaseSRWLockExclusive(&global_mutex);
    return 0;
#endif
}

static int state_wait_for_release()
{
#ifdef __linux__
    return pthread_cond_wait(&release_cond, &global_mutex);
#endif

#ifdef _WIN32
    return SleepConditionVariableSRW(&release_cond,
            &global_mutex, INFINITE, 0) == 0 ? 1 : 0;
#endif
}

static int state_signal_release()
{
#ifdef __linux__
    return pthread_cond_signal(&release_cond);
#endif

#ifdef _WIN32
    WakeConditionVariable(&release_cond);
    return 0;
#endif
}

static struct {
    char owner[ORSL_MAX_TAG_LEN + 1];
    unsigned long rsrv_cnt;
} rsrv_data[ORSL_MAX_CARDS];

static int check_args(const int n, const int *__restrict inds,
                      const ORSLBusySet *__restrict bsets,
                      const ORSLTag __restrict tag)
{
    int i;
    int card_specified[ORSL_MAX_CARDS];
    if (tag == NULL) return -1;
    if (strlen((char *)tag) > ORSL_MAX_TAG_LEN) return -1;
    if (n < 0 || n >= ORSL_MAX_CARDS) return -1;
    if (n != 0 && (inds == NULL || bsets == NULL)) return -1;
    for (i = 0; i < ORSL_MAX_CARDS; i++)
        card_specified[i] = 0;
    for (i = 0; i < n; i++) {
        int ind = inds[i];
        if (ind < 0 || ind >= ORSL_MAX_CARDS) return -1;
        if (card_specified[ind]) return -1;
        card_specified[ind] = 1;
    }
    return 0;
}

static int check_bsets(const int n, const ORSLBusySet *bsets)
{
    int i;
    for (i = 0; i < n; i++)
        if (bsets[i].type == BUSY_SET_PARTIAL) return -1;
    return 0;
}

static int can_reserve_card(int card, const ORSLBusySet *__restrict bset,
                            const ORSLTag __restrict tag)
{
    assert(tag != NULL);
    assert(bset != NULL);
    assert(strlen((char *)tag) < ORSL_MAX_TAG_LEN);
    assert(bset->type != BUSY_SET_PARTIAL);

    return (bset->type == BUSY_SET_EMPTY ||
            ((rsrv_data[card].rsrv_cnt == 0 ||
            strncmp((char *)tag,
                rsrv_data[card].owner, ORSL_MAX_TAG_LEN) == 0) &&
            rsrv_data[card].rsrv_cnt < ULONG_MAX)) ? 0 : - 1;
}

static void reserve_card(int card, const ORSLBusySet *__restrict bset,
                         const ORSLTag __restrict tag)
{
    assert(tag != NULL);
    assert(bset != NULL);
    assert(strlen((char *)tag) < ORSL_MAX_TAG_LEN);
    assert(bset->type != BUSY_SET_PARTIAL);

    if (bset->type == BUSY_SET_EMPTY)
        return;

    assert(rsrv_data[card].rsrv_cnt == 0 ||
            strncmp((char *)tag,
                rsrv_data[card].owner, ORSL_MAX_TAG_LEN) == 0);
    assert(rsrv_data[card].rsrv_cnt < ULONG_MAX);

    if (rsrv_data[card].rsrv_cnt == 0)
        strncpy(rsrv_data[card].owner, (char *)tag, ORSL_MAX_TAG_LEN);
    rsrv_data[card].owner[ORSL_MAX_TAG_LEN] = '\0';
    rsrv_data[card].rsrv_cnt++;
}

static int can_release_card(int card, const ORSLBusySet *__restrict bset,
                            const ORSLTag __restrict tag)
{
    assert(tag != NULL);
    assert(bset != NULL);
    assert(strlen((char *)tag) < ORSL_MAX_TAG_LEN);
    assert(bset->type != BUSY_SET_PARTIAL);

    return (bset->type == BUSY_SET_EMPTY || (rsrv_data[card].rsrv_cnt > 0 &&
                strncmp((char *)tag,
                    rsrv_data[card].owner, ORSL_MAX_TAG_LEN) == 0)) ? 0 : 1;
}

static void release_card(int card, const ORSLBusySet *__restrict bset,
                         const ORSLTag __restrict tag)
{
    assert(tag != NULL);
    assert(bset != NULL);
    assert(strlen((char *)tag) < ORSL_MAX_TAG_LEN);
    assert(bset->type != BUSY_SET_PARTIAL);

    if (bset->type == BUSY_SET_EMPTY)
        return;

    assert(strncmp((char *)tag,
                rsrv_data[card].owner, ORSL_MAX_TAG_LEN) == 0);
    assert(rsrv_data[card].rsrv_cnt > 0);

    rsrv_data[card].rsrv_cnt--;
}

int ORSLReserve0(const int n, const int *__restrict inds,
                const ORSLBusySet *__restrict bsets,
                const ORSLTag __restrict tag)
{
    int i, ok;

    if (n == 0) return 0;
    if (check_args(n, inds, bsets, tag) != 0) return EINVAL;
    if (check_bsets(n, bsets) != 0) return ENOSYS;

    state_lock();

    /* Loop until we find that all the resources we want are available */
    do {
        ok = 1;
        for (i = 0; i < n; i++)
            if (can_reserve_card(inds[i], &bsets[i], tag) != 0) {
                ok = 0;
                /* Wait for someone to release some resources */
                state_wait_for_release();
                break;
            }
    } while (!ok);

    /* At this point we are good to reserve_card the resources we want */
    for (i = 0; i < n; i++)
        reserve_card(inds[i], &bsets[i], tag);

    state_unlock();
    return 0;
}

int ORSLTryReserve0(const int n, const int *__restrict inds,
                   const ORSLBusySet *__restrict bsets,
                   const ORSLTag __restrict tag)
{
    int i, rc = EBUSY;

    if (n == 0) return 0;
    if (check_args(n, inds, bsets, tag) != 0) return EINVAL;
    if (check_bsets(n, bsets) != 0) return ENOSYS;

    state_lock();

    /* Check resource availability once */
    for (i = 0; i < n; i++)
        if (can_reserve_card(inds[i], &bsets[i], tag) != 0)
            goto bail_out;

    /* At this point we are good to reserve the resources we want */
    for (i = 0; i < n; i++)
        reserve_card(inds[i], &bsets[i], tag);

    rc = 0;

bail_out:
    state_unlock();
    return rc;
}

int ORSLReservePartial0(const ORSLPartialGranularity gran, const int n,
                       const int *__restrict inds, ORSLBusySet *__restrict bsets,
                       const ORSLTag __restrict tag)
{
    int rc = EBUSY;
    int i, num_avail = n;

    if (n == 0) return 0;
    if (gran != GRAN_CARD && gran != GRAN_THREAD) return EINVAL;
    if (gran != GRAN_CARD) return EINVAL;
    if (check_args(n, inds, bsets, tag) != 0) return EINVAL;
    if (check_bsets(n, bsets) != 0) return ENOSYS;

    state_lock();

    /* Check resource availability once; remove unavailable resources from the
     * user-provided list */
    for (i = 0; i < n; i++)
        if (can_reserve_card(inds[i], &bsets[i], tag) != 0) {
            num_avail--;
            bsets[i].type = BUSY_SET_EMPTY;
        }

    if (num_avail == 0)
        goto bail_out;

    /* At this point we are good to reserve the resources we want */
    for (i = 0; i < n; i++)
        reserve_card(inds[i], &bsets[i], tag);

    rc = 0;

bail_out:
    state_unlock();
    return rc;
}

int ORSLRelease0(const int n, const int *__restrict inds,
                const ORSLBusySet *__restrict bsets,
                const ORSLTag __restrict tag)
{
    int i, rc = EPERM;

    if (n == 0) return 0;
    if (check_args(n, inds, bsets, tag) != 0) return EINVAL;
    if (check_bsets(n, bsets) != 0) return ENOSYS;

    state_lock();

    /* Check that we can release all the resources */
    for (i = 0; i < n; i++)
        if (can_release_card(inds[i], &bsets[i], tag) != 0)
            goto bail_out;

    /* At this point we are good to release the resources we want */
    for (i = 0; i < n; i++)
        release_card(inds[i], &bsets[i], tag);

    state_signal_release();

    rc = 0;

bail_out:
    state_unlock();
    return rc;
}

/* vim:set et: */
