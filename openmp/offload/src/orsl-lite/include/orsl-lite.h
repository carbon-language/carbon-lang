//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef _ORSL_LITE_H_
#define _ORSL_LITE_H_

#ifndef TARGET_WINNT
#include <sched.h>
#else
#define cpu_set_t int
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Type of a ORSLBusySet */
typedef enum ORSLBusySetType {
    BUSY_SET_EMPTY = 0,     /**< Empty set */
    BUSY_SET_PARTIAL = 1,   /**< Non-empty set that omits some threads */
    BUSY_SET_FULL = 2       /**< A set that includes all threads on the card */
} BusySetType;

/** ORSLBusySet encapsulation */
typedef struct ORSLBusySet {
    BusySetType type;   /**< Set type */
#ifdef __linux__
    cpu_set_t cpu_set;  /**< CPU mask (unused for BUSY_SET_EMPTY and
                           BUSY_SET_PARTIAL sets) represented by the standard
                           Linux CPU set type -- cpu_set_t. Threads are numbered
                           starting from 0. The maximal possible thread number
                           is system-specific. See CPU_SET(3) family of macros
                           for more details. Unused in ORSL Lite. */
#endif
} ORSLBusySet;

/** Client tag */
typedef char* ORSLTag;

/** Maximal length of tag in characters */
#define ORSL_MAX_TAG_LEN 128

/** Maximal number of cards that can be managed by ORSL */
#define ORSL_MAX_CARDS 32

/** Reserves computational resources on a set of cards. Blocks.
 *
 * If any of the resources cannot be reserved, this function will block until
 * they become available. Reservation can be recursive if performed by the
 * same tag. A recursively reserved resource must be released the same number
 * of times it was reserved.
 *
 * @see ORSLTryReserve
 *
 * @param[in]  n      Number of cards to reserve resources on. Cannot be < 0
 *                    or > ORSL_MAX_CARDS.
 *
 * @param[in]  inds   Indices of the cards: an integer array with n elements.
 *                    Cannot be NULL if n > 0. Valid card indices are from 0
 *                    to ORSL_MAX_CARDS-1. Cannot contain duplicate elements.
 *
 * @param[in]  bsets  Requested resources on each of the card. Cannot be NULL
 *                    if n > 0.
 *
 * @param[in]  tag    ORSLTag of the calling client. Cannot be NULL. Length
 *                    must not exeed ORSL_MAX_TAG_LEN.
 *
 * @returns    0      if the resources were successfully reserved
 *
 * @returns    EINVAL if any of the arguments is invalid
 *
 * @returns    EAGAIN limit of recursive reservations reached
 *                    (not in ORSL Lite)
 *
 * @returns    ENOSYS (in ORSL Lite) if type of any of the busy sets is
 *                    equal to BUSY_SET_PARTIAL
 */
int ORSLReserve(const int n, const int *__restrict inds,
                const ORSLBusySet *__restrict bsets,
                const ORSLTag __restrict tag);

/** Reserves computational resources on a set of cards. Does not block.
 *
 * If any of the resources cannot be reserved, this function will return
 * immediately. Reservation can be recursive if performed by the same tag.
 * A recursively reserved resource must be released the same number of times
 * it was reserved.
 *
 * @see ORSLReserve
 *
 * @param[in]  n      Number of cards to reserve resources on. Cannot be < 0
 *                    or > ORSL_MAX_CARDS.
 *
 * @param[in]  inds     Indices of the cards: an integer array with n elements.
 *                      Cannot be NULL if n > 0. Valid card indices are from 0
 *                      to ORSL_MAX_CARDS-1. Cannot contain duplicate elements.
 *
 * @param[inout] bsets  Requested resources on each of the card. Cannot be
 *                      NULL if n > 0.
 *
 * @param[in]    tag    ORSLTag of the calling client. Cannot be NULL. Length
 *                      must not exceed ORSL_MAX_TAG_LEN.
 *
 * @returns      0      if the resources were successfully reserved
 *
 * @returns      EBUSY  if some of the requested resources are busy
 *
 * @returns      EINVAL if any of the arguments is invalid
 *
 * @returns      EAGAIN limit of recursive reservations reached
 *                      (not in ORSL Lite)
 *
 * @returns      ENOSYS (in ORSL Lite) if type of any of the busy sets is
 *                      equal to BUSY_SET_PARTIAL
 */
int ORSLTryReserve(const int n, const int *__restrict inds,
                   const ORSLBusySet *__restrict bsets,
                   const ORSLTag __restrict tag);

/** Granularify of partial reservation */
typedef enum ORSLPartialGranularity {
    GRAN_CARD = 0, /**< Card granularity */
    GRAN_THREAD = 1 /**< Thread granularity */
} ORSLPartialGranularity;

/** Requests reservation of some of computational resources on a set of cards.
 * Does not block. Updates user-provided bsets to indicate which resources
 * were reserved.
 *
 * If any of the resources cannot be reserved, this function will update busy
 * sets provided by the caller to reflect what resources were actually
 * reserved. This function supports two granularity modes: 'card' and
 * 'thread'.  When granularity is set to 'card', a failure to reserve a thread
 * on the card will imply that reservation has failed for the whole card. When
 * granularity is set to 'thread', reservation on a card will be considered
 * successful as long as at least one thread on the card was successfully
 * reserved. Reservation can be recursive if performed by the same tag. A
 * recursively reserved resource must be released the same number of times it
 * was reserved.
 *
 * @param[in]  gran   Reservation granularity
 *
 * @param[in]  n      Number of cards to reserve resources on. Cannot be < 0
 *                    or > ORSL_MAX_CARDS.
 *
 * @param[in]  inds   Indices of the cards: an integer array with n elements.
 *                    Cannot be NULL if n > 0. Valid card indices are from 0
 *                    to ORSL_MAX_CARDS-1. Cannot contain duplicate elements.
 *
 * @param[in]  bsets  Requested resources on each of the card. Cannot be NULL
 *                    if n > 0.
 *
 * @param[in]  tag    ORSLTag of the calling client. Cannot be NULL. Length
 *                    must not exceed ORSL_MAX_TAG_LEN.
 *
 * @returns    0      if at least some of the resources were successfully
 *                    reserved
 *
 * @returns    EBUSY  if all of the requested resources are busy
 *
 * @returns    EINVAL if any of the arguments is invalid
 *
 * @returns    EAGAIN limit of recursive reservations reached
 *                    (not in ORSL Lite)
 *
 * @returns    ENOSYS (in ORSL Lite) if type of any of the busy sets is
 *                    equal to BUSY_SET_PARTIAL
 */
int ORSLReservePartial(const ORSLPartialGranularity gran, const int n,
                       const int *__restrict inds,
                       ORSLBusySet *__restrict bsets,
                       const ORSLTag __restrict tag);

/** Releases previously reserved computational resources on a set of cards.
 *
 * This function will fail if any of the resources to be released were not
 * reserved by the calling client.
 *
 * @see ORSLReserve
 * @see ORSLTryReserve
 * @see ORSLReservePartial
 *
 * @param[in]  n      Number of cards to reserve resources on. Cannot be < 0
 *                    or > ORSL_MAX_CARDS.
 *
 * @param[in]  inds   Indices of the cards: an integer array with n elements.
 *                    Cannot be NULL if n > 0. Valid card indices are from 0
 *                    to ORSL_MAX_CARDS-1. Cannot contain duplicate elements.
 *
 * @param[in]  bsets  Requested resources on each of the card. Cannot be NULL
 *                    if n > 0.
 *
 * @param[in]  tag    ORSLTag of the calling client. Cannot be NULL. Length
 *                    must not exceed ORSL_MAX_TAG_LEN.
 *
 * @returns    0      if the resources were successfully released
 *
 * @returns    EINVAL if any of the arguments is invalid
 *
 * @returns    EPERM  the calling client did not reserve some of the
 *                    resources it is trying to release.
 *
 * @returns    ENOSYS (in ORSL Lite) if type of any of the busy sets is
 *                    equal to BUSY_SET_PARTIAL
 */
int ORSLRelease(const int n, const int *__restrict inds,
                const ORSLBusySet *__restrict bsets,
                const ORSLTag __restrict tag);

#ifdef __cplusplus
}
#endif

#endif
