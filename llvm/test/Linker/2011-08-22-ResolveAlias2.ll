; This file is used by 2011-08-22-ResolveAlias.ll
; RUN: true

%struct.HexxagonBoard = type { %struct.BitBoard64, %struct.BitBoard64 }
%struct.BitBoard64 = type { i32, i32 }
%union.pthread_attr_t = type { [56 x i8] }
%union.pthread_mutex_t = type { [40 x i8] }
%struct.timespec = type { i64, i64 }
%union.pthread_mutexattr_t = type { [4 x i8] }
%union.pthread_cond_t = type { [48 x i8] }

@_ZN13HexxagonBoardC1ERKS_ = alias void (%struct.HexxagonBoard*, %struct.HexxagonBoard*)* @_ZN13HexxagonBoardC2ERKS_
@_ZL20__gthrw_pthread_oncePiPFvvE = weak alias i32 (i32*, void ()*)* @pthread_once
@_ZL27__gthrw_pthread_getspecificj = weak alias i8* (i32)* @pthread_getspecific
@_ZL27__gthrw_pthread_setspecificjPKv = weak alias i32 (i32, i8*)* @pthread_setspecific
@_ZL22__gthrw_pthread_createPmPK14pthread_attr_tPFPvS3_ES3_ = weak alias i32 (i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*)* @pthread_create
@_ZL20__gthrw_pthread_joinmPPv = weak alias i32 (i64, i8**)* @pthread_join
@_ZL21__gthrw_pthread_equalmm = weak alias i32 (i64, i64)* @pthread_equal
@_ZL20__gthrw_pthread_selfv = weak alias i64 ()* @pthread_self
@_ZL22__gthrw_pthread_detachm = weak alias i32 (i64)* @pthread_detach
@_ZL22__gthrw_pthread_cancelm = weak alias i32 (i64)* @pthread_cancel
@_ZL19__gthrw_sched_yieldv = weak alias i32 ()* @sched_yield
@_ZL26__gthrw_pthread_mutex_lockP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*)* @pthread_mutex_lock
@_ZL29__gthrw_pthread_mutex_trylockP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*)* @pthread_mutex_trylock
@_ZL31__gthrw_pthread_mutex_timedlockP15pthread_mutex_tPK8timespec = weak alias i32 (%union.pthread_mutex_t*, %struct.timespec*)* @pthread_mutex_timedlock
@_ZL28__gthrw_pthread_mutex_unlockP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*)* @pthread_mutex_unlock
@_ZL26__gthrw_pthread_mutex_initP15pthread_mutex_tPK19pthread_mutexattr_t = weak alias i32 (%union.pthread_mutex_t*, %union.pthread_mutexattr_t*)* @pthread_mutex_init
@_ZL29__gthrw_pthread_mutex_destroyP15pthread_mutex_t = weak alias i32 (%union.pthread_mutex_t*)* @pthread_mutex_destroy
@_ZL30__gthrw_pthread_cond_broadcastP14pthread_cond_t = weak alias i32 (%union.pthread_cond_t*)* @pthread_cond_broadcast
@_ZL27__gthrw_pthread_cond_signalP14pthread_cond_t = weak alias i32 (%union.pthread_cond_t*)* @pthread_cond_signal
@_ZL25__gthrw_pthread_cond_waitP14pthread_cond_tP15pthread_mutex_t = weak alias i32 (%union.pthread_cond_t*, %union.pthread_mutex_t*)* @pthread_cond_wait
@_ZL30__gthrw_pthread_cond_timedwaitP14pthread_cond_tP15pthread_mutex_tPK8timespec = weak alias i32 (%union.pthread_cond_t*, %union.pthread_mutex_t*, %struct.timespec*)* @pthread_cond_timedwait
@_ZL28__gthrw_pthread_cond_destroyP14pthread_cond_t = weak alias i32 (%union.pthread_cond_t*)* @pthread_cond_destroy
@_ZL26__gthrw_pthread_key_createPjPFvPvE = weak alias i32 (i32*, void (i8*)*)* @pthread_key_create
@_ZL26__gthrw_pthread_key_deletej = weak alias i32 (i32)* @pthread_key_delete
@_ZL30__gthrw_pthread_mutexattr_initP19pthread_mutexattr_t = weak alias i32 (%union.pthread_mutexattr_t*)* @pthread_mutexattr_init
@_ZL33__gthrw_pthread_mutexattr_settypeP19pthread_mutexattr_ti = weak alias i32 (%union.pthread_mutexattr_t*, i32)* @pthread_mutexattr_settype
@_ZL33__gthrw_pthread_mutexattr_destroyP19pthread_mutexattr_t = weak alias i32 (%union.pthread_mutexattr_t*)* @pthread_mutexattr_destroy

define void @_ZN13HexxagonBoardC2ERKS_(%struct.HexxagonBoard*, %struct.HexxagonBoard*) uwtable align 2 {
  ret void
}

define weak i32 @pthread_once(i32*, void ()*) {
  ret i32 0
}

define weak i8* @pthread_getspecific(i32) {
  ret i8* null
}

define weak i32 @pthread_setspecific(i32, i8*) {
  ret i32 0
}

define weak i32 @pthread_create(i64*, %union.pthread_attr_t*, i8* (i8*)*, i8*) {
  ret i32 0
}

define weak i32 @pthread_join(i64, i8**) {
  ret i32 0
}

define weak i32 @pthread_equal(i64, i64) {
  ret i32 0
}

define weak i64 @pthread_self() {
  ret i64 0
}

define weak i32 @pthread_detach(i64) {
  ret i32 0
}

define weak i32 @pthread_cancel(i64) {
  ret i32 0
}

define weak i32 @sched_yield() {
  ret i32 0
}

define weak i32 @pthread_mutex_lock(%union.pthread_mutex_t*) {
  ret i32 0
}

define weak i32 @pthread_mutex_trylock(%union.pthread_mutex_t*) {
  ret i32 0
}

define weak i32 @pthread_mutex_timedlock(%union.pthread_mutex_t*, %struct.timespec*) {
  ret i32 0
}

define weak i32 @pthread_mutex_unlock(%union.pthread_mutex_t*) {
  ret i32 0
}

define weak i32 @pthread_mutex_init(%union.pthread_mutex_t*, %union.pthread_mutexattr_t*) {
  ret i32 0
}

define weak i32 @pthread_mutex_destroy(%union.pthread_mutex_t*) {
  ret i32 0
}

define weak i32 @pthread_cond_broadcast(%union.pthread_cond_t*) {
  ret i32 0
}

define weak i32 @pthread_cond_signal(%union.pthread_cond_t*) {
  ret i32 0
}

define weak i32 @pthread_cond_wait(%union.pthread_cond_t*, %union.pthread_mutex_t*) {
  ret i32 0
}

define weak i32 @pthread_cond_timedwait(%union.pthread_cond_t*, %union.pthread_mutex_t*, %struct.timespec*) {
  ret i32 0
}

define weak i32 @pthread_cond_destroy(%union.pthread_cond_t*) {
  ret i32 0
}

define weak i32 @pthread_key_create(i32*, void (i8*)*) {
  ret i32 0
}

define weak i32 @pthread_key_delete(i32) {
  ret i32 0
}

define weak i32 @pthread_mutexattr_init(%union.pthread_mutexattr_t*) {
  ret i32 0
}

define weak i32 @pthread_mutexattr_settype(%union.pthread_mutexattr_t*, i32) {
  ret i32 0
}

define weak i32 @pthread_mutexattr_destroy(%union.pthread_mutexattr_t*) {
  ret i32 0
}
