//===----------- device.h - Target independent OpenMP target RTL ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for managing devices that are handled by RTL plugins.
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_DEVICE_H
#define _OMPTARGET_DEVICE_H

#include <cassert>
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "omptarget.h"
#include "rtl.h"

// Forward declarations.
struct RTLInfoTy;
struct __tgt_bin_desc;
struct __tgt_target_table;

using map_var_info_t = void *;

// enum for OMP_TARGET_OFFLOAD; keep in sync with kmp.h definition
enum kmp_target_offload_kind {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};
typedef enum kmp_target_offload_kind kmp_target_offload_kind_t;

/// Map between host data and target data.
struct HostDataToTargetTy {
  const uintptr_t HstPtrBase; // host info.
  const uintptr_t HstPtrBegin;
  const uintptr_t HstPtrEnd;       // non-inclusive.
  const map_var_info_t HstPtrName; // Optional source name of mapped variable.

  const uintptr_t TgtPtrBegin; // target info.

private:
  static const uint64_t INFRefCount = ~(uint64_t)0;
  static std::string refCountToStr(uint64_t RefCount) {
    return RefCount == INFRefCount ? "INF" : std::to_string(RefCount);
  }

  struct StatesTy {
    StatesTy(uint64_t DRC, uint64_t HRC)
        : DynRefCount(DRC), HoldRefCount(HRC) {}
    /// The dynamic reference count is the standard reference count as of OpenMP
    /// 4.5.  The hold reference count is an OpenMP extension for the sake of
    /// OpenACC support.
    ///
    /// The 'ompx_hold' map type modifier is permitted only on "omp target" and
    /// "omp target data", and "delete" is permitted only on "omp target exit
    /// data" and associated runtime library routines.  As a result, we really
    /// need to implement "reset" functionality only for the dynamic reference
    /// counter.  Likewise, only the dynamic reference count can be infinite
    /// because, for example, omp_target_associate_ptr and "omp declare target
    /// link" operate only on it.  Nevertheless, it's actually easier to follow
    /// the code (and requires less assertions for special cases) when we just
    /// implement these features generally across both reference counters here.
    /// Thus, it's the users of this class that impose those restrictions.
    ///
    uint64_t DynRefCount;
    uint64_t HoldRefCount;
    /// This mutex will be locked when data movement is issued. For targets that
    /// doesn't support async data movement, this mutex can guarantee that after
    /// it is released, memory region on the target is update to date. For
    /// targets that support async data movement, this can guarantee that data
    /// movement has been issued. This mutex *must* be locked right before
    /// releasing the mapping table lock.
    std::mutex UpdateMtx;
    /// Pointer to the event corresponding to the data update of this map.
    /// Note: At present this event is created when the first data transfer from
    /// host to device is issued, and only being used for H2D. It is not used
    /// for data transfer in another direction (device to host). It is still
    /// unclear whether we need it for D2H. If in the future we need similar
    /// mechanism for D2H, and if the event cannot be shared between them, Event
    /// should be written as <tt>void *Event[2]</tt>.
    void *Event = nullptr;
  };
  // When HostDataToTargetTy is used by std::set, std::set::iterator is const
  // use unique_ptr to make States mutable.
  const std::unique_ptr<StatesTy> States;

public:
  HostDataToTargetTy(uintptr_t BP, uintptr_t B, uintptr_t E, uintptr_t TB,
                     bool UseHoldRefCount, map_var_info_t Name = nullptr,
                     bool IsINF = false)
      : HstPtrBase(BP), HstPtrBegin(B), HstPtrEnd(E), HstPtrName(Name),
        TgtPtrBegin(TB), States(std::make_unique<StatesTy>(UseHoldRefCount ? 0
                                                           : IsINF ? INFRefCount
                                                                   : 1,
                                                           !UseHoldRefCount ? 0
                                                           : IsINF ? INFRefCount
                                                                   : 1)) {}

  /// Get the total reference count.  This is smarter than just getDynRefCount()
  /// + getHoldRefCount() because it handles the case where at least one is
  /// infinity and the other is non-zero.
  uint64_t getTotalRefCount() const {
    if (States->DynRefCount == INFRefCount ||
        States->HoldRefCount == INFRefCount)
      return INFRefCount;
    return States->DynRefCount + States->HoldRefCount;
  }

  /// Get the dynamic reference count.
  uint64_t getDynRefCount() const { return States->DynRefCount; }

  /// Get the hold reference count.
  uint64_t getHoldRefCount() const { return States->HoldRefCount; }

  /// Get the event bound to this data map.
  void *getEvent() const { return States->Event; }

  /// Set the event bound to this data map.
  void setEvent(void *Event) const { States->Event = Event; }

  /// Reset the specified reference count unless it's infinity.  Reset to 1
  /// (even if currently 0) so it can be followed by a decrement.
  void resetRefCount(bool UseHoldRefCount) const {
    uint64_t &ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    if (ThisRefCount != INFRefCount)
      ThisRefCount = 1;
  }

  /// Increment the specified reference count unless it's infinity.
  void incRefCount(bool UseHoldRefCount) const {
    uint64_t &ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    if (ThisRefCount != INFRefCount) {
      ++ThisRefCount;
      assert(ThisRefCount < INFRefCount && "refcount overflow");
    }
  }

  /// Decrement the specified reference count unless it's infinity or zero, and
  /// return the total reference count.
  uint64_t decRefCount(bool UseHoldRefCount) const {
    uint64_t &ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    uint64_t OtherRefCount =
        UseHoldRefCount ? States->DynRefCount : States->HoldRefCount;
    (void)OtherRefCount;
    if (ThisRefCount != INFRefCount) {
      if (ThisRefCount > 0)
        --ThisRefCount;
      else
        assert(OtherRefCount > 0 && "total refcount underflow");
    }
    return getTotalRefCount();
  }

  /// Is the dynamic (and thus the total) reference count infinite?
  bool isDynRefCountInf() const { return States->DynRefCount == INFRefCount; }

  /// Convert the dynamic reference count to a debug string.
  std::string dynRefCountToStr() const {
    return refCountToStr(States->DynRefCount);
  }

  /// Convert the hold reference count to a debug string.
  std::string holdRefCountToStr() const {
    return refCountToStr(States->HoldRefCount);
  }

  /// Should one decrement of the specified reference count (after resetting it
  /// if \c AfterReset) remove this mapping?
  bool decShouldRemove(bool UseHoldRefCount, bool AfterReset = false) const {
    uint64_t ThisRefCount =
        UseHoldRefCount ? States->HoldRefCount : States->DynRefCount;
    uint64_t OtherRefCount =
        UseHoldRefCount ? States->DynRefCount : States->HoldRefCount;
    if (OtherRefCount > 0)
      return false;
    if (AfterReset)
      return ThisRefCount != INFRefCount;
    return ThisRefCount == 1;
  }

  void lock() const { States->UpdateMtx.lock(); }

  void unlock() const { States->UpdateMtx.unlock(); }
};

typedef uintptr_t HstPtrBeginTy;
inline bool operator<(const HostDataToTargetTy &lhs, const HstPtrBeginTy &rhs) {
  return lhs.HstPtrBegin < rhs;
}
inline bool operator<(const HstPtrBeginTy &lhs, const HostDataToTargetTy &rhs) {
  return lhs < rhs.HstPtrBegin;
}
inline bool operator<(const HostDataToTargetTy &lhs,
                      const HostDataToTargetTy &rhs) {
  return lhs.HstPtrBegin < rhs.HstPtrBegin;
}

typedef std::set<HostDataToTargetTy, std::less<>> HostDataToTargetListTy;

struct LookupResult {
  struct {
    unsigned IsContained : 1;
    unsigned ExtendsBefore : 1;
    unsigned ExtendsAfter : 1;
  } Flags;

  HostDataToTargetListTy::iterator Entry;

  LookupResult() : Flags({0, 0, 0}), Entry() {}
};

/// This struct will be returned by \p DeviceTy::getTargetPointer which provides
/// more data than just a target pointer.
struct TargetPointerResultTy {
  struct {
    /// If the map table entry is just created
    unsigned IsNewEntry : 1;
    /// If the pointer is actually a host pointer (when unified memory enabled)
    unsigned IsHostPointer : 1;
  } Flags = {0, 0};

  /// The iterator to the corresponding map table entry
  HostDataToTargetListTy::iterator MapTableEntry{};

  /// The corresponding target pointer
  void *TargetPointer = nullptr;
};

/// Map for shadow pointers
struct ShadowPtrValTy {
  void *HstPtrVal;
  void *TgtPtrAddr;
  void *TgtPtrVal;
};
typedef std::map<void *, ShadowPtrValTy> ShadowPtrListTy;

///
struct PendingCtorDtorListsTy {
  std::list<void *> PendingCtors;
  std::list<void *> PendingDtors;
};
typedef std::map<__tgt_bin_desc *, PendingCtorDtorListsTy>
    PendingCtorsDtorsPerLibrary;

struct DeviceTy {
  int32_t DeviceID;
  RTLInfoTy *RTL;
  int32_t RTLDeviceID;

  bool IsInit;
  std::once_flag InitFlag;
  bool HasPendingGlobals;

  HostDataToTargetListTy HostDataToTargetMap;
  PendingCtorsDtorsPerLibrary PendingCtorsDtors;

  ShadowPtrListTy ShadowPtrMap;

  std::mutex DataMapMtx, PendingGlobalsMtx, ShadowMtx;

  // NOTE: Once libomp gains full target-task support, this state should be
  // moved into the target task in libomp.
  std::map<int32_t, uint64_t> LoopTripCnt;

  DeviceTy(RTLInfoTy *RTL);
  // DeviceTy is not copyable
  DeviceTy(const DeviceTy &D) = delete;
  DeviceTy &operator=(const DeviceTy &D) = delete;

  ~DeviceTy();

  // Return true if data can be copied to DstDevice directly
  bool isDataExchangable(const DeviceTy &DstDevice);

  LookupResult lookupMapping(void *HstPtrBegin, int64_t Size);
  /// Get the target pointer based on host pointer begin and base. If the
  /// mapping already exists, the target pointer will be returned directly. In
  /// addition, if required, the memory region pointed by \p HstPtrBegin of size
  /// \p Size will also be transferred to the device. If the mapping doesn't
  /// exist, and if unified shared memory is not enabled, a new mapping will be
  /// created and the data will also be transferred accordingly. nullptr will be
  /// returned because of any of following reasons:
  /// - Data allocation failed;
  /// - The user tried to do an illegal mapping;
  /// - Data transfer issue fails.
  TargetPointerResultTy
  getTargetPointer(void *HstPtrBegin, void *HstPtrBase, int64_t Size,
                   map_var_info_t HstPtrName, bool HasFlagTo,
                   bool HasFlagAlways, bool IsImplicit, bool UpdateRefCount,
                   bool HasCloseModifier, bool HasPresentModifier,
                   bool HasHoldModifier, AsyncInfoTy &AsyncInfo);
  void *getTgtPtrBegin(void *HstPtrBegin, int64_t Size);
  void *getTgtPtrBegin(void *HstPtrBegin, int64_t Size, bool &IsLast,
                       bool UpdateRefCount, bool UseHoldRefCount,
                       bool &IsHostPtr, bool MustContain = false,
                       bool ForceDelete = false);
  /// For the map entry for \p HstPtrBegin, decrement the reference count
  /// specified by \p HasHoldModifier and, if the the total reference count is
  /// then zero, deallocate the corresponding device storage and remove the map
  /// entry.  Return \c OFFLOAD_SUCCESS if the map entry existed, and return
  /// \c OFFLOAD_FAIL if not.  It is the caller's responsibility to skip calling
  /// this function if the map entry is not expected to exist because
  /// \p HstPtrBegin uses shared memory.
  int deallocTgtPtr(void *HstPtrBegin, int64_t Size, bool HasHoldModifier);
  int associatePtr(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size);
  int disassociatePtr(void *HstPtrBegin);

  // calls to RTL
  int32_t initOnce();
  __tgt_target_table *load_binary(void *Img);

  // device memory allocation/deallocation routines
  /// Allocates \p Size bytes on the device, host or shared memory space
  /// (depending on \p Kind) and returns the address/nullptr when
  /// succeeds/fails. \p HstPtr is an address of the host data which the
  /// allocated target data will be associated with. If it is unknown, the
  /// default value of \p HstPtr is nullptr. Note: this function doesn't do
  /// pointer association. Actually, all the __tgt_rtl_data_alloc
  /// implementations ignore \p HstPtr. \p Kind dictates what allocator should
  /// be used (host, shared, device).
  void *allocData(int64_t Size, void *HstPtr = nullptr,
                  int32_t Kind = TARGET_ALLOC_DEFAULT);
  /// Deallocates memory which \p TgtPtrBegin points at and returns
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails.
  int32_t deleteData(void *TgtPtrBegin);

  // Data transfer. When AsyncInfo is nullptr, the transfer will be
  // synchronous.
  // Copy data from host to device
  int32_t submitData(void *TgtPtrBegin, void *HstPtrBegin, int64_t Size,
                     AsyncInfoTy &AsyncInfo);
  // Copy data from device back to host
  int32_t retrieveData(void *HstPtrBegin, void *TgtPtrBegin, int64_t Size,
                       AsyncInfoTy &AsyncInfo);
  // Copy data from current device to destination device directly
  int32_t dataExchange(void *SrcPtr, DeviceTy &DstDev, void *DstPtr,
                       int64_t Size, AsyncInfoTy &AsyncInfo);

  int32_t runRegion(void *TgtEntryPtr, void **TgtVarsPtr, ptrdiff_t *TgtOffsets,
                    int32_t TgtVarsSize, AsyncInfoTy &AsyncInfo);
  int32_t runTeamRegion(void *TgtEntryPtr, void **TgtVarsPtr,
                        ptrdiff_t *TgtOffsets, int32_t TgtVarsSize,
                        int32_t NumTeams, int32_t ThreadLimit,
                        uint64_t LoopTripCount, AsyncInfoTy &AsyncInfo);

  /// Synchronize device/queue/event based on \p AsyncInfo and return
  /// OFFLOAD_SUCCESS/OFFLOAD_FAIL when succeeds/fails.
  int32_t synchronize(AsyncInfoTy &AsyncInfo);

  /// Calls the corresponding print in the \p RTLDEVID
  /// device RTL to obtain the information of the specific device.
  bool printDeviceInfo(int32_t RTLDevID);

  /// Event related interfaces.
  /// {
  /// Create an event.
  int32_t createEvent(void **Event);

  /// Record the event based on status in AsyncInfo->Queue at the moment the
  /// function is called.
  int32_t recordEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Wait for an event. This function can be blocking or non-blocking,
  /// depending on the implmentation. It is expected to set a dependence on the
  /// event such that corresponding operations shall only start once the event
  /// is fulfilled.
  int32_t waitEvent(void *Event, AsyncInfoTy &AsyncInfo);

  /// Synchronize the event. It is expected to block the thread.
  int32_t syncEvent(void *Event);

  /// Destroy the event.
  int32_t destroyEvent(void *Event);
  /// }

private:
  // Call to RTL
  void init(); // To be called only via DeviceTy::initOnce()
};

extern bool device_is_ready(int device_num);

/// Struct for the data required to handle plugins
struct PluginManager {
  /// RTLs identified on the host
  RTLsTy RTLs;

  /// Devices associated with RTLs
  std::vector<std::unique_ptr<DeviceTy>> Devices;
  std::mutex RTLsMtx; ///< For RTLs and Devices

  /// Translation table retreived from the binary
  HostEntriesBeginToTransTableTy HostEntriesBeginToTransTable;
  std::mutex TrlTblMtx; ///< For Translation Table
  /// Host offload entries in order of image registration
  std::vector<__tgt_offload_entry *> HostEntriesBeginRegistrationOrder;

  /// Map from ptrs on the host to an entry in the Translation Table
  HostPtrToTableMapTy HostPtrToTableMap;
  std::mutex TblMapMtx; ///< For HostPtrToTableMap

  // Store target policy (disabled, mandatory, default)
  kmp_target_offload_kind_t TargetOffloadPolicy = tgt_default;
  std::mutex TargetOffloadMtx; ///< For TargetOffloadPolicy
};

extern PluginManager *PM;

#endif
