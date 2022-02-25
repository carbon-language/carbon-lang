//===----------- rtl.cpp - Target independent OpenMP target RTL -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for handling RTL plugins.
//
//===----------------------------------------------------------------------===//

#include "rtl.h"
#include "device.h"
#include "private.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>

// List of all plugins that can support offloading.
static const char *RTLNames[] = {
    /* PowerPC target       */ "libomptarget.rtl.ppc64.so",
    /* x86_64 target        */ "libomptarget.rtl.x86_64.so",
    /* CUDA target          */ "libomptarget.rtl.cuda.so",
    /* AArch64 target       */ "libomptarget.rtl.aarch64.so",
    /* SX-Aurora VE target  */ "libomptarget.rtl.ve.so",
    /* AMDGPU target        */ "libomptarget.rtl.amdgpu.so",
    /* Remote target        */ "libomptarget.rtl.rpc.so",
};

PluginManager *PM;

#if OMPTARGET_PROFILE_ENABLED
static char *ProfileTraceFile = nullptr;
#endif

__attribute__((constructor(101))) void init() {
  DP("Init target library!\n");

  bool UseEventsForAtomicTransfers = true;
  if (const char *ForceAtomicMap = getenv("LIBOMPTARGET_MAP_FORCE_ATOMIC")) {
    std::string ForceAtomicMapStr(ForceAtomicMap);
    if (ForceAtomicMapStr == "false" || ForceAtomicMapStr == "FALSE")
      UseEventsForAtomicTransfers = false;
    else if (ForceAtomicMapStr != "true" && ForceAtomicMapStr != "TRUE")
      fprintf(stderr,
              "Warning: 'LIBOMPTARGET_MAP_FORCE_ATOMIC' accepts only "
              "'true'/'TRUE' or 'false'/'FALSE' as options, '%s' ignored\n",
              ForceAtomicMap);
  }

  PM = new PluginManager(UseEventsForAtomicTransfers);

#ifdef OMPTARGET_PROFILE_ENABLED
  ProfileTraceFile = getenv("LIBOMPTARGET_PROFILE");
  // TODO: add a configuration option for time granularity
  if (ProfileTraceFile)
    llvm::timeTraceProfilerInitialize(500 /* us */, "libomptarget");
#endif
}

__attribute__((destructor(101))) void deinit() {
  DP("Deinit target library!\n");
  delete PM;

#ifdef OMPTARGET_PROFILE_ENABLED
  if (ProfileTraceFile) {
    // TODO: add env var for file output
    if (auto E = llvm::timeTraceProfilerWrite(ProfileTraceFile, "-"))
      fprintf(stderr, "Error writing out the time trace\n");

    llvm::timeTraceProfilerCleanup();
  }
#endif
}

void RTLsTy::LoadRTLs() {
  // Parse environment variable OMP_TARGET_OFFLOAD (if set)
  PM->TargetOffloadPolicy =
      (kmp_target_offload_kind_t)__kmpc_get_target_offload();
  if (PM->TargetOffloadPolicy == tgt_disabled) {
    return;
  }

  DP("Loading RTLs...\n");

  // Attempt to open all the plugins and, if they exist, check if the interface
  // is correct and if they are supporting any devices.
  for (auto *Name : RTLNames) {
    DP("Loading library '%s'...\n", Name);
    void *dynlib_handle = dlopen(Name, RTLD_NOW);

    if (!dynlib_handle) {
      // Library does not exist or cannot be found.
      DP("Unable to load library '%s': %s!\n", Name, dlerror());
      continue;
    }

    DP("Successfully loaded library '%s'!\n", Name);

    AllRTLs.emplace_back();

    // Retrieve the RTL information from the runtime library.
    RTLInfoTy &R = AllRTLs.back();

    bool ValidPlugin = true;

    if (!(*((void **)&R.is_valid_binary) =
              dlsym(dynlib_handle, "__tgt_rtl_is_valid_binary")))
      ValidPlugin = false;
    if (!(*((void **)&R.number_of_devices) =
              dlsym(dynlib_handle, "__tgt_rtl_number_of_devices")))
      ValidPlugin = false;
    if (!(*((void **)&R.init_device) =
              dlsym(dynlib_handle, "__tgt_rtl_init_device")))
      ValidPlugin = false;
    if (!(*((void **)&R.load_binary) =
              dlsym(dynlib_handle, "__tgt_rtl_load_binary")))
      ValidPlugin = false;
    if (!(*((void **)&R.data_alloc) =
              dlsym(dynlib_handle, "__tgt_rtl_data_alloc")))
      ValidPlugin = false;
    if (!(*((void **)&R.data_submit) =
              dlsym(dynlib_handle, "__tgt_rtl_data_submit")))
      ValidPlugin = false;
    if (!(*((void **)&R.data_retrieve) =
              dlsym(dynlib_handle, "__tgt_rtl_data_retrieve")))
      ValidPlugin = false;
    if (!(*((void **)&R.data_delete) =
              dlsym(dynlib_handle, "__tgt_rtl_data_delete")))
      ValidPlugin = false;
    if (!(*((void **)&R.run_region) =
              dlsym(dynlib_handle, "__tgt_rtl_run_target_region")))
      ValidPlugin = false;
    if (!(*((void **)&R.run_team_region) =
              dlsym(dynlib_handle, "__tgt_rtl_run_target_team_region")))
      ValidPlugin = false;

    // Invalid plugin
    if (!ValidPlugin) {
      DP("Invalid plugin as necessary interface is not found.\n");
      AllRTLs.pop_back();
      continue;
    }

    // No devices are supported by this RTL?
    if (!(R.NumberOfDevices = R.number_of_devices())) {
      // The RTL is invalid! Will pop the object from the RTLs list.
      DP("No devices supported in this RTL\n");
      AllRTLs.pop_back();
      continue;
    }

    R.LibraryHandler = dynlib_handle;

#ifdef OMPTARGET_DEBUG
    R.RTLName = Name;
#endif

    DP("Registering RTL %s supporting %d devices!\n", R.RTLName.c_str(),
       R.NumberOfDevices);

    // Optional functions
    *((void **)&R.init_requires) =
        dlsym(dynlib_handle, "__tgt_rtl_init_requires");
    *((void **)&R.data_submit_async) =
        dlsym(dynlib_handle, "__tgt_rtl_data_submit_async");
    *((void **)&R.data_retrieve_async) =
        dlsym(dynlib_handle, "__tgt_rtl_data_retrieve_async");
    *((void **)&R.run_region_async) =
        dlsym(dynlib_handle, "__tgt_rtl_run_target_region_async");
    *((void **)&R.run_team_region_async) =
        dlsym(dynlib_handle, "__tgt_rtl_run_target_team_region_async");
    *((void **)&R.synchronize) = dlsym(dynlib_handle, "__tgt_rtl_synchronize");
    *((void **)&R.data_exchange) =
        dlsym(dynlib_handle, "__tgt_rtl_data_exchange");
    *((void **)&R.data_exchange_async) =
        dlsym(dynlib_handle, "__tgt_rtl_data_exchange_async");
    *((void **)&R.is_data_exchangable) =
        dlsym(dynlib_handle, "__tgt_rtl_is_data_exchangable");
    *((void **)&R.register_lib) =
        dlsym(dynlib_handle, "__tgt_rtl_register_lib");
    *((void **)&R.unregister_lib) =
        dlsym(dynlib_handle, "__tgt_rtl_unregister_lib");
    *((void **)&R.supports_empty_images) =
        dlsym(dynlib_handle, "__tgt_rtl_supports_empty_images");
    *((void **)&R.set_info_flag) =
        dlsym(dynlib_handle, "__tgt_rtl_set_info_flag");
    *((void **)&R.print_device_info) =
        dlsym(dynlib_handle, "__tgt_rtl_print_device_info");
    *((void **)&R.create_event) =
        dlsym(dynlib_handle, "__tgt_rtl_create_event");
    *((void **)&R.record_event) =
        dlsym(dynlib_handle, "__tgt_rtl_record_event");
    *((void **)&R.wait_event) = dlsym(dynlib_handle, "__tgt_rtl_wait_event");
    *((void **)&R.sync_event) = dlsym(dynlib_handle, "__tgt_rtl_sync_event");
    *((void **)&R.destroy_event) =
        dlsym(dynlib_handle, "__tgt_rtl_destroy_event");
    *((void **)&R.release_async_info) =
        dlsym(dynlib_handle, "__tgt_rtl_release_async_info");
    *((void **)&R.init_async_info) =
        dlsym(dynlib_handle, "__tgt_rtl_init_async_info");
    *((void **)&R.init_device_info) =
        dlsym(dynlib_handle, "__tgt_rtl_init_device_info");
  }

  DP("RTLs loaded!\n");

  return;
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering libs

static void RegisterImageIntoTranslationTable(TranslationTable &TT,
                                              RTLInfoTy &RTL,
                                              __tgt_device_image *image) {

  // same size, as when we increase one, we also increase the other.
  assert(TT.TargetsTable.size() == TT.TargetsImages.size() &&
         "We should have as many images as we have tables!");

  // Resize the Targets Table and Images to accommodate the new targets if
  // required
  unsigned TargetsTableMinimumSize = RTL.Idx + RTL.NumberOfDevices;

  if (TT.TargetsTable.size() < TargetsTableMinimumSize) {
    TT.TargetsImages.resize(TargetsTableMinimumSize, 0);
    TT.TargetsTable.resize(TargetsTableMinimumSize, 0);
  }

  // Register the image in all devices for this target type.
  for (int32_t i = 0; i < RTL.NumberOfDevices; ++i) {
    // If we are changing the image we are also invalidating the target table.
    if (TT.TargetsImages[RTL.Idx + i] != image) {
      TT.TargetsImages[RTL.Idx + i] = image;
      TT.TargetsTable[RTL.Idx + i] = 0; // lazy initialization of target table.
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functionality for registering Ctors/Dtors

static void RegisterGlobalCtorsDtorsForImage(__tgt_bin_desc *desc,
                                             __tgt_device_image *img,
                                             RTLInfoTy *RTL) {

  for (int32_t i = 0; i < RTL->NumberOfDevices; ++i) {
    DeviceTy &Device = *PM->Devices[RTL->Idx + i];
    Device.PendingGlobalsMtx.lock();
    Device.HasPendingGlobals = true;
    for (__tgt_offload_entry *entry = img->EntriesBegin;
         entry != img->EntriesEnd; ++entry) {
      if (entry->flags & OMP_DECLARE_TARGET_CTOR) {
        DP("Adding ctor " DPxMOD " to the pending list.\n",
           DPxPTR(entry->addr));
        Device.PendingCtorsDtors[desc].PendingCtors.push_back(entry->addr);
      } else if (entry->flags & OMP_DECLARE_TARGET_DTOR) {
        // Dtors are pushed in reverse order so they are executed from end
        // to beginning when unregistering the library!
        DP("Adding dtor " DPxMOD " to the pending list.\n",
           DPxPTR(entry->addr));
        Device.PendingCtorsDtors[desc].PendingDtors.push_front(entry->addr);
      }

      if (entry->flags & OMP_DECLARE_TARGET_LINK) {
        DP("The \"link\" attribute is not yet supported!\n");
      }
    }
    Device.PendingGlobalsMtx.unlock();
  }
}

void RTLsTy::RegisterRequires(int64_t flags) {
  // TODO: add more elaborate check.
  // Minimal check: only set requires flags if previous value
  // is undefined. This ensures that only the first call to this
  // function will set the requires flags. All subsequent calls
  // will be checked for compatibility.
  assert(flags != OMP_REQ_UNDEFINED &&
         "illegal undefined flag for requires directive!");
  if (RequiresFlags == OMP_REQ_UNDEFINED) {
    RequiresFlags = flags;
    return;
  }

  // If multiple compilation units are present enforce
  // consistency across all of them for require clauses:
  //  - reverse_offload
  //  - unified_address
  //  - unified_shared_memory
  if ((RequiresFlags & OMP_REQ_REVERSE_OFFLOAD) !=
      (flags & OMP_REQ_REVERSE_OFFLOAD)) {
    FATAL_MESSAGE0(
        1, "'#pragma omp requires reverse_offload' not used consistently!");
  }
  if ((RequiresFlags & OMP_REQ_UNIFIED_ADDRESS) !=
      (flags & OMP_REQ_UNIFIED_ADDRESS)) {
    FATAL_MESSAGE0(
        1, "'#pragma omp requires unified_address' not used consistently!");
  }
  if ((RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY) !=
      (flags & OMP_REQ_UNIFIED_SHARED_MEMORY)) {
    FATAL_MESSAGE0(
        1,
        "'#pragma omp requires unified_shared_memory' not used consistently!");
  }

  // TODO: insert any other missing checks

  DP("New requires flags %" PRId64 " compatible with existing %" PRId64 "!\n",
     flags, RequiresFlags);
}

void RTLsTy::initRTLonce(RTLInfoTy &R) {
  // If this RTL is not already in use, initialize it.
  if (!R.isUsed && R.NumberOfDevices != 0) {
    // Initialize the device information for the RTL we are about to use.
    const size_t Start = PM->Devices.size();
    PM->Devices.reserve(Start + R.NumberOfDevices);
    for (int32_t device_id = 0; device_id < R.NumberOfDevices; device_id++) {
      PM->Devices.push_back(std::make_unique<DeviceTy>(&R));
      // global device ID
      PM->Devices[Start + device_id]->DeviceID = Start + device_id;
      // RTL local device ID
      PM->Devices[Start + device_id]->RTLDeviceID = device_id;
    }

    // Initialize the index of this RTL and save it in the used RTLs.
    R.Idx = (UsedRTLs.empty())
                ? 0
                : UsedRTLs.back()->Idx + UsedRTLs.back()->NumberOfDevices;
    assert((size_t)R.Idx == Start &&
           "RTL index should equal the number of devices used so far.");
    R.isUsed = true;
    UsedRTLs.push_back(&R);

    DP("RTL " DPxMOD " has index %d!\n", DPxPTR(R.LibraryHandler), R.Idx);
  }
}

void RTLsTy::initAllRTLs() {
  for (auto &R : AllRTLs)
    initRTLonce(R);
}

void RTLsTy::RegisterLib(__tgt_bin_desc *desc) {
  PM->RTLsMtx.lock();
  // Register the images with the RTLs that understand them, if any.
  for (int32_t i = 0; i < desc->NumDeviceImages; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    RTLInfoTy *FoundRTL = nullptr;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image.
    for (auto &R : AllRTLs) {
      if (!R.is_valid_binary(img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL %s!\n",
           DPxPTR(img->ImageStart), R.RTLName.c_str());
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL %s!\n",
         DPxPTR(img->ImageStart), R.RTLName.c_str());

      initRTLonce(R);

      // Initialize (if necessary) translation table for this library.
      PM->TrlTblMtx.lock();
      if (!PM->HostEntriesBeginToTransTable.count(desc->HostEntriesBegin)) {
        PM->HostEntriesBeginRegistrationOrder.push_back(desc->HostEntriesBegin);
        TranslationTable &TransTable =
            (PM->HostEntriesBeginToTransTable)[desc->HostEntriesBegin];
        TransTable.HostTable.EntriesBegin = desc->HostEntriesBegin;
        TransTable.HostTable.EntriesEnd = desc->HostEntriesEnd;
      }

      // Retrieve translation table for this library.
      TranslationTable &TransTable =
          (PM->HostEntriesBeginToTransTable)[desc->HostEntriesBegin];

      DP("Registering image " DPxMOD " with RTL %s!\n", DPxPTR(img->ImageStart),
         R.RTLName.c_str());
      RegisterImageIntoTranslationTable(TransTable, R, img);
      PM->TrlTblMtx.unlock();
      FoundRTL = &R;

      // Load ctors/dtors for static objects
      RegisterGlobalCtorsDtorsForImage(desc, img, FoundRTL);

      // if an RTL was found we are done - proceed to register the next image
      break;
    }

    if (!FoundRTL) {
      DP("No RTL found for image " DPxMOD "!\n", DPxPTR(img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();

  DP("Done registering entries!\n");
}

void RTLsTy::UnregisterLib(__tgt_bin_desc *desc) {
  DP("Unloading target library!\n");

  PM->RTLsMtx.lock();
  // Find which RTL understands each image, if any.
  for (int32_t i = 0; i < desc->NumDeviceImages; ++i) {
    // Obtain the image.
    __tgt_device_image *img = &desc->DeviceImages[i];

    RTLInfoTy *FoundRTL = NULL;

    // Scan the RTLs that have associated images until we find one that supports
    // the current image. We only need to scan RTLs that are already being used.
    for (auto *R : UsedRTLs) {

      assert(R->isUsed && "Expecting used RTLs.");

      if (!R->is_valid_binary(img)) {
        DP("Image " DPxMOD " is NOT compatible with RTL " DPxMOD "!\n",
           DPxPTR(img->ImageStart), DPxPTR(R->LibraryHandler));
        continue;
      }

      DP("Image " DPxMOD " is compatible with RTL " DPxMOD "!\n",
         DPxPTR(img->ImageStart), DPxPTR(R->LibraryHandler));

      FoundRTL = R;

      // Execute dtors for static objects if the device has been used, i.e.
      // if its PendingCtors list has been emptied.
      for (int32_t i = 0; i < FoundRTL->NumberOfDevices; ++i) {
        DeviceTy &Device = *PM->Devices[FoundRTL->Idx + i];
        Device.PendingGlobalsMtx.lock();
        if (Device.PendingCtorsDtors[desc].PendingCtors.empty()) {
          AsyncInfoTy AsyncInfo(Device);
          for (auto &dtor : Device.PendingCtorsDtors[desc].PendingDtors) {
            int rc = target(nullptr, Device, dtor, 0, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, 1, 1, true /*team*/,
                            AsyncInfo);
            if (rc != OFFLOAD_SUCCESS) {
              DP("Running destructor " DPxMOD " failed.\n", DPxPTR(dtor));
            }
          }
          // Remove this library's entry from PendingCtorsDtors
          Device.PendingCtorsDtors.erase(desc);
          // All constructors have been issued, wait for them now.
          if (AsyncInfo.synchronize() != OFFLOAD_SUCCESS)
            DP("Failed synchronizing destructors kernels.\n");
        }
        Device.PendingGlobalsMtx.unlock();
      }

      DP("Unregistered image " DPxMOD " from RTL " DPxMOD "!\n",
         DPxPTR(img->ImageStart), DPxPTR(R->LibraryHandler));

      break;
    }

    // if no RTL was found proceed to unregister the next image
    if (!FoundRTL) {
      DP("No RTLs in use support the image " DPxMOD "!\n",
         DPxPTR(img->ImageStart));
    }
  }
  PM->RTLsMtx.unlock();
  DP("Done unregistering images!\n");

  // Remove entries from PM->HostPtrToTableMap
  PM->TblMapMtx.lock();
  for (__tgt_offload_entry *cur = desc->HostEntriesBegin;
       cur < desc->HostEntriesEnd; ++cur) {
    PM->HostPtrToTableMap.erase(cur->addr);
  }

  // Remove translation table for this descriptor.
  auto TransTable =
      PM->HostEntriesBeginToTransTable.find(desc->HostEntriesBegin);
  if (TransTable != PM->HostEntriesBeginToTransTable.end()) {
    DP("Removing translation table for descriptor " DPxMOD "\n",
       DPxPTR(desc->HostEntriesBegin));
    PM->HostEntriesBeginToTransTable.erase(TransTable);
  } else {
    DP("Translation table for descriptor " DPxMOD " cannot be found, probably "
       "it has been already removed.\n",
       DPxPTR(desc->HostEntriesBegin));
  }

  PM->TblMapMtx.unlock();

  // TODO: Remove RTL and the devices it manages if it's not used anymore?
  // TODO: Write some RTL->unload_image(...) function?

  DP("Done unregistering library!\n");
}
