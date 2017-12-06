//===------ omptarget.cpp - Target independent OpenMP target RTL -- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <omptarget.h>

#include "device.h"
#include "private.h"
#include "rtl.h"

#include <cassert>
#include <vector>

#ifdef OMPTARGET_DEBUG
int DebugLevel = 0;
#endif // OMPTARGET_DEBUG

////////////////////////////////////////////////////////////////////////////////
/// adds a target shared library to the target execution image
EXTERN void __tgt_register_lib(__tgt_bin_desc *desc) {
  RTLs.RegisterLib(desc);
}

////////////////////////////////////////////////////////////////////////////////
/// unloads a target shared library
EXTERN void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  RTLs.UnregisterLib(desc);
}

/// Map global data and execute pending ctors
static int InitLibrary(DeviceTy& Device) {
  /*
   * Map global data
   */
  int32_t device_id = Device.DeviceID;
  int rc = OFFLOAD_SUCCESS;

  Device.PendingGlobalsMtx.lock();
  TrlTblMtx.lock();
  for (HostEntriesBeginToTransTableTy::iterator
      ii = HostEntriesBeginToTransTable.begin();
      ii != HostEntriesBeginToTransTable.end(); ++ii) {
    TranslationTable *TransTable = &ii->second;
    if (TransTable->TargetsTable[device_id] != 0) {
      // Library entries have already been processed
      continue;
    }

    // 1) get image.
    assert(TransTable->TargetsImages.size() > (size_t)device_id &&
           "Not expecting a device ID outside the table's bounds!");
    __tgt_device_image *img = TransTable->TargetsImages[device_id];
    if (!img) {
      DP("No image loaded for device id %d.\n", device_id);
      rc = OFFLOAD_FAIL;
      break;
    }
    // 2) load image into the target table.
    __tgt_target_table *TargetTable =
        TransTable->TargetsTable[device_id] = Device.load_binary(img);
    // Unable to get table for this image: invalidate image and fail.
    if (!TargetTable) {
      DP("Unable to generate entries table for device id %d.\n", device_id);
      TransTable->TargetsImages[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // Verify whether the two table sizes match.
    size_t hsize =
        TransTable->HostTable.EntriesEnd - TransTable->HostTable.EntriesBegin;
    size_t tsize = TargetTable->EntriesEnd - TargetTable->EntriesBegin;

    // Invalid image for these host entries!
    if (hsize != tsize) {
      DP("Host and Target tables mismatch for device id %d [%zx != %zx].\n",
         device_id, hsize, tsize);
      TransTable->TargetsImages[device_id] = 0;
      TransTable->TargetsTable[device_id] = 0;
      rc = OFFLOAD_FAIL;
      break;
    }

    // process global data that needs to be mapped.
    Device.DataMapMtx.lock();
    __tgt_target_table *HostTable = &TransTable->HostTable;
    for (__tgt_offload_entry *CurrDeviceEntry = TargetTable->EntriesBegin,
                             *CurrHostEntry = HostTable->EntriesBegin,
                             *EntryDeviceEnd = TargetTable->EntriesEnd;
         CurrDeviceEntry != EntryDeviceEnd;
         CurrDeviceEntry++, CurrHostEntry++) {
      if (CurrDeviceEntry->size != 0) {
        // has data.
        assert(CurrDeviceEntry->size == CurrHostEntry->size &&
               "data size mismatch");

        // Fortran may use multiple weak declarations for the same symbol,
        // therefore we must allow for multiple weak symbols to be loaded from
        // the fat binary. Treat these mappings as any other "regular" mapping.
        // Add entry to map.
        if (Device.getTgtPtrBegin(CurrHostEntry->addr, CurrHostEntry->size))
          continue;
        DP("Add mapping from host " DPxMOD " to device " DPxMOD " with size %zu"
            "\n", DPxPTR(CurrHostEntry->addr), DPxPTR(CurrDeviceEntry->addr),
            CurrDeviceEntry->size);
        Device.HostDataToTargetMap.push_front(HostDataToTargetTy(
            (uintptr_t)CurrHostEntry->addr /*HstPtrBase*/,
            (uintptr_t)CurrHostEntry->addr /*HstPtrBegin*/,
            (uintptr_t)CurrHostEntry->addr + CurrHostEntry->size /*HstPtrEnd*/,
            (uintptr_t)CurrDeviceEntry->addr /*TgtPtrBegin*/,
            INF_REF_CNT /*RefCount*/));
      }
    }
    Device.DataMapMtx.unlock();
  }
  TrlTblMtx.unlock();

  if (rc != OFFLOAD_SUCCESS) {
    Device.PendingGlobalsMtx.unlock();
    return rc;
  }

  /*
   * Run ctors for static objects
   */
  if (!Device.PendingCtorsDtors.empty()) {
    // Call all ctors for all libraries registered so far
    for (auto &lib : Device.PendingCtorsDtors) {
      if (!lib.second.PendingCtors.empty()) {
        DP("Has pending ctors... call now\n");
        for (auto &entry : lib.second.PendingCtors) {
          void *ctor = entry;
          int rc = target(device_id, ctor, 0, NULL, NULL, NULL,
                          NULL, 1, 1, true /*team*/);
          if (rc != OFFLOAD_SUCCESS) {
            DP("Running ctor " DPxMOD " failed.\n", DPxPTR(ctor));
            Device.PendingGlobalsMtx.unlock();
            return OFFLOAD_FAIL;
          }
        }
        // Clear the list to indicate that this device has been used
        lib.second.PendingCtors.clear();
        DP("Done with pending ctors for lib " DPxMOD "\n", DPxPTR(lib.first));
      }
    }
  }
  Device.HasPendingGlobals = false;
  Device.PendingGlobalsMtx.unlock();

  return OFFLOAD_SUCCESS;
}

// Check whether a device has been initialized, global ctors have been
// executed and global data has been mapped; do so if not already done.
static int CheckDevice(int32_t device_id) {
  // Is device ready?
  if (!device_is_ready(device_id)) {
    DP("Device %d is not ready.\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Get device info.
  DeviceTy &Device = Devices[device_id];

  // Check whether global data has been mapped for this device
  Device.PendingGlobalsMtx.lock();
  bool hasPendingGlobals = Device.HasPendingGlobals;
  Device.PendingGlobalsMtx.unlock();
  if (hasPendingGlobals && InitLibrary(Device) != OFFLOAD_SUCCESS) {
    DP("Failed to init globals on device %d\n", device_id);
    return OFFLOAD_FAIL;
  }

  return OFFLOAD_SUCCESS;
}

// Following datatypes and functions (tgt_oldmap_type, combined_entry_t,
// translate_map, cleanup_map) will be removed once the compiler starts using
// the new map types.

// Old map types
enum tgt_oldmap_type {
  OMP_TGT_OLDMAPTYPE_TO          = 0x001, // copy data from host to device
  OMP_TGT_OLDMAPTYPE_FROM        = 0x002, // copy data from device to host
  OMP_TGT_OLDMAPTYPE_ALWAYS      = 0x004, // copy regardless of the ref. count
  OMP_TGT_OLDMAPTYPE_DELETE      = 0x008, // force unmapping of data
  OMP_TGT_OLDMAPTYPE_MAP_PTR     = 0x010, // map pointer as well as pointee
  OMP_TGT_OLDMAPTYPE_FIRST_MAP   = 0x020, // first occurrence of mapped variable
  OMP_TGT_OLDMAPTYPE_RETURN_PTR  = 0x040, // return TgtBase addr of mapped data
  OMP_TGT_OLDMAPTYPE_PRIVATE_PTR = 0x080, // private variable - not mapped
  OMP_TGT_OLDMAPTYPE_PRIVATE_VAL = 0x100  // copy by value - not mapped
};

// Temporary functions for map translation and cleanup
struct combined_entry_t {
  int num_members; // number of members in combined entry
  void *base_addr; // base address of combined entry
  void *begin_addr; // begin address of combined entry
  void *end_addr; // size of combined entry
};

static void translate_map(int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t &new_arg_num,
    void **&new_args_base, void **&new_args, int64_t *&new_arg_sizes,
    int64_t *&new_arg_types, bool is_target_construct) {
  if (arg_num <= 0) {
    DP("Nothing to translate\n");
    new_arg_num = 0;
    return;
  }

  // array of combined entries
  combined_entry_t *cmb_entries =
      (combined_entry_t *) alloca(arg_num * sizeof(combined_entry_t));
  // number of combined entries
  long num_combined = 0;
  // old entry is MAP_PTR?
  bool *is_ptr_old = (bool *) alloca(arg_num * sizeof(bool));
  // old entry is member of member_of[old] cmb_entry
  int *member_of = (int *) alloca(arg_num * sizeof(int));
  // temporary storage for modifications of the original arg_types
  int64_t *mod_arg_types = (int64_t *) alloca(arg_num  *sizeof(int64_t));

  DP("Translating %d map entries\n", arg_num);
  for (int i = 0; i < arg_num; ++i) {
    member_of[i] = -1;
    is_ptr_old[i] = false;
    mod_arg_types[i] = arg_types[i];
    // Scan previous entries to see whether this entry shares the same base
    for (int j = 0; j < i; ++j) {
      void *new_begin_addr = NULL;
      void *new_end_addr = NULL;

      if (mod_arg_types[i] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
        if (args_base[i] == args[j]) {
          if (!(mod_arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR)) {
            DP("Entry %d has the same base as entry %d's begin address\n", i,
                j);
            new_begin_addr = args_base[i];
            new_end_addr = (char *)args_base[i] + sizeof(void *);
            assert(arg_sizes[j] == sizeof(void *));
            is_ptr_old[j] = true;
          } else {
            DP("Entry %d has the same base as entry %d's begin address, but "
                "%d's base was a MAP_PTR too\n", i, j, j);
            int32_t to_from_always_delete =
                OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM |
                OMP_TGT_OLDMAPTYPE_ALWAYS | OMP_TGT_OLDMAPTYPE_DELETE;
            if (mod_arg_types[j] & to_from_always_delete) {
              DP("Resetting to/from/always/delete flags for entry %d because "
                  "it is only a pointer to pointer\n", j);
              mod_arg_types[j] &= ~to_from_always_delete;
            }
          }
        }
      } else {
        if (!(mod_arg_types[i] & OMP_TGT_OLDMAPTYPE_FIRST_MAP) &&
            args_base[i] == args_base[j]) {
          DP("Entry %d has the same base address as entry %d\n", i, j);
          new_begin_addr = args[i];
          new_end_addr = (char *)args[i] + arg_sizes[i];
        }
      }

      // If we have combined the entry with a previous one
      if (new_begin_addr) {
        int id;
        if(member_of[j] == -1) {
          // We have a new entry
          id = num_combined++;
          DP("Creating new combined entry %d for old entry %d\n", id, j);
          // Initialize new entry
          cmb_entries[id].num_members = 1;
          cmb_entries[id].base_addr = args_base[j];
          if (mod_arg_types[j] & OMP_TGT_OLDMAPTYPE_MAP_PTR) {
            cmb_entries[id].begin_addr = args_base[j];
            cmb_entries[id].end_addr = (char *)args_base[j] + arg_sizes[j];
          } else {
            cmb_entries[id].begin_addr = args[j];
            cmb_entries[id].end_addr = (char *)args[j] + arg_sizes[j];
          }
          member_of[j] = id;
        } else {
          // Reuse existing combined entry
          DP("Reusing existing combined entry %d\n", member_of[j]);
          id = member_of[j];
        }

        // Update combined entry
        DP("Adding entry %d to combined entry %d\n", i, id);
        cmb_entries[id].num_members++;
        // base_addr stays the same
        cmb_entries[id].begin_addr =
            std::min(cmb_entries[id].begin_addr, new_begin_addr);
        cmb_entries[id].end_addr =
            std::max(cmb_entries[id].end_addr, new_end_addr);
        member_of[i] = id;
        break;
      }
    }
  }

  DP("New entries: %ld combined + %d original\n", num_combined, arg_num);
  new_arg_num = arg_num + num_combined;
  new_args_base = (void **) malloc(new_arg_num * sizeof(void *));
  new_args = (void **) malloc(new_arg_num * sizeof(void *));
  new_arg_sizes = (int64_t *) malloc(new_arg_num * sizeof(int64_t));
  new_arg_types = (int64_t *) malloc(new_arg_num * sizeof(int64_t));

  const int64_t alignment = 8;

  int next_id = 0; // next ID
  int next_cid = 0; // next combined ID
  int *combined_to_new_id = (int *) alloca(num_combined * sizeof(int));
  for (int i = 0; i < arg_num; ++i) {
    // It is member_of
    if (member_of[i] == next_cid) {
      int cid = next_cid++; // ID of this combined entry
      int nid = next_id++; // ID of the new (global) entry
      combined_to_new_id[cid] = nid;
      DP("Combined entry %3d will become new entry %3d\n", cid, nid);

      int64_t padding = (int64_t)cmb_entries[cid].begin_addr % alignment;
      if (padding) {
        DP("Using a padding of %" PRId64 " for begin address " DPxMOD "\n",
            padding, DPxPTR(cmb_entries[cid].begin_addr));
        cmb_entries[cid].begin_addr =
            (char *)cmb_entries[cid].begin_addr - padding;
      }

      new_args_base[nid] = cmb_entries[cid].base_addr;
      new_args[nid] = cmb_entries[cid].begin_addr;
      new_arg_sizes[nid] = (int64_t) ((char *)cmb_entries[cid].end_addr -
          (char *)cmb_entries[cid].begin_addr);
      new_arg_types[nid] = OMP_TGT_MAPTYPE_TARGET_PARAM;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", "
          "size %" PRId64 ", type 0x%" PRIx64 "\n", nid,
          DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
          new_arg_types[nid]);
    } else if (member_of[i] != -1) {
      DP("Combined entry %3d has been encountered before, do nothing\n",
          member_of[i]);
    }

    // Now that the combined entry (the one the old entry was a member of) has
    // been inserted into the new arguments list, proceed with the old entry.
    int nid = next_id++;
    DP("Old entry %3d will become new entry %3d\n", i, nid);

    new_args_base[nid] = args_base[i];
    new_args[nid] = args[i];
    new_arg_sizes[nid] = arg_sizes[i];
    int64_t old_type = mod_arg_types[i];

    if (is_ptr_old[i]) {
      // Reset TO and FROM flags
      old_type &= ~(OMP_TGT_OLDMAPTYPE_TO | OMP_TGT_OLDMAPTYPE_FROM);
    }

    if (member_of[i] == -1) {
      if (!is_target_construct)
        old_type &= ~OMP_TGT_MAPTYPE_TARGET_PARAM;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", size %" PRId64
          ", type 0x%" PRIx64 " (old entry %d not MEMBER_OF)\n", nid,
          DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
          new_arg_types[nid], i);
    } else {
      // Old entry is not FIRST_MAP
      old_type &= ~OMP_TGT_OLDMAPTYPE_FIRST_MAP;
      // Add MEMBER_OF
      int new_member_of = combined_to_new_id[member_of[i]];
      old_type |= ((int64_t)new_member_of + 1) << 48;
      new_arg_types[nid] = old_type;
      DP("Entry %3d: base_addr " DPxMOD ", begin_addr " DPxMOD ", size %" PRId64
        ", type 0x%" PRIx64 " (old entry %d MEMBER_OF %d)\n", nid,
        DPxPTR(new_args_base[nid]), DPxPTR(new_args[nid]), new_arg_sizes[nid],
        new_arg_types[nid], i, new_member_of);
    }
  }
}

static void cleanup_map(int32_t new_arg_num, void **new_args_base,
    void **new_args, int64_t *new_arg_sizes, int64_t *new_arg_types,
    int32_t arg_num, void **args_base) {
  if (new_arg_num > 0) {
    int offset = new_arg_num - arg_num;
    for (int32_t i = 0; i < arg_num; ++i) {
      // Restore old base address
      args_base[i] = new_args_base[i+offset];
    }
    free(new_args_base);
    free(new_args);
    free(new_arg_sizes);
    free(new_arg_types);
  }
}

static short member_of(int64_t type) {
  return ((type & OMP_TGT_MAPTYPE_MEMBER_OF) >> 48) - 1;
}

/// Internal function to do the mapping and transfer the data to the device
static int target_data_begin(DeviceTy &Device, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  // process each input.
  int rc = OFFLOAD_SUCCESS;
  for (int32_t i = 0; i < arg_num; ++i) {
    // Ignore private variables and arrays - there is no mapping for them.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    // Address of pointer on the host and device, respectively.
    void *Pointer_HstPtrBegin, *Pointer_TgtPtrBegin;
    bool IsNew, Pointer_IsNew;
    bool IsImplicit = arg_types[i] & OMP_TGT_MAPTYPE_IMPLICIT;
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF);
    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Has a pointer entry: \n");
      // base is address of pointer.
      Pointer_TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBase, HstPtrBase,
          sizeof(void *), Pointer_IsNew, IsImplicit, UpdateRef);
      if (!Pointer_TgtPtrBegin) {
        DP("Call to getOrAllocTgtPtr returned null pointer (device failure or "
            "illegal mapping).\n");
      }
      DP("There are %zu bytes allocated at target address " DPxMOD " - is%s new"
          "\n", sizeof(void *), DPxPTR(Pointer_TgtPtrBegin),
          (Pointer_IsNew ? "" : " not"));
      Pointer_HstPtrBegin = HstPtrBase;
      // modify current entry.
      HstPtrBase = *(void **)HstPtrBase;
      UpdateRef = true; // subsequently update ref count of pointee
    }

    void *TgtPtrBegin = Device.getOrAllocTgtPtr(HstPtrBegin, HstPtrBase,
        arg_sizes[i], IsNew, IsImplicit, UpdateRef);
    if (!TgtPtrBegin && arg_sizes[i]) {
      // If arg_sizes[i]==0, then the argument is a pointer to NULL, so
      // getOrAlloc() returning NULL is not an error.
      DP("Call to getOrAllocTgtPtr returned null pointer (device failure or "
          "illegal mapping).\n");
    }
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
        " - is%s new\n", arg_sizes[i], DPxPTR(TgtPtrBegin),
        (IsNew ? "" : " not"));

    if (arg_types[i] & OMP_TGT_MAPTYPE_RETURN_PARAM) {
      void *ret_ptr;
      if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)
        ret_ptr = Pointer_TgtPtrBegin;
      else {
        bool IsLast; // not used
        ret_ptr = Device.getTgtPtrBegin(HstPtrBegin, 0, IsLast, false);
      }

      DP("Returning device pointer " DPxMOD "\n", DPxPTR(ret_ptr));
      args_base[i] = ret_ptr;
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      bool copy = false;
      if (IsNew || (arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS)) {
        copy = true;
      } else if (arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) {
        // Copy data only if the "parent" struct has RefCount==1.
        short parent_idx = member_of(arg_types[i]);
        long parent_rc = Device.getMapEntryRefCnt(args[parent_idx]);
        assert(parent_rc > 0 && "parent struct not found");
        if (parent_rc == 1) {
          copy = true;
        }
      }

      if (copy) {
        DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
            arg_sizes[i], DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
        int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Copying data to device failed.\n");
          rc = OFFLOAD_FAIL;
        }
      }
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      DP("Update pointer (" DPxMOD ") -> [" DPxMOD "]\n",
          DPxPTR(Pointer_TgtPtrBegin), DPxPTR(TgtPtrBegin));
      uint64_t Delta = (uint64_t)HstPtrBegin - (uint64_t)HstPtrBase;
      void *TgtPtrBase = (void *)((uint64_t)TgtPtrBegin - Delta);
      int rt = Device.data_submit(Pointer_TgtPtrBegin, &TgtPtrBase,
          sizeof(void *));
      if (rt != OFFLOAD_SUCCESS) {
        DP("Copying data to device failed.\n");
        rc = OFFLOAD_FAIL;
      }
      // create shadow pointers for this entry
      Device.ShadowMtx.lock();
      Device.ShadowPtrMap[Pointer_HstPtrBegin] = {HstPtrBase,
          Pointer_TgtPtrBegin, TgtPtrBase};
      Device.ShadowMtx.unlock();
    }
  }

  return rc;
}

EXTERN void __tgt_target_data_begin_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_begin(device_id, arg_num, args_base, args, arg_sizes,
                          arg_types);
}

/// creates host-to-target data mapping, stores it in the
/// libomptarget.so internal structure (an entry in a stack of data maps)
/// and passes the data to the device.
EXTERN void __tgt_target_data_begin(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data begin region for device %ld with %d mappings\n", device_id,
     arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
    DP("Use default device id %ld\n", device_id);
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_begin(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_begin(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

/// Internal function to undo the mapping and retrieve the data from the device.
static int target_data_end(DeviceTy &Device, int32_t arg_num, void **args_base,
    void **args, int64_t *arg_sizes, int64_t *arg_types) {
  int rc = OFFLOAD_SUCCESS;
  // process each input.
  for (int32_t i = arg_num - 1; i >= 0; --i) {
    // Ignore private variables and arrays - there is no mapping for them.
    // Also, ignore the use_device_ptr directive, it has no effect here.
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    bool IsLast;
    bool UpdateRef = !(arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ);
    bool ForceDelete = arg_types[i] & OMP_TGT_MAPTYPE_DELETE;

    // If PTR_AND_OBJ, HstPtrBegin is address of pointee
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast,
        UpdateRef);
    DP("There are %" PRId64 " bytes allocated at target address " DPxMOD
        " - is%s last\n", arg_sizes[i], DPxPTR(TgtPtrBegin),
        (IsLast ? "" : " not"));

    bool DelEntry = IsLast || ForceDelete;

    if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
        !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
      DelEntry = false; // protect parent struct from being deallocated
    }

    if ((arg_types[i] & OMP_TGT_MAPTYPE_FROM) || DelEntry) {
      // Move data back to the host
      if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
        bool Always = arg_types[i] & OMP_TGT_MAPTYPE_ALWAYS;
        bool CopyMember = false;
        if ((arg_types[i] & OMP_TGT_MAPTYPE_MEMBER_OF) &&
            !(arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ)) {
          // Copy data only if the "parent" struct has RefCount==1.
          short parent_idx = member_of(arg_types[i]);
          long parent_rc = Device.getMapEntryRefCnt(args[parent_idx]);
          assert(parent_rc > 0 && "parent struct not found");
          if (parent_rc == 1) {
            CopyMember = true;
          }
        }

        if (DelEntry || Always || CopyMember) {
          DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
              arg_sizes[i], DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
          int rt = Device.data_retrieve(HstPtrBegin, TgtPtrBegin, arg_sizes[i]);
          if (rt != OFFLOAD_SUCCESS) {
            DP("Copying data from device failed.\n");
            rc = OFFLOAD_FAIL;
          }
        }
      }

      // If we copied back to the host a struct/array containing pointers, we
      // need to restore the original host pointer values from their shadow
      // copies. If the struct is going to be deallocated, remove any remaining
      // shadow pointer entries for this struct.
      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + arg_sizes[i];
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;

        // An STL map is sorted on its keys; use this property
        // to quickly determine when to break out of the loop.
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;

        // If we copied the struct to the host, we need to restore the pointer.
        if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
          DP("Restoring original host pointer value " DPxMOD " for host "
              "pointer " DPxMOD "\n", DPxPTR(it->second.HstPtrVal),
              DPxPTR(ShadowHstPtrAddr));
          *ShadowHstPtrAddr = it->second.HstPtrVal;
        }
        // If the struct is to be deallocated, remove the shadow entry.
        if (DelEntry) {
          DP("Removing shadow pointer " DPxMOD "\n", DPxPTR(ShadowHstPtrAddr));
          Device.ShadowPtrMap.erase(it);
        }
      }
      Device.ShadowMtx.unlock();

      // Deallocate map
      if (DelEntry) {
        int rt = Device.deallocTgtPtr(HstPtrBegin, arg_sizes[i], ForceDelete);
        if (rt != OFFLOAD_SUCCESS) {
          DP("Deallocating data from device failed.\n");
          rc = OFFLOAD_FAIL;
        }
      }
    }
  }

  return rc;
}

/// passes data from the target, releases target memory and destroys
/// the host-target mapping (top entry from the stack of data maps)
/// created by the last __tgt_target_data_begin.
EXTERN void __tgt_target_data_end(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data end region with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  RTLsMtx.lock();
  size_t Devices_size = Devices.size();
  RTLsMtx.unlock();
  if (Devices_size <= (size_t)device_id) {
    DP("Device ID  %ld does not have a matching RTL.\n", device_id);
    return;
  }

  DeviceTy &Device = Devices[device_id];
  if (!Device.IsInit) {
    DP("uninit device: ignore");
    return;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, false);

  //target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
  target_data_end(Device, new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);
}

EXTERN void __tgt_target_data_end_nowait(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_end(device_id, arg_num, args_base, args, arg_sizes,
                        arg_types);
}

/// passes data to/from the target.
EXTERN void __tgt_target_data_update(int64_t device_id, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering data update with %d mappings\n", arg_num);

  // No devices available?
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DeviceTy& Device = Devices[device_id];

  // process each input.
  for (int32_t i = 0; i < arg_num; ++i) {
    if ((arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) ||
        (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE))
      continue;

    void *HstPtrBegin = args[i];
    int64_t MapSize = arg_sizes[i];
    bool IsLast;
    void *TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, MapSize, IsLast,
        false);

    if (arg_types[i] & OMP_TGT_MAPTYPE_FROM) {
      DP("Moving %" PRId64 " bytes (tgt:" DPxMOD ") -> (hst:" DPxMOD ")\n",
          arg_sizes[i], DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBegin));
      Device.data_retrieve(HstPtrBegin, TgtPtrBegin, MapSize);

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original host pointer value " DPxMOD " for host pointer "
            DPxMOD "\n", DPxPTR(it->second.HstPtrVal),
            DPxPTR(ShadowHstPtrAddr));
        *ShadowHstPtrAddr = it->second.HstPtrVal;
      }
      Device.ShadowMtx.unlock();
    }

    if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
      DP("Moving %" PRId64 " bytes (hst:" DPxMOD ") -> (tgt:" DPxMOD ")\n",
          arg_sizes[i], DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBegin));
      Device.data_submit(TgtPtrBegin, HstPtrBegin, MapSize);

      uintptr_t lb = (uintptr_t) HstPtrBegin;
      uintptr_t ub = (uintptr_t) HstPtrBegin + MapSize;
      Device.ShadowMtx.lock();
      for (ShadowPtrListTy::iterator it = Device.ShadowPtrMap.begin();
          it != Device.ShadowPtrMap.end(); ++it) {
        void **ShadowHstPtrAddr = (void**) it->first;
        if ((uintptr_t) ShadowHstPtrAddr < lb)
          continue;
        if ((uintptr_t) ShadowHstPtrAddr >= ub)
          break;
        DP("Restoring original target pointer value " DPxMOD " for target "
            "pointer " DPxMOD "\n", DPxPTR(it->second.TgtPtrVal),
            DPxPTR(it->second.TgtPtrAddr));
        Device.data_submit(it->second.TgtPtrAddr,
            &it->second.TgtPtrVal, sizeof(void *));
      }
      Device.ShadowMtx.unlock();
    }
  }
}

EXTERN void __tgt_target_data_update_nowait(
    int64_t device_id, int32_t arg_num, void **args_base, void **args,
    int64_t *arg_sizes, int64_t *arg_types, int32_t depNum, void *depList,
    int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  __tgt_target_data_update(device_id, arg_num, args_base, args, arg_sizes,
                           arg_types);
}

/// performs the same actions as data_begin in case arg_num is
/// non-zero and initiates run of the offloaded region on the target platform;
/// if arg_num is non-zero after the region execution is done it also
/// performs the same action as data_update and data_end above. This function
/// returns 0 if it was able to transfer the execution to a target and an
/// integer different from zero otherwise.
int target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types,
    int32_t team_num, int32_t thread_limit, int IsTeamConstruct) {
  DeviceTy &Device = Devices[device_id];

  // Find the table information in the map or look it up in the translation
  // tables.
  TableMap *TM = 0;
  TblMapMtx.lock();
  HostPtrToTableMapTy::iterator TableMapIt = HostPtrToTableMap.find(host_ptr);
  if (TableMapIt == HostPtrToTableMap.end()) {
    // We don't have a map. So search all the registered libraries.
    TrlTblMtx.lock();
    for (HostEntriesBeginToTransTableTy::iterator
             ii = HostEntriesBeginToTransTable.begin(),
             ie = HostEntriesBeginToTransTable.end();
         !TM && ii != ie; ++ii) {
      // get the translation table (which contains all the good info).
      TranslationTable *TransTable = &ii->second;
      // iterate over all the host table entries to see if we can locate the
      // host_ptr.
      __tgt_offload_entry *begin = TransTable->HostTable.EntriesBegin;
      __tgt_offload_entry *end = TransTable->HostTable.EntriesEnd;
      __tgt_offload_entry *cur = begin;
      for (uint32_t i = 0; cur < end; ++cur, ++i) {
        if (cur->addr != host_ptr)
          continue;
        // we got a match, now fill the HostPtrToTableMap so that we
        // may avoid this search next time.
        TM = &HostPtrToTableMap[host_ptr];
        TM->Table = TransTable;
        TM->Index = i;
        break;
      }
    }
    TrlTblMtx.unlock();
  } else {
    TM = &TableMapIt->second;
  }
  TblMapMtx.unlock();

  // No map for this host pointer found!
  if (!TM) {
    DP("Host ptr " DPxMOD " does not have a matching target pointer.\n",
       DPxPTR(host_ptr));
    return OFFLOAD_FAIL;
  }

  // get target table.
  TrlTblMtx.lock();
  assert(TM->Table->TargetsTable.size() > (size_t)device_id &&
         "Not expecting a device ID outside the table's bounds!");
  __tgt_target_table *TargetTable = TM->Table->TargetsTable[device_id];
  TrlTblMtx.unlock();
  assert(TargetTable && "Global data has not been mapped\n");

  // Move data to device.
  int rc = target_data_begin(Device, arg_num, args_base, args, arg_sizes,
      arg_types);

  if (rc != OFFLOAD_SUCCESS) {
    DP("Call to target_data_begin failed, skipping target execution.\n");
    // Call target_data_end to dealloc whatever target_data_begin allocated
    // and return OFFLOAD_FAIL.
    target_data_end(Device, arg_num, args_base, args, arg_sizes, arg_types);
    return OFFLOAD_FAIL;
  }

  std::vector<void *> tgt_args;
  std::vector<ptrdiff_t> tgt_offsets;

  // List of (first-)private arrays allocated for this target region
  std::vector<void *> fpArrays;

  for (int32_t i = 0; i < arg_num; ++i) {
    if (!(arg_types[i] & OMP_TGT_MAPTYPE_TARGET_PARAM)) {
      // This is not a target parameter, do not push it into tgt_args.
      continue;
    }
    void *HstPtrBegin = args[i];
    void *HstPtrBase = args_base[i];
    void *TgtPtrBegin;
    ptrdiff_t TgtBaseOffset;
    bool IsLast; // unused.
    if (arg_types[i] & OMP_TGT_MAPTYPE_LITERAL) {
      DP("Forwarding first-private value " DPxMOD " to the target construct\n",
          DPxPTR(HstPtrBase));
      TgtPtrBegin = HstPtrBase;
      TgtBaseOffset = 0;
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PRIVATE) {
      // Allocate memory for (first-)private array
      TgtPtrBegin = Device.RTL->data_alloc(Device.RTLDeviceID,
          arg_sizes[i], HstPtrBegin);
      if (!TgtPtrBegin) {
        DP ("Data allocation for %sprivate array " DPxMOD " failed\n",
            (arg_types[i] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
            DPxPTR(HstPtrBegin));
        rc = OFFLOAD_FAIL;
        break;
      } else {
        fpArrays.push_back(TgtPtrBegin);
        TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
        void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
        DP("Allocated %" PRId64 " bytes of target memory at " DPxMOD " for "
            "%sprivate array " DPxMOD " - pushing target argument " DPxMOD "\n",
            arg_sizes[i], DPxPTR(TgtPtrBegin),
            (arg_types[i] & OMP_TGT_MAPTYPE_TO ? "first-" : ""),
            DPxPTR(HstPtrBegin), DPxPTR(TgtPtrBase));
#endif
        // If first-private, copy data from host
        if (arg_types[i] & OMP_TGT_MAPTYPE_TO) {
          int rt = Device.data_submit(TgtPtrBegin, HstPtrBegin, arg_sizes[i]);
          if (rt != OFFLOAD_SUCCESS) {
            DP ("Copying data to device failed.\n");
            rc = OFFLOAD_FAIL;
            break;
          }
        }
      }
    } else if (arg_types[i] & OMP_TGT_MAPTYPE_PTR_AND_OBJ) {
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBase, sizeof(void *), IsLast,
          false);
      TgtBaseOffset = 0; // no offset for ptrs.
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD " to "
         "object " DPxMOD "\n", DPxPTR(TgtPtrBegin), DPxPTR(HstPtrBase),
         DPxPTR(HstPtrBase));
    } else {
      TgtPtrBegin = Device.getTgtPtrBegin(HstPtrBegin, arg_sizes[i], IsLast,
          false);
      TgtBaseOffset = (intptr_t)HstPtrBase - (intptr_t)HstPtrBegin;
#ifdef OMPTARGET_DEBUG
      void *TgtPtrBase = (void *)((intptr_t)TgtPtrBegin + TgtBaseOffset);
      DP("Obtained target argument " DPxMOD " from host pointer " DPxMOD "\n",
          DPxPTR(TgtPtrBase), DPxPTR(HstPtrBegin));
#endif
    }
    tgt_args.push_back(TgtPtrBegin);
    tgt_offsets.push_back(TgtBaseOffset);
  }

  assert(tgt_args.size() == tgt_offsets.size() &&
      "Size mismatch in arguments and offsets");

  // Pop loop trip count
  uint64_t ltc = Device.loopTripCnt;
  Device.loopTripCnt = 0;

  // Launch device execution.
  if (rc == OFFLOAD_SUCCESS) {
    DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
        TargetTable->EntriesBegin[TM->Index].name,
        DPxPTR(TargetTable->EntriesBegin[TM->Index].addr), TM->Index);
    if (IsTeamConstruct) {
      rc = Device.run_team_region(TargetTable->EntriesBegin[TM->Index].addr,
          &tgt_args[0], &tgt_offsets[0], tgt_args.size(), team_num,
          thread_limit, ltc);
    } else {
      rc = Device.run_region(TargetTable->EntriesBegin[TM->Index].addr,
          &tgt_args[0], &tgt_offsets[0], tgt_args.size());
    }
  } else {
    DP("Errors occurred while obtaining target arguments, skipping kernel "
        "execution\n");
  }

  // Deallocate (first-)private arrays
  for (auto it : fpArrays) {
    int rt = Device.RTL->data_delete(Device.RTLDeviceID, it);
    if (rt != OFFLOAD_SUCCESS) {
      DP("Deallocation of (first-)private arrays failed.\n");
      rc = OFFLOAD_FAIL;
    }
  }

  // Move data from device.
  int rt = target_data_end(Device, arg_num, args_base, args, arg_sizes,
      arg_types);

  if (rt != OFFLOAD_SUCCESS) {
    DP("Call to target_data_end failed.\n");
    rc = OFFLOAD_FAIL;
  }

  return rc;
}

EXTERN int __tgt_target(int64_t device_id, void *host_ptr, int32_t arg_num,
    void **args_base, void **args, int64_t *arg_sizes, int64_t *arg_types) {
  DP("Entering target region with entry point " DPxMOD " and device Id %ld\n",
     DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //    arg_types, 0, 0, false /*team*/, false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, 0, 0, false /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t depNum, void *depList, int32_t noAliasDepNum,
    void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
                      arg_types);
}

EXTERN int __tgt_target_teams(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit) {
  DP("Entering target region with entry point " DPxMOD " and device Id %ld\n",
     DPxPTR(host_ptr), device_id);

  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return OFFLOAD_FAIL;
  }

  // Translate maps
  int32_t new_arg_num;
  void **new_args_base;
  void **new_args;
  int64_t *new_arg_sizes;
  int64_t *new_arg_types;
  translate_map(arg_num, args_base, args, arg_sizes, arg_types, new_arg_num,
      new_args_base, new_args, new_arg_sizes, new_arg_types, true);

  //return target(device_id, host_ptr, arg_num, args_base, args, arg_sizes,
  //              arg_types, team_num, thread_limit, true /*team*/,
  //              false /*recursive*/);
  int rc = target(device_id, host_ptr, new_arg_num, new_args_base, new_args,
      new_arg_sizes, new_arg_types, team_num, thread_limit, true /*team*/);

  // Cleanup translation memory
  cleanup_map(new_arg_num, new_args_base, new_args, new_arg_sizes,
      new_arg_types, arg_num, args_base);

  return rc;
}

EXTERN int __tgt_target_teams_nowait(int64_t device_id, void *host_ptr,
    int32_t arg_num, void **args_base, void **args, int64_t *arg_sizes,
    int64_t *arg_types, int32_t team_num, int32_t thread_limit, int32_t depNum,
    void *depList, int32_t noAliasDepNum, void *noAliasDepList) {
  if (depNum + noAliasDepNum > 0)
    __kmpc_omp_taskwait(NULL, 0);

  return __tgt_target_teams(device_id, host_ptr, arg_num, args_base, args,
                            arg_sizes, arg_types, team_num, thread_limit);
}


// The trip count mechanism will be revised - this scheme is not thread-safe.
EXTERN void __kmpc_push_target_tripcount(int64_t device_id,
    uint64_t loop_tripcount) {
  if (device_id == OFFLOAD_DEVICE_DEFAULT) {
    device_id = omp_get_default_device();
  }

  if (CheckDevice(device_id) != OFFLOAD_SUCCESS) {
    DP("Failed to get device %ld ready\n", device_id);
    return;
  }

  DP("__kmpc_push_target_tripcount(%ld, %" PRIu64 ")\n", device_id,
      loop_tripcount);
  Devices[device_id].loopTripCnt = loop_tripcount;
}

