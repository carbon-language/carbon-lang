/* -*- mode: C++; c-basic-offset: 4; tab-width: 4 vi:set tabstop=4 expandtab: -*/
//===-- RemoteProcInfo.hpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//  This file defines the primary object created when unw_create_addr_space()
//  is called.  This object tracks the list of known images in memory 
//  (dylibs, bundles, etc), it maintains a link to a RemoteRegisterMap for this
//  architecture, it caches the remote process memory in a local store and all
//  read/writes are filtered through its accessors which will use the memory
//  caches.  It maintains a logging level set by the driver program and puts
//  timing/debug messages out on a FILE* provided to it.

//  RemoteProcInfo is not specific to any particular unwind so it does not
//  maintain an "arg" argument (an opaque pointer that the driver program uses
//  to track the process/thread being unwound).  

#ifndef __REMOTE_PROC_INFO_HPP__
#define __REMOTE_PROC_INFO_HPP__

#if defined (SUPPORT_REMOTE_UNWINDING)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <sys/time.h>
#include <mach-o/loader.h>
#include <mach-o/getsect.h>
#include <mach/i386/thread_status.h>
#include <Availability.h>

#include <map>
#include <vector>
#include <algorithm>

#include "FileAbstraction.hpp"
#include "libunwind.h"
#include "InternalMacros.h"
#include "dwarf2.h"
#include "RemoteUnwindProfile.h"
#include "Registers.hpp"
#include "RemoteRegisterMap.hpp"

namespace lldb_private
{
class RemoteProcInfo;

///
/// unw_addr_space_remote is the concrete instance that a unw_addr_space_t points to when examining
/// a remote process.
///
struct unw_addr_space_remote
{
   enum unw_as_type type;      // should always be UNW_REMOTE
   RemoteProcInfo* ras;
};

class RemoteMemoryBlob
{
public:
  typedef void (*free_callback_with_arg)(void *, void*);
  typedef void (*free_callback)(void *);

  /* This object is constructed with a callback to free the memory;
     that callback takes a pointer to the memory region and optionally
     takes an additional argument -- the "void* arg" passed around for
     remote unwinds, in case the driver program allocated this e.g. with
     mach_vm_read, and needs the token to vm_deallocate it.  */

  RemoteMemoryBlob (uint8_t *buf, free_callback_with_arg to_free, 
                 uint64_t startaddr, uint64_t len, uint64_t mh, void *arg) : 
                    fBuf(buf), fToFree(NULL), fToFreeWithArg(to_free),
                    fStartAddr(startaddr), fLen(len), fMachHeader(mh),
                    fArg(arg) { }
  RemoteMemoryBlob (uint8_t *buf, free_callback to_free, uint64_t startaddr, 
                 uint64_t len, uint64_t mh, void *arg) : 
                    fBuf(buf), fToFree(to_free), fToFreeWithArg(NULL), 
                    fStartAddr(startaddr), fLen(len), fMachHeader(mh),
                    fArg(NULL) { }

  // the following is to create a dummy RMB object for lower_bound's use in
  // searching.
  RemoteMemoryBlob (uint64_t startaddr) : fBuf(NULL), fToFree(NULL),
                    fToFreeWithArg(NULL), fStartAddr(startaddr), fLen(0),
                    fMachHeader(-1), fArg(NULL) { }
  ~RemoteMemoryBlob () {
    if (fToFreeWithArg)
      fToFreeWithArg(fBuf, fArg);
    else if (fToFree)
      fToFree(fBuf);
  }
  bool contains_addr (uint64_t addr) {
    if (fStartAddr <= addr &&  addr < fStartAddr + fLen)
      return true;
    else
      return false;
  }
  uint8_t *get_blob_range (uint64_t remote_process_addr, int len) {
    if (this->contains_addr (remote_process_addr) == false)
      return NULL;
    if (this->contains_addr (remote_process_addr + len) == false)
      return NULL;
    return fBuf + (remote_process_addr - fStartAddr);
  }
  uint64_t getMh ()       const { return fMachHeader; }
  uint64_t getStartAddr() const { return fStartAddr; }
  uint64_t getLength()    const { return fLen; }
private:
  uint8_t *fBuf;
  free_callback fToFree;
  free_callback_with_arg fToFreeWithArg;
  uint64_t fStartAddr;
  uint64_t fLen;
  uint64_t fMachHeader;
  void    *fArg;
};

inline bool operator<(const RemoteMemoryBlob &b1, const RemoteMemoryBlob &b2) {
    if (b1.getStartAddr() < b2.getStartAddr())
        return true;
    else
        return false;
}

// One of these for each image in memory (executable, dylib, bundle, etc)

struct RemoteImageEntry
{
  RemoteImageEntry () : mach_header(0), text_start(0), text_end(0), eh_frame_start(0), eh_frame_len(0), compact_unwind_info_start(0), compact_unwind_info_len(0) { }
  ~RemoteImageEntry () {
      std::map<uint64_t, RemoteUnwindProfile *>::iterator i;
      for (i = profiles.begin(); i != profiles.end(); ++i)
        delete i->second;
  }
  uint64_t mach_header;
  uint64_t text_start;
  uint64_t text_end;
  uint64_t eh_frame_start;
  uint64_t eh_frame_len;
  uint64_t compact_unwind_info_start;
  uint64_t compact_unwind_info_len;

  // unwind profiles created for thsi binary image so far,
  // key is the start address of the profile.
  std::map<uint64_t, RemoteUnwindProfile *> profiles;

  // a list of function address bounds for this binary image -
  // end addresses should be accurate and not inferred from potentially
  // incomplete start-address data (e.g. nlist records).
  std::vector<FuncBounds> func_bounds;
};

class RemoteImages
{
public:
    RemoteImages (unw_targettype_t targarch) : fTargetArch(targarch) { }
    ~RemoteImages ();
    void removeAllImageProfiles();
    void removeOneImageProfiles(uint64_t mh);
    RemoteImageEntry *remoteEntryForTextAddr (uint64_t pc);
    bool addFuncBounds (uint64_t mh, std::vector<FuncBounds> &startAddrs);
    bool haveFuncBounds (uint64_t mh);
    bool findFuncBounds (uint32_t pc, uint32_t &startAddr, uint32_t &endAddr);
    bool findFuncBounds (uint64_t pc, uint64_t &startAddr, uint64_t &endAddr);
    void addImage (uint64_t mh, uint64_t text_start, uint64_t text_end, uint64_t eh_frame, uint64_t eh_frame_len, uint64_t compact_unwind_start, uint64_t compact_unwind_len);
    bool addProfile (RemoteProcInfo* procinfo, unw_accessors_t *acc, unw_addr_space_t as, uint64_t start, uint64_t end, void *arg);
    RemoteUnwindProfile* findProfileByTextAddr (uint64_t pc);
    bool addMemBlob (RemoteMemoryBlob *blob);
    uint8_t *getMemBlobMemory (uint64_t addr, int len);
private:
    RemoteImages();
    std::map<uint64_t, RemoteImageEntry> fImages;
    std::vector<RemoteMemoryBlob *> fMemBlobs;
    unw_targettype_t fTargetArch;
};

RemoteImages::~RemoteImages () {
    std::map<uint64_t, std::vector<RemoteMemoryBlob *> >::iterator i;
    std::vector<RemoteMemoryBlob *>::iterator j;
    for (j = fMemBlobs.begin(); j != fMemBlobs.end(); ++j) {
        delete *j;
    }
    fMemBlobs.erase(fMemBlobs.begin(), fMemBlobs.end());
}

void RemoteImages::removeAllImageProfiles() {
    fImages.erase(fImages.begin(), fImages.end());
    std::vector<RemoteMemoryBlob *>::iterator j;
    for (j = fMemBlobs.begin(); j != fMemBlobs.end(); ++j)
        delete *j;
    fMemBlobs.erase(fMemBlobs.begin(), fMemBlobs.end());
}

void RemoteImages::removeOneImageProfiles(uint64_t mh) {
    std::map<uint64_t, RemoteImageEntry>::iterator i;
    i = fImages.find(mh);
    if (i != fImages.end())
        fImages.erase(i);

    std::vector<RemoteMemoryBlob *>::iterator j;
    for (j = fMemBlobs.begin(); j != fMemBlobs.end(); ++j) {
        if ((*j)->getMh() == mh) {
            delete *j;
            break; 
        }
    }
    if (j != fMemBlobs.end())
        fMemBlobs.erase(j);
}

RemoteImageEntry *RemoteImages::remoteEntryForTextAddr (uint64_t pc) {
    std::map<uint64_t, RemoteImageEntry>::iterator i = fImages.lower_bound (pc);
    if (i == fImages.begin() && i == fImages.end())
        return NULL;
    if (i == fImages.end()) {
        --i;
    } else {
        if (i != fImages.begin() && i->first != pc)
          --i;
    }
    if (i->second.text_start <= pc && i->second.text_end > pc)
      {
        return &(i->second);
      }
    else
      {
        return NULL;
      }
}

bool RemoteImages::addFuncBounds (uint64_t mh, std::vector<FuncBounds> &startAddrs) {
    RemoteImageEntry *img = NULL;
    std::map<uint64_t, RemoteImageEntry>::iterator i = fImages.find (mh);
    if (i == fImages.end())
        return false;
    img = &i->second;
    img->func_bounds = startAddrs;
    std::sort(img->func_bounds.begin(), img->func_bounds.end());
    return true;
}

bool RemoteImages::haveFuncBounds (uint64_t mh) {
    RemoteImageEntry *img = NULL;
    std::map<uint64_t, RemoteImageEntry>::iterator i = fImages.find (mh);
    if (i == fImages.end())
        return false;
    img = &i->second;
    if (img->func_bounds.size() > 0)
        return true;
    return false;
}

bool RemoteImages::findFuncBounds (uint64_t pc, uint64_t &startAddr, uint64_t &endAddr) {
    RemoteImageEntry *img = NULL;
    startAddr = endAddr = 0;
    std::map<uint64_t, RemoteImageEntry>::iterator i = fImages.lower_bound (pc);
    if (i == fImages.begin() && i == fImages.end())
        return false;
    if (i == fImages.end()) {
        --i;
    } else {
        if (i != fImages.begin() && i->first != pc)
            --i;
    }
    if (i->second.text_start <= pc && i->second.text_end > pc)
      {
        img = &i->second;
      }
    else
      return false;
    std::vector<FuncBounds>::iterator j;
    j = std::lower_bound(img->func_bounds.begin(), img->func_bounds.end(), FuncBounds (pc, pc));
    if (j == img->func_bounds.begin() && j == img->func_bounds.end())
        return false;
    if (j == img->func_bounds.end()) {
        --j;
    } else {
        if (j != img->func_bounds.begin() && j->fStart != pc)
            --j;
    }
    if (j->fStart <= pc && j->fEnd > pc) {
        startAddr = j->fStart;
        endAddr = j->fEnd;
        return true;
    }
    return false;
}

// Add 32-bit version of findFuncBounds so we can avoid templatizing all of these functions
// just to handle 64 and 32 bit unwinds.

bool RemoteImages::findFuncBounds (uint32_t pc, uint32_t &startAddr, uint32_t &endAddr) {
    uint64_t big_startAddr = startAddr;
    uint64_t big_endAddr = endAddr;
    bool ret;
    ret = findFuncBounds (pc, big_startAddr, big_endAddr);
    startAddr = (uint32_t) big_startAddr & 0xffffffff;
    endAddr = (uint32_t) big_endAddr & 0xffffffff;
    return ret;
}

// Make sure we don't cache the same memory range more than once
// I'm not checking the length of the blobs to check for overlap -
// as this is used today, the only duplication will be with the same
// start address.

bool 
RemoteImages::addMemBlob (RemoteMemoryBlob *blob) { 

    if (fMemBlobs.empty())
    {
        fMemBlobs.push_back(blob);
    }
    else
    {
        std::vector<RemoteMemoryBlob *>::iterator pos;

        pos = std::lower_bound (fMemBlobs.begin(), fMemBlobs.end(), blob);

        if (pos != fMemBlobs.end() && (*pos)->getStartAddr() == blob->getStartAddr())
            return false;

        fMemBlobs.insert (pos, blob);
    }
    return true;
}

uint8_t *RemoteImages::getMemBlobMemory (uint64_t addr, int len) {
    uint8_t *res = NULL;
    std::vector<RemoteMemoryBlob *>::iterator j;
    RemoteMemoryBlob *searchobj = new RemoteMemoryBlob(addr);
    j = std::lower_bound (fMemBlobs.begin(), fMemBlobs.end(), searchobj);
    delete searchobj;
    if (j == fMemBlobs.end() && j == fMemBlobs.begin())
        return NULL;
    if (j == fMemBlobs.end()) {
        --j;
    } else {
        if (j != fMemBlobs.begin() && (*j)->getStartAddr() != addr)
            --j;
    }
    res = (*j)->get_blob_range (addr, len);
    if (res != NULL)
        return res;
    for (j = fMemBlobs.begin(); j != fMemBlobs.end(); ++j) {
        res = (*j)->get_blob_range (addr, len);
        if (res != NULL)
            break;
    }
    return res;
}

void RemoteImages::addImage (uint64_t mh, uint64_t text_start, 
                             uint64_t text_end, uint64_t eh_frame, 
                             uint64_t eh_frame_len, 
                             uint64_t compact_unwind_start,
                             uint64_t compact_unwind_len) {
    struct RemoteImageEntry img;
    img.mach_header = mh;
    img.text_start = text_start;
    img.text_end = text_end;
    img.eh_frame_start = eh_frame;
    img.eh_frame_len = eh_frame_len;
    img.compact_unwind_info_start = compact_unwind_start;
    img.compact_unwind_info_len = compact_unwind_len;
    fImages[mh] = img;
}

// The binary image for this start/end address must already be present
bool RemoteImages::addProfile (RemoteProcInfo* procinfo, unw_accessors_t *acc, unw_addr_space_t as, uint64_t start, uint64_t end, void *arg) {
    RemoteImageEntry *img = NULL;
    std::map<uint64_t, RemoteImageEntry>::iterator i = fImages.lower_bound (start);
    if (i == fImages.begin() && i == fImages.end())
        return false;
    if (i == fImages.end()) {
        --i;
    } else {
        if (i != fImages.begin() && i->first != start) {
            --i;
        }
    }
    if (i->second.text_start <= start && i->second.text_end > start)
      {
        img = &i->second;
      }
    else
      return false;
    RemoteUnwindProfile* profile = new RemoteUnwindProfile;
    if (AssemblyParse (procinfo, acc, as, start, end, *profile, arg)) {
        img->profiles[start] = profile;
        return true;
    }
    return false;
}

RemoteUnwindProfile* RemoteImages::findProfileByTextAddr (uint64_t pc) {
    RemoteImageEntry *img = NULL;
    std::map<uint64_t, RemoteImageEntry>::iterator i = fImages.lower_bound (pc);
    if (i == fImages.begin() && i == fImages.end())
        return NULL;
    if (i == fImages.end()) {
        --i;
    } else {
        if (i != fImages.begin() && i->first != pc)
          --i;
    }
    if (i->second.text_start <= pc && i->second.text_end > pc)
      {
        img = &i->second;
      }
    else
      return NULL;
    std::map<uint64_t, RemoteUnwindProfile *>::iterator j;
    j = img->profiles.lower_bound (pc);
    if (j == img->profiles.begin() && j == img->profiles.end())
        return NULL;
    if (j == img->profiles.end()) {
        --j;
    } else {
        if (j != img->profiles.begin() && j->first != pc)
          --j;
    }
    if (j->second->fStart <= pc && j->second->fEnd > pc)
      {
        return j->second;
      }
    return NULL;
}

///
/// RemoteProcInfo is used as a template parameter to UnwindCursor when 
/// unwinding a thread that has a custom set of accessors.  It calls the 
/// custom accessors for all data.
///
class RemoteProcInfo
{
public:

// libunwind documentation specifies that unw_create_addr_space defaults to 
//  UNW_CACHE_NONE but that's going to work very poorly for us so we're 
// defaulting to UNW_CACHE_GLOBAL.

    RemoteProcInfo(unw_accessors_t* accessors, unw_targettype_t targarch) : 
                    fAccessors(*accessors), fCachingPolicy(UNW_CACHE_GLOBAL), 
                    fTargetArch(targarch), fImages(targarch), fLogging(NULL), 
                    fLogLevel(UNW_LOG_LEVEL_NONE)
    {
        fWrapper.type = UNW_REMOTE;
        fWrapper.ras = this;
        fRemoteRegisterMap = new RemoteRegisterMap(accessors, targarch);
        if (fTargetArch == UNW_TARGET_X86_64 || fTargetArch == UNW_TARGET_I386
            || fTargetArch == UNW_TARGET_ARM)
            fLittleEndian = true;
        else
            fLittleEndian = false;
    }

    ~RemoteProcInfo () {
        delete fRemoteRegisterMap;
    }

    bool haveProfile (uint64_t pc) {
        if (fImages.findProfileByTextAddr (pc))
          return true;
        else
          return false;
    }

    // returns NULL if profile does not yet exist.
    RemoteUnwindProfile* findProfile (uint64_t pc) {
        return fImages.findProfileByTextAddr (pc);
    }

    // returns NULL if the binary image is not yet added.
    bool addProfile (unw_accessors_t *acc, unw_addr_space_t as, uint64_t start, uint64_t end, void *arg) {
        if (fImages.addProfile (this, acc, as, start, end, arg))
          return true;
        else
          return false;
    }

    bool haveImageEntry (uint64_t pc, void *arg);

    bool getImageAddresses (uint64_t pc, uint64_t &mh, uint64_t &text_start, uint64_t &text_end, 
                            uint64_t &eh_frame_start, uint64_t &eh_frame_len, uint64_t &compact_unwind_start, 
                            void *arg);
    bool getImageAddresses (uint64_t pc, uint32_t &mh, uint32_t &text_start, uint32_t &text_end, 
                            uint32_t &eh_frame_start, uint32_t &eh_frame_len, uint32_t &compact_unwind_start, 
                            void *arg);

    bool addFuncBounds (uint64_t mh, std::vector<FuncBounds> &startAddrs)    { return fImages.addFuncBounds (mh, startAddrs); }
    bool haveFuncBounds (uint64_t mh)                                        { return fImages.haveFuncBounds (mh); }
    bool findStartAddr (uint64_t pc, uint32_t &startAddr, uint32_t &endAddr) { return fImages.findFuncBounds (pc, startAddr, endAddr); }
    bool findStartAddr (uint64_t pc, uint64_t &startAddr, uint64_t &endAddr) { return fImages.findFuncBounds (pc, startAddr, endAddr); }
    uint8_t *getMemBlobMemory (uint64_t addr, int len) { return fImages.getMemBlobMemory (addr, len); }


    // Functions to pull memory from the target into the debugger.

    int getBytes(uint64_t addr, uint64_t extent, uint8_t* buf, void* arg)
    {
        int err = readRaw(addr, extent, buf, arg);

        if(err)
            return 0;

        return 1;
    }

#define DECLARE_INT_ACCESSOR(bits)                                              \
    uint##bits##_t get##bits(uint64_t addr, void* arg)                            \
    {                                                                           \
        uint##bits##_t ret;                                                     \
        int err = readRaw(addr, (unw_word_t)(bits / 8), (uint8_t*)&ret, arg);   \
                                                                                \
        if(err)                                                                 \
            ABORT("Invalid memory access in the target");                       \
                                                                                \
        return ret;                                                             \
    }
    DECLARE_INT_ACCESSOR(8)
    DECLARE_INT_ACCESSOR(16)
    DECLARE_INT_ACCESSOR(32)
    DECLARE_INT_ACCESSOR(64)
#undef DECLARE_INT_ACCESSOR

// 'err' is set to 0 if there were no errors reading this
// memory.  Non-zero values indicate that the memory was not
// read successfully.  This method should be preferred over the
// method above which asserts on failure.

#define DECLARE_INT_ACCESSOR_ERR(bits)                                          \
    uint##bits##_t get##bits(uint64_t addr, int &err, void* arg)                  \
    {                                                                           \
        uint##bits##_t ret;                                                     \
        err = readRaw(addr, (unw_word_t)(bits / 8), (uint8_t*)&ret, arg);       \
                                                                                \
        return ret;                                                             \
    }
    DECLARE_INT_ACCESSOR_ERR(8)
    DECLARE_INT_ACCESSOR_ERR(16)
    DECLARE_INT_ACCESSOR_ERR(32)
    DECLARE_INT_ACCESSOR_ERR(64)
#undef DECLARE_INT_ACCESSOR_ERR

    double getDouble(uint64_t addr, void* arg)
    {
        double ret;
        int err = readRaw(addr, (unw_word_t)(sizeof(ret) / 8), (uint8_t*)&ret, arg);
        if(err)
            ABORT("Invalid memory access in the target");
        return ret;
    }

    v128 getVector(uint64_t addr, void* arg)
    {
        v128 ret;
        int err = readRaw(addr, (unw_word_t)(sizeof(ret) / 8), (uint8_t*)&ret, arg);
        if(err)
            ABORT("Invalid memory access in the target");
        return ret;
    }

    // Pull an unsigned LEB128 from the target into the debugger as a uint64_t.
    uint64_t getULEB128(uint64_t& addr, uint64_t end, void* arg)
    {
        uint64_t lAddr = addr;
        uint64_t ret = 0;
        uint8_t shift = 0;
        uint64_t byte;
        do {
            if(lAddr == end)
                ABORT("Truncated LEB128 number in the target");

            byte = (uint64_t)get8(lAddr, arg);
            lAddr++;

            if(((shift == 63) && (byte > 0x01)) || (shift > 63))
                ABORT("LEB128 number is larger than is locally representible");

            ret |= ((byte & 0x7f) << shift);
            shift += 7;
        } while((byte & 0x80) == 0x80);
        addr = lAddr;
        return ret;
    }

    // Pull an unsigned LEB128 from the target into the debugger as a uint64_t.
    uint64_t getULEB128(uint32_t& addr, uint32_t end, void* arg)
    {
        uint32_t lAddr = addr;
        uint64_t ret = 0;
        uint8_t shift = 0;
        uint64_t byte;
        do {
            if(lAddr == end)
                ABORT("Truncated LEB128 number in the target");

            byte = (uint64_t)get8(lAddr, arg);
            lAddr++;

            if(((shift == 63) && (byte > 0x01)) || (shift > 63))
                ABORT("LEB128 number is larger than is locally representible");

            ret |= ((byte & 0x7f) << shift);
            shift += 7;
        } while((byte & 0x80) == 0x80);
        addr = lAddr;
        return ret;
    }


    // Pull a signed LEB128 from the target into the debugger as a uint64_t.
    int64_t getSLEB128(uint64_t& addr, uint64_t end, void* arg)
    {
        uint64_t lAddr = addr;
        uint64_t ret = 0;
        uint8_t shift = 0;
        uint64_t byte;
        do {
            if(lAddr == end)
                ABORT("Truncated LEB128 number in the target");
            byte = (uint64_t)get8(lAddr, arg);
            lAddr++;
            if(((shift == 63) && (byte > 0x01)) || (shift > 63))
                ABORT("LEB128 number is larger than is locally representible");
            ret |= ((byte & 0x7f) << shift);
            shift += 7;
        } while((byte & 0x80) == 0x80);
        // Sign-extend
        if((shift < (sizeof(int64_t) * 8)) && (byte & 0x40))
            ret |= -(1 << shift);
        addr = lAddr;
        return ret;
    }

    // Pull a signed LEB128 from the target into the debugger as a uint64_t.
    int64_t getSLEB128(uint32_t& addr, uint32_t end, void* arg)
    {
        uint32_t lAddr = addr;
        uint64_t ret = 0;
        uint8_t shift = 0;
        uint64_t byte;
        do {
            if(lAddr == end)
                ABORT("Truncated LEB128 number in the target");
            byte = (uint64_t)get8(lAddr, arg);
            lAddr++;
            if(((shift == 63) && (byte > 0x01)) || (shift > 63))
                ABORT("LEB128 number is larger than is locally representible");
            ret |= ((byte & 0x7f) << shift);
            shift += 7;
        } while((byte & 0x80) == 0x80);
        // Sign-extend
        if((shift < (sizeof(int64_t) * 8)) && (byte & 0x40))
            ret |= -(1 << shift);
        addr = lAddr;
        return ret;
    }


    uint64_t getP (uint64_t addr, void *arg) {
        switch (fTargetArch) {
            case UNW_TARGET_X86_64:
              return get64(addr, arg);
              break;
            case UNW_TARGET_I386:
              return get32(addr, arg);
              break;
        }
        ABORT("Unknown target architecture.");
        return 0;
    }

    uint64_t getP (uint64_t addr, int& err, void *arg) {
        switch (fTargetArch) {
            case UNW_TARGET_X86_64:
              return get64(addr, err, arg);
              break;
            case UNW_TARGET_I386:
              return get32(addr, err, arg);
              break;
        }
        ABORT("Unknown target architecture.");
        return 0;
    }

    bool findFunctionName(uint64_t addr, char *buf, size_t bufLen, unw_word_t *offset, void* arg);
    bool findFunctionBounds(uint64_t addr, uint64_t& low, uint64_t& high, void* arg);
    int setCachingPolicy(unw_caching_policy_t policy);

    void setLoggingLevel(FILE *f, unw_log_level_t level);
    void logInfo(const char *fmt, ...);
    void logAPI(const char *fmt, ...);
    void logVerbose(const char *fmt, ...);
    void logDebug(const char *fmt, ...);
    struct timeval *timestamp_start ();
    void timestamp_stop (struct timeval *tstart, const char *fmt, ...);

    void flushAllCaches()                       { fImages.removeAllImageProfiles(); }
    void flushCacheByMachHeader(uint64_t mh)    { fImages.removeOneImageProfiles(mh); }
    unw_targettype_t getTargetArch()            { return fTargetArch; }
    unw_accessors_t* getAccessors ()            { return &fAccessors; }
    RemoteRegisterMap* getRegisterMap()         { return fRemoteRegisterMap; }
    unw_addr_space_t wrap ()                    { return (unw_addr_space_t) &fWrapper; }
    bool remoteIsLittleEndian ()                { return fLittleEndian; }
    unw_log_level_t getDebugLoggingLevel()      { return fLogLevel; }
    bool addMemBlob (RemoteMemoryBlob *blob)    { return fImages.addMemBlob(blob); }
    unw_caching_policy_t getCachingPolicy()     { return fCachingPolicy; }

private:
    int readRaw(uint64_t addr, uint64_t extent, uint8_t *valp, void* arg)
    {
        uint8_t *t = this->getMemBlobMemory (addr, extent);
        if (t) {
            memcpy (valp, t, extent);
            return 0;
        }
        return fAccessors.access_raw((unw_addr_space_t)this, addr, extent, valp, 0, arg);
    }

    struct unw_addr_space_remote    fWrapper;
    unw_accessors_t                 fAccessors;
    unw_caching_policy_t            fCachingPolicy;
    unw_targettype_t                fTargetArch;
    unw_addr_space_t                fAddrSpace;
    RemoteImages                    fImages;
    RemoteRegisterMap               *fRemoteRegisterMap;
    FILE                            *fLogging;
    unw_log_level_t                 fLogLevel;
    bool                            fLittleEndian;
};

// Find an image containing the given pc, returns false if absent and
// we can't add it via the accessors.
bool RemoteProcInfo::haveImageEntry (uint64_t pc, void *arg) {
    if (fImages.remoteEntryForTextAddr (pc) == NULL) {
        unw_word_t mh, text_start, text_end, eh_frame, eh_frame_len, compact_unwind, compact_unwind_len;
        if (fAccessors.find_image_info (wrap(), pc, &mh, &text_start, 
                                        &text_end, &eh_frame, &eh_frame_len, &compact_unwind, &compact_unwind_len, arg) == UNW_ESUCCESS) {
            fImages.addImage (mh, text_start, text_end, eh_frame, eh_frame_len, compact_unwind, compact_unwind_len);
            if (fCachingPolicy != UNW_CACHE_NONE) {
                if (compact_unwind_len != 0) {
                    logVerbose ("Creating RemoteMemoryBlob of compact unwind info image at mh 0x%llx, %lld bytes", mh, (uint64_t) compact_unwind_len);
                    uint8_t *buf = (uint8_t*) malloc (compact_unwind_len);
                    if (this->getBytes (compact_unwind, compact_unwind_len, buf, arg)) {
                        RemoteMemoryBlob *b = new RemoteMemoryBlob(buf, free, compact_unwind, compact_unwind_len, mh, NULL);
                        if (fImages.addMemBlob (b) == false)
                            delete b;
                    }
                } else if (eh_frame_len != 0) {
                    logVerbose ("Creating RemoteMemoryBlob of eh_frame for image at mh 0x%llx, %lld bytes", mh, (uint64_t) compact_unwind_len);
                    uint8_t *buf = (uint8_t*) malloc (eh_frame_len);
                    if (this->getBytes (eh_frame, eh_frame_len, buf, arg)) {
                        RemoteMemoryBlob *b = new RemoteMemoryBlob(buf, free, eh_frame, eh_frame_len, mh, NULL);
                        if (fImages.addMemBlob (b) == false)
                            delete b;
                    }
                }
            }
        } else {
            return false;  /// find_image_info failed
        }
    } else {
        return true;
    }
    return true;
}

bool RemoteProcInfo::getImageAddresses (uint64_t pc, uint64_t &mh, uint64_t &text_start, uint64_t &text_end, 
                        uint64_t &eh_frame_start, uint64_t &eh_frame_len, uint64_t &compact_unwind_start, 
                        void *arg) {
    // Make sure we have this RemoteImageEntry already - fetch it now if needed.
    if (haveImageEntry (pc, arg) == false) {
        return false;
    }
    RemoteImageEntry *r = fImages.remoteEntryForTextAddr (pc);
    if (r) {
        mh = r->mach_header;
        text_start = r->text_start;
        text_end = r->text_end;
        eh_frame_start = r->eh_frame_start;
        eh_frame_len = r->eh_frame_len;
        compact_unwind_start = r->compact_unwind_info_start;
        return true;
    }
    return false;
}


bool RemoteProcInfo::findFunctionName(uint64_t addr, char *buf, size_t bufLen, unw_word_t *offset, void* arg)
{
    if(fAccessors.get_proc_name(wrap(), addr, buf, bufLen, offset, arg) == UNW_ESUCCESS)
        return true;
    else
        return false;
}

bool RemoteProcInfo::findFunctionBounds(uint64_t addr, uint64_t& low, uint64_t& high, void* arg)
{
    if (fAccessors.get_proc_bounds(wrap(), addr, &low, &high, arg) == UNW_ESUCCESS
        && high != 0)
        return true;
    else
        return false;
}

int RemoteProcInfo::setCachingPolicy(unw_caching_policy_t policy)
{
    if(policy == UNW_CACHE_NONE && fCachingPolicy != UNW_CACHE_NONE)
    {
        flushAllCaches();
    }

    if(!(policy == UNW_CACHE_NONE || policy == UNW_CACHE_GLOBAL || policy == UNW_CACHE_PER_THREAD))
        return UNW_EINVAL;

    fCachingPolicy = policy;

    return UNW_ESUCCESS;
}

void RemoteProcInfo::setLoggingLevel(FILE *f, unw_log_level_t level)
{
    fLogLevel = level;
    fLogging = f;
}

void RemoteProcInfo::logInfo(const char *fmt, ...)
{
    if (fLogging == NULL || fLogLevel == UNW_LOG_LEVEL_NONE)
        return;
    if (fLogLevel & UNW_LOG_LEVEL_INFO) {
        va_list ap;
        va_start (ap, fmt);
        vfprintf (fLogging, fmt, ap);
        fputs ("\n", fLogging);
        va_end (ap);
    }
}

void RemoteProcInfo::logAPI(const char *fmt, ...)
{
    if (fLogging == NULL || fLogLevel == UNW_LOG_LEVEL_NONE)
        return;
    if (fLogLevel & UNW_LOG_LEVEL_API) {
        va_list ap;
        va_start (ap, fmt);
        vfprintf (fLogging, fmt, ap);
        fputs ("\n", fLogging);
        va_end (ap);
    }
}

void RemoteProcInfo::logVerbose(const char *fmt, ...)
{
    if (fLogging == NULL || fLogLevel == UNW_LOG_LEVEL_NONE)
        return;
    if (fLogLevel & UNW_LOG_LEVEL_VERBOSE) {
        va_list ap;
        va_start (ap, fmt);
        vfprintf (fLogging, fmt, ap);
        fputs ("\n", fLogging);
        va_end (ap);
    }
}

void RemoteProcInfo::logDebug(const char *fmt, ...)
{
    if (fLogging == NULL || fLogLevel == UNW_LOG_LEVEL_NONE)
        return;
    if (fLogLevel & UNW_LOG_LEVEL_DEBUG) {
        va_list ap;
        va_start (ap, fmt);
        vfprintf (fLogging, fmt, ap);
        fputs ("\n", fLogging);
        va_end (ap);
    }
}

struct timeval *RemoteProcInfo::timestamp_start ()
{
    if (fLogging == NULL || fLogLevel == UNW_LOG_LEVEL_NONE)
        return NULL;
    if (fLogLevel & UNW_LOG_LEVEL_TIMINGS) {
        struct timeval *t = (struct timeval *) malloc (sizeof (struct timeval));
        if (gettimeofday (t, NULL) != 0) {
            free (t);
            return NULL;
        }
        return t;
    }
    return NULL;
}

void RemoteProcInfo::timestamp_stop (struct timeval *tstart, const char *fmt, ...)
{
    if (fLogging == NULL || fLogLevel == UNW_LOG_LEVEL_NONE || tstart == NULL)
        return;
    if (fLogLevel & UNW_LOG_LEVEL_TIMINGS) {
        struct timeval tend;
        if (gettimeofday (&tend, NULL) != 0) {
            free (tstart);
            return;
        }
        struct timeval result;
        timersub (&tend, tstart, &result);
        va_list ap;
        va_start (ap, fmt);
        vprintf (fmt, ap);
        printf (" duration %0.5fs\n", (double) ((result.tv_sec * 1000000) + result.tv_usec) / 1000000.0);
        va_end (ap);
        free (tstart);
    }
}


// Initialize the register context at the start of a remote unwind.

void getRemoteContext (RemoteProcInfo* procinfo, Registers_x86_64& r, void *arg) {
    unw_accessors_t* accessors = procinfo->getAccessors();
    unw_addr_space_t addrSpace = procinfo->wrap();
    RemoteRegisterMap* regmap = procinfo->getRegisterMap();
    uint64_t rv;

    // now that we have a selected process/thread, ask about the valid registers.
    regmap->scan_caller_regs (addrSpace, arg);

#define FILLREG(reg) {int caller_reg; regmap->unwind_regno_to_caller_regno ((reg), caller_reg); accessors->access_reg (addrSpace, caller_reg, &rv, 0, arg); r.setRegister ((reg), rv);}
    FILLREG (UNW_X86_64_RAX);
    FILLREG (UNW_X86_64_RDX);
    FILLREG (UNW_X86_64_RCX);
    FILLREG (UNW_X86_64_RBX);
    FILLREG (UNW_X86_64_RSI);
    FILLREG (UNW_X86_64_RDI);
    FILLREG (UNW_X86_64_RBP);
    FILLREG (UNW_X86_64_RSP);
    FILLREG (UNW_X86_64_R8);
    FILLREG (UNW_X86_64_R9);
    FILLREG (UNW_X86_64_R10);
    FILLREG (UNW_X86_64_R11);
    FILLREG (UNW_X86_64_R12);
    FILLREG (UNW_X86_64_R13);
    FILLREG (UNW_X86_64_R14);
    FILLREG (UNW_X86_64_R15);
    FILLREG (UNW_REG_IP);
#undef FILLREG
}

void getRemoteContext (RemoteProcInfo* procinfo, Registers_x86& r, void *arg) {
    unw_accessors_t* accessors = procinfo->getAccessors();
    unw_addr_space_t addrSpace = procinfo->wrap();
    RemoteRegisterMap* regmap = procinfo->getRegisterMap();
    uint64_t rv;

    // now that we have a selected process/thread, ask about the valid registers.
    regmap->scan_caller_regs (addrSpace, arg);

#define FILLREG(reg) {int caller_reg; regmap->unwind_regno_to_caller_regno ((reg), caller_reg); accessors->access_reg (addrSpace, caller_reg, &rv, 0, arg); r.setRegister ((reg), rv);}
    FILLREG (UNW_X86_EAX);
    FILLREG (UNW_X86_ECX);
    FILLREG (UNW_X86_EDX);
    FILLREG (UNW_X86_EBX);
    FILLREG (UNW_X86_EBP);
    FILLREG (UNW_X86_ESP);
    FILLREG (UNW_X86_ESI);
    FILLREG (UNW_X86_EDI);
    FILLREG (UNW_REG_IP);
#undef FILLREG
}

}; // namespace lldb_private



#endif // SUPPORT_REMOTE_UNWINDING
#endif // __REMOTE_PROC_INFO_HPP__
