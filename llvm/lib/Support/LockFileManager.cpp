//===--- LockFileManager.cpp - File-level Locking Utility------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/LockFileManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <sys/stat.h>
#include <sys/types.h>
#if LLVM_ON_WIN32
#include <windows.h>
#endif
#if LLVM_ON_UNIX
#include <unistd.h>
#endif
using namespace llvm;

/// \brief Attempt to read the lock file with the given name, if it exists.
///
/// \param LockFileName The name of the lock file to read.
///
/// \returns The process ID of the process that owns this lock file
Optional<std::pair<std::string, int> >
LockFileManager::readLockFile(StringRef LockFileName) {
  // Read the owning host and PID out of the lock file. If it appears that the
  // owning process is dead, the lock file is invalid.
  std::unique_ptr<MemoryBuffer> MB;
  if (MemoryBuffer::getFile(LockFileName, MB)) {
    sys::fs::remove(LockFileName);
    return None;
  }

  StringRef Hostname;
  StringRef PIDStr;
  std::tie(Hostname, PIDStr) = getToken(MB->getBuffer(), " ");
  PIDStr = PIDStr.substr(PIDStr.find_first_not_of(" "));
  int PID;
  if (!PIDStr.getAsInteger(10, PID))
    return std::make_pair(std::string(Hostname), PID);

  // Delete the lock file. It's invalid anyway.
  sys::fs::remove(LockFileName);
  return None;
}

bool LockFileManager::processStillExecuting(StringRef Hostname, int PID) {
#if LLVM_ON_UNIX && !defined(__ANDROID__)
  char MyHostname[256];
  MyHostname[255] = 0;
  MyHostname[0] = 0;
  gethostname(MyHostname, 255);
  // Check whether the process is dead. If so, we're done.
  if (MyHostname == Hostname && getsid(PID) == -1 && errno == ESRCH)
    return false;
#endif

  return true;
}

LockFileManager::LockFileManager(StringRef FileName)
{
  this->FileName = FileName;
  if (error_code EC = sys::fs::make_absolute(this->FileName)) {
    Error = EC;
    return;
  }
  LockFileName = this->FileName;
  LockFileName += ".lock";

  // If the lock file already exists, don't bother to try to create our own
  // lock file; it won't work anyway. Just figure out who owns this lock file.
  if ((Owner = readLockFile(LockFileName)))
    return;

  // Create a lock file that is unique to this instance.
  UniqueLockFileName = LockFileName;
  UniqueLockFileName += "-%%%%%%%%";
  int UniqueLockFileID;
  if (error_code EC
        = sys::fs::createUniqueFile(UniqueLockFileName.str(),
                                    UniqueLockFileID,
                                    UniqueLockFileName)) {
    Error = EC;
    return;
  }

  // Write our process ID to our unique lock file.
  {
    raw_fd_ostream Out(UniqueLockFileID, /*shouldClose=*/true);

#if LLVM_ON_UNIX
    // FIXME: move getpid() call into LLVM
    char hostname[256];
    hostname[255] = 0;
    hostname[0] = 0;
    gethostname(hostname, 255);
    Out << hostname << ' ' << getpid();
#else
    Out << "localhost 1";
#endif
    Out.close();

    if (Out.has_error()) {
      // We failed to write out PID, so make up an excuse, remove the
      // unique lock file, and fail.
      Error = make_error_code(errc::no_space_on_device);
      sys::fs::remove(UniqueLockFileName.c_str());
      return;
    }
  }

  while (1) {
    // Create a link from the lock file name. If this succeeds, we're done.
    error_code EC =
        sys::fs::create_link(UniqueLockFileName.str(), LockFileName.str());
    if (EC == errc::success)
      return;

    if (EC != errc::file_exists) {
      Error = EC;
      return;
    }

    // Someone else managed to create the lock file first. Read the process ID
    // from the lock file.
    if ((Owner = readLockFile(LockFileName))) {
      // Wipe out our unique lock file (it's useless now)
      sys::fs::remove(UniqueLockFileName.str());
      return;
    }

    if (!sys::fs::exists(LockFileName.str())) {
      // The previous owner released the lock file before we could read it.
      // Try to get ownership again.
      continue;
    }

    // There is a lock file that nobody owns; try to clean it up and get
    // ownership.
    if ((EC = sys::fs::remove(LockFileName.str()))) {
      Error = EC;
      return;
    }
  }
}

LockFileManager::LockFileState LockFileManager::getState() const {
  if (Owner)
    return LFS_Shared;

  if (Error)
    return LFS_Error;

  return LFS_Owned;
}

LockFileManager::~LockFileManager() {
  if (getState() != LFS_Owned)
    return;

  // Since we own the lock, remove the lock file and our own unique lock file.
  sys::fs::remove(LockFileName.str());
  sys::fs::remove(UniqueLockFileName.str());
}

void LockFileManager::waitForUnlock() {
  if (getState() != LFS_Shared)
    return;

#if LLVM_ON_WIN32
  unsigned long Interval = 1;
#else
  struct timespec Interval;
  Interval.tv_sec = 0;
  Interval.tv_nsec = 1000000;
#endif
  // Don't wait more than five minutes for the file to appear.
  unsigned MaxSeconds = 300;
  bool LockFileGone = false;
  do {
    // Sleep for the designated interval, to allow the owning process time to
    // finish up and remove the lock file.
    // FIXME: Should we hook in to system APIs to get a notification when the
    // lock file is deleted?
#if LLVM_ON_WIN32
    Sleep(Interval);
#else
    nanosleep(&Interval, NULL);
#endif
    bool LockFileJustDisappeared = false;

    // If the lock file is still expected to be there, check whether it still
    // is.
    if (!LockFileGone) {
      bool Exists;
      if (!sys::fs::exists(LockFileName.str(), Exists) && !Exists) {
        LockFileGone = true;
        LockFileJustDisappeared = true;
      }
    }

    // If the lock file is no longer there, check if the original file is
    // available now.
    if (LockFileGone) {
      if (sys::fs::exists(FileName.str())) {
        return;
      }

      // The lock file is gone, so now we're waiting for the original file to
      // show up. If this just happened, reset our waiting intervals and keep
      // waiting.
      if (LockFileJustDisappeared) {
        MaxSeconds = 5;

#if LLVM_ON_WIN32
        Interval = 1;
#else
        Interval.tv_sec = 0;
        Interval.tv_nsec = 1000000;
#endif
        continue;
      }
    }

    // If we're looking for the lock file to disappear, but the process
    // owning the lock died without cleaning up, just bail out.
    if (!LockFileGone &&
        !processStillExecuting((*Owner).first, (*Owner).second)) {
      return;
    }

    // Exponentially increase the time we wait for the lock to be removed.
#if LLVM_ON_WIN32
    Interval *= 2;
#else
    Interval.tv_sec *= 2;
    Interval.tv_nsec *= 2;
    if (Interval.tv_nsec >= 1000000000) {
      ++Interval.tv_sec;
      Interval.tv_nsec -= 1000000000;
    }
#endif
  } while (
#if LLVM_ON_WIN32
           Interval < MaxSeconds * 1000
#else
           Interval.tv_sec < (time_t)MaxSeconds
#endif
           );

  // Give up.
}
