//===- Support/FileUtilities.cpp - File System Utilities ------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#include "Support/FileUtilities.h"
#include "Config/unistd.h"
#include "Config/fcntl.h"
#include "Config/sys/stat.h"
#include "Config/sys/types.h"
#include "Config/sys/mman.h"
#include <fstream>
#include <iostream>
#include <cstdio>
using namespace llvm;

/// CheckMagic - Returns true IFF the file named FN begins with Magic. FN must
/// name a readable file.
///
bool llvm::CheckMagic(const std::string &FN, const std::string &Magic) {
  char buf[1 + Magic.size ()];
  std::ifstream f (FN.c_str ());
  f.read (buf, Magic.size ());
  buf[Magic.size ()] = '\0';
  return Magic == buf;
}

/// IsArchive - Returns true IFF the file named FN appears to be a "ar" library
/// archive. The file named FN must exist.
///
bool llvm::IsArchive(const std::string &FN) {
  // Inspect the beginning of the file to see if it contains the "ar"
  // library archive format magic string.
  return CheckMagic (FN, "!<arch>\012");
}

/// IsBytecode - Returns true IFF the file named FN appears to be an LLVM
/// bytecode file. The file named FN must exist.
///
bool llvm::IsBytecode(const std::string &FN) {
  // Inspect the beginning of the file to see if it contains the LLVM
  // bytecode format magic string.
  return CheckMagic (FN, "llvm");
}

/// IsSharedObject - Returns trus IFF the file named FN appears to be a shared
/// object with an ELF header. The file named FN must exist.
///
bool llvm::IsSharedObject(const std::string &FN) {
  // Inspect the beginning of the file to see if it contains the ELF shared
  // object magic string.
  static const char elfMagic[] = { 0x7f, 'E', 'L', 'F', '\0' };
  return CheckMagic(FN, elfMagic);
}

/// FileOpenable - Returns true IFF Filename names an existing regular
/// file which we can successfully open.
///
bool llvm::FileOpenable(const std::string &Filename) {
  struct stat s;
  if (stat (Filename.c_str (), &s) == -1)
    return false; // Cannot stat file
  if (!S_ISREG (s.st_mode))
    return false; // File is not a regular file
  std::ifstream FileStream (Filename.c_str ());
  if (!FileStream)
    return false; // File is not openable
  return true;
}

/// DiffFiles - Compare the two files specified, returning true if they are
/// different or if there is a file error.  If you specify a string to fill in
/// for the error option, it will set the string to an error message if an error
/// occurs, allowing the caller to distinguish between a failed diff and a file
/// system error.
///
bool llvm::DiffFiles(const std::string &FileA, const std::string &FileB,
                     std::string *Error) {
  std::ifstream FileAStream(FileA.c_str());
  if (!FileAStream) {
    if (Error) *Error = "Couldn't open file '" + FileA + "'";
    return true;
  }

  std::ifstream FileBStream(FileB.c_str());
  if (!FileBStream) {
    if (Error) *Error = "Couldn't open file '" + FileB + "'";
    return true;
  }

  // Compare the two files...
  int C1, C2;
  do {
    C1 = FileAStream.get();
    C2 = FileBStream.get();
    if (C1 != C2) return true;
  } while (C1 != EOF);

  return false;
}


/// MoveFileOverIfUpdated - If the file specified by New is different than Old,
/// or if Old does not exist, move the New file over the Old file.  Otherwise,
/// remove the New file.
///
void llvm::MoveFileOverIfUpdated(const std::string &New,
                                 const std::string &Old) {
  if (DiffFiles(New, Old)) {
    if (std::rename(New.c_str(), Old.c_str()))
      std::cerr << "Error renaming '" << New << "' to '" << Old << "'!\n";
  } else {
    std::remove(New.c_str());
  }  
}

/// removeFile - Delete the specified file
///
void llvm::removeFile(const std::string &Filename) {
  std::remove(Filename.c_str());
}

/// getUniqueFilename - Return a filename with the specified prefix.  If the
/// file does not exist yet, return it, otherwise add a suffix to make it
/// unique.
///
std::string llvm::getUniqueFilename(const std::string &FilenameBase) {
  if (!std::ifstream(FilenameBase.c_str()))
    return FilenameBase;    // Couldn't open the file? Use it!

  // Create a pattern for mkstemp...
  char *FNBuffer = new char[FilenameBase.size()+8];
  strcpy(FNBuffer, FilenameBase.c_str());
  strcpy(FNBuffer+FilenameBase.size(), "-XXXXXX");

  // Agree on a temporary file name to use....
  int TempFD;
  if ((TempFD = mkstemp(FNBuffer)) == -1) {
    std::cerr << "bugpoint: ERROR: Cannot create temporary file in the current "
	      << " directory!\n";
    exit(1);
  }

  // We don't need to hold the temp file descriptor... we will trust that no one
  // will overwrite/delete the file while we are working on it...
  close(TempFD);
  std::string Result(FNBuffer);
  delete[] FNBuffer;
  return Result;
}

static bool AddPermissionsBits (const std::string &Filename, mode_t bits) {
  // Get the umask value from the operating system.  We want to use it
  // when changing the file's permissions. Since calling umask() sets
  // the umask and returns its old value, we must call it a second
  // time to reset it to the user's preference.
  mode_t mask = umask (0777); // The arg. to umask is arbitrary...
  umask (mask);

  // Get the file's current mode.
  struct stat st;
  if ((stat (Filename.c_str(), &st)) == -1)
    return false;

  // Change the file to have whichever permissions bits from 'bits'
  // that the umask would not disable.
  if ((chmod(Filename.c_str(), (st.st_mode | (bits & ~mask)))) == -1)
    return false;

  return true;
}

/// MakeFileExecutable - Make the file named Filename executable by
/// setting whichever execute permissions bits the process's current
/// umask would allow. Filename must name an existing file or
/// directory.  Returns true on success, false on error.
///
bool llvm::MakeFileExecutable(const std::string &Filename) {
  return AddPermissionsBits(Filename, 0111);
}

/// MakeFileReadable - Make the file named Filename readable by
/// setting whichever read permissions bits the process's current
/// umask would allow. Filename must name an existing file or
/// directory.  Returns true on success, false on error.
///
bool llvm::MakeFileReadable(const std::string &Filename) {
  return AddPermissionsBits(Filename, 0444);
}

/// getFileSize - Return the size of the specified file in bytes, or -1 if the
/// file cannot be read or does not exist.
long long llvm::getFileSize(const std::string &Filename) {
  struct stat StatBuf;
  if (stat(Filename.c_str(), &StatBuf) == -1)
    return -1;
  return StatBuf.st_size;  
}

/// getFileTimestamp - Get the last modified time for the specified file in an
/// unspecified format.  This is useful to allow checking to see if a file was
/// updated since that last time the timestampt was aquired.  If the file does
/// not exist or there is an error getting the time-stamp, zero is returned.
unsigned long long llvm::getFileTimestamp(const std::string &Filename) {
  struct stat StatBuf;
  if (stat(Filename.c_str(), &StatBuf) == -1)
    return 0;
  return StatBuf.st_mtime;  
}

/// ReadFileIntoAddressSpace - Attempt to map the specific file into the
/// address space of the current process for reading.  If this succeeds,
/// return the address of the buffer and the length of the file mapped.  On
/// failure, return null.
void *llvm::ReadFileIntoAddressSpace(const std::string &Filename, 
                                     unsigned &Length) {
#ifdef HAVE_MMAP_FILE
  Length = getFileSize(Filename);
  if ((int)Length == -1) return 0;

  FDHandle FD(open(Filename.c_str(), O_RDONLY));
  if (FD == -1) return 0;

  // mmap in the file all at once...
  void *Buffer = (void*)mmap(0, Length, PROT_READ, MAP_PRIVATE, FD, 0);

  if (Buffer == (void*)MAP_FAILED)
    return 0;
  return Buffer;
#else
  // FIXME: implement with read/write
  return 0;
#endif
}

/// UnmapFileFromAddressSpace - Remove the specified file from the current
/// address space.
void llvm::UnmapFileFromAddressSpace(void *Buffer, unsigned Length) {
#ifdef HAVE_MMAP_FILE
  munmap((char*)Buffer, Length);
#else
  free(Buffer);
#endif
}

//===----------------------------------------------------------------------===//
// FDHandle class implementation
//

FDHandle::~FDHandle() throw() {
  if (FD != -1) close(FD);
}

FDHandle &FDHandle::operator=(int fd) throw() {
  if (FD != -1) close(FD);
  FD = fd;
  return *this;
}

