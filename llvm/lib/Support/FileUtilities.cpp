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

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/System/Path.h"
#include "llvm/Config/unistd.h"
#include "llvm/Config/fcntl.h"
#include "llvm/Config/sys/types.h"
#include "llvm/Config/sys/stat.h"
#include "llvm/Config/sys/mman.h"
#include "llvm/Config/alloca.h"
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <iostream>
using namespace llvm;

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


/// CopyFile - Copy the specified source file to the specified destination,
/// overwriting destination if it exists.  This returns true on failure.
///
bool llvm::CopyFile(const std::string &Dest, const std::string &Src) {
  FDHandle InFD(open(Src.c_str(), O_RDONLY));
  if (InFD == -1) return true;

  FileRemover FR(Dest);

  FDHandle OutFD(open(Dest.c_str(), O_WRONLY|O_CREAT, 0666));
  if (OutFD == -1) return true;

  char Buffer[16*1024];
  while (ssize_t Amt = read(InFD, Buffer, 16*1024)) {
    if (Amt == -1) {
      if (errno != EINTR) return true;  // Error reading the file.
    } else {
      char *BufPtr = Buffer;
      while (Amt) {
        ssize_t AmtWritten = write(OutFD, BufPtr, Amt);
        if (AmtWritten == -1) {
          if (errno != EINTR) return true;  // Error writing the file.
        } else {
          Amt -= AmtWritten;
          BufPtr += AmtWritten;
        }
      }
    }
  }

  FR.releaseFile();  // Success!
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
#if defined(HAVE_MKSTEMP) && !defined(_MSC_VER)
  int TempFD;
  if ((TempFD = mkstemp(FNBuffer)) == -1) {
    // FIXME: this should return an emtpy string or something and allow the
    // caller to deal with the error!
    std::cerr << "bugpoint: ERROR: Cannot create temporary file in the current "
	      << " directory!\n";
    exit(1);
  }

  // We don't need to hold the temp file descriptor... we will trust that no one
  // will overwrite/delete the file while we are working on it...
  close(TempFD);
#else
  // If we don't have mkstemp, use the old and obsolete mktemp function.
  if (mktemp(FNBuffer) == 0) {
    // FIXME: this should return an emtpy string or something and allow the
    // caller to deal with the error!
    std::cerr << "bugpoint: ERROR: Cannot create temporary file in the current "
              << " directory!\n";
    exit(1);
  }
#endif

  std::string Result(FNBuffer);
  delete[] FNBuffer;
  return Result;
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

