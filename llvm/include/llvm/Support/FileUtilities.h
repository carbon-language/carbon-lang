//===- llvm/Support/FileUtilities.h - File System Utilities -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a family of utility functions which are useful for doing
// various things with files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILEUTILITIES_H
#define LLVM_SUPPORT_FILEUTILITIES_H

#include <string>

namespace llvm {

/// DiffFiles - Compare the two files specified, returning true if they are
/// different or if there is a file error.  If you specify a string to fill in
/// for the error option, it will set the string to an error message if an error
/// occurs, allowing the caller to distinguish between a failed diff and a file
/// system error.
///
bool DiffFiles(const std::string &FileA, const std::string &FileB,
               std::string *Error = 0);

/// MoveFileOverIfUpdated - If the file specified by New is different than Old,
/// or if Old does not exist, move the New file over the Old file.  Otherwise,
/// remove the New file.
///
void MoveFileOverIfUpdated(const std::string &New, const std::string &Old);
 
/// removeFile - Delete the specified file.
///
void removeFile(const std::string &Filename);

/// FDHandle - Simple handle class to make sure a file descriptor gets closed
/// when the object is destroyed.  This handle acts similarly to an
/// std::auto_ptr, in that the copy constructor and assignment operators
/// transfer ownership of the handle.  This means that FDHandle's do not have
/// value semantics.
///
class FDHandle {
  int FD;
public:
  FDHandle() : FD(-1) {}
  FDHandle(int fd) : FD(fd) {}
  FDHandle(FDHandle &RHS) : FD(RHS.FD) {
    RHS.FD = -1;       // Transfer ownership
  }

  ~FDHandle() throw();

  /// get - Get the current file descriptor, without releasing ownership of it.
  int get() const { return FD; }
  operator int() const { return FD; }

  FDHandle &operator=(int fd) throw();

  FDHandle &operator=(FDHandle &RHS) {
    int fd = RHS.FD;
    RHS.FD = -1;       // Transfer ownership
    return operator=(fd);
  }

  /// release - Take ownership of the file descriptor away from the FDHandle
  /// object, so that the file is not closed when the FDHandle is destroyed.
  int release() {
    int Ret = FD;
    FD = -1;
    return Ret;
  }
};

  /// FileRemover - This class is a simple object meant to be stack allocated.
  /// If an exception is thrown from a region, the object removes the filename
  /// specified (if deleteIt is true).
  ///
  class FileRemover {
    std::string Filename;
    bool DeleteIt;
  public:
    FileRemover(const std::string &filename, bool deleteIt = true)
      : Filename(filename), DeleteIt(deleteIt) {}
    
    ~FileRemover() {
      if (DeleteIt) removeFile(Filename);
    }

    /// releaseFile - Take ownership of the file away from the FileRemover so it
    /// will not be removed when the object is destroyed.
    void releaseFile() { DeleteIt = false; }
  };
} // End llvm namespace

#endif
