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
#include "Config/sys/stat.h"
#include "Config/sys/types.h"
#include <fstream>
#include <iostream>
#include <cstdio>

/// FileOpenable - Returns true IFF Filename names an existing regular
/// file which we can successfully open.
///
bool FileOpenable (const std::string &Filename) {
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
bool DiffFiles(const std::string &FileA, const std::string &FileB,
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
void MoveFileOverIfUpdated(const std::string &New, const std::string &Old) {
  if (DiffFiles(New, Old)) {
    if (std::rename(New.c_str(), Old.c_str()))
      std::cerr << "Error renaming '" << New << "' to '" << Old << "'!\n";
  } else {
    std::remove(New.c_str());
  }  
}

/// removeFile - Delete the specified file
///
void removeFile(const std::string &Filename) {
  std::remove(Filename.c_str());
}

/// getUniqueFilename - Return a filename with the specified prefix.  If the
/// file does not exist yet, return it, otherwise add a suffix to make it
/// unique.
///
std::string getUniqueFilename(const std::string &FilenameBase) {
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
bool MakeFileExecutable (const std::string &Filename) {
  return AddPermissionsBits (Filename, 0111);
}

/// MakeFileReadable - Make the file named Filename readable by
/// setting whichever read permissions bits the process's current
/// umask would allow. Filename must name an existing file or
/// directory.  Returns true on success, false on error.
///
bool MakeFileReadable (const std::string &Filename) {
  return AddPermissionsBits (Filename, 0444);
}
