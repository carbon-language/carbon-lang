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
#include "llvm/System/Path.h"
#include "llvm/System/MappedFile.h"
#include "llvm/ADT/StringExtras.h"
#include <cmath>
#include <cstring>
#include <cctype>
using namespace llvm;

static bool isNumberChar(char C) {
  switch (C) {
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
  case '.': case '+': case '-':
  case 'D':  // Strange exponential notation.
  case 'd':  // Strange exponential notation.
  case 'e':
  case 'E': return true;
  default: return false;
  }
}

static char *BackupNumber(char *Pos, char *FirstChar) {
  // If we didn't stop in the middle of a number, don't backup.
  if (!isNumberChar(*Pos)) return Pos;

  // Otherwise, return to the start of the number.
  while (Pos > FirstChar && isNumberChar(Pos[-1]))
    --Pos;
  return Pos;
}

/// CompareNumbers - compare two numbers, returning true if they are different.
static bool CompareNumbers(char *&F1P, char *&F2P, char *F1End, char *F2End,
                           double AbsTolerance, double RelTolerance,
                           std::string *ErrorMsg) {
  char *F1NumEnd, *F2NumEnd;
  double V1 = 0.0, V2 = 0.0;

  // If one of the positions is at a space and the other isn't, chomp up 'til
  // the end of the space.
  while (isspace(*F1P) && F1P != F1End)
    ++F1P;
  while (isspace(*F2P) && F2P != F2End)
    ++F2P;

  // If we stop on numbers, compare their difference.  Note that some ugliness
  // is built into this to permit support for numbers that use "D" or "d" as
  // their exponential marker, e.g. "1.234D45".  This occurs in 200.sixtrack in
  // spec2k.
  if (isNumberChar(*F1P) && isNumberChar(*F2P)) {
    bool isDNotation;
    do {
      isDNotation = false;
      V1 = strtod(F1P, &F1NumEnd);
      V2 = strtod(F2P, &F2NumEnd);

      if (*F1NumEnd == 'D' || *F1NumEnd == 'd') {
        *F1NumEnd = 'e';  // Strange exponential notation!
        isDNotation = true;
      }
      if (*F2NumEnd == 'D' || *F2NumEnd == 'd') {
        *F2NumEnd = 'e';  // Strange exponential notation!
        isDNotation = true;
      }
    } while (isDNotation);
  } else {
    // Otherwise, the diff failed.
    F1NumEnd = F1P;
    F2NumEnd = F2P;
  }

  if (F1NumEnd == F1P || F2NumEnd == F2P) {
    if (ErrorMsg) *ErrorMsg = "Comparison failed, not a numeric difference.";
    return true;
  }

  // Check to see if these are inside the absolute tolerance
  if (AbsTolerance < std::abs(V1-V2)) {
    // Nope, check the relative tolerance...
    double Diff;
    if (V2)
      Diff = std::abs(V1/V2 - 1.0);
    else if (V1)
      Diff = std::abs(V2/V1 - 1.0);
    else
      Diff = 0;  // Both zero.
    if (Diff > RelTolerance) {
      if (ErrorMsg) {
        *ErrorMsg = "Compared: " + ftostr(V1) + " and " + ftostr(V2) +
                    ": diff = " + ftostr(Diff) + "\n";
        *ErrorMsg += "Out of tolerance: rel/abs: " + ftostr(RelTolerance) +
                     "/" + ftostr(AbsTolerance);
      }
      return true;
    }
  }

  // Otherwise, advance our read pointers to the end of the numbers.
  F1P = F1NumEnd;  F2P = F2NumEnd;
  return false;
}

// PadFileIfNeeded - If the files are not identical, we will have to be doing
// numeric comparisons in here.  There are bad cases involved where we (i.e.,
// strtod) might run off the beginning or end of the file if it starts or ends
// with a number.  Because of this, if needed, we pad the file so that it starts
// and ends with a null character.
static void PadFileIfNeeded(char *&FileStart, char *&FileEnd, char *&FP) {
  if (FileStart-FileEnd < 2 ||
      isNumberChar(FileStart[0]) || isNumberChar(FileEnd[-1])) {
    unsigned FileLen = FileEnd-FileStart;
    char *NewFile = new char[FileLen+2];
    NewFile[0] = 0;              // Add null padding
    NewFile[FileLen+1] = 0;      // Add null padding
    memcpy(NewFile+1, FileStart, FileLen);
    FP = NewFile+(FP-FileStart)+1;
    FileStart = NewFile+1;
    FileEnd = FileStart+FileLen;
  }
}

/// DiffFilesWithTolerance - Compare the two files specified, returning 0 if the
/// files match, 1 if they are different, and 2 if there is a file error.  This
/// function differs from DiffFiles in that you can specify an absolete and
/// relative FP error that is allowed to exist.  If you specify a string to fill
/// in for the error option, it will set the string to an error message if an
/// error occurs, allowing the caller to distinguish between a failed diff and a
/// file system error.
///
int llvm::DiffFilesWithTolerance(const sys::Path &FileA,
                                 const sys::Path &FileB,
                                 double AbsTol, double RelTol,
                                 std::string *Error) {
  sys::FileStatus FileAStat, FileBStat;
  if (FileA.getFileStatus(FileAStat, Error) ||
      FileB.getFileStatus(FileBStat, Error))
    return 2;
  // Check for zero length files because some systems croak when you try to
  // mmap an empty file.
  size_t A_size = FileAStat.getSize();
  size_t B_size = FileBStat.getSize();

  // If they are both zero sized then they're the same
  if (A_size == 0 && B_size == 0)
    return 0;
  // If only one of them is zero sized then they can't be the same
  if ((A_size == 0 || B_size == 0))
    return 1;

  try {
    // Now its safe to mmap the files into memory becasue both files
    // have a non-zero size.
    sys::MappedFile F1(FileA);
    sys::MappedFile F2(FileB);
    F1.map();
    F2.map();

    // Okay, now that we opened the files, scan them for the first difference.
    char *File1Start = F1.charBase();
    char *File2Start = F2.charBase();
    char *File1End = File1Start+A_size;
    char *File2End = File2Start+B_size;
    char *F1P = File1Start;
    char *F2P = File2Start;

    if (A_size == B_size) {
      // Are the buffers identical?
      if (std::memcmp(File1Start, File2Start, A_size) == 0)
        return 0;

      if (AbsTol == 0 && RelTol == 0)
        return 1;   // Files different!
    }

    char *OrigFile1Start = File1Start;
    char *OrigFile2Start = File2Start;

    // If the files need padding, do so now.
    PadFileIfNeeded(File1Start, File1End, F1P);
    PadFileIfNeeded(File2Start, File2End, F2P);

    bool CompareFailed = false;
    while (1) {
      // Scan for the end of file or next difference.
      while (F1P < File1End && F2P < File2End && *F1P == *F2P)
        ++F1P, ++F2P;

      if (F1P >= File1End || F2P >= File2End) break;

      // Okay, we must have found a difference.  Backup to the start of the
      // current number each stream is at so that we can compare from the
      // beginning.
      F1P = BackupNumber(F1P, File1Start);
      F2P = BackupNumber(F2P, File2Start);

      // Now that we are at the start of the numbers, compare them, exiting if
      // they don't match.
      if (CompareNumbers(F1P, F2P, File1End, File2End, AbsTol, RelTol, Error)) {
        CompareFailed = true;
        break;
      }
    }

    // Okay, we reached the end of file.  If both files are at the end, we
    // succeeded.
    bool F1AtEnd = F1P >= File1End;
    bool F2AtEnd = F2P >= File2End;
    if (!CompareFailed && (!F1AtEnd || !F2AtEnd)) {
      // Else, we might have run off the end due to a number: backup and retry.
      if (F1AtEnd && isNumberChar(F1P[-1])) --F1P;
      if (F2AtEnd && isNumberChar(F2P[-1])) --F2P;
      F1P = BackupNumber(F1P, File1Start);
      F2P = BackupNumber(F2P, File2Start);

      // Now that we are at the start of the numbers, compare them, exiting if
      // they don't match.
      if (CompareNumbers(F1P, F2P, File1End, File2End, AbsTol, RelTol, Error))
        CompareFailed = true;

      // If we found the end, we succeeded.
      if (F1P < File1End || F2P < File2End)
        CompareFailed = true;
    }

    if (OrigFile1Start != File1Start)
      delete[] (File1Start-1);   // Back up past null byte
    if (OrigFile2Start != File2Start)
      delete[] (File2Start-1);   // Back up past null byte
    return CompareFailed;
  } catch (const std::string &Msg) {
    if (Error) *Error = Msg;
    return 2;
  } catch (...) {
    *Error = "Unknown Exception Occurred";
    return 2;
  }
}
