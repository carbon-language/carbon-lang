//===- fpcmp.cpp - A fuzzy "cmp" that permits floating point noise --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// fpcmp is a tool that basically works like the 'cmp' tool, except that it can
// tolerate errors due to floating point noise, with the -r option.
//
//===----------------------------------------------------------------------===//

#include "Support/CommandLine.h"
#include "Support/FileUtilities.h"
#include <iostream>
#include <cmath>

using namespace llvm;

namespace {
  cl::opt<std::string>
  File1(cl::Positional, cl::desc("<input file #1>"), cl::Required);
  cl::opt<std::string>
  File2(cl::Positional, cl::desc("<input file #2>"), cl::Required);

  cl::opt<double>
  RelTolerance("r", cl::desc("Relative error tolerated"), cl::init(0));
  cl::opt<double>
  AbsTolerance("a", cl::desc("Absolute error tolerated"), cl::init(0));
}


/// OpenFile - mmap the specified file into the address space for reading, and
/// return the length and address of the buffer.
static void OpenFile(const std::string &Filename, unsigned &Len, char* &BufPtr){
  BufPtr = (char*)ReadFileIntoAddressSpace(Filename, Len);
  if (BufPtr == 0) {
    std::cerr << "Error: cannot open file '" << Filename << "'\n";
    exit(2);
  }
}

static bool isNumberChar(char C) {
  switch (C) {
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9': 
  case '.': case '+': case '-':
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

static void CompareNumbers(char *&F1P, char *&F2P, char *F1End, char *F2End) {
  char *F1NumEnd, *F2NumEnd;
  double V1, V2; 
  // If we stop on numbers, compare their difference.
  if (isNumberChar(*F1P) && isNumberChar(*F2P)) {
    V1 = strtod(F1P, &F1NumEnd);
    V2 = strtod(F2P, &F2NumEnd);
  } else {
    // Otherwise, the diff failed.
    F1NumEnd = F1P;
    F2NumEnd = F2P;
  }

  if (F1NumEnd == F1P || F2NumEnd == F2P) {
    std::cerr << "Comparison failed, not a numeric difference.\n";
    exit(1);
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
      std::cerr << "Compared: " << V1 << " and " << V2 << ": diff = "
                << Diff << "\n";
      std::cerr << "Out of tolerance: rel/abs: " << RelTolerance
                << "/" << AbsTolerance << "\n";
      exit(1);
    }
  }

  // Otherwise, advance our read pointers to the end of the numbers.
  F1P = F1NumEnd;  F2P = F2NumEnd;
}


int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  // mmap in the files.
  unsigned File1Len, File2Len;
  char *File1Start, *File2Start;
  OpenFile(File1, File1Len, File1Start);
  OpenFile(File2, File2Len, File2Start);

  // Okay, now that we opened the files, scan them for the first difference.
  char *File1End = File1Start+File1Len;
  char *File2End = File2Start+File2Len;
  char *F1P = File1Start;
  char *F2P = File2Start;
  
  while (1) {
    // Scan for the end of file or first difference.
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
    CompareNumbers(F1P, F2P, File1End, File2End);
  }

  // Okay, we reached the end of file.  If both files are at the end, we
  // succeeded.
  if (F1P >= File1End && F2P >= File2End) return 0;

  // Otherwise, we might have run off the end due to a number, backup and retry.
  F1P = BackupNumber(F1P, File1Start);
  F2P = BackupNumber(F2P, File2Start);

  // Now that we are at the start of the numbers, compare them, exiting if
  // they don't match.
  CompareNumbers(F1P, F2P, File1End, File2End);

  // If we found the end, we succeeded.
  if (F1P >= File1End && F2P >= File2End) return 0;

  return 1;
}

