//===-- SourceFile.cpp - SourceFile implementation for the debugger -------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file implements the SourceFile class for the LLVM debugger.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/SourceFile.h"
#include "Support/SlowOperationInformer.h"
#include "Support/FileUtilities.h"
#include <iostream>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
using namespace llvm;

/// readFile - Load Filename into FileStart and FileEnd.
///
void SourceFile::readFile() {
  ssize_t FileSize = getFileSize(Filename);
  if (FileSize != -1) {
    FDHandle FD(open(Filename.c_str(), O_RDONLY));
    if (FD != -1) {
      char *FilePos = new char[FileSize];
      FileStart = FilePos;

      // If this takes a long time, inform the user what we are doing.
      SlowOperationInformer SOI("loading source file '" + Filename + "'");

      try {
        // Read in the whole buffer.
        unsigned Amount = FileSize;
        while (Amount) {
          unsigned AmountToRead = 512*1024;
          if (Amount < AmountToRead) AmountToRead = Amount;
          ssize_t ReadAmount = read(FD, FilePos, AmountToRead);
          if (ReadAmount < 0 && errno == EINTR)
            continue;
          else if (ReadAmount <= 0) {
            // Couldn't read whole file just free memory and continue.
            throw "Error reading file '" + Filename + "'!";
          }
          Amount -= ReadAmount;
          FilePos += ReadAmount;
          
          SOI.progress(FileSize-Amount, FileSize);
        }

      } catch (const std::string &Msg) {
        std::cout << Msg << "\n";
        // If the user cancels the operation, clean up after ourselves.
        delete [] FileStart;
        FileStart = 0;
        return;
      }
      
      FileEnd = FileStart+FileSize;
    }
  }
}

/// calculateLineOffsets - Compute the LineOffset vector for the current file.
///
void SourceFile::calculateLineOffsets() const {
  assert(LineOffset.empty() && "Line offsets already computed!");
  const char *BufPtr = FileStart;
  do {
    LineOffset.push_back(BufPtr-FileStart);

    // Scan until we get to a newline.
    while (BufPtr != FileEnd && *BufPtr != '\n' && *BufPtr != '\r')
      ++BufPtr;

    if (BufPtr != FileEnd) {
      ++BufPtr;               // Skip over the \n or \r
      if (BufPtr[-1] == '\r' && BufPtr != FileEnd && BufPtr[0] == '\n')
        ++BufPtr;   // Skip over dos/windows style \r\n's
    }
  } while (BufPtr != FileEnd);
}


/// getSourceLine - Given a line number, return the start and end of the line
/// in the file.  If the line number is invalid, or if the file could not be
/// loaded, null pointers are returned for the start and end of the file. Note
/// that line numbers start with 0, not 1.
void SourceFile::getSourceLine(unsigned LineNo, const char *&LineStart,
                               const char *&LineEnd) const {
  LineStart = LineEnd = 0;
  if (FileStart == 0) return;  // Couldn't load file, return null pointers
  if (LineOffset.empty()) calculateLineOffsets();

  // Asking for an out-of-range line number?
  if (LineNo >= LineOffset.size()) return;

  // Otherwise, they are asking for a valid line, which we can fulfill.
  LineStart = FileStart+LineOffset[LineNo];

  if (LineNo+1 < LineOffset.size())
    LineEnd = FileStart+LineOffset[LineNo+1];
  else
    LineEnd = FileEnd;

  // If the line ended with a newline, strip it off.
  while (LineEnd != LineStart && (LineEnd[-1] == '\n' || LineEnd[-1] == '\r'))
    --LineEnd;

  assert(LineEnd >= LineStart && "We somehow got our pointers swizzled!");
}
 
