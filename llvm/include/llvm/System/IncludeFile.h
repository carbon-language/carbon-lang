//===- llvm/Support/IncludeFile.h - Ensure Linking Of Library ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IncludeFile class.
//
//===----------------------------------------------------------------------===//

/// This class is used as a facility to make sure that the implementation of a 
/// header file is included into a tool that uses the header.  This is solely 
/// to overcome problems linking .a files and not getting the implementation 
/// of compilation units we need. This is commonly an issue with the various
/// Passes but also occurs elsewhere in LLVM. We like to use .a files because
/// they link faster and provide the smallest executables. However, sometimes
/// those executables are too small, if the program doesn't reference something
/// that might be needed, especially by a loaded share object. This little class
/// helps to resolve that problem. The basic strategy is to use this class in
/// a header file and pass the address of a variable to the constructor. If the
/// variable is defined in the header file's corresponding .cpp file then all
/// tools/libraries that #include the header file will require the .cpp as well.
/// For example:<br/>
/// <tt>extern int LinkMyCodeStub;</tt><br/>
/// <tt>static IncludeFile LinkMyModule(&LinkMyCodeStub);</tt><br/>
/// @brief Class to ensure linking of corresponding object file.

#ifndef LLVM_SUPPORT_INCLUDEFILE_H
#define LLVM_SUPPORT_INCLUDEFILE_H

namespace llvm {
struct IncludeFile {
  IncludeFile(void *);
};
}

#endif
