//===- llvm/System/IncludeFile.h - Ensure Linking Of Library ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the FORCE_DEFINING_FILE_TO_BE_LINKED and DEFINE_FILE_FOR
// macros.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_INCLUDEFILE_H
#define LLVM_SYSTEM_INCLUDEFILE_H

/// This macro is the public interface that IncludeFile.h exports. This gives
/// us the option to implement the "link the definition" capability in any 
/// manner that we choose. All header files that depend on a specific .cpp
/// file being linked at run time should use this macro instead of the
/// IncludeFile class directly. 
/// 
/// For example, foo.h would use:<br/>
/// <tt>FORCE_DEFINING_FILE_TO_BE_LINKED(foo)</tt><br/>
/// 
/// And, foo.cp would use:<br/>
/// <tt>DEFINING_FILE_FOR(foo)</tt><br/>
#define FORCE_DEFINING_FILE_TO_BE_LINKED(name) \
  namespace llvm { \
    extern char name ## LinkVar; \
    static IncludeFile name ## LinkObj ( &name ## LinkVar ); \
  } 

/// This macro is the counterpart to FORCE_DEFINING_FILE_TO_BE_LINKED. It should
/// be used in a .cpp file to define the name referenced in a header file that
/// will cause linkage of the .cpp file. It should only be used at extern level.
#define DEFINING_FILE_FOR(name) namespace llvm { char name ## LinkVar; }

namespace llvm {

/// This class is used in the implementation of FORCE_DEFINING_FILE_TO_BE_LINKED
/// macro to make sure that the implementation of a header file is included 
/// into a tool that uses the header.  This is solely 
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
struct IncludeFile {
  IncludeFile(void *);
};

}

#endif
