//===- llvm/System/Program.h ------------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Program class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_PROGRAM_H
#define LLVM_SYSTEM_PROGRAM_H

#include "llvm/System/Path.h"
#include <vector>

namespace llvm {
namespace sys {

  /// This class provides an abstraction for programs that are executable by the
  /// operating system. It provides a platform generic way to find executable
  /// programs from the path and to execute them. The sys::Path class is used to
  /// locate the Program.
  /// @since 1.4
  /// @brief An abstraction for finding and executing programs.
  class Program {
    /// @name Methods
    /// @{
    public:
      /// This static constructor (factory) will attempt to locate a program in
      /// the operating system's file system using some pre-determined set of 
      /// locations to search (e.g. the PATH on Unix). 
      /// @returns A Path object initialized to the path of the program or a
      /// Path object that is empty (invalid) if the program could not be found.
      /// @throws nothing
      /// @brief Construct a Program by finding it by name.
      static Path FindProgramByName(const std::string& name);

      /// This function executes the program using the \p arguments provided and
      /// waits for the program to exit. This function will block the current
      /// program until the invoked program exits. The invoked program will 
      /// inherit the stdin, stdout, and stderr file descriptors, the
      /// environment and other configuration settings of the inoking program.
      /// If Path::executable() does not return true when this function is
      /// called then a std::string is thrown. 
      /// Path::executable() returns true.
      /// @param path A sys::Path object providing the path of the program to be
      /// executed. It is presumed this is the result of the FindProgramByName
      /// method.
      /// @param args A vector of strings that are passed to the program.
      /// The first element should *not* be the name of the program.
      /// @returns an integer result code indicating the status of the program.
      /// @throws std::string on a variety of error conditions or if the invoked 
      /// program aborted abnormally.
      /// @see FindProgrambyName
      /// @brief Executes the program with the given set of \p arguments.
      static int ExecuteAndWait(const Path& path, 
                                const std::vector<std::string>& args);
    /// @}
  };
}
}

// vim: sw=2

#endif
