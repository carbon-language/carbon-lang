//===- Win32/Signals.cpp - Win32 Signals Implementation ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Signals class.
//
//===----------------------------------------------------------------------===//

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be generic UNIX code (see ../Unix/Signals.cpp)
//===----------------------------------------------------------------------===//

// RemoveFileOnSignal - The public API
void llvm::RemoveFileOnSignal(const std::string &Filename) {
}

// RemoveDirectoryOnSignal - The public API
void llvm::RemoveDirectoryOnSignal(const llvm::sys::Path& path) {
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void llvm::PrintStackTraceOnErrorSignal() {
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
