//===- Win32/SysConfig.cpp - Win32 System Configuration ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some functions for managing system configuration on Win32.
//
//===----------------------------------------------------------------------===//

namespace llvm {

// Some LLVM programs such as bugpoint produce core files as a normal part of
// their operation. To prevent the disk from filling up, this configuration item
// does what's necessary to prevent their generation.
void sys::PreventCoreFiles() {
  // Windows doesn't do core files, so nothing to do.
  // Although...  it might be nice to prevent the do-you-want-to-debug
  // dialog box from coming up.  Or maybe not...
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
