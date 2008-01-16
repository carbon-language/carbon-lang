//===-- GRConstants.h- Simple, Path-Sens. Constant Prop. ---------*- C++ -*-==//
//   
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//               Constant Propagation via Graph Reachability
//
//  This files defines the interface to use the 'GRConstants' path-sensitive
//  constant-propagation analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GRCONSTANTS
#define LLVM_CLANG_GRCONSTANTS

namespace clang {
  
  /// RunGRConstants - This is a simple driver to run the GRConstants analysis
  ///  on a provided CFG.  This interface will eventually be replaced with
  ///  something more elaborate as the requirements on the interface become
  ///  clearer.
  void RunGRConstants(CFG& cfg);
  
} // end clang namespace


#endif
