//===--- Action.h - Abstract compilation steps ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_ACTION_H_
#define CLANG_DRIVER_ACTION_H_

namespace clang {
namespace driver {

/// Action - Represent an abstract compilation step to perform. 
///
/// An action represents an edge in the compilation graph; typically
/// it is a job to transform an input using some tool.
class Action {
public:
  
};

} // end namespace driver
} // end namespace clang

#endif
