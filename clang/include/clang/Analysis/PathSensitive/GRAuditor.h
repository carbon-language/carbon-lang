//==- GRAuditor.h - Observers of the creation of ExplodedNodes------*- C++ -*-//
//             
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRAuditor and its primary subclasses, an interface
//  to audit the creation of ExplodedNodes.  This interface can be used
//  to implement simple checkers that do not mutate analysis state but
//  instead operate by perfoming simple logical checks at key monitoring
//  locations (e.g., function calls).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRAUDITOR
#define LLVM_CLANG_ANALYSIS_GRAUDITOR

namespace clang {

class ExplodedNode;
class GRStateManager;
  
class GRAuditor {
public:
  virtual ~GRAuditor() {}
  virtual bool Audit(ExplodedNode* N, GRStateManager& M) = 0;
};
  
  
} // end clang namespace

#endif
