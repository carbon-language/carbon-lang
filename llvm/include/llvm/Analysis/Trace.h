//===- llvm/Analysis/Trace.h - Represent one trace of LLVM code -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents a single trace of LLVM basic blocks.  A trace is a
// single entry, multiple exit, region of code that is often hot.  Trace-based
// optimizations treat traces almost like they are a large, strange, basic
// block: because the trace path is assumed to be hot, optimizations for the
// fall-through path are made at the expense of the non-fall-through paths.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TRACE_H
#define LLVM_ANALYSIS_TRACE_H

#include <iosfwd>
#include <vector>

namespace llvm { 
  class BasicBlock;
  class Function;
  class Module;

class Trace {
  typedef std::vector<BasicBlock *> BasicBlockListType;
  BasicBlockListType BasicBlocks;

public:
  /// contains - Returns true if this trace contains the given basic
  /// block.
  ///
  inline bool contains (const BasicBlock *X) {
    for (unsigned i = 0, e = BasicBlocks.size(); i != e; ++i)
      if (BasicBlocks[i] == X)
        return true;
    return false;
  }

  /// Trace ctor - Make a new trace from a vector of basic blocks,
  /// residing in the function which is the parent of the first
  /// basic block in the vector.
  ///
  Trace (const std::vector<BasicBlock *> &vBB) : BasicBlocks (vBB) {
  }

  /// getEntryBasicBlock - Return the entry basic block (first block)
  /// of the trace.
  ///
  BasicBlock *getEntryBasicBlock () const { return BasicBlocks[0]; }

  /// getFunction - Return this trace's parent function.
  ///
  Function *getFunction () const;

  /// getModule - Return this Module that contains this trace's parent
  /// function.
  ///
  Module *getModule () const;

  /// print - Write trace to output stream.
  ///
  void print (std::ostream &O) const;

  /// dump - Debugger convenience method; writes trace to standard error
  /// output stream.
  ///
  void dump () const;

  // BasicBlock iterators...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;

  iterator                begin()       { return BasicBlocks.begin(); }
  const_iterator          begin() const { return BasicBlocks.begin(); }
  iterator                end  ()       { return BasicBlocks.end();   }
  const_iterator          end  () const { return BasicBlocks.end();   }

  reverse_iterator       rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator       rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator rend  () const { return BasicBlocks.rend();   }

  unsigned                 size() const { return BasicBlocks.size(); }
  bool                    empty() const { return BasicBlocks.empty(); }

  BasicBlock *operator[] (unsigned i) const { return BasicBlocks[i]; }
  BasicBlock *getBlock (unsigned i)   const { return BasicBlocks[i]; }

  /// Returns true if B1 and B2 appear on a path from START to an exit
  /// block => B1 appears before B2. If START is not provided, defaults
  /// to 0, which means use getEntryBasicBlock().
  ///
  bool dominates (const BasicBlock *B1, const BasicBlock *B2,
		  const BasicBlock *start = 0);
};

} // end namespace llvm

#endif // TRACE_H
