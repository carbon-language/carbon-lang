//===-- llvm/CodeGen/MBasicBlock.h - Machine Specific BB rep ----*- C++ -*-===//
//
// This class provides a way to represent a basic block in a machine-specific
// form.  A basic block is represented as a list of machine specific
// instructions.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_MBASICBLOCK_H
#define CODEGEN_MBASICBLOCK_H

#include "llvm/CodeGen/MInstruction.h"
#include "Support/ilist"

class MBasicBlock {
  MBasicBlock *Prev, *Next;
  iplist<MInstruction> InstList;
  // FIXME: we should maintain a pointer to the function we are embedded into!
public:
  MBasicBlock() {}

  // Provide accessors for the MBasicBlock list...
  typedef iplist<MInstruction> InstListType;
  typedef InstListType::iterator iterator;
  typedef InstListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  //===--------------------------------------------------------------------===//
  /// Instruction iterator methods
  ///
  inline iterator                begin()       { return InstList.begin(); }
  inline const_iterator          begin() const { return InstList.begin(); }
  inline iterator                end  ()       { return InstList.end();   }
  inline const_iterator          end  () const { return InstList.end();   }

  inline reverse_iterator       rbegin()       { return InstList.rbegin(); }
  inline const_reverse_iterator rbegin() const { return InstList.rbegin(); }
  inline reverse_iterator       rend  ()       { return InstList.rend();   }
  inline const_reverse_iterator rend  () const { return InstList.rend();   }

  inline unsigned                 size() const { return InstList.size(); }
  inline bool                    empty() const { return InstList.empty(); }
  inline const MInstruction     &front() const { return InstList.front(); }
  inline       MInstruction     &front()       { return InstList.front(); }
  inline const MInstruction      &back() const { return InstList.back(); }
  inline       MInstruction      &back()       { return InstList.back(); }

  /// getInstList() - Return the underlying instruction list container.  You
  /// need to access it directly if you want to modify it currently.
  ///
  const InstListType &getInstList() const { return InstList; }
        InstListType &getInstList()       { return InstList; }

private:   // Methods used to maintain doubly linked list of blocks...
  friend class ilist_traits<MBasicBlock>;

  MBasicBlock *getPrev() const { return Prev; }
  MBasicBlock *getNext() const { return Next; }
  void setPrev(MBasicBlock *P) { Prev = P; }
  void setNext(MBasicBlock *N) { Next = N; }
};

#endif
