//===-- llvm/CodeGen/MachineBasicBlock.h ------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBASICBLOCK_H
#define LLVM_CODEGEN_MACHINEBASICBLOCK_H

#include "llvm/CodeGen/MachineInstr.h"
#include "Support/GraphTraits.h"
#include "Support/ilist"
#include <iosfwd>

namespace llvm {
  class MachineFunction;

// ilist_traits
template <>
class ilist_traits<MachineInstr> {
  // this is only set by the MachineBasicBlock owning the ilist
  friend class MachineBasicBlock;
  MachineBasicBlock* parent;

public:
  ilist_traits<MachineInstr>() : parent(0) { }

  static MachineInstr* getPrev(MachineInstr* N) { return N->prev; }
  static MachineInstr* getNext(MachineInstr* N) { return N->next; }

  static const MachineInstr*
  getPrev(const MachineInstr* N) { return N->prev; }

  static const MachineInstr*
  getNext(const MachineInstr* N) { return N->next; }

  static void setPrev(MachineInstr* N, MachineInstr* prev) { N->prev = prev; }
  static void setNext(MachineInstr* N, MachineInstr* next) { N->next = next; }

  static MachineInstr* createNode();
  void addNodeToList(MachineInstr* N);
  void removeNodeFromList(MachineInstr* N);
  void transferNodesFromList(
      iplist<MachineInstr, ilist_traits<MachineInstr> >& toList,
      ilist_iterator<MachineInstr> first,
      ilist_iterator<MachineInstr> last);
};

class BasicBlock;

class MachineBasicBlock {
public:
  typedef ilist<MachineInstr> Instructions;
  Instructions Insts;
  MachineBasicBlock *Prev, *Next;
  const BasicBlock *BB;
  std::vector<MachineBasicBlock *> Predecessors;
  std::vector<MachineBasicBlock *> Successors;
  int Number;
  MachineFunction *Parent;

public:
  MachineBasicBlock(const BasicBlock *bb = 0) : Prev(0), Next(0), BB(bb),
                                                Number(-1), Parent(0) {
    Insts.parent = this;
  }
  ~MachineBasicBlock() {}
  
  /// getBasicBlock - Return the LLVM basic block that this instance
  /// corresponded to originally.
  ///
  const BasicBlock *getBasicBlock() const { return BB; }

  /// getParent - Return the MachineFunction containing this basic block.
  ///
  const MachineFunction *getParent() const { return Parent; }
  MachineFunction *getParent() { return Parent; }

  typedef ilist<MachineInstr>::iterator                       iterator;
  typedef ilist<MachineInstr>::const_iterator           const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  unsigned size() const { return Insts.size(); }
  bool empty() const { return Insts.empty(); }

  MachineInstr& front() { return Insts.front(); }
  MachineInstr& back()  { return Insts.back(); }

  iterator                begin()       { return Insts.begin();  }
  const_iterator          begin() const { return Insts.begin();  }
  iterator                  end()       { return Insts.end();    }
  const_iterator            end() const { return Insts.end();    }
  reverse_iterator       rbegin()       { return Insts.rbegin(); }
  const_reverse_iterator rbegin() const { return Insts.rbegin(); }
  reverse_iterator       rend  ()       { return Insts.rend();   }
  const_reverse_iterator rend  () const { return Insts.rend();   }

  // Machine-CFG iterators
  typedef std::vector<MachineBasicBlock *>::iterator       pred_iterator;
  typedef std::vector<MachineBasicBlock *>::const_iterator const_pred_iterator;
  typedef std::vector<MachineBasicBlock *>::iterator       succ_iterator;
  typedef std::vector<MachineBasicBlock *>::const_iterator const_succ_iterator;
  
  pred_iterator        pred_begin()       { return Predecessors.begin (); }
  const_pred_iterator  pred_begin() const { return Predecessors.begin (); }
  pred_iterator        pred_end()         { return Predecessors.end ();   }
  const_pred_iterator  pred_end()   const { return Predecessors.end ();   }
  unsigned             pred_size()  const { return Predecessors.size ();  }
  succ_iterator        succ_begin()       { return Successors.begin ();   }
  const_succ_iterator  succ_begin() const { return Successors.begin ();   }
  succ_iterator        succ_end()         { return Successors.end ();     }
  const_succ_iterator  succ_end()   const { return Successors.end ();     }
  unsigned             succ_size()  const { return Successors.size ();    }

  // Machine-CFG mutators

  /// addSuccessor - Add succ as a successor of this MachineBasicBlock.
  /// The Predecessors list of succ is automatically updated.
  ///
  void addSuccessor (MachineBasicBlock *succ) {
    Successors.push_back (succ);
    succ->addPredecessor (this);
  }

  /// removeSuccessor - Remove succ from the successors list of this
  /// MachineBasicBlock. The Predecessors list of succ is automatically updated.
  ///
  void removeSuccessor (MachineBasicBlock *succ) {
    succ->removePredecessor (this);
    std::vector<MachineBasicBlock *>::iterator goner =
      std::find (Successors.begin(), Successors.end (), succ);
    Successors.erase (goner);
  }

  /// getFirstTerminator - returns an iterator to the first terminator
  /// instruction of this basic block. If a terminator does not exist,
  /// it returns end()
  iterator getFirstTerminator();

  void push_back(MachineInstr *MI) { Insts.push_back(MI); }
  template<typename IT>
  void insert(iterator I, IT S, IT E) { Insts.insert(I, S, E); }
  iterator insert(iterator I, MachineInstr *M) { return Insts.insert(I, M); }

  // erase - Remove the specified element or range from the instruction list.
  // These functions delete any instructions removed.
  //
  iterator erase(iterator I)             { return Insts.erase(I); }
  iterator erase(iterator I, iterator E) { return Insts.erase(I, E); }
  MachineInstr *remove(MachineInstr *I)  { return Insts.remove(I); }
  void clear()                           { Insts.clear(); }

  // Debugging methods.
  void dump() const;
  void print(std::ostream &OS) const;

  /// getNumber - MachineBasicBlocks are uniquely numbered at the function
  /// level, unless they're not in a MachineFunction yet, in which case this
  /// will return -1.
  ///
  int getNumber() const { return Number; }

private:   // Methods used to maintain doubly linked list of blocks...
  friend class ilist_traits<MachineBasicBlock>;

  MachineBasicBlock *getPrev() const { return Prev; }
  MachineBasicBlock *getNext() const { return Next; }
  void setPrev(MachineBasicBlock *P) { Prev = P; }
  void setNext(MachineBasicBlock *N) { Next = N; }

  // Machine-CFG mutators

  /// addPredecessor - Remove pred as a predecessor of this MachineBasicBlock.
  /// Don't do this unless you know what you're doing, because it doesn't
  /// update pred's successors list. Use pred->addSuccessor instead.
  ///
  void addPredecessor (MachineBasicBlock *pred) {
    Predecessors.push_back (pred);
  }

  /// removePredecessor - Remove pred as a predecessor of this
  /// MachineBasicBlock. Don't do this unless you know what you're
  /// doing, because it doesn't update pred's successors list. Use
  /// pred->removeSuccessor instead.
  ///
  void removePredecessor (MachineBasicBlock *pred) {
    std::vector<MachineBasicBlock *>::iterator goner =
      std::find (Predecessors.begin(), Predecessors.end (), pred);
    Predecessors.erase (goner);
  }
};


//===--------------------------------------------------------------------===//
// GraphTraits specializations for machine basic block graphs (machine-CFGs)
//===--------------------------------------------------------------------===//

// Provide specializations of GraphTraits to be able to treat a
// MachineFunction as a graph of MachineBasicBlocks...
//

template <> struct GraphTraits<MachineBasicBlock *> {
  typedef MachineBasicBlock NodeType;
  typedef MachineBasicBlock::succ_iterator ChildIteratorType;

  static NodeType *getEntryNode(MachineBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->succ_end();
  }
};

template <> struct GraphTraits<const MachineBasicBlock *> {
  typedef const MachineBasicBlock NodeType;
  typedef MachineBasicBlock::const_succ_iterator ChildIteratorType;

  static NodeType *getEntryNode(const MachineBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->succ_end();
  }
};

// Provide specializations of GraphTraits to be able to treat a
// MachineFunction as a graph of MachineBasicBlocks... and to walk it
// in inverse order.  Inverse order for a function is considered
// to be when traversing the predecessor edges of a MBB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<MachineBasicBlock*> > {
  typedef MachineBasicBlock NodeType;
  typedef MachineBasicBlock::pred_iterator ChildIteratorType;
  static NodeType *getEntryNode(Inverse<MachineBasicBlock *> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->pred_end();
  }
};

template <> struct GraphTraits<Inverse<const MachineBasicBlock*> > {
  typedef const MachineBasicBlock NodeType;
  typedef MachineBasicBlock::const_pred_iterator ChildIteratorType;
  static NodeType *getEntryNode(Inverse<const MachineBasicBlock*> G) {
    return G.Graph; 
  }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return N->pred_end();
  }
};

} // End llvm namespace

#endif
