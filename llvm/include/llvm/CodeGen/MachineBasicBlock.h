//===-- llvm/CodeGen/MachineBasicBlock.h ------------------------*- C++ -*-===//
// 
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEBASICBLOCK_H
#define LLVM_CODEGEN_MACHINEBASICBLOCK_H

#include "llvm/Annotation.h"
#include <vector>
class BasicBlock;
class MachineInstr;

extern AnnotationID MCFBB_AID;

class MachineBasicBlock : public Annotation {
  std::vector<MachineInstr*> Insts;
public:
  MachineBasicBlock() : Annotation(MCFBB_AID) {}
  ~MachineBasicBlock() {}
  
  // Static methods to retrieve or destroy the MachineBasicBlock
  // object for a given basic block.
  static MachineBasicBlock& get(const BasicBlock *BB) {
    return *(MachineBasicBlock*)
      ((Annotable*)BB)->getOrCreateAnnotation(MCFBB_AID);
  }
  
  static void destroy(const BasicBlock *BB) {
    ((Annotable*)BB)->deleteAnnotation(MCFBB_AID);
  }
  
  typedef std::vector<MachineInstr*>::iterator                iterator;
  typedef std::vector<MachineInstr*>::const_iterator    const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  unsigned size() const { return Insts.size(); }
  bool empty() const { return Insts.empty(); }

  MachineInstr * operator[](unsigned i) const { return Insts[i]; }
  MachineInstr *&operator[](unsigned i)       { return Insts[i]; }

  MachineInstr *front() const { return Insts.front(); }
  MachineInstr *back()  const { return Insts.back(); }

  iterator                begin()       { return Insts.begin();  }
  const_iterator          begin() const { return Insts.begin();  }
  iterator                  end()       { return Insts.end();    }
  const_iterator            end() const { return Insts.end();    }
  reverse_iterator       rbegin()       { return Insts.rbegin(); }
  const_reverse_iterator rbegin() const { return Insts.rbegin(); }
  reverse_iterator       rend  ()       { return Insts.rend();   }
  const_reverse_iterator rend  () const { return Insts.rend();   }

  void push_back(MachineInstr *MI) { Insts.push_back(MI); }
  template<typename IT>
  void insert(iterator I, IT S, IT E) { Insts.insert(I, S, E); }
  iterator insert(iterator I, MachineInstr *M) { return Insts.insert(I, M); }

  // erase - Remove the specified element or range from the instruction list.
  // These functions do not delete any instructions removed.
  //
  iterator erase(iterator I)             { return Insts.erase(I); }
  iterator erase(iterator I, iterator E) { return Insts.erase(I, E); }

  MachineInstr *pop_back() {
    MachineInstr *R = back();
    Insts.pop_back();
    return R;
  }
};


#endif
