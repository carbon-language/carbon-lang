//===-- llvm/CodeGen/MachineFunction.h --------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Collect native machine code for a function.  This class contains a list of
// MachineBasicBlock instances that make up the current compiled function.
//
// This class also contains pointers to various classes which hold
// target-specific information about the generated code.
//   
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEFUNCTION_H
#define LLVM_CODEGEN_MACHINEFUNCTION_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "Support/Annotation.h"

namespace llvm {

// ilist_traits
template <>
class ilist_traits<MachineBasicBlock> {
  // this is only set by the MachineFunction owning the ilist
  friend class MachineFunction;
  MachineFunction* Parent;
  
public:
  ilist_traits<MachineBasicBlock>() : Parent(0) { }
  
  static MachineBasicBlock* getPrev(MachineBasicBlock* N) { return N->Prev; }
  static MachineBasicBlock* getNext(MachineBasicBlock* N) { return N->Next; }
  
  static const MachineBasicBlock*
  getPrev(const MachineBasicBlock* N) { return N->Prev; }
  
  static const MachineBasicBlock*
  getNext(const MachineBasicBlock* N) { return N->Next; }
  
  static void setPrev(MachineBasicBlock* N, MachineBasicBlock* prev) { N->Prev = prev; }
  static void setNext(MachineBasicBlock* N, MachineBasicBlock* next) { N->Next = next; }
  
  static MachineBasicBlock* createNode();
  void addNodeToList(MachineBasicBlock* N);
  void removeNodeFromList(MachineBasicBlock* N);
  void transferNodesFromList(
			     iplist<MachineBasicBlock, ilist_traits<MachineBasicBlock> >& toList,
			     ilist_iterator<MachineBasicBlock> first,
			     ilist_iterator<MachineBasicBlock> last);
};
  


class Function;
class TargetMachine;
class SSARegMap;
class MachineFunctionInfo;
class MachineFrameInfo;
class MachineConstantPool;

class MachineFunction : private Annotation {
  const Function *Fn;
  const TargetMachine &Target;

  // List of machine basic blocks in function
  ilist<MachineBasicBlock> BasicBlocks;

  // Keeping track of mapping from SSA values to registers
  SSARegMap *SSARegMapping;

  // Used to keep track of frame and constant area information for sparc be
  MachineFunctionInfo *MFInfo;

  // Keep track of objects allocated on the stack.
  MachineFrameInfo *FrameInfo;

  // Keep track of constants which are spilled to memory
  MachineConstantPool *ConstantPool;

  // Function-level unique numbering for MachineBasicBlocks
  int NextMBBNumber;

public:
  MachineFunction(const Function *Fn, const TargetMachine &TM);
  ~MachineFunction();

  /// getFunction - Return the LLVM function that this machine code represents
  ///
  const Function *getFunction() const { return Fn; }

  /// getTarget - Return the target machine this machine code is compiled with
  ///
  const TargetMachine &getTarget() const { return Target; }

  /// SSARegMap Interface... Keep track of information about each SSA virtual
  /// register, such as which register class it belongs to.
  ///
  SSARegMap *getSSARegMap() const { return SSARegMapping; }
  void clearSSARegMap();

  /// getFrameInfo - Return the frame info object for the current function.
  /// This object contains information about objects allocated on the stack
  /// frame of the current function in an abstract way.
  ///
  MachineFrameInfo *getFrameInfo() const { return FrameInfo; }

  /// getConstantPool - Return the constant pool object for the current
  /// function.
  MachineConstantPool *getConstantPool() const { return ConstantPool; }

  /// MachineFunctionInfo - Keep track of various per-function pieces of
  /// information for the sparc backend.
  ///
  MachineFunctionInfo *getInfo() const { return MFInfo; }

  /// getNextMBBNumber - Returns the next unique number to be assigned
  /// to a MachineBasicBlock in this MachineFunction.
  ///
  int getNextMBBNumber() { return NextMBBNumber++; }

  /// print - Print out the MachineFunction in a format suitable for debugging
  /// to the specified stream.
  ///
  void print(std::ostream &OS) const;

  /// dump - Print the current MachineFunction to cerr, useful for debugger use.
  ///
  void dump() const;

  /// construct - Allocate and initialize a MachineFunction for a given Function
  /// and Target
  ///
  static MachineFunction& construct(const Function *F, const TargetMachine &TM);

  /// destruct - Destroy the MachineFunction corresponding to a given Function
  ///
  static void destruct(const Function *F);

  /// get - Return a handle to a MachineFunction corresponding to the given
  /// Function.  This should not be called before "construct()" for a given
  /// Function.
  ///
  static MachineFunction& get(const Function *F);

  // Provide accessors for the MachineBasicBlock list...
  typedef ilist<MachineBasicBlock> BasicBlockListType;
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  // Provide accessors for basic blocks...
  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }
 
  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  unsigned                  size() const { return BasicBlocks.size(); }
  bool                     empty() const { return BasicBlocks.empty(); }
  const MachineBasicBlock &front() const { return BasicBlocks.front(); }
        MachineBasicBlock &front()       { return BasicBlocks.front(); }
  const MachineBasicBlock & back() const { return BasicBlocks.back(); }
        MachineBasicBlock & back()       { return BasicBlocks.back(); }
};

} // End llvm namespace

#endif
