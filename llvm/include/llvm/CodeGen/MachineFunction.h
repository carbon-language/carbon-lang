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
#include "Support/ilist"

namespace llvm {

class Function;
class TargetMachine;
class FunctionPass;
class SSARegMap;
class MachineFunctionInfo;
class MachineFrameInfo;
class MachineConstantPool;

FunctionPass *createMachineFunctionPrinterPass();

class MachineFunction : private Annotation {
  const Function *Fn;
  const TargetMachine &Target;

  // List of machine basic blocks in function
  iplist<MachineBasicBlock> BasicBlocks;

  // Keeping track of mapping from SSA values to registers
  SSARegMap *SSARegMapping;

  // Used to keep track of frame and constant area information for sparc be
  MachineFunctionInfo *MFInfo;

  // Keep track of objects allocated on the stack.
  MachineFrameInfo *FrameInfo;

  // Keep track of constants which are spilled to memory
  MachineConstantPool *ConstantPool;

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


  /// print - Print out the MachineFunction in a format suitable for debugging
  /// to the specified stream.
  ///
  void print(std::ostream &OS) const;

  /// dump - Print the current MachineFunction to cerr, useful for debugger use.
  ///
  void dump() const;

  // The next three methods are used to construct, destruct, and retrieve the
  // MachineFunction object for the given method.
  //
  // construct() -- Allocates and initializes for a given method and target
  // get()       -- Returns a handle to the object.
  //                This should not be called before "construct()"
  //                for a given Method.
  // destruct()  -- Destroy the MachineFunction object
  // 
  static MachineFunction& construct(const Function *F, const TargetMachine &TM);
  static void destruct(const Function *F);
  static MachineFunction& get(const Function *F);

  // Provide accessors for the MachineBasicBlock list...
  typedef iplist<MachineBasicBlock> BasicBlockListType;
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
