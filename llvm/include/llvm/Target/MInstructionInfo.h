//===- Target/MInstructionInfo.h - Target Instruction Information -*-C++-*-===//
//
// MInstruction's are completely generic instructions that provide very little
// interpretation upon their arguments and sementics.  This file defines an
// interface that should be used to get information about the semantics of the
// actual instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MINSTRUCTIONINFO_H
#define LLVM_CODEGEN_MINSTRUCTIONINFO_H

#include <assert.h>
#include <iosfwd>
class MInstruction;
class MRegisterInfo;

/// MInstructionDesc - This record contains all of the information known about a
/// particular instruction.  Note that several instructions with the same
/// mnemonic may be represented in the target machine as different instructions.
///
struct MInstructionDesc {
  const char *Name;     // Assembly language mnemonic for the instruction.
  unsigned   Flags;    // Flags identifying inst properties (defined below)
  unsigned TSFlags;    // Target Specific Flags
};

/// MIF namespace - This namespace contains flags that pertain to machine
/// instructions
///
namespace MIF {
  enum {
    // Memory flags...
    LOAD               = 1 << 0,   // This instruction loads from memory
    STORE              = 1 << 1,   // This instruction stores to memory

    // Control flow flags...
    CALL               = 1 << 2,   // This instruction calls another function
    RET                = 1 << 3,   // This instruction returns from function
    BRANCH             = 1 << 4,   // This instruction is a branch
  };
};

/// MInstructionInfo base class - We assume that the target defines a static
/// array of MInstructionDesc objects that represent all of the machine
/// instructions that the target has.  As such, we simply have to track a
/// pointer to this array so that we can turn an instruction opcode into an
/// instruction descriptor.
///
class MInstructionInfo {
  const MInstructionDesc *Desc;    // Pointer to the descriptor array
  unsigned NumInstructions;        // Number of entries in the array
protected:
  MInstructionInfo(const MInstructionDesc *D, unsigned NI)
    : Desc(D), NumInstructions(NI) {}
public:

  enum {                           // Target independant constants
    PHIOpcode = 0,                 /// Opcode for PHI instruction
    NoOpOpcode = 1,                /// Opcode for noop instruction
  };

  /// getRegisterInfo - MInstructionInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const = 0;

  const MInstructionDesc &operator[](unsigned Opcode) const {
    assert(Opcode < NumInstructions &&
           "Attempting to access record for invalid opcode!");
    return Desc[Opcode];
  }

  /// Provide a get method, equivalent to [], but more useful if we have a
  /// pointer to this object.
  const MInstructionDesc &get(unsigned Opcode) const {
    return operator[](Opcode);
  }

  virtual void print(const MInstruction *MI, std::ostream &O) const = 0;

};

#endif
