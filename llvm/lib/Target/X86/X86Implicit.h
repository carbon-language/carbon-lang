//===-- X86Implicit.h - All the implicit uses and defs for X86 ops --------===//
//
// This defines a class which maps X86 opcodes to the registers that they
// implicitly modify or use.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include <map>
#include <vector>

class X86Implicit {
public:
  typedef std::map <X86::Opcode, std::vector <X86::Register> > ImplicitMap;
  ImplicitMap implicitUses;
  ImplicitMap implicitDefs;
  X86Implicit ();
};
