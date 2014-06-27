//===-- llvm/GlobalObject.h - Class to represent a global object *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This represents an independent object. That is, a function or a global
// variable, but not an alias.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GLOBALOBJECT_H
#define LLVM_IR_GLOBALOBJECT_H

#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"

namespace llvm {
class Comdat;
class Module;

class GlobalObject : public GlobalValue {
  GlobalObject(const GlobalObject &) LLVM_DELETED_FUNCTION;

protected:
  GlobalObject(Type *Ty, ValueTy VTy, Use *Ops, unsigned NumOps,
               LinkageTypes Linkage, const Twine &Name)
      : GlobalValue(Ty, VTy, Ops, NumOps, Linkage, Name), ObjComdat(nullptr) {
    setGlobalValueSubClassData(0);
  }

  std::string Section;     // Section to emit this into, empty means default
  Comdat *ObjComdat;
public:
  unsigned getAlignment() const {
    return (1u << getGlobalValueSubClassData()) >> 1;
  }
  void setAlignment(unsigned Align);

  bool hasSection() const { return !StringRef(getSection()).empty(); }
  const char *getSection() const { return Section.c_str(); }
  void setSection(StringRef S);

  bool hasComdat() const { return getComdat() != nullptr; }
  const Comdat *getComdat() const { return ObjComdat; }
  Comdat *getComdat() { return ObjComdat; }
  void setComdat(Comdat *C) { ObjComdat = C; }

  void copyAttributesFrom(const GlobalValue *Src) override;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::FunctionVal ||
           V->getValueID() == Value::GlobalVariableVal;
  }
};

} // End llvm namespace

#endif
