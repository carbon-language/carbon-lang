//===-- llvm/Constant.h - Constant class definition --------------*- C++ -*--=//
//
// This file contains the declaration of the Constant class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANT_H
#define LLVM_CONSTANT_H

#include "llvm/User.h"

class Constant : public User {
protected:
  inline Constant(const Type *Ty) : User(Ty, Value::ConstantVal) {}
  ~Constant() {}

  void destroyConstantImpl();
public:
  /// setName - Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  /// Static constructor to get a '0' constant of arbitrary type...
  static Constant *getNullValue(const Type *Ty);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const = 0;

  virtual void print(std::ostream &O) const;

  /// isConstantExpr - Return true if this is a ConstantExpr
  virtual bool isConstantExpr() const { return false; }


  /// destroyConstant - Called if some element of this constant is no longer
  /// valid.  At this point only other constants may be on the use_list for this
  /// constant.  Any constants on our Use list must also be destroy'd.  The
  /// implementation must be sure to remove the constant from the list of
  /// available cached constants.  Implementations should call
  /// destroyConstantImpl as the last thing they do, to destroy all users and
  /// delete this.
  ///
  /// Note that this call is only valid on non-primitive constants: You cannot
  /// destroy an integer constant for example.  This API is used to delete
  /// constants that have ConstantPointerRef's embeded in them when the module
  /// is deleted, and it is used by GlobalDCE to remove ConstantPointerRefs that
  /// are unneeded, allowing globals to be DCE'd.
  ///
  virtual void destroyConstant() { assert(0 && "Not reached!"); }

  
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Constant *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::ConstantVal;
  }

  // WARNING: Only to be used by Bytecode & Assembly Parsers!  USER CODE SHOULD
  // NOT USE THIS!!
  // Returns the number of uses of OldV that were replaced.
  virtual unsigned mutateReferences(Value* OldV, Value *NewV) { return 0; }
  // END WARNING!!
};

#endif
