//===----- CGCall.h - Encapsulate calling convention details ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGCALL_H
#define CLANG_CODEGEN_CGCALL_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Value.h"
#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"

#include "CGValue.h"

// FIXME: Restructure so we don't have to expose so much stuff.
#include "ABIInfo.h"

namespace llvm {
  struct AttributeWithIndex;
  class Function;
  class Type;
  class Value;

  template<typename T, unsigned> class SmallVector;
}

namespace clang {
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class ObjCMethodDecl;
  class VarDecl;

namespace CodeGen {
  typedef llvm::SmallVector<llvm::AttributeWithIndex, 8> AttributeListType;

  struct CallArg {
    RValue RV;
    QualType Ty;
    bool NeedsCopy;
    CallArg(RValue rv, QualType ty, bool needscopy)
    : RV(rv), Ty(ty), NeedsCopy(needscopy)
    { }
  };

  /// CallArgList - Type for representing both the value and type of
  /// arguments in a call.
  class CallArgList :
    public llvm::SmallVector<CallArg, 16> {
  public:
    void add(RValue rvalue, QualType type, bool needscopy = false) {
      push_back(CallArg(rvalue, type, needscopy));
    }
  };

  /// FunctionArgList - Type for representing both the decl and type
  /// of parameters to a function. The decl must be either a
  /// ParmVarDecl or ImplicitParamDecl.
  class FunctionArgList : public llvm::SmallVector<const VarDecl*, 16> {
  };

  /// CGFunctionInfo - Class to encapsulate the information about a
  /// function definition.
  class CGFunctionInfo : public llvm::FoldingSetNode {
    struct ArgInfo {
      CanQualType type;
      ABIArgInfo info;
    };

    /// The LLVM::CallingConv to use for this function (as specified by the
    /// user).
    unsigned CallingConvention;

    /// The LLVM::CallingConv to actually use for this function, which may
    /// depend on the ABI.
    unsigned EffectiveCallingConvention;

    /// Whether this function is noreturn.
    bool NoReturn;

    unsigned NumArgs;
    ArgInfo *Args;

    /// How many arguments to pass inreg.
    bool HasRegParm;
    unsigned RegParm;

  public:
    typedef const ArgInfo *const_arg_iterator;
    typedef ArgInfo *arg_iterator;

    CGFunctionInfo(unsigned CallingConvention, bool NoReturn,
                   bool HasRegParm, unsigned RegParm, CanQualType ResTy,
                   const CanQualType *ArgTys, unsigned NumArgTys);
    ~CGFunctionInfo() { delete[] Args; }

    const_arg_iterator arg_begin() const { return Args + 1; }
    const_arg_iterator arg_end() const { return Args + 1 + NumArgs; }
    arg_iterator arg_begin() { return Args + 1; }
    arg_iterator arg_end() { return Args + 1 + NumArgs; }

    unsigned  arg_size() const { return NumArgs; }

    bool isNoReturn() const { return NoReturn; }

    /// getCallingConvention - Return the user specified calling
    /// convention.
    unsigned getCallingConvention() const { return CallingConvention; }

    /// getEffectiveCallingConvention - Return the actual calling convention to
    /// use, which may depend on the ABI.
    unsigned getEffectiveCallingConvention() const {
      return EffectiveCallingConvention;
    }
    void setEffectiveCallingConvention(unsigned Value) {
      EffectiveCallingConvention = Value;
    }

    bool getHasRegParm() const { return HasRegParm; }
    unsigned getRegParm() const { return RegParm; }

    CanQualType getReturnType() const { return Args[0].type; }

    ABIArgInfo &getReturnInfo() { return Args[0].info; }
    const ABIArgInfo &getReturnInfo() const { return Args[0].info; }

    void Profile(llvm::FoldingSetNodeID &ID) {
      ID.AddInteger(getCallingConvention());
      ID.AddBoolean(NoReturn);
      ID.AddBoolean(HasRegParm);
      ID.AddInteger(RegParm);
      getReturnType().Profile(ID);
      for (arg_iterator it = arg_begin(), ie = arg_end(); it != ie; ++it)
        it->type.Profile(ID);
    }
    template<class Iterator>
    static void Profile(llvm::FoldingSetNodeID &ID,
                        const FunctionType::ExtInfo &Info,
                        CanQualType ResTy,
                        Iterator begin,
                        Iterator end) {
      ID.AddInteger(Info.getCC());
      ID.AddBoolean(Info.getNoReturn());
      ID.AddBoolean(Info.getHasRegParm());
      ID.AddInteger(Info.getRegParm());
      ResTy.Profile(ID);
      for (; begin != end; ++begin) {
        CanQualType T = *begin; // force iterator to be over canonical types
        T.Profile(ID);
      }
    }
  };
  
  /// ReturnValueSlot - Contains the address where the return value of a 
  /// function can be stored, and whether the address is volatile or not.
  class ReturnValueSlot {
    llvm::PointerIntPair<llvm::Value *, 1, bool> Value;

  public:
    ReturnValueSlot() {}
    ReturnValueSlot(llvm::Value *Value, bool IsVolatile)
      : Value(Value, IsVolatile) {}

    bool isNull() const { return !getValue(); }
    
    bool isVolatile() const { return Value.getInt(); }
    llvm::Value *getValue() const { return Value.getPointer(); }
  };
  
}  // end namespace CodeGen
}  // end namespace clang

#endif
