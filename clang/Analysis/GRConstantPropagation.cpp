//===-- GRConstantPropagation.cpp --------------------------------*- C++ -*-==//
//             
//              [ Constant Propagation via Graph Reachability ]
//   
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines a simple analysis that performs path-sensitive
//  constant propagation within a function.  An example use of this analysis
//  is to perform simple checks for NULL dereferences.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/AST/Expr.h"
#include "clang/AST/CFG.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;
using llvm::APInt;
using llvm::APFloat;
using llvm::dyn_cast;
using llvm::cast;

//===----------------------------------------------------------------------===//
// ConstV - Represents a variant over APInt, APFloat, and const char
//===----------------------------------------------------------------------===//

namespace {
class ConstV {
  uintptr_t Data;
public:
  enum VariantType { VTString = 0x0, VTObjCString = 0x1,
                     VTFloat  = 0x2, VTInt = 0x3,
                     Flags    = 0x3 };
  
  ConstV(const StringLiteral* v) 
    : Data(reinterpret_cast<uintptr_t>(v) | VTString) {}
  
  ConstV(const ObjCStringLiteral* v)
    : Data(reinterpret_cast<uintptr_t>(v) | VTObjCString) {} 
           
  ConstV(llvm::APInt* v)
    : Data(reinterpret_cast<uintptr_t>(v) | VTInt) {}
  
  ConstV(llvm::APFloat* v)
    : Data(reinterpret_cast<uintptr_t>(v) | VTFloat) {}
  

  inline void* getData() const { return (void*) (Data & ~Flags); }
  inline VariantType getVT() const { return (VariantType) (Data & Flags); } 
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(getData());
  }
};
} // end anonymous namespace

// Overload machinery for casting from ConstV to contained classes.

namespace llvm {

#define CV_OBJ_CAST(CLASS,FLAG)\
template<> inline bool isa<CLASS,ConstV>(const ConstV& V) {\
  return V.getVT() == FLAG;\
}\
\
template <> struct cast_retty_impl<CLASS, ConstV> {\
  typedef const CLASS* ret_type;\
};
  
CV_OBJ_CAST(APInt,ConstV::VTInt)
CV_OBJ_CAST(APFloat,ConstV::VTFloat)
CV_OBJ_CAST(StringLiteral,ConstV::VTString)
CV_OBJ_CAST(ObjCStringLiteral,ConstV::VTObjCString)  

#undef CV_OBJ_CAST
  
template <> struct simplify_type<ConstV> {
  typedef void* SimpleType;
  static SimpleType getSimplifiedValue(const ConstV &Val) { 
    return Val.getData();
  }
};

} // end llvm namespace


