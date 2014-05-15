//===- ThreadSafetyTIL.cpp -------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT in the llvm repository for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Analysis/Analyses/ThreadSafetyTraverse.h"

namespace clang {
namespace threadSafety {
namespace til {


StringRef getUnaryOpcodeString(TIL_UnaryOpcode Op) {
  switch (Op) {
    case UOP_Minus:    return "-";
    case UOP_BitNot:   return "~";
    case UOP_LogicNot: return "!";
  }
  return "";
}


StringRef getBinaryOpcodeString(TIL_BinaryOpcode Op) {
  switch (Op) {
    case BOP_Mul:      return "*";
    case BOP_Div:      return "/";
    case BOP_Rem:      return "%";
    case BOP_Add:      return "+";
    case BOP_Sub:      return "-";
    case BOP_Shl:      return "<<";
    case BOP_Shr:      return ">>";
    case BOP_BitAnd:   return "&";
    case BOP_BitXor:   return "^";
    case BOP_BitOr:    return "|";
    case BOP_Eq:       return "==";
    case BOP_Neq:      return "!=";
    case BOP_Lt:       return "<";
    case BOP_Leq:      return "<=";
    case BOP_LogicAnd: return "&&";
    case BOP_LogicOr:  return "||";
  }
  return "";
}



// If E is a variable, then trace back through any aliases or redundant
// Phi nodes to find the canonical definition.
SExpr *getCanonicalVal(SExpr *E) {
  while (auto *V = dyn_cast<Variable>(E)) {
    SExpr *D;
    do {
      if (V->kind() != Variable::VK_Let)
        return V;
      D = V->definition();
      auto *V2 = dyn_cast<Variable>(D);
      if (V2)
        V = V2;
      else
        break;
    } while (true);

    if (ThreadSafetyTIL::isTrivial(D))
      return D;

    if (Phi *Ph = dyn_cast<Phi>(D)) {
      if (Ph->status() == Phi::PH_Incomplete)
        simplifyIncompleteArg(V, Ph);

      if (Ph->status() == Phi::PH_SingleVal) {
        E = Ph->values()[0];
        continue;
      }
    }
    return V;
  }
  return E;
}


// Trace the arguments of an incomplete Phi node to see if they have the same
// canonical definition.  If so, mark the Phi node as redundant.
// getCanonicalVal() will recursively call simplifyIncompletePhi().
void simplifyIncompleteArg(Variable *V, til::Phi *Ph) {
  assert(Ph && Ph->status() == Phi::PH_Incomplete);

  // eliminate infinite recursion -- assume that this node is not redundant.
  Ph->setStatus(Phi::PH_MultiVal);

  SExpr *E0 = getCanonicalVal(Ph->values()[0]);
  for (unsigned i=1, n=Ph->values().size(); i<n; ++i) {
    SExpr *Ei = getCanonicalVal(Ph->values()[i]);
    if (Ei == V)
      continue;  // Recursive reference to itself.  Don't count.
    if (Ei != E0) {
      return;    // Status is already set to MultiVal.
    }
  }
  Ph->setStatus(Phi::PH_SingleVal);
  // Eliminate Redundant Phi node.
  V->setDefinition(Ph->values()[0]);
}


}  // end namespace til
}  // end namespace threadSafety
}  // end namespace clang

