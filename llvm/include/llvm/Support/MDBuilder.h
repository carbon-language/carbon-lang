//===---- llvm/Support/MDBuilder.h - Builder for LLVM metadata --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MDBuilder class, which is used as a convenient way to
// create LLVM metadata with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MDBUILDER_H
#define LLVM_SUPPORT_MDBUILDER_H

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/APInt.h"

namespace llvm {

  class MDBuilder {
    LLVMContext &Context;

    MDString *getFastString() {
      return createString("fast");
    }
  public:
    MDBuilder(LLVMContext &context) : Context(context) {}

    /// \brief Return the given string as metadata.
    MDString *createString(StringRef Str) {
      return MDString::get(Context, Str);
    }

    //===------------------------------------------------------------------===//
    // FPMath metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata with appropriate settings for 'fast math'.
    MDNode *createFastFPMath() {
      return MDNode::get(Context, getFastString());
    }

    /// \brief Return metadata with the given settings.  Special values for the
    /// Accuracy parameter are 0.0, which means the default (maximal precision)
    /// setting; and negative values which all mean 'fast'.
    MDNode *createFPMath(float Accuracy) {
      if (Accuracy == 0.0)
        return 0;
      if (Accuracy < 0.0)
        return MDNode::get(Context, getFastString());
      assert(Accuracy > 0.0 && "Invalid fpmath accuracy!");
      Value *Op = ConstantFP::get(Type::getFloatTy(Context), Accuracy);
      return MDNode::get(Context, Op);
    }


    //===------------------------------------------------------------------===//
    // Range metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata describing the range [Lo, Hi).
    MDNode *createRange(const APInt &Lo, const APInt &Hi) {
      assert(Lo.getBitWidth() == Hi.getBitWidth() && "Mismatched bitwidths!");
      // If the range is everything then it is useless.
      if (Hi == Lo)
        return 0;

      // Return the range [Lo, Hi).
      Type *Ty = IntegerType::get(Context, Lo.getBitWidth());
      Value *Range[2] = { ConstantInt::get(Ty, Lo), ConstantInt::get(Ty, Hi) };
      return MDNode::get(Context, Range);
    }


    //===------------------------------------------------------------------===//
    // TBAA metadata.
    //===------------------------------------------------------------------===//

    /// \brief Return metadata appropriate for a TBAA root node.  Each returned
    /// node is distinct from all other metadata and will never be identified
    /// (uniqued) with anything else.
    MDNode *createAnonymousTBAARoot() {
      // To ensure uniqueness the root node is self-referential.
      MDNode *Dummy = MDNode::getTemporary(Context, ArrayRef<Value*>());
      MDNode *Root = MDNode::get(Context, Dummy);
      // At this point we have
      //   !0 = metadata !{}            <- dummy
      //   !1 = metadata !{metadata !0} <- root
      // Replace the dummy operand with the root node itself and delete the dummy.
      Root->replaceOperandWith(0, Root);
      MDNode::deleteTemporary(Dummy);
      // We now have
      //   !1 = metadata !{metadata !1} <- self-referential root
      return Root;
    }

    /// \brief Return metadata appropriate for a TBAA root node with the given
    /// name.  This may be identified (uniqued) with other roots with the same
    /// name.
    MDNode *createTBAARoot(StringRef Name) {
      return MDNode::get(Context, createString(Name));
    }

    /// \brief Return metadata for a non-root TBAA node with the given name,
    /// parent in the TBAA tree, and value for 'pointsToConstantMemory'.
    MDNode *createTBAANode(StringRef Name, MDNode *Parent,
                           bool isConstant = false) {
      if (isConstant) {
        Constant *Flags = ConstantInt::get(Type::getInt64Ty(Context), 1);
        Value *Ops[3] = { createString(Name), Parent, Flags };
        return MDNode::get(Context, Ops);
      } else {
        Value *Ops[2] = { createString(Name), Parent };
        return MDNode::get(Context, Ops);
      }
    }

  };

} // end namespace llvm

#endif
