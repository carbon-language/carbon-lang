//===-- Relooper.h - Top-level interface for WebAssembly  ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//
///
/// \file
/// \brief This defines an optimized C++ implemention of the Relooper
/// algorithm, originally developed as part of Emscripten, which
/// generates a structured AST from arbitrary control flow.
///
//===-------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <set>

namespace llvm {

namespace Relooper {

struct Block;
struct Shape;

///
/// Info about a branching from one block to another
///
struct Branch {
  enum FlowType {
    Direct = 0, // We will directly reach the right location through other
                // means, no need for continue or break
    Break = 1,
    Continue = 2,
    Nested = 3 // This code is directly reached, but we must be careful to
               // ensure it is nested in an if - it is not reached
    // unconditionally, other code paths exist alongside it that we need to make
    // sure do not intertwine
  };
  Shape
      *Ancestor; // If not nullptr, this shape is the relevant one for purposes
                 // of getting to the target block. We break or continue on it
  Branch::FlowType
      Type;     // If Ancestor is not nullptr, this says whether to break or
                // continue
  bool Labeled; // If a break or continue, whether we need to use a label
  const char *Condition; // The condition for which we branch. For example,
                         // "my_var == 1". Conditions are checked one by one.
                         // One of the conditions should have nullptr as the
                         // condition, in which case it is the default
                         // FIXME: move from char* to LLVM data structures
  const char *Code; // If provided, code that is run right before the branch is
                    // taken. This is useful for phis
                    // FIXME: move from char* to LLVM data structures

  Branch(const char *ConditionInit, const char *CodeInit = nullptr);
  ~Branch();
};

typedef SetVector<Block *> BlockSet;
typedef MapVector<Block *, Branch *> BlockBranchMap;
typedef MapVector<Block *, std::unique_ptr<Branch>> OwningBlockBranchMap;

///
/// Represents a basic block of code - some instructions that end with a
/// control flow modifier (a branch, return or throw).
///
struct Block {
  // Branches become processed after we finish the shape relevant to them. For
  // example, when we recreate a loop, branches to the loop start become
  // continues and are now processed. When we calculate what shape to generate
  // from a set of blocks, we ignore processed branches. Blocks own the Branch
  // objects they use, and destroy them when done.
  OwningBlockBranchMap BranchesOut;
  BlockSet BranchesIn;
  OwningBlockBranchMap ProcessedBranchesOut;
  BlockSet ProcessedBranchesIn;
  Shape *Parent; // The shape we are directly inside
  int Id; // A unique identifier, defined when added to relooper. Note that this
          // uniquely identifies a *logical* block - if we split it, the two
          // instances have the same content *and* the same Id
  const char *Code;      // The string representation of the code in this block.
                         // Owning pointer (we copy the input)
                         // FIXME: move from char* to LLVM data structures
  const char *BranchVar; // A variable whose value determines where we go; if
                         // this is not nullptr, emit a switch on that variable
                         // FIXME: move from char* to LLVM data structures
  bool IsCheckedMultipleEntry; // If true, we are a multiple entry, so reaching
                               // us requires setting the label variable

  Block(const char *CodeInit, const char *BranchVarInit);
  ~Block();

  void AddBranchTo(Block *Target, const char *Condition,
                   const char *Code = nullptr);
};

///
/// Represents a structured control flow shape
///
struct Shape {
  int Id; // A unique identifier. Used to identify loops, labels are Lx where x
          // is the Id. Defined when added to relooper
  Shape *Next;    // The shape that will appear in the code right after this one
  Shape *Natural; // The shape that control flow gets to naturally (if there is
                  // Next, then this is Next)

  /// Discriminator for LLVM-style RTTI (dyn_cast<> et al.)
  enum ShapeKind { SK_Simple, SK_Multiple, SK_Loop };

private:
  ShapeKind Kind;

public:
  ShapeKind getKind() const { return Kind; }

  Shape(ShapeKind KindInit) : Id(-1), Next(nullptr), Kind(KindInit) {}
};

///
/// Simple: No control flow at all, just instructions.
///
struct SimpleShape : public Shape {
  Block *Inner;

  SimpleShape() : Shape(SK_Simple), Inner(nullptr) {}

  static bool classof(const Shape *S) { return S->getKind() == SK_Simple; }
};

///
/// A shape that may be implemented with a labeled loop.
///
struct LabeledShape : public Shape {
  bool Labeled; // If we have a loop, whether it needs to be labeled

  LabeledShape(ShapeKind KindInit) : Shape(KindInit), Labeled(false) {}
};

// Blocks with the same id were split and are identical, so we just care about
// ids in Multiple entries
typedef std::map<int, Shape *> IdShapeMap;

///
/// Multiple: A shape with more than one entry. If the next block to
///           be entered is among them, we run it and continue to
///           the next shape, otherwise we continue immediately to the
///           next shape.
///
struct MultipleShape : public LabeledShape {
  IdShapeMap InnerMap; // entry block ID -> shape
  int Breaks; // If we have branches on us, we need a loop (or a switch). This
              // is a counter of requirements,
              // if we optimize it to 0, the loop is unneeded
  bool UseSwitch; // Whether to switch on label as opposed to an if-else chain

  MultipleShape() : LabeledShape(SK_Multiple), Breaks(0), UseSwitch(false) {}

  static bool classof(const Shape *S) { return S->getKind() == SK_Multiple; }
};

///
/// Loop: An infinite loop.
///
struct LoopShape : public LabeledShape {
  Shape *Inner;

  LoopShape() : LabeledShape(SK_Loop), Inner(nullptr) {}

  static bool classof(const Shape *S) { return S->getKind() == SK_Loop; }
};

} // namespace Relooper

} // namespace llvm
