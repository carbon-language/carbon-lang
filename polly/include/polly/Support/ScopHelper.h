//===------ Support/ScopHelper.h -- Some Helper Functions for Scop. -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with LLVM-IR.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_IRHELPER_H
#define POLLY_SUPPORT_IRHELPER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {
class Type;
class Instruction;
class LoadInst;
class LoopInfo;
class Loop;
class ScalarEvolution;
class SCEV;
class Value;
class PHINode;
class Region;
class Pass;
class BasicBlock;
class StringRef;
class DataLayout;
class DominatorTree;
class RegionInfo;
class TerminatorInst;
class ScalarEvolution;
}

namespace polly {
class Scop;

/// @brief Type to remap values.
using ValueMapT = llvm::DenseMap<llvm::AssertingVH<llvm::Value>,
                                 llvm::AssertingVH<llvm::Value>>;

/// @brief Type for a set of invariant loads.
using InvariantLoadsSetTy = llvm::SetVector<llvm::AssertingVH<llvm::LoadInst>>;

/// Temporary Hack for extended regiontree.
///
/// @brief Cast the region to loop.
///
/// @param R  The Region to be casted.
/// @param LI The LoopInfo to help the casting.
///
/// @return If there is a a loop that has the same entry and exit as the region,
///         return the loop, otherwise, return null.
llvm::Loop *castToLoop(const llvm::Region &R, llvm::LoopInfo &LI);

/// @brief Check if the PHINode has any incoming Invoke edge.
///
/// @param PN The PHINode to check.
///
/// @return If the PHINode has an incoming BB that jumps to the parent BB
///         of the PHINode with an invoke instruction, return true,
///         otherwise, return false.
bool hasInvokeEdge(const llvm::PHINode *PN);

llvm::Value *getPointerOperand(llvm::Instruction &Inst);

/// @brief Simplify the region to have a single unconditional entry edge and a
/// single exit edge.
///
/// Although this function allows DT and RI to be null, regions only work
/// properly if the DominatorTree (for Region::contains) and RegionInfo are kept
/// up-to-date.
///
/// @param R  The region to be simplified
/// @param DT DominatorTree to be updated.
/// @param LI LoopInfo to be updated.
/// @param RI RegionInfo to be updated.
void simplifyRegion(llvm::Region *R, llvm::DominatorTree *DT,
                    llvm::LoopInfo *LI, llvm::RegionInfo *RI);

/// @brief Split the entry block of a function to store the newly inserted
///        allocations outside of all Scops.
///
/// @param EntryBlock The entry block of the current function.
/// @param P          The pass that currently running.
///
void splitEntryBlockForAlloca(llvm::BasicBlock *EntryBlock, llvm::Pass *P);

/// @brief Wrapper for SCEVExpander extended to all Polly features.
///
/// This wrapper will internally call the SCEVExpander but also makes sure that
/// all additional features not represented in SCEV (e.g., SDiv/SRem are not
/// black boxes but can be part of the function) will be expanded correctly.
///
/// The parameters are the same as for the creation of a SCEVExpander as well
/// as the call to SCEVExpander::expandCodeFor:
///
/// @param S    The current Scop.
/// @param SE   The Scalar Evolution pass.
/// @param DL   The module data layout.
/// @param Name The suffix added to the new instruction names.
/// @param E    The expression for which code is actually generated.
/// @param Ty   The type of the resulting code.
/// @param IP   The insertion point for the new code.
/// @param VMap A remaping of values used in @p E.
llvm::Value *expandCodeFor(Scop &S, llvm::ScalarEvolution &SE,
                           const llvm::DataLayout &DL, const char *Name,
                           const llvm::SCEV *E, llvm::Type *Ty,
                           llvm::Instruction *IP, ValueMapT *VMap = nullptr);

/// @brief Check if the block is a error block.
///
/// A error block is currently any block that fullfills at least one of
/// the following conditions:
///
///  - It is terminated by an unreachable instruction
///  - It contains a call to a non-pure function that is not immediately
///    dominated by a loop header and that does not dominate the region exit.
///    This is a heuristic to pick only error blocks that are conditionally
///    executed and can be assumed to be not executed at all without the domains
///    beeing available.
///
/// @param BB The block to check.
/// @param R  The analyzed region.
/// @param LI The loop info analysis.
/// @param DT The dominator tree of the function.
///
/// @return True if the block is a error block, false otherwise.
bool isErrorBlock(llvm::BasicBlock &BB, const llvm::Region &R,
                  llvm::LoopInfo &LI, const llvm::DominatorTree &DT);

/// @brief Return the condition for the terminator @p TI.
///
/// For unconditional branches the "i1 true" condition will be returned.
///
/// @param TI The terminator to get the condition from.
///
/// @return The condition of @p TI and nullptr if none could be extracted.
llvm::Value *getConditionFromTerminator(llvm::TerminatorInst *TI);

/// @brief Check if @p LInst can be hoisted in @p R.
///
/// @param LInst The load to check.
/// @param R     The analyzed region.
/// @param LI    The loop info.
/// @param SE    The scalar evolution analysis.
///
/// @return True if @p LInst can be hoisted in @p R.
bool isHoistableLoad(llvm::LoadInst *LInst, llvm::Region &R, llvm::LoopInfo &LI,
                     llvm::ScalarEvolution &SE);

/// @brief Return true iff @p V is an intrinsic that we ignore during code
///        generation.
bool isIgnoredIntrinsic(const llvm::Value *V);

/// @brief Check whether a value an be synthesized by the code generator.
///
/// Some value will be recalculated only from information that is code generated
/// from the polyhedral representation. For such instructions we do not need to
/// ensure that their operands are available during code generation.
///
/// @param V The value to check.
/// @param LI The LoopInfo analysis.
/// @param SE The scalar evolution database.
/// @param R The region out of which SSA names are parameters.
/// @return If the instruction I can be regenerated from its
///         scalar evolution representation, return true,
///         otherwise return false.
bool canSynthesize(const llvm::Value *V, const llvm::LoopInfo *LI,
                   llvm::ScalarEvolution *SE, const llvm::Region *R);
}
#endif
