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

namespace llvm {
class Type;
class Instruction;
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
class ScalarEvolution;
}

namespace polly {
class Scop;
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
llvm::Value *expandCodeFor(Scop &S, llvm::ScalarEvolution &SE,
                           const llvm::DataLayout &DL, const char *Name,
                           const llvm::SCEV *E, llvm::Type *Ty,
                           llvm::Instruction *IP);
}
#endif
