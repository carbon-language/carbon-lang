//===- IR/OpenMPIRBuilder.h - OpenMP encoding builder for LLVM IR - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpenMPIRBuilder class and helpers used as a convenient
// way to create LLVM instructions for OpenMP directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OPENMP_OMPIRBUILDER_H
#define LLVM_FRONTEND_OPENMP_OMPIRBUILDER_H

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Allocator.h"
#include <forward_list>

namespace llvm {
class CanonicalLoopInfo;

/// An interface to create LLVM-IR for OpenMP directives.
///
/// Each OpenMP directive has a corresponding public generator method.
class OpenMPIRBuilder {
public:
  /// Create a new OpenMPIRBuilder operating on the given module \p M. This will
  /// not have an effect on \p M (see initialize).
  OpenMPIRBuilder(Module &M) : M(M), Builder(M.getContext()) {}
  ~OpenMPIRBuilder();

  /// Initialize the internal state, this will put structures types and
  /// potentially other helpers into the underlying module. Must be called
  /// before any other method and only once!
  void initialize();

  /// Finalize the underlying module, e.g., by outlining regions.
  /// \param Fn                    The function to be finalized. If not used,
  ///                              all functions are finalized.
  /// \param AllowExtractorSinking Flag to include sinking instructions,
  ///                              emitted by CodeExtractor, in the
  ///                              outlined region. Default is false.
  void finalize(Function *Fn = nullptr, bool AllowExtractorSinking = false);

  /// Add attributes known for \p FnID to \p Fn.
  void addAttributes(omp::RuntimeFunction FnID, Function &Fn);

  /// Type used throughout for insertion points.
  using InsertPointTy = IRBuilder<>::InsertPoint;

  /// Callback type for variable finalization (think destructors).
  ///
  /// \param CodeGenIP is the insertion point at which the finalization code
  ///                  should be placed.
  ///
  /// A finalize callback knows about all objects that need finalization, e.g.
  /// destruction, when the scope of the currently generated construct is left
  /// at the time, and location, the callback is invoked.
  using FinalizeCallbackTy = std::function<void(InsertPointTy CodeGenIP)>;

  struct FinalizationInfo {
    /// The finalization callback provided by the last in-flight invocation of
    /// createXXXX for the directive of kind DK.
    FinalizeCallbackTy FiniCB;

    /// The directive kind of the innermost directive that has an associated
    /// region which might require finalization when it is left.
    omp::Directive DK;

    /// Flag to indicate if the directive is cancellable.
    bool IsCancellable;
  };

  /// Push a finalization callback on the finalization stack.
  ///
  /// NOTE: Temporary solution until Clang CG is gone.
  void pushFinalizationCB(const FinalizationInfo &FI) {
    FinalizationStack.push_back(FI);
  }

  /// Pop the last finalization callback from the finalization stack.
  ///
  /// NOTE: Temporary solution until Clang CG is gone.
  void popFinalizationCB() { FinalizationStack.pop_back(); }

  /// Callback type for body (=inner region) code generation
  ///
  /// The callback takes code locations as arguments, each describing a
  /// location at which code might need to be generated or a location that is
  /// the target of control transfer.
  ///
  /// \param AllocaIP is the insertion point at which new alloca instructions
  ///                 should be placed.
  /// \param CodeGenIP is the insertion point at which the body code should be
  ///                  placed.
  /// \param ContinuationBB is the basic block target to leave the body.
  ///
  /// Note that all blocks pointed to by the arguments have terminators.
  using BodyGenCallbackTy =
      function_ref<void(InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                        BasicBlock &ContinuationBB)>;

  // This is created primarily for sections construct as llvm::function_ref
  // (BodyGenCallbackTy) is not storable (as described in the comments of
  // function_ref class - function_ref contains non-ownable reference
  // to the callable.
  using StorableBodyGenCallbackTy =
      std::function<void(InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                         BasicBlock &ContinuationBB)>;

  /// Callback type for loop body code generation.
  ///
  /// \param CodeGenIP is the insertion point where the loop's body code must be
  ///                  placed. This will be a dedicated BasicBlock with a
  ///                  conditional branch from the loop condition check and
  ///                  terminated with an unconditional branch to the loop
  ///                  latch.
  /// \param IndVar    is the induction variable usable at the insertion point.
  using LoopBodyGenCallbackTy =
      function_ref<void(InsertPointTy CodeGenIP, Value *IndVar)>;

  /// Callback type for variable privatization (think copy & default
  /// constructor).
  ///
  /// \param AllocaIP is the insertion point at which new alloca instructions
  ///                 should be placed.
  /// \param CodeGenIP is the insertion point at which the privatization code
  ///                  should be placed.
  /// \param Original The value being copied/created, should not be used in the
  ///                 generated IR.
  /// \param Inner The equivalent of \p Original that should be used in the
  ///              generated IR; this is equal to \p Original if the value is
  ///              a pointer and can thus be passed directly, otherwise it is
  ///              an equivalent but different value.
  /// \param ReplVal The replacement value, thus a copy or new created version
  ///                of \p Inner.
  ///
  /// \returns The new insertion point where code generation continues and
  ///          \p ReplVal the replacement value.
  using PrivatizeCallbackTy = function_ref<InsertPointTy(
      InsertPointTy AllocaIP, InsertPointTy CodeGenIP, Value &Original,
      Value &Inner, Value *&ReplVal)>;

  /// Description of a LLVM-IR insertion point (IP) and a debug/source location
  /// (filename, line, column, ...).
  struct LocationDescription {
    template <typename T, typename U>
    LocationDescription(const IRBuilder<T, U> &IRB)
        : IP(IRB.saveIP()), DL(IRB.getCurrentDebugLocation()) {}
    LocationDescription(const InsertPointTy &IP) : IP(IP) {}
    LocationDescription(const InsertPointTy &IP, const DebugLoc &DL)
        : IP(IP), DL(DL) {}
    InsertPointTy IP;
    DebugLoc DL;
  };

  /// Emitter methods for OpenMP directives.
  ///
  ///{

  /// Generator for '#omp barrier'
  ///
  /// \param Loc The location where the barrier directive was encountered.
  /// \param DK The kind of directive that caused the barrier.
  /// \param ForceSimpleCall Flag to force a simple (=non-cancellation) barrier.
  /// \param CheckCancelFlag Flag to indicate a cancel barrier return value
  ///                        should be checked and acted upon.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy createBarrier(const LocationDescription &Loc, omp::Directive DK,
                              bool ForceSimpleCall = false,
                              bool CheckCancelFlag = true);

  /// Generator for '#omp cancel'
  ///
  /// \param Loc The location where the directive was encountered.
  /// \param IfCondition The evaluated 'if' clause expression, if any.
  /// \param CanceledDirective The kind of directive that is cancled.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy createCancel(const LocationDescription &Loc, Value *IfCondition,
                             omp::Directive CanceledDirective);

  /// Generator for '#omp parallel'
  ///
  /// \param Loc The insert and source location description.
  /// \param AllocaIP The insertion points to be used for alloca instructions.
  /// \param BodyGenCB Callback that will generate the region code.
  /// \param PrivCB Callback to copy a given variable (think copy constructor).
  /// \param FiniCB Callback to finalize variable copies.
  /// \param IfCondition The evaluated 'if' clause expression, if any.
  /// \param NumThreads The evaluated 'num_threads' clause expression, if any.
  /// \param ProcBind The value of the 'proc_bind' clause (see ProcBindKind).
  /// \param IsCancellable Flag to indicate a cancellable parallel region.
  ///
  /// \returns The insertion position *after* the parallel.
  IRBuilder<>::InsertPoint
  createParallel(const LocationDescription &Loc, InsertPointTy AllocaIP,
                 BodyGenCallbackTy BodyGenCB, PrivatizeCallbackTy PrivCB,
                 FinalizeCallbackTy FiniCB, Value *IfCondition,
                 Value *NumThreads, omp::ProcBindKind ProcBind,
                 bool IsCancellable);

  /// Generator for the control flow structure of an OpenMP canonical loop.
  ///
  /// This generator operates on the logical iteration space of the loop, i.e.
  /// the caller only has to provide a loop trip count of the loop as defined by
  /// base language semantics. The trip count is interpreted as an unsigned
  /// integer. The induction variable passed to \p BodyGenCB will be of the same
  /// type and run from 0 to \p TripCount - 1. It is up to the callback to
  /// convert the logical iteration variable to the loop counter variable in the
  /// loop body.
  ///
  /// \param Loc       The insert and source location description. The insert
  ///                  location can be between two instructions or the end of a
  ///                  degenerate block (e.g. a BB under construction).
  /// \param BodyGenCB Callback that will generate the loop body code.
  /// \param TripCount Number of iterations the loop body is executed.
  /// \param Name      Base name used to derive BB and instruction names.
  ///
  /// \returns An object representing the created control flow structure which
  ///          can be used for loop-associated directives.
  CanonicalLoopInfo *createCanonicalLoop(const LocationDescription &Loc,
                                         LoopBodyGenCallbackTy BodyGenCB,
                                         Value *TripCount,
                                         const Twine &Name = "loop");

  /// Generator for the control flow structure of an OpenMP canonical loop.
  ///
  /// Instead of a logical iteration space, this allows specifying user-defined
  /// loop counter values using increment, upper- and lower bounds. To
  /// disambiguate the terminology when counting downwards, instead of lower
  /// bounds we use \p Start for the loop counter value in the first body
  /// iteration.
  ///
  /// Consider the following limitations:
  ///
  ///  * A loop counter space over all integer values of its bit-width cannot be
  ///    represented. E.g using uint8_t, its loop trip count of 256 cannot be
  ///    stored into an 8 bit integer):
  ///
  ///      DO I = 0, 255, 1
  ///
  ///  * Unsigned wrapping is only supported when wrapping only "once"; E.g.
  ///    effectively counting downwards:
  ///
  ///      for (uint8_t i = 100u; i > 0; i += 127u)
  ///
  ///
  /// TODO: May need to add additional parameters to represent:
  ///
  ///  * Allow representing downcounting with unsigned integers.
  ///
  ///  * Sign of the step and the comparison operator might disagree:
  ///
  ///      for (int i = 0; i < 42; i -= 1u)
  ///
  //
  /// \param Loc       The insert and source location description.
  /// \param BodyGenCB Callback that will generate the loop body code.
  /// \param Start     Value of the loop counter for the first iterations.
  /// \param Stop      Loop counter values past this will stop the loop.
  /// \param Step      Loop counter increment after each iteration; negative
  ///                  means counting down.
  /// \param IsSigned  Whether Start, Stop and Step are signed integers.
  /// \param InclusiveStop Whether \p Stop itself is a valid value for the loop
  ///                      counter.
  /// \param ComputeIP Insertion point for instructions computing the trip
  ///                  count. Can be used to ensure the trip count is available
  ///                  at the outermost loop of a loop nest. If not set,
  ///                  defaults to the preheader of the generated loop.
  /// \param Name      Base name used to derive BB and instruction names.
  ///
  /// \returns An object representing the created control flow structure which
  ///          can be used for loop-associated directives.
  CanonicalLoopInfo *createCanonicalLoop(const LocationDescription &Loc,
                                         LoopBodyGenCallbackTy BodyGenCB,
                                         Value *Start, Value *Stop, Value *Step,
                                         bool IsSigned, bool InclusiveStop,
                                         InsertPointTy ComputeIP = {},
                                         const Twine &Name = "loop");

  /// Collapse a loop nest into a single loop.
  ///
  /// Merges loops of a loop nest into a single CanonicalLoopNest representation
  /// that has the same number of innermost loop iterations as the origin loop
  /// nest. The induction variables of the input loops are derived from the
  /// collapsed loop's induction variable. This is intended to be used to
  /// implement OpenMP's collapse clause. Before applying a directive,
  /// collapseLoops normalizes a loop nest to contain only a single loop and the
  /// directive's implementation does not need to handle multiple loops itself.
  /// This does not remove the need to handle all loop nest handling by
  /// directives, such as the ordered(<n>) clause or the simd schedule-clause
  /// modifier of the worksharing-loop directive.
  ///
  /// Example:
  /// \code
  ///   for (int i = 0; i < 7; ++i) // Canonical loop "i"
  ///     for (int j = 0; j < 9; ++j) // Canonical loop "j"
  ///       body(i, j);
  /// \endcode
  ///
  /// After collapsing with Loops={i,j}, the loop is changed to
  /// \code
  ///   for (int ij = 0; ij < 63; ++ij) {
  ///     int i = ij / 9;
  ///     int j = ij % 9;
  ///     body(i, j);
  ///   }
  /// \endcode
  ///
  /// In the current implementation, the following limitations apply:
  ///
  ///  * All input loops have an induction variable of the same type.
  ///
  ///  * The collapsed loop will have the same trip count integer type as the
  ///    input loops. Therefore it is possible that the collapsed loop cannot
  ///    represent all iterations of the input loops. For instance, assuming a
  ///    32 bit integer type, and two input loops both iterating 2^16 times, the
  ///    theoretical trip count of the collapsed loop would be 2^32 iteration,
  ///    which cannot be represented in an 32-bit integer. Behavior is undefined
  ///    in this case.
  ///
  ///  * The trip counts of every input loop must be available at \p ComputeIP.
  ///    Non-rectangular loops are not yet supported.
  ///
  ///  * At each nest level, code between a surrounding loop and its nested loop
  ///    is hoisted into the loop body, and such code will be executed more
  ///    often than before collapsing (or not at all if any inner loop iteration
  ///    has a trip count of 0). This is permitted by the OpenMP specification.
  ///
  /// \param DL        Debug location for instructions added for collapsing,
  ///                  such as instructions to compute/derive the input loop's
  ///                  induction variables.
  /// \param Loops     Loops in the loop nest to collapse. Loops are specified
  ///                  from outermost-to-innermost and every control flow of a
  ///                  loop's body must pass through its directly nested loop.
  /// \param ComputeIP Where additional instruction that compute the collapsed
  ///                  trip count. If not set, defaults to before the generated
  ///                  loop.
  ///
  /// \returns The CanonicalLoopInfo object representing the collapsed loop.
  CanonicalLoopInfo *collapseLoops(DebugLoc DL,
                                   ArrayRef<CanonicalLoopInfo *> Loops,
                                   InsertPointTy ComputeIP);

  /// Modifies the canonical loop to be a statically-scheduled workshare loop.
  ///
  /// This takes a \p LoopInfo representing a canonical loop, such as the one
  /// created by \p createCanonicalLoop and emits additional instructions to
  /// turn it into a workshare loop. In particular, it calls to an OpenMP
  /// runtime function in the preheader to obtain the loop bounds to be used in
  /// the current thread, updates the relevant instructions in the canonical
  /// loop and calls to an OpenMP runtime finalization function after the loop.
  ///
  /// TODO: Workshare loops with static scheduling may contain up to two loops
  /// that fulfill the requirements of an OpenMP canonical loop. One for
  /// iterating over all iterations of a chunk and another one for iterating
  /// over all chunks that are executed on the same thread. Returning
  /// CanonicalLoopInfo objects representing them may eventually be useful for
  /// the apply clause planned in OpenMP 6.0, but currently whether these are
  /// canonical loops is irrelevant.
  ///
  /// \param DL       Debug location for instructions added for the
  ///                 workshare-loop construct itself.
  /// \param CLI      A descriptor of the canonical loop to workshare.
  /// \param AllocaIP An insertion point for Alloca instructions usable in the
  ///                 preheader of the loop.
  /// \param NeedsBarrier Indicates whether a barrier must be inserted after
  ///                     the loop.
  /// \param Chunk    The size of loop chunk considered as a unit when
  ///                 scheduling. If \p nullptr, defaults to 1.
  ///
  /// \returns Point where to insert code after the workshare construct.
  InsertPointTy applyStaticWorkshareLoop(DebugLoc DL, CanonicalLoopInfo *CLI,
                                         InsertPointTy AllocaIP,
                                         bool NeedsBarrier,
                                         Value *Chunk = nullptr);

  /// Modifies the canonical loop to be a dynamically-scheduled workshare loop.
  ///
  /// This takes a \p LoopInfo representing a canonical loop, such as the one
  /// created by \p createCanonicalLoop and emits additional instructions to
  /// turn it into a workshare loop. In particular, it calls to an OpenMP
  /// runtime function in the preheader to obtain, and then in each iteration
  /// to update the loop counter.
  ///
  /// \param DL       Debug location for instructions added for the
  ///                 workshare-loop construct itself.
  /// \param CLI      A descriptor of the canonical loop to workshare.
  /// \param AllocaIP An insertion point for Alloca instructions usable in the
  ///                 preheader of the loop.
  /// \param SchedType Type of scheduling to be passed to the init function.
  /// \param NeedsBarrier Indicates whether a barrier must be insterted after
  ///                     the loop.
  /// \param Chunk    The size of loop chunk considered as a unit when
  ///                 scheduling. If \p nullptr, defaults to 1.
  ///
  /// \returns Point where to insert code after the workshare construct.
  InsertPointTy applyDynamicWorkshareLoop(DebugLoc DL, CanonicalLoopInfo *CLI,
                                          InsertPointTy AllocaIP,
                                          omp::OMPScheduleType SchedType,
                                          bool NeedsBarrier,
                                          Value *Chunk = nullptr);

  /// Modifies the canonical loop to be a workshare loop.
  ///
  /// This takes a \p LoopInfo representing a canonical loop, such as the one
  /// created by \p createCanonicalLoop and emits additional instructions to
  /// turn it into a workshare loop. In particular, it calls to an OpenMP
  /// runtime function in the preheader to obtain the loop bounds to be used in
  /// the current thread, updates the relevant instructions in the canonical
  /// loop and calls to an OpenMP runtime finalization function after the loop.
  ///
  /// \param DL       Debug location for instructions added for the
  ///                 workshare-loop construct itself.
  /// \param CLI      A descriptor of the canonical loop to workshare.
  /// \param AllocaIP An insertion point for Alloca instructions usable in the
  ///                 preheader of the loop.
  /// \param NeedsBarrier Indicates whether a barrier must be insterted after
  ///                     the loop.
  ///
  /// \returns Point where to insert code after the workshare construct.
  InsertPointTy applyWorkshareLoop(DebugLoc DL, CanonicalLoopInfo *CLI,
                                   InsertPointTy AllocaIP, bool NeedsBarrier);

  /// Tile a loop nest.
  ///
  /// Tiles the loops of \p Loops by the tile sizes in \p TileSizes. Loops in
  /// \p/ Loops must be perfectly nested, from outermost to innermost loop
  /// (i.e. Loops.front() is the outermost loop). The trip count llvm::Value
  /// of every loop and every tile sizes must be usable in the outermost
  /// loop's preheader. This implies that the loop nest is rectangular.
  ///
  /// Example:
  /// \code
  ///   for (int i = 0; i < 15; ++i) // Canonical loop "i"
  ///     for (int j = 0; j < 14; ++j) // Canonical loop "j"
  ///         body(i, j);
  /// \endcode
  ///
  /// After tiling with Loops={i,j} and TileSizes={5,7}, the loop is changed to
  /// \code
  ///   for (int i1 = 0; i1 < 3; ++i1)
  ///     for (int j1 = 0; j1 < 2; ++j1)
  ///       for (int i2 = 0; i2 < 5; ++i2)
  ///         for (int j2 = 0; j2 < 7; ++j2)
  ///           body(i1*3+i2, j1*3+j2);
  /// \endcode
  ///
  /// The returned vector are the loops {i1,j1,i2,j2}. The loops i1 and j1 are
  /// referred to the floor, and the loops i2 and j2 are the tiles. Tiling also
  /// handles non-constant trip counts, non-constant tile sizes and trip counts
  /// that are not multiples of the tile size. In the latter case the tile loop
  /// of the last floor-loop iteration will have fewer iterations than specified
  /// as its tile size.
  ///
  ///
  /// @param DL        Debug location for instructions added by tiling, for
  ///                  instance the floor- and tile trip count computation.
  /// @param Loops     Loops to tile. The CanonicalLoopInfo objects are
  ///                  invalidated by this method, i.e. should not used after
  ///                  tiling.
  /// @param TileSizes For each loop in \p Loops, the tile size for that
  ///                  dimensions.
  ///
  /// \returns A list of generated loops. Contains twice as many loops as the
  ///          input loop nest; the first half are the floor loops and the
  ///          second half are the tile loops.
  std::vector<CanonicalLoopInfo *>
  tileLoops(DebugLoc DL, ArrayRef<CanonicalLoopInfo *> Loops,
            ArrayRef<Value *> TileSizes);

  /// Generator for '#omp flush'
  ///
  /// \param Loc The location where the flush directive was encountered
  void createFlush(const LocationDescription &Loc);

  /// Generator for '#omp taskwait'
  ///
  /// \param Loc The location where the taskwait directive was encountered.
  void createTaskwait(const LocationDescription &Loc);

  /// Generator for '#omp taskyield'
  ///
  /// \param Loc The location where the taskyield directive was encountered.
  void createTaskyield(const LocationDescription &Loc);

  /// Functions used to generate reductions. Such functions take two Values
  /// representing LHS and RHS of the reduction, respectively, and a reference
  /// to the value that is updated to refer to the reduction result.
  using ReductionGenTy =
      function_ref<InsertPointTy(InsertPointTy, Value *, Value *, Value *&)>;

  /// Functions used to generate atomic reductions. Such functions take two
  /// Values representing pointers to LHS and RHS of the reduction. They are
  /// expected to atomically update the LHS to the reduced value.
  using AtomicReductionGenTy =
      function_ref<InsertPointTy(InsertPointTy, Value *, Value *)>;

  /// Information about an OpenMP reduction.
  struct ReductionInfo {
    ReductionInfo(Value *Variable, Value *PrivateVariable,
                  ReductionGenTy ReductionGen,
                  AtomicReductionGenTy AtomicReductionGen)
        : Variable(Variable), PrivateVariable(PrivateVariable),
          ReductionGen(ReductionGen), AtomicReductionGen(AtomicReductionGen) {}

    /// Returns the type of the element being reduced.
    Type *getElementType() const {
      return Variable->getType()->getPointerElementType();
    }

    /// Reduction variable of pointer type.
    Value *Variable;

    /// Thread-private partial reduction variable.
    Value *PrivateVariable;

    /// Callback for generating the reduction body. The IR produced by this will
    /// be used to combine two values in a thread-safe context, e.g., under
    /// lock or within the same thread, and therefore need not be atomic.
    ReductionGenTy ReductionGen;

    /// Callback for generating the atomic reduction body, may be null. The IR
    /// produced by this will be used to atomically combine two values during
    /// reduction. If null, the implementation will use the non-atomic version
    /// along with the appropriate synchronization mechanisms.
    AtomicReductionGenTy AtomicReductionGen;
  };

  // TODO: provide atomic and non-atomic reduction generators for reduction
  // operators defined by the OpenMP specification.

  /// Generator for '#omp reduction'.
  ///
  /// Emits the IR instructing the runtime to perform the specific kind of
  /// reductions. Expects reduction variables to have been privatized and
  /// initialized to reduction-neutral values separately. Emits the calls to
  /// runtime functions as well as the reduction function and the basic blocks
  /// performing the reduction atomically and non-atomically.
  ///
  /// The code emitted for the following:
  ///
  /// \code
  ///   type var_1;
  ///   type var_2;
  ///   #pragma omp <directive> reduction(reduction-op:var_1,var_2)
  ///   /* body */;
  /// \endcode
  ///
  /// corresponds to the following sketch.
  ///
  /// \code
  /// void _outlined_par() {
  ///   // N is the number of different reductions.
  ///   void *red_array[] = {privatized_var_1, privatized_var_2, ...};
  ///   switch(__kmpc_reduce(..., N, /*size of data in red array*/, red_array,
  ///                        _omp_reduction_func,
  ///                        _gomp_critical_user.reduction.var)) {
  ///   case 1: {
  ///     var_1 = var_1 <reduction-op> privatized_var_1;
  ///     var_2 = var_2 <reduction-op> privatized_var_2;
  ///     // ...
  ///    __kmpc_end_reduce(...);
  ///     break;
  ///   }
  ///   case 2: {
  ///     _Atomic<ReductionOp>(var_1, privatized_var_1);
  ///     _Atomic<ReductionOp>(var_2, privatized_var_2);
  ///     // ...
  ///     break;
  ///   }
  ///   default: break;
  ///   }
  /// }
  ///
  /// void _omp_reduction_func(void **lhs, void **rhs) {
  ///   *(type *)lhs[0] = *(type *)lhs[0] <reduction-op> *(type *)rhs[0];
  ///   *(type *)lhs[1] = *(type *)lhs[1] <reduction-op> *(type *)rhs[1];
  ///   // ...
  /// }
  /// \endcode
  ///
  /// \param Loc                The location where the reduction was
  ///                           encountered. Must be within the associate
  ///                           directive and after the last local access to the
  ///                           reduction variables.
  /// \param AllocaIP           An insertion point suitable for allocas usable
  ///                           in reductions.
  /// \param ReductionInfos     A list of info on each reduction variable.
  /// \param IsNoWait           A flag set if the reduction is marked as nowait.
  InsertPointTy createReductions(const LocationDescription &Loc,
                                 InsertPointTy AllocaIP,
                                 ArrayRef<ReductionInfo> ReductionInfos,
                                 bool IsNoWait = false);

  ///}

  /// Return the insertion point used by the underlying IRBuilder.
  InsertPointTy getInsertionPoint() { return Builder.saveIP(); }

  /// Update the internal location to \p Loc.
  bool updateToLocation(const LocationDescription &Loc) {
    Builder.restoreIP(Loc.IP);
    Builder.SetCurrentDebugLocation(Loc.DL);
    return Loc.IP.getBlock() != nullptr;
  }

  /// Return the function declaration for the runtime function with \p FnID.
  FunctionCallee getOrCreateRuntimeFunction(Module &M,
                                            omp::RuntimeFunction FnID);

  Function *getOrCreateRuntimeFunctionPtr(omp::RuntimeFunction FnID);

  /// Return the (LLVM-IR) string describing the source location \p LocStr.
  Constant *getOrCreateSrcLocStr(StringRef LocStr);

  /// Return the (LLVM-IR) string describing the default source location.
  Constant *getOrCreateDefaultSrcLocStr();

  /// Return the (LLVM-IR) string describing the source location identified by
  /// the arguments.
  Constant *getOrCreateSrcLocStr(StringRef FunctionName, StringRef FileName,
                                 unsigned Line, unsigned Column);

  /// Return the (LLVM-IR) string describing the DebugLoc \p DL. Use \p F as
  /// fallback if \p DL does not specify the function name.
  Constant *getOrCreateSrcLocStr(DebugLoc DL, Function *F = nullptr);

  /// Return the (LLVM-IR) string describing the source location \p Loc.
  Constant *getOrCreateSrcLocStr(const LocationDescription &Loc);

  /// Return an ident_t* encoding the source location \p SrcLocStr and \p Flags.
  /// TODO: Create a enum class for the Reserve2Flags
  Value *getOrCreateIdent(Constant *SrcLocStr,
                          omp::IdentFlag Flags = omp::IdentFlag(0),
                          unsigned Reserve2Flags = 0);

  /// Generate control flow and cleanup for cancellation.
  ///
  /// \param CancelFlag Flag indicating if the cancellation is performed.
  /// \param CanceledDirective The kind of directive that is cancled.
  /// \param ExitCB Extra code to be generated in the exit block.
  void emitCancelationCheckImpl(Value *CancelFlag,
                                omp::Directive CanceledDirective,
                                FinalizeCallbackTy ExitCB = {});

  /// Generate a barrier runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  /// \param DK The directive which caused the barrier
  /// \param ForceSimpleCall Flag to force a simple (=non-cancellation) barrier.
  /// \param CheckCancelFlag Flag to indicate a cancel barrier return value
  ///                        should be checked and acted upon.
  ///
  /// \returns The insertion point after the barrier.
  InsertPointTy emitBarrierImpl(const LocationDescription &Loc,
                                omp::Directive DK, bool ForceSimpleCall,
                                bool CheckCancelFlag);

  /// Generate a flush runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitFlush(const LocationDescription &Loc);

  /// The finalization stack made up of finalize callbacks currently in-flight,
  /// wrapped into FinalizationInfo objects that reference also the finalization
  /// target block and the kind of cancellable directive.
  SmallVector<FinalizationInfo, 8> FinalizationStack;

  /// Return true if the last entry in the finalization stack is of kind \p DK
  /// and cancellable.
  bool isLastFinalizationInfoCancellable(omp::Directive DK) {
    return !FinalizationStack.empty() &&
           FinalizationStack.back().IsCancellable &&
           FinalizationStack.back().DK == DK;
  }

  /// Generate a taskwait runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitTaskwaitImpl(const LocationDescription &Loc);

  /// Generate a taskyield runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitTaskyieldImpl(const LocationDescription &Loc);

  /// Return the current thread ID.
  ///
  /// \param Ident The ident (ident_t*) describing the query origin.
  Value *getOrCreateThreadID(Value *Ident);

  /// The underlying LLVM-IR module
  Module &M;

  /// The LLVM-IR Builder used to create IR.
  IRBuilder<> Builder;

  /// Map to remember source location strings
  StringMap<Constant *> SrcLocStrMap;

  /// Map to remember existing ident_t*.
  DenseMap<std::pair<Constant *, uint64_t>, Value *> IdentMap;

  /// Helper that contains information about regions we need to outline
  /// during finalization.
  struct OutlineInfo {
    using PostOutlineCBTy = std::function<void(Function &)>;
    PostOutlineCBTy PostOutlineCB;
    BasicBlock *EntryBB, *ExitBB;

    /// Collect all blocks in between EntryBB and ExitBB in both the given
    /// vector and set.
    void collectBlocks(SmallPtrSetImpl<BasicBlock *> &BlockSet,
                       SmallVectorImpl<BasicBlock *> &BlockVector);

    /// Return the function that contains the region to be outlined.
    Function *getFunction() const { return EntryBB->getParent(); }
  };

  /// Collection of regions that need to be outlined during finalization.
  SmallVector<OutlineInfo, 16> OutlineInfos;

  /// Collection of owned canonical loop objects that eventually need to be
  /// free'd.
  std::forward_list<CanonicalLoopInfo> LoopInfos;

  /// Add a new region that will be outlined later.
  void addOutlineInfo(OutlineInfo &&OI) { OutlineInfos.emplace_back(OI); }

  /// An ordered map of auto-generated variables to their unique names.
  /// It stores variables with the following names: 1) ".gomp_critical_user_" +
  /// <critical_section_name> + ".var" for "omp critical" directives; 2)
  /// <mangled_name_for_global_var> + ".cache." for cache for threadprivate
  /// variables.
  StringMap<AssertingVH<Constant>, BumpPtrAllocator> InternalVars;

  /// Create the global variable holding the offload mappings information.
  GlobalVariable *createOffloadMaptypes(SmallVectorImpl<uint64_t> &Mappings,
                                        std::string VarName);

  /// Create the global variable holding the offload names information.
  GlobalVariable *
  createOffloadMapnames(SmallVectorImpl<llvm::Constant *> &Names,
                        std::string VarName);

  struct MapperAllocas {
    AllocaInst *ArgsBase = nullptr;
    AllocaInst *Args = nullptr;
    AllocaInst *ArgSizes = nullptr;
  };

  /// Create the allocas instruction used in call to mapper functions.
  void createMapperAllocas(const LocationDescription &Loc,
                           InsertPointTy AllocaIP, unsigned NumOperands,
                           struct MapperAllocas &MapperAllocas);

  /// Create the call for the target mapper function.
  /// \param Loc The source location description.
  /// \param MapperFunc Function to be called.
  /// \param SrcLocInfo Source location information global.
  /// \param MaptypesArg The argument types.
  /// \param MapnamesArg The argument names.
  /// \param MapperAllocas The AllocaInst used for the call.
  /// \param DeviceID Device ID for the call.
  /// \param NumOperands Number of operands in the call.
  void emitMapperCall(const LocationDescription &Loc, Function *MapperFunc,
                      Value *SrcLocInfo, Value *MaptypesArg, Value *MapnamesArg,
                      struct MapperAllocas &MapperAllocas, int64_t DeviceID,
                      unsigned NumOperands);

public:
  /// Generator for __kmpc_copyprivate
  ///
  /// \param Loc The source location description.
  /// \param BufSize Number of elements in the buffer.
  /// \param CpyBuf List of pointers to data to be copied.
  /// \param CpyFn function to call for copying data.
  /// \param DidIt flag variable; 1 for 'single' thread, 0 otherwise.
  ///
  /// \return The insertion position *after* the CopyPrivate call.

  InsertPointTy createCopyPrivate(const LocationDescription &Loc,
                                  llvm::Value *BufSize, llvm::Value *CpyBuf,
                                  llvm::Value *CpyFn, llvm::Value *DidIt);

  /// Generator for '#omp single'
  ///
  /// \param Loc The source location description.
  /// \param BodyGenCB Callback that will generate the region code.
  /// \param FiniCB Callback to finalize variable copies.
  /// \param DidIt Local variable used as a flag to indicate 'single' thread
  ///
  /// \returns The insertion position *after* the single call.
  InsertPointTy createSingle(const LocationDescription &Loc,
                             BodyGenCallbackTy BodyGenCB,
                             FinalizeCallbackTy FiniCB, llvm::Value *DidIt);

  /// Generator for '#omp master'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region code.
  /// \param FiniCB Callback to finalize variable copies.
  ///
  /// \returns The insertion position *after* the master.
  InsertPointTy createMaster(const LocationDescription &Loc,
                             BodyGenCallbackTy BodyGenCB,
                             FinalizeCallbackTy FiniCB);

  /// Generator for '#omp masked'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region code.
  /// \param FiniCB Callback to finialize variable copies.
  ///
  /// \returns The insertion position *after* the masked.
  InsertPointTy createMasked(const LocationDescription &Loc,
                             BodyGenCallbackTy BodyGenCB,
                             FinalizeCallbackTy FiniCB, Value *Filter);

  /// Generator for '#omp critical'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region body code.
  /// \param FiniCB Callback to finalize variable copies.
  /// \param CriticalName name of the lock used by the critical directive
  /// \param HintInst Hint Instruction for hint clause associated with critical
  ///
  /// \returns The insertion position *after* the critical.
  InsertPointTy createCritical(const LocationDescription &Loc,
                               BodyGenCallbackTy BodyGenCB,
                               FinalizeCallbackTy FiniCB,
                               StringRef CriticalName, Value *HintInst);

  /// Generator for '#omp sections'
  ///
  /// \param Loc The insert and source location description.
  /// \param AllocaIP The insertion points to be used for alloca instructions.
  /// \param SectionCBs Callbacks that will generate body of each section.
  /// \param PrivCB Callback to copy a given variable (think copy constructor).
  /// \param FiniCB Callback to finalize variable copies.
  /// \param IsCancellable Flag to indicate a cancellable parallel region.
  /// \param IsNowait If true, barrier - to ensure all sections are executed
  /// before moving forward will not be generated.
  /// \returns The insertion position *after* the sections.
  InsertPointTy createSections(const LocationDescription &Loc,
                               InsertPointTy AllocaIP,
                               ArrayRef<StorableBodyGenCallbackTy> SectionCBs,
                               PrivatizeCallbackTy PrivCB,
                               FinalizeCallbackTy FiniCB, bool IsCancellable,
                               bool IsNowait);

  /// Generator for '#omp section'
  ///
  /// \param Loc The insert and source location description.
  /// \param BodyGenCB Callback that will generate the region body code.
  /// \param FiniCB Callback to finalize variable copies.
  /// \returns The insertion position *after* the section.
  InsertPointTy createSection(const LocationDescription &Loc,
                              BodyGenCallbackTy BodyGenCB,
                              FinalizeCallbackTy FiniCB);

  /// Generate conditional branch and relevant BasicBlocks through which private
  /// threads copy the 'copyin' variables from Master copy to threadprivate
  /// copies.
  ///
  /// \param IP insertion block for copyin conditional
  /// \param MasterVarPtr a pointer to the master variable
  /// \param PrivateVarPtr a pointer to the threadprivate variable
  /// \param IntPtrTy Pointer size type
  /// \param BranchtoEnd Create a branch between the copyin.not.master blocks
  //				 and copy.in.end block
  ///
  /// \returns The insertion point where copying operation to be emitted.
  InsertPointTy createCopyinClauseBlocks(InsertPointTy IP, Value *MasterAddr,
                                         Value *PrivateAddr,
                                         llvm::IntegerType *IntPtrTy,
                                         bool BranchtoEnd = true);

  /// Create a runtime call for kmpc_Alloc
  ///
  /// \param Loc The insert and source location description.
  /// \param Size Size of allocated memory space
  /// \param Allocator Allocator information instruction
  /// \param Name Name of call Instruction for OMP_alloc
  ///
  /// \returns CallInst to the OMP_Alloc call
  CallInst *createOMPAlloc(const LocationDescription &Loc, Value *Size,
                           Value *Allocator, std::string Name = "");

  /// Create a runtime call for kmpc_free
  ///
  /// \param Loc The insert and source location description.
  /// \param Addr Address of memory space to be freed
  /// \param Allocator Allocator information instruction
  /// \param Name Name of call Instruction for OMP_Free
  ///
  /// \returns CallInst to the OMP_Free call
  CallInst *createOMPFree(const LocationDescription &Loc, Value *Addr,
                          Value *Allocator, std::string Name = "");

  /// Create a runtime call for kmpc_threadprivate_cached
  ///
  /// \param Loc The insert and source location description.
  /// \param Pointer pointer to data to be cached
  /// \param Size size of data to be cached
  /// \param Name Name of call Instruction for callinst
  ///
  /// \returns CallInst to the thread private cache call.
  CallInst *createCachedThreadPrivate(const LocationDescription &Loc,
                                      llvm::Value *Pointer,
                                      llvm::ConstantInt *Size,
                                      const llvm::Twine &Name = Twine(""));

  /// The `omp target` interface
  ///
  /// For more information about the usage of this interface,
  /// \see openmp/libomptarget/deviceRTLs/common/include/target.h
  ///
  ///{

  /// Create a runtime call for kmpc_target_init
  ///
  /// \param Loc The insert and source location description.
  /// \param IsSPMD Flag to indicate if the kernel is an SPMD kernel or not.
  /// \param RequiresFullRuntime Indicate if a full device runtime is necessary.
  InsertPointTy createTargetInit(const LocationDescription &Loc, bool IsSPMD, bool RequiresFullRuntime);

  /// Create a runtime call for kmpc_target_deinit
  ///
  /// \param Loc The insert and source location description.
  /// \param IsSPMD Flag to indicate if the kernel is an SPMD kernel or not.
  /// \param RequiresFullRuntime Indicate if a full device runtime is necessary.
  void createTargetDeinit(const LocationDescription &Loc, bool IsSPMD, bool RequiresFullRuntime);

  ///}

  /// Declarations for LLVM-IR types (simple, array, function and structure) are
  /// generated below. Their names are defined and used in OpenMPKinds.def. Here
  /// we provide the declarations, the initializeTypes function will provide the
  /// values.
  ///
  ///{
#define OMP_TYPE(VarName, InitValue) Type *VarName = nullptr;
#define OMP_ARRAY_TYPE(VarName, ElemTy, ArraySize)                             \
  ArrayType *VarName##Ty = nullptr;                                            \
  PointerType *VarName##PtrTy = nullptr;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  FunctionType *VarName = nullptr;                                             \
  PointerType *VarName##Ptr = nullptr;
#define OMP_STRUCT_TYPE(VarName, StrName, ...)                                 \
  StructType *VarName = nullptr;                                               \
  PointerType *VarName##Ptr = nullptr;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  ///}

private:
  /// Create all simple and struct types exposed by the runtime and remember
  /// the llvm::PointerTypes of them for easy access later.
  void initializeTypes(Module &M);

  /// Common interface for generating entry calls for OMP Directives.
  /// if the directive has a region/body, It will set the insertion
  /// point to the body
  ///
  /// \param OMPD Directive to generate entry blocks for
  /// \param EntryCall Call to the entry OMP Runtime Function
  /// \param ExitBB block where the region ends.
  /// \param Conditional indicate if the entry call result will be used
  ///        to evaluate a conditional of whether a thread will execute
  ///        body code or not.
  ///
  /// \return The insertion position in exit block
  InsertPointTy emitCommonDirectiveEntry(omp::Directive OMPD, Value *EntryCall,
                                         BasicBlock *ExitBB,
                                         bool Conditional = false);

  /// Common interface to finalize the region
  ///
  /// \param OMPD Directive to generate exiting code for
  /// \param FinIP Insertion point for emitting Finalization code and exit call
  /// \param ExitCall Call to the ending OMP Runtime Function
  /// \param HasFinalize indicate if the directive will require finalization
  ///         and has a finalization callback in the stack that
  ///        should be called.
  ///
  /// \return The insertion position in exit block
  InsertPointTy emitCommonDirectiveExit(omp::Directive OMPD,
                                        InsertPointTy FinIP,
                                        Instruction *ExitCall,
                                        bool HasFinalize = true);

  /// Common Interface to generate OMP inlined regions
  ///
  /// \param OMPD Directive to generate inlined region for
  /// \param EntryCall Call to the entry OMP Runtime Function
  /// \param ExitCall Call to the ending OMP Runtime Function
  /// \param BodyGenCB Body code generation callback.
  /// \param FiniCB Finalization Callback. Will be called when finalizing region
  /// \param Conditional indicate if the entry call result will be used
  ///        to evaluate a conditional of whether a thread will execute
  ///        body code or not.
  /// \param HasFinalize indicate if the directive will require finalization
  ///        and has a finalization callback in the stack that
  ///        should be called.
  /// \param IsCancellable if HasFinalize is set to true, indicate if the
  ///        the directive should be cancellable.
  /// \return The insertion point after the region

  InsertPointTy
  EmitOMPInlinedRegion(omp::Directive OMPD, Instruction *EntryCall,
                       Instruction *ExitCall, BodyGenCallbackTy BodyGenCB,
                       FinalizeCallbackTy FiniCB, bool Conditional = false,
                       bool HasFinalize = true, bool IsCancellable = false);

  /// Get the platform-specific name separator.
  /// \param Parts different parts of the final name that needs separation
  /// \param FirstSeparator First separator used between the initial two
  ///        parts of the name.
  /// \param Separator separator used between all of the rest consecutive
  ///        parts of the name
  static std::string getNameWithSeparators(ArrayRef<StringRef> Parts,
                                           StringRef FirstSeparator,
                                           StringRef Separator);

  /// Gets (if variable with the given name already exist) or creates
  /// internal global variable with the specified Name. The created variable has
  /// linkage CommonLinkage by default and is initialized by null value.
  /// \param Ty Type of the global variable. If it is exist already the type
  /// must be the same.
  /// \param Name Name of the variable.
  Constant *getOrCreateOMPInternalVariable(Type *Ty, const Twine &Name,
                                           unsigned AddressSpace = 0);

  /// Returns corresponding lock object for the specified critical region
  /// name. If the lock object does not exist it is created, otherwise the
  /// reference to the existing copy is returned.
  /// \param CriticalName Name of the critical region.
  ///
  Value *getOMPCriticalRegionLock(StringRef CriticalName);

  /// Callback type for Atomic Expression update
  /// ex:
  /// \code{.cpp}
  /// unsigned x = 0;
  /// #pragma omp atomic update
  /// x = Expr(x_old);  //Expr() is any legal operation
  /// \endcode
  ///
  /// \param XOld the value of the atomic memory address to use for update
  /// \param IRB reference to the IRBuilder to use
  ///
  /// \returns Value to update X to.
  using AtomicUpdateCallbackTy =
      const function_ref<Value *(Value *XOld, IRBuilder<> &IRB)>;

private:
  enum AtomicKind { Read, Write, Update, Capture };

  /// Determine whether to emit flush or not
  ///
  /// \param Loc    The insert and source location description.
  /// \param AO     The required atomic ordering
  /// \param AK     The OpenMP atomic operation kind used.
  ///
  /// \returns		wether a flush was emitted or not
  bool checkAndEmitFlushAfterAtomic(const LocationDescription &Loc,
                                    AtomicOrdering AO, AtomicKind AK);

  /// Emit atomic update for constructs: X = X BinOp Expr ,or X = Expr BinOp X
  /// For complex Operations: X = UpdateOp(X) => CmpExch X, old_X, UpdateOp(X)
  /// Only Scalar data types.
  ///
  /// \param AllocIP	  Instruction to create AllocaInst before.
  /// \param X			    The target atomic pointer to be updated
  /// \param Expr		    The value to update X with.
  /// \param AO			    Atomic ordering of the generated atomic
  ///                   instructions.
  /// \param RMWOp		  The binary operation used for update. If
  ///                   operation is not supported by atomicRMW,
  ///                   or belong to {FADD, FSUB, BAD_BINOP}.
  ///                   Then a `cmpExch` based	atomic will be generated.
  /// \param UpdateOp 	Code generator for complex expressions that cannot be
  ///                   expressed through atomicrmw instruction.
  /// \param VolatileX	     true if \a X volatile?
  /// \param IsXLHSInRHSPart true if \a X is Left H.S. in Right H.S. part of
  ///                        the update expression, false otherwise.
  ///                        (e.g. true for X = X BinOp Expr)
  ///
  /// \returns A pair of the old value of X before the update, and the value
  ///          used for the update.
  std::pair<Value *, Value *> emitAtomicUpdate(Instruction *AllocIP, Value *X,
                                               Value *Expr, AtomicOrdering AO,
                                               AtomicRMWInst::BinOp RMWOp,
                                               AtomicUpdateCallbackTy &UpdateOp,
                                               bool VolatileX,
                                               bool IsXLHSInRHSPart);

  /// Emit the binary op. described by \p RMWOp, using \p Src1 and \p Src2 .
  ///
  /// \Return The instruction
  Value *emitRMWOpAsInstruction(Value *Src1, Value *Src2,
                                AtomicRMWInst::BinOp RMWOp);

public:
  /// a struct to pack relevant information while generating atomic Ops
  struct AtomicOpValue {
    Value *Var = nullptr;
    bool IsSigned = false;
    bool IsVolatile = false;
  };

  /// Emit atomic Read for : V = X --- Only Scalar data types.
  ///
  /// \param Loc    The insert and source location description.
  /// \param X			The target pointer to be atomically read
  /// \param V			Memory address where to store atomically read
  /// 					    value
  /// \param AO			Atomic ordering of the generated atomic
  /// 					    instructions.
  ///
  /// \return Insertion point after generated atomic read IR.
  InsertPointTy createAtomicRead(const LocationDescription &Loc,
                                 AtomicOpValue &X, AtomicOpValue &V,
                                 AtomicOrdering AO);

  /// Emit atomic write for : X = Expr --- Only Scalar data types.
  ///
  /// \param Loc    The insert and source location description.
  /// \param X			The target pointer to be atomically written to
  /// \param Expr		The value to store.
  /// \param AO			Atomic ordering of the generated atomic
  ///               instructions.
  ///
  /// \return Insertion point after generated atomic Write IR.
  InsertPointTy createAtomicWrite(const LocationDescription &Loc,
                                  AtomicOpValue &X, Value *Expr,
                                  AtomicOrdering AO);

  /// Emit atomic update for constructs: X = X BinOp Expr ,or X = Expr BinOp X
  /// For complex Operations: X = UpdateOp(X) => CmpExch X, old_X, UpdateOp(X)
  /// Only Scalar data types.
  ///
  /// \param Loc      The insert and source location description.
  /// \param AllocIP  Instruction to create AllocaInst before.
  /// \param X        The target atomic pointer to be updated
  /// \param Expr     The value to update X with.
  /// \param AO       Atomic ordering of the generated atomic instructions.
  /// \param RMWOp    The binary operation used for update. If operation
  ///                 is	not supported by atomicRMW, or belong to
  ///	                {FADD, FSUB, BAD_BINOP}. Then a `cmpExch` based
  ///                 atomic will be generated.
  /// \param UpdateOp 	Code generator for complex expressions that cannot be
  ///                   expressed through atomicrmw instruction.
  /// \param IsXLHSInRHSPart true if \a X is Left H.S. in Right H.S. part of
  ///                        the update expression, false otherwise.
  ///	                       (e.g. true for X = X BinOp Expr)
  ///
  /// \return Insertion point after generated atomic update IR.
  InsertPointTy createAtomicUpdate(const LocationDescription &Loc,
                                   Instruction *AllocIP, AtomicOpValue &X,
                                   Value *Expr, AtomicOrdering AO,
                                   AtomicRMWInst::BinOp RMWOp,
                                   AtomicUpdateCallbackTy &UpdateOp,
                                   bool IsXLHSInRHSPart);

  /// Emit atomic update for constructs: --- Only Scalar data types
  /// V = X; X = X BinOp Expr ,
  /// X = X BinOp Expr; V = X,
  /// V = X; X = Expr BinOp X,
  /// X = Expr BinOp X; V = X,
  /// V = X; X = UpdateOp(X),
  /// X = UpdateOp(X); V = X,
  ///
  /// \param Loc        The insert and source location description.
  /// \param AllocIP    Instruction to create AllocaInst before.
  /// \param X          The target atomic pointer to be updated
  /// \param V          Memory address where to store captured value
  /// \param Expr       The value to update X with.
  /// \param AO         Atomic ordering of the generated atomic instructions
  /// \param RMWOp      The binary operation used for update. If
  ///                   operation is not supported by atomicRMW, or belong to
  ///	                  {FADD, FSUB, BAD_BINOP}. Then a cmpExch based
  ///                   atomic will be generated.
  /// \param UpdateOp   Code generator for complex expressions that cannot be
  ///                   expressed through atomicrmw instruction.
  /// \param UpdateExpr true if X is an in place update of the form
  ///                   X = X BinOp Expr or X = Expr BinOp X
  /// \param IsXLHSInRHSPart true if X is Left H.S. in Right H.S. part of the
  ///                        update expression, false otherwise.
  ///                        (e.g. true for X = X BinOp Expr)
  /// \param IsPostfixUpdate true if original value of 'x' must be stored in
  ///                        'v', not an updated one.
  ///
  /// \return Insertion point after generated atomic capture IR.
  InsertPointTy
  createAtomicCapture(const LocationDescription &Loc, Instruction *AllocIP,
                      AtomicOpValue &X, AtomicOpValue &V, Value *Expr,
                      AtomicOrdering AO, AtomicRMWInst::BinOp RMWOp,
                      AtomicUpdateCallbackTy &UpdateOp, bool UpdateExpr,
                      bool IsPostfixUpdate, bool IsXLHSInRHSPart);

  /// Create the control flow structure of a canonical OpenMP loop.
  ///
  /// The emitted loop will be disconnected, i.e. no edge to the loop's
  /// preheader and no terminator in the AfterBB. The OpenMPIRBuilder's
  /// IRBuilder location is not preserved.
  ///
  /// \param DL        DebugLoc used for the instructions in the skeleton.
  /// \param TripCount Value to be used for the trip count.
  /// \param F         Function in which to insert the BasicBlocks.
  /// \param PreInsertBefore  Where to insert BBs that execute before the body,
  ///                         typically the body itself.
  /// \param PostInsertBefore Where to insert BBs that execute after the body.
  /// \param Name      Base name used to derive BB
  ///                  and instruction names.
  ///
  /// \returns The CanonicalLoopInfo that represents the emitted loop.
  CanonicalLoopInfo *createLoopSkeleton(DebugLoc DL, Value *TripCount,
                                        Function *F,
                                        BasicBlock *PreInsertBefore,
                                        BasicBlock *PostInsertBefore,
                                        const Twine &Name = {});
};

/// Class to represented the control flow structure of an OpenMP canonical loop.
///
/// The control-flow structure is standardized for easy consumption by
/// directives associated with loops. For instance, the worksharing-loop
/// construct may change this control flow such that each loop iteration is
/// executed on only one thread. The constraints of a canonical loop in brief
/// are:
///
///  * The number of loop iterations must have been computed before entering the
///    loop.
///
///  * Has an (unsigned) logical induction variable that starts at zero and
///    increments by one.
///
///  * The loop's CFG itself has no side-effects. The OpenMP specification
///    itself allows side-effects, but the order in which they happen, including
///    how often or whether at all, is unspecified. We expect that the frontend
///    will emit those side-effect instructions somewhere (e.g. before the loop)
///    such that the CanonicalLoopInfo itself can be side-effect free.
///
/// Keep in mind that CanonicalLoopInfo is meant to only describe a repeated
/// execution of a loop body that satifies these constraints. It does NOT
/// represent arbitrary SESE regions that happen to contain a loop. Do not use
/// CanonicalLoopInfo for such purposes.
///
/// The control flow can be described as follows:
///
///     Preheader
///        |
///  /-> Header
///  |     |
///  |    Cond---\
///  |     |     |
///  |    Body   |
///  |    | |    |
///  |   <...>   |
///  |    | |    |
///   \--Latch   |
///              |
///             Exit
///              |
///            After
///
/// The loop is thought to start at PreheaderIP (at the Preheader's terminator,
/// including) and end at AfterIP (at the After's first instruction, excluding).
/// That is, instructions in the Preheader and After blocks (except the
/// Preheader's terminator) are out of CanonicalLoopInfo's control and may have
/// side-effects. Typically, the Preheader is used to compute the loop's trip
/// count. The instructions from BodyIP (at the Body block's first instruction,
/// excluding) until the Latch are also considered outside CanonicalLoopInfo's
/// control and thus can have side-effects. The body block is the single entry
/// point into the loop body, which may contain arbitrary control flow as long
/// as all control paths eventually branch to the Latch block.
///
/// TODO: Consider adding another standardized BasicBlock between Body CFG and
/// Latch to guarantee that there is only a single edge to the latch. It would
/// make loop transformations easier to not needing to consider multiple
/// predecessors of the latch (See redirectAllPredecessorsTo) and would give us
/// an equivalant to PreheaderIP, AfterIP and BodyIP for inserting code that
/// executes after each body iteration.
///
/// There must be no loop-carried dependencies through llvm::Values. This is
/// equivalant to that the Latch has no PHINode and the Header's only PHINode is
/// for the induction variable.
///
/// All code in Header, Cond, Latch and Exit (plus the terminator of the
/// Preheader) are CanonicalLoopInfo's responsibility and their build-up checked
/// by assertOK(). They are expected to not be modified unless explicitly
/// modifying the CanonicalLoopInfo through a methods that applies a OpenMP
/// loop-associated construct such as applyWorkshareLoop, tileLoops, unrollLoop,
/// etc. These methods usually invalidate the CanonicalLoopInfo and re-use its
/// basic blocks. After invalidation, the CanonicalLoopInfo must not be used
/// anymore as its underlying control flow may not exist anymore.
/// Loop-transformation methods such as tileLoops, collapseLoops and unrollLoop
/// may also return a new CanonicalLoopInfo that can be passed to other
/// loop-associated construct implementing methods. These loop-transforming
/// methods may either create a new CanonicalLoopInfo usually using
/// createLoopSkeleton and invalidate the input CanonicalLoopInfo, or reuse and
/// modify one of the input CanonicalLoopInfo and return it as representing the
/// modified loop. What is done is an implementation detail of
/// transformation-implementing method and callers should always assume that the
/// CanonicalLoopInfo passed to it is invalidated and a new object is returned.
/// Returned CanonicalLoopInfo have the same structure and guarantees as the one
/// created by createCanonicalLoop, such that transforming methods do not have
/// to special case where the CanonicalLoopInfo originated from.
///
/// Generally, methods consuming CanonicalLoopInfo do not need an
/// OpenMPIRBuilder::InsertPointTy as argument, but use the locations of the
/// CanonicalLoopInfo to insert new or modify existing instructions. Unless
/// documented otherwise, methods consuming CanonicalLoopInfo do not invalidate
/// any InsertPoint that is outside CanonicalLoopInfo's control. Specifically,
/// any InsertPoint in the Preheader, After or Block can still be used after
/// calling such a method.
///
/// TODO: Provide mechanisms for exception handling and cancellation points.
///
/// Defined outside OpenMPIRBuilder because nested classes cannot be
/// forward-declared, e.g. to avoid having to include the entire OMPIRBuilder.h.
class CanonicalLoopInfo {
  friend class OpenMPIRBuilder;

private:
  BasicBlock *Preheader = nullptr;
  BasicBlock *Header = nullptr;
  BasicBlock *Cond = nullptr;
  BasicBlock *Body = nullptr;
  BasicBlock *Latch = nullptr;
  BasicBlock *Exit = nullptr;
  BasicBlock *After = nullptr;

  /// Add the control blocks of this loop to \p BBs.
  ///
  /// This does not include any block from the body, including the one returned
  /// by getBody().
  ///
  /// FIXME: This currently includes the Preheader and After blocks even though
  /// their content is (mostly) not under CanonicalLoopInfo's control.
  /// Re-evaluated whether this makes sense.
  void collectControlBlocks(SmallVectorImpl<BasicBlock *> &BBs);

public:
  /// Returns whether this object currently represents the IR of a loop. If
  /// returning false, it may have been consumed by a loop transformation or not
  /// been intialized. Do not use in this case;
  bool isValid() const { return Header; }

  /// The preheader ensures that there is only a single edge entering the loop.
  /// Code that must be execute before any loop iteration can be emitted here,
  /// such as computing the loop trip count and begin lifetime markers. Code in
  /// the preheader is not considered part of the canonical loop.
  BasicBlock *getPreheader() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Preheader;
  }

  /// The header is the entry for each iteration. In the canonical control flow,
  /// it only contains the PHINode for the induction variable.
  BasicBlock *getHeader() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Header;
  }

  /// The condition block computes whether there is another loop iteration. If
  /// yes, branches to the body; otherwise to the exit block.
  BasicBlock *getCond() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Cond;
  }

  /// The body block is the single entry for a loop iteration and not controlled
  /// by CanonicalLoopInfo. It can contain arbitrary control flow but must
  /// eventually branch to the \p Latch block.
  BasicBlock *getBody() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Body;
  }

  /// Reaching the latch indicates the end of the loop body code. In the
  /// canonical control flow, it only contains the increment of the induction
  /// variable.
  BasicBlock *getLatch() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Latch;
  }

  /// Reaching the exit indicates no more iterations are being executed.
  BasicBlock *getExit() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Exit;
  }

  /// The after block is intended for clean-up code such as lifetime end
  /// markers. It is separate from the exit block to ensure, analogous to the
  /// preheader, it having just a single entry edge and being free from PHI
  /// nodes should there be multiple loop exits (such as from break
  /// statements/cancellations).
  BasicBlock *getAfter() const {
    assert(isValid() && "Requires a valid canonical loop");
    return After;
  }

  /// Returns the llvm::Value containing the number of loop iterations. It must
  /// be valid in the preheader and always interpreted as an unsigned integer of
  /// any bit-width.
  Value *getTripCount() const {
    assert(isValid() && "Requires a valid canonical loop");
    Instruction *CmpI = &Cond->front();
    assert(isa<CmpInst>(CmpI) && "First inst must compare IV with TripCount");
    return CmpI->getOperand(1);
  }

  /// Returns the instruction representing the current logical induction
  /// variable. Always unsigned, always starting at 0 with an increment of one.
  Instruction *getIndVar() const {
    assert(isValid() && "Requires a valid canonical loop");
    Instruction *IndVarPHI = &Header->front();
    assert(isa<PHINode>(IndVarPHI) && "First inst must be the IV PHI");
    return IndVarPHI;
  }

  /// Return the type of the induction variable (and the trip count).
  Type *getIndVarType() const {
    assert(isValid() && "Requires a valid canonical loop");
    return getIndVar()->getType();
  }

  /// Return the insertion point for user code before the loop.
  OpenMPIRBuilder::InsertPointTy getPreheaderIP() const {
    assert(isValid() && "Requires a valid canonical loop");
    return {Preheader, std::prev(Preheader->end())};
  };

  /// Return the insertion point for user code in the body.
  OpenMPIRBuilder::InsertPointTy getBodyIP() const {
    assert(isValid() && "Requires a valid canonical loop");
    return {Body, Body->begin()};
  };

  /// Return the insertion point for user code after the loop.
  OpenMPIRBuilder::InsertPointTy getAfterIP() const {
    assert(isValid() && "Requires a valid canonical loop");
    return {After, After->begin()};
  };

  Function *getFunction() const {
    assert(isValid() && "Requires a valid canonical loop");
    return Header->getParent();
  }

  /// Consistency self-check.
  void assertOK() const;

  /// Invalidate this loop. That is, the underlying IR does not fulfill the
  /// requirements of an OpenMP canonical loop anymore.
  void invalidate();
};

} // end namespace llvm

#endif // LLVM_FRONTEND_OPENMP_OMPIRBUILDER_H
