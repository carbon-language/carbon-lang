//===------ polly/ScopInfo.h - Create Scops from LLVM IR --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the Scop
// detection derived from their LLVM-IR code.
//
// This representation is shared among several tools in the polyhedral
// community, which are e.g. CLooG, Pluto, Loopo, Graphite.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_INFO_H
#define POLLY_SCOP_INFO_H

#include "polly/ScopDetection.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/RegionPass.h"
#include "isl/ctx.h"

#include <forward_list>

using namespace llvm;

namespace llvm {
class Loop;
class LoopInfo;
class PHINode;
class ScalarEvolution;
class SCEV;
class SCEVAddRecExpr;
class Type;
}

struct isl_ctx;
struct isl_map;
struct isl_basic_map;
struct isl_id;
struct isl_set;
struct isl_union_set;
struct isl_union_map;
struct isl_space;
struct isl_ast_build;
struct isl_constraint;
struct isl_pw_multi_aff;

namespace polly {

class IRAccess;
class Scop;
class ScopStmt;
class ScopInfo;
class TempScop;
class SCEVAffFunc;
class Comparison;

/// @brief A class to store information about arrays in the SCoP.
///
/// Objects are accessible via the ScoP, MemoryAccess or the id associated with
/// the MemoryAccess access function.
///
class ScopArrayInfo {
public:
  /// @brief Construct a ScopArrayInfo object.
  ///
  /// @param BasePtr        The array base pointer.
  /// @param ElementType    The type of the elements stored in the array.
  /// @param IslCtx         The isl context used to create the base pointer id.
  /// @param DimensionSizes A vector containing the size of each dimension.
  ScopArrayInfo(Value *BasePtr, Type *ElementType, isl_ctx *IslCtx,
                const SmallVector<const SCEV *, 4> &DimensionSizes);

  /// @brief Destructor to free the isl id of the base pointer.
  ~ScopArrayInfo();

  /// @brief Return the base pointer.
  Value *getBasePtr() const { return BasePtr; }

  /// @brief Return the number of dimensions.
  unsigned getNumberOfDimensions() const { return DimensionSizes.size(); }

  /// @brief Return the size of dimension @p dim.
  const SCEV *getDimensionSize(unsigned dim) const {
    assert(dim < getNumberOfDimensions() && "Invalid dimension");
    return DimensionSizes[dim];
  }

  /// @brief Get the type of the elements stored in this array.
  Type *getElementType() const { return ElementType; }

  /// @brief Get element size in bytes.
  int getElemSizeInBytes() const;

  /// @brief Get the name of this memory reference.
  std::string getName() const;

  /// @brief Return the isl id for the base pointer.
  __isl_give isl_id *getBasePtrId() const;

  /// @brief Dump a readable representation to stderr.
  void dump() const;

  /// @brief Print a readable representation to @p OS.
  void print(raw_ostream &OS) const;

  /// @brief Access the ScopArrayInfo associated with an access function.
  static const ScopArrayInfo *
  getFromAccessFunction(__isl_keep isl_pw_multi_aff *PMA);

  /// @brief Access the ScopArrayInfo associated with an isl Id.
  static const ScopArrayInfo *getFromId(__isl_take isl_id *Id);

private:
  /// @brief The base pointer.
  Value *BasePtr;

  /// @brief The type of the elements stored in this array.
  Type *ElementType;

  /// @brief The isl id for the base pointer.
  isl_id *Id;

  /// @brief The sizes of each dimension.
  SmallVector<const SCEV *, 4> DimensionSizes;
};

/// @brief Represent memory accesses in statements.
class MemoryAccess {
public:
  /// @brief The access type of a memory access
  ///
  /// There are three kind of access types:
  ///
  /// * A read access
  ///
  /// A certain set of memory locations are read and may be used for internal
  /// calculations.
  ///
  /// * A must-write access
  ///
  /// A certain set of memory locations is definitely written. The old value is
  /// replaced by a newly calculated value. The old value is not read or used at
  /// all.
  ///
  /// * A may-write access
  ///
  /// A certain set of memory locations may be written. The memory location may
  /// contain a new value if there is actually a write or the old value may
  /// remain, if no write happens.
  enum AccessType { READ, MUST_WRITE, MAY_WRITE };

  /// @brief Reduction access type
  ///
  /// Commutative and associative binary operations suitable for reductions
  enum ReductionType {
    RT_NONE, ///< Indicate no reduction at all
    RT_ADD,  ///< Addition
    RT_MUL,  ///< Multiplication
    RT_BOR,  ///< Bitwise Or
    RT_BXOR, ///< Bitwise XOr
    RT_BAND, ///< Bitwise And
  };

private:
  MemoryAccess(const MemoryAccess &) = delete;
  const MemoryAccess &operator=(const MemoryAccess &) = delete;

  isl_map *AccessRelation;
  enum AccessType AccType;

  /// @brief The base address (e.g., A for A[i+j]).
  Value *BaseAddr;

  std::string BaseName;
  __isl_give isl_basic_map *createBasicAccessMap(ScopStmt *Statement);
  ScopStmt *Statement;

  /// @brief Reduction type for reduction like accesses, RT_NONE otherwise
  ///
  /// An access is reduction like if it is part of a load-store chain in which
  /// both access the same memory location (use the same LLVM-IR value
  /// as pointer reference). Furthermore, between the load and the store there
  /// is exactly one binary operator which is known to be associative and
  /// commutative.
  ///
  /// TODO:
  ///
  /// We can later lift the constraint that the same LLVM-IR value defines the
  /// memory location to handle scops such as the following:
  ///
  ///    for i
  ///      for j
  ///        sum[i+j] = sum[i] + 3;
  ///
  /// Here not all iterations access the same memory location, but iterations
  /// for which j = 0 holds do. After lifing the equality check in ScopInfo,
  /// subsequent transformations do not only need check if a statement is
  /// reduction like, but they also need to verify that that the reduction
  /// property is only exploited for statement instances that load from and
  /// store to the same data location. Doing so at dependence analysis time
  /// could allow us to handle the above example.
  ReductionType RedType = RT_NONE;

  /// @brief The access instruction of this memory access.
  Instruction *Inst;

  /// Updated access relation read from JSCOP file.
  isl_map *newAccessRelation;

  /// @brief A unique identifier for this memory access.
  ///
  /// The identifier is unique between all memory accesses belonging to the same
  /// scop statement.
  isl_id *Id;

  void assumeNoOutOfBound(const IRAccess &Access);

  /// @brief Compute bounds on an over approximated  access relation.
  ///
  /// @param ElementSize The size of one element accessed.
  void computeBoundsOnAccessRelation(unsigned ElementSize);

  /// @brief Get the original access function as read from IR.
  __isl_give isl_map *getOriginalAccessRelation() const;

  /// @brief Return the space in which the access relation lives in.
  __isl_give isl_space *getOriginalAccessRelationSpace() const;

  /// @brief Get the new access function imported or set by a pass
  __isl_give isl_map *getNewAccessRelation() const;

  /// @brief Fold the memory access to consider parameteric offsets
  ///
  /// To recover memory accesses with array size parameters in the subscript
  /// expression we post-process the delinearization results.
  ///
  /// We would normally recover from an access A[exp0(i) * N + exp1(i)] into an
  /// array A[][N] the 2D access A[exp0(i)][exp1(i)]. However, another valid
  /// delinearization is A[exp0(i) - 1][exp1(i) + N] which - depending on the
  /// range of exp1(i) - may be preferrable. Specifically, for cases where we
  /// know exp1(i) is negative, we want to choose the latter expression.
  ///
  /// As we commonly do not have any information about the range of exp1(i),
  /// we do not choose one of the two options, but instead create a piecewise
  /// access function that adds the (-1, N) offsets as soon as exp1(i) becomes
  /// negative. For a 2D array such an access function is created by applying
  /// the piecewise map:
  ///
  /// [i,j] -> [i, j] :      j >= 0
  /// [i,j] -> [i-1, j+N] :  j <  0
  ///
  /// We can generalize this mapping to arbitrary dimensions by applying this
  /// piecewise mapping pairwise from the rightmost to the leftmost access
  /// dimension. It would also be possible to cover a wider range by introducing
  /// more cases and adding multiple of Ns to these cases. However, this has
  /// not yet been necessary.
  /// The introduction of different cases necessarily complicates the memory
  /// access function, but cases that can be statically proven to not happen
  /// will be eliminated later on.
  __isl_give isl_map *foldAccess(const IRAccess &Access,
                                 __isl_take isl_map *AccessRelation,
                                 ScopStmt *Statement);

public:
  /// @brief Create a memory access from an access in LLVM-IR.
  ///
  /// @param Access     The memory access.
  /// @param AccInst    The access instruction.
  /// @param Statement  The statement that contains the access.
  /// @param SAI        The ScopArrayInfo object for this base pointer.
  /// @param Identifier An identifier that is unique for all memory accesses
  ///                   belonging to the same scop statement.
  MemoryAccess(const IRAccess &Access, Instruction *AccInst,
               ScopStmt *Statement, const ScopArrayInfo *SAI, int Identifier);

  ~MemoryAccess();

  /// @brief Get the type of a memory access.
  enum AccessType getType() { return AccType; }

  /// @brief Is this a reduction like access?
  bool isReductionLike() const { return RedType != RT_NONE; }

  /// @brief Is this a read memory access?
  bool isRead() const { return AccType == MemoryAccess::READ; }

  /// @brief Is this a must-write memory access?
  bool isMustWrite() const { return AccType == MemoryAccess::MUST_WRITE; }

  /// @brief Is this a may-write memory access?
  bool isMayWrite() const { return AccType == MemoryAccess::MAY_WRITE; }

  /// @brief Is this a write memory access?
  bool isWrite() const { return isMustWrite() || isMayWrite(); }

  /// @brief Check if a new access relation was imported or set by a pass.
  bool hasNewAccessRelation() const { return newAccessRelation; }

  /// @brief Return the newest access relation of this access.
  ///
  /// There are two possibilities:
  ///   1) The original access relation read from the LLVM-IR.
  ///   2) A new access relation imported from a json file or set by another
  ///      pass (e.g., for privatization).
  ///
  /// As 2) is by construction "newer" than 1) we return the new access
  /// relation if present.
  ///
  isl_map *getAccessRelation() const {
    return hasNewAccessRelation() ? getNewAccessRelation()
                                  : getOriginalAccessRelation();
  }

  /// @brief Return the access relation after the schedule was applied.
  __isl_give isl_pw_multi_aff *
  applyScheduleToAccessRelation(__isl_take isl_union_map *Schedule) const;

  /// @brief Get an isl string representing the access function read from IR.
  std::string getOriginalAccessRelationStr() const;

  /// @brief Get the base address of this access (e.g. A for A[i+j]).
  Value *getBaseAddr() const { return BaseAddr; }

  /// @brief Get the base array isl_id for this access.
  __isl_give isl_id *getArrayId() const;

  /// @brief Get the ScopArrayInfo object for the base address.
  const ScopArrayInfo *getScopArrayInfo() const;

  /// @brief Return a string representation of the accesse's reduction type.
  const std::string getReductionOperatorStr() const;

  /// @brief Return a string representation of the reduction type @p RT.
  static const std::string getReductionOperatorStr(ReductionType RT);

  const std::string &getBaseName() const { return BaseName; }

  /// @brief Return the access instruction of this memory access.
  Instruction *getAccessInstruction() const { return Inst; }

  /// Get the stride of this memory access in the specified Schedule. Schedule
  /// is a map from the statement to a schedule where the innermost dimension is
  /// the dimension of the innermost loop containing the statement.
  __isl_give isl_set *getStride(__isl_take const isl_map *Schedule) const;

  /// Is the stride of the access equal to a certain width? Schedule is a map
  /// from the statement to a schedule where the innermost dimension is the
  /// dimension of the innermost loop containing the statement.
  bool isStrideX(__isl_take const isl_map *Schedule, int StrideWidth) const;

  /// Is consecutive memory accessed for a given statement instance set?
  /// Schedule is a map from the statement to a schedule where the innermost
  /// dimension is the dimension of the innermost loop containing the
  /// statement.
  bool isStrideOne(__isl_take const isl_map *Schedule) const;

  /// Is always the same memory accessed for a given statement instance set?
  /// Schedule is a map from the statement to a schedule where the innermost
  /// dimension is the dimension of the innermost loop containing the
  /// statement.
  bool isStrideZero(__isl_take const isl_map *Schedule) const;

  /// @brief Check if this is a scalar memory access.
  bool isScalar() const;

  /// @brief Get the statement that contains this memory access.
  ScopStmt *getStatement() const { return Statement; }

  /// @brief Get the reduction type of this access
  ReductionType getReductionType() const { return RedType; }

  /// @brief Set the updated access relation read from JSCOP file.
  void setNewAccessRelation(__isl_take isl_map *newAccessRelation);

  /// @brief Mark this a reduction like access
  void markAsReductionLike(ReductionType RT) { RedType = RT; }

  /// @brief Align the parameters in the access relation to the scop context
  void realignParams();

  /// @brief Get identifier for the memory access.
  ///
  /// This identifier is unique for all accesses that belong to the same scop
  /// statement.
  __isl_give isl_id *getId() const;

  /// @brief Print the MemoryAccess.
  ///
  /// @param OS The output stream the MemoryAccess is printed to.
  void print(raw_ostream &OS) const;

  /// @brief Print the MemoryAccess to stderr.
  void dump() const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              MemoryAccess::ReductionType RT);

///===----------------------------------------------------------------------===//
/// @brief Statement of the Scop
///
/// A Scop statement represents an instruction in the Scop.
///
/// It is further described by its iteration domain, its schedule and its data
/// accesses.
/// At the moment every statement represents a single basic block of LLVM-IR.
class ScopStmt {
public:
  /// @brief List to hold all (scalar) memory accesses mapped to an instruction.
  using MemoryAccessList = std::forward_list<MemoryAccess>;

private:
  ScopStmt(const ScopStmt &) = delete;
  const ScopStmt &operator=(const ScopStmt &) = delete;

  /// Polyhedral description
  //@{

  /// The Scop containing this ScopStmt
  Scop &Parent;

  /// The iteration domain describes the set of iterations for which this
  /// statement is executed.
  ///
  /// Example:
  ///     for (i = 0; i < 100 + b; ++i)
  ///       for (j = 0; j < i; ++j)
  ///         S(i,j);
  ///
  /// 'S' is executed for different values of i and j. A vector of all
  /// induction variables around S (i, j) is called iteration vector.
  /// The domain describes the set of possible iteration vectors.
  ///
  /// In this case it is:
  ///
  ///     Domain: 0 <= i <= 100 + b
  ///             0 <= j <= i
  ///
  /// A pair of statement and iteration vector (S, (5,3)) is called statement
  /// instance.
  isl_set *Domain;

  /// The schedule map describes the execution order of the statement
  /// instances.
  ///
  /// A statement and its iteration domain do not give any information about the
  /// order in time in which the different statement instances are executed.
  /// This information is provided by the schedule.
  ///
  /// The schedule maps every instance of each statement into a multi
  /// dimensional schedule space. This space can be seen as a multi
  /// dimensional clock.
  ///
  /// Example:
  ///
  /// <S,(5,4)>  may be mapped to (5,4) by this schedule:
  ///
  /// s0 = i (Year of execution)
  /// s1 = j (Day of execution)
  ///
  /// or to (9, 20) by this schedule:
  ///
  /// s0 = i + j (Year of execution)
  /// s1 = 20 (Day of execution)
  ///
  /// The order statement instances are executed is defined by the
  /// schedule vectors they are mapped to. A statement instance
  /// <A, (i, j, ..)> is executed before a statement instance <B, (i', ..)>, if
  /// the schedule vector of A is lexicographic smaller than the schedule
  /// vector of B.
  isl_map *Schedule;

  /// The memory accesses of this statement.
  ///
  /// The only side effects of a statement are its memory accesses.
  typedef SmallVector<MemoryAccess *, 8> MemoryAccessVec;
  MemoryAccessVec MemAccs;

  /// @brief Mapping from instructions to (scalar) memory accesses.
  DenseMap<const Instruction *, MemoryAccessList *> InstructionToAccess;

  //@}

  /// @brief A SCoP statement represents either a basic block (affine/precise
  ///        case) or a whole region (non-affine case). Only one of the
  ///        following two members will therefore be set and indicate which
  ///        kind of statement this is.
  ///
  ///{

  /// @brief The BasicBlock represented by this statement (in the affine case).
  BasicBlock *BB;

  /// @brief The region represented by this statement (in the non-affine case).
  Region *R;

  ///}

  /// @brief The isl AST build for the new generated AST.
  isl_ast_build *Build;

  std::vector<Loop *> NestLoops;

  std::string BaseName;

  /// Build the statement.
  //@{
  __isl_give isl_set *buildConditionSet(const Comparison &Cmp);
  __isl_give isl_set *addConditionsToDomain(__isl_take isl_set *Domain,
                                            TempScop &tempScop,
                                            const Region &CurRegion);
  __isl_give isl_set *addLoopBoundsToDomain(__isl_take isl_set *Domain,
                                            TempScop &tempScop);
  __isl_give isl_set *buildDomain(TempScop &tempScop, const Region &CurRegion);
  void buildSchedule(SmallVectorImpl<unsigned> &ScheduleVec);

  /// @brief Create the accesses for instructions in @p Block.
  ///
  /// @param tempScop       The template SCoP.
  /// @param Block          The basic block for which accesses should be
  ///                       created.
  /// @param isApproximated Flag to indicate blocks that might not be executed,
  ///                       hence for which write accesses need to be modeled as
  ///                       may-write accesses.
  void buildAccesses(TempScop &tempScop, BasicBlock *Block,
                     bool isApproximated = false);

  /// @brief Detect and mark reductions in the ScopStmt
  void checkForReductions();

  /// @brief Collect loads which might form a reduction chain with @p StoreMA
  void
  collectCandiateReductionLoads(MemoryAccess *StoreMA,
                                llvm::SmallVectorImpl<MemoryAccess *> &Loads);
  //@}

  /// @brief Derive assumptions about parameter values from GetElementPtrInst
  ///
  /// In case a GEP instruction references into a fixed size array e.g., an
  /// access A[i][j] into an array A[100x100], LLVM-IR does not guarantee that
  /// the subscripts always compute values that are within array bounds. In this
  /// function we derive the set of parameter values for which all accesses are
  /// within bounds and add the assumption that the scop is only every executed
  /// with this set of parameter values.
  ///
  /// Example:
  ///
  ///   void foo(float A[][20], long n, long m {
  ///     for (long i = 0; i < n; i++)
  ///       for (long j = 0; j < m; j++)
  ///         A[i][j] = ...
  ///
  /// This loop yields out-of-bound accesses if m is at least 20 and at the same
  /// time at least one iteration of the outer loop is executed. Hence, we
  /// assume:
  ///
  ///   n <= 0 or m <= 20.
  ///
  /// TODO: The location where the GEP instruction is executed is not
  /// necessarily the location where the memory is actually accessed. As a
  /// result scanning for GEP[s] is imprecise. Even though this is not a
  /// correctness problem, this imprecision may result in missed optimizations
  /// or non-optimal run-time checks.
  void deriveAssumptionsFromGEP(GetElementPtrInst *Inst);

  /// @brief Scan @p Block and derive assumptions about parameter values.
  void deriveAssumptions(BasicBlock *Block);

  /// Create the ScopStmt from a BasicBlock.
  ScopStmt(Scop &parent, TempScop &tempScop, const Region &CurRegion,
           BasicBlock &bb, SmallVectorImpl<Loop *> &NestLoops,
           SmallVectorImpl<unsigned> &ScheduleVec);

  /// Create an overapproximating ScopStmt for the region @p R.
  ScopStmt(Scop &parent, TempScop &tempScop, const Region &CurRegion, Region &R,
           SmallVectorImpl<Loop *> &NestLoops,
           SmallVectorImpl<unsigned> &ScheduleVec);

  friend class Scop;

public:
  ~ScopStmt();

  /// @brief Get an isl_ctx pointer.
  isl_ctx *getIslCtx() const;

  /// @brief Get the iteration domain of this ScopStmt.
  ///
  /// @return The iteration domain of this ScopStmt.
  __isl_give isl_set *getDomain() const;

  /// @brief Get the space of the iteration domain
  ///
  /// @return The space of the iteration domain
  __isl_give isl_space *getDomainSpace() const;

  /// @brief Get the id of the iteration domain space
  ///
  /// @return The id of the iteration domain space
  __isl_give isl_id *getDomainId() const;

  /// @brief Get an isl string representing this domain.
  std::string getDomainStr() const;

  /// @brief Get the schedule function of this ScopStmt.
  ///
  /// @return The schedule function of this ScopStmt.
  __isl_give isl_map *getSchedule() const;
  void setSchedule(__isl_take isl_map *Schedule);

  /// @brief Get an isl string representing this schedule.
  std::string getScheduleStr() const;

  /// @brief Get the BasicBlock represented by this ScopStmt (if any).
  ///
  /// @return The BasicBlock represented by this ScopStmt, or null if the
  ///         statement represents a region.
  BasicBlock *getBasicBlock() const { return BB; }

  /// @brief Return true if this statement represents a single basic block.
  bool isBlockStmt() const { return BB != nullptr; }

  /// @brief Get the region represented by this ScopStmt (if any).
  ///
  /// @return The region represented by this ScopStmt, or null if the statement
  ///         represents a basic block.
  Region *getRegion() const { return R; }

  /// @brief Return true if this statement represents a whole region.
  bool isRegionStmt() const { return R != nullptr; }

  /// @brief Return the (scalar) memory accesses for @p Inst.
  const MemoryAccessList &getAccessesFor(const Instruction *Inst) const {
    MemoryAccessList *MAL = lookupAccessesFor(Inst);
    assert(MAL && "Cannot get memory accesses because they do not exist!");
    return *MAL;
  }

  /// @brief Return the (scalar) memory accesses for @p Inst if any.
  MemoryAccessList *lookupAccessesFor(const Instruction *Inst) const {
    auto It = InstructionToAccess.find(Inst);
    return It == InstructionToAccess.end() ? nullptr : It->getSecond();
  }

  /// @brief Return the __first__ (scalar) memory access for @p Inst.
  const MemoryAccess &getAccessFor(const Instruction *Inst) const {
    MemoryAccess *MA = lookupAccessFor(Inst);
    assert(MA && "Cannot get memory access because it does not exist!");
    return *MA;
  }

  /// @brief Return the __first__ (scalar) memory access for @p Inst if any.
  MemoryAccess *lookupAccessFor(const Instruction *Inst) const {
    auto It = InstructionToAccess.find(Inst);
    return It == InstructionToAccess.end() ? nullptr
                                           : &It->getSecond()->front();
  }

  void setBasicBlock(BasicBlock *Block) {
    // TODO: Handle the case where the statement is a region statement, thus
    //       the entry block was split and needs to be changed in the region R.
    assert(BB && "Cannot set a block for a region statement");
    BB = Block;
  }

  typedef MemoryAccessVec::iterator iterator;
  typedef MemoryAccessVec::const_iterator const_iterator;

  iterator begin() { return MemAccs.begin(); }
  iterator end() { return MemAccs.end(); }
  const_iterator begin() const { return MemAccs.begin(); }
  const_iterator end() const { return MemAccs.end(); }

  unsigned getNumParams() const;
  unsigned getNumIterators() const;
  unsigned getNumSchedule() const;

  Scop *getParent() { return &Parent; }
  const Scop *getParent() const { return &Parent; }

  const char *getBaseName() const;

  /// @brief Set the isl AST build.
  void setAstBuild(__isl_keep isl_ast_build *B) { Build = B; }

  /// @brief Get the isl AST build.
  __isl_keep isl_ast_build *getAstBuild() const { return Build; }

  /// @brief Restrict the domain of the statement.
  ///
  /// @param NewDomain The new statement domain.
  void restrictDomain(__isl_take isl_set *NewDomain);

  /// @brief Get the loop for a dimension.
  ///
  /// @param Dimension The dimension of the induction variable
  /// @return The loop at a certain dimension.
  const Loop *getLoopForDimension(unsigned Dimension) const;

  /// @brief Align the parameters in the statement to the scop context
  void realignParams();

  /// @brief Print the ScopStmt.
  ///
  /// @param OS The output stream the ScopStmt is printed to.
  void print(raw_ostream &OS) const;

  /// @brief Print the ScopStmt to stderr.
  void dump() const;
};

/// @brief Print ScopStmt S to raw_ostream O.
static inline raw_ostream &operator<<(raw_ostream &O, const ScopStmt &S) {
  S.print(O);
  return O;
}

///===----------------------------------------------------------------------===//
/// @brief Static Control Part
///
/// A Scop is the polyhedral representation of a control flow region detected
/// by the Scop detection. It is generated by translating the LLVM-IR and
/// abstracting its effects.
///
/// A Scop consists of a set of:
///
///   * A set of statements executed in the Scop.
///
///   * A set of global parameters
///   Those parameters are scalar integer values, which are constant during
///   execution.
///
///   * A context
///   This context contains information about the values the parameters
///   can take and relations between different parameters.
class Scop {
public:
  /// @brief Type to represent a pair of minimal/maximal access to an array.
  using MinMaxAccessTy = std::pair<isl_pw_multi_aff *, isl_pw_multi_aff *>;

  /// @brief Vector of minimal/maximal accesses to different arrays.
  using MinMaxVectorTy = SmallVector<MinMaxAccessTy, 4>;

  /// @brief Vector of minimal/maximal access vectors one for each alias group.
  using MinMaxVectorVectorTy = SmallVector<MinMaxVectorTy *, 4>;

private:
  Scop(const Scop &) = delete;
  const Scop &operator=(const Scop &) = delete;

  ScalarEvolution *SE;

  /// The underlying Region.
  Region &R;

  /// Flag to indicate that the scheduler actually optimized the SCoP.
  bool IsOptimized;

  /// Max loop depth.
  unsigned MaxLoopDepth;

  typedef std::vector<std::unique_ptr<ScopStmt>> StmtSet;
  /// The statements in this Scop.
  StmtSet Stmts;

  /// Parameters of this Scop
  typedef SmallVector<const SCEV *, 8> ParamVecType;
  ParamVecType Parameters;

  /// The isl_ids that are used to represent the parameters
  typedef std::map<const SCEV *, int> ParamIdType;
  ParamIdType ParameterIds;

  /// Isl context.
  isl_ctx *IslCtx;

  /// @brief A map from basic blocks to SCoP statements.
  DenseMap<BasicBlock *, ScopStmt *> StmtMap;

  /// Constraints on parameters.
  isl_set *Context;

  typedef MapVector<const Value *, const ScopArrayInfo *> ArrayInfoMapTy;
  /// @brief A map to remember ScopArrayInfo objects for all base pointers.
  ArrayInfoMapTy ScopArrayInfoMap;

  /// @brief The assumptions under which this scop was built.
  ///
  /// When constructing a scop sometimes the exact representation of a statement
  /// or condition would be very complex, but there is a common case which is a
  /// lot simpler, but which is only valid under certain assumptions. The
  /// assumed context records the assumptions taken during the construction of
  /// this scop and that need to be code generated as a run-time test.
  isl_set *AssumedContext;

  /// @brief The set of minimal/maximal accesses for each alias group.
  ///
  /// When building runtime alias checks we look at all memory instructions and
  /// build so called alias groups. Each group contains a set of accesses to
  /// different base arrays which might alias with each other. However, between
  /// alias groups there is no aliasing possible.
  ///
  /// In a program with int and float pointers annotated with tbaa information
  /// we would probably generate two alias groups, one for the int pointers and
  /// one for the float pointers.
  ///
  /// During code generation we will create a runtime alias check for each alias
  /// group to ensure the SCoP is executed in an alias free environment.
  MinMaxVectorVectorTy MinMaxAliasGroups;

  /// Create the static control part with a region, max loop depth of this
  /// region and parameters used in this region.
  Scop(TempScop &TempScop, LoopInfo &LI, ScalarEvolution &SE, ScopDetection &SD,
       isl_ctx *ctx);

  /// @brief Check if a basic block is trivial.
  ///
  /// A trivial basic block does not contain any useful calculation. Therefore,
  /// it does not need to be represented as a polyhedral statement.
  ///
  /// @param BB The basic block to check
  /// @param tempScop TempScop returning further information regarding the Scop.
  ///
  /// @return True if the basic block is trivial, otherwise false.
  static bool isTrivialBB(BasicBlock *BB, TempScop &tempScop);

  /// @brief Build the Context of the Scop.
  void buildContext();

  /// @brief Add the bounds of the parameters to the context.
  void addParameterBounds();

  /// @brief Simplify the assumed context.
  void simplifyAssumedContext();

  /// @brief Create a new SCoP statement for either @p BB or @p R.
  ///
  /// Either @p BB or @p R should be non-null. A new statement for the non-null
  /// argument will be created and added to the statement vector and map.
  ///
  /// @param BB         The basic block we build the statement for (or null)
  /// @param R          The region we build the statement for (or null).
  /// @param tempScop   The temp SCoP we use as model.
  /// @param CurRegion  The SCoP region.
  /// @param NestLoops  A vector of all surrounding loops.
  /// @param Schedule   The position of the new statement as schedule.
  void addScopStmt(BasicBlock *BB, Region *R, TempScop &tempScop,
                   const Region &CurRegion, SmallVectorImpl<Loop *> &NestLoops,
                   SmallVectorImpl<unsigned> &Schedule);

  /// Build the Scop and Statement with precalculated scop information.
  void buildScop(TempScop &TempScop, const Region &CurRegion,
                 // Loops in Scop containing CurRegion
                 SmallVectorImpl<Loop *> &NestLoops,
                 // The schedule numbers
                 SmallVectorImpl<unsigned> &Schedule, LoopInfo &LI,
                 ScopDetection &SD);

  /// @name Helper function for printing the Scop.
  ///
  ///{
  void printContext(raw_ostream &OS) const;
  void printArrayInfo(raw_ostream &OS) const;
  void printStatements(raw_ostream &OS) const;
  void printAliasAssumptions(raw_ostream &OS) const;
  ///}

  friend class ScopInfo;

public:
  ~Scop();

  ScalarEvolution *getSE() const;

  /// @brief Get the count of parameters used in this Scop.
  ///
  /// @return The count of parameters used in this Scop.
  inline ParamVecType::size_type getNumParams() const {
    return Parameters.size();
  }

  /// @brief Get a set containing the parameters used in this Scop
  ///
  /// @return The set containing the parameters used in this Scop.
  inline const ParamVecType &getParams() const { return Parameters; }

  /// @brief Take a list of parameters and add the new ones to the scop.
  void addParams(std::vector<const SCEV *> NewParameters);

  int getNumArrays() { return ScopArrayInfoMap.size(); }

  typedef iterator_range<ArrayInfoMapTy::iterator> array_range;
  typedef iterator_range<ArrayInfoMapTy::const_iterator> const_array_range;

  inline array_range arrays() {
    return array_range(ScopArrayInfoMap.begin(), ScopArrayInfoMap.end());
  }

  inline const_array_range arrays() const {
    return const_array_range(ScopArrayInfoMap.begin(), ScopArrayInfoMap.end());
  }

  /// @brief Return the isl_id that represents a certain parameter.
  ///
  /// @param Parameter A SCEV that was recognized as a Parameter.
  ///
  /// @return The corresponding isl_id or NULL otherwise.
  isl_id *getIdForParam(const SCEV *Parameter) const;

  /// @name Parameter Iterators
  ///
  /// These iterators iterate over all parameters of this Scop.
  //@{
  typedef ParamVecType::iterator param_iterator;
  typedef ParamVecType::const_iterator const_param_iterator;

  param_iterator param_begin() { return Parameters.begin(); }
  param_iterator param_end() { return Parameters.end(); }
  const_param_iterator param_begin() const { return Parameters.begin(); }
  const_param_iterator param_end() const { return Parameters.end(); }
  //@}

  /// @brief Get the maximum region of this static control part.
  ///
  /// @return The maximum region of this static control part.
  inline const Region &getRegion() const { return R; }
  inline Region &getRegion() { return R; }

  /// @brief Get the maximum depth of the loop.
  ///
  /// @return The maximum depth of the loop.
  inline unsigned getMaxLoopDepth() const { return MaxLoopDepth; }

  /// @brief Get the schedule dimension number of this Scop.
  ///
  /// @return The schedule dimension number of this Scop.
  inline unsigned getScheduleDim() const {
    unsigned maxScheduleDim = 0;

    for (const_iterator SI = begin(), SE = end(); SI != SE; ++SI)
      maxScheduleDim = std::max(maxScheduleDim, (*SI)->getNumSchedule());

    return maxScheduleDim;
  }

  /// @brief Mark the SCoP as optimized by the scheduler.
  void markAsOptimized() { IsOptimized = true; }

  /// @brief Check if the SCoP has been optimized by the scheduler.
  bool isOptimized() const { return IsOptimized; }

  /// @brief Get the name of this Scop.
  std::string getNameStr() const;

  /// @brief Get the constraint on parameter of this Scop.
  ///
  /// @return The constraint on parameter of this Scop.
  __isl_give isl_set *getContext() const;
  __isl_give isl_space *getParamSpace() const;

  /// @brief Get the assumed context for this Scop.
  ///
  /// @return The assumed context of this Scop.
  __isl_give isl_set *getAssumedContext() const;

  /// @brief Add assumptions to assumed context.
  ///
  /// The assumptions added will be assumed to hold during the execution of the
  /// scop. However, as they are generally not statically provable, at code
  /// generation time run-time checks will be generated that ensure the
  /// assumptions hold.
  ///
  /// WARNING: We currently exploit in simplifyAssumedContext the knowledge
  ///          that assumptions do not change the set of statement instances
  ///          executed.
  ///
  /// @param Set A set describing relations between parameters that are assumed
  ///            to hold.
  void addAssumption(__isl_take isl_set *Set);

  /// @brief Build all alias groups for this SCoP.
  ///
  /// @returns True if __no__ error occurred, false otherwise.
  bool buildAliasGroups(AliasAnalysis &AA);

  //// @brief Drop all constant dimensions from statment schedules.
  ///
  ///  Schedule dimensions that are constant accross the scop do not carry
  ///  any information, but would cost compile time due to the increased number
  ///  of schedule dimensions. To not pay this cost, we remove them.
  void dropConstantScheduleDims();

  /// @brief Return all alias groups for this SCoP.
  const MinMaxVectorVectorTy &getAliasGroups() const {
    return MinMaxAliasGroups;
  }

  /// @brief Get an isl string representing the context.
  std::string getContextStr() const;

  /// @brief Get an isl string representing the assumed context.
  std::string getAssumedContextStr() const;

  /// @brief Return the stmt for the given @p BB or nullptr if none.
  ScopStmt *getStmtForBasicBlock(BasicBlock *BB) const;

  /// @brief Return the number of statements in the SCoP.
  size_t getSize() const { return Stmts.size(); }

  /// @name Statements Iterators
  ///
  /// These iterators iterate over all statements of this Scop.
  //@{
  typedef StmtSet::iterator iterator;
  typedef StmtSet::const_iterator const_iterator;

  iterator begin() { return Stmts.begin(); }
  iterator end() { return Stmts.end(); }
  const_iterator begin() const { return Stmts.begin(); }
  const_iterator end() const { return Stmts.end(); }

  typedef StmtSet::reverse_iterator reverse_iterator;
  typedef StmtSet::const_reverse_iterator const_reverse_iterator;

  reverse_iterator rbegin() { return Stmts.rbegin(); }
  reverse_iterator rend() { return Stmts.rend(); }
  const_reverse_iterator rbegin() const { return Stmts.rbegin(); }
  const_reverse_iterator rend() const { return Stmts.rend(); }
  //@}

  /// @brief Return the (possibly new) ScopArrayInfo object for @p Access.
  ///
  /// @param ElementType The type of the elements stored in this array.
  const ScopArrayInfo *
  getOrCreateScopArrayInfo(Value *BasePtr, Type *ElementType,
                           const SmallVector<const SCEV *, 4> &Sizes);

  /// @brief Return the cached ScopArrayInfo object for @p BasePtr.
  const ScopArrayInfo *getScopArrayInfo(Value *BasePtr);

  void setContext(isl_set *NewContext);

  /// @brief Align the parameters in the statement to the scop context
  void realignParams();

  /// @brief Print the static control part.
  ///
  /// @param OS The output stream the static control part is printed to.
  void print(raw_ostream &OS) const;

  /// @brief Print the ScopStmt to stderr.
  void dump() const;

  /// @brief Get the isl context of this static control part.
  ///
  /// @return The isl context of this static control part.
  isl_ctx *getIslCtx() const;

  /// @brief Get a union set containing the iteration domains of all statements.
  __isl_give isl_union_set *getDomains();

  /// @brief Get a union map of all may-writes performed in the SCoP.
  __isl_give isl_union_map *getMayWrites();

  /// @brief Get a union map of all must-writes performed in the SCoP.
  __isl_give isl_union_map *getMustWrites();

  /// @brief Get a union map of all writes performed in the SCoP.
  __isl_give isl_union_map *getWrites();

  /// @brief Get a union map of all reads performed in the SCoP.
  __isl_give isl_union_map *getReads();

  /// @brief Get the schedule of all the statements in the SCoP.
  __isl_give isl_union_map *getSchedule();

  /// @brief Intersects the domains of all statements in the SCoP.
  ///
  /// @return true if a change was made
  bool restrictDomains(__isl_take isl_union_set *Domain);
};

/// @brief Print Scop scop to raw_ostream O.
static inline raw_ostream &operator<<(raw_ostream &O, const Scop &scop) {
  scop.print(O);
  return O;
}

///===---------------------------------------------------------------------===//
/// @brief Build the Polly IR (Scop and ScopStmt) on a Region.
///
class ScopInfo : public RegionPass {
  //===-------------------------------------------------------------------===//
  ScopInfo(const ScopInfo &) = delete;
  const ScopInfo &operator=(const ScopInfo &) = delete;

  // The Scop
  Scop *scop;
  isl_ctx *ctx;

  void clear() {
    if (scop) {
      delete scop;
      scop = 0;
    }
  }

public:
  static char ID;
  explicit ScopInfo();
  ~ScopInfo();

  /// @brief Try to build the Polly IR of static control part on the current
  ///        SESE-Region.
  ///
  /// @return If the current region is a valid for a static control part,
  ///         return the Polly IR representing this static control part,
  ///         return null otherwise.
  Scop *getScop() { return scop; }
  const Scop *getScop() const { return scop; }

  /// @name RegionPass interface
  //@{
  virtual bool runOnRegion(Region *R, RGPassManager &RGM);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void releaseMemory() { clear(); }
  virtual void print(raw_ostream &OS, const Module *) const {
    if (scop)
      scop->print(OS);
    else
      OS << "Invalid Scop!\n";
  }
  //@}
};

} // end namespace polly

namespace llvm {
class PassRegistry;
void initializeScopInfoPass(llvm::PassRegistry &);
}

#endif
