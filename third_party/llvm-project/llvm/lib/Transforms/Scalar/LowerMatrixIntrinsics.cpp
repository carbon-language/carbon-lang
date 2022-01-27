//===- LowerMatrixIntrinsics.cpp -  Lower matrix intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower matrix intrinsics to vector operations.
//
// TODO:
//  * Improve fusion:
//   * Support more cases, e.g. multiply-add, multiply-sub, operands/results
//     transposed.
//   * Improve cost-modeling, e.g. choose different number of rows/columns
//     columns for tiles, consider cost of copies on alias.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/MatrixUtils.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "lower-matrix-intrinsics"

static cl::opt<bool>
    FuseMatrix("fuse-matrix", cl::init(true), cl::Hidden,
               cl::desc("Enable/disable fusing matrix instructions."));
// TODO: Allow and use non-square tiles.
static cl::opt<unsigned> TileSize(
    "fuse-matrix-tile-size", cl::init(4), cl::Hidden,
    cl::desc(
        "Tile size for matrix instruction fusion using square-shaped tiles."));
static cl::opt<bool> TileUseLoops("fuse-matrix-use-loops", cl::init(false),
                                  cl::Hidden,
                                  cl::desc("Generate loop nest for tiling."));
static cl::opt<bool> ForceFusion(
    "force-fuse-matrix", cl::init(false), cl::Hidden,
    cl::desc("Force matrix instruction fusion even if not profitable."));
static cl::opt<bool> AllowContractEnabled(
    "matrix-allow-contract", cl::init(false), cl::Hidden,
    cl::desc("Allow the use of FMAs if available and profitable. This may "
             "result in different results, due to less rounding error."));

enum class MatrixLayoutTy { ColumnMajor, RowMajor };

static cl::opt<MatrixLayoutTy> MatrixLayout(
    "matrix-default-layout", cl::init(MatrixLayoutTy::ColumnMajor),
    cl::desc("Sets the default matrix layout"),
    cl::values(clEnumValN(MatrixLayoutTy::ColumnMajor, "column-major",
                          "Use column-major layout"),
               clEnumValN(MatrixLayoutTy::RowMajor, "row-major",
                          "Use row-major layout")));

/// Helper function to either return Scope, if it is a subprogram or the
/// attached subprogram for a local scope.
static DISubprogram *getSubprogram(DIScope *Scope) {
  if (auto *Subprogram = dyn_cast<DISubprogram>(Scope))
    return Subprogram;
  return cast<DILocalScope>(Scope)->getSubprogram();
}

namespace {

// Given an element pointer \p BasePtr to the start of a (sub) matrix, compute
// the start address of vector \p VecIdx with type (\p EltType x \p NumElements)
// assuming \p Stride elements between start two consecutive vectors.
// \p Stride must be >= \p NumElements.
// For column-major matrixes, the function computes the address of a column
// vectors and \p NumElements must be set to the number of elements in a column
// (= number of rows of the matrix). For row-major matrixes, the function
// computes the address of a row vector and \p NumElements must be set to the
// number of elements in a column (= number of columns of the matrix).
//
// Consider a 4x4 matrix in column-mjaor layout like below
//
//      0       1      2      3
// 0   v_0_0  v_0_1  v_0_2  v_0_3
// 1   v_1_0  v_1_1  v_1_2  v_1_3
// 2   v_2_0  v_2_1  v_2_2  v_2_3
// 3   v_3_0  v_3_1  v_3_2  v_3_3

// To compute the column addresses for a 2x3 sub-matrix at row 1 and column 1,
// we need a pointer to the first element of the submatrix as base pointer.
// Then we can use computeVectorAddr to compute the addresses for the columns
// of the sub-matrix.
//
// Column 0: computeVectorAddr(Base, 0 (column), 4 (stride), 2 (num rows), ..)
//           -> just returns Base
// Column 1: computeVectorAddr(Base, 1 (column), 4 (stride), 2 (num rows), ..)
//           -> returns Base + (1 * 4)
// Column 2: computeVectorAddr(Base, 2 (column), 4 (stride), 2 (num rows), ..)
//           -> returns Base + (2 * 4)
//
// The graphic below illustrates the number of elements in a column (marked
// with |) and the number of skipped elements (marked with }).
//
//         v_0_0  v_0_1 {v_0_2 {v_0_3
//                Base   Col 1  Col 2
//                  |     |      |
//         v_1_0 |v_1_1 |v_1_2 |v_1_3
//         v_2_0 |v_2_1 |v_2_2 |v_2_3
//         v_3_0 {v_3_1 {v_3_2  v_3_3
//
Value *computeVectorAddr(Value *BasePtr, Value *VecIdx, Value *Stride,
                         unsigned NumElements, Type *EltType,
                         IRBuilder<> &Builder) {

  assert((!isa<ConstantInt>(Stride) ||
          cast<ConstantInt>(Stride)->getZExtValue() >= NumElements) &&
         "Stride must be >= the number of elements in the result vector.");
  unsigned AS = cast<PointerType>(BasePtr->getType())->getAddressSpace();

  // Compute the start of the vector with index VecIdx as VecIdx * Stride.
  Value *VecStart = Builder.CreateMul(VecIdx, Stride, "vec.start");

  // Get pointer to the start of the selected vector. Skip GEP creation,
  // if we select vector 0.
  if (isa<ConstantInt>(VecStart) && cast<ConstantInt>(VecStart)->isZero())
    VecStart = BasePtr;
  else
    VecStart = Builder.CreateGEP(EltType, BasePtr, VecStart, "vec.gep");

  // Cast elementwise vector start pointer to a pointer to a vector
  // (EltType x NumElements)*.
  auto *VecType = FixedVectorType::get(EltType, NumElements);
  Type *VecPtrType = PointerType::get(VecType, AS);
  return Builder.CreatePointerCast(VecStart, VecPtrType, "vec.cast");
}

/// LowerMatrixIntrinsics contains the methods used to lower matrix intrinsics.
///
/// Currently, the lowering for each matrix intrinsic is done as follows:
/// 1. Propagate the shape information from intrinsics to connected
/// instructions.
/// 2. Lower instructions with shape information (assuming column-major layout).
///  The lowering works similarly using row-major layout.
///  2.1. Get column vectors for each argument. If we already lowered the
///       definition of an argument, use the produced column vectors directly.
///       If not, split the operand vector containing an embedded matrix into
///       a set of column vectors,
///  2.2. Lower the instruction in terms of column major operations, which
///       yields a set of column vectors containing result matrix. Note that we
///       lower all instructions that have shape information. Besides the
///       intrinsics, this includes stores for example.
///  2.3. Update uses of the lowered instruction. If we have shape information
///       for a user, there is nothing to do, as we will look up the result
///       column matrix when lowering the user. For other uses, we embed the
///       result matrix in a flat vector and update the use.
///  2.4. Cache the result column matrix for the instruction we lowered
/// 3. After we lowered all instructions in a function, remove the now
///    obsolete instructions.
///
class LowerMatrixIntrinsics {
  Function &Func;
  const DataLayout &DL;
  const TargetTransformInfo &TTI;
  AliasAnalysis *AA;
  DominatorTree *DT;
  LoopInfo *LI;
  OptimizationRemarkEmitter *ORE;

  /// Contains estimates of the number of operations (loads, stores, compute) required to lower a matrix operation.
  struct OpInfoTy {
    /// Number of stores emitted to generate this matrix.
    unsigned NumStores = 0;
    /// Number of loads emitted to generate this matrix.
    unsigned NumLoads = 0;
    /// Number of compute operations emitted to generate this matrix.
    unsigned NumComputeOps = 0;
    /// Most of the time transposes can be fused with matrix multiplies or can
    /// be folded away via algebraic simplifications.  This is the number of
    /// transposes that we failed to make "free" via such optimizations.
    unsigned NumExposedTransposes = 0;

    OpInfoTy &operator+=(const OpInfoTy &RHS) {
      NumStores += RHS.NumStores;
      NumLoads += RHS.NumLoads;
      NumComputeOps += RHS.NumComputeOps;
      NumExposedTransposes += RHS.NumExposedTransposes;
      return *this;
    }
  };

  /// Wrapper class representing a matrix as a set of vectors, either in row or
  /// column major layout. All vectors must have the same vector type.
  class MatrixTy {
    SmallVector<Value *, 16> Vectors;

    OpInfoTy OpInfo;

    bool IsColumnMajor = true;

  public:
    MatrixTy() : IsColumnMajor(MatrixLayout == MatrixLayoutTy::ColumnMajor) {}
    MatrixTy(ArrayRef<Value *> Vectors)
        : Vectors(Vectors.begin(), Vectors.end()),
          IsColumnMajor(MatrixLayout == MatrixLayoutTy::ColumnMajor) {}
    MatrixTy(unsigned NumRows, unsigned NumColumns, Type *EltTy)
        : IsColumnMajor(MatrixLayout == MatrixLayoutTy::ColumnMajor) {

      unsigned D = isColumnMajor() ? NumColumns : NumRows;
      for (unsigned J = 0; J < D; ++J)
        addVector(UndefValue::get(FixedVectorType::get(
            EltTy, isColumnMajor() ? NumRows : NumColumns)));
    }

    Value *getVector(unsigned i) const { return Vectors[i]; }
    Value *getColumn(unsigned i) const {
      assert(isColumnMajor() && "only supported for column-major matrixes");
      return Vectors[i];
    }
    Value *getRow(unsigned i) const {
      assert(!isColumnMajor() && "only supported for row-major matrixes");
      return Vectors[i];
    }

    void setVector(unsigned i, Value *V) { Vectors[i] = V; }

    Type *getElementType() const { return getVectorTy()->getElementType(); }

    unsigned getNumVectors() const {
      if (isColumnMajor())
        return getNumColumns();
      return getNumRows();
    }

    unsigned getNumColumns() const {
      if (isColumnMajor())
        return Vectors.size();
      else {
        assert(Vectors.size() > 0 && "Cannot call getNumRows without columns");
        return cast<FixedVectorType>(Vectors[0]->getType())->getNumElements();
      }
    }
    unsigned getNumRows() const {
      if (isColumnMajor()) {
        assert(Vectors.size() > 0 && "Cannot call getNumRows without columns");
        return cast<FixedVectorType>(Vectors[0]->getType())->getNumElements();
      } else
        return Vectors.size();
    }

    void addVector(Value *V) { Vectors.push_back(V); }
    VectorType *getColumnTy() {
      assert(isColumnMajor() && "only supported for column-major matrixes");
      return getVectorTy();
    }

    VectorType *getVectorTy() const {
      return cast<VectorType>(Vectors[0]->getType());
    }

    iterator_range<SmallVector<Value *, 8>::iterator> columns() {
      assert(isColumnMajor() &&
             "columns() only supported for column-major matrixes");
      return make_range(Vectors.begin(), Vectors.end());
    }

    iterator_range<SmallVector<Value *, 8>::iterator> vectors() {
      return make_range(Vectors.begin(), Vectors.end());
    }

    /// Embed the vectors of the matrix into a flat vector by concatenating
    /// them.
    Value *embedInVector(IRBuilder<> &Builder) const {
      return Vectors.size() == 1 ? Vectors[0]
                                 : concatenateVectors(Builder, Vectors);
    }

    MatrixTy &addNumLoads(unsigned N) {
      OpInfo.NumLoads += N;
      return *this;
    }

    void setNumLoads(unsigned N) { OpInfo.NumLoads = N; }

    MatrixTy &addNumStores(unsigned N) {
      OpInfo.NumStores += N;
      return *this;
    }

    MatrixTy &addNumExposedTransposes(unsigned N) {
      OpInfo.NumExposedTransposes += N;
      return *this;
    }

    MatrixTy &addNumComputeOps(unsigned N) {
      OpInfo.NumComputeOps += N;
      return *this;
    }

    unsigned getNumStores() const { return OpInfo.NumStores; }
    unsigned getNumLoads() const { return OpInfo.NumLoads; }
    unsigned getNumComputeOps() const { return OpInfo.NumComputeOps; }

    const OpInfoTy &getOpInfo() const { return OpInfo; }

    bool isColumnMajor() const { return IsColumnMajor; }

    unsigned getStride() const {
      if (isColumnMajor())
        return getNumRows();
      return getNumColumns();
    }

    /// Extract a vector of \p NumElts starting at index (\p I, \p J). If the
    /// matrix is column-major, the result vector is extracted from a column
    /// vector, otherwise from a row vector.
    Value *extractVector(unsigned I, unsigned J, unsigned NumElts,
                         IRBuilder<> &Builder) const {
      Value *Vec = isColumnMajor() ? getColumn(J) : getRow(I);
      return Builder.CreateShuffleVector(
          Vec, createSequentialMask(isColumnMajor() ? I : J, NumElts, 0),
          "block");
    }
  };

  struct ShapeInfo {
    unsigned NumRows;
    unsigned NumColumns;

    bool IsColumnMajor;

    ShapeInfo(unsigned NumRows = 0, unsigned NumColumns = 0)
        : NumRows(NumRows), NumColumns(NumColumns),
          IsColumnMajor(MatrixLayout == MatrixLayoutTy::ColumnMajor) {}

    ShapeInfo(Value *NumRows, Value *NumColumns)
        : ShapeInfo(cast<ConstantInt>(NumRows)->getZExtValue(),
                    cast<ConstantInt>(NumColumns)->getZExtValue()) {}

    bool operator==(const ShapeInfo &other) {
      return NumRows == other.NumRows && NumColumns == other.NumColumns;
    }
    bool operator!=(const ShapeInfo &other) { return !(*this == other); }

    /// Returns true if shape-information is defined, meaning both dimensions
    /// are != 0.
    operator bool() const {
      assert(NumRows == 0 || NumColumns != 0);
      return NumRows != 0;
    }

    unsigned getStride() const {
      if (IsColumnMajor)
        return NumRows;
      return NumColumns;
    }

    unsigned getNumVectors() const {
      if (IsColumnMajor)
        return NumColumns;
      return NumRows;
    }
  };

  /// Maps instructions to their shape information. The shape information
  /// describes the shape to be used while lowering. This matches the shape of
  /// the result value of the instruction, with the only exceptions being store
  /// instructions and the matrix_column_major_store intrinsics. For those, the
  /// shape information indicates that those instructions should be lowered
  /// using shape information as well.  A ValueMap is used so that when
  /// sub-passes like optimizeTransposes performs RAUW the map stays
  /// up-to-date.
  ValueMap<Value *, ShapeInfo> ShapeMap;

  /// List of instructions to remove. While lowering, we are not replacing all
  /// users of a lowered instruction, if shape information is available and
  /// those need to be removed after we finished lowering.
  SmallVector<Instruction *, 16> ToRemove;

  /// Map from instructions to their produced column matrix.
  MapVector<Value *, MatrixTy> Inst2ColumnMatrix;

private:
  static FastMathFlags getFastMathFlags(Instruction *Inst) {
    FastMathFlags FMF;

    if (isa<FPMathOperator>(*Inst))
      FMF = Inst->getFastMathFlags();

    FMF.setAllowContract(AllowContractEnabled || FMF.allowContract());

    return FMF;
  }

public:
  LowerMatrixIntrinsics(Function &F, TargetTransformInfo &TTI,
                        AliasAnalysis *AA, DominatorTree *DT, LoopInfo *LI,
                        OptimizationRemarkEmitter *ORE)
      : Func(F), DL(F.getParent()->getDataLayout()), TTI(TTI), AA(AA), DT(DT),
        LI(LI), ORE(ORE) {}

  unsigned getNumOps(Type *VT) {
    assert(isa<VectorType>(VT) && "Expected vector type");
    return getNumOps(VT->getScalarType(),
                     cast<FixedVectorType>(VT)->getNumElements());
  }

  /// Is this the minimal version executed in the backend pipelines.
  bool isMinimal() const {
    return !DT;
  }

  /// Return the estimated number of vector ops required for an operation on
  /// \p VT * N.
  unsigned getNumOps(Type *ST, unsigned N) {
    return std::ceil((ST->getPrimitiveSizeInBits() * N).getFixedSize() /
                     double(TTI.getRegisterBitWidth(
                                   TargetTransformInfo::RGK_FixedWidthVector)
                                .getFixedSize()));
  }

  /// Return the set of vectors that a matrix value is lowered to.
  ///
  /// If we lowered \p MatrixVal, just return the cache result matrix. Otherwise
  /// split the flat vector \p MatrixVal containing a matrix with shape \p SI
  /// into vectors.
  MatrixTy getMatrix(Value *MatrixVal, const ShapeInfo &SI,
                     IRBuilder<> &Builder) {
    VectorType *VType = dyn_cast<VectorType>(MatrixVal->getType());
    assert(VType && "MatrixVal must be a vector type");
    assert(cast<FixedVectorType>(VType)->getNumElements() ==
               SI.NumRows * SI.NumColumns &&
           "The vector size must match the number of matrix elements");

    // Check if we lowered MatrixVal using shape information. In that case,
    // return the existing matrix, if it matches the requested shape
    // information. If there is a mis-match, embed the result in a flat
    // vector and split it later.
    auto Found = Inst2ColumnMatrix.find(MatrixVal);
    if (Found != Inst2ColumnMatrix.end()) {
      MatrixTy &M = Found->second;
      // Return the found matrix, if its shape matches the requested shape
      // information
      if (SI.NumRows == M.getNumRows() && SI.NumColumns == M.getNumColumns())
        return M;

      MatrixVal = M.embedInVector(Builder);
    }

    // Otherwise split MatrixVal.
    SmallVector<Value *, 16> SplitVecs;
    for (unsigned MaskStart = 0;
         MaskStart < cast<FixedVectorType>(VType)->getNumElements();
         MaskStart += SI.getStride()) {
      Value *V = Builder.CreateShuffleVector(
          MatrixVal, createSequentialMask(MaskStart, SI.getStride(), 0),
          "split");
      SplitVecs.push_back(V);
    }

    return {SplitVecs};
  }

  /// If \p V already has a known shape return false.  Otherwise set the shape
  /// for instructions that support it.
  bool setShapeInfo(Value *V, ShapeInfo Shape) {
    assert(Shape && "Shape not set");
    if (isa<UndefValue>(V) || !supportsShapeInfo(V))
      return false;

    auto SIter = ShapeMap.find(V);
    if (SIter != ShapeMap.end()) {
      LLVM_DEBUG(dbgs() << "  not overriding existing shape: "
                        << SIter->second.NumRows << " "
                        << SIter->second.NumColumns << " for " << *V << "\n");
      return false;
    }

    ShapeMap.insert({V, Shape});
    LLVM_DEBUG(dbgs() << "  " << Shape.NumRows << " x " << Shape.NumColumns
                      << " for " << *V << "\n");
    return true;
  }

  bool isUniformShape(Value *V) {
    Instruction *I = dyn_cast<Instruction>(V);
    if (!I)
      return true;

    switch (I->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul: // Scalar multiply.
    case Instruction::FNeg:
    case Instruction::Add:
    case Instruction::Mul:
    case Instruction::Sub:
      return true;
    default:
      return false;
    }
  }

  /// Returns true if shape information can be used for \p V. The supported
  /// instructions must match the instructions that can be lowered by this pass.
  bool supportsShapeInfo(Value *V) {
    Instruction *Inst = dyn_cast<Instruction>(V);
    if (!Inst)
      return false;

    IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst);
    if (II)
      switch (II->getIntrinsicID()) {
      case Intrinsic::matrix_multiply:
      case Intrinsic::matrix_transpose:
      case Intrinsic::matrix_column_major_load:
      case Intrinsic::matrix_column_major_store:
        return true;
      default:
        return false;
      }
    return isUniformShape(V) || isa<StoreInst>(V) || isa<LoadInst>(V);
  }

  /// Propagate the shape information of instructions to their users.
  /// The work list contains instructions for which we can compute the shape,
  /// either based on the information provided by matrix intrinsics or known
  /// shapes of operands.
  SmallVector<Instruction *, 32>
  propagateShapeForward(SmallVectorImpl<Instruction *> &WorkList) {
    SmallVector<Instruction *, 32> NewWorkList;
    // Pop an element for which we guaranteed to have at least one of the
    // operand shapes.  Add the shape for this and then add users to the work
    // list.
    LLVM_DEBUG(dbgs() << "Forward-propagate shapes:\n");
    while (!WorkList.empty()) {
      Instruction *Inst = WorkList.pop_back_val();

      // New entry, set the value and insert operands
      bool Propagate = false;

      Value *MatrixA;
      Value *MatrixB;
      Value *M;
      Value *N;
      Value *K;
      if (match(Inst, m_Intrinsic<Intrinsic::matrix_multiply>(
                          m_Value(MatrixA), m_Value(MatrixB), m_Value(M),
                          m_Value(N), m_Value(K)))) {
        Propagate = setShapeInfo(Inst, {M, K});
      } else if (match(Inst, m_Intrinsic<Intrinsic::matrix_transpose>(
                                 m_Value(MatrixA), m_Value(M), m_Value(N)))) {
        // Flip dimensions.
        Propagate = setShapeInfo(Inst, {N, M});
      } else if (match(Inst, m_Intrinsic<Intrinsic::matrix_column_major_store>(
                                 m_Value(MatrixA), m_Value(), m_Value(),
                                 m_Value(), m_Value(M), m_Value(N)))) {
        Propagate = setShapeInfo(Inst, {N, M});
      } else if (match(Inst, m_Intrinsic<Intrinsic::matrix_column_major_load>(
                                 m_Value(), m_Value(), m_Value(), m_Value(M),
                                 m_Value(N)))) {
        Propagate = setShapeInfo(Inst, {M, N});
      } else if (match(Inst, m_Store(m_Value(MatrixA), m_Value()))) {
        auto OpShape = ShapeMap.find(MatrixA);
        if (OpShape != ShapeMap.end())
          setShapeInfo(Inst, OpShape->second);
        continue;
      } else if (isUniformShape(Inst)) {
        // Find the first operand that has a known shape and use that.
        for (auto &Op : Inst->operands()) {
          auto OpShape = ShapeMap.find(Op.get());
          if (OpShape != ShapeMap.end()) {
            Propagate |= setShapeInfo(Inst, OpShape->second);
            break;
          }
        }
      }

      if (Propagate) {
        NewWorkList.push_back(Inst);
        for (auto *User : Inst->users())
          if (ShapeMap.count(User) == 0)
            WorkList.push_back(cast<Instruction>(User));
      }
    }

    return NewWorkList;
  }

  /// Propagate the shape to operands of instructions with shape information.
  /// \p Worklist contains the instruction for which we already know the shape.
  SmallVector<Instruction *, 32>
  propagateShapeBackward(SmallVectorImpl<Instruction *> &WorkList) {
    SmallVector<Instruction *, 32> NewWorkList;

    auto pushInstruction = [](Value *V,
                              SmallVectorImpl<Instruction *> &WorkList) {
      Instruction *I = dyn_cast<Instruction>(V);
      if (I)
        WorkList.push_back(I);
    };
    // Pop an element with known shape.  Traverse the operands, if their shape
    // derives from the result shape and is unknown, add it and add them to the
    // worklist.
    LLVM_DEBUG(dbgs() << "Backward-propagate shapes:\n");
    while (!WorkList.empty()) {
      Value *V = WorkList.pop_back_val();

      size_t BeforeProcessingV = WorkList.size();
      if (!isa<Instruction>(V))
        continue;

      Value *MatrixA;
      Value *MatrixB;
      Value *M;
      Value *N;
      Value *K;
      if (match(V, m_Intrinsic<Intrinsic::matrix_multiply>(
                       m_Value(MatrixA), m_Value(MatrixB), m_Value(M),
                       m_Value(N), m_Value(K)))) {
        if (setShapeInfo(MatrixA, {M, N}))
          pushInstruction(MatrixA, WorkList);

        if (setShapeInfo(MatrixB, {N, K}))
          pushInstruction(MatrixB, WorkList);

      } else if (match(V, m_Intrinsic<Intrinsic::matrix_transpose>(
                              m_Value(MatrixA), m_Value(M), m_Value(N)))) {
        // Flip dimensions.
        if (setShapeInfo(MatrixA, {M, N}))
          pushInstruction(MatrixA, WorkList);
      } else if (match(V, m_Intrinsic<Intrinsic::matrix_column_major_store>(
                              m_Value(MatrixA), m_Value(), m_Value(), m_Value(),
                              m_Value(M), m_Value(N)))) {
        if (setShapeInfo(MatrixA, {M, N})) {
          pushInstruction(MatrixA, WorkList);
        }
      } else if (isa<LoadInst>(V) ||
                 match(V, m_Intrinsic<Intrinsic::matrix_column_major_load>())) {
        // Nothing to do, no matrix input.
      } else if (isa<StoreInst>(V)) {
        // Nothing to do.  We forward-propagated to this so we would just
        // backward propagate to an instruction with an already known shape.
      } else if (isUniformShape(V)) {
        // Propagate to all operands.
        ShapeInfo Shape = ShapeMap[V];
        for (Use &U : cast<Instruction>(V)->operands()) {
          if (setShapeInfo(U.get(), Shape))
            pushInstruction(U.get(), WorkList);
        }
      }
      // After we discovered new shape info for new instructions in the
      // worklist, we use their users as seeds for the next round of forward
      // propagation.
      for (size_t I = BeforeProcessingV; I != WorkList.size(); I++)
        for (User *U : WorkList[I]->users())
          if (isa<Instruction>(U) && V != U)
            NewWorkList.push_back(cast<Instruction>(U));
    }
    return NewWorkList;
  }

  /// Try moving transposes in order to fold them away or into multiplies.
  void optimizeTransposes() {
    auto ReplaceAllUsesWith = [this](Instruction &Old, Value *New) {
      // We need to remove Old from the ShapeMap otherwise RAUW will replace it
      // with New. We should only add New it it supportsShapeInfo so we insert
      // it conditionally instead.
      auto S = ShapeMap.find(&Old);
      if (S != ShapeMap.end()) {
        ShapeMap.erase(S);
        if (supportsShapeInfo(New))
          ShapeMap.insert({New, S->second});
      }
      Old.replaceAllUsesWith(New);
    };

    // First sink all transposes inside matmuls, hoping that we end up with NN,
    // NT or TN variants.
    for (BasicBlock &BB : reverse(Func)) {
      for (auto II = BB.rbegin(); II != BB.rend();) {
        Instruction &I = *II;
        // We may remove II.  By default continue on the next/prev instruction.
        ++II;
        // If we were to erase II, move again.
        auto EraseFromParent = [&II](Value *V) {
          auto *Inst = cast<Instruction>(V);
          if (Inst->use_empty()) {
            if (Inst == &*II) {
              ++II;
            }
            Inst->eraseFromParent();
          }
        };

        // If we're creating a new instruction, continue from there.
        Instruction *NewInst = nullptr;

        IRBuilder<> IB(&I);
        MatrixBuilder<IRBuilder<>> Builder(IB);

        Value *TA, *TAMA, *TAMB;
        ConstantInt *R, *K, *C;
        if (match(&I, m_Intrinsic<Intrinsic::matrix_transpose>(m_Value(TA)))) {

          // Transpose of a transpose is a nop
          Value *TATA;
          if (match(TA,
                    m_Intrinsic<Intrinsic::matrix_transpose>(m_Value(TATA)))) {
            ReplaceAllUsesWith(I, TATA);
            EraseFromParent(&I);
            EraseFromParent(TA);
          }

          // (A * B)^t -> B^t * A^t
          // RxK KxC      CxK   KxR
          else if (match(TA, m_Intrinsic<Intrinsic::matrix_multiply>(
                                 m_Value(TAMA), m_Value(TAMB), m_ConstantInt(R),
                                 m_ConstantInt(K), m_ConstantInt(C)))) {
            Value *T0 = Builder.CreateMatrixTranspose(TAMB, K->getZExtValue(),
                                                      C->getZExtValue(),
                                                      TAMB->getName() + "_t");
            // We are being run after shape prop, add shape for newly created
            // instructions so that we lower them later.
            setShapeInfo(T0, {C, K});
            Value *T1 = Builder.CreateMatrixTranspose(TAMA, R->getZExtValue(),
                                                      K->getZExtValue(),
                                                      TAMA->getName() + "_t");
            setShapeInfo(T1, {K, R});
            NewInst = Builder.CreateMatrixMultiply(T0, T1, C->getZExtValue(),
                                                   K->getZExtValue(),
                                                   R->getZExtValue(), "mmul");
            ReplaceAllUsesWith(I, NewInst);
            EraseFromParent(&I);
            EraseFromParent(TA);
          }
        }

        // If we replaced I with a new instruction, continue from there.
        if (NewInst)
          II = std::next(BasicBlock::reverse_iterator(NewInst));
      }
    }

    // If we have a TT matmul, lift the transpose.  We may be able to fold into
    // consuming multiply.
    for (BasicBlock &BB : Func) {
      for (BasicBlock::iterator II = BB.begin(); II != BB.end();) {
        Instruction *I = &*II;
        // We may remove I.
        ++II;
        Value *A, *B, *AT, *BT;
        ConstantInt *R, *K, *C;
        // A^t * B ^t -> (B * A)^t
        if (match(&*I, m_Intrinsic<Intrinsic::matrix_multiply>(
                           m_Value(A), m_Value(B), m_ConstantInt(R),
                           m_ConstantInt(K), m_ConstantInt(C))) &&
            match(A, m_Intrinsic<Intrinsic::matrix_transpose>(m_Value(AT))) &&
            match(B, m_Intrinsic<Intrinsic::matrix_transpose>(m_Value((BT))))) {
          IRBuilder<> IB(&*I);
          MatrixBuilder<IRBuilder<>> Builder(IB);
          Value *M = Builder.CreateMatrixMultiply(
              BT, AT, C->getZExtValue(), K->getZExtValue(), R->getZExtValue());
          setShapeInfo(M, {C, R});
          Instruction *NewInst = Builder.CreateMatrixTranspose(
              M, C->getZExtValue(), R->getZExtValue());
          ReplaceAllUsesWith(*I, NewInst);
          if (I->use_empty())
            I->eraseFromParent();
          if (A->use_empty())
            cast<Instruction>(A)->eraseFromParent();
          if (A != B && B->use_empty())
            cast<Instruction>(B)->eraseFromParent();
        }
      }
    }
  }

  bool Visit() {
    SmallVector<Instruction *, 32> WorkList;

    // Initially only the shape of matrix intrinsics is known.
    // Initialize the work list with ops carrying shape information.
    for (BasicBlock &BB : Func)
      for (Instruction &Inst : BB) {
        IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
        if (!II)
          continue;

        switch (II->getIntrinsicID()) {
        case Intrinsic::matrix_multiply:
        case Intrinsic::matrix_transpose:
        case Intrinsic::matrix_column_major_load:
        case Intrinsic::matrix_column_major_store:
          WorkList.push_back(&Inst);
          break;
        default:
          break;
        }
      }

    // Avoid unnecessary work if there are no matrix intrinsics in the function.
    if (WorkList.empty())
      return false;

    // Propagate shapes until nothing changes any longer.
    while (!WorkList.empty()) {
      WorkList = propagateShapeForward(WorkList);
      WorkList = propagateShapeBackward(WorkList);
    }

    if (!isMinimal()) {
      optimizeTransposes();
      LLVM_DEBUG({
        dbgs() << "Dump after matrix transpose optimization:\n";
        Func.dump();
      });
    }

    bool Changed = false;
    SmallVector<CallInst *, 16> MaybeFusableInsts;
    SmallVector<Instruction *, 16> MatrixInsts;

    // First, collect all instructions with shape information and candidates for
    // fusion (currently only matrix multiplies).
    ReversePostOrderTraversal<Function *> RPOT(&Func);
    for (auto *BB : RPOT)
      for (Instruction &I : *BB) {
        if (ShapeMap.find(&I) == ShapeMap.end())
          continue;
        if (match(&I, m_Intrinsic<Intrinsic::matrix_multiply>()))
          MaybeFusableInsts.push_back(cast<CallInst>(&I));
        MatrixInsts.push_back(&I);
      }

    // Second, try to fuse candidates.
    SmallPtrSet<Instruction *, 16> FusedInsts;
    for (CallInst *CI : MaybeFusableInsts)
      LowerMatrixMultiplyFused(CI, FusedInsts);
    Changed = !FusedInsts.empty();

    // Third, lower remaining instructions with shape information.
    for (Instruction *Inst : MatrixInsts) {
      if (FusedInsts.count(Inst))
        continue;

      IRBuilder<> Builder(Inst);

      if (CallInst *CInst = dyn_cast<CallInst>(Inst))
        Changed |= VisitCallInst(CInst);

      Value *Op1;
      Value *Op2;
      if (auto *BinOp = dyn_cast<BinaryOperator>(Inst))
        Changed |= VisitBinaryOperator(BinOp);
      if (auto *UnOp = dyn_cast<UnaryOperator>(Inst))
        Changed |= VisitUnaryOperator(UnOp);
      if (match(Inst, m_Load(m_Value(Op1))))
        Changed |= VisitLoad(cast<LoadInst>(Inst), Op1, Builder);
      else if (match(Inst, m_Store(m_Value(Op1), m_Value(Op2))))
        Changed |= VisitStore(cast<StoreInst>(Inst), Op1, Op2, Builder);
    }

    if (ORE) {
      RemarkGenerator RemarkGen(Inst2ColumnMatrix, *ORE, Func);
      RemarkGen.emitRemarks();
    }

    // Delete the instructions backwards, as it has a reduced likelihood of
    // having to update as many def-use and use-def chains.
    //
    // Because we add to ToRemove during fusion we can't guarantee that defs
    // are before uses.  Change uses to undef temporarily as these should get
    // removed as well.
    //
    // For verification, we keep track of where we changed uses to undefs in
    // UndefedInsts and then check that we in fact remove them.
    SmallSet<Instruction *, 16> UndefedInsts;
    for (auto *Inst : reverse(ToRemove)) {
      for (Use &U : llvm::make_early_inc_range(Inst->uses())) {
        if (auto *Undefed = dyn_cast<Instruction>(U.getUser()))
          UndefedInsts.insert(Undefed);
        U.set(UndefValue::get(Inst->getType()));
      }
      Inst->eraseFromParent();
      UndefedInsts.erase(Inst);
    }
    if (!UndefedInsts.empty()) {
      // If we didn't remove all undefed instructions, it's a hard error.
      dbgs() << "Undefed but present instructions:\n";
      for (auto *I : UndefedInsts)
        dbgs() << *I << "\n";
      llvm_unreachable("Undefed but instruction not removed");
    }

    return Changed;
  }

  /// Turns \p BasePtr into an elementwise pointer to \p EltType.
  Value *createElementPtr(Value *BasePtr, Type *EltType, IRBuilder<> &Builder) {
    unsigned AS = cast<PointerType>(BasePtr->getType())->getAddressSpace();
    Type *EltPtrType = PointerType::get(EltType, AS);
    return Builder.CreatePointerCast(BasePtr, EltPtrType);
  }

  /// Replace intrinsic calls
  bool VisitCallInst(CallInst *Inst) {
    if (!Inst->getCalledFunction() || !Inst->getCalledFunction()->isIntrinsic())
      return false;

    switch (Inst->getCalledFunction()->getIntrinsicID()) {
    case Intrinsic::matrix_multiply:
      LowerMultiply(Inst);
      break;
    case Intrinsic::matrix_transpose:
      LowerTranspose(Inst);
      break;
    case Intrinsic::matrix_column_major_load:
      LowerColumnMajorLoad(Inst);
      break;
    case Intrinsic::matrix_column_major_store:
      LowerColumnMajorStore(Inst);
      break;
    default:
      return false;
    }
    return true;
  }

  /// Compute the alignment for a column/row \p Idx with \p Stride between them.
  /// The address at \p Idx == 0 has alignment \p A. If \p Stride is a
  /// ConstantInt, reduce the initial alignment based on the byte offset. For
  /// non-ConstantInt strides, return the common alignment of the initial
  /// alignment and the element size in bytes.
  Align getAlignForIndex(unsigned Idx, Value *Stride, Type *ElementTy,
                         MaybeAlign A) const {
    Align InitialAlign = DL.getValueOrABITypeAlignment(A, ElementTy);
    if (Idx == 0)
      return InitialAlign;

    TypeSize ElementSizeInBits = DL.getTypeSizeInBits(ElementTy);
    if (auto *ConstStride = dyn_cast<ConstantInt>(Stride)) {
      uint64_t StrideInBytes =
          ConstStride->getZExtValue() * ElementSizeInBits / 8;
      return commonAlignment(InitialAlign, Idx * StrideInBytes);
    }
    return commonAlignment(InitialAlign, ElementSizeInBits / 8);
  }

  /// Load a matrix with \p Shape starting at \p Ptr and using \p Stride between
  /// vectors.
  MatrixTy loadMatrix(Type *Ty, Value *Ptr, MaybeAlign MAlign, Value *Stride,
                      bool IsVolatile, ShapeInfo Shape, IRBuilder<> &Builder) {
    auto *VType = cast<VectorType>(Ty);
    Type *EltTy = VType->getElementType();
    Type *VecTy = FixedVectorType::get(EltTy, Shape.getStride());
    Value *EltPtr = createElementPtr(Ptr, EltTy, Builder);
    MatrixTy Result;
    for (unsigned I = 0, E = Shape.getNumVectors(); I < E; ++I) {
      Value *GEP = computeVectorAddr(
          EltPtr, Builder.getIntN(Stride->getType()->getScalarSizeInBits(), I),
          Stride, Shape.getStride(), EltTy, Builder);
      Value *Vector = Builder.CreateAlignedLoad(
          VecTy, GEP, getAlignForIndex(I, Stride, EltTy, MAlign),
          IsVolatile, "col.load");

      Result.addVector(Vector);
    }
    return Result.addNumLoads(getNumOps(Result.getVectorTy()) *
                              Result.getNumVectors());
  }

  /// Loads a sub-matrix with shape \p ResultShape from a \p R x \p C matrix,
  /// starting at \p MatrixPtr[I][J].
  MatrixTy loadMatrix(Value *MatrixPtr, MaybeAlign Align, bool IsVolatile,
                      ShapeInfo MatrixShape, Value *I, Value *J,
                      ShapeInfo ResultShape, Type *EltTy,
                      IRBuilder<> &Builder) {

    Value *Offset = Builder.CreateAdd(
        Builder.CreateMul(J, Builder.getInt64(MatrixShape.getStride())), I);

    unsigned AS = cast<PointerType>(MatrixPtr->getType())->getAddressSpace();
    Value *EltPtr =
        Builder.CreatePointerCast(MatrixPtr, PointerType::get(EltTy, AS));
    Value *TileStart = Builder.CreateGEP(EltTy, EltPtr, Offset);
    auto *TileTy = FixedVectorType::get(EltTy, ResultShape.NumRows *
                                                   ResultShape.NumColumns);
    Type *TilePtrTy = PointerType::get(TileTy, AS);
    Value *TilePtr =
        Builder.CreatePointerCast(TileStart, TilePtrTy, "col.cast");

    return loadMatrix(TileTy, TilePtr, Align,
                      Builder.getInt64(MatrixShape.getStride()), IsVolatile,
                      ResultShape, Builder);
  }

  /// Lower a load instruction with shape information.
  void LowerLoad(Instruction *Inst, Value *Ptr, MaybeAlign Align, Value *Stride,
                 bool IsVolatile, ShapeInfo Shape) {
    IRBuilder<> Builder(Inst);
    finalizeLowering(Inst,
                     loadMatrix(Inst->getType(), Ptr, Align, Stride, IsVolatile,
                                Shape, Builder),
                     Builder);
  }

  /// Lowers llvm.matrix.column.major.load.
  ///
  /// The intrinsic loads a matrix from memory using a stride between columns.
  void LowerColumnMajorLoad(CallInst *Inst) {
    assert(MatrixLayout == MatrixLayoutTy::ColumnMajor &&
           "Intrinsic only supports column-major layout!");
    Value *Ptr = Inst->getArgOperand(0);
    Value *Stride = Inst->getArgOperand(1);
    LowerLoad(Inst, Ptr, Inst->getParamAlign(0), Stride,
              cast<ConstantInt>(Inst->getArgOperand(2))->isOne(),
              {Inst->getArgOperand(3), Inst->getArgOperand(4)});
  }

  /// Stores a sub-matrix \p StoreVal into the \p R x \p C matrix starting at \p
  /// MatrixPtr[I][J].
  void storeMatrix(const MatrixTy &StoreVal, Value *MatrixPtr,
                   MaybeAlign MAlign, bool IsVolatile, ShapeInfo MatrixShape,
                   Value *I, Value *J, Type *EltTy, IRBuilder<> &Builder) {
    Value *Offset = Builder.CreateAdd(
        Builder.CreateMul(J, Builder.getInt64(MatrixShape.getStride())), I);

    unsigned AS = cast<PointerType>(MatrixPtr->getType())->getAddressSpace();
    Value *EltPtr =
        Builder.CreatePointerCast(MatrixPtr, PointerType::get(EltTy, AS));
    Value *TileStart = Builder.CreateGEP(EltTy, EltPtr, Offset);
    auto *TileTy = FixedVectorType::get(EltTy, StoreVal.getNumRows() *
                                                   StoreVal.getNumColumns());
    Type *TilePtrTy = PointerType::get(TileTy, AS);
    Value *TilePtr =
        Builder.CreatePointerCast(TileStart, TilePtrTy, "col.cast");

    storeMatrix(TileTy, StoreVal, TilePtr, MAlign,
                Builder.getInt64(MatrixShape.getStride()), IsVolatile, Builder);
  }

  /// Store matrix \p StoreVal starting at \p Ptr and using \p Stride between
  /// vectors.
  MatrixTy storeMatrix(Type *Ty, MatrixTy StoreVal, Value *Ptr,
                       MaybeAlign MAlign, Value *Stride, bool IsVolatile,
                       IRBuilder<> &Builder) {
    auto VType = cast<VectorType>(Ty);
    Value *EltPtr = createElementPtr(Ptr, VType->getElementType(), Builder);
    for (auto Vec : enumerate(StoreVal.vectors())) {
      Value *GEP = computeVectorAddr(
          EltPtr,
          Builder.getIntN(Stride->getType()->getScalarSizeInBits(),
                          Vec.index()),
          Stride, StoreVal.getStride(), VType->getElementType(), Builder);
      Builder.CreateAlignedStore(Vec.value(), GEP,
                                 getAlignForIndex(Vec.index(), Stride,
                                                  VType->getElementType(),
                                                  MAlign),
                                 IsVolatile);
    }
    return MatrixTy().addNumStores(getNumOps(StoreVal.getVectorTy()) *
                                   StoreVal.getNumVectors());
  }

  /// Lower a store instruction with shape information.
  void LowerStore(Instruction *Inst, Value *Matrix, Value *Ptr, MaybeAlign A,
                  Value *Stride, bool IsVolatile, ShapeInfo Shape) {
    IRBuilder<> Builder(Inst);
    auto StoreVal = getMatrix(Matrix, Shape, Builder);
    finalizeLowering(Inst,
                     storeMatrix(Matrix->getType(), StoreVal, Ptr, A, Stride,
                                 IsVolatile, Builder),
                     Builder);
  }

  /// Lowers llvm.matrix.column.major.store.
  ///
  /// The intrinsic store a matrix back memory using a stride between columns.
  void LowerColumnMajorStore(CallInst *Inst) {
    assert(MatrixLayout == MatrixLayoutTy::ColumnMajor &&
           "Intrinsic only supports column-major layout!");
    Value *Matrix = Inst->getArgOperand(0);
    Value *Ptr = Inst->getArgOperand(1);
    Value *Stride = Inst->getArgOperand(2);
    LowerStore(Inst, Matrix, Ptr, Inst->getParamAlign(1), Stride,
               cast<ConstantInt>(Inst->getArgOperand(3))->isOne(),
               {Inst->getArgOperand(4), Inst->getArgOperand(5)});
  }

  // Set elements I..I+NumElts-1 to Block
  Value *insertVector(Value *Col, unsigned I, Value *Block,
                      IRBuilder<> &Builder) {

    // First, bring Block to the same size as Col
    unsigned BlockNumElts =
        cast<FixedVectorType>(Block->getType())->getNumElements();
    unsigned NumElts = cast<FixedVectorType>(Col->getType())->getNumElements();
    assert(NumElts >= BlockNumElts && "Too few elements for current block");

    Block = Builder.CreateShuffleVector(
        Block, createSequentialMask(0, BlockNumElts, NumElts - BlockNumElts));

    // If Col is 7 long and I is 2 and BlockNumElts is 2 the mask is: 0, 1, 7,
    // 8, 4, 5, 6
    SmallVector<int, 16> Mask;
    unsigned i;
    for (i = 0; i < I; i++)
      Mask.push_back(i);

    unsigned VecNumElts =
        cast<FixedVectorType>(Col->getType())->getNumElements();
    for (; i < I + BlockNumElts; i++)
      Mask.push_back(i - I + VecNumElts);

    for (; i < VecNumElts; i++)
      Mask.push_back(i);

    return Builder.CreateShuffleVector(Col, Block, Mask);
  }

  Value *createMulAdd(Value *Sum, Value *A, Value *B, bool UseFPOp,
                      IRBuilder<> &Builder, bool AllowContraction,
                      unsigned &NumComputeOps) {
    NumComputeOps += getNumOps(A->getType());
    if (!Sum)
      return UseFPOp ? Builder.CreateFMul(A, B) : Builder.CreateMul(A, B);

    if (UseFPOp) {
      if (AllowContraction) {
        // Use fmuladd for floating point operations and let the backend decide
        // if that's profitable.
        Function *FMulAdd = Intrinsic::getDeclaration(
            Func.getParent(), Intrinsic::fmuladd, A->getType());
        return Builder.CreateCall(FMulAdd, {A, B, Sum});
      }
      NumComputeOps += getNumOps(A->getType());
      Value *Mul = Builder.CreateFMul(A, B);
      return Builder.CreateFAdd(Sum, Mul);
    }

    NumComputeOps += getNumOps(A->getType());
    Value *Mul = Builder.CreateMul(A, B);
    return Builder.CreateAdd(Sum, Mul);
  }

  /// Cache \p Matrix as result of \p Inst and update the uses of \p Inst. For
  /// users with shape information, there's nothing to do: they will use the
  /// cached value when they are lowered. For other users, \p Matrix is
  /// flattened and the uses are updated to use it. Also marks \p Inst for
  /// deletion.
  void finalizeLowering(Instruction *Inst, MatrixTy Matrix,
                        IRBuilder<> &Builder) {
    auto inserted = Inst2ColumnMatrix.insert(std::make_pair(Inst, Matrix));
    (void)inserted;
    assert(inserted.second && "multiple matrix lowering mapping");

    ToRemove.push_back(Inst);
    Value *Flattened = nullptr;
    for (Use &U : llvm::make_early_inc_range(Inst->uses())) {
      if (ShapeMap.find(U.getUser()) == ShapeMap.end()) {
        if (!Flattened)
          Flattened = Matrix.embedInVector(Builder);
        U.set(Flattened);
      }
    }
  }

  /// Compute \p Result += \p A * \p B for input matrices with left-associating
  /// addition.
  ///
  /// We can fold a transpose into the operand that is used to extract scalars.
  /// This is the first operands with row-major and the second with
  /// column-major.  If \p IsScalarMatrixTransposed we assume the appropriate
  /// operand is transposed.
  void emitMatrixMultiply(MatrixTy &Result, const MatrixTy &A,
                          const MatrixTy &B, IRBuilder<> &Builder, bool IsTiled,
                          bool IsScalarMatrixTransposed, FastMathFlags FMF) {
    const unsigned VF = std::max<unsigned>(
        TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
                .getFixedSize() /
            Result.getElementType()->getPrimitiveSizeInBits().getFixedSize(),
        1U);
    unsigned R = Result.getNumRows();
    unsigned C = Result.getNumColumns();
    unsigned M = A.getNumColumns();

    bool IsFP = Result.getElementType()->isFloatingPointTy();
    assert(A.isColumnMajor() == B.isColumnMajor() &&
           Result.isColumnMajor() == A.isColumnMajor() &&
           "operands must agree on matrix layout");
    unsigned NumComputeOps = 0;

    Builder.setFastMathFlags(FMF);

    if (A.isColumnMajor()) {
      // Multiply columns from the first operand with scalars from the second
      // operand. Then move along the K axes and accumulate the columns.  With
      // this the adds can be vectorized without reassociation.
      for (unsigned J = 0; J < C; ++J) {
        unsigned BlockSize = VF;
        // If Result is zero, we don't need to accumulate in the K==0 iteration.
        bool isSumZero = isa<ConstantAggregateZero>(Result.getColumn(J));

        for (unsigned I = 0; I < R; I += BlockSize) {
          // Gradually lower the vectorization factor to cover the remainder.
          while (I + BlockSize > R)
            BlockSize /= 2;

          Value *Sum = IsTiled ? Result.extractVector(I, J, BlockSize, Builder)
                               : nullptr;
          for (unsigned K = 0; K < M; ++K) {
            Value *L = A.extractVector(I, K, BlockSize, Builder);
            Value *RH = Builder.CreateExtractElement(
                B.getColumn(IsScalarMatrixTransposed ? K : J),
                IsScalarMatrixTransposed ? J : K);
            Value *Splat = Builder.CreateVectorSplat(BlockSize, RH, "splat");
            Sum =
                createMulAdd(isSumZero && K == 0 ? nullptr : Sum, L, Splat,
                             IsFP, Builder, FMF.allowContract(), NumComputeOps);
          }
          Result.setVector(J,
                           insertVector(Result.getVector(J), I, Sum, Builder));
        }
      }
    } else {
      // Multiply rows from the second operand with scalars from the first
      // operand. Then move along the K axes and accumulate the rows.  With this
      // the adds can be vectorized without reassociation.
      for (unsigned I = 0; I < R; ++I) {
        unsigned BlockSize = VF;
        bool isSumZero = isa<ConstantAggregateZero>(Result.getRow(I));
        for (unsigned J = 0; J < C; J += BlockSize) {
          // Gradually lower the vectorization factor to cover the remainder.
          while (J + BlockSize > C)
            BlockSize /= 2;

          Value *Sum = nullptr;
          for (unsigned K = 0; K < M; ++K) {
            Value *R = B.extractVector(K, J, BlockSize, Builder);
            Value *LH = Builder.CreateExtractElement(
                A.getVector(IsScalarMatrixTransposed ? K : I),
                IsScalarMatrixTransposed ? I : K);
            Value *Splat = Builder.CreateVectorSplat(BlockSize, LH, "splat");
            Sum =
                createMulAdd(isSumZero && K == 0 ? nullptr : Sum, Splat, R,
                             IsFP, Builder, FMF.allowContract(), NumComputeOps);
          }
          Result.setVector(I,
                           insertVector(Result.getVector(I), J, Sum, Builder));
        }
      }
    }
    Result.addNumComputeOps(NumComputeOps);
  }

  /// Ensure that the memory in \p Load does not alias \p Store by potentially
  /// copying it to a new location.  This new or otherwise the original location
  /// is returned.
  Value *getNonAliasingPointer(LoadInst *Load, StoreInst *Store,
                               CallInst *MatMul) {
    MemoryLocation StoreLoc = MemoryLocation::get(Store);
    MemoryLocation LoadLoc = MemoryLocation::get(Load);

    // If we can statically determine noalias we're good.
    if (AA->isNoAlias(LoadLoc, StoreLoc))
      return Load->getPointerOperand();

    // Create code to check if the memory locations of the Load and Store
    // overlap and if they do, copy Load's operand to a new buffer.

    // First, create  new blocks for 2n part of the check and the copy.
    BasicBlock *Check0 = MatMul->getParent();
    // FIXME: Use lazy DTU and update SplitBlock to accept a DTU instead of a
    // DT. Manually collect dominator tree updates, to avoid unnecessary work,
    // as we adjust Check0 and Check1's branches.
    SmallVector<DominatorTree::UpdateType, 4> DTUpdates;
    for (BasicBlock *Succ : successors(Check0))
      DTUpdates.push_back({DT->Delete, Check0, Succ});

    BasicBlock *Check1 =
        SplitBlock(MatMul->getParent(), MatMul, (DomTreeUpdater *)nullptr, LI,
                   nullptr, "alias_cont");
    BasicBlock *Copy =
        SplitBlock(MatMul->getParent(), MatMul, (DomTreeUpdater *)nullptr, LI,
                   nullptr, "copy");
    BasicBlock *Fusion =
        SplitBlock(MatMul->getParent(), MatMul, (DomTreeUpdater *)nullptr, LI,
                   nullptr, "no_alias");

    // Check if the loaded memory location begins before the end of the store
    // location. If the condition holds, they might overlap, otherwise they are
    // guaranteed to not overlap.
    IRBuilder<> Builder(MatMul);
    Check0->getTerminator()->eraseFromParent();
    Builder.SetInsertPoint(Check0);
    Type *IntPtrTy = Builder.getIntPtrTy(Load->getModule()->getDataLayout());
    Value *StoreBegin = Builder.CreatePtrToInt(
        const_cast<Value *>(StoreLoc.Ptr), IntPtrTy, "store.begin");
    Value *StoreEnd = Builder.CreateAdd(
        StoreBegin, ConstantInt::get(IntPtrTy, StoreLoc.Size.getValue()),
        "store.end", true, true);
    Value *LoadBegin = Builder.CreatePtrToInt(const_cast<Value *>(LoadLoc.Ptr),
                                              IntPtrTy, "load.begin");
    Builder.CreateCondBr(Builder.CreateICmpULT(LoadBegin, StoreEnd), Check1,
                         Fusion);

    // Check if the store begins before the end of the load location. If the
    // condition holds, they alias, otherwise they are guaranteed to not
    // overlap.
    Check1->getTerminator()->eraseFromParent();
    Builder.SetInsertPoint(Check1, Check1->begin());
    Value *LoadEnd = Builder.CreateAdd(
        LoadBegin, ConstantInt::get(IntPtrTy, LoadLoc.Size.getValue()),
        "load.end", true, true);
    Builder.CreateCondBr(Builder.CreateICmpULT(StoreBegin, LoadEnd), Copy,
                         Fusion);

    // Copy load operand to new alloca.
    Builder.SetInsertPoint(Copy, Copy->begin());
    AllocaInst *NewLd =
        Builder.CreateAlloca(Load->getType(), Load->getPointerAddressSpace());
    Builder.CreateMemCpy(NewLd, NewLd->getAlign(),
                         Load->getPointerOperand(), Load->getAlign(),
                         LoadLoc.Size.getValue());
    Builder.SetInsertPoint(Fusion, Fusion->begin());
    PHINode *PHI = Builder.CreatePHI(Load->getPointerOperandType(), 3);
    PHI->addIncoming(Load->getPointerOperand(), Check0);
    PHI->addIncoming(Load->getPointerOperand(), Check1);
    PHI->addIncoming(NewLd, Copy);

    // Adjust DT.
    DTUpdates.push_back({DT->Insert, Check0, Check1});
    DTUpdates.push_back({DT->Insert, Check0, Fusion});
    DTUpdates.push_back({DT->Insert, Check1, Copy});
    DTUpdates.push_back({DT->Insert, Check1, Fusion});
    DT->applyUpdates(DTUpdates);
    return PHI;
  }

  bool isFusionProfitable(CallInst *MatMul) {
    if (ForceFusion)
      return true;

    ShapeInfo LShape(MatMul->getArgOperand(2), MatMul->getArgOperand(3));
    ShapeInfo RShape(MatMul->getArgOperand(3), MatMul->getArgOperand(4));

    const unsigned R = LShape.NumRows;
    const unsigned C = RShape.NumColumns;
    const unsigned M = LShape.NumColumns;
    auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();

    const unsigned VF = std::max<unsigned>(
        TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
                .getFixedSize() /
            EltType->getPrimitiveSizeInBits().getFixedSize(),
        1U);

    // Cost model for tiling
    //
    // For tiling to be beneficial, we need reuse either along the R or
    // the C axis.  We vectorize along the R axis so that means at least
    // 3 elements.
    // TODO: Also consider cost of copying if operands alias.
    if (R <= VF && C == 1)
      return false;
    // Then we need enough elements to exceed the number of vector
    // registers we have.  Note that this is an oversimplification since
    // fusing also takes some extra loads which may exceed the number of
    // reloads necessary.
    unsigned Op0Regs = (R + VF - 1) / VF * M;
    unsigned Op1Regs = (M + VF - 1) / VF * C;
    return Op0Regs + Op1Regs >
           TTI.getNumberOfRegisters(TTI.getRegisterClassForType(true));
  }

  MatrixTy getZeroMatrix(Type *EltType, unsigned R, unsigned C) {
    MatrixTy Res;
    auto *ColumType = FixedVectorType::get(EltType, R);
    for (unsigned I = 0; I < C; ++I)
      Res.addVector(ConstantAggregateZero::get(ColumType));
    return Res;
  }

  void createTiledLoops(CallInst *MatMul, Value *LPtr, ShapeInfo LShape,
                        Value *RPtr, ShapeInfo RShape, StoreInst *Store) {
    auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();

    // Create the main tiling loop nest.
    TileInfo TI(LShape.NumRows, RShape.NumColumns, LShape.NumColumns, TileSize);
    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
    Instruction *InsertI = cast<Instruction>(MatMul);
    BasicBlock *Start = InsertI->getParent();
    BasicBlock *End =
        SplitBlock(InsertI->getParent(), InsertI, DT, LI, nullptr, "continue");
    IRBuilder<> Builder(MatMul);
    BasicBlock *InnerBody = TI.CreateTiledLoops(Start, End, Builder, DTU, *LI);

    Type *TileVecTy =
        FixedVectorType::get(MatMul->getType()->getScalarType(), TileSize);
    MatrixTy TileResult;
    // Insert in the inner loop header.
    Builder.SetInsertPoint(TI.InnerLoopHeader->getTerminator());
    // Create PHI nodes for the result columns to accumulate across iterations.
    SmallVector<PHINode *, 4> ColumnPhis;
    for (unsigned I = 0; I < TileSize; I++) {
      auto *Phi = Builder.CreatePHI(TileVecTy, 2, "result.vec." + Twine(I));
      Phi->addIncoming(ConstantAggregateZero::get(TileVecTy),
                       TI.RowLoopHeader->getSingleSuccessor());
      TileResult.addVector(Phi);
      ColumnPhis.push_back(Phi);
    }

    // Insert in the inner loop body, which computes
    //   Res += Load(CurrentRow, K) * Load(K, CurrentColumn)
    Builder.SetInsertPoint(InnerBody->getTerminator());
    // Load tiles of the operands.
    MatrixTy A = loadMatrix(LPtr, {}, false, LShape, TI.CurrentRow, TI.CurrentK,
                            {TileSize, TileSize}, EltType, Builder);
    MatrixTy B = loadMatrix(RPtr, {}, false, RShape, TI.CurrentK, TI.CurrentCol,
                            {TileSize, TileSize}, EltType, Builder);
    emitMatrixMultiply(TileResult, A, B, Builder, true, false,
                       getFastMathFlags(MatMul));
    // Store result after the inner loop is done.
    Builder.SetInsertPoint(TI.RowLoopLatch->getTerminator());
    storeMatrix(TileResult, Store->getPointerOperand(), Store->getAlign(),
                Store->isVolatile(), {LShape.NumRows, RShape.NumColumns},
                TI.CurrentRow, TI.CurrentCol, EltType, Builder);

    for (unsigned I = 0; I < TileResult.getNumVectors(); I++)
      ColumnPhis[I]->addIncoming(TileResult.getVector(I), TI.InnerLoopLatch);

    // Force unrolling of a few iterations of the inner loop, to make sure there
    // is enough work per iteration.
    // FIXME: The unroller should make this decision directly instead, but
    // currently the cost-model is not up to the task.
    unsigned InnerLoopUnrollCount = std::min(10u, LShape.NumColumns / TileSize);
    addStringMetadataToLoop(LI->getLoopFor(TI.InnerLoopHeader),
                            "llvm.loop.unroll.count", InnerLoopUnrollCount);
  }

  void emitSIMDTiling(CallInst *MatMul, LoadInst *LoadOp0, LoadInst *LoadOp1,
                      StoreInst *Store,
                      SmallPtrSetImpl<Instruction *> &FusedInsts) {
    assert(MatrixLayout == MatrixLayoutTy::ColumnMajor &&
           "Tiling only supported for column-major matrixes at the moment!");
    if (!isFusionProfitable(MatMul))
      return;

    ShapeInfo LShape(MatMul->getArgOperand(2), MatMul->getArgOperand(3));
    ShapeInfo RShape(MatMul->getArgOperand(3), MatMul->getArgOperand(4));

    const unsigned R = LShape.NumRows;
    const unsigned C = RShape.NumColumns;
    const unsigned M = LShape.NumColumns;
    auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();

    Value *APtr = getNonAliasingPointer(LoadOp0, Store, MatMul);
    Value *BPtr = getNonAliasingPointer(LoadOp1, Store, MatMul);
    Value *CPtr = Store->getPointerOperand();

    if (TileUseLoops && (R % TileSize == 0 && C % TileSize == 0))
      createTiledLoops(MatMul, APtr, LShape, BPtr, RShape, Store);
    else {
      IRBuilder<> Builder(Store);
      for (unsigned J = 0; J < C; J += TileSize)
        for (unsigned I = 0; I < R; I += TileSize) {
          const unsigned TileR = std::min(R - I, unsigned(TileSize));
          const unsigned TileC = std::min(C - J, unsigned(TileSize));
          MatrixTy Res = getZeroMatrix(EltType, TileR, TileC);

          for (unsigned K = 0; K < M; K += TileSize) {
            const unsigned TileM = std::min(M - K, unsigned(TileSize));
            MatrixTy A =
                loadMatrix(APtr, LoadOp0->getAlign(), LoadOp0->isVolatile(),
                           LShape, Builder.getInt64(I), Builder.getInt64(K),
                           {TileR, TileM}, EltType, Builder);
            MatrixTy B =
                loadMatrix(BPtr, LoadOp1->getAlign(), LoadOp1->isVolatile(),
                           RShape, Builder.getInt64(K), Builder.getInt64(J),
                           {TileM, TileC}, EltType, Builder);
            emitMatrixMultiply(Res, A, B, Builder, true, false,
                               getFastMathFlags(MatMul));
          }
          storeMatrix(Res, CPtr, Store->getAlign(), Store->isVolatile(), {R, M},
                      Builder.getInt64(I), Builder.getInt64(J), EltType,
                      Builder);
        }
    }

    // Mark eliminated instructions as fused and remove them.
    FusedInsts.insert(Store);
    FusedInsts.insert(MatMul);
    Store->eraseFromParent();
    MatMul->eraseFromParent();
    if (LoadOp0->hasNUses(0)) {
      FusedInsts.insert(LoadOp0);
      LoadOp0->eraseFromParent();
    }
    if (LoadOp1 != LoadOp0 && LoadOp1->hasNUses(0)) {
      FusedInsts.insert(LoadOp1);
      LoadOp1->eraseFromParent();
    }
  }

  /// Try to lower matrix multiply chains by fusing operations.
  ///
  /// Call finalizeLowering on lowered instructions.  Instructions that are
  /// completely eliminated by fusion are added to \p FusedInsts.
  void LowerMatrixMultiplyFused(CallInst *MatMul,
                                SmallPtrSetImpl<Instruction *> &FusedInsts) {
    if (!FuseMatrix || !DT)
      return;

    assert(AA && LI && "Analyses should be available");

    Value *A = MatMul->getArgOperand(0);
    Value *B = MatMul->getArgOperand(1);

    // We can fold the transpose into the operand that is used to fetch scalars.
    Value *T;
    if (MatrixLayout == MatrixLayoutTy::ColumnMajor
            ? match(B, m_Intrinsic<Intrinsic::matrix_transpose>(m_Value(T)))
            : match(A, m_Intrinsic<Intrinsic::matrix_transpose>(m_Value(T)))) {
      IRBuilder<> Builder(MatMul);
      auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();
      ShapeInfo LShape(MatMul->getArgOperand(2), MatMul->getArgOperand(3));
      ShapeInfo RShape(MatMul->getArgOperand(3), MatMul->getArgOperand(4));
      const unsigned R = LShape.NumRows;
      const unsigned M = LShape.NumColumns;
      const unsigned C = RShape.NumColumns;

      MatrixTy MA;
      MatrixTy MB;

      Value *Transpose;
      if (MatrixLayout == MatrixLayoutTy::ColumnMajor) {
        MA = getMatrix(A, ShapeInfo(R, M), Builder);
        MB = getMatrix(T, ShapeInfo(C, M), Builder);
        Transpose = B;
      } else {
        MA = getMatrix(T, ShapeInfo(R, M), Builder);
        MB = getMatrix(B, ShapeInfo(C, M), Builder);
        Transpose = A;
      }

      // Initialize the output
      MatrixTy Result(R, C, EltType);

      emitMatrixMultiply(Result, MA, MB, Builder, false, true,
                         getFastMathFlags(MatMul));

      FusedInsts.insert(MatMul);
      if (Transpose->hasOneUse()) {
        FusedInsts.insert(cast<Instruction>(Transpose));
        ToRemove.push_back(cast<Instruction>(Transpose));
        // TODO: add a fake entry for the folded instruction so that this is
        // included in the expression in the remark.
        Inst2ColumnMatrix[Transpose] = MatrixTy(M, C, EltType);
      }
      finalizeLowering(MatMul, Result, Builder);
      return;
    }

    if (!MatMul->hasOneUse() || MatrixLayout != MatrixLayoutTy::ColumnMajor)
      return;

    // Lower {ld, ld} -> matmul -> st chains.  No need to call finalizeLowering
    // since the single store user will be lowered as part of this.
    auto *LoadOp0 = dyn_cast<LoadInst>(A);
    auto *LoadOp1 = dyn_cast<LoadInst>(B);
    auto *Store = dyn_cast<StoreInst>(*MatMul->user_begin());
    if (LoadOp0 && LoadOp1 && Store) {
      // The store address must dominate the MatMul instruction, otherwise
      // we create invalid IR.
      SetVector<Value *> WorkList;
      WorkList.insert(Store->getOperand(1));
      SmallVector<Instruction *> ToHoist;
      for (unsigned I = 0; I != WorkList.size(); ++I) {
        Value *Current = WorkList[I];
        auto *CurrI = dyn_cast<Instruction>(Current);
        if (!CurrI)
          continue;
        if (isa<PHINode>(CurrI))
          return;
        if (DT->dominates(CurrI, MatMul))
          continue;
        if (CurrI->mayHaveSideEffects() || CurrI->mayReadFromMemory())
          return;
        ToHoist.push_back(CurrI);
        WorkList.insert(CurrI->op_begin(), CurrI->op_end());
      }

      sort(ToHoist, [this](Instruction *A, Instruction *B) {
        return DT->dominates(A, B);
      });
      for (Instruction *I : ToHoist)
        I->moveBefore(MatMul);

      emitSIMDTiling(MatMul, LoadOp0, LoadOp1, Store, FusedInsts);
      return;
    }
  }

  /// Lowers llvm.matrix.multiply.
  void LowerMultiply(CallInst *MatMul) {
    IRBuilder<> Builder(MatMul);
    auto *EltType = cast<VectorType>(MatMul->getType())->getElementType();
    ShapeInfo LShape(MatMul->getArgOperand(2), MatMul->getArgOperand(3));
    ShapeInfo RShape(MatMul->getArgOperand(3), MatMul->getArgOperand(4));

    const MatrixTy &Lhs = getMatrix(MatMul->getArgOperand(0), LShape, Builder);
    const MatrixTy &Rhs = getMatrix(MatMul->getArgOperand(1), RShape, Builder);
    assert(Lhs.getElementType() == Rhs.getElementType() &&
           "Matrix multiply argument element types do not match.");

    const unsigned R = LShape.NumRows;
    const unsigned C = RShape.NumColumns;
    assert(LShape.NumColumns == RShape.NumRows);

    // Initialize the output
    MatrixTy Result(R, C, EltType);
    assert(Lhs.getElementType() == Result.getElementType() &&
           "Matrix multiply result element type does not match arguments.");

    emitMatrixMultiply(Result, Lhs, Rhs, Builder, false, false,
                       getFastMathFlags(MatMul));
    finalizeLowering(MatMul, Result, Builder);
  }

  /// Lowers llvm.matrix.transpose.
  void LowerTranspose(CallInst *Inst) {
    MatrixTy Result;
    IRBuilder<> Builder(Inst);
    Value *InputVal = Inst->getArgOperand(0);
    VectorType *VectorTy = cast<VectorType>(InputVal->getType());
    ShapeInfo ArgShape(Inst->getArgOperand(1), Inst->getArgOperand(2));
    MatrixTy InputMatrix = getMatrix(InputVal, ArgShape, Builder);

    const unsigned NewNumVecs =
        InputMatrix.isColumnMajor() ? ArgShape.NumRows : ArgShape.NumColumns;
    const unsigned NewNumElts =
        InputMatrix.isColumnMajor() ? ArgShape.NumColumns : ArgShape.NumRows;

    for (unsigned I = 0; I < NewNumVecs; ++I) {
      // Build a single result vector. First initialize it.
      Value *ResultVector = UndefValue::get(
          FixedVectorType::get(VectorTy->getElementType(), NewNumElts));
      // Go through the old elements and insert it into the resulting vector.
      for (auto J : enumerate(InputMatrix.vectors())) {
        Value *Elt = Builder.CreateExtractElement(J.value(), I);
        // Row and column indices are transposed.
        ResultVector =
            Builder.CreateInsertElement(ResultVector, Elt, J.index());
      }
      Result.addVector(ResultVector);
    }

    // TODO: Improve estimate of operations needed for transposes. Currently we
    // just count the insertelement/extractelement instructions, but do not
    // account for later simplifications/combines.
    finalizeLowering(
        Inst,
        Result.addNumComputeOps(2 * ArgShape.NumRows * ArgShape.NumColumns)
            .addNumExposedTransposes(1),
        Builder);
  }

  /// Lower load instructions, if shape information is available.
  bool VisitLoad(LoadInst *Inst, Value *Ptr, IRBuilder<> &Builder) {
    auto I = ShapeMap.find(Inst);
    if (I == ShapeMap.end())
      return false;

    LowerLoad(Inst, Ptr, Inst->getAlign(),
              Builder.getInt64(I->second.getStride()), Inst->isVolatile(),
              I->second);
    return true;
  }

  bool VisitStore(StoreInst *Inst, Value *StoredVal, Value *Ptr,
                  IRBuilder<> &Builder) {
    auto I = ShapeMap.find(StoredVal);
    if (I == ShapeMap.end())
      return false;

    LowerStore(Inst, StoredVal, Ptr, Inst->getAlign(),
               Builder.getInt64(I->second.getStride()), Inst->isVolatile(),
               I->second);
    return true;
  }

  /// Lower binary operators, if shape information is available.
  bool VisitBinaryOperator(BinaryOperator *Inst) {
    auto I = ShapeMap.find(Inst);
    if (I == ShapeMap.end())
      return false;

    Value *Lhs = Inst->getOperand(0);
    Value *Rhs = Inst->getOperand(1);

    IRBuilder<> Builder(Inst);
    ShapeInfo &Shape = I->second;

    MatrixTy Result;
    MatrixTy A = getMatrix(Lhs, Shape, Builder);
    MatrixTy B = getMatrix(Rhs, Shape, Builder);
    assert(A.isColumnMajor() == B.isColumnMajor() &&
           Result.isColumnMajor() == A.isColumnMajor() &&
           "operands must agree on matrix layout");

    Builder.setFastMathFlags(getFastMathFlags(Inst));

    // Helper to perform binary op on vectors.
    auto BuildVectorOp = [&Builder, Inst](Value *LHS, Value *RHS) {
      switch (Inst->getOpcode()) {
      case Instruction::Add:
        return Builder.CreateAdd(LHS, RHS);
      case Instruction::Mul:
        return Builder.CreateMul(LHS, RHS);
      case Instruction::Sub:
        return Builder.CreateSub(LHS, RHS);
      case Instruction::FAdd:
        return Builder.CreateFAdd(LHS, RHS);
      case Instruction::FMul:
        return Builder.CreateFMul(LHS, RHS);
      case Instruction::FSub:
        return Builder.CreateFSub(LHS, RHS);
      default:
        llvm_unreachable("Unsupported binary operator for matrix");
      }
    };

    for (unsigned I = 0; I < Shape.getNumVectors(); ++I)
      Result.addVector(BuildVectorOp(A.getVector(I), B.getVector(I)));

    finalizeLowering(Inst,
                     Result.addNumComputeOps(getNumOps(Result.getVectorTy()) *
                                             Result.getNumVectors()),
                     Builder);
    return true;
  }

  /// Lower unary operators, if shape information is available.
  bool VisitUnaryOperator(UnaryOperator *Inst) {
    auto I = ShapeMap.find(Inst);
    if (I == ShapeMap.end())
      return false;

    Value *Op = Inst->getOperand(0);

    IRBuilder<> Builder(Inst);
    ShapeInfo &Shape = I->second;

    MatrixTy Result;
    MatrixTy M = getMatrix(Op, Shape, Builder);

    Builder.setFastMathFlags(getFastMathFlags(Inst));

    // Helper to perform unary op on vectors.
    auto BuildVectorOp = [&Builder, Inst](Value *Op) {
      switch (Inst->getOpcode()) {
      case Instruction::FNeg:
        return Builder.CreateFNeg(Op);
      default:
        llvm_unreachable("Unsupported unary operator for matrix");
      }
    };

    for (unsigned I = 0; I < Shape.getNumVectors(); ++I)
      Result.addVector(BuildVectorOp(M.getVector(I)));

    finalizeLowering(Inst,
                     Result.addNumComputeOps(getNumOps(Result.getVectorTy()) *
                                             Result.getNumVectors()),
                     Builder);
    return true;
  }

  /// Helper to linearize a matrix expression tree into a string. Currently
  /// matrix expressions are linarized by starting at an expression leaf and
  /// linearizing bottom up.
  struct ExprLinearizer {
    unsigned LengthToBreak = 100;
    std::string Str;
    raw_string_ostream Stream;
    unsigned LineLength = 0;
    const DataLayout &DL;

    /// Mapping from instructions to matrixes. It is used to identify
    /// matrix instructions.
    const MapVector<Value *, MatrixTy> &Inst2Matrix;

    /// Mapping from values to the leaves of all expressions that the value is
    /// part of.
    const DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared;

    /// Set of matrix expressions in the scope of a given DISubprogram.
    const SmallSetVector<Value *, 32> &ExprsInSubprogram;

    /// Leaf node of the expression to linearize.
    Value *Leaf;

    /// Used to keep track of sub-expressions that get reused while linearizing
    /// the expression. Re-used sub-expressions are marked as (reused).
    SmallPtrSet<Value *, 8> ReusedExprs;

    ExprLinearizer(const DataLayout &DL,
                   const MapVector<Value *, MatrixTy> &Inst2Matrix,
                   const DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared,
                   const SmallSetVector<Value *, 32> &ExprsInSubprogram,
                   Value *Leaf)
        : Stream(Str), DL(DL), Inst2Matrix(Inst2Matrix), Shared(Shared),
          ExprsInSubprogram(ExprsInSubprogram), Leaf(Leaf) {}

    void indent(unsigned N) {
      LineLength += N;
      for (unsigned i = 0; i < N; i++)
        Stream << " ";
    }

    void lineBreak() {
      Stream << "\n";
      LineLength = 0;
    }

    void maybeIndent(unsigned Indent) {
      if (LineLength >= LengthToBreak)
        lineBreak();

      if (LineLength == 0)
        indent(Indent);
    }

    void write(StringRef S) {
      LineLength += S.size();
      Stream << S;
    }

    Value *getUnderlyingObjectThroughLoads(Value *V) {
      if (Value *Ptr = getPointerOperand(V))
        return getUnderlyingObjectThroughLoads(Ptr);
      else if (V->getType()->isPointerTy())
        return getUnderlyingObject(V);
      return V;
    }

    /// Returns true if \p V is a matrix value in the given subprogram.
    bool isMatrix(Value *V) const { return ExprsInSubprogram.count(V); }

    /// If \p V is a matrix value, print its shape as as NumRows x NumColumns to
    /// \p SS.
    void prettyPrintMatrixType(Value *V, raw_string_ostream &SS) {
      auto M = Inst2Matrix.find(V);
      if (M == Inst2Matrix.end())
        SS << "unknown";
      else {
        SS << M->second.getNumRows();
        SS << "x";
        SS << M->second.getNumColumns();
      }
    }

    /// Write the called function name. Handles calls to llvm.matrix.*
    /// specially: we write the name, followed by the dimensions of the input
    /// matrixes, followed by the scalar type name.
    void writeFnName(CallInst *CI) {
      if (!CI->getCalledFunction())
        write("<no called fn>");
      else {
        StringRef Name = CI->getCalledFunction()->getName();
        if (!Name.startswith("llvm.matrix")) {
          write(Name);
          return;
        }
        auto *II = cast<IntrinsicInst>(CI);
        write(Intrinsic::getBaseName(II->getIntrinsicID())
                  .drop_front(StringRef("llvm.matrix.").size()));
        write(".");
        std::string Tmp;
        raw_string_ostream SS(Tmp);

        switch (II->getIntrinsicID()) {
        case Intrinsic::matrix_multiply:
          prettyPrintMatrixType(II->getOperand(0), SS);
          SS << ".";
          prettyPrintMatrixType(II->getOperand(1), SS);
          SS << "." << *II->getType()->getScalarType();
          break;
        case Intrinsic::matrix_transpose:
          prettyPrintMatrixType(II->getOperand(0), SS);
          SS << "." << *II->getType()->getScalarType();
          break;
        case Intrinsic::matrix_column_major_load:
          prettyPrintMatrixType(II, SS);
          SS << "." << *II->getType()->getScalarType();
          break;
        case Intrinsic::matrix_column_major_store:
          prettyPrintMatrixType(II->getOperand(0), SS);
          SS << "." << *II->getOperand(0)->getType()->getScalarType();
          break;
        default:
          llvm_unreachable("Unhandled case");
        }
        SS.flush();
        write(Tmp);
      }
    }

    unsigned getNumShapeArgs(CallInst *CI) const {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::matrix_multiply:
          return 3;
        case Intrinsic::matrix_transpose:
          return 2;
        case Intrinsic::matrix_column_major_load:
        case Intrinsic::matrix_column_major_store:
          return 3;
        default:
          return 0;
        }
      }
      return 0;
    }

    /// Special printing for values: for pointers, we print if they refer to an
    /// (function) external address or a stack address, for other values we
    /// either print the constant or "scalar"/"matrix" for other values.
    void write(Value *V) {
      V = getUnderlyingObjectThroughLoads(V);
      if (V->getType()->isPointerTy()) {
        if (isa<AllocaInst>(V)) {
          Stream << "stack addr";
          LineLength += StringRef("stack addr").size();
        } else {
          Stream << "addr";
          LineLength += StringRef("addr").size();
        }
        if (!V->getName().empty()) {
          Stream << " %" << V->getName() << "";
          LineLength += V->getName().size() + 2;
        }
        return;
      }

      std::string Tmp;
      raw_string_ostream TmpStream(Tmp);

      if (auto *CI = dyn_cast<ConstantInt>(V))
        TmpStream << CI->getValue();
      else if (isa<Constant>(V))
        TmpStream << "constant";
      else {
        if (isMatrix(V))
          TmpStream << "matrix";
        else
          TmpStream << "scalar";
      }
      TmpStream.flush();
      Tmp = std::string(StringRef(Tmp).trim());
      LineLength += Tmp.size();
      Stream << Tmp;
    }

    /// Linearize expression \p Expr starting at an indentation of \p Indent.
    /// Expressions that are re-used multiple times are prefixed with (reused)
    /// at the re-used root instruction.
    void linearizeExpr(Value *Expr, unsigned Indent, bool ParentReused,
                       bool ParentShared) {
      auto *I = cast<Instruction>(Expr);
      maybeIndent(Indent);
      SmallVector<Value *, 8> Ops;

      // Is Expr shared with other expression leaves?
      bool ExprShared = false;

      // Deal with shared subtrees. Mark them as shared, if required.
      if (!ParentShared) {
        auto SI = Shared.find(Expr);
        assert(SI != Shared.end() && SI->second.count(Leaf));

        for (Value *S : SI->second) {
          if (S == Leaf)
            continue;
          DebugLoc DL = cast<Instruction>(S)->getDebugLoc();
          write("shared with remark at line " + std::to_string(DL.getLine()) +
                " column " + std::to_string(DL.getCol()) + " (");
        }
        ExprShared = SI->second.size() > 1;
      }

      bool Reused = !ReusedExprs.insert(Expr).second;
      if (Reused && !ParentReused)
        write("(reused) ");

      if (auto *CI = dyn_cast<CallInst>(I)) {
        writeFnName(CI);

        Ops.append(CI->arg_begin(), CI->arg_end() - getNumShapeArgs(CI));
      } else if (isa<BitCastInst>(Expr)) {
        // Special case bitcasts, which are used to materialize matrixes from
        // non-matrix ops.
        write("matrix");
        return;
      } else {
        Ops.append(I->value_op_begin(), I->value_op_end());
        write(std::string(I->getOpcodeName()));
      }

      write(std::string("("));

      unsigned NumOpsToBreak = 1;
      if (match(Expr, m_Intrinsic<Intrinsic::matrix_column_major_load>()))
        NumOpsToBreak = 2;

      for (Value *Op : Ops) {
        if (Ops.size() > NumOpsToBreak)
          lineBreak();

        maybeIndent(Indent + 1);
        if (isMatrix(Op))
          linearizeExpr(Op, Indent + 1, Reused, ExprShared);
        else
          write(Op);
        if (Op != Ops.back())
          write(", ");
      }

      write(")");
    }

    const std::string &getResult() {
      Stream.flush();
      return Str;
    }
  };

  /// Generate remarks for matrix operations in a function. To generate remarks
  /// for matrix expressions, the following approach is used:
  /// 1. Use the inlined-at debug information to group matrix operations to the
  ///    DISubprograms they are contained in.
  /// 2. Collect leaves of matrix expressions (done in
  ///    RemarkGenerator::getExpressionLeaves) for each subprogram - expression
  //     mapping.  Leaves are lowered matrix instructions without other matrix
  //     users (like stores) in the current subprogram.
  /// 3. For each leaf, create a remark containing a linearizied version of the
  ///    matrix expression. The expression is linearized by a recursive
  ///    bottom-up traversal of the matrix operands, starting at a leaf. Note
  ///    that multiple leaves can share sub-expressions. Shared subexpressions
  ///    are explicitly marked as shared().
  struct RemarkGenerator {
    const MapVector<Value *, MatrixTy> &Inst2Matrix;
    OptimizationRemarkEmitter &ORE;
    Function &Func;
    const DataLayout &DL;

    RemarkGenerator(const MapVector<Value *, MatrixTy> &Inst2Matrix,
                    OptimizationRemarkEmitter &ORE, Function &Func)
        : Inst2Matrix(Inst2Matrix), ORE(ORE), Func(Func),
          DL(Func.getParent()->getDataLayout()) {}

    /// Return all leaves of the expressions in \p ExprsInSubprogram. Those are
    /// instructions in Inst2Matrix returning void or without any users in
    /// \p ExprsInSubprogram. Currently that should only include stores.
    SmallVector<Value *, 4>
    getExpressionLeaves(const SmallSetVector<Value *, 32> &ExprsInSubprogram) {
      SmallVector<Value *, 4> Leaves;
      for (auto *Expr : ExprsInSubprogram)
        if (Expr->getType()->isVoidTy() ||
            !any_of(Expr->users(), [&ExprsInSubprogram](User *U) {
              return ExprsInSubprogram.count(U);
            }))
          Leaves.push_back(Expr);
      return Leaves;
    }

    /// Recursively traverse expression \p V starting at \p Leaf and add \p Leaf
    /// to all visited expressions in \p Shared. Limit the matrix operations to
    /// the ones in \p ExprsInSubprogram.
    void collectSharedInfo(Value *Leaf, Value *V,
                           const SmallSetVector<Value *, 32> &ExprsInSubprogram,
                           DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared) {

      if (!ExprsInSubprogram.count(V))
        return;

      auto I = Shared.insert({V, {}});
      I.first->second.insert(Leaf);

      for (Value *Op : cast<Instruction>(V)->operand_values())
        collectSharedInfo(Leaf, Op, ExprsInSubprogram, Shared);
    }

    /// Calculate the number of exclusive and shared op counts for expression
    /// starting at \p V. Expressions used multiple times are counted once.
    /// Limit the matrix operations to the ones in \p ExprsInSubprogram.
    std::pair<OpInfoTy, OpInfoTy>
    sumOpInfos(Value *Root, SmallPtrSetImpl<Value *> &ReusedExprs,
               const SmallSetVector<Value *, 32> &ExprsInSubprogram,
               DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared) const {
      if (!ExprsInSubprogram.count(Root))
        return {};

      // Already counted this expression. Stop.
      if (!ReusedExprs.insert(Root).second)
        return {};

      OpInfoTy SharedCount;
      OpInfoTy Count;

      auto I = Shared.find(Root);
      auto CM = Inst2Matrix.find(Root);
      if (I->second.size() == 1)
        Count = CM->second.getOpInfo();
      else
        SharedCount = CM->second.getOpInfo();

      for (Value *Op : cast<Instruction>(Root)->operand_values()) {
        auto C = sumOpInfos(Op, ReusedExprs, ExprsInSubprogram, Shared);
        Count += C.first;
        SharedCount += C.second;
      }
      return {Count, SharedCount};
    }

    void emitRemarks() {
      if (!ORE.allowExtraAnalysis(DEBUG_TYPE))
        return;

      // Map matrix operations to their containting subprograms, by traversing
      // the inlinedAt chain. If the function does not have a DISubprogram, we
      // only map them to the containing function.
      MapVector<DISubprogram *, SmallVector<Value *, 8>> Subprog2Exprs;
      for (auto &KV : Inst2Matrix) {
        if (Func.getSubprogram()) {
          auto *I = cast<Instruction>(KV.first);
          DILocation *Context = I->getDebugLoc();
          while (Context) {
            auto I =
                Subprog2Exprs.insert({getSubprogram(Context->getScope()), {}});
            I.first->second.push_back(KV.first);
            Context = DebugLoc(Context).getInlinedAt();
          }
        } else {
          auto I = Subprog2Exprs.insert({nullptr, {}});
          I.first->second.push_back(KV.first);
        }
      }
      for (auto &KV : Subprog2Exprs) {
        SmallSetVector<Value *, 32> ExprsInSubprogram(KV.second.begin(),
                                                      KV.second.end());
        auto Leaves = getExpressionLeaves(ExprsInSubprogram);

        DenseMap<Value *, SmallPtrSet<Value *, 2>> Shared;
        for (Value *Leaf : Leaves)
          collectSharedInfo(Leaf, Leaf, ExprsInSubprogram, Shared);

        // Generate remarks for each leaf.
        for (auto *L : Leaves) {

          DebugLoc Loc = cast<Instruction>(L)->getDebugLoc();
          DILocation *Context = cast<Instruction>(L)->getDebugLoc();
          while (Context) {
            if (getSubprogram(Context->getScope()) == KV.first) {
              Loc = Context;
              break;
            }
            Context = DebugLoc(Context).getInlinedAt();
          }

          SmallPtrSet<Value *, 8> ReusedExprs;
          OpInfoTy Counts, SharedCounts;
          std::tie(Counts, SharedCounts) =
              sumOpInfos(L, ReusedExprs, ExprsInSubprogram, Shared);

          OptimizationRemark Rem(DEBUG_TYPE, "matrix-lowered", Loc,
                                 cast<Instruction>(L)->getParent());

          Rem << "Lowered with ";
          Rem << ore::NV("NumStores", Counts.NumStores) << " stores, "
              << ore::NV("NumLoads", Counts.NumLoads) << " loads, "
              << ore::NV("NumComputeOps", Counts.NumComputeOps)
              << " compute ops, "
              << ore::NV("NumExposedTransposes", Counts.NumExposedTransposes)
              << " exposed transposes";

          if (SharedCounts.NumStores > 0 || SharedCounts.NumLoads > 0 ||
              SharedCounts.NumComputeOps > 0) {
            Rem << ",\nadditionally "
                << ore::NV("NumStores", SharedCounts.NumStores) << " stores, "
                << ore::NV("NumLoads", SharedCounts.NumLoads) << " loads, "
                << ore::NV("NumFPOps", SharedCounts.NumComputeOps)
                << " compute ops"
                << " are shared with other expressions";
          }

          Rem << ("\n" + linearize(L, Shared, ExprsInSubprogram, DL));
          ORE.emit(Rem);
        }
      }
    }

    std::string
    linearize(Value *L,
              const DenseMap<Value *, SmallPtrSet<Value *, 2>> &Shared,
              const SmallSetVector<Value *, 32> &ExprsInSubprogram,
              const DataLayout &DL) {
      ExprLinearizer Lin(DL, Inst2Matrix, Shared, ExprsInSubprogram, L);
      Lin.linearizeExpr(L, 0, false, false);
      return Lin.getResult();
    }
  };
};
} // namespace

PreservedAnalyses LowerMatrixIntrinsicsPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  OptimizationRemarkEmitter *ORE = nullptr;
  AAResults *AA = nullptr;
  DominatorTree *DT = nullptr;
  LoopInfo *LI = nullptr;

  if (!Minimal) {
    ORE = &AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
    AA = &AM.getResult<AAManager>(F);
    DT = &AM.getResult<DominatorTreeAnalysis>(F);
    LI = &AM.getResult<LoopAnalysis>(F);
  }

  LowerMatrixIntrinsics LMT(F, TTI, AA, DT, LI, ORE);
  if (LMT.Visit()) {
    PreservedAnalyses PA;
    if (!Minimal) {
      PA.preserve<LoopAnalysis>();
      PA.preserve<DominatorTreeAnalysis>();
    }
    return PA;
  }
  return PreservedAnalyses::all();
}

void LowerMatrixIntrinsicsPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<LowerMatrixIntrinsicsPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << "<";
  if (Minimal)
    OS << "minimal";
  OS << ">";
}

namespace {

class LowerMatrixIntrinsicsLegacyPass : public FunctionPass {
public:
  static char ID;

  LowerMatrixIntrinsicsLegacyPass() : FunctionPass(ID) {
    initializeLowerMatrixIntrinsicsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    LowerMatrixIntrinsics LMT(F, TTI, &AA, &DT, &LI, &ORE);
    bool C = LMT.Visit();
    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
  }
};
} // namespace

static const char pass_name[] = "Lower the matrix intrinsics";
char LowerMatrixIntrinsicsLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LowerMatrixIntrinsicsLegacyPass, DEBUG_TYPE, pass_name,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LowerMatrixIntrinsicsLegacyPass, DEBUG_TYPE, pass_name,
                    false, false)

Pass *llvm::createLowerMatrixIntrinsicsPass() {
  return new LowerMatrixIntrinsicsLegacyPass();
}

namespace {

/// A lightweight version of the matrix lowering pass that only requires TTI.
/// Advanced features that require DT, AA or ORE like tiling are disabled. This
/// is used to lower matrix intrinsics if the main lowering pass is not run, for
/// example with -O0.
class LowerMatrixIntrinsicsMinimalLegacyPass : public FunctionPass {
public:
  static char ID;

  LowerMatrixIntrinsicsMinimalLegacyPass() : FunctionPass(ID) {
    initializeLowerMatrixIntrinsicsMinimalLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    LowerMatrixIntrinsics LMT(F, TTI, nullptr, nullptr, nullptr, nullptr);
    bool C = LMT.Visit();
    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // namespace

static const char pass_name_minimal[] = "Lower the matrix intrinsics (minimal)";
char LowerMatrixIntrinsicsMinimalLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LowerMatrixIntrinsicsMinimalLegacyPass,
                      "lower-matrix-intrinsics-minimal", pass_name_minimal,
                      false, false)
INITIALIZE_PASS_END(LowerMatrixIntrinsicsMinimalLegacyPass,
                    "lower-matrix-intrinsics-minimal", pass_name_minimal, false,
                    false)

Pass *llvm::createLowerMatrixIntrinsicsMinimalPass() {
  return new LowerMatrixIntrinsicsMinimalLegacyPass();
}
