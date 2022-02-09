//===-- FirBuilder.h -- FIR operation builder -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
#define FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir {
class AbstractArrayBox;
class ExtendedValue;
class BoxValue;

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder {
public:
  explicit FirOpBuilder(mlir::Operation *op, const fir::KindMapping &kindMap)
      : OpBuilder{op}, kindMap{kindMap} {}
  explicit FirOpBuilder(mlir::OpBuilder &builder,
                        const fir::KindMapping &kindMap)
      : OpBuilder{builder}, kindMap{kindMap} {}

  /// Get the current Region of the insertion point.
  mlir::Region &getRegion() { return *getBlock()->getParent(); }

  /// Get the current Module
  mlir::ModuleOp getModule() {
    return getRegion().getParentOfType<mlir::ModuleOp>();
  }

  /// Get the current Function
  mlir::FuncOp getFunction() {
    return getRegion().getParentOfType<mlir::FuncOp>();
  }

  /// Get a reference to the kind map.
  const fir::KindMapping &getKindMap() { return kindMap; }

  /// The LHS and RHS are not always in agreement in terms of
  /// type. In some cases, the disagreement is between COMPLEX and other scalar
  /// types. In that case, the conversion must insert/extract out of a COMPLEX
  /// value to have the proper semantics and be strongly typed. For e.g for
  /// converting an integer/real to a complex, the real part is filled using
  /// the integer/real after type conversion and the imaginary part is zero.
  mlir::Value convertWithSemantics(mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val);

  /// Get the entry block of the current Function
  mlir::Block *getEntryBlock() { return &getFunction().front(); }

  /// Get the block for adding Allocas. If OpenMP is enabled then get the
  /// the alloca block from an Operation which can be Outlined. Otherwise
  /// use the entry block of the current Function
  mlir::Block *getAllocaBlock();

  /// Safely create a reference type to the type `eleTy`.
  mlir::Type getRefType(mlir::Type eleTy);

  /// Create a sequence of `eleTy` with `rank` dimensions of unknown size.
  mlir::Type getVarLenSeqTy(mlir::Type eleTy, unsigned rank = 1);

  /// Get character length type
  mlir::Type getCharacterLengthType() { return getIndexType(); }

  /// Get the integer type whose bit width corresponds to the width of pointer
  /// types, or is bigger.
  mlir::Type getIntPtrType() {
    // TODO: Delay the need of such type until codegen or find a way to use
    // llvm::DataLayout::getPointerSizeInBits here.
    return getI64Type();
  }

  /// Get the mlir real type that implements fortran REAL(kind).
  mlir::Type getRealType(int kind);

  /// Create a null constant memory reference of type \p ptrType.
  /// If \p ptrType is not provided, !fir.ref<none> type will be used.
  mlir::Value createNullConstant(mlir::Location loc, mlir::Type ptrType = {});

  /// Create an integer constant of type \p type and value \p i.
  mlir::Value createIntegerConstant(mlir::Location loc, mlir::Type integerType,
                                    std::int64_t i);

  /// Create a real constant from an integer value.
  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 llvm::APFloat::integerPart val);

  /// Create a real constant from an APFloat value.
  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 const llvm::APFloat &val);

  /// Create a real constant of type \p realType with a value zero.
  mlir::Value createRealZeroConstant(mlir::Location loc, mlir::Type realType) {
    return createRealConstant(loc, realType, 0u);
  }

  /// Create a slot for a local on the stack. Besides the variable's type and
  /// shape, it may be given name, pinned, or target attributes.
  mlir::Value allocateLocal(mlir::Location loc, mlir::Type ty,
                            llvm::StringRef uniqName, llvm::StringRef name,
                            bool pinned, llvm::ArrayRef<mlir::Value> shape,
                            llvm::ArrayRef<mlir::Value> lenParams,
                            bool asTarget = false);
  mlir::Value allocateLocal(mlir::Location loc, mlir::Type ty,
                            llvm::StringRef uniqName, llvm::StringRef name,
                            llvm::ArrayRef<mlir::Value> shape,
                            llvm::ArrayRef<mlir::Value> lenParams,
                            bool asTarget = false);

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::StringRef name = {},
                              mlir::ValueRange shape = {},
                              mlir::ValueRange lenParams = {},
                              llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

  /// Create an unnamed and untracked temporary on the stack.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              mlir::ValueRange shape) {
    return createTemporary(loc, type, llvm::StringRef{}, shape);
  }

  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    return createTemporary(loc, type, llvm::StringRef{}, {}, {}, attrs);
  }

  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::StringRef name,
                              llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    return createTemporary(loc, type, name, {}, {}, attrs);
  }

  /// Create a global value.
  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name,
                             mlir::StringAttr linkage = {},
                             mlir::Attribute value = {}, bool isConst = false);

  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name, bool isConst,
                             std::function<void(FirOpBuilder &)> bodyBuilder,
                             mlir::StringAttr linkage = {});

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name,
                                     mlir::StringAttr linkage = {},
                                     mlir::Attribute value = {}) {
    return createGlobal(loc, type, name, linkage, value, /*isConst=*/true);
  }

  fir::GlobalOp
  createGlobalConstant(mlir::Location loc, mlir::Type type,
                       llvm::StringRef name,
                       std::function<void(FirOpBuilder &)> bodyBuilder,
                       mlir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, /*isConst=*/true, bodyBuilder,
                        linkage);
  }

  /// Convert a StringRef string into a fir::StringLitOp.
  fir::StringLitOp createStringLitOp(mlir::Location loc,
                                     llvm::StringRef string);

  //===--------------------------------------------------------------------===//
  // Linkage helpers (inline). The default linkage is external.
  //===--------------------------------------------------------------------===//

  mlir::StringAttr createCommonLinkage() { return getStringAttr("common"); }

  mlir::StringAttr createInternalLinkage() { return getStringAttr("internal"); }

  mlir::StringAttr createLinkOnceLinkage() { return getStringAttr("linkonce"); }

  mlir::StringAttr createWeakLinkage() { return getStringAttr("weak"); }

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  mlir::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), name);
  }

  static mlir::FuncOp getNamedFunction(mlir::ModuleOp module,
                                       llvm::StringRef name);

  fir::GlobalOp getNamedGlobal(llvm::StringRef name) {
    return getNamedGlobal(getModule(), name);
  }

  static fir::GlobalOp getNamedGlobal(mlir::ModuleOp module,
                                      llvm::StringRef name);

  /// Lazy creation of fir.convert op.
  mlir::Value createConvert(mlir::Location loc, mlir::Type toTy,
                            mlir::Value val);

  /// Create a new FuncOp. If the function may have already been created, use
  /// `addNamedFunction` instead.
  mlir::FuncOp createFunction(mlir::Location loc, llvm::StringRef name,
                              mlir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty);
  }

  static mlir::FuncOp createFunction(mlir::Location loc, mlir::ModuleOp module,
                                     llvm::StringRef name,
                                     mlir::FunctionType ty);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  mlir::FuncOp addNamedFunction(mlir::Location loc, llvm::StringRef name,
                                mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(loc, name, ty);
  }

  static mlir::FuncOp addNamedFunction(mlir::Location loc,
                                       mlir::ModuleOp module,
                                       llvm::StringRef name,
                                       mlir::FunctionType ty) {
    if (auto func = getNamedFunction(module, name))
      return func;
    return createFunction(loc, module, name, ty);
  }

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Location loc, mlir::Value val) {
    return createConvert(loc, getIndexType(), val);
  }

  /// Construct one of the two forms of shape op from an array box.
  mlir::Value genShape(mlir::Location loc, const fir::AbstractArrayBox &arr);
  mlir::Value genShape(mlir::Location loc, llvm::ArrayRef<mlir::Value> shift,
                       llvm::ArrayRef<mlir::Value> exts);
  mlir::Value genShape(mlir::Location loc, llvm::ArrayRef<mlir::Value> exts);

  /// Create one of the shape ops given an extended value. For a boxed value,
  /// this may create a `fir.shift` op.
  mlir::Value createShape(mlir::Location loc, const fir::ExtendedValue &exv);

  /// Create a boxed value (Fortran descriptor) to be passed to the runtime.
  /// \p exv is an extended value holding a memory reference to the object that
  /// must be boxed. This function will crash if provided something that is not
  /// a memory reference type.
  /// Array entities are boxed with a shape and character with their length.
  mlir::Value createBox(mlir::Location loc, const fir::ExtendedValue &exv);

  /// Create constant i1 with value 1. if \p b is true or 0. otherwise
  mlir::Value createBool(mlir::Location loc, bool b) {
    return createIntegerConstant(loc, getIntegerType(1), b ? 1 : 0);
  }

  //===--------------------------------------------------------------------===//
  // If-Then-Else generation helper
  //===--------------------------------------------------------------------===//

  /// Helper class to create if-then-else in a structured way:
  /// Usage: genIfOp().genThen([&](){...}).genElse([&](){...}).end();
  /// Alternatively, getResults() can be used instead of end() to end the ifOp
  /// and get the ifOp results.
  class IfBuilder {
  public:
    IfBuilder(fir::IfOp ifOp, FirOpBuilder &builder)
        : ifOp{ifOp}, builder{builder} {}
    template <typename CC>
    IfBuilder &genThen(CC func) {
      builder.setInsertionPointToStart(&ifOp.thenRegion().front());
      func();
      return *this;
    }
    template <typename CC>
    IfBuilder &genElse(CC func) {
      assert(!ifOp.elseRegion().empty() && "must have else region");
      builder.setInsertionPointToStart(&ifOp.elseRegion().front());
      func();
      return *this;
    }
    void end() { builder.setInsertionPointAfter(ifOp); }

    /// End the IfOp and return the results if any.
    mlir::Operation::result_range getResults() {
      end();
      return ifOp.getResults();
    }

    fir::IfOp &getIfOp() { return ifOp; };

  private:
    fir::IfOp ifOp;
    FirOpBuilder &builder;
  };

  /// Create an IfOp and returns an IfBuilder that can generate the else/then
  /// bodies.
  IfBuilder genIfOp(mlir::Location loc, mlir::TypeRange results,
                    mlir::Value cdt, bool withElseRegion) {
    auto op = create<fir::IfOp>(loc, results, cdt, withElseRegion);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with no "else" region, and no result values.
  /// Usage: genIfThen(loc, cdt).genThen(lambda).end();
  IfBuilder genIfThen(mlir::Location loc, mlir::Value cdt) {
    auto op = create<fir::IfOp>(loc, llvm::None, cdt, false);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with an "else" region, and no result values.
  /// Usage: genIfThenElse(loc, cdt).genThen(lambda).genElse(lambda).end();
  IfBuilder genIfThenElse(mlir::Location loc, mlir::Value cdt) {
    auto op = create<fir::IfOp>(loc, llvm::None, cdt, true);
    return IfBuilder(op, *this);
  }

  /// Generate code testing \p addr is not a null address.
  mlir::Value genIsNotNull(mlir::Location loc, mlir::Value addr);

  /// Generate code testing \p addr is a null address.
  mlir::Value genIsNull(mlir::Location loc, mlir::Value addr);

private:
  const KindMapping &kindMap;
};

} // namespace fir

namespace fir::factory {

//===----------------------------------------------------------------------===//
// ExtendedValue inquiry helpers
//===----------------------------------------------------------------------===//

/// Read or get character length from \p box that must contain a character
/// entity. If the length value is contained in the ExtendedValue, this will
/// not generate any code, otherwise this will generate a read of the fir.box
/// describing the entity.
mlir::Value readCharLen(fir::FirOpBuilder &builder, mlir::Location loc,
                        const fir::ExtendedValue &box);

/// Read or get the extent in dimension \p dim of the array described by \p box.
mlir::Value readExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                       const fir::ExtendedValue &box, unsigned dim);

/// Read extents from \p box.
llvm::SmallVector<mlir::Value> readExtents(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           const fir::BoxValue &box);

/// Get extents from \p box. For fir::BoxValue and
/// fir::MutableBoxValue, this will generate code to read the extents.
llvm::SmallVector<mlir::Value> getExtents(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          const fir::ExtendedValue &box);

//===----------------------------------------------------------------------===//
// String literal helper helpers
//===----------------------------------------------------------------------===//

/// Create a !fir.char<1> string literal global and returns a
/// fir::CharBoxValue with its address and length.
fir::ExtendedValue createStringLiteral(fir::FirOpBuilder &, mlir::Location,
                                       llvm::StringRef string);

/// Unique a compiler generated identifier. A short prefix should be provided
/// to hint at the origin of the identifier.
std::string uniqueCGIdent(llvm::StringRef prefix, llvm::StringRef name);

/// Lowers the extents from the sequence type to Values.
/// Any unknown extents are lowered to undefined values.
llvm::SmallVector<mlir::Value> createExtents(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             fir::SequenceType seqTy);

//===----------------------------------------------------------------------===//
// Location helpers
//===----------------------------------------------------------------------===//

/// Generate a string literal containing the file name and return its address
mlir::Value locationToFilename(fir::FirOpBuilder &, mlir::Location);

/// Generate a constant of the given type with the location line number
mlir::Value locationToLineNo(fir::FirOpBuilder &, mlir::Location, mlir::Type);

/// Builds and returns the type of a ragged array header used to cache mask
/// evaluations. RaggedArrayHeader is defined in
/// flang/include/flang/Runtime/ragged.h.
mlir::TupleType getRaggedArrayHeaderType(fir::FirOpBuilder &builder);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
