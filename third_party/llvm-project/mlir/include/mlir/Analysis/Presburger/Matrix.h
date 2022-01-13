//===- Matrix.h - MLIR Matrix Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple 2D matrix class that supports reading, writing, resizing,
// swapping rows, and swapping columns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIX_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIX_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace mlir {

/// This is a class to represent a resizable matrix.
///
/// More columns and rows can be reserved than are currently used. The data is
/// stored as a single 1D array, viewed as a 2D matrix with nRows rows and
/// nReservedColumns columns, stored in row major form. Thus the element at
/// (i, j) is stored at data[i*nReservedColumns + j]. The reserved but unused
/// columns always have all zero values. The reserved rows are just reserved
/// space in the underlying SmallVector's capacity.
class Matrix {
public:
  Matrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// The number of reserved rows and columns will be at least the number
  /// specified, and will always be sufficient to accomodate the number of rows
  /// and columns specified.
  ///
  /// Initially, the entries are initialized to ero.
  Matrix(unsigned rows, unsigned columns, unsigned reservedRows = 0,
         unsigned reservedColumns = 0);

  /// Return the identity matrix of the specified dimension.
  static Matrix identity(unsigned dimension);

  /// Access the element at the specified row and column.
  int64_t &at(unsigned row, unsigned column);
  int64_t at(unsigned row, unsigned column) const;
  int64_t &operator()(unsigned row, unsigned column);
  int64_t operator()(unsigned row, unsigned column) const;

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const;

  unsigned getNumColumns() const;

  /// Return the maximum number of rows/columns that can be added without
  /// incurring a reallocation.
  unsigned getNumReservedRows() const;
  unsigned getNumReservedColumns() const;

  /// Reserve enough space to resize to the specified number of rows without
  /// reallocations.
  void reserveRows(unsigned rows);

  /// Get an ArrayRef corresponding to the specified row.
  ArrayRef<int64_t> getRow(unsigned row) const;

  /// Insert columns having positions pos, pos + 1, ... pos + count - 1.
  /// Columns that were at positions 0 to pos - 1 will stay where they are;
  /// columns that were at positions pos to nColumns - 1 will be pushed to the
  /// right. pos should be at most nColumns.
  void insertColumns(unsigned pos, unsigned count);
  void insertColumn(unsigned pos);

  /// Insert rows having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// rows that were at positions pos to nColumns - 1 will be pushed to the
  /// right. pos should be at most nRows.
  void insertRows(unsigned pos, unsigned count);
  void insertRow(unsigned pos);

  /// Remove the columns having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// columns that were at positions pos + count - 1 or later will be pushed to
  /// the right. The columns to be deleted must be valid rows: pos + count - 1
  /// must be at most nColumns - 1.
  void removeColumns(unsigned pos, unsigned count);
  void removeColumn(unsigned pos);

  /// Remove the rows having positions pos, pos + 1, ... pos + count - 1.
  /// Rows that were at positions 0 to pos - 1 will stay where they are;
  /// rows that were at positions pos + count - 1 or later will be pushed to the
  /// right. The rows to be deleted must be valid rows: pos + count - 1 must be
  /// at most nRows - 1.
  void removeRows(unsigned pos, unsigned count);
  void removeRow(unsigned pos);

  void copyRow(unsigned sourceRow, unsigned targetRow);

  /// Add `scale` multiples of the source row to the target row.
  void addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale);

  /// Add `scale` multiples of the source column to the target column.
  void addToColumn(unsigned sourceColumn, unsigned targetColumn, int64_t scale);

  /// Negate the specified column.
  void negateColumn(unsigned column);

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resizeVertically(unsigned newNRows);

  /// Add an extra row at the bottom of the matrix and return its position.
  unsigned appendExtraRow();

  /// Print the matrix.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Return whether the Matrix is in a consistent state with all its
  /// invariants satisfied.
  bool hasConsistentState() const;

private:
  /// The current number of rows, columns, and reserved columns. The underlying
  /// data vector is viewed as an nRows x nReservedColumns matrix, of which the
  /// first nColumns columns are currently in use, and the remaining are
  /// reserved columns filled with zeros.
  unsigned nRows, nColumns, nReservedColumns;

  /// Stores the data. data.size() is equal to nRows * nReservedColumns.
  /// data.capacity() / nReservedColumns is the number of reserved rows.
  SmallVector<int64_t, 64> data;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
