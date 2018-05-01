// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_SEMANTICS_STATEMENTMAP_H_
#define FORTRAN_SEMANTICS_STATEMENTMAP_H_

#include "Stmt.h"
#include <functional>
#include <map>
#include <vector>

namespace Fortran::semantics {


//
// A StatementMap describes the relations between statements in a program unit
// and also hold information that are common to each statement (label, provenance,
// ...)
//
// In the parse-tree each struct representing a statement (see Stmt.def)
// is associated with an index in the statement map.
//
// Valid indices are 1 to Size().
//
// The index None=0 represents no statement.
//
// In the map, each statement belong to one of 4 groups Single,
// Start, Part, and End (see Stmt.def)
//
// Also, the statements are stored in 2 dimensions:
//  - the body dimension for Single and Start statements.
//  - the construct dimension for Start, Part and End statements
//
// In the construct dimension, a sequence of statement is always
// composed of one Start, any number of Part and one End.
//
// Only Start and Part statements can have a body and they are
// the 'parent' of each statement in that body.
//
// The mapping of Fortran statements to the 4 groups Single, Start,
// Part, and End should be obvious.
//
// For instance, consider the following piece of code
//
//   DO i=1,n
//      PRINT *, "in DO body"
//      IF (i==1) THEN
//        PRINT *, "in IF body"
//        CYCLE
//      ELSE IF (i==2)
//        PRINT *, "in ELSE IF body"
//      ELSE
//        PRINT *, "in ELSE body"
//      ENDIF
//      PRINT *, "also in DO body"
//   ENDDO
//
// The statement map Dump() for that piece of code shall look like that
//
//     10:    > NonLabelDo
//     11:    | | Print
//     12:    | > IfThen
//     13:    | | | Print
//     14:    | | | Cycle
//     15:    | + ElseIf
//     16:    | | | Print
//     17:    | + Else
//     18:    | | | Print
//     19:    | < EndIf
//     20:    | | Print
//     21:    < EndDo
//
// The '>', '+' and '>' are used to mark the Start, Part and End statements.
//
// In that dump, we see that:
//  - The DO construct is comprised of statements 10 and 21
//  - The IF construct is comprised of statements 12, 15, 17 and 19
//  - The body of statement 10 is comprised of statements 11, 12, and 20
//  - The body of statement 12 is comprised of statements 13 and 14
//  - The body of statement 15 is comprised of statement 16
//  - The body of statement 17 is comprised of statement 18
//
// For the few Fortran constructs that do not provide an explicit 'End'
// statement, a dummy entry will be provided in the map. For instance,
// the construct
//
//    IF (cond) CYCLE
//
// could dump as follow:
//
//    30:  > If
//    31:  | | Cycle
//    32:  < DummyEndIf
//
//
class StatementMap {
public:
  typedef int Index;  // should become an opaque type.
  
  static constexpr Index None = 0;
  
private:
  struct Entry {
    StmtClass sclass;
    StmtGroup group;
    
    int label;      // The label associated to the statement (1..99999) or 0
    
    // Relations to other statements.
    Index parent;
    Index prev_in_body;
    Index next_in_body;
    Index prev_in_construct;
    Index next_in_construct;
  };
  
  std::vector<Entry> entries_;

  std::map<Index, int> label_do_map_;

  
  Entry &at(Index index);
  const Entry &at(Index index) const;

public:
  // Add a statement to the map.
  Index Add(StmtClass sclass, int label);

  // Set the label required to close a LabelDo
//  void SetLabelDoLabel(Index do_stmt, int label);

//  int GetLabelDoLabel(Index do_stmt) const;

  //
  // 'prev_index' shall be a Start or Part statement.
  //
  // Assert that the next statement in the construct is of an expected class.
  //
  //
  void AssertNextInConstruct(Index prev_index) const;

  // Provide the number of statements in the map.
  // Reminder: the proper indices are 1..Size()
  int Size() const;

  // The index of the first statement added to the map
  Index First() const { return 1; }

  // The index of the last statement  added to the map
  Index Last() const { return Size(); }

  //
  // Specialize the StmtClass of an existing statement.
  //
  //
  void Specialize(Index index, StmtClass oldclass, StmtClass newclass);

  StmtClass GetClass(Index index) const { return at(index).sclass; }

  StmtGroup GetGroup(Index index) const { return at(index).group; }

  // Provide the numerical label associated to that statement or 0.
  // Be aware that labels are not necessarily unique and so cannot 
  // be used to identify a statement within the whole map.
  int GetLabel(Index index) const { return at(index).label; }

  Index GetParent(Index index) const { return at(index).parent; }


  // Dump all the statements in the map in sequence so without
  // relying on the statement 'hierarchy'. This is only useful
  // to debug the statement map 'hierarchy'.
  void DumpFlat(std::ostream &out, bool verbose = true) const;

  void DumpBody(std::ostream &out, Index index, bool rec = true, int level = 0,
                bool verbose = false) const;
  void DumpStmt(std::ostream &out, Index index, bool verbose = false) const;
  void Dump(std::ostream &out, Index index, bool rec = true, int level = 0,
            bool verbose = false) const;
  
  void CheckIndex(Index index) const;
  
  Index Next(Index index) const;
  Index Prev(Index index) const;
  
  bool EmptyBody(Index index) const;
  
  Index FirstInBody(Index index) const;
  Index LastInBody(Index index) const;
  
  // Functionally equivalent to LastInBody(PreviousPartOfConstruct(index))
  Index LastInPreviousBody(Index index) const;
  Index FindPrevInConstruct(Index index, StmtClass sclass) const;
  
  Index FindNextInConstruct(Index index, StmtClass sclass) const;

  Index PrevInConstruct(Index index) const;

  Index NextInConstruct(Index index) const;

  // Find the Start component of a construct specified by
  // any of its component.
  Index StartOfConstruct(Index index) const;

  // Find the End component of a construct specified by
  // any of its component.
  Index EndOfConstruct(Index index) const;

  // Find the last component of a construct.
  //
  // Unlike EndOfConstruct(), this visitor does not require the presence of
  // any End construct and so can be safely used during the creation of
  // the statement map
  //
  Index LastOfConstruct(Index index) const;

  // Visit all the statements that compose a construct.
  //
  // 'stmt' shall be a construct component (so in group Start, Part or End)
  void VisitConstruct(Index stmt, std::function<bool(Index)> action) const;

  // Visit all the statements that compose a construct in reverse order.
  //
  // 'stmt' shall be a construct component (so in group Start, Part or End)
  //
  void VisitConstructRev(Index stmt, std::function<bool(Index)> action) const;
};

}  // namespace Fortran::semantics

#endif  // FORTRAN_SEMANTICS_STATEMENTMAP_H_
