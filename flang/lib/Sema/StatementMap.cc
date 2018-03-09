
#include "flang/Sema/StatementMap.h"

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define FAIL(msg) \
  do { \
    std::cerr << "FATAL " << __FILE__ << ":" << __LINE__ << ":\n   " << msg \
              << "\n"; \
    exit(1); \
  } while (0)
#define INTERNAL_ERROR FAIL("Internal Error")

namespace Fortran::semantics {

StatementMap::Entry &StatementMap::Get(Index index) {
  if (!(( First() <= index) && (index <= Last()))) {
    FAIL("Illegal Stmt index " << index << " (expect " 
         << First() << " .." << Last() << ")");
    exit(1);
  }
  return entries_[index - 1];
}

const StatementMap::Entry &StatementMap::Get(Index index) const {
  if (!((First() <= index) && (index <= Last()))) {
    FAIL("Illegal Stmt index " << index << " (expect " 
         << First() << ".." << Last() << ")");
    exit(1);
  }
  return entries_[index - 1];
}

StatementMap::Index StatementMap::Add(StmtClass sclass, int label) {
  Entry self;
  self.sclass = sclass;
  self.group = StmtClassToGroup(sclass);
  self.label = label;

  self.parent = None;
  self.prev_in_body = None;
  self.next_in_body = None;
  self.prev_in_construct = None;
  self.next_in_construct = None;

  Index self_index = Last() + 1;

  if (Size() == 0) {
    // Special case of the first entry.
    entries_.push_back(self);
    return self_index;
  }

  Index prev_index = self_index - 1;
  auto prev_group = Get(prev_index).group;
  if (prev_group == StmtGroup::End) {
    // When inserting after a closed construct, do as if
    // that was after a single statement
    prev_index = StartOfConstruct(prev_index);
    prev_group = StmtGroup::Single;
  }

  Entry &prev = Get(prev_index);

  if (self.group == StmtGroup::Single || self.group == StmtGroup::Start) {
    if (prev_group == StmtGroup::Start || prev_group == StmtGroup::Part) {
      // Insert 'self' as first statement in body of 'prev;
      self.parent = prev_index;
    } else if (prev_group == StmtGroup::Single) {
      // Insert 'self' after 'prev' (in the same body)
      prev.next_in_body = self_index;
      self.prev_in_body = prev_index;
      self.parent = prev.parent;
    } else {
      INTERNAL_ERROR;
    }
  } else if (self.group == StmtGroup::Part || self.group == StmtGroup::End) {
    if (prev_group == StmtGroup::Start || prev_group == StmtGroup::Part) {
      // Close the empty body of 'prev'
      assert(prev.next_in_construct == None);
      prev.next_in_construct = self_index;
      self.prev_in_construct = prev_index;
    } else if (prev_group == StmtGroup::Single) {
      // Close a non-empty body ending with 'prev'
      assert(prev.next_in_body == None);
      if (prev.parent == None) {
        DumpFlat(std::cerr);
      }
      assert(prev.parent != None);
      Get(prev.parent).next_in_construct = self_index;
      self.prev_in_construct = prev.parent;
    } else {
      INTERNAL_ERROR;
    }
  }

  // TODO: Beware of the reallocation cost of calling push_back() for each
  // statement.
  // const int chunksize = 128;
  // entries_.reserve( (entries_.size()/chunksize+1)*chunksize );

  entries_.push_back(self);


  if ( self.group == StmtGroup::Part || self.group == StmtGroup::End ) {
    assert( self.prev_in_construct != None) ;
    assert( NextInConstruct(self.prev_in_construct) == self_index) ;
    assert( PrevInConstruct(self_index) == self.prev_in_construct) ;
    AssertNextInConstruct(self.prev_in_construct);
  }

  return self_index;
}

//
// 'prev_index' shall be a Start or Part statement.
//
// Assert that the next statement in the construct is of an expected class.
//
//
void StatementMap::AssertNextInConstruct(Index prev_index) const {

  if (prev_index == None) return;  
  auto &prev = Get(prev_index);
  if (prev.next_in_construct == None) return;
  auto &next = Get(prev.next_in_construct);


#define ORDER(A, B) (prev.sclass == StmtClass::A && next.sclass == StmtClass::B)

  if (  // If THEN ... [ELSEIF]* ... [ELSE] ... Endif
      ORDER(IfThen, ElseIf) || ORDER(IfThen, Else) || ORDER(IfThen, EndIf) ||
      ORDER(ElseIf, ElseIf) || ORDER(ElseIf, Else) || ORDER(ElseIf, EndIf) ||
      ORDER(Else, EndIf) ||
      // PROGRAM ... [CONTAINS] ... END
      ORDER(Program, Contains) || ORDER(Program, EndProgram) ||
      ORDER(Contains, EndProgram) ||
      // MODULE
      ORDER(Module, Contains) || ORDER(Module, EndModule) ||
      ORDER(Contains, EndModule) ||
      // SUBROUTINE
      ORDER(Subroutine, Contains) || ORDER(Subroutine, EndSubroutine) ||
      ORDER(Contains, EndSubroutine) ||
      // FUNCTION
      ORDER(Function, Contains) || ORDER(Function, EndFunction) ||
      ORDER(Contains, EndFunction) ||
      // IF ..
      ORDER(If, DummyEndIf) ||
      // FORALL ..
      ORDER(Forall, DummyEndForall) || ORDER(ForallConstruct, EndForall) ||
      // WHERE ..
      ORDER(Where, DummyEndWhere) || ORDER(WhereConstruct, MaskedElsewhere) ||
      ORDER(WhereConstruct, EndWhere) || ORDER(ElseWhere, EndWhere) ||
      ORDER(MaskedElsewhere, EndWhere) ||
      ORDER(MaskedElsewhere, MaskedElsewhere) ||
      // ASSOCIATE
      ORDER(Associate, EndAssociate) ||
      // BLOCK
      ORDER(Block, EndBlock) ||
      // CHANGE TEAM
      ORDER(ChangeTeam, EndChangeTeam) ||
      // CRITICAL
      ORDER(Critical, EndCritical) ||
      // TYPE
      ORDER(DerivedType, EndType) ||
      // ENUM
      ORDER(EnumDef, EndEnum) ||
      // INTERFACE
      ORDER(Interface, EndInterface) ||
      // DO var=...
      ORDER(NonLabelDo, EndDo) ||
      // DO WHILE ...
      ORDER(NonLabelDoWhile, EndDo) ||
      // DO CONCURRENT
      ORDER(NonLabelDoConcurrent, EndDo) ||
      // DO <label> var=....
      ORDER(LabelDo, EndDo) ||
      ORDER(LabelDo, DummyEndDo) ||
      // DO <label> WHILE ...
      ORDER(LabelDoWhile, EndDo) ||
      ORDER(LabelDoWhile, DummyEndDo) ||
      // DO <label> CONCURRENT ...
      ORDER(LabelDoConcurrent, EndDo) ||
      ORDER(LabelDoConcurrent, DummyEndDo) ||
     // SELECT CASE
      ORDER(SelectCase, EndSelect) || ORDER(SelectCase, Case) ||
      ORDER(Case, Case) || ORDER(Case, EndSelect) ||
      ORDER(SelectCase, CaseDefault) || ORDER(CaseDefault, Case) ||
      ORDER(Case, CaseDefault) || ORDER(CaseDefault, CaseDefault) ||
      ORDER(CaseDefault, EndSelect) ||

      // SELECT RANK
      ORDER(SelectRank, EndSelect) || ORDER(SelectRank, SelectRankCase) ||
      ORDER(SelectRank, SelectRankStar) || ORDER(SelectRank, SelectRankDefault) ||
      
      ORDER(SelectRankCase, SelectRankCase) || ORDER(SelectRankCase, SelectRankStar) ||
      ORDER(SelectRankCase, SelectRankDefault) || ORDER(SelectRankCase, EndSelect) ||

      ORDER(SelectRankStar, SelectRankCase) || ORDER(SelectRankStar, SelectRankStar) ||
      ORDER(SelectRankStar, SelectRankDefault) || ORDER(SelectRankStar, EndSelect) ||

      ORDER(SelectRankDefault, SelectRankCase) || ORDER(SelectRankDefault, SelectRankStar) ||
      ORDER(SelectRankDefault, SelectRankDefault) ||ORDER(SelectRankDefault, EndSelect) ||

      // SELECT TYPE
      ORDER(SelectType, EndSelect) || ORDER(SelectType, TypeGuard) || 
      ORDER(SelectType, ClassGuard) ||  ORDER(SelectType, ClassDefault) || 
      
      ORDER(TypeGuard, EndSelect) || ORDER(TypeGuard, TypeGuard) || 
      ORDER(TypeGuard, ClassGuard) ||  ORDER(TypeGuard, ClassDefault) || 

      ORDER(ClassGuard, EndSelect) || ORDER(ClassGuard, TypeGuard) || 
      ORDER(ClassGuard, ClassGuard) ||  ORDER(ClassGuard, ClassDefault) || 

      ORDER(ClassDefault, EndSelect) || ORDER(ClassDefault, TypeGuard) || 
      ORDER(ClassDefault, ClassGuard) ||  ORDER(ClassDefault, ClassDefault) || 
      
      // STRUCTURE (PGI)
      ORDER(Structure, EndStructure)) {
    // That looks good
  } else {
    // Todo: print location of both statement
    FAIL("Found " << StmtClassName(next.sclass) << " after "
                  << StmtClassName(prev.sclass));
  }
#undef ORDER
}

// Provide the number of statements in the map.
// Reminder: the proper indices are 1..Size()
int StatementMap::Size() const { return entries_.size(); }

//
// Specialize the StmtClass of an existing statement.
//
//
void StatementMap::Specialize( Index index, StmtClass oldc, StmtClass newc) {
  Entry &self = Get(index);
  
  if (self.sclass != oldc) {
    INTERNAL_ERROR;
  }
  if (self.group != StmtClassToGroup(oldc)) {
    INTERNAL_ERROR;
  }
  if (self.group != StmtClassToGroup(newc)) {
    INTERNAL_ERROR;
  }

  // Only a few specializations are allowed
  if ((oldc == StmtClass::Case && newc == StmtClass::CaseDefault) ||
      (oldc == StmtClass::TypeGuard && newc == StmtClass::ClassDefault) ||
      (oldc == StmtClass::TypeGuard && newc == StmtClass::ClassGuard) ||
      (oldc == StmtClass::SelectRankCase && newc == StmtClass::SelectRankStar) ||
      (oldc == StmtClass::SelectRankCase && newc == StmtClass::SelectRankDefault) ||
      (oldc == StmtClass::Access && newc == StmtClass::PublicAccess) ||
      (oldc == StmtClass::Access && newc == StmtClass::PrivateAccess) || 
      (oldc == StmtClass::LabelDo && newc == StmtClass::LabelDoWhile)|| 
      (oldc == StmtClass::LabelDo && newc == StmtClass::LabelDoConcurrent) || 
      (oldc == StmtClass::NonLabelDo && newc == StmtClass::NonLabelDoWhile) || 
      (oldc == StmtClass::NonLabelDo && newc == StmtClass::NonLabelDoConcurrent)) {
    self.sclass = newc;

    // Paranoid mode! More consistency checks
    AssertNextInConstruct(self.prev_in_construct);
    AssertNextInConstruct(index);

  } else {
    INTERNAL_ERROR;
  }
}

void StatementMap::DumpFlat(std::ostream &out, bool verbose) const {
  for (Index i = First() ; i <= Last(); i++) {
    out << std::setw(4) << std::right << i << ": ";
    DumpStmt(out, i, verbose);
  }
}

void StatementMap::DumpBody(
    std::ostream &out, Index index, bool rec, int level, bool verbose) const {
  for (Index i = FirstInBody(index); i != None; i = Next(i)) {
    Dump(out, i, rec, level, verbose);
  }
}

void StatementMap::DumpStmt(
    std::ostream &out, Index index, bool verbose) const {
  const auto &stmt = Get(index);

  switch (stmt.group) {
  case StmtGroup::Single: out << "| "; break;
  case StmtGroup::Start: out << "> "; break;
  case StmtGroup::Part: out << "+ "; break;
  case StmtGroup::End: out << "< "; break;
  }

  out << StmtClassName(stmt.sclass);

  if (verbose) {
    // Enable this while debugging the Map
    if (stmt.parent) out << " parent=" << stmt.parent;
    if (stmt.prev_in_body) out << " prev_in_body=" << stmt.prev_in_body;
    if (stmt.next_in_body) out << " next_in_body=" << stmt.next_in_body;
    if (stmt.prev_in_construct)
      out << " prev_in_construct=" << stmt.prev_in_construct;
    if (stmt.next_in_construct)
      out << " next_in_construct=" << stmt.next_in_construct;
  }
  out << "\n";
}

void StatementMap::Dump(
    std::ostream &out, Index index, bool rec, int level, bool verbose) const {
  while (index != None) {
    const auto &stmt = Get(index);
    out << std::setw(4) << std::right << index << ": ";

    for (int i = 0; i < level; i++)
      out << "|  ";

    DumpStmt(out, index, verbose);

    if (rec &&
        (stmt.group == StmtGroup::Start || stmt.group == StmtGroup::Part)) {
      DumpBody(out, index, rec, level + 1, verbose);
      index = NextInConstruct(index);
    } else {
      index = None;
    }
  }
}

void StatementMap::CheckIndex(Index index) const {
  assert(index > 0 && index < int(entries_.size()));
}

StatementMap::Index StatementMap::Next(Index index) const {
  auto &e = Get(index);
  if (e.group == StmtGroup::Single) {
    return e.next_in_body;
  } else if (e.group == StmtGroup::Start) {
    return e.next_in_body;
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

StatementMap::Index StatementMap::Prev(Index index) const {
  auto &e = Get(index);
  if (e.group == StmtGroup::Single) {
    return e.prev_in_body;
  } else if (e.group == StmtGroup::Start) {
    return e.prev_in_body;
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

bool StatementMap::EmptyBody(Index index) const {
  auto &e = Get(index);
  if (e.group == StmtGroup::Start) {
    if (e.next_in_construct == None) {
      // The body construction is not finished yet
      INTERNAL_ERROR;
      return None;
    } else {
      return (e.next_in_construct == index + 1);
    }
  } else if (e.group == StmtGroup::Part) {
    if (e.next_in_construct == None) {
      // The body construction is not finished yet
      INTERNAL_ERROR;
      return None;
    }
    return (e.next_in_construct == index + 1);
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

StatementMap::Index StatementMap::FirstInBody(StatementMap::Index index) const {
  auto &e = Get(index);
  if (e.group == StmtGroup::Start) {
    if (e.next_in_construct == index + 1)
      return None;  // an empty body
    else
      return index + 1;
  } else if (e.group == StmtGroup::Part) {
    if (e.next_in_construct == index + 1)
      return None;  // an empty body
    else
      return index + 1;
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

StatementMap::Index StatementMap::LastInBody(StatementMap::Index index) const {
  CheckIndex(index);
  auto &stmt = Get(index);
  if (stmt.group == StmtGroup::Start) {
    if (stmt.next_in_construct == index + 1) {
      return None;  //  empty body
    } else if (stmt.next_in_construct == None) {
      // We are probably querying an incomplete construct
      INTERNAL_ERROR;
      return None;
    } else {
      return LastInPreviousBody(stmt.next_in_construct);
    }
  } else if (stmt.group == StmtGroup::Part) {
    if (stmt.next_in_construct == index + 1) {
      return None;  // empty body
    } else if (stmt.next_in_construct == None) {
      // We are probably querying an incomplete construct
      INTERNAL_ERROR;
      return None;
    } else {
      return LastInPreviousBody(stmt.next_in_construct);
    }
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

// Functionnally equivalent to LastInBody(PreviousPartOfConstruct(index))
StatementMap::Index StatementMap::LastInPreviousBody(
    StatementMap::Index index) const {
  auto &stmt = Get(index);
  if (stmt.group == StmtGroup::Part || stmt.group == StmtGroup::End) {
    auto &prev = Get(index - 1);
    if (prev.group == StmtGroup::Single) {
      return index - 1;
    } else if (prev.group == StmtGroup::End) {
      return StartOfConstruct(index - 1);
    } else if (prev.group == StmtGroup::Start) {
      return None;
    } else if (prev.group == StmtGroup::Part) {
      return None;
    } else {
      INTERNAL_ERROR;
      return None;
    }
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

StatementMap::Index StatementMap::FindPrevInConstruct(
    StatementMap::Index index, StmtClass sclass) const {
  while (true) {
    index = PrevInConstruct(index);
    if (index == None) return None;
    if (Get(index).sclass == sclass) return index;
  }
}

StatementMap::Index StatementMap::FindNextInConstruct(
    StatementMap::Index index, StmtClass sclass) const {
  while (true) {
    index = NextInConstruct(index);
    if (index == None) return None;
    if (Get(index).sclass == sclass) return index;
  }
}

StatementMap::Index StatementMap::PrevInConstruct(
    StatementMap::Index index) const {
  auto &stmt = Get(index);
  if (stmt.group == StmtGroup::Start) {
    return None;
  } else if (stmt.group == StmtGroup::Part) {
    return stmt.prev_in_construct;
  } else if (stmt.group == StmtGroup::End) {
    return stmt.prev_in_construct;
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

StatementMap::Index StatementMap::NextInConstruct(
    StatementMap::Index index) const {
  auto &stmt = Get(index);
  if (stmt.group == StmtGroup::Start) {
    return stmt.next_in_construct;
  } else if (stmt.group == StmtGroup::Part) {
    return stmt.next_in_construct;
  } else if (stmt.group == StmtGroup::End) {
    return None;
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

// Find the Start component of a construct specified by
// any of its component.
StatementMap::Index StatementMap::StartOfConstruct(
    StatementMap::Index index) const {
  auto &stmt = Get(index);
  if (stmt.group == StmtGroup::Start) {
    return index;
  } else if (stmt.group == StmtGroup::Part || stmt.group == StmtGroup::End) {
    return StartOfConstruct(stmt.prev_in_construct);
  } else {
    INTERNAL_ERROR;
    return None;
  }
}

// Find the End component of a construct specified by
// any of its component.
StatementMap::Index StatementMap::EndOfConstruct(
    StatementMap::Index index) const {
  while (true) {
    auto &stmt = Get(index);
    if (stmt.group == StmtGroup::Start || stmt.group == StmtGroup::Part) {
      index = stmt.next_in_construct;
    } else if (stmt.group == StmtGroup::End) {
      return index;
    } else {
      INTERNAL_ERROR;
      return None;
    }
  }
}

// Find the last component of a construct.
//
// Unlike EndOfConstruct(), this visitor does not require the presence of
// any End construct and so can be safely used during the creation of
// the statement map
//
StatementMap::Index StatementMap::LastOfConstruct(
    StatementMap::Index index) const {
  while (true) {
    auto &stmt = Get(index);
    if (stmt.group == StmtGroup::Start || stmt.group == StmtGroup::Part) {
      if (stmt.next_in_construct == None)
        return index;  // end of an incomplete construct
      index = stmt.next_in_construct;
    } else if (stmt.group == StmtGroup::End) {
      return index;  // end of an complete construct
    } else {
      INTERNAL_ERROR;
      return None;
    }
  }
}

// Visit all the statements that compose a construct.
//
// 'stmt' shall be a construct component (so in group Start, Part or End)
void StatementMap::VisitConstruct(
    StatementMap::Index stmt, std::function<bool(Index)> action) const {
  Index start = StartOfConstruct(stmt);
  for (Index at = start; at != None; at = NextInConstruct(at)) {
    if (!action(at)) break;
  }
}

// Visit all the statements that compose a construct in reverse order.
//
// 'stmt' shall be a construct component (so in group Start, Part or End)
//
void StatementMap::VisitConstructRev(
    Index stmt, std::function<bool(Index)> action) const {
  // Reminder: Using LastOfConstruct instead of EndOfConstruct
  //           because the visitor shall usable while constructing
  //           the statement map.
  std::cerr << "#" << stmt;
  Index start = LastOfConstruct(stmt);
  std::cerr << "%" << start;
  for (Index at = start; at != None; at = PrevInConstruct(at)) {
    std::cerr << "@" << at;
    if (!action(at)) break;
  }
}

// // Set the label required to close a LabelDo
// void StatementMap::SetLabelDoLabel(Index do_stmt, int label) {
//   assert(Get(do_stmt).sclass == StmtClass::LabelDo);
//   assert(label != 0);
//   label_do_map_[do_stmt] = label;
// }

// int StatementMap::GetLabelDoLabel(Index do_stmt) const {
//   auto it = label_do_map_.find(do_stmt);
//   if (it != label_do_map_.end()) {
//     return it->second;
//   }
//   // In theory that should never happen
//   INTERNAL_ERROR;
//   return 0;
// }

}  // namespace Fortran::semantics
