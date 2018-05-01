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


#include "../../lib/parser/format-specification.h"
#include "../../lib/parser/idioms.h"
#include "../../lib/parser/indirection.h"
#include "../../lib/parser/parse-tree-visitor.h"
#include "../../lib/parser/parse-tree.h"
#include "../../lib/parser/provenance.h"

#include "scope.h"
#include "StatementMap.h"
#include "ParseTreeDump.h"
#include "LabelTable.h"
#include "SemanticData.h"
#include "GetValue.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <vector>

namespace psr = Fortran::parser ;
namespace sm  = Fortran::semantics ;

using sm::StmtClass;
using sm::StmtGroup;
using sm::GetSema;
using sm::InitSema;
using sm::Scope ;
using sm::LabelTable ;
using sm::LabelTableStack ;


#define TODO  do { std::cerr << "NOT YET IMPLEMENTED " << __FILE__ << ":" << __LINE__ << "\n" ; exit(1) ; } while(0)
#define CONSUME(x) do { (void)x ; } while(0) 

#if 1
#define TRACE_CALL()  do { std::cerr << "*** call " << __PRETTY_FUNCTION__ << "\n" ; } while(0)
#else
#define TRACE_CALL()  do {} while(0)
#endif

#define TRACE(msg) do { std::cerr << msg << "\n" ; } while(0)
#define FAIL(msg)  do { std::cerr << "FATAL " << __FILE__ << ":" << __LINE__ << ":\n   " << msg << "\n" ; exit(1) ; } while(0)
#define INTERNAL_ERROR FAIL("Internal Error")

//
// Make it easy to look to env variables  (for tracing & debugging)
//
// return true iff the env var 'name' is set and its first characters is neither '0' or 'n'
// 
bool ENV(const char *name) {
  return ( getenv(name) &&  !( getenv(name)[0] == '0' || getenv(name)[0] == 'n') );
}

// A simple helper to remove 'const' without having to make the type explicit (which can be 
// annoying since combined types can be quite long in the parse-tree)
//
// For example.
//   Using std::const_cast: 
//      auto &uses = std::const_cast<std::list<Statement<Indirection<UseStmt>>>&>(const_use_list);
//   Using unconst:
//      auto &uses = unconst(const_uses);
//
template <typename T> static inline T & unconst(const T&x) {
  return const_cast<T&>(x) ;
}


static void dump(LabelTable &labels) 
{    
#if 1
  TRACE( "==== Label Table ====");
  for ( int i=1 ; i<=99999 ;i++) {
    psr::Provenance p;
    if ( labels.find(i,p) ) {
      TRACE( "  #" << i << " at " << p.offset() ) ;
    }
  }
  TRACE( "=====================");
#endif
}


using SMap = sm::StatementMap;

namespace Fortran::parser { 


class Pass1 : public LabelTableStack 
{
  
public:
  
  Pass1(const psr::CookedSource &source) :
    source_(source), 
    current_label_(-1) ,
    current_smap_(0)
  {
    system_scope_  = &Scope::systemScope;
    unit_scope_    = &Scope::globalScope;
    current_scope_ = nullptr ;
  }   

public:

  // Remark: Ideally those fields shall not be directly accessible.
  //         Make them private in a base class? 
  
  const psr::CookedSource &source_ ;
  int current_label_; // hold the value of a statement label until it get consumed (-1 means none) 
  Provenance current_label_loc_; 
  const Scope * system_scope_ ; 
  Scope * unit_scope_; 
  Scope * current_scope_ ; 
  SMap * current_smap_ ;

  std::map<SMap::Index, sm::Name> construct_name_ ;

  // Provide all label-DO statements that are still open.
  // The key is the LabelDoStmt index
  // The value is the required label.
  //
  // TODO: WARNING: That table is shared by all program, functions 
  // and subroutines in the same unit. However, the labels are not
  // shared which means that there is a risk of conflict.
  //
  // Fortunately, that table life time is quite short and it is
  // supposed to empty itself. Some assert(opened_label_do_.empty()) 
  // shall be inserted before and after switching context.
  //
  std::map<SMap::Index,int> opened_label_do_;

public:
  
  Scope * EnterScope(Scope &s) { 
    assert( &s.parent() == current_scope_ ) ;
    current_scope_ = &s ; 
    return &s;
  } 

  void LeaveScope(Scope::Kind k) {
    assert( current_scope_->kind() == k ) ; 
    current_scope_ = const_cast<Scope*>(&current_scope_->parent()) ; 
  }

  Provenance GetProvenance(const CharBlock &source) {
    return source_.GetProvenance(source.begin()).start() ;
  }

  // Trace the location and label of any x with an accessible Statement<> in its type.
  template <typename T> void TraceStatementInfo(const T &x) { 
    auto & s = GetStatementValue(x) ;
    // TODO: compilation will fail is 's' is not of type Statement<...>.
    // Do we have a type trait to detect Statement<>?  
    //  if constexpr ( s is a Statement<> ) {
    if ( s.label ) {
      TRACE("stmt: loc=" << s.provenance.offset() << " label=" << s.label ) ; 
    } else {
      TRACE("stmt: loc=" << s.provenance.offset()  ) ; 
    }
    // } else {  TRACE("stmt: none") ; } 
  } 

  void SpecializeDoStmt( SMap::Index dostmt , const std::optional<psr::LoopControl> & control) 
  {
    auto & smap = GetStatementMap(); 

    StmtClass do_class = smap.GetClass(dostmt);
    StmtClass do_while_class;
    StmtClass do_concurrent_class;

    if ( do_class == StmtClass::NonLabelDo ) {
      do_while_class = StmtClass::NonLabelDoWhile;
      do_concurrent_class = StmtClass::NonLabelDoConcurrent;
    } else if ( do_class == StmtClass::LabelDo ) {
      do_while_class = StmtClass::LabelDoWhile;
      do_concurrent_class = StmtClass::LabelDoConcurrent;
    } else {
      INTERNAL_ERROR;
      return ;
    }

    if ( control ) { 
      std::visit( 
        visitors{
         [&] (const psr::LoopBounds<psr::ScalarIntExpr> &x) {
           // keep the do_class. Do nothing
         },
         [&] (const psr::LoopControl::Concurrent &x) {
           smap.Specialize(dostmt, do_class, do_concurrent_class);
         },
         [&] (const psr::ScalarLogicalExpr&x) {
           smap.Specialize(dostmt, do_class, do_while_class);
         },
         [](const auto &x) {
           // TODO
           return semantics::DeclTypeSpec::MakeTypeStar();
         },
       },
       control->u);
    } else {
      smap.Specialize(dostmt, do_class, do_while_class);
    }
       
  }


public:

  void ClearConstructNames() {
    construct_name_.clear() ;
  }
  

  //
  // TODO: Should operate with Symbol instead of Name
  //
  std::optional<sm::Name> GetConstructName(SMap::Index stmt) {
    auto it = construct_name_.find(stmt);
    if ( it == construct_name_.end() ) {
      return std::nullopt ; 
    }
    return it->second;
  }

  // 
  void 
  SetConstructName(SMap::Index stmt, const psr::Name &name) {
    construct_name_.insert_or_assign(stmt, name.ToString() );
  }

  // 
  void 
  SetConstructName(SMap::Index stmt, const std::optional<psr::Name>& name) {
    if (name) {
      SetConstructName(stmt,*name) ;
    }
  }

  void 
  CheckStatementName( SMap::Index part_or_end, const std::optional<psr::Name> &found, bool required) {
    
    auto & smap = GetStatementMap() ;

    assert( smap.GetGroup(part_or_end) == StmtGroup::Part ||
            smap.GetGroup(part_or_end) == StmtGroup::End );
            
    SMap::Index start = smap.StartOfConstruct(part_or_end); 
    assert( smap.GetGroup(start) == StmtGroup::Start );

    

    std::optional<sm::Name> expect = GetConstructName(start);
   
    // TODO: Get the location from part_or_end
    const char * text = StmtClassText( smap.GetClass(part_or_end) ) ;
    if ( expect ) {      
      if ( found ) {
        if ( found->ToString() != *expect ) {
          FAIL("In statement #" << part_or_end 
               << ": Unexpected name '" << found->ToString() << "' in " << text
               << "' (expected '" << *expect << "') ");
        } 
      } else if ( required) {
        FAIL("In statement #" << part_or_end 
             << ": Missing name '" << *expect << "' in " << text);
      }
    } else if ( found ) {
      FAIL("In statement #" << part_or_end 
           << ": Unexpected name '" << found->ToString() << "' in " << text << "' (none expected) ");
    }
  }

public:


  SMap & GetStatementMap()
  {
    assert(current_smap_) ;
    return *current_smap_ ;
  }
  
  void OpenLabelDo(SMap::Index dostmt, int label) 
  { 
    opened_label_do_[dostmt] = label ;
  }
  
  void CloseLabelDo(SMap::Index dostmt) 
  { 
    opened_label_do_.erase(dostmt) ;
  }

  // If stmt is an opened LabelDo, then return its label.  
  // In all other cases, return 0.
  //
  // This function can be safely called even if stmt is 
  // not a LabelStmt
  // 
  int GetLabelOfOpenedLabelDo(SMap::Index stmt) 
  {         
    auto it = opened_label_do_.find(stmt);
    if ( it != opened_label_do_.end() ) 
      return it->second ;
    else
      return 0 ; 
  }

  // return true if the specified label matches any currently opened LabelDo
  bool MatchAnyOpenedLabelDo(int label) 
  {         
    if ( label > 0 ) {
      auto smap = GetStatementMap() ; 
      for (auto it : opened_label_do_ ) {
        if ( it.second == label )
          return true;
      }
    }
    return false; 
  }
  

  bool ValidEndOfLabelDo(StmtClass sclass) 
  {
    

    if ( sclass == StmtClass::Continue ) return true ; 
    if ( sclass == StmtClass::EndDo ) return true ; 


    // Add below, all statement classes that we want to allow for 
    // backward compatibility with old codes or standards.
    if (true) {
      // Most action statements should be legal 
      // TODO: did I miss something?
      if ( sclass == StmtClass::Allocate) return true ;  
      if ( sclass == StmtClass::ArithmeticIf) return true ; 
      if ( sclass == StmtClass::Assign) return true ;  
      if ( sclass == StmtClass::AssignedGoto) return true ; 
      if ( sclass == StmtClass::Assignment) return true ; 
      if ( sclass == StmtClass::Backspace) return true ;  
      if ( sclass == StmtClass::Call) return true ;  
      if ( sclass == StmtClass::Close) return true ; 
      if ( sclass == StmtClass::ComputedGoto) return true ;  
      if ( sclass == StmtClass::Cycle) return true ;  
      if ( sclass == StmtClass::Deallocate) return true ; 
      if ( sclass == StmtClass::Endfile) return true ;  
      if ( sclass == StmtClass::EventPost) return true ; 
      if ( sclass == StmtClass::EventWait) return true ;  
      if ( sclass == StmtClass::Exit) return true ;  
      if ( sclass == StmtClass::FailImage) return true ;  
      if ( sclass == StmtClass::Flush) return true ; 
      if ( sclass == StmtClass::Forall) return true ; 
      if ( sclass == StmtClass::FormTeam) return true ;  
      if ( sclass == StmtClass::Goto) return true ; 
      if ( sclass == StmtClass::Inquire) return true ;  
      if ( sclass == StmtClass::Lock) return true ; 
      if ( sclass == StmtClass::Nullify) return true ;  
      if ( sclass == StmtClass::Open) return true ; 
      if ( sclass == StmtClass::Pause) return true ; 
      if ( sclass == StmtClass::PointerAssignment) return true ;  
      if ( sclass == StmtClass::Print ) return true ; 
      if ( sclass == StmtClass::Print) return true ; 
      if ( sclass == StmtClass::Read) return true ;  
      if ( sclass == StmtClass::Redimension) return true ;  
      if ( sclass == StmtClass::Return) return true ;  
      if ( sclass == StmtClass::Rewind) return true ; 
      if ( sclass == StmtClass::Stop) return true ;  
      if ( sclass == StmtClass::SyncAll) return true ; 
      if ( sclass == StmtClass::SyncImages) return true ;  
      if ( sclass == StmtClass::SyncMemory) return true ; 
      if ( sclass == StmtClass::SyncTeam) return true ;  
      if ( sclass == StmtClass::Unlock) return true ;  
      if ( sclass == StmtClass::Wait) return true ; 
      if ( sclass == StmtClass::Where) return true ;  
      if ( sclass == StmtClass::Write) return true ; 

      // Is the following legal? Maybe
      if ( sclass == StmtClass::Format) return true ;  // TODO: is that legal?
      //if ( sclass == StmtClass::Data) return true ;  // TODO: is that legal?
      //if ( sclass == StmtClass::Entry) return true ; // TODO: is that legal?

      // TODO Is there anything special to do here for Cycle and Exit? Probably not.
      
      // I do not know how standard that is but GFortran 
      // is accepting non-construct IF as in
      //
      //    DO 666 i=1,10
      //    666 IF (i>4) PRINT *,i 
      //
      // The InitStmt function below was designed to 
      // also handle that case. 
      //
      if ( sclass == StmtClass::If ) return true ; 

    }

    return false ; 
  }


  void CheckNoOpenedLabelDo( SMap::Index stmt, int label ) {
      if ( MatchAnyOpenedLabelDo(label) ) {
        FAIL("Statement with label " << label << " is not properly"
             " nested to close all corresponding label DO statement");
      }
  }
  
  // Check if adding a statement of class 'sclass' with the 
  // specified 'label' can legally close some opened DoLabel.
  // 
  // Note: This is also the place where LabelDo terminated by
  // a Enddo are marked as closed.
  //
  // TODO: add a Provenance argument for the stmt 
  //
  void CheckValidityOfEndingLabelDo(StmtClass sclass, int stmt_label ) 
  { 
    auto & smap = GetStatementMap() ;
    if ( smap.Size() == 0 ) 
      return ;
    
    StmtGroup sgroup = StmtClassToGroup(sclass) ;
    SMap::Index last = smap.Last(); 
    SMap::Index label_do = SMap::None;
    if ( smap.GetGroup(last) == StmtGroup::Single ) {
      label_do = smap.GetParent(last) ;
    } else if ( smap.GetGroup(last) == StmtGroup::End ) {
      label_do = smap.GetParent(smap.StartOfConstruct(last)) ;
    } 
    if ( label_do != SMap::None ) {
      if ( smap.GetClass(label_do) == StmtClass::LabelDo ) {
        if (  GetLabelOfOpenedLabelDo(label_do) == stmt_label ) {
          if ( ! ValidEndOfLabelDo(sclass) ) {
            FAIL("Statement cannot end a DO label");
          } else if ( sclass == StmtClass::EndDo ) {
            CloseLabelDo(label_do);
          }
        } else if ( sclass == StmtClass::EndDo ) {
          FAIL("ENDDO label does not match previous DO-label");
        } else if ( sgroup==StmtGroup::Part || sgroup==StmtGroup::End ) {
          FAIL("Unterminated DO-label statement");
        }
      }
    }
  }

  void CloseLabelDoLoopsWithStmtLabel(SMap::Index closing_stmt) 
  {
    auto & smap = GetStatementMap() ;
    auto sclass = smap.GetClass(closing_stmt);
    auto sgroup = smap.GetGroup(closing_stmt);
    
    // Ending a LabelDo using a construct is handled when 
    // the 'end' of that construct is inserted so not now.
    if ( sgroup == StmtGroup::Start )
      return ;

    // LabelDo cannot be closed by a label on a statement part
    if ( sgroup == StmtGroup::Part )
      return ;

    if ( sgroup == StmtGroup::End && sclass != StmtClass::EndDo ) {
      // Try to close using the label on construct that was just ended.
      // Note: this is usually not legal except in a few rare cases.
      //       see ValidEndOfLabelDo() for more details.
      closing_stmt = smap.StartOfConstruct(closing_stmt);
    }
    
    int closing_label = smap.GetLabel(closing_stmt)  ;
    if ( closing_label == 0 ) 
      return; 
      
    SMap::Index parent ;

    if ( sclass == StmtClass::EndDo ) {
      // Skip the loop that we just closed by adding an
      // ENDDO into the map.
      parent = smap.GetParent(smap.StartOfConstruct(closing_stmt)) ;
    } else {
      parent = smap.GetParent(closing_stmt) ;          
    }

    // Insert a DummyEndDo for each surrounding LabelDo that
    // matches the label of the added statement or construct.
    //
    // By construction the LabelDo loops should be perfectly nested
    // except if some directives are inserted (e.g. OpenMP parallel,
    // OpenACC loop, ...).  Though, we have to wonder if loop directives 
    // shall be allowed to break a loop nest sharing the same end-label.
    //
    while (  parent != SMap::None 
             && GetLabelOfOpenedLabelDo(parent) == closing_label
             ) 
      {              
        auto name = GetConstructName(parent);
        if (name) {
          //
          // DummyEndDo cannot be used to close a named LabelDo.
          // For instance, the following is not legal:
          //
          //   foobar: DO 666 i=1,n
          //   ...
          //   666 CONTINUE
          //
          FAIL("Statement #" << closing_stmt 
               << " cannot be used to close named DO label #" 
               << parent) ;
        }
        smap.Add( StmtClass::DummyEndDo, 0 ) ; 
        CloseLabelDo(parent) ;
        parent = smap.GetParent(parent) ;
      }
    
    if ( MatchAnyOpenedLabelDo(closing_label) ) {
      FAIL("Statement " <<  closing_stmt << " is not properly"
           " nested to close all corresponding label DO statement");
    }
    
  }
  
  // Initialize a statement.
  
  template<typename T> 
  sm::Semantic<T> & InitStmt( const T &stmt, StmtClass sclass  )
  {
    auto & sema = InitSema(stmt);
    auto & smap = GetStatementMap() ;

    // Consume the label installed by the surrounding Statement<...>
    int stmt_label = 0 ;
    if ( current_label_ >=  0 ) { 
      stmt_label = current_label_ ;

      // Special case: The statement after the non-construct IF, 
      // FORALL and WHERE do not have an associated Statement<...>
      // so simulate an 'unset' label for those
      if ( std::is_same<T,psr::IfStmt>::value |
           std::is_same<T,psr::ForallStmt>::value |
           std::is_same<T,psr::WhereStmt>::value) {
        current_label_ = 0 ;      
      } else {
        // else mark the label as consumed
        current_label_ = -1 ;
      }

       if ( stmt_label != 0 ) {
        LabelTable & label_table = GetLabelTable() ;
        Provenance old_loc ; 
        if ( label_table.find(stmt_label, old_loc) ) {
          FAIL("Duplicate label " << stmt_label 
               << "at @" << current_label_loc_.offset() 
               << "and @" << old_loc.offset() ) ;          
        } else {
          label_table.add( stmt_label, current_label_loc_) ;
        }
      }

    } else {
      FAIL("No label to consume in " << __PRETTY_FUNCTION__ );      
    }
        
    CheckValidityOfEndingLabelDo(sclass, stmt_label) ;

    // Now, add our statement.
    int stmt_index = smap.Add( sclass, stmt_label ) ;    

    // and then close as many LabelDo as possible using the 
    // label of the statement we just added (or the label 
    // of the construct that the statement just ended).
    CloseLabelDoLoopsWithStmtLabel(stmt_index);
    
    // If the label of the added statement was supposed to 
    // close some opened LabelDo then they should now be all 
    // be closed. 
    if ( MatchAnyOpenedLabelDo(stmt_label) ) {
      FAIL("Statement with label " <<  stmt_index << " is not properly"
           " nested to close all corresponding label DO statement");
    }
    
    sema.stmt_index = stmt_index; 
    return sema;
  }

public:

  template <typename T> bool Pre(const T &x) { 
    if ( ENV("TRACE_FALLBACK")  )  
      TRACE( "*** fallback Pre(" << sm::GetTypeName(x) << ")" )  ;
    
    //  TRACE( "*** fallback " << __PRETTY_FUNCTION__  ) ; 
    return true ;
  }
  
  template <typename T> void Post(const T &) { 
  //  if ( ENV("TRACE_FALLBACK") )  
  //    TRACE( "*** fallback " << __PRETTY_FUNCTION__  ) ; 
  }
  
  // fallback for std::variant


  template <typename... A> bool Pre(const std::variant<A...> &) { 
    //std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true;
  }
  
  template <typename... A> void Post(const std::variant<A...> &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }
  
  // fallback for std::tuple

  template <typename... A> bool Pre(const std::tuple<A...> &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true;
  }

  template <typename... A> void Post(const std::tuple<A...> &) { 
    //  std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }

  // fallback for std::string

  bool Pre(const std::string &x) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true ;
  }

  void Post(const std::string &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }

  // fallback for Indirection<>

  template <typename T> bool Pre(const psr::Indirection<T> &x) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
    return true ;
  }

  template <typename T> void Post(const psr::Indirection<T> &) { 
    // std::cerr << "@@@ fallback " << __PRETTY_FUNCTION__  << "\n" ; 
  }



  //  ========== Statement<>  ===========

  template <typename T>
  bool Pre(const psr::Statement<T> &x) { 

    // Each and every label must be consumed by a statement.
    if ( current_label_ != -1 ) {
      TRACE("*** Label " << current_label_ << " (" << current_label_loc_.offset() 
            << ") was not consumed in " << __PRETTY_FUNCTION__ );
    }

    // Store the label for the next statement.
    // The value 0 indicates not label but zero shall be consumed like any other
    // label. 
    current_label_ = 0 ; 
    current_label_loc_ = GetProvenance(x.source) ;
    if ( x.label.has_value() ) {
      //
      // TODO: The parser stores the label in a std::uint64_t but does not report overflow
      //       which means that the following labels are currently accepted as valid and
      ///      we have no ways to detect them.
      //         18446744073709551617 = 2^64+1 = 1
      //         18446744073709551618 = 2^64+2 = 2
      //         ...
      //
      if ( 1 <= x.label.value() && x.label.value() <= 99999 ) {
        current_label_ = x.label.value() ; 
      } else {
        FAIL( "##### Illegal label value " << x.label.value() << " at @" << current_label_loc_.offset() ) ;
      }
    } 
    return true ; 
  }

  template <typename T>
  void Post(const psr::Statement<T> &x) { 
    // Check that the label was consumed
    // Each Statement shall be associated to a class acting as a statement 
    if ( current_label_!=-1 )  {
      TRACE("*** Label " << current_label_ << " (" << current_label_loc_.offset() 
            << ") was not consumed in " << __PRETTY_FUNCTION__ );
      current_label_=-1 ;
    }
  }

  
  // ========== ProgramUnit  ===========

  bool Pre(const ProgramUnit &x) { 
    TRACE_CALL() ;
    current_scope_ = unit_scope_;
    auto & sema = InitSema(x);

    // Install the statement map for future GetStatementMap() 
    sema.statement_map = new SMap;
    assert(!current_smap_);
    current_smap_ = sema.statement_map;
   
    ClearConstructNames() ;    

    return true ; 
  }

  void Post(const ProgramUnit &x) {
    TRACE_CALL() ;
    std::cerr << "DUMP STMT " << GetStatementMap().Size() << "\n";
    //  GetStatementMap().DumpFlat(std::cerr);
    //  std::cerr << "========================\n";
    GetStatementMap().Dump(std::cerr,1,true);
    current_smap_ = 0;

#if 0 
    // #### rewrite StmtFunctionStmt into AssignmentStmt
    // #### This is an experiment.
    // #### Currently not working
    
    const SpecificationPart & specif_part = std::get<SpecificationPart>(x.t) ;
    auto &const_decl_list = std::get< std::list<DeclarationConstruct> >(specif_part.t);   
    // Reminder: ExecutionPart = std::list<ExecutionPartConstruct>
    auto &const_exec_list = std::get< std::list<ExecutionPartConstruct> >(x.t);  

    // We are going to move elements from decl_list to exec_list so get rid of the const.
    auto &decl_list = unconst(const_decl_list);
    auto &exec_part = unconst(const_exec_part);
    
    // The goal is to remove some StmtFunctionStmt at the end of decl_list
    // and to insert them in the same order at the begining of excl_part
    //
    // for instance:
    // 
    //  - Before:
    //       ! decl_list contains 4 elements 
    //       integer,intent(in) :: i,n
    //       integer :: a,b(n),c(n)
    //       b(i) = i*10     ! StmtFunctionStmt
    //       c(i) = i*i      ! StmtFunctionStmt        
    //       ! exec_part contains 1 element
    //       print *, "hello" , b(1), c(1)
    //
    //  - After:
    //       ! decl_list contains 2 elements 
    //       integer,intent(in) :: i,n
    //       integer :: a, b(n), c(n)
    //       ! exec_part contains 3 element
    //       b(i) = i*10
    //       c(i) = i*i              
    //       print *, "hello" , b(1), c(1)
    // 

    // For the purpose of that experiment, convert all StmtFunctionStmt 
    // found at the end of decl_list.
    // The final code shall be more selective. 
    
    // A stupid alias for readability purpose
    typedef Statement<Indirection<StmtFunctionStmt>>> StmtFunctionStmt_type;

    while (true) {
      if (decl_list.empty())
        break ;
      DeclarationConstruct &last = decl_list.back() ; 
      if ( ! std::holds_alternative<StmtFunctionStmt_type>(last.v) )
        break ;
      auto & func_stmt = GetValue( std::get<StmtFunctionStmt_type>(last.v) ) ;
      
      auto & funcname = unconst(func_stmt.name());
      auto & arglist  = unconst(func_stmt.args());
      auto & rhs      = GetValue(unconst(func_stmt.expr()));               
                                 
      psr::DataRef base(...);
      std::list<SectionSubscript> sslist;
      for ( Name &index : arglist ){
        // SectionSubscript -> Expr -> Designator -> DataRef -> Name = index
        SectionSubscript ss(...);
        sslist.push_back(ss); 
      }
      psr::ArrayElement elem(base,sslist);
      ExecutionPartConstruct(ExecutableConstruct(StatementActionStmt(
      decl_list.pop_back() ;
    }
#endif

    ClearConstructNames() ;    
  }

  //  ========== MainProgram  ===========

  bool Pre(const MainProgram &x) { 
    TRACE_CALL() ;
    auto &sema = InitSema(x); 

    // const ProgramStmt * program_stmt = GetOptValue( std::get<x.PROG>(x.t) ) ;

    // Install the scope 
    sema.scope_provider = EnterScope( current_scope_->MakeScope(Scope::Kind::MainProgram) )  ; 

    // Install the label table
    sema.label_table = PushLabelTable( new LabelTable ) ;

    return true ; 
  }

  void Post(const MainProgram &x) { 
    auto &sema = GetSema(x); 
    dump(GetLabelTable()) ; 
    PopLabelTable(sema.label_table)  ;     
    LeaveScope(Scope::Kind::MainProgram)  ;     
    TRACE_CALL() ;
  }

  //  ========== FunctionSubprogram  ===========

  bool Pre(const FunctionSubprogram &x) { 
    TRACE_CALL() ;
    auto &sema = InitSema(x); 

    // const FunctionStmt    & function_stmt = GetValue(std::get<x.FUNC>(x.t)) ; 
    // const EndFunctionStmt & end_stmt      = GetValue(std::get<x.END>(x.t)) ; 

    sema.scope_provider = EnterScope( current_scope_->MakeScope(Scope::Kind::Subprogram) )  ; 

    // Install the label table
    sema.label_table = PushLabelTable( new LabelTable ) ;

    return true ; 
  }

  void Post(const FunctionSubprogram &x) { 
    TRACE_CALL() ;
    auto &sema = GetSema(x); 
    dump(GetLabelTable()) ; 
    PopLabelTable(sema.label_table)  ;     
    LeaveScope(Scope::Kind::Subprogram);
  }

  //  ========== SubroutineSubprogram  ===========

  bool Pre(const SubroutineSubprogram &x) { 
    TRACE_CALL() ;
    auto &sema = InitSema(x); 

    // const SubroutineStmt    & subroutine_stmt = GetValue(std::get<x.SUBR>(x.t)) ; 
    // const EndSubroutineStmt & end_stmt      = GetValue(std::get<x.END>(x.t)) ; 

    sema.scope_provider = EnterScope( current_scope_->MakeScope(Scope::Kind::Subprogram) )  ; 

    // Install the label table
    sema.label_table = PushLabelTable( new LabelTable ) ;

    return true ; 
  }

  void Post(const SubroutineSubprogram &x) { 
    TRACE_CALL() ;
    auto &sema = GetSema(x); 
    dump(GetLabelTable()) ; 
    PopLabelTable(sema.label_table)  ;  
    if ( ! opened_label_do_.empty() ) 
      INTERNAL_ERROR;
    LeaveScope(Scope::Kind::Subprogram)  ; 
  }


  //  ========== Module  ===========

  bool Pre(const Module &x) { 
    TRACE_CALL() ;
    auto &sema = InitSema(x); 

    // const ModuleStmt    & module_stmt = GetValue(std::get<x.MOD>(x.t)) ; 
    // const EndModuleStmt & end_stmt    = GetValue(std::get<x.END>(x.t)) ; 

    sema.scope_provider = EnterScope( current_scope_->MakeScope(Scope::Kind::Module) )  ; 

    // Install the label table
    sema.label_table = PushLabelTable( new LabelTable ) ;

    return true ; 
  }

  void Post(const Module &x) { 
    TRACE_CALL() ;
    auto &sema = GetSema(x); 
    dump(GetLabelTable()) ; 
    PopLabelTable(sema.label_table)  ;     
    LeaveScope(Scope::Kind::Module)  ;     
  }


  // =========== BlockData =========== 

  bool Pre(const BlockData &x) { 
    TRACE_CALL() ;
    return true ; 
  }

  void Post(const BlockData &x) { 
    TRACE_CALL() ;
  }


  // =========== DerivedTypeDef =========== 
  // WARNING: there is also a sm::DerivedTypeDef defined in types.h 
  bool Pre(const psr::DerivedTypeDef &x) { 
    TRACE_CALL() ;
    auto &sema = InitSema(x); 
    (void) sema ; 
    sema.scope_provider=0; // TODO: Install a scope?
    return true ; 
  }

  void Post(const psr::DerivedTypeDef &x) { 
    TRACE_CALL() ;
  }

  // =========== DerivedTypeStmt =========== 

  bool Pre(const DerivedTypeStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::DerivedType);

    auto &name = std::get<1>(x.t);
    SetConstructName(sema.stmt_index, name);

    return true ; 
  }

  void Post(const DerivedTypeStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndTypeStmt =========== 

  bool Pre(const EndTypeStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndType);

    auto &name = x.v ;
    CheckStatementName(sema.stmt_index, name, false); 

    return true ; 
  }

  void Post(const EndTypeStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ModuleStmt =========== 

  bool Pre(const ModuleStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::Module);     

    auto &name = x.v;
    SetConstructName(sema.stmt_index, name);

    return true ; 
  }

  void Post(const ModuleStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndModuleStmt =========== 

  bool Pre(const EndModuleStmt &x) { 
    TRACE_CALL() ;
     auto &sema = InitStmt(x, StmtClass::EndModule); 

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, false); 

    return true ; 
  }

  void Post(const EndModuleStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SubroutineStmt =========== 

  bool Pre(const SubroutineStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::Subroutine);
 
    auto &name = std::get<psr::Name>(x.t);
    SetConstructName(sema.stmt_index, name);

    return true ; 
  }

  void Post(const SubroutineStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndSubroutineStmt =========== 

  bool Pre(const EndSubroutineStmt &x) { 
    TRACE_CALL() ;
    auto &sema =  InitStmt(x, StmtClass::EndSubroutine); 

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, false); 
    return true ; 
  }

  // =========== FunctionStmt =========== 

  bool Pre(const FunctionStmt &x) { 
    TRACE_CALL() ;
    auto &sema =  InitStmt(x, StmtClass::Function); 

    auto &name = std::get<psr::Name>(x.t);
    SetConstructName(sema.stmt_index, name);

    return true ; 
  }

  void Post(const FunctionStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndFunctionStmt =========== 

  bool Pre(const EndFunctionStmt &x) { 
    TRACE_CALL() ;
    auto &sema =  InitStmt(x, StmtClass::EndFunction);
    
    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, false); 

    return true ; 
  }

  void Post(const EndFunctionStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== TypeDeclarationStmt =========== 

  bool Pre(const TypeDeclarationStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::TypeDeclaration); 
    return true ; 
  }

  void Post(const TypeDeclarationStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== ImplicitStmt =========== 

  bool Pre(const ImplicitStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Implicit); 
    return true ; 
  }

  void Post(const ImplicitStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== UseStmt =========== 

  bool Pre(const UseStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Use); 
    return true ; 
  }

  void Post(const UseStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== PrintStmt =========== 

  bool Pre(const PrintStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Print); 
    return true ; 
  }

  void Post(const PrintStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AssignmentStmt =========== 

  bool Pre(const AssignmentStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Assignment); 
    

    return true ; 
  }

  void Post(const AssignmentStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ProgramStmt =========== 

  bool Pre(const ProgramStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::Program); 
    
    auto &name = x.v;
    SetConstructName(sema.stmt_index, name);

       
    return true ; 
  }

  void Post(const ProgramStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndProgramStmt =========== 

  bool Pre(const EndProgramStmt &x) { 
    TRACE_CALL() ;
    auto &sema =InitStmt(x, StmtClass::EndProgram);

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, false); 

    return true ; 
  }

  void Post(const EndProgramStmt &x) {     
    TRACE_CALL() ;
  }

  
  // =========== ComponentDefStmt =========== 

  bool Pre(const ComponentDefStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::ComponentDef);
    return true ; 
  }

  void Post(const ComponentDefStmt &x) { 
    TRACE_CALL() ;
  }
    // =========== AccessStmt =========== 

  bool Pre(const AccessStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Access);
    return true ; 
  }

  void Post(const AccessStmt &x) { 
    TRACE_CALL() ;
  }

    // =========== AllocatableStmt =========== 

  bool Pre(const AllocatableStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Allocatable);
    return true ; 
  }

  void Post(const AllocatableStmt &x) { 
    TRACE_CALL() ;
  }
  
  // =========== AsynchronousStmt =========== 

  bool Pre(const AsynchronousStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Asynchronous);    
    return true ; 
  }

  void Post(const AsynchronousStmt &x) { 
    TRACE_CALL() ;
  }
  // =========== BindStmt =========== 

  bool Pre(const BindStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Bind);    
    return true ; 
  }

  void Post(const BindStmt &x) { 
    TRACE_CALL() ;
  }
  // =========== CodimensionStmt =========== 

  bool Pre(const CodimensionStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Codimension);    
    return true ; 
  }

  void Post(const CodimensionStmt &x) { 
    TRACE_CALL() ;
  }
  
  // =========== ContainsStmt =========== 

  bool Pre(const ContainsStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Contains);    
    return true ; 
  }

  void Post(const ContainsStmt &x) { 
    TRACE_CALL() ;
  }
  
  // =========== DimensionStmt =========== 

  bool Pre(const DimensionStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Dimension);
    return true ; 
  }

  void Post(const DimensionStmt &x) { 
    TRACE_CALL() ;
  }
  // =========== ExternalStmt =========== 

  bool Pre(const ExternalStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::External);
    return true ; 
  }

  void Post(const ExternalStmt &x) { 
    TRACE_CALL() ;
  }
  
  // =========== IntentStmt =========== 

  bool Pre(const IntentStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Intent);
    return true ; 
  }

  void Post(const IntentStmt &x) { 
    TRACE_CALL() ;
  }
  
  // =========== IntrinsicStmt =========== 

  bool Pre(const IntrinsicStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Intrinsic);
    return true ; 
  }

  void Post(const IntrinsicStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== NamelistStmt =========== 

  bool Pre(const NamelistStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Namelist);
    return true ; 
  }

  void Post(const NamelistStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== OptionalStmt =========== 

  bool Pre(const OptionalStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Optional);
    return true ; 
  }

  void Post(const OptionalStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== PointerStmt =========== 

  bool Pre(const PointerStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Pointer);
    return true ; 
  }

  void Post(const PointerStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ProtectedStmt =========== 

  bool Pre(const ProtectedStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Protected);
    return true ; 
  }

  void Post(const ProtectedStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SaveStmt =========== 

  bool Pre(const SaveStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Save);
    return true ; 
  }

  void Post(const SaveStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== TargetStmt =========== 

  bool Pre(const TargetStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Target);
    return true ; 
  }

  void Post(const TargetStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ValueStmt =========== 

  bool Pre(const ValueStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Value);
    return true ; 
  }

  void Post(const ValueStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== VolatileStmt =========== 

  bool Pre(const VolatileStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Volatile);
    return true ; 
  }

  void Post(const VolatileStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== CommonStmt =========== 

  bool Pre(const CommonStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Common);
    return true ; 
  }

  void Post(const CommonStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EquivalenceStmt =========== 

  bool Pre(const EquivalenceStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Equivalence);
    return true ; 
  }

  void Post(const EquivalenceStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== BasedPointerStmt =========== 

  bool Pre(const BasedPointerStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::BasedPointer);
    return true ; 
  }

  void Post(const BasedPointerStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== GenericStmt =========== 

  bool Pre(const GenericStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Generic);
    return true ; 
  }

  void Post(const GenericStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ParameterStmt =========== 

  bool Pre(const ParameterStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Parameter);
    return true ; 
  }

  void Post(const ParameterStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== BlockStmt =========== 
  
  bool Pre(const BlockStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::Block);
    
    auto &name = x.v;
    SetConstructName(sema.stmt_index, name);
    
    return true ; 
  }

  void Post(const BlockStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== EndBlockStmt =========== 
  
  bool Pre(const EndBlockStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndBlock);

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndBlockStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== ForallConstructStmt =========== 
  
  bool Pre(const ForallConstructStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::ForallConstruct);
    
    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);
    
    return true ; 
  }

  void Post(const ForallConstructStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== EndForallStmt =========== 
  
  bool Pre(const EndForallStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndForall);

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndForallStmt &x) { 
    TRACE_CALL() ;
  }



  // =========== AssociateStmt =========== 
  
  bool Pre(const AssociateStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::Associate);
    
    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);
    
    return true ; 
  }

  void Post(const AssociateStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== EndAssociateStmt =========== 
  
  bool Pre(const EndAssociateStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndAssociate);

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndAssociateStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ChangeTeamStmt =========== 
  
  bool Pre(const ChangeTeamStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::ChangeTeam);
    
    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);
    
    return true ; 
  }

  void Post(const ChangeTeamStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndChangeTeamStmt =========== 
  
  bool Pre(const EndChangeTeamStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndChangeTeam);

    auto &name = std::get<1>(x.t);
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndChangeTeamStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== CriticalStmt =========== 
  
  bool Pre(const CriticalStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::Critical);
    
    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);
    
    return true ; 
  }

  void Post(const CriticalStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndCriticalStmt =========== 
  
  bool Pre(const EndCriticalStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndCritical);

    auto &name = x.v ;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndCriticalStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== EnumeratorDef =========== 

  bool Pre(const EnumeratorDefStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EnumeratorDef);
    return true ; 
  }

  void Post(const EnumeratorDefStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EnumDefStmt =========== 
  
  bool Pre(const EnumDefStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EnumDef);
    return true ; 
  }

  void Post(const EnumDefStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndEnumStmt =========== 

  bool Pre(const EndEnumStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EndEnum);
    return true ; 
  }

  void Post(const EndEnumStmt &x) {
    TRACE_CALL() ;
  }

  // =========== InterfaceStmt =========== 

  bool Pre(const InterfaceStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Interface);
    
    // TODO: Compare the [generic-spec] with the END INTERFACE statement
    return true ; 
  }

  void Post(const InterfaceStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndInterfaceStmt =========== 

  bool Pre(const EndInterfaceStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EndInterface);
    // TODO: Compare the [generic-spec] with the INTERFACE statement
    return true ; 
  }

  void Post(const EndInterfaceStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== IfThenStmt =========== 

  bool Pre(const IfThenStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::IfThen);

    auto &name = std::get<0>(x.t) ;
    SetConstructName(sema.stmt_index, name);
  
    return true ; 
  }

  void Post(const IfThenStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ElseIfStmt =========== 

  bool Pre(const ElseIfStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::ElseIf);

    auto &name = std::get<1>(x.t);
    CheckStatementName(sema.stmt_index, name, false); 
    
    return true ; 
  }

  void Post(const ElseIfStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ElseStmt =========== 

  bool Pre(const ElseStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::Else);

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, false); 
    
    return true ; 
  }

  void Post(const ElseStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndIfStmt =========== 

  bool Pre(const EndIfStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndIf);

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndIfStmt &x) { 
    TRACE_CALL() ;
  }
 
  // =========== NonLabelDoStmt =========== 

  bool Pre(const NonLabelDoStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::NonLabelDo);

    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);

    // Specialize from StmtClass::LabelDo to StmtClass::NonLabelDoWhile or
    // StmtClass::NonLabelDoConcurrent where applicable
    SpecializeDoStmt( sema.stmt_index , std::get<std::optional<LoopControl>>(x.t) );

    return true ; 
  }

  void Post(const NonLabelDoStmt &x) { 
    TRACE_CALL() ;
 }
 
  // =========== LabelDoStmt =========== 

  bool Pre(const LabelDoStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::LabelDo);
    
    auto &sema = GetSema(x);
    // auto &smap = GetStatementMap() ;

    int end_label = int(std::get<1>(x.t)) ; 
    assert( 1 <= end_label && end_label <= 99999); // TODO: proper error
    OpenLabelDo( sema.stmt_index, end_label);

    Provenance previous_loc;
    if ( GetLabelTable().find(end_label, previous_loc) ) {
      // Early fail when the label already exists.
      // This is actually optional since a "Duplicate Label" or "unexpected END statement"
      // error will occur. 
      FAIL("Label " << end_label << " required by DO statement is already declared") ;
    }

    // And also record the construct name 
    auto &name = std::get<0>(x.t) ;
    SetConstructName(sema.stmt_index, name);

    // Specialize from StmtClass::LabelDo to StmtClass::LabelDoWhile or
    // StmtClass::LabelDoConcurrent where applicable
    SpecializeDoStmt( sema.stmt_index , std::get<std::optional<LoopControl>>(x.t) );

    return true ; 
  }

  void Post(const LabelDoStmt &x) { 
    TRACE_CALL() ;
  }
 
  // =========== EndDoStmt =========== 

  bool Pre(const EndDoStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::EndDo); 

    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndDoStmt &x) { 
    TRACE_CALL() ;
  }
 

  // =========== IfStmt =========== 

  bool Pre(const IfStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::If);
    return true ; 
  }

  void Post(const IfStmt &x) { 
    TRACE_CALL() ;
    GetStatementMap().Add( StmtClass::DummyEndIf, 0); 
  }

   // =========== SelectCaseStmt =========== 

  bool Pre(const SelectCaseStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::SelectCase);

    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);    
    
    return true ; 
  }

  void Post(const SelectCaseStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== CaseStmt =========== 

  bool Pre(const CaseStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Case);
    
    auto &sema = GetSema(x);
    auto &smap = GetStatementMap() ;
    
    // Detect and specialize CASE into CASE DEFAULT 
    if ( std::holds_alternative<psr::Default>(std::get<0>(x.t).u) )
      {
        // So this is a CASE DEFAULT

        // Let's verify that there not already one
        SMap::Index default_stmt = SMap::None; 
#if 1
        // Method 1: Manually visit the construct elements
        // smap.VisistConstruct( sema.stmt_index,
        smap.VisitConstructRev( sema.stmt_index,
                              [&]( SMap::Index at ) -> bool 
                              {
                                if ( smap.GetClass(at) == StmtClass::CaseDefault ) {
                                  default_stmt = at ;
                                  return false;
                                }
                                return true; 
                              } );
#else     
        // Method 2: Search for an previous StmtClass::CaseDefault
        default_stmt = smap.FindPrevInConstruct(sema.stmt_index,
                                                StmtClass::CaseDefault);
#endif
        if (default_stmt != SMap::None)
          {
            FAIL(" Duplicate CASE DEFAULT #" << default_stmt << " and #"
                 << sema.stmt_index) ; 
          }

        // 
        smap.Specialize(sema.stmt_index,
                        StmtClass::Case,
                        StmtClass::CaseDefault);
      }
    
    // Check the construct name 
    auto &name = std::get<1>(x.t);
    CheckStatementName(sema.stmt_index, name, false); 

    return true ; 
  }

  void Post(const CaseStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndSelectStmt =========== 

  bool Pre(const EndSelectStmt &x) {
    // Reminder: EndSelectStmt can end SelectCaseStmt, SelectTypeStmt or SelectRankStmt
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::EndSelect);

    // Check the construct name 
    auto &name = x.v;
    CheckStatementName(sema.stmt_index, name, true); 

    return true ; 
  }

  void Post(const EndSelectStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SelectRankStmt =========== 

  bool Pre(const SelectRankStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::SelectRank);

    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);    

    return true ; 
  }

  void Post(const SelectRankStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SelectRankCaseStmt =========== 

  bool Pre(const SelectRankCaseStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::SelectRankCase);  

    auto &sema = GetSema(x);
    auto &smap = GetStatementMap() ;
    
    // Detect and specialize to StmtClass::SelectRankDefault 
    // or StmtClass::SelectRankStar
    if ( std::holds_alternative<psr::Default>(std::get<0>(x.t).u) )
      {
        // This is a RANK DEFAULT statement
        
        // Let's check that this is the only one
        SMap::Index default_stmt = SMap::None; 
        default_stmt = smap.FindPrevInConstruct(sema.stmt_index,
                                                StmtClass::SelectRankDefault);
        if (default_stmt != SMap::None)
          {
            FAIL(" Duplicate RANK DEFAULT in #" << default_stmt << " and #"
                 << sema.stmt_index) ; 
          }
        
        // And specialize to SelectRankSelectRankDefault in the SMap
        smap.Specialize(sema.stmt_index,
                        StmtClass::SelectRankCase,
                        StmtClass::SelectRankDefault);

      }
    else if ( std::holds_alternative<psr::Star>(std::get<0>(x.t).u) )
      {
        // This is a RANK(*) statement
        
        // Let's check that this is the only one
        SMap::Index default_stmt = SMap::None; 
        default_stmt = smap.FindPrevInConstruct(sema.stmt_index,
                                                StmtClass::SelectRankStar);
        if (default_stmt != SMap::None)
          {
            FAIL(" Duplicate RANK(*) in #" << default_stmt << " and #"
                 << sema.stmt_index) ; 
          }

        // And specialize to SelectRankStar in the SMap
        smap.Specialize(sema.stmt_index,
                        StmtClass::SelectRankCase,
                        StmtClass::SelectRankStar);

         // TODO: Install a scope to 'redeclare' the variable 
     }
    else
      {
        // This is a RANK(expr) statement

        // TODO: evaluate the constant expression 
        // TODO: compare the expression to other case (shall be unique)
        // TODO: Install a scope to declare the variable with given rank.        
      }

    
    // Check the construct name 
    auto &name = std::get<1>(x.t);
    CheckStatementName(sema.stmt_index, name, false); 


    return true ; 
  }

  void Post(const SelectRankCaseStmt &x) {         
    TRACE_CALL() ;
  }

  // =========== SelectTypeStmt =========== 

  bool Pre(const SelectTypeStmt &x) { 
    TRACE_CALL() ;
    auto &sema = InitStmt(x, StmtClass::SelectType);

    auto &name = std::get<0>(x.t);
    SetConstructName(sema.stmt_index, name);    

    return true ; 
  }

  void Post(const SelectTypeStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== TypeGuardStmt =========== 
  // 
  // That is 
  //    TYPE IS (...) 
  // or 
  //    CLASS IS (...) 
  // or
  //    CLASS DEFAULT
  // 

  bool Pre(const TypeGuardStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::TypeGuard);
    auto &sema = GetSema(x);
    auto &smap = GetStatementMap() ;
    
    // Provide the proper specialization:
    //  TYPE  IS      = StmtClass::TypeGuard  
    //  CLASS IS      = StmtClass::ClassGuard  
    //  CLASS DEFAULT = StmtClass::ClassDefault  
    
    if ( std::holds_alternative<psr::Default>(std::get<0>(x.t).u) )
      {
        // This is a CLASS DEFAULT statement
        
        // Let's check that this is the only one
        SMap::Index default_stmt = SMap::None; 
        default_stmt = smap.FindPrevInConstruct(sema.stmt_index,
                                                StmtClass::ClassDefault);
        if (default_stmt != SMap::None)
          {
            FAIL(" Duplicate RANK DEFAULT in #" << default_stmt
                 << " and #" << sema.stmt_index) ; 
          }
        
        // Specialize from TypeGuard to ClassDefault in the SMap
        smap.Specialize(sema.stmt_index,
                        StmtClass::TypeGuard,
                        StmtClass::ClassDefault);

      }
    else if ( std::holds_alternative<psr::DerivedTypeSpec>(std::get<0>(x.t).u) )
      {
        // This is a CLASS IS (...) statement

        // Specialize from TypeGuard to ClassGuard in the SMap
        smap.Specialize(sema.stmt_index,
                        StmtClass::TypeGuard,
                        StmtClass::ClassGuard);

         // TODO: ...
     }
    else
      {
        // This is a TYPE IS (...) statement
        // TODO: ...
      }

    // Check the construct name.
    auto &name = std::get<1>(x.t);
    CheckStatementName(sema.stmt_index, name, false); 


    return true ; 
  }

  void Post(const TypeGuardStmt &x) { 
    TRACE_CALL() ;
  }

  

  // =========== ProcedureDeclarationStmt =========== 

  bool Pre(const ProcedureDeclarationStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::ProcedureDeclaration);
    return true ; 
  }

  void Post(const ProcedureDeclarationStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== StructureStmt =========== 

  bool Pre(const StructureStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Structure);
    return true ; 
  }

  void Post(const StructureStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== StructureDef::EndStructureStmt =========== 

  bool Pre(const StructureDef::EndStructureStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EndStructure);
    return true ; 
  }

  void Post(const StructureDef::EndStructureStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== FormatStmt =========== 

  bool Pre(const FormatStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Format);
    return true ; 
  }

  void Post(const FormatStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EntryStmt =========== 

  bool Pre(const EntryStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Entry);
    return true ; 
  }

  void Post(const EntryStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ImportStmt =========== 

  bool Pre(const ImportStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Import);
    return true ; 
  }

  void Post(const ImportStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AllocateStmt =========== 

  bool Pre(const AllocateStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Allocate);
    return true ; 
  }

  void Post(const AllocateStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== BackspaceStmt =========== 

  bool Pre(const BackspaceStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Backspace);
    return true ; 
  }

  void Post(const BackspaceStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== CallStmt =========== 

  bool Pre(const CallStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Call);
    return true ; 
  }

  void Post(const CallStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== CloseStmt =========== 

  bool Pre(const CloseStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Close);
    return true ; 
  }

  void Post(const CloseStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ContinueStmt =========== 

  bool Pre(const ContinueStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Continue);
    return true ; 
  }

  void Post(const ContinueStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== DeallocateStmt =========== 

  bool Pre(const DeallocateStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Deallocate);
    return true ; 
  }

  void Post(const DeallocateStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EndfileStmt =========== 

  bool Pre(const EndfileStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Endfile);
    return true ; 
  }

  void Post(const EndfileStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EventPostStmt =========== 

  bool Pre(const EventPostStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EventPost);
    return true ; 
  }

  void Post(const EventPostStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== EventWaitStmt =========== 

  bool Pre(const EventWaitStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::EventWait);
    return true ; 
  }

  void Post(const EventWaitStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== CycleStmt =========== 

  bool Pre(const CycleStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::Cycle);

    
    auto & smap = GetStatementMap() ;

    // the optional name of the construct we are looking for. 
    std::optional<sm::Name> target_name ;
    if (x.v) {
      target_name = x.v->ToString();
    }

    // Reminder: Unlike in EXIT, the target of a CYCLE statement is always 
    // a DO so the target resolution are similar but not identical.
    
    SMap::Index target_do = SMap::None ;    
    SMap::Index construct = sema.stmt_index ; 

    // Note: At that point, 'construct' refers to the CYCLE statment which 
    // is not a construct index (i.e. a Start statement). However, in the 
    // loop below, 'construct' will be assigned a proper construct index. 

    bool done=false ; 
    while (!done) 
      {
        construct = smap.StartOfConstruct(smap.GetParent(construct));
        assert( smap.GetGroup(construct) == StmtGroup::Start ) ; 
        auto construct_name = GetConstructName(construct);

        StmtClass construct_class =  smap.GetClass(construct);
        switch(construct_class) {
        case StmtClass::LabelDo:
        case StmtClass::LabelDoWhile:
        case StmtClass::NonLabelDo:
        case StmtClass::NonLabelDoWhile:
          if ( ! target_name ) {
            // The default target is the first loop
            target_do = construct;
            done = true;
          } else if ( construct_name == target_name ) {
            target_do = construct;
            done = true;
          } 
          break;
          
        case StmtClass::LabelDoConcurrent:
        case StmtClass::NonLabelDoConcurrent: 
          // C1135 A cycle-stmt shall not appear within a CHANGE TEAM, CRITICAL, or DO 
          // CONCURRENT construct if it belongs to an outer construct.
          //           
          // Simply speaking, a DO CONCURRENT should either match or fail.
          //
          if ( ! target_name ) {
            // The default target is the first loop
            target_do = construct;
            done = true;
          } else if ( construct_name == target_name ) {
            target_do = construct;
            done = true;
          } else {
            FAIL("CYCLE statement cannot be used to exit a " << StmtClassText(construct_class) << " statement");
            done = true;
          }
          break;


        case StmtClass::ChangeTeam:
        case StmtClass::Critical:
          // C1135 A cycle-stmt shall not appear within a CHANGE TEAM, CRITICAL, or DO 
          // CONCURRENT construct if it belongs to an outer construct.
          //
          FAIL("CYCLE statement cannot be used to exit a " << StmtClassText(construct_class) <<  " statement");
          done = true;
          break ;

        case StmtClass::IfThen:
        case StmtClass::SelectCase:
        case StmtClass::SelectRank:
        case StmtClass::SelectType:
        case StmtClass::Block:
        case StmtClass::Associate:
        case StmtClass::WhereConstruct: 
          // A CYCLE statement can be used to exit those constructs but they are proper targets
          break;

        case StmtClass::Program:
        case StmtClass::Function:
        case StmtClass::Subroutine:
          // We need to stop here. 
          done = true; 
          break;

        case StmtClass::If:
          // This is a non-construct IF that owns the EXIT statement
          break;

        default:
          // TODO: If you hit that internal error then that means that
          // we forgot to handle a construct that is susceptible to 
          // contain an EXIT statement
          INTERNAL_ERROR;

        }
      }

    if ( target_do == SMap::None ) {
      if ( target_name ) {
        FAIL("No construct named '" << *target_name << "' found arount CYCLE statement" ) ;
      } else {
        FAIL("No loop found arount CYCLE statement" ) ;
      }
    }
    
    TRACE("Target of CYCLE statement #" << sema.stmt_index << " is statement #" << target_do );

    // TODO: Do something with target_do

    return true ; 
  }

  void Post(const CycleStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ExitStmt =========== 

  bool Pre(const ExitStmt &x) { 
    TRACE_CALL() ;
    auto & sema = InitStmt(x, StmtClass::Exit);

    auto & smap = GetStatementMap() ;

    // the name of the construct we are looking for (can be null)
    std::optional<sm::Name> target_name;
    if (x.v) 
      target_name = x.v->ToString(); 

    // Remark: I am currently search the target construct by 
    // only considering its identifer but this is actually incorrect
    // because of scopes.
    // 
    // For instance, consider the following piece of code
    //
    //  outer: do i=1,n
    //  inner: do j=1,n
    //    block
    //      import, only :: A,i
    //      outer: do k=1,3
    //        ...
    //      enddo outer
    //      if (A(i)==0) EXIT outer 
    //      A(i) = 42 
    //    end block
    //  enddo inner
    //  enddo outer
    //
    //  The current implemntation would match the i-loop even though
    //  its name should not be visible because of the IMPORT, ONLY 
    //  statement.
    //
    //  The proper way should be: 
    //    - resolve the name to an existing symbol 
    //    - fail is that symbol is not a construct name (by design 
    //      there is no issue with forward references)
    //    - and then explore the parent constructs to match their
    //      respective symbol (stored in the Statement Map?)
    //  
    // Remark: if the symbol holds the SIndex of the construct 
    //   then the match should be done usng that. 
    //
    SMap::Index target_construct = SMap::None ;    
    SMap::Index construct = sema.stmt_index ; 

    // At that point, construct refers to EXIT statment which is not a
    // construct index (i.e. a Start statement). However, in the loop below, 
    // it will be assigned a proper construct index. 

    bool done=false ; 
    while (!done) 
      {
        construct = smap.StartOfConstruct(smap.GetParent(construct));
        assert( smap.GetGroup(construct) == StmtGroup::Start ) ; 
        auto construct_name = GetConstructName(construct);

        StmtClass construct_class =  smap.GetClass(construct);
        switch(construct_class) {
        case StmtClass::LabelDo:
        case StmtClass::LabelDoWhile:
        case StmtClass::NonLabelDo:
        case StmtClass::NonLabelDoWhile:
          if ( ! target_name ) {
            // The default target is the first loop
            target_construct = construct;
            done = true;
          } else if ( construct_name == target_name ) {
            target_construct = construct;
            done = true;
          } 
          break;
          
        case StmtClass::LabelDoConcurrent:
        case StmtClass::NonLabelDoConcurrent: 
        case StmtClass::ChangeTeam:
        case StmtClass::Critical:
          //
          //  C1167 An exit-stmt shall not appear within a CHANGE TEAM, CRITICAL, or DO CONCURRENT construct
          //  if it belongs to that construct or an outer construct.
          //
          FAIL("EXIT statement cannot be used to exit a " << StmtClassText(construct_class) << " statement");
          break ;

        case StmtClass::IfThen:
        case StmtClass::SelectCase:
        case StmtClass::SelectRank:
        case StmtClass::SelectType:
        case StmtClass::Block:
        case StmtClass::Associate:
        case StmtClass::WhereConstruct: 
          // Those constructs that can be 'exited' if explicitly named
          if ( target_name ) { 
            if ( construct_name == target_name ) {
              target_construct = construct;
              done = true;
            }
          } 
          break;

        case StmtClass::Program:
        case StmtClass::Function:
        case StmtClass::Subroutine:
          // We need to stop here. 
          done = true; 
          break;

        case StmtClass::If:
          // This is a non-construct IF that owns the EXIT statement
          break;

        default:
          // TODO: If you hit that internal error then that means that
          // we forgot to handle a construct that is susceptible to 
          // contain an EXIT statement
          INTERNAL_ERROR;

        }
      }

    if ( target_construct == SMap::None ) {
      if ( target_name ) {
        FAIL("No construct named '" << *target_name << "' found arount EXIT statement" ) ;
      } else {
        FAIL("No loop found arount EXIT statement" ) ;
      }
    }
    
    TRACE("Target of EXIT statement #" << sema.stmt_index << " is statement #" << target_construct );
    // TODO: Do something with target_construct


    return true ; 
  }

  void Post(const ExitStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== FailImageStmt =========== 

  bool Pre(const FailImageStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::FailImage);
    return true ; 
  }

  void Post(const FailImageStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== FlushStmt =========== 

  bool Pre(const FlushStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Flush);
    return true ; 
  }

  void Post(const FlushStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== FormTeamStmt =========== 

  bool Pre(const FormTeamStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::FormTeam);
    return true ; 
  }

  void Post(const FormTeamStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== GotoStmt =========== 

  bool Pre(const GotoStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Goto);
    return true ; 
  }

  void Post(const GotoStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== InquireStmt =========== 

  bool Pre(const InquireStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Inquire);
    return true ; 
  }

  void Post(const InquireStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== LockStmt =========== 

  bool Pre(const LockStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Lock);
    return true ; 
  }

  void Post(const LockStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== NullifyStmt =========== 

  bool Pre(const NullifyStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Nullify);
    return true ; 
  }

  void Post(const NullifyStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== OpenStmt =========== 

  bool Pre(const OpenStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Open);
    return true ; 
  }

  void Post(const OpenStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== PointerAssignmentStmt =========== 

  bool Pre(const PointerAssignmentStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::PointerAssignment);
    return true ; 
  }

  void Post(const PointerAssignmentStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ReadStmt =========== 

  bool Pre(const ReadStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Read);
    return true ; 
  }

  void Post(const ReadStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ReturnStmt =========== 

  bool Pre(const ReturnStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Return);
    return true ; 
  }

  void Post(const ReturnStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== RewindStmt =========== 

  bool Pre(const RewindStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Rewind);
    return true ; 
  }

  void Post(const RewindStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== StmtFunctionStmt =========== 

  bool Pre(const StmtFunctionStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Rewind);
    return true ; 
  }

  void Post(const StmtFunctionStmt &x) {     
    TRACE_CALL() ;
  }
  
  // =========== StopStmt =========== 

  bool Pre(const StopStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Stop);
    return true ; 
  }

  void Post(const StopStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SyncAllStmt =========== 

  bool Pre(const SyncAllStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::SyncAll);
    return true ; 
  }

  void Post(const SyncAllStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SyncImagesStmt =========== 

  bool Pre(const SyncImagesStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::SyncImages);
    return true ; 
  }

  void Post(const SyncImagesStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SyncMemoryStmt =========== 

  bool Pre(const SyncMemoryStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::SyncMemory);
    return true ; 
  }

  void Post(const SyncMemoryStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== SyncTeamStmt =========== 

  bool Pre(const SyncTeamStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::SyncTeam);
    return true ; 
  }

  void Post(const SyncTeamStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== UnlockStmt =========== 

  bool Pre(const UnlockStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Unlock);
    return true ; 
  }

  void Post(const UnlockStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== WaitStmt =========== 

  bool Pre(const WaitStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Wait);
    return true ; 
  }

  void Post(const WaitStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== WhereStmt =========== 

  bool Pre(const WhereStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Where);
    return true ; 
  }

  void Post(const WhereStmt &x) { 
    TRACE_CALL() ;
    GetStatementMap().Add( StmtClass::DummyEndWhere, 0); 
  }

  // =========== WriteStmt =========== 

  bool Pre(const WriteStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Write);
    return true ; 
  }

  void Post(const WriteStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ComputedGotoStmt =========== 

  bool Pre(const ComputedGotoStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::ComputedGoto);
    return true ; 
  }

  void Post(const ComputedGotoStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== ForallStmt =========== 

  bool Pre(const ForallStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Forall);
    return true ; 
  }

  void Post(const ForallStmt &x) { 
    TRACE_CALL() ;
    GetStatementMap().Add( StmtClass::DummyEndForall, 0); 
  }

  // =========== ArithmeticIfStmt =========== 

  bool Pre(const ArithmeticIfStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::ArithmeticIf);
    return true ; 
  }

  void Post(const ArithmeticIfStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AssignStmt =========== 

  bool Pre(const AssignStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Assign);
    
    return true ; 
  }

  void Post(const AssignStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== AssignedGotoStmt =========== 

  bool Pre(const AssignedGotoStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::AssignedGoto);
    return true ; 
  }

  void Post(const AssignedGotoStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== PauseStmt =========== 

  bool Pre(const PauseStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Pause);
    return true ; 
  }

  void Post(const PauseStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== PrivateStmt =========== 

  bool Pre(const PrivateStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::Private);
    return true ; 
  }

  void Post(const PrivateStmt &x) { 
    TRACE_CALL() ;
  }

  // =========== TypeBoundProcedureStmt =========== 

  bool Pre(const TypeBoundProcedureStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::TypeBoundProcedure);
    return true ; 
  }

  void Post(const TypeBoundProcedureStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== FinalProcedureStmt =========== 

  bool Pre(const FinalProcedureStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::FinalProcedure);
    return true ; 
  }

  void Post(const FinalProcedureStmt &x) { 
    TRACE_CALL() ;
  }


  // =========== TypeBoundGenericStmt =========== 

  bool Pre(const TypeBoundGenericStmt &x) { 
    TRACE_CALL() ;
    InitStmt(x, StmtClass::TypeBoundGeneric);
    return true ; 
  }

  void Post(const TypeBoundGenericStmt &x) { 
    TRACE_CALL() ;
  }



public:
  
  void run(const ProgramUnit &p) {  
    assert( NoLabelTable() ) ; 
    current_scope_ = unit_scope_;
    Walk(p,*this) ;
    assert( current_scope_ == unit_scope_ ) ;
  }

} ;

}  // of namespace Fortran::parser 


void DoSemanticAnalysis( const psr::CookedSource & source,const psr::Program &all) 
{ 
  psr::Pass1 pass1(source) ;
  for (const psr::ProgramUnit &unit : all.v) {
    TRACE("===========================================================================================================");
    psr::DumpTree(unit);
    TRACE("===========================================================================================================");
    pass1.run(unit) ; 
  } 
}

