//===--- SemaDecl.cpp - Semantic Analysis for Declarations ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declarations.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Builtins.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/Attr.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallSet.h"
using namespace clang;

// C99: 6.7.5p3: Used by ParseDeclarator/ParseField to make sure we have
// a constant expression of type int with a value greater than zero.
bool Sema::VerifyConstantArrayType(const ArrayType *Array,
                                   SourceLocation DeclLoc) { 
  const Expr *Size = Array->getSize();
  if (Size == 0) return false;  // incomplete type.
  
  if (!Size->getType()->isIntegerType()) {
    Diag(Size->getLocStart(), diag::err_array_size_non_int, 
         Size->getType().getAsString(), Size->getSourceRange());
    return true;
  }

  // Verify that the size of the array is an integer constant expr.
  SourceLocation Loc;
  llvm::APSInt SizeVal(32);
  if (!Size->isIntegerConstantExpr(SizeVal, &Loc)) {
    // FIXME: This emits the diagnostic to enforce 6.7.2.1p8, but the message
    // is wrong.  It is also wrong for static variables.
    // FIXME: This is also wrong for:
    // int sub1(int i, char *pi) { typedef int foo[i];
    // struct bar {foo f1; int f2:3; int f3:4} *p; }
    Diag(DeclLoc, diag::err_typecheck_illegal_vla, Size->getSourceRange());
    return true;
  }
  
  // We have a constant expression with an integer type, now make sure 
  // value greater than zero (C99 6.7.5.2p1).
  
  // FIXME: This check isn't specific to static VLAs, this should be moved
  // elsewhere or replicated.  'int X[-1];' inside a function should emit an
  // error.
  if (SizeVal.isSigned()) {
    llvm::APSInt Zero(SizeVal.getBitWidth());
    Zero.setIsUnsigned(false);
    if (SizeVal < Zero) {
      Diag(DeclLoc, diag::err_typecheck_negative_array_size,
           Size->getSourceRange());
      return true;
    } else if (SizeVal == 0) {
      // GCC accepts zero sized static arrays.
      Diag(DeclLoc, diag::err_typecheck_zero_array_size, 
           Size->getSourceRange());
    }
  }
  return false;
}

Sema::DeclTy *Sema::isTypeName(const IdentifierInfo &II, Scope *S) const {
  return dyn_cast_or_null<TypedefDecl>(II.getFETokenInfo<Decl>());
}

void Sema::PopScope(SourceLocation Loc, Scope *S) {
  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    Decl *D = static_cast<Decl*>(*I);
    assert(D && "This decl didn't get pushed??");
    IdentifierInfo *II = D->getIdentifier();
    if (!II) continue;
    
    // Unlink this decl from the identifier.  Because the scope contains decls
    // in an unordered collection, and because we have multiple identifier
    // namespaces (e.g. tag, normal, label),the decl may not be the first entry.
    if (II->getFETokenInfo<Decl>() == D) {
      // Normal case, no multiple decls in different namespaces.
      II->setFETokenInfo(D->getNext());
    } else {
      // Scan ahead.  There are only three namespaces in C, so this loop can
      // never execute more than 3 times.
      Decl *SomeDecl = II->getFETokenInfo<Decl>();
      while (SomeDecl->getNext() != D) {
        SomeDecl = SomeDecl->getNext();
        assert(SomeDecl && "Didn't find this decl on its identifier's chain!");
      }
      SomeDecl->setNext(D->getNext());
    }
    
    // This will have to be revisited for C++: there we want to nest stuff in
    // namespace decls etc.  Even for C, we might want a top-level translation
    // unit decl or something.
    if (!CurFunctionDecl)
      continue;

    // Chain this decl to the containing function, it now owns the memory for
    // the decl.
    D->setNext(CurFunctionDecl->getDeclChain());
    CurFunctionDecl->setDeclChain(D);
  }
}

/// LookupScopedDecl - Look up the inner-most declaration in the specified
/// namespace.
Decl *Sema::LookupScopedDecl(IdentifierInfo *II, unsigned NSI,
                             SourceLocation IdLoc, Scope *S) {
  if (II == 0) return 0;
  Decl::IdentifierNamespace NS = (Decl::IdentifierNamespace)NSI;
  
  // Scan up the scope chain looking for a decl that matches this identifier
  // that is in the appropriate namespace.  This search should not take long, as
  // shadowing of names is uncommon, and deep shadowing is extremely uncommon.
  for (Decl *D = II->getFETokenInfo<Decl>(); D; D = D->getNext())
    if (D->getIdentifierNamespace() == NS)
      return D;
  
  // If we didn't find a use of this identifier, and if the identifier
  // corresponds to a compiler builtin, create the decl object for the builtin
  // now, injecting it into translation unit scope, and return it.
  if (NS == Decl::IDNS_Ordinary) {
    // If this is a builtin on some other target, or if this builtin varies
    // across targets (e.g. in type), emit a diagnostic and mark the translation
    // unit non-portable for using it.
    if (II->isNonPortableBuiltin()) {
      // Only emit this diagnostic once for this builtin.
      II->setNonPortableBuiltin(false);
      Context.Target.DiagnoseNonPortability(IdLoc,
                                            diag::port_target_builtin_use);
    }
    // If this is a builtin on this (or all) targets, create the decl.
    if (unsigned BuiltinID = II->getBuiltinID())
      return LazilyCreateBuiltin(II, BuiltinID, S);
  }
  return 0;
}

/// LazilyCreateBuiltin - The specified Builtin-ID was first used at file scope.
/// lazily create a decl for it.
Decl *Sema::LazilyCreateBuiltin(IdentifierInfo *II, unsigned bid, Scope *S) {
  Builtin::ID BID = (Builtin::ID)bid;

  QualType R = Context.BuiltinInfo.GetBuiltinType(BID, Context);
  FunctionDecl *New = new FunctionDecl(SourceLocation(), II, R,
                                       FunctionDecl::Extern, 0);
  
  // Find translation-unit scope to insert this function into.
  while (S->getParent())
    S = S->getParent();
  S->AddDecl(New);
  
  // Add this decl to the end of the identifier info.
  if (Decl *LastDecl = II->getFETokenInfo<Decl>()) {
    // Scan until we find the last (outermost) decl in the id chain. 
    while (LastDecl->getNext())
      LastDecl = LastDecl->getNext();
    // Insert before (outside) it.
    LastDecl->setNext(New);
  } else {
    II->setFETokenInfo(New);
  }    
  // Make sure clients iterating over decls see this.
  LastInGroupList.push_back(New);
  
  return New;
}

/// MergeTypeDefDecl - We just parsed a typedef 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
TypedefDecl *Sema::MergeTypeDefDecl(TypedefDecl *New, Decl *OldD) {
  // Verify the old decl was also a typedef.
  TypedefDecl *Old = dyn_cast<TypedefDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind,
         New->getName());
    Diag(OldD->getLocation(), diag::err_previous_definition);
    return New;
  }
  
  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // TODO: This is totally simplistic.  It should handle merging functions
  // together etc, merging extern int X; int X; ...
  Diag(New->getLocation(), diag::err_redefinition, New->getName());
  Diag(Old->getLocation(), diag::err_previous_definition);
  return New;
}

/// MergeFunctionDecl - We just parsed a function 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
FunctionDecl *Sema::MergeFunctionDecl(FunctionDecl *New, Decl *OldD) {
  // Verify the old decl was also a function.
  FunctionDecl *Old = dyn_cast<FunctionDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind,
         New->getName());
    Diag(OldD->getLocation(), diag::err_previous_definition);
    return New;
  }
  
  // This is not right, but it's a start.  If 'Old' is a function prototype with
  // the same type as 'New', silently allow this.  FIXME: We should link up decl
  // objects here.
  if (Old->getBody() == 0 && 
      Old->getCanonicalType() == New->getCanonicalType()) {
    return New;
  }
  
  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // TODO: This is totally simplistic.  It should handle merging functions
  // together etc, merging extern int X; int X; ...
  Diag(New->getLocation(), diag::err_redefinition, New->getName());
  Diag(Old->getLocation(), diag::err_previous_definition);
  return New;
}

/// MergeVarDecl - We just parsed a variable 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
/// FIXME: Need to carefully consider tentative definition rules (C99 6.9.2p2).
/// For example, we incorrectly complain about i1, i4 from C99 6.9.2p4.
/// 
VarDecl *Sema::MergeVarDecl(VarDecl *New, Decl *OldD) {
  // Verify the old decl was also a variable.
  VarDecl *Old = dyn_cast<VarDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind,
         New->getName());
    Diag(OldD->getLocation(), diag::err_previous_definition);
    return New;
  }
  // Verify the types match.
  if (Old->getCanonicalType() != New->getCanonicalType()) {
    Diag(New->getLocation(), diag::err_redefinition, New->getName());
    Diag(Old->getLocation(), diag::err_previous_definition);
    return New;
  }
  // We've verified the types match, now check if Old is "extern".
  if (Old->getStorageClass() != VarDecl::Extern) {
    Diag(New->getLocation(), diag::err_redefinition, New->getName());
    Diag(Old->getLocation(), diag::err_previous_definition);
  }
  return New;
}

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclTy *Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  // TODO: emit error on 'int;' or 'const enum foo;'.
  // TODO: emit error on 'typedef int;'
  // if (!DS.isMissingDeclaratorOk()) Diag(...);
  
  return 0;
}

Sema::DeclTy *
Sema::ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init, 
                      DeclTy *lastDeclarator) {
  Decl *LastDeclarator = (Decl*)lastDeclarator;
  IdentifierInfo *II = D.getIdentifier();
  
  // See if this is a redefinition of a variable in the same scope.
  Decl *PrevDecl = LookupScopedDecl(II, Decl::IDNS_Ordinary,
                                    D.getIdentifierLoc(), S);
  if (PrevDecl && !S->isDeclScope(PrevDecl))
    PrevDecl = 0;   // If in outer scope, it isn't the same thing.

  Decl *New;
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    TypedefDecl *NewTD = ParseTypedefDecl(S, D, LastDeclarator);
    if (!NewTD) return 0;

    // Handle attributes prior to checking for duplicates in MergeVarDecl
    HandleDeclAttributes(NewTD, D.getDeclSpec().getAttributes(),
                         D.getAttributes());
    // Merge the decl with the existing one if appropriate.
    if (PrevDecl) {
      NewTD = MergeTypeDefDecl(NewTD, PrevDecl);
      if (NewTD == 0) return 0;
    }
    New = NewTD;
    if (S->getParent() == 0) {
      // C99 6.7.7p2: If a typedef name specifies a variably modified type
      // then it shall have block scope.
      if (ArrayType *ary = dyn_cast<ArrayType>(NewTD->getUnderlyingType())) {
        if (VerifyConstantArrayType(ary, D.getIdentifierLoc()))
          return 0;
      }
    }
  } else if (D.isFunctionDeclarator()) {
    QualType R = GetTypeForDeclarator(D, S);
    if (R.isNull()) return 0; // FIXME: "auto func();" passes through...
    
    FunctionDecl::StorageClass SC;
    switch (D.getDeclSpec().getStorageClassSpec()) {
      default: assert(0 && "Unknown storage class!");
      case DeclSpec::SCS_auto:        
      case DeclSpec::SCS_register:
        Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_func,
             R.getAsString());
        return 0;
      case DeclSpec::SCS_unspecified: SC = FunctionDecl::None; break;
      case DeclSpec::SCS_extern:      SC = FunctionDecl::Extern; break;
      case DeclSpec::SCS_static:      SC = FunctionDecl::Static; break;
    }

    FunctionDecl *NewFD = new FunctionDecl(D.getIdentifierLoc(), II, R, SC,
                                           LastDeclarator);
    
    // Merge the decl with the existing one if appropriate.
    if (PrevDecl) {
      NewFD = MergeFunctionDecl(NewFD, PrevDecl);
      if (NewFD == 0) return 0;
    }
    New = NewFD;
  } else {
    QualType R = GetTypeForDeclarator(D, S);
    if (R.isNull()) return 0;

    VarDecl *NewVD;
    VarDecl::StorageClass SC;
    switch (D.getDeclSpec().getStorageClassSpec()) {
      default: assert(0 && "Unknown storage class!");
      case DeclSpec::SCS_unspecified: SC = VarDecl::None; break;
      case DeclSpec::SCS_extern:      SC = VarDecl::Extern; break;
      case DeclSpec::SCS_static:      SC = VarDecl::Static; break;
      case DeclSpec::SCS_auto:        SC = VarDecl::Auto; break;
      case DeclSpec::SCS_register:    SC = VarDecl::Register; break;
    }    
    if (S->getParent() == 0) {
      // File scope. C99 6.9.2p2: A declaration of an identifier for and 
      // object that has file scope without an initializer, and without a
      // storage-class specifier or with the storage-class specifier "static",
      // constitutes a tentative definition. Note: A tentative definition with
      // external linkage is valid (C99 6.2.2p5).
      if (!Init && SC == VarDecl::Static) {
        // C99 6.9.2p3: If the declaration of an identifier for an object is
        // a tentative definition and has internal linkage (C99 6.2.2p3), the  
        // declared type shall not be an incomplete type.
        if (R->isIncompleteType()) {
          Diag(D.getIdentifierLoc(), diag::err_typecheck_decl_incomplete_type,
               R.getAsString());
          return 0;
        }
      }
      // C99 6.9p2: The storage-class specifiers auto and register shall not
      // appear in the declaration specifiers in an external declaration.
      if (SC == VarDecl::Auto || SC == VarDecl::Register) {
        Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_fscope,
             R.getAsString());
        return 0;
      }
      // C99 6.7.5.2p2: If an identifier is declared to be an object with 
      // static storage duration, it shall not have a variable length array.
      if (ArrayType *ary = dyn_cast<ArrayType>(R.getCanonicalType())) {
        if (VerifyConstantArrayType(ary, D.getIdentifierLoc()))
          return 0;
      }
      NewVD = new FileVarDecl(D.getIdentifierLoc(), II, R, SC, LastDeclarator);
    } else { 
      // Block scope. C99 6.7p7: If an identifier for an object is declared with
      // no linkage (C99 6.2.2p6), the type for the object shall be complete...
      if (SC != VarDecl::Extern) {
        if (R->isIncompleteType()) {
          Diag(D.getIdentifierLoc(), diag::err_typecheck_decl_incomplete_type,
               R.getAsString());
          return 0;
        }
      }
      if (SC == VarDecl::Static) {
        // C99 6.7.5.2p2: If an identifier is declared to be an object with 
        // static storage duration, it shall not have a variable length array.
        if (ArrayType *ary = dyn_cast<ArrayType>(R.getCanonicalType())) {
          if (VerifyConstantArrayType(ary, D.getIdentifierLoc()))
            return 0;
        }
      }
      NewVD = new BlockVarDecl(D.getIdentifierLoc(), II, R, SC, LastDeclarator);
    }    
    // Handle attributes prior to checking for duplicates in MergeVarDecl
    HandleDeclAttributes(NewVD, D.getDeclSpec().getAttributes(),
                         D.getAttributes());
     
    // Merge the decl with the existing one if appropriate.
    if (PrevDecl) {
      NewVD = MergeVarDecl(NewVD, PrevDecl);
      if (NewVD == 0) return 0;
    }
    New = NewVD;
  }
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    New->setNext(II->getFETokenInfo<Decl>());
    II->setFETokenInfo(New);
    S->AddDecl(New);
  }
  
  if (S->getParent() == 0)
    AddTopLevelDecl(New, LastDeclarator);
  
  return New;
}

/// The declarators are chained together backwards, reverse the list.
Sema::DeclTy *Sema::FinalizeDeclaratorGroup(Scope *S, DeclTy *group) {
  // Often we have single declarators, handle them quickly.
  Decl *Group = static_cast<Decl*>(group);
  if (Group == 0 || Group->getNextDeclarator() == 0) return Group;
  
  Decl *NewGroup = 0;
  while (Group) {
    Decl *Next = Group->getNextDeclarator();
    Group->setNextDeclarator(NewGroup);
    NewGroup = Group;
    Group = Next;
  }
  return NewGroup;
}
  
ParmVarDecl *
Sema::ParseParamDeclarator(DeclaratorChunk &FTI, unsigned ArgNo,
                           Scope *FnScope) {
  const DeclaratorChunk::ParamInfo &PI = FTI.Fun.ArgInfo[ArgNo];

  IdentifierInfo *II = PI.Ident;
  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // Can this happen for params?  We already checked that they don't conflict
  // among each other.  Here they can only shadow globals, which is ok.
  if (Decl *PrevDecl = LookupScopedDecl(II, Decl::IDNS_Ordinary,
                                        PI.IdentLoc, FnScope)) {
    
  }
  
  // FIXME: Handle storage class (auto, register). No declarator?
  // TODO: Chain to previous parameter with the prevdeclarator chain?
  ParmVarDecl *New = new ParmVarDecl(PI.IdentLoc, II, 
                                     QualType::getFromOpaquePtr(PI.TypeInfo), 
                                     VarDecl::None, 0);

  // If this has an identifier, add it to the scope stack.
  if (II) {
    New->setNext(II->getFETokenInfo<Decl>());
    II->setFETokenInfo(New);
    FnScope->AddDecl(New);
  }

  return New;
}
  

Sema::DeclTy *Sema::ParseStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
  assert(CurFunctionDecl == 0 && "Function parsing confused");
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;
  
  // Verify 6.9.1p6: 'every identifier in the identifier list shall be declared'
  // for a K&R function.
  if (!FTI.hasPrototype) {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
      if (FTI.ArgInfo[i].TypeInfo == 0) {
        Diag(FTI.ArgInfo[i].IdentLoc, diag::ext_param_not_declared,
             FTI.ArgInfo[i].Ident->getName());
        // Implicitly declare the argument as type 'int' for lack of a better
        // type.
        FTI.ArgInfo[i].TypeInfo = Context.IntTy.getAsOpaquePtr();
      }
    }
   
    // Since this is a function definition, act as though we have information
    // about the arguments.
    FTI.hasPrototype = true;
  } else {
    // FIXME: Diagnose arguments without names in C.
    
  }
  
  Scope *GlobalScope = FnBodyScope->getParent();
  
  FunctionDecl *FD =
    static_cast<FunctionDecl*>(ParseDeclarator(GlobalScope, D, 0, 0));
  CurFunctionDecl = FD;
  
  // Create Decl objects for each parameter, adding them to the FunctionDecl.
  llvm::SmallVector<ParmVarDecl*, 16> Params;
  
  // Check for C99 6.7.5.3p10 - foo(void) is a non-varargs function that takes
  // no arguments, not a function that takes a single void argument.
  if (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
      FTI.ArgInfo[0].TypeInfo == Context.VoidTy.getAsOpaquePtr()) {
    // empty arg list, don't push any params.
  } else {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
      Params.push_back(ParseParamDeclarator(D.getTypeObject(0), i,FnBodyScope));
  }
  
  FD->setParams(&Params[0], Params.size());
  
  return FD;
}

Sema::DeclTy *Sema::ParseFunctionDefBody(DeclTy *D, StmtTy *Body) {
  FunctionDecl *FD = static_cast<FunctionDecl*>(D);
  FD->setBody((Stmt*)Body);
  
  assert(FD == CurFunctionDecl && "Function parsing confused");
  CurFunctionDecl = 0;
  
  // Verify and clean out per-function state.
  
  // Check goto/label use.
  for (llvm::DenseMap<IdentifierInfo*, LabelStmt*>::iterator
       I = LabelMap.begin(), E = LabelMap.end(); I != E; ++I) {
    // Verify that we have no forward references left.  If so, there was a goto
    // or address of a label taken, but no definition of it.  Label fwd
    // definitions are indicated with a null substmt.
    if (I->second->getSubStmt() == 0) {
      LabelStmt *L = I->second;
      // Emit error.
      Diag(L->getIdentLoc(), diag::err_undeclared_label_use, L->getName());
      
      // At this point, we have gotos that use the bogus label.  Stitch it into
      // the function body so that they aren't leaked and that the AST is well
      // formed.
      L->setSubStmt(new NullStmt(L->getIdentLoc()));
      cast<CompoundStmt>((Stmt*)Body)->push_back(L);
    }
  }
  LabelMap.clear();
  
  return FD;
}


/// ImplicitlyDefineFunction - An undeclared identifier was used in a function
/// call, forming a call to an implicitly defined function (per C99 6.5.1p2).
Decl *Sema::ImplicitlyDefineFunction(SourceLocation Loc, IdentifierInfo &II,
                                     Scope *S) {
  if (getLangOptions().C99)  // Extension in C99.
    Diag(Loc, diag::ext_implicit_function_decl, II.getName());
  else  // Legal in C90, but warn about it.
    Diag(Loc, diag::warn_implicit_function_decl, II.getName());
  
  // FIXME: handle stuff like:
  // void foo() { extern float X(); }
  // void bar() { X(); }  <-- implicit decl for X in another scope.

  // Set a Declarator for the implicit definition: int foo();
  const char *Dummy;
  DeclSpec DS;
  bool Error = DS.SetTypeSpecType(DeclSpec::TST_int, Loc, Dummy);
  Error = Error; // Silence warning.
  assert(!Error && "Error setting up implicit decl!");
  Declarator D(DS, Declarator::BlockContext);
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, 0, 0, Loc));
  D.SetIdentifier(&II, Loc);
  
  // Find translation-unit scope to insert this function into.
  while (S->getParent())
    S = S->getParent();
  
  return static_cast<Decl*>(ParseDeclarator(S, D, 0, 0));
}


TypedefDecl *Sema::ParseTypedefDecl(Scope *S, Declarator &D,
                                    Decl *LastDeclarator) {
  assert(D.getIdentifier() && "Wrong callback for declspec without declarator");
  
  QualType T = GetTypeForDeclarator(D, S);
  if (T.isNull()) return 0;
  
  // Scope manipulation handled by caller.
  return new TypedefDecl(D.getIdentifierLoc(), D.getIdentifier(), T,
                         LastDeclarator);
}


/// ParseTag - This is invoked when we see 'struct foo' or 'struct {'.  In the
/// former case, Name will be non-null.  In the later case, Name will be null.
/// TagType indicates what kind of tag this is. TK indicates whether this is a
/// reference/declaration/definition of a tag.
Sema::DeclTy *Sema::ParseTag(Scope *S, unsigned TagType, TagKind TK,
                             SourceLocation KWLoc, IdentifierInfo *Name,
                             SourceLocation NameLoc, AttributeList *Attr) {
  // If this is a use of an existing tag, it must have a name.
  assert((Name != 0 || TK == TK_Definition) &&
         "Nameless record must be a definition!");
  
  Decl::Kind Kind;
  switch (TagType) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = Decl::Struct; break;
  case DeclSpec::TST_union:  Kind = Decl::Union; break;
//case DeclSpec::TST_class:  Kind = Decl::Class; break;
  case DeclSpec::TST_enum:   Kind = Decl::Enum; break;
  }
  
  // If this is a named struct, check to see if there was a previous forward
  // declaration or definition.
  if (TagDecl *PrevDecl = 
          dyn_cast_or_null<TagDecl>(LookupScopedDecl(Name, Decl::IDNS_Tag,
                                                     NameLoc, S))) {
    
    // If this is a use of a previous tag, or if the tag is already declared in
    // the same scope (so that the definition/declaration completes or
    // rementions the tag), reuse the decl.
    if (TK == TK_Reference || S->isDeclScope(PrevDecl)) {
      // Make sure that this wasn't declared as an enum and now used as a struct
      // or something similar.
      if (PrevDecl->getKind() != Kind) {
        Diag(KWLoc, diag::err_use_with_wrong_tag, Name->getName());
        Diag(PrevDecl->getLocation(), diag::err_previous_use);
      }
      
      // If this is a use or a forward declaration, we're good.
      if (TK != TK_Definition)
        return PrevDecl;

      // Diagnose attempts to redefine a tag.
      if (PrevDecl->isDefinition()) {
        Diag(NameLoc, diag::err_redefinition, Name->getName());
        Diag(PrevDecl->getLocation(), diag::err_previous_definition);
        // If this is a redefinition, recover by making this struct be
        // anonymous, which will make any later references get the previous
        // definition.
        Name = 0;
      } else {
        // Okay, this is definition of a previously declared or referenced tag.
        // Move the location of the decl to be the definition site.
        PrevDecl->setLocation(NameLoc);
        return PrevDecl;
      }
    }
    // If we get here, this is a definition of a new struct type in a nested
    // scope, e.g. "struct foo; void bar() { struct foo; }", just create a new
    // type.
  }
  
  // If there is an identifier, use the location of the identifier as the
  // location of the decl, otherwise use the location of the struct/union
  // keyword.
  SourceLocation Loc = NameLoc.isValid() ? NameLoc : KWLoc;
  
  // Otherwise, if this is the first time we've seen this tag, create the decl.
  TagDecl *New;
  switch (Kind) {
  default: assert(0 && "Unknown tag kind!");
  case Decl::Enum:
    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // enum X { A, B, C } D;    D should chain to X.
    New = new EnumDecl(Loc, Name, 0);
    // If this is an undefined enum, warn.
    if (TK != TK_Definition) Diag(Loc, diag::ext_forward_ref_enum);
    break;
  case Decl::Union:
  case Decl::Struct:
  case Decl::Class:
    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // struct X { int A; } D;    D should chain to X.
    New = new RecordDecl(Kind, Loc, Name, 0);
    break;
  }    
  
  // If this has an identifier, add it to the scope stack.
  if (Name) {
    New->setNext(Name->getFETokenInfo<Decl>());
    Name->setFETokenInfo(New);
    S->AddDecl(New);
  }
  
  return New;
}

/// ParseField - Each field of a struct/union/class is passed into this in order
/// to create a FieldDecl object for it.
Sema::DeclTy *Sema::ParseField(Scope *S, DeclTy *TagDecl,
                               SourceLocation DeclStart, 
                               Declarator &D, ExprTy *BitfieldWidth) {
  IdentifierInfo *II = D.getIdentifier();
  Expr *BitWidth = (Expr*)BitfieldWidth;
  
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();
  
  // FIXME: Unnamed fields can be handled in various different ways, for
  // example, unnamed unions inject all members into the struct namespace!
  
  
  if (BitWidth) {
    // TODO: Validate.
    //printf("WARNING: BITFIELDS IGNORED!\n");
    
    // 6.7.2.1p3
    // 6.7.2.1p4
    
  } else {
    // Not a bitfield.

    // validate II.
    
  }
  
  QualType T = GetTypeForDeclarator(D, S);
  if (T.isNull()) return 0;
  
  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (ArrayType *ary = dyn_cast<ArrayType>(T.getCanonicalType())) {
    if (VerifyConstantArrayType(ary, Loc))
      return 0;
  }
  
  // FIXME: Chain fielddecls together.
  return new FieldDecl(Loc, II, T, 0);
}

void Sema::ParseRecordBody(SourceLocation RecLoc, DeclTy *RecDecl,
                           DeclTy **Fields, unsigned NumFields) {
  RecordDecl *Record = cast<RecordDecl>(static_cast<Decl*>(RecDecl));
  if (Record->isDefinition()) {
    // Diagnose code like:
    //     struct S { struct S {} X; };
    // We discover this when we complete the outer S.  Reject and ignore the
    // outer S.
    Diag(Record->getLocation(), diag::err_nested_redefinition,
         Record->getKindName());
    Diag(RecLoc, diag::err_previous_definition);
    return;
  }

  // Verify that all the fields are okay.
  unsigned NumNamedMembers = 0;
  llvm::SmallVector<FieldDecl*, 32> RecFields;
  llvm::SmallSet<const IdentifierInfo*, 32> FieldIDs;
  
  for (unsigned i = 0; i != NumFields; ++i) {
    FieldDecl *FD = cast_or_null<FieldDecl>(static_cast<Decl*>(Fields[i]));
    if (!FD) continue;  // Already issued a diagnostic.
    
    // Get the type for the field.
    Type *FDTy = FD->getType().getCanonicalType().getTypePtr();
    
    // C99 6.7.2.1p2 - A field may not be a function type.
    if (isa<FunctionType>(FDTy)) {
      Diag(FD->getLocation(), diag::err_field_declared_as_function,
           FD->getName());
      delete FD;
      continue;
    }

    // C99 6.7.2.1p2 - A field may not be an incomplete type except...
    if (FDTy->isIncompleteType()) {
      if (i != NumFields-1 ||                   // ... that the last member ...
          Record->getKind() != Decl::Struct ||  // ... of a structure ...
          !isa<ArrayType>(FDTy)) {         //... may have incomplete array type.
        Diag(FD->getLocation(), diag::err_field_incomplete, FD->getName());
        delete FD;
        continue;
      }
      if (NumNamedMembers < 1) {      //... must have more than named member ...
        Diag(FD->getLocation(), diag::err_flexible_array_empty_struct,
             FD->getName());
        delete FD;
        continue;
      }
      
      // Okay, we have a legal flexible array member at the end of the struct.
      Record->setHasFlexibleArrayMember(true);
    }
    
    
    /// C99 6.7.2.1p2 - a struct ending in a flexible array member cannot be the
    /// field of another structure or the element of an array.
    if (RecordType *FDTTy = dyn_cast<RecordType>(FDTy)) {
      if (FDTTy->getDecl()->hasFlexibleArrayMember()) {
        // If this is a member of a union, then entire union becomes "flexible".
        if (Record->getKind() == Decl::Union) {
          Record->setHasFlexibleArrayMember(true);
        } else {
          // If this is a struct/class and this is not the last element, reject
          // it.  Note that GCC supports variable sized arrays in the middle of
          // structures.
          if (i != NumFields-1) {
            Diag(FD->getLocation(), diag::err_variable_sized_type_in_struct,
                 FD->getName());
            delete FD;
            continue;
          }

          // We support flexible arrays at the end of structs in other structs
          // as an extension.
          Diag(FD->getLocation(), diag::ext_flexible_array_in_struct,
               FD->getName());
          Record->setHasFlexibleArrayMember(true);
        }
      }
    }
    
    // Keep track of the number of named members.
    if (IdentifierInfo *II = FD->getIdentifier()) {
      // Detect duplicate member names.
      if (!FieldIDs.insert(II)) {
        Diag(FD->getLocation(), diag::err_duplicate_member, II->getName());
        // Find the previous decl.
        SourceLocation PrevLoc;
        for (unsigned i = 0, e = RecFields.size(); ; ++i) {
          assert(i != e && "Didn't find previous def!");
          if (RecFields[i]->getIdentifier() == II) {
            PrevLoc = RecFields[i]->getLocation();
            break;
          }
        }
        Diag(PrevLoc, diag::err_previous_definition);
        delete FD;
        continue;
      }
      ++NumNamedMembers;
    }
    
    // Remember good fields.
    RecFields.push_back(FD);
  }
 
  
  // Okay, we successfully defined 'Record'.
  Record->defineBody(&RecFields[0], RecFields.size());
}

Sema::DeclTy *Sema::ParseEnumConstant(Scope *S, DeclTy *theEnumDecl,
                                      DeclTy *lastEnumConst,
                                      SourceLocation IdLoc, IdentifierInfo *Id,
                                      SourceLocation EqualLoc, ExprTy *val) {
  theEnumDecl = theEnumDecl;  // silence unused warning.
  EnumConstantDecl *LastEnumConst =
    cast_or_null<EnumConstantDecl>(static_cast<Decl*>(lastEnumConst));
  Expr *Val = static_cast<Expr*>(val);

  // Verify that there isn't already something declared with this name in this
  // scope.
  if (Decl *PrevDecl = LookupScopedDecl(Id, Decl::IDNS_Ordinary, IdLoc, S)) {
    if (S->isDeclScope(PrevDecl)) {
      if (isa<EnumConstantDecl>(PrevDecl))
        Diag(IdLoc, diag::err_redefinition_of_enumerator, Id->getName());
      else
        Diag(IdLoc, diag::err_redefinition, Id->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
      // FIXME: Don't leak memory: delete Val;
      return 0;
    }
  }

  llvm::APSInt EnumVal(32);
  QualType EltTy;
  if (Val) {
    // C99 6.7.2.2p2: Make sure we have an integer constant expression.
    SourceLocation ExpLoc;
    if (!Val->isIntegerConstantExpr(EnumVal, &ExpLoc)) {
      Diag(ExpLoc, diag::err_enum_value_not_integer_constant_expr, 
           Id->getName());
      // FIXME: Don't leak memory: delete Val;
      return 0;
    }
    EltTy = Val->getType();
  } else if (LastEnumConst) {
    // Assign the last value + 1.
    EnumVal = LastEnumConst->getInitVal();
    ++EnumVal;
    // FIXME: detect overflow!
    EltTy = LastEnumConst->getType();
  } else {
    // First value, set to zero.
    EltTy = Context.IntTy;
    // FIXME: Resize EnumVal to the size of int.
  }
  
  // TODO: Default promotions to int/uint.
  
  // TODO: If the result value doesn't fit in an int, it must be a long or long
  // long value.  ISO C does not support this, but GCC does as an extension,
  // emit a warning.
  
  EnumConstantDecl *New = new EnumConstantDecl(IdLoc, Id, EltTy, Val, EnumVal,
                                               LastEnumConst);
  
  // Register this decl in the current scope stack.
  New->setNext(Id->getFETokenInfo<Decl>());
  Id->setFETokenInfo(New);
  S->AddDecl(New);
  return New;
}

void Sema::ParseEnumBody(SourceLocation EnumLoc, DeclTy *EnumDeclX,
                         DeclTy **Elements, unsigned NumElements) {
  EnumDecl *Enum = cast<EnumDecl>(static_cast<Decl*>(EnumDeclX));
  assert(!Enum->isDefinition() && "Enum redefinitions can't reach here");
  
  // Verify that all the values are okay, and reverse the list.
  EnumConstantDecl *EltList = 0;
  for (unsigned i = 0; i != NumElements; ++i) {
    EnumConstantDecl *ECD =
      cast_or_null<EnumConstantDecl>(static_cast<Decl*>(Elements[i]));
    if (!ECD) continue;  // Already issued a diagnostic.

    ECD->setNextDeclarator(EltList);
    EltList = ECD;
  }
  
  Enum->defineElements(EltList);
}

void Sema::AddTopLevelDecl(Decl *current, Decl *last) {
  if (!current) return;

  // If this is a top-level decl that is chained to some other (e.g. int A,B,C;)
  // remember this in the LastInGroupList list.
  if (last)
    LastInGroupList.push_back((Decl*)last);
}

void Sema::HandleDeclAttribute(Decl *New, AttributeList *rawAttr) {
  if (strcmp(rawAttr->getAttributeName()->getName(), "vector_size") == 0) {
    if (ValueDecl *vDecl = dyn_cast<ValueDecl>(New)) {
      QualType newType = HandleVectorTypeAttribute(vDecl->getType(), rawAttr);
      if (!newType.isNull()) // install the new vector type into the decl
        vDecl->setType(newType);
    } 
    if (TypedefDecl *tDecl = dyn_cast<TypedefDecl>(New)) {
      QualType newType = HandleVectorTypeAttribute(tDecl->getUnderlyingType(), 
                                                   rawAttr);
      if (!newType.isNull()) // install the new vector type into the decl
        tDecl->setUnderlyingType(newType);
    }
  }
  // FIXME: add other attributes...
}

void Sema::HandleDeclAttributes(Decl *New, AttributeList *declspec_prefix,
                                AttributeList *declarator_postfix) {
  while (declspec_prefix) {
    HandleDeclAttribute(New, declspec_prefix);
    declspec_prefix = declspec_prefix->getNext();
  }
  while (declarator_postfix) {
    HandleDeclAttribute(New, declarator_postfix);
    declarator_postfix = declarator_postfix->getNext();
  }
}

QualType Sema::HandleVectorTypeAttribute(QualType curType, 
                                      AttributeList *rawAttr) {
  // check the attribute arugments.
  if (rawAttr->getNumArgs() != 1) {
    Diag(rawAttr->getAttributeLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return QualType();
  }
  Expr *sizeExpr = static_cast<Expr *>(rawAttr->getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize)) {
    Diag(rawAttr->getAttributeLoc(), diag::err_attribute_vector_size_not_int,
         sizeExpr->getSourceRange());
    return QualType();
  }
  // navigate to the base type - we need to provide for vector pointers, 
  // vector arrays, and functions returning vectors.
  Type *canonType = curType.getCanonicalType().getTypePtr();
  
  while (canonType->isPointerType() || canonType->isArrayType() ||
         canonType->isFunctionType()) {
    if (PointerType *PT = dyn_cast<PointerType>(canonType))
      canonType = PT->getPointeeType().getTypePtr();
    else if (ArrayType *AT = dyn_cast<ArrayType>(canonType))
      canonType = AT->getElementType().getTypePtr();
    else if (FunctionType *FT = dyn_cast<FunctionType>(canonType))
      canonType = FT->getResultType().getTypePtr();
  }
  // the base type must be integer or float.
  if (!(canonType->isIntegerType() || canonType->isRealFloatingType())) {
    Diag(rawAttr->getAttributeLoc(), diag::err_attribute_invalid_vector_type,
         curType.getCanonicalType().getAsString());
    return QualType();
  }
  BuiltinType *baseType = cast<BuiltinType>(canonType);
  unsigned typeSize = baseType->getSize();
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = vecSize.getZExtValue() * 8; 
  
  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    Diag(rawAttr->getAttributeLoc(), diag::err_attribute_invalid_size,
         sizeExpr->getSourceRange());
    return QualType();
  }
  if (vectorSize == 0) {
    Diag(rawAttr->getAttributeLoc(), diag::err_attribute_zero_size,
         sizeExpr->getSourceRange());
    return QualType();
  }
  // Since OpenCU requires 3 element vectors (OpenCU 5.1.2), we don't restrict
  // the number of elements to be a power of two (unlike GCC).
  // Instantiate the vector type, the number of elements is > 0.
  return Context.convertToVectorType(curType, vectorSize/typeSize);
}

