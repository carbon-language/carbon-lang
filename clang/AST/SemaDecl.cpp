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
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
using namespace llvm;
using namespace clang;


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

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclTy *Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  // TODO: emit error on 'int;' or 'const enum foo;'.
  // TODO: emit error on 'typedef int;'
  // if (!DS.isMissingDeclaratorOk()) Diag(...);
  
  // TODO: Register 'struct foo;' with the type system as an opaque struct.
  
  // TODO: Check that we don't already have 'union foo;' or something else
  // that conflicts.
  return 0;
}

/// LookupScopedDecl - Look up the inner-most declaration in the specified
/// namespace.
static Decl *LookupScopedDecl(IdentifierInfo *II, Decl::IdentifierNamespace NS){
  if (II == 0) return 0;
  
  // Scan up the scope chain looking for a decl that matches this identifier
  // that is in the appropriate namespace.  This search should not take long, as
  // shadowing of names is uncommon, and deep shadowing is extremely uncommon.
  for (Decl *D = II->getFETokenInfo<Decl>(); D; D = D->getNext())
    if (D->getIdentifierNamespace() == NS)
      return D;
  return 0;
}


Action::DeclTy *
Sema::ParseDeclarator(Scope *S, Declarator &D, ExprTy *Init, 
                      DeclTy *LastInGroup) {
  IdentifierInfo *II = D.getIdentifier();
  
  if (Decl *PrevDecl = LookupScopedDecl(II, Decl::IDNS_Ordinary)) {
    // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
    if (S->isDeclScope(PrevDecl)) {
      // TODO: This is totally simplistic.  It should handle merging functions
      // together etc, merging extern int X; int X; ...
      Diag(D.getIdentifierLoc(), diag::err_redefinition, II->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
    }
  }
  
  Decl *New;
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    New = ParseTypedefDecl(S, D);
    if (!New) return 0;
  } else if (D.isFunctionDeclarator()) {
    TypeRef R = GetTypeForDeclarator(D, S);
    if (R.isNull()) return 0;
    New = new FunctionDecl(D.getIdentifierLoc(), II, R);
  } else {
    TypeRef R = GetTypeForDeclarator(D, S);
    if (R.isNull()) return 0;
    New = new VarDecl(D.getIdentifierLoc(), II, R);
  }
  
  
  // If this has an identifier, add it to the scope stack.
  if (II) {
    New->setNext(II->getFETokenInfo<Decl>());
    II->setFETokenInfo(New);
    S->AddDecl(New);
  }
  
  // If this is a top-level decl that is chained to some other (e.g. int A,B,C;)
  // remember this in the LastInGroupList list.
  if (LastInGroup && S->getParent() == 0)
    LastInGroupList.push_back((Decl*)LastInGroup);
  
  return New;
}

VarDecl *
Sema::ParseParamDeclarator(DeclaratorChunk &FTI, unsigned ArgNo,
                           Scope *FnScope) {
  const DeclaratorChunk::ParamInfo &PI = FTI.Fun.ArgInfo[ArgNo];

  IdentifierInfo *II = PI.Ident;
  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // Can this happen for params?  We already checked that they don't conflict
  // among each other.  Here they can only shadow globals, which is ok.
  if (Decl *PrevDecl = LookupScopedDecl(II, Decl::IDNS_Ordinary)) {
    
  }
  
  VarDecl *New = new VarDecl(PI.IdentLoc, II, static_cast<Type*>(PI.TypeInfo));
  
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
        Diag(FTI.ArgInfo[i].IdentLoc, diag::err_param_not_declared,
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
  SmallVector<VarDecl*, 16> Params;
  
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
  assert(!Error && "Error setting up implicit decl!");
  Declarator D(DS, Declarator::BlockContext);
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, 0, 0, Loc));
  D.SetIdentifier(&II, Loc);
  
  Decl *Result = static_cast<Decl*>(ParseDeclarator(S, D, 0, 0));
  
  // Visit this implicit declaration like any other top-level form.
  LastInGroupList.push_back(Result);
  return Result;
}


Decl *Sema::ParseTypedefDecl(Scope *S, Declarator &D) {
  assert(D.getIdentifier() && "Wrong callback for declspec withotu declarator");
  
  TypeRef T = GetTypeForDeclarator(D, S);
  if (T.isNull()) return 0;
  
  // Scope manipulation handled by caller.
  return new TypedefDecl(D.getIdentifierLoc(), D.getIdentifier(), T);
}


/// ParseTag - This is invoked when we see 'struct foo' or 'struct {'.  In the
/// former case, Name will be non-null.  In the later case, Name will be null.
/// TagType indicates what kind of tag this is. TK indicates whether this is a
/// reference/declaration/definition of a tag.
Sema::DeclTy *Sema::ParseTag(Scope *S, unsigned TagType, TagKind TK,
                             SourceLocation KWLoc, IdentifierInfo *Name,
                             SourceLocation NameLoc) {
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
          dyn_cast_or_null<TagDecl>(LookupScopedDecl(Name, Decl::IDNS_Tag))) {
    
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
    New = new EnumDecl(Loc, Name);
    // If this is an undefined enum, warn.
    if (TK != TK_Definition) Diag(Loc, diag::ext_forward_ref_enum);
    break;
  case Decl::Union:
  case Decl::Struct:
  case Decl::Class:
    New = new RecordDecl(Kind, Loc, Name);
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
  
  if (BitWidth) {
    // TODO: Validate.
    printf("WARNING: BITFIELDS IGNORED!\n");
    
    // 6.7.2.1p3
    // 6.7.2.1p4
    
  } else {
    // Not a bitfield.

    // validate II.
    
  }
  
  TypeRef T = GetTypeForDeclarator(D, S);
  if (T.isNull()) return 0;

  return new FieldDecl(Loc, II, T);
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
  SmallVector<Decl*, 32> RecFields;
  SmallSet<const IdentifierInfo*, 32> FieldIDs;
  
  for (unsigned i = 0; i != NumFields; ++i) {
    FieldDecl *FD = cast_or_null<FieldDecl>(static_cast<Decl*>(Fields[i]));
    if (!FD) continue;  // Already issued a diagnostic.
    
    // Get the type for the field.
    Type *FDTy = FD->getType()->getCanonicalType();
    
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

Sema::DeclTy *Sema::ParseEnumConstant(Scope *S, DeclTy *EnumDeclX,
                                      SourceLocation IdLoc, IdentifierInfo *Id,
                                      SourceLocation EqualLoc, ExprTy *Val) {
  EnumDecl *TheEnumDecl = cast<EnumDecl>(static_cast<Decl*>(EnumDeclX));

  // Verify that there isn't already something declared with this name in this
  // scope.
  if (Decl *PrevDecl = LookupScopedDecl(Id, Decl::IDNS_Ordinary)) {
    if (S->isDeclScope(PrevDecl)) {
      if (isa<EnumConstantDecl>(PrevDecl))
        Diag(IdLoc, diag::err_redefinition_of_enumerator, Id->getName());
      else
        Diag(IdLoc, diag::err_redefinition, Id->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
      return 0;
    }
  }
  
  TypeRef Ty = Context.getTagDeclType(TheEnumDecl);
  EnumConstantDecl *New = new EnumConstantDecl(IdLoc, Id, Ty);
  
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
  
  // Verify that all the values are okay.
  SmallVector<EnumConstantDecl*, 32> Values;
  for (unsigned i = 0; i != NumElements; ++i) {
    EnumConstantDecl *ECD =
      cast_or_null<EnumConstantDecl>(static_cast<Decl*>(Elements[i]));
    if (!ECD) continue;  // Already issued a diagnostic.
    
    Values.push_back(ECD);
  }
  
  Enum->defineElements(&Values[0], Values.size());
}

