//===--- DeclObjC.h - Classes for representing declarations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclObjC interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLOBJC_H
#define LLVM_CLANG_AST_DECLOBJC_H

#include "clang/AST/Decl.h"

namespace clang {
class Expr;
class Stmt;
class FunctionDecl;
class AttributeList;
class RecordDecl;
class ObjCIvarDecl;
class ObjCMethodDecl;
class ObjCProtocolDecl;
class ObjCCategoryDecl;
class ObjCPropertyDecl;
class ObjCPropertyImplDecl;

class ObjCListBase {
  void operator=(const ObjCListBase &);     // DO NOT IMPLEMENT
  ObjCListBase(const ObjCListBase&);        // DO NOT IMPLEMENT
protected:
  /// List is an array of pointers to objects that are not owned by this object.
  void **List;
  unsigned NumElts;

public:
  ObjCListBase() : List(0), NumElts(0) {}
  ~ObjCListBase() {
    assert(List == 0 && "Destroy should have been called before dtor");
  }
  
  void Destroy(ASTContext &Ctx);
  
  unsigned size() const { return NumElts; }
  bool empty() const { return NumElts == 0; }
  
protected:
  void set(void *const* InList, unsigned Elts, ASTContext &Ctx);
};
  
  
/// ObjCList - This is a simple template class used to hold various lists of
/// decls etc, which is heavily used by the ObjC front-end.  This only use case
/// this supports is setting the list all at once and then reading elements out
/// of it.
template <typename T>
class ObjCList : public ObjCListBase {
public:
  void set(T* const* InList, unsigned Elts, ASTContext &Ctx) {
    ObjCListBase::set(reinterpret_cast<void*const*>(InList), Elts, Ctx);
  }
  
  typedef T* const * iterator;
  iterator begin() const { return (iterator)List; }
  iterator end() const { return (iterator)List+NumElts; }
  
  T* operator[](unsigned Idx) const {
    assert(Idx < NumElts && "Invalid access");
    return (T*)List[Idx];
  }
};

  

/// ObjCMethodDecl - Represents an instance or class method declaration.
/// ObjC methods can be declared within 4 contexts: class interfaces,
/// categories, protocols, and class implementations. While C++ member
/// functions leverage C syntax, Objective-C method syntax is modeled after 
/// Smalltalk (using colons to specify argument types/expressions). 
/// Here are some brief examples:
///
/// Setter/getter instance methods:
/// - (void)setMenu:(NSMenu *)menu;
/// - (NSMenu *)menu; 
/// 
/// Instance method that takes 2 NSView arguments:
/// - (void)replaceSubview:(NSView *)oldView with:(NSView *)newView;
///
/// Getter class method:
/// + (NSMenu *)defaultMenu;
///
/// A selector represents a unique name for a method. The selector names for
/// the above methods are setMenu:, menu, replaceSubview:with:, and defaultMenu.
///
class ObjCMethodDecl : public NamedDecl, public DeclContext {
public:
  enum ImplementationControl { None, Required, Optional };
private:
  /// Bitfields must be first fields in this class so they pack with those
  /// declared in class Decl.
  /// instance (true) or class (false) method.
  bool IsInstance : 1;
  bool IsVariadic : 1;
  
  // Synthesized declaration method for a property setter/getter
  bool IsSynthesized : 1;
  
  // NOTE: VC++ treats enums as signed, avoid using ImplementationControl enum
  /// @required/@optional
  unsigned DeclImplementation : 2;
  
  // NOTE: VC++ treats enums as signed, avoid using the ObjCDeclQualifier enum
  /// in, inout, etc.
  unsigned objcDeclQualifier : 6;
  
  // Type of this method.
  QualType MethodDeclType;
  /// ParamInfo - List of pointers to VarDecls for the formal parameters of this
  /// Method.
  ObjCList<ParmVarDecl> ParamInfo;
  
  /// List of attributes for this method declaration.
  SourceLocation EndLoc; // the location of the ';' or '{'.
  
  // The following are only used for method definitions, null otherwise.
  // FIXME: space savings opportunity, consider a sub-class.
  Stmt *Body;

  /// SelfDecl - Decl for the implicit self parameter. This is lazily
  /// constructed by createImplicitParams.
  ImplicitParamDecl *SelfDecl;
  /// CmdDecl - Decl for the implicit _cmd parameter. This is lazily
  /// constructed by createImplicitParams.
  ImplicitParamDecl *CmdDecl;
  
  ObjCMethodDecl(SourceLocation beginLoc, SourceLocation endLoc,
                 Selector SelInfo, QualType T,
                 DeclContext *contextDecl,
                 bool isInstance = true,
                 bool isVariadic = false,
                 bool isSynthesized = false,
                 ImplementationControl impControl = None)
  : NamedDecl(ObjCMethod, contextDecl, beginLoc, SelInfo),
    DeclContext(ObjCMethod),
    IsInstance(isInstance), IsVariadic(isVariadic),
    IsSynthesized(isSynthesized),
    DeclImplementation(impControl), objcDeclQualifier(OBJC_TQ_None),
    MethodDeclType(T), 
    EndLoc(endLoc), Body(0), SelfDecl(0), CmdDecl(0) {}

  virtual ~ObjCMethodDecl() {}
  
public:
  
  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);

  static ObjCMethodDecl *Create(ASTContext &C,
                                SourceLocation beginLoc, 
                                SourceLocation endLoc, Selector SelInfo,
                                QualType T, DeclContext *contextDecl,
                                bool isInstance = true,
                                bool isVariadic = false,
                                bool isSynthesized = false,
                                ImplementationControl impControl = None);
  
  ObjCDeclQualifier getObjCDeclQualifier() const {
    return ObjCDeclQualifier(objcDeclQualifier);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QV) { objcDeclQualifier = QV; }
  
  // Location information, modeled after the Stmt API.
  SourceLocation getLocStart() const { return getLocation(); }
  SourceLocation getLocEnd() const { return EndLoc; }
  SourceRange getSourceRange() const { 
    return SourceRange(getLocation(), EndLoc); 
  }
    
  ObjCInterfaceDecl *getClassInterface();
  const ObjCInterfaceDecl *getClassInterface() const {
    return const_cast<ObjCMethodDecl*>(this)->getClassInterface();
  }
  
  Selector getSelector() const { return getDeclName().getObjCSelector(); }
  unsigned getSynthesizedMethodSize() const;
  QualType getResultType() const { return MethodDeclType; }
  
  // Iterator access to formal parameters.
  unsigned param_size() const { return ParamInfo.size(); }
  typedef ObjCList<ParmVarDecl>::iterator param_iterator;
  param_iterator param_begin() const { return ParamInfo.begin(); }
  param_iterator param_end() const { return ParamInfo.end(); }

  void setMethodParams(ParmVarDecl *const *List, unsigned Num,
                       ASTContext &C) {
    ParamInfo.set(List, Num, C);
  }

  /// createImplicitParams - Used to lazily create the self and cmd
  /// implict parameters. This must be called prior to using getSelfDecl()
  /// or getCmdDecl(). The call is ignored if the implicit paramters
  /// have already been created.
  void createImplicitParams(ASTContext &Context, const ObjCInterfaceDecl *ID);

  ImplicitParamDecl * getSelfDecl() const { return SelfDecl; }
  ImplicitParamDecl * getCmdDecl() const { return CmdDecl; }
  
  bool isInstanceMethod() const { return IsInstance; }
  bool isVariadic() const { return IsVariadic; }
  
  bool isClassMethod() const { return !IsInstance; }

  bool isSynthesized() const { return IsSynthesized; }
  void setIsSynthesized() { IsSynthesized = true; }
  
  // Related to protocols declared in  @protocol
  void setDeclImplementation(ImplementationControl ic) { 
    DeclImplementation = ic; 
  }
  ImplementationControl getImplementationControl() const { 
    return ImplementationControl(DeclImplementation); 
  }

  virtual CompoundStmt *getBody() const { return (CompoundStmt*) Body; }
  void setBody(CompoundStmt *B) { Body = (Stmt*) B; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjCMethod; }
  static bool classof(const ObjCMethodDecl *D) { return true; }
  static DeclContext *castToDeclContext(const ObjCMethodDecl *D) {
    return static_cast<DeclContext *>(const_cast<ObjCMethodDecl*>(D));
  }
  static ObjCMethodDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<ObjCMethodDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// ObjCContainerDecl - Represents a container for method declarations.
/// Current sub-classes are ObjCInterfaceDecl, ObjCCategoryDecl, and
/// ObjCProtocolDecl. 
/// FIXME: Use for ObjC implementation decls.
///
class ObjCContainerDecl : public NamedDecl, public DeclContext {
  SourceLocation AtEndLoc; // marks the end of the method container.
  // FIXME. In the long term, all TU variables declared in class scope belong
  // to class's decl context. This waits till we can establish class's
  // context before processing all decls in the class.
  /// Instance variables in the interface.
  ObjCList<VarDecl> TUVars;
public:

  ObjCContainerDecl(Kind DK, DeclContext *DC, SourceLocation L, 
                    IdentifierInfo *Id)
    : NamedDecl(DK, DC, L, Id), DeclContext(DK) {}

  virtual ~ObjCContainerDecl() {}

  // Iterator access to properties.
  typedef specific_decl_iterator<ObjCPropertyDecl> prop_iterator;
  prop_iterator prop_begin() const { 
    return prop_iterator(decls_begin());
  }
  prop_iterator prop_end() const { 
    return prop_iterator(decls_end());
  }
  
  // Iterator access to instance/class methods.
  typedef specific_decl_iterator<ObjCMethodDecl> method_iterator;
  method_iterator meth_begin() const { 
    return method_iterator(decls_begin());
  }
  method_iterator meth_end() const { 
    return method_iterator(decls_end());
  }

  typedef filtered_decl_iterator<ObjCMethodDecl, 
                                 &ObjCMethodDecl::isInstanceMethod> 
    instmeth_iterator;
  instmeth_iterator instmeth_begin() const {
    return instmeth_iterator(decls_begin());
  }
  instmeth_iterator instmeth_end() const {
    return instmeth_iterator(decls_end());
  }

  typedef filtered_decl_iterator<ObjCMethodDecl, 
                                 &ObjCMethodDecl::isClassMethod> 
    classmeth_iterator;
  classmeth_iterator classmeth_begin() const {
    return classmeth_iterator(decls_begin());
  }
  classmeth_iterator classmeth_end() const {
    return classmeth_iterator(decls_end());
  }

  // Get the local instance/class method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector Sel) const;
  ObjCMethodDecl *getClassMethod(Selector Sel) const;
  ObjCMethodDecl *getMethod(Selector Sel, bool isInstance) const {
    return isInstance ? getInstanceMethod(Sel) : getClassMethod(Sel);
  }
  
  typedef ObjCList<VarDecl>::iterator tuvar_iterator;
  tuvar_iterator tuvar_begin() const { return TUVars.begin(); }
  tuvar_iterator tuvar_end() const { return TUVars.end(); }
  unsigned tuvar_size() const { return TUVars.size(); }
  void setTUVarList(VarDecl * const *List, unsigned Num, ASTContext &C) {
    TUVars.set(List, Num, C);
  }
  
  ObjCPropertyDecl *FindPropertyDeclaration(IdentifierInfo *PropertyId) const;

  // Marks the end of the container.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }
  void setAtEndLoc(SourceLocation L) { AtEndLoc = L; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() >= ObjCContainerFirst && 
           D->getKind() <= ObjCContainerLast;
  }
  static bool classof(const ObjCContainerDecl *D) { return true; }

  static DeclContext *castToDeclContext(const ObjCContainerDecl *D) {
    return static_cast<DeclContext *>(const_cast<ObjCContainerDecl*>(D));
  }
  static ObjCContainerDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<ObjCContainerDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// ObjCInterfaceDecl - Represents an ObjC class declaration. For example:
///
///   // MostPrimitive declares no super class (not particularly useful).
///   @interface MostPrimitive 
///     // no instance variables or methods.
///   @end
///
///   // NSResponder inherits from NSObject & implements NSCoding (a protocol). 
///   @interface NSResponder : NSObject <NSCoding>
///   { // instance variables are represented by ObjCIvarDecl.
///     id nextResponder; // nextResponder instance variable.
///   }
///   - (NSResponder *)nextResponder; // return a pointer to NSResponder.
///   - (void)mouseMoved:(NSEvent *)theEvent; // return void, takes a pointer
///   @end                                    // to an NSEvent.
///
///   Unlike C/C++, forward class declarations are accomplished with @class.
///   Unlike C/C++, @class allows for a list of classes to be forward declared.
///   Unlike C++, ObjC is a single-rooted class model. In Cocoa, classes
///   typically inherit from NSObject (an exception is NSProxy).
///
class ObjCInterfaceDecl : public ObjCContainerDecl {
  /// TypeForDecl - This indicates the Type object that represents this
  /// TypeDecl.  It is a cache maintained by ASTContext::getObjCInterfaceType
  Type *TypeForDecl;
  friend class ASTContext;
  
  /// Class's super class.
  ObjCInterfaceDecl *SuperClass;
  
  /// Protocols referenced in interface header declaration
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  /// Instance variables in the interface.
  ObjCList<ObjCIvarDecl> IVars;
  
  /// List of categories defined for this class.
  /// FIXME: Why is this a linked list??
  ObjCCategoryDecl *CategoryList;
    
  bool ForwardDecl:1; // declared with @class.
  bool InternalInterface:1; // true - no @interface for @implementation
  
  SourceLocation ClassLoc; // location of the class identifier.
  SourceLocation SuperClassLoc; // location of the super class identifier.
  SourceLocation EndLoc; // marks the '>', '}', or identifier.

  ObjCInterfaceDecl(DeclContext *DC, SourceLocation atLoc, IdentifierInfo *Id,
                    SourceLocation CLoc, bool FD, bool isInternal);
  
  virtual ~ObjCInterfaceDecl() {}
  
public:

  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);

  static ObjCInterfaceDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation atLoc,
                                   IdentifierInfo *Id, 
                                   SourceLocation ClassLoc = SourceLocation(),
                                   bool ForwardDecl = false,
                                   bool isInternal = false);
  const ObjCList<ObjCProtocolDecl> &getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  
  ObjCCategoryDecl *FindCategoryDeclaration(IdentifierInfo *CategoryId) const;

  typedef ObjCList<ObjCProtocolDecl>::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  
  typedef ObjCList<ObjCIvarDecl>::iterator ivar_iterator;
  ivar_iterator ivar_begin() const { return IVars.begin(); }
  ivar_iterator ivar_end() const { return IVars.end(); }
  unsigned ivar_size() const { return IVars.size(); }
  bool ivar_empty() const { return IVars.empty(); }
    
  /// setProtocolList - Set the list of protocols that this interface
  /// implements.
  void setProtocolList(ObjCProtocolDecl *const* List, unsigned Num,
                       ASTContext &C) {
    ReferencedProtocols.set(List, Num, C);
  }
   
  void setIVarList(ObjCIvarDecl * const *List, unsigned Num, ASTContext &C) {
    IVars.set(List, Num, C);
  }
  FieldDecl *lookupFieldDeclForIvar(ASTContext &Context, 
                                    const ObjCIvarDecl *ivar);

  bool isForwardDecl() const { return ForwardDecl; }
  void setForwardDecl(bool val) { ForwardDecl = val; }
  
  ObjCInterfaceDecl *getSuperClass() const { return SuperClass; }
  void setSuperClass(ObjCInterfaceDecl * superCls) { SuperClass = superCls; }
  
  ObjCCategoryDecl* getCategoryList() const { return CategoryList; }
  void setCategoryList(ObjCCategoryDecl *category) { 
    CategoryList = category;
  }
  
  /// isSuperClassOf - Return true if this class is the specified class or is a
  /// super class of the specified interface class.
  bool isSuperClassOf(const ObjCInterfaceDecl *I) const {
    // If RHS is derived from LHS it is OK; else it is not OK.
    while (I != NULL) {
      if (this == I)
        return true;
      I = I->getSuperClass();
    }
    return false;
  }
  
  ObjCIvarDecl *lookupInstanceVariable(IdentifierInfo *IVarName,
                                       ObjCInterfaceDecl *&ClassDeclared);
  ObjCIvarDecl *lookupInstanceVariable(IdentifierInfo *IVarName) {
    ObjCInterfaceDecl *ClassDeclared;
    return lookupInstanceVariable(IVarName, ClassDeclared);
  }

  // Lookup a method. First, we search locally. If a method isn't
  // found, we search referenced protocols and class categories.
  ObjCMethodDecl *lookupInstanceMethod(Selector Sel);
  ObjCMethodDecl *lookupClassMethod(Selector Sel);

  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); } // '@'interface
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  SourceLocation getClassLoc() const { return ClassLoc; }
  void setSuperClassLoc(SourceLocation Loc) { SuperClassLoc = Loc; }
  SourceLocation getSuperClassLoc() const { return SuperClassLoc; }
    
  /// ImplicitInterfaceDecl - check that this is an implicitely declared
  /// ObjCInterfaceDecl node. This is for legacy objective-c @implementation
  /// declaration without an @interface declaration.
  bool ImplicitInterfaceDecl() const { return InternalInterface; }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCInterface; }
  static bool classof(const ObjCInterfaceDecl *D) { return true; }
};

/// ObjCIvarDecl - Represents an ObjC instance variable. In general, ObjC
/// instance variables are identical to C. The only exception is Objective-C
/// supports C++ style access control. For example:
///
///   @interface IvarExample : NSObject
///   {
///     id defaultToProtected;
///   @public:
///     id canBePublic; // same as C++.
///   @protected:
///     id canBeProtected; // same as C++.
///   @package:
///     id canBePackage; // framework visibility (not available in C++).
///   }
///
class ObjCIvarDecl : public FieldDecl {
public:
  enum AccessControl {
    None, Private, Protected, Public, Package
  };
  
private:
  ObjCIvarDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
               QualType T, AccessControl ac, Expr *BW)
    : FieldDecl(ObjCIvar, DC, L, Id, T, BW, /*Mutable=*/false), 
      DeclAccess(ac) {}
  
public:
  static ObjCIvarDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              IdentifierInfo *Id, QualType T,
                              AccessControl ac, Expr *BW = NULL);
    
  void setAccessControl(AccessControl ac) { DeclAccess = ac; }

  AccessControl getAccessControl() const { return AccessControl(DeclAccess); }

  AccessControl getCanonicalAccessControl() const {
    return DeclAccess == None ? Protected : AccessControl(DeclAccess);
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjCIvar; }
  static bool classof(const ObjCIvarDecl *D) { return true; }
private:
  // NOTE: VC++ treats enums as signed, avoid using the AccessControl enum
  unsigned DeclAccess : 3;
};

  
/// ObjCAtDefsFieldDecl - Represents a field declaration created by an
///  @defs(...).
class ObjCAtDefsFieldDecl : public FieldDecl {
private:
  ObjCAtDefsFieldDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
                      QualType T, Expr *BW)
    : FieldDecl(ObjCAtDefsField, DC, L, Id, T, BW, /*Mutable=*/false) {}
  
public:
  static ObjCAtDefsFieldDecl *Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L,
                                     IdentifierInfo *Id, QualType T,
                                     Expr *BW);
    
  virtual void Destroy(ASTContext& C);

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjCAtDefsField; }
  static bool classof(const ObjCAtDefsFieldDecl *D) { return true; }
};

/// ObjCProtocolDecl - Represents a protocol declaration. ObjC protocols
/// declare a pure abstract type (i.e no instance variables are permitted). 
/// Protocols orginally drew inspiration from C++ pure virtual functions (a C++ 
/// feature with nice semantics and lousy syntax:-). Here is an example:
///
/// @protocol NSDraggingInfo <refproto1, refproto2>
/// - (NSWindow *)draggingDestinationWindow;
/// - (NSImage *)draggedImage;
/// @end
///
/// This says that NSDraggingInfo requires two methods and requires everything
/// that the two "referenced protocols" 'refproto1' and 'refproto2' require as
/// well.
///
/// @interface ImplementsNSDraggingInfo : NSObject <NSDraggingInfo>
/// @end
///
/// ObjC protocols inspired Java interfaces. Unlike Java, ObjC classes and
/// protocols are in distinct namespaces. For example, Cocoa defines both
/// an NSObject protocol and class (which isn't allowed in Java). As a result, 
/// protocols are referenced using angle brackets as follows:
///
/// id <NSDraggingInfo> anyObjectThatImplementsNSDraggingInfo;
///
class ObjCProtocolDecl : public ObjCContainerDecl {
  /// Referenced protocols
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  bool isForwardProtoDecl; // declared with @protocol.
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
  
  ObjCProtocolDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id)
    : ObjCContainerDecl(ObjCProtocol, DC, L, Id), 
      isForwardProtoDecl(true) {
  }
  
  virtual ~ObjCProtocolDecl() {}
  
public:
  static ObjCProtocolDecl *Create(ASTContext &C, DeclContext *DC, 
                                  SourceLocation L, IdentifierInfo *Id);

  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);
  
  const ObjCList<ObjCProtocolDecl> &getReferencedProtocols() const { 
    return ReferencedProtocols;
  }
  typedef ObjCList<ObjCProtocolDecl>::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  
  /// setProtocolList - Set the list of protocols that this interface
  /// implements.
  void setProtocolList(ObjCProtocolDecl *const*List, unsigned Num,
                       ASTContext &C) {
    ReferencedProtocols.set(List, Num, C);
  }
  
  ObjCProtocolDecl *lookupProtocolNamed(IdentifierInfo *PName);
  
  // Lookup a method. First, we search locally. If a method isn't
  // found, we search referenced protocols and class categories.
  ObjCMethodDecl *lookupInstanceMethod(Selector Sel);
  ObjCMethodDecl *lookupClassMethod(Selector Sel);

  bool isForwardDecl() const { return isForwardProtoDecl; }
  void setForwardDecl(bool val) { isForwardProtoDecl = val; }

  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); } // '@'protocol
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCProtocol; }
  static bool classof(const ObjCProtocolDecl *D) { return true; }
};
  
/// ObjCClassDecl - Specifies a list of forward class declarations. For example:
///
/// @class NSCursor, NSImage, NSPasteboard, NSWindow;
///
class ObjCClassDecl : public Decl {
  ObjCList<ObjCInterfaceDecl> ForwardDecls;
  
  ObjCClassDecl(DeclContext *DC, SourceLocation L, 
                ObjCInterfaceDecl *const *Elts, unsigned nElts, ASTContext &C);
  virtual ~ObjCClassDecl() {}
public:
  
  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);
  
  static ObjCClassDecl *Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                               ObjCInterfaceDecl *const *Elts, unsigned nElts);
  
  typedef ObjCList<ObjCInterfaceDecl>::iterator iterator;
  iterator begin() const { return ForwardDecls.begin(); }
  iterator end() const { return ForwardDecls.end(); }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCClass; }
  static bool classof(const ObjCClassDecl *D) { return true; }
};

/// ObjCForwardProtocolDecl - Specifies a list of forward protocol declarations.
/// For example:
/// 
/// @protocol NSTextInput, NSChangeSpelling, NSDraggingInfo;
/// 
class ObjCForwardProtocolDecl : public Decl {
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  ObjCForwardProtocolDecl(DeclContext *DC, SourceLocation L,
                          ObjCProtocolDecl *const *Elts, unsigned nElts,
                          ASTContext &C);  
  virtual ~ObjCForwardProtocolDecl() {}
  
public:
  static ObjCForwardProtocolDecl *Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L, 
                                         ObjCProtocolDecl *const *Elts,
                                         unsigned Num);

  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);
  
  typedef ObjCList<ObjCProtocolDecl>::iterator iterator;
  iterator begin() const { return ReferencedProtocols.begin(); }
  iterator end() const { return ReferencedProtocols.end(); }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCForwardProtocol;
  }
  static bool classof(const ObjCForwardProtocolDecl *D) { return true; }
};

/// ObjCCategoryDecl - Represents a category declaration. A category allows
/// you to add methods to an existing class (without subclassing or modifying
/// the original class interface or implementation:-). Categories don't allow 
/// you to add instance data. The following example adds "myMethod" to all
/// NSView's within a process:
///
/// @interface NSView (MyViewMethods)
/// - myMethod;
/// @end
///
/// Cateogries also allow you to split the implementation of a class across
/// several files (a feature more naturally supported in C++).
///
/// Categories were originally inspired by dynamic languages such as Common
/// Lisp and Smalltalk.  More traditional class-based languages (C++, Java) 
/// don't support this level of dynamism, which is both powerful and dangerous.
///
class ObjCCategoryDecl : public ObjCContainerDecl {
  /// Interface belonging to this category
  ObjCInterfaceDecl *ClassInterface;
  
  /// referenced protocols in this category.
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  /// Next category belonging to this class.
  /// FIXME: this should not be a singly-linked list.  Move storage elsewhere.
  ObjCCategoryDecl *NextClassCategory;
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  
  ObjCCategoryDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id)
    : ObjCContainerDecl(ObjCCategory, DC, L, Id),
      ClassInterface(0), NextClassCategory(0){
  }
public:
  
  static ObjCCategoryDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation L, IdentifierInfo *Id);
  
  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  void setClassInterface(ObjCInterfaceDecl *IDecl) { ClassInterface = IDecl; }
  
  /// setProtocolList - Set the list of protocols that this interface
  /// implements.
  void setProtocolList(ObjCProtocolDecl *const*List, unsigned Num,
                              ASTContext &C) {
    ReferencedProtocols.set(List, Num, C);
  }
  
  const ObjCList<ObjCProtocolDecl> &getReferencedProtocols() const { 
    return ReferencedProtocols;
  }
  
  typedef ObjCList<ObjCProtocolDecl>::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  
  ObjCCategoryDecl *getNextClassCategory() const { return NextClassCategory; }
  void insertNextClassCategory() {
    NextClassCategory = ClassInterface->getCategoryList();
    ClassInterface->setCategoryList(this);
  }
  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); } // '@'interface
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCCategory; }
  static bool classof(const ObjCCategoryDecl *D) { return true; }
};

class ObjCImplDecl : public Decl, public DeclContext {
  /// Class interface for this category implementation
  ObjCInterfaceDecl *ClassInterface;
  
  /// implemented instance methods
  llvm::SmallVector<ObjCMethodDecl*, 16> InstanceMethods;
  
  /// implemented class methods
  llvm::SmallVector<ObjCMethodDecl*, 16> ClassMethods;
  
  /// Property Implementations in this category
  llvm::SmallVector<ObjCPropertyImplDecl*, 8> PropertyImplementations;
  
  SourceLocation EndLoc;  
  
protected:
  ObjCImplDecl(Kind DK, DeclContext *DC, SourceLocation L,
               ObjCInterfaceDecl *classInterface)
      : Decl(DK, DC, L), DeclContext(DK),
        ClassInterface(classInterface) {}
  
public:
  virtual ~ObjCImplDecl() {}
  
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  
  void addInstanceMethod(ObjCMethodDecl *method) {
    InstanceMethods.push_back(method);
  }
  void addClassMethod(ObjCMethodDecl *method) {
    ClassMethods.push_back(method);
  }   
  
  // Get the local instance/class method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector Sel) const;
  ObjCMethodDecl *getClassMethod(Selector Sel) const;
  ObjCMethodDecl *getMethod(Selector Sel, bool isInstance) const {
    return isInstance ? getInstanceMethod(Sel) : getClassMethod(Sel);
  }
  
  void addPropertyImplementation(ObjCPropertyImplDecl *property) {
    PropertyImplementations.push_back(property);
  }
  
  ObjCPropertyImplDecl *FindPropertyImplDecl(IdentifierInfo *propertyId) const;
  ObjCPropertyImplDecl *FindPropertyImplIvarDecl(IdentifierInfo *ivarId) const;
  
  typedef llvm::SmallVector<ObjCPropertyImplDecl*, 8>::const_iterator
    propimpl_iterator;
  propimpl_iterator propimpl_begin() const { 
    return PropertyImplementations.begin(); 
  }
  propimpl_iterator propimpl_end() const { 
    return PropertyImplementations.end(); 
  }
  
  typedef llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator
    instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods.begin(); }
  instmeth_iterator instmeth_end() const { return InstanceMethods.end(); }
  
  typedef llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator
    classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods.begin(); }
  classmeth_iterator classmeth_end() const { return ClassMethods.end(); }
  
  
  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); }
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
};
  
/// ObjCCategoryImplDecl - An object of this class encapsulates a category 
/// @implementation declaration. If a category class has declaration of a 
/// property, its implementation must be specified in the category's 
/// @implementation declaration. Example:
/// @interface I @end
/// @interface I(CATEGORY)
///    @property int p1, d1;
/// @end
/// @implementation I(CATEGORY)
///  @dynamic p1,d1;
/// @end
///
/// ObjCCategoryImplDecl
class ObjCCategoryImplDecl : public ObjCImplDecl {
  // Category name
  IdentifierInfo *Id;

  ObjCCategoryImplDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
                       ObjCInterfaceDecl *classInterface)
    : ObjCImplDecl(ObjCCategoryImpl, DC, L, classInterface), Id(Id) {}
public:
  static ObjCCategoryImplDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L, IdentifierInfo *Id,
                                      ObjCInterfaceDecl *classInterface);
        
  /// getIdentifier - Get the identifier that names the class
  /// interface associated with this implementation.
  IdentifierInfo *getIdentifier() const { 
    return Id; 
  }
  
  /// getNameAsCString - Get the name of identifier for the class
  /// interface associated with this implementation as a C string
  /// (const char*).
  const char *getNameAsCString() const {
    return Id ? Id->getName() : "";
  }
  
  /// @brief Get the name of the class associated with this interface.
  std::string getNameAsString() const {
    return Id ? Id->getName() : "";
  }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCCategoryImpl;}
  static bool classof(const ObjCCategoryImplDecl *D) { return true; }
  static DeclContext *castToDeclContext(const ObjCCategoryImplDecl *D) {
    return static_cast<DeclContext *>(const_cast<ObjCCategoryImplDecl*>(D));
  }
  static ObjCCategoryImplDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<ObjCCategoryImplDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// ObjCImplementationDecl - Represents a class definition - this is where
/// method definitions are specified. For example:
///
/// @code
/// @implementation MyClass
/// - (void)myMethod { /* do something */ }
/// @end
/// @endcode
///
/// Typically, instance variables are specified in the class interface, 
/// *not* in the implementation. Nevertheless (for legacy reasons), we
/// allow instance variables to be specified in the implementation.  When
/// specified, they need to be *identical* to the interface.
///
class ObjCImplementationDecl : public ObjCImplDecl {  
  /// Implementation Class's super class.
  ObjCInterfaceDecl *SuperClass;
    
  /// Instance variables declared in the @implementation.
  ObjCList<ObjCIvarDecl> IVars;

  ObjCImplementationDecl(DeclContext *DC, SourceLocation L, 
                         ObjCInterfaceDecl *classInterface,
                         ObjCInterfaceDecl *superDecl)
    : ObjCImplDecl(ObjCImplementation, DC, L, classInterface), 
       SuperClass(superDecl){}
public:  
  static ObjCImplementationDecl *Create(ASTContext &C, DeclContext *DC, 
                                        SourceLocation L, 
                                        ObjCInterfaceDecl *classInterface,
                                        ObjCInterfaceDecl *superDecl);
  
  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);

  void setIVarList(ObjCIvarDecl *const *InArray, unsigned Num, ASTContext &C) {
    IVars.set(InArray, Num, C);
  }
  
  /// getIdentifier - Get the identifier that names the class
  /// interface associated with this implementation.
  IdentifierInfo *getIdentifier() const { 
    return getClassInterface()->getIdentifier(); 
  }

  /// getNameAsCString - Get the name of identifier for the class
  /// interface associated with this implementation as a C string
  /// (const char*).
  const char *getNameAsCString() const {
    assert(getIdentifier() && "Name is not a simple identifier");
    return getIdentifier()->getName();
  }

  /// @brief Get the name of the class associated with this interface.
  std::string getNameAsString() const {
    return getClassInterface()->getNameAsString();
  }

  const ObjCInterfaceDecl *getSuperClass() const { return SuperClass; }
  ObjCInterfaceDecl *getSuperClass() { return SuperClass; }
  
  void setSuperClass(ObjCInterfaceDecl * superCls) { SuperClass = superCls; }
    
  typedef ObjCList<ObjCIvarDecl>::iterator ivar_iterator;
  ivar_iterator ivar_begin() const { return IVars.begin(); }
  ivar_iterator ivar_end() const { return IVars.end(); }
  unsigned ivar_size() const { return IVars.size(); }
  bool ivar_empty() const { return IVars.empty(); }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCImplementation;
  }
  static bool classof(const ObjCImplementationDecl *D) { return true; }
  static DeclContext *castToDeclContext(const ObjCImplementationDecl *D) {
    return static_cast<DeclContext *>(const_cast<ObjCImplementationDecl*>(D));
  }
  static ObjCImplementationDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<ObjCImplementationDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// ObjCCompatibleAliasDecl - Represents alias of a class. This alias is 
/// declared as @compatibility_alias alias class.
class ObjCCompatibleAliasDecl : public NamedDecl {
  /// Class that this is an alias of.
  ObjCInterfaceDecl *AliasedClass;
  
  ObjCCompatibleAliasDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
                          ObjCInterfaceDecl* aliasedClass)
    : NamedDecl(ObjCCompatibleAlias, DC, L, Id), AliasedClass(aliasedClass) {}
public:
  static ObjCCompatibleAliasDecl *Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L, IdentifierInfo *Id,
                                         ObjCInterfaceDecl* aliasedClass);

  const ObjCInterfaceDecl *getClassInterface() const { return AliasedClass; }
  ObjCInterfaceDecl *getClassInterface() { return AliasedClass; }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCCompatibleAlias;
  }
  static bool classof(const ObjCCompatibleAliasDecl *D) { return true; }
  
};

/// ObjCPropertyDecl - Represents one property declaration in an interface.
/// For example:
/// @property (assign, readwrite) int MyProperty;
///
class ObjCPropertyDecl : public NamedDecl {
public:
  enum PropertyAttributeKind {
    OBJC_PR_noattr    = 0x00, 
    OBJC_PR_readonly  = 0x01, 
    OBJC_PR_getter    = 0x02,
    OBJC_PR_assign    = 0x04, 
    OBJC_PR_readwrite = 0x08, 
    OBJC_PR_retain    = 0x10,
    OBJC_PR_copy      = 0x20, 
    OBJC_PR_nonatomic = 0x40,
    OBJC_PR_setter    = 0x80
  };

  enum SetterKind { Assign, Retain, Copy };
  enum PropertyControl { None, Required, Optional };
private:
  QualType DeclType;
  unsigned PropertyAttributes : 8;
  
  // @required/@optional
  unsigned PropertyImplementation : 2;
  
  Selector GetterName;    // getter name of NULL if no getter
  Selector SetterName;    // setter name of NULL if no setter
  
  ObjCMethodDecl *GetterMethodDecl; // Declaration of getter instance method
  ObjCMethodDecl *SetterMethodDecl; // Declaration of setter instance method

  ObjCPropertyDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id, 
                   QualType T)
    : NamedDecl(ObjCProperty, DC, L, Id), DeclType(T),
      PropertyAttributes(OBJC_PR_noattr), PropertyImplementation(None),
      GetterName(Selector()), 
      SetterName(Selector()),
      GetterMethodDecl(0), SetterMethodDecl(0) {}
public:
  static ObjCPropertyDecl *Create(ASTContext &C, DeclContext *DC, 
                                  SourceLocation L, 
                                  IdentifierInfo *Id, QualType T,
                                  PropertyControl propControl = None);
  QualType getType() const { return DeclType; }
  
  PropertyAttributeKind getPropertyAttributes() const {
    return PropertyAttributeKind(PropertyAttributes);
  }
  void setPropertyAttributes(PropertyAttributeKind PRVal) { 
    PropertyAttributes |= PRVal;
  }

 void makeitReadWriteAttribute(void) {
    PropertyAttributes &= ~OBJC_PR_readonly;
    PropertyAttributes |= OBJC_PR_readwrite;
 } 

  // Helper methods for accessing attributes.

  /// isReadOnly - Return true iff the property has a setter.
  bool isReadOnly() const {
    return (PropertyAttributes & OBJC_PR_readonly);
  }

  /// getSetterKind - Return the method used for doing assignment in
  /// the property setter. This is only valid if the property has been
  /// defined to have a setter.
  SetterKind getSetterKind() const {
    if (PropertyAttributes & OBJC_PR_retain)
      return Retain;
    if (PropertyAttributes & OBJC_PR_copy)
      return Copy;
    return Assign;
  }

  Selector getGetterName() const { return GetterName; }
  void setGetterName(Selector Sel) { GetterName = Sel; }
  
  Selector getSetterName() const { return SetterName; }
  void setSetterName(Selector Sel) { SetterName = Sel; }
  
  ObjCMethodDecl *getGetterMethodDecl() const { return GetterMethodDecl; }
  void setGetterMethodDecl(ObjCMethodDecl *gDecl) { GetterMethodDecl = gDecl; }

  ObjCMethodDecl *getSetterMethodDecl() const { return SetterMethodDecl; }
  void setSetterMethodDecl(ObjCMethodDecl *gDecl) { SetterMethodDecl = gDecl; }
  
  // Related to @optional/@required declared in @protocol
  void setPropertyImplementation(PropertyControl pc) {
    PropertyImplementation = pc;
  }
  PropertyControl getPropertyImplementation() const {
    return PropertyControl(PropertyImplementation);
  }  
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCProperty;
  }
  static bool classof(const ObjCPropertyDecl *D) { return true; }
};

/// ObjCPropertyImplDecl - Represents implementation declaration of a property 
/// in a class or category implementation block. For example:
/// @synthesize prop1 = ivar1;
///
class ObjCPropertyImplDecl : public Decl {
public:
  enum Kind {
    Synthesize,
    Dynamic
  };
private:
  SourceLocation AtLoc;   // location of @synthesize or @dynamic
  /// Property declaration being implemented
  ObjCPropertyDecl *PropertyDecl;

  /// Null for @dynamic. Required for @synthesize.
  ObjCIvarDecl *PropertyIvarDecl;

  ObjCPropertyImplDecl(DeclContext *DC, SourceLocation atLoc, SourceLocation L,
                       ObjCPropertyDecl *property, 
                       Kind PK, 
                       ObjCIvarDecl *ivarDecl)
    : Decl(ObjCPropertyImpl, DC, L), AtLoc(atLoc), 
      PropertyDecl(property), PropertyIvarDecl(ivarDecl) {
    assert (PK == Dynamic || PropertyIvarDecl);
  }
  
public:
  static ObjCPropertyImplDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation atLoc, SourceLocation L, 
                                      ObjCPropertyDecl *property, 
                                      Kind PK, 
                                      ObjCIvarDecl *ivarDecl);

  SourceLocation getLocStart() const { return AtLoc; }

  ObjCPropertyDecl *getPropertyDecl() const {
    return PropertyDecl;
  }
  
  Kind getPropertyImplementation() const {
    return PropertyIvarDecl ? Synthesize : Dynamic;
  }
  
  ObjCIvarDecl *getPropertyIvarDecl() const {
    return PropertyIvarDecl;
  }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCPropertyImpl;
  }
  static bool classof(const ObjCPropertyImplDecl *D) { return true; }  
};

}  // end namespace clang
#endif
