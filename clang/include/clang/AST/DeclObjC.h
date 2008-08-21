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
#include "clang/Basic/IdentifierTable.h"

namespace clang {
class Expr;
class Stmt;
class FunctionDecl;
class AttributeList;
class ObjCIvarDecl;
class ObjCMethodDecl;
class ObjCProtocolDecl;
class ObjCCategoryDecl;
class ObjCPropertyDecl;
class ObjCPropertyImplDecl;
  
  
/// ObjCList - This is a simple template class used to hold various lists of
/// decls etc, which is heavily used by the ObjC front-end.  This only use case
/// this supports is setting the list all at once and then reading elements out
/// of it.
template <typename T>
class ObjCList {
  /// List is a new[]'d array of pointers to objects that are not owned by this
  /// list.
  T **List;
  unsigned NumElts;
  
  void operator=(const ObjCList &); // DO NOT IMPLEMENT
  ObjCList(const ObjCList&);        // DO NOT IMPLEMENT
public:
  ObjCList() : List(0), NumElts(0) {}
  ~ObjCList() {
    delete[] List;
  }

  void set(T* const* InList, unsigned Elts) {
    assert(List == 0 && "Elements already set!");
    List = new T*[Elts];
    NumElts = Elts;
    memcpy(List, InList, sizeof(T*)*Elts);
  }
  
  typedef T* const * iterator;
  iterator begin() const { return List; }
  iterator end() const { return List+NumElts; }
  
  unsigned size() const { return NumElts; }
  bool empty() const { return NumElts == 0; }
  
  T* operator[](unsigned idx) const {
    assert(idx < NumElts && "Invalid access");
    return List[idx];
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
class ObjCMethodDecl : public Decl, public DeclContext {
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
  
  // Context this method is declared in.
  NamedDecl *MethodContext;
  
  // A unigue name for this method.
  Selector SelName;
  
  // Type of this method.
  QualType MethodDeclType;
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this Method.  This is null if there are no formals.  
  ParmVarDecl **ParamInfo;
  unsigned NumMethodParams;
  
  /// List of attributes for this method declaration.
  SourceLocation EndLoc; // the location of the ';' or '{'.
  
  // The following are only used for method definitions, null otherwise.
  // FIXME: space savings opportunity, consider a sub-class.
  Stmt *Body;
  // Decls for implicit parameters
  ImplicitParamDecl *SelfDecl;
  ImplicitParamDecl *CmdDecl;
  
  ObjCMethodDecl(SourceLocation beginLoc, SourceLocation endLoc,
                 Selector SelInfo, QualType T,
                 Decl *contextDecl,
                 bool isInstance = true,
                 bool isVariadic = false,
                 bool isSynthesized = false,
                 ImplementationControl impControl = None)
  : Decl(ObjCMethod, beginLoc),
    DeclContext(ObjCMethod),
    IsInstance(isInstance), IsVariadic(isVariadic),
    IsSynthesized(isSynthesized),
    DeclImplementation(impControl), objcDeclQualifier(OBJC_TQ_None),
    MethodContext(static_cast<NamedDecl*>(contextDecl)),
    SelName(SelInfo), MethodDeclType(T), 
    ParamInfo(0), NumMethodParams(0), 
    EndLoc(endLoc), Body(0), SelfDecl(0), CmdDecl(0) {}

  virtual ~ObjCMethodDecl();
  
public:
  
  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);

  static ObjCMethodDecl *Create(ASTContext &C,
                                SourceLocation beginLoc, 
                                SourceLocation endLoc, Selector SelInfo,
                                QualType T, Decl *contextDecl,
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
  
  NamedDecl *getMethodContext() const { return MethodContext; }
  
  ObjCInterfaceDecl *getClassInterface();
  const ObjCInterfaceDecl *getClassInterface() const {
    return const_cast<ObjCMethodDecl*>(this)->getClassInterface();
  }
  
  Selector getSelector() const { return SelName; }
  unsigned getSynthesizedMethodSize() const;
  QualType getResultType() const { return MethodDeclType; }
  
  // Iterator access to formal parameters.
  unsigned param_size() const { return NumMethodParams; }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  param_iterator param_begin() { return ParamInfo; }
  param_iterator param_end() { return ParamInfo+param_size(); }
  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const { return ParamInfo+param_size(); }
  
  unsigned getNumParams() const { return NumMethodParams; }
  ParmVarDecl *getParamDecl(unsigned i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }  
  void setParamDecl(int i, ParmVarDecl *pDecl) {
    ParamInfo[i] = pDecl;
  }  
  void setMethodParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  ImplicitParamDecl * getSelfDecl() const { return SelfDecl; }
  void setSelfDecl(ImplicitParamDecl *decl) { SelfDecl = decl; }
  ImplicitParamDecl * getCmdDecl() const { return CmdDecl; }
  void setCmdDecl(ImplicitParamDecl *decl) { CmdDecl = decl; }
  
  bool isInstance() const { return IsInstance; }
  bool isVariadic() const { return IsVariadic; }
  
  bool isSynthesized() const { return IsSynthesized; }
  
  // Related to protocols declared in  @protocol
  void setDeclImplementation(ImplementationControl ic) { 
    DeclImplementation = ic; 
  }
  ImplementationControl getImplementationControl() const { 
    return ImplementationControl(DeclImplementation); 
  }

  virtual Stmt *getBody() const { return Body; }
  void setBody(Stmt *B) { Body = B; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjCMethod; }
  static bool classof(const ObjCMethodDecl *D) { return true; }
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
class ObjCInterfaceDecl : public NamedDecl, public DeclContext {
  /// TypeForDecl - This indicates the Type object that represents this
  /// TypeDecl.  It is a cache maintained by ASTContext::getObjCInterfaceType
  Type *TypeForDecl;
  friend class ASTContext;
  
  /// Class's super class.
  ObjCInterfaceDecl *SuperClass;
  
  /// Protocols referenced in interface header declaration
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  /// Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjCIvarDecl **Ivars;   // Null if not defined.
  unsigned NumIvars;      // 0 if none.
  
  /// instance methods
  ObjCMethodDecl **InstanceMethods;  // Null if not defined
  unsigned NumInstanceMethods;  // 0 if none.
  
  /// class methods
  ObjCMethodDecl **ClassMethods;  // Null if not defined
  unsigned NumClassMethods;  // 0 if none
  
  /// List of categories defined for this class.
  ObjCCategoryDecl *CategoryList;
    
  /// class properties
  ObjCPropertyDecl **PropertyDecl;  // Null if no property
  unsigned NumPropertyDecl;  // 0 if none.
  
  bool ForwardDecl:1; // declared with @class.
  bool InternalInterface:1; // true - no @interface for @implementation
  
  SourceLocation ClassLoc; // location of the class identifier.
  SourceLocation SuperClassLoc; // location of the super class identifier.
  SourceLocation EndLoc; // marks the '>', '}', or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.

  ObjCInterfaceDecl(SourceLocation atLoc, IdentifierInfo *Id,
                    SourceLocation CLoc, bool FD, bool isInternal)
    : NamedDecl(ObjCInterface, atLoc, Id), DeclContext(ObjCInterface),
      TypeForDecl(0), SuperClass(0),
      Ivars(0), NumIvars(0),
      InstanceMethods(0), NumInstanceMethods(0), 
      ClassMethods(0), NumClassMethods(0),
      CategoryList(0), PropertyDecl(0), NumPropertyDecl(0),
      ForwardDecl(FD), InternalInterface(isInternal),
      ClassLoc(CLoc) {
      }
  
  virtual ~ObjCInterfaceDecl();
  
public:

  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);

  static ObjCInterfaceDecl *Create(ASTContext &C,
                                   SourceLocation atLoc,
                                   IdentifierInfo *Id, 
                                   SourceLocation ClassLoc = SourceLocation(),
                                   bool ForwardDecl = false,
                                   bool isInternal = false);
  const ObjCList<ObjCProtocolDecl> &getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  
  ObjCPropertyDecl *FindPropertyDeclaration(IdentifierInfo *PropertyId) const;
  ObjCCategoryDecl *FindCategoryDeclaration(IdentifierInfo *CategoryId) const;
  ObjCIvarDecl *FindIvarDeclaration(IdentifierInfo *IvarId) const;

  typedef ObjCList<ObjCProtocolDecl>::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  
  typedef ObjCIvarDecl * const *ivar_iterator;
  ivar_iterator ivar_begin() const { return Ivars; }
  ivar_iterator ivar_end() const { return Ivars + ivar_size();}
  unsigned ivar_size() const { return NumIvars; }
  bool ivar_empty() const { return NumIvars == 0; }
  
  unsigned getNumInstanceMethods() const { return NumInstanceMethods; }
  unsigned getNumClassMethods() const { return NumClassMethods; }
  
  typedef ObjCMethodDecl * const * instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods; }
  instmeth_iterator instmeth_end() const {
    return InstanceMethods+NumInstanceMethods;
  }
  
  typedef ObjCMethodDecl * const * classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods; }
  classmeth_iterator classmeth_end() const {
    return ClassMethods+NumClassMethods;
  }

  /// addReferencedProtocols - Set the list of protocols that this interface
  /// implements.
  void addReferencedProtocols(ObjCProtocolDecl *const*List, unsigned NumRPs) {
    ReferencedProtocols.set(List, NumRPs);
  }
   
  void addInstanceVariablesToClass(ObjCIvarDecl **ivars, unsigned numIvars,
                                   SourceLocation RBracLoc);

  void addMethods(ObjCMethodDecl **insMethods, unsigned numInsMembers,
                  ObjCMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEnd);
  
  void addProperties(ObjCPropertyDecl **Properties, unsigned NumProperties);
  
  void mergeProperties(ObjCPropertyDecl **Properties, unsigned NumProperties);
  
  void addPropertyMethods(ASTContext &Context,
                          ObjCPropertyDecl* Property,
                          llvm::SmallVector<ObjCMethodDecl*, 32> &insMethods);
  
  typedef ObjCPropertyDecl * const * classprop_iterator;
  classprop_iterator classprop_begin() const { return PropertyDecl; }
  classprop_iterator classprop_end() const {
    return PropertyDecl+NumPropertyDecl;
  }
  
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

                                                                           
  // Get the local instance method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector Sel) {
    for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); 
         I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
    return 0;
  }
  // Get the local class method declared in this interface.
  ObjCMethodDecl *getClassMethod(Selector Sel) {
    for (classmeth_iterator I = classmeth_begin(), E = classmeth_end(); 
         I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
    return 0;
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
  
  // We also need to record the @end location.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }
  
  unsigned getNumPropertyDecl() const { return NumPropertyDecl; }
  
  ObjCPropertyDecl * const * getPropertyDecl() const { return PropertyDecl; }
  ObjCPropertyDecl **getPropertyDecl() { return PropertyDecl; }

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
  ObjCIvarDecl(SourceLocation L, IdentifierInfo *Id, QualType T,
               AccessControl ac, Expr *BW)
    : FieldDecl(ObjCIvar, L, Id, T, BW), DeclAccess(ac) {}
  
public:
  static ObjCIvarDecl *Create(ASTContext &C, SourceLocation L,
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
  ObjCAtDefsFieldDecl(SourceLocation L, IdentifierInfo *Id,
                      QualType T, Expr *BW)
  : FieldDecl(ObjCAtDefsField, L, Id, T, BW) {}
  
public:
  static ObjCAtDefsFieldDecl *Create(ASTContext &C, SourceLocation L,
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
class ObjCProtocolDecl : public NamedDecl {
  /// Referenced protocols
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  /// protocol instance methods
  ObjCMethodDecl **InstanceMethods;  // Null if not defined
  unsigned NumInstanceMethods;  // 0 if none

  /// protocol class methods
  ObjCMethodDecl **ClassMethods;  // Null if not defined
  unsigned NumClassMethods;  // 0 if none
  
  /// protocol properties
  ObjCPropertyDecl **PropertyDecl;  // Null if no property
  unsigned NumPropertyDecl;  // 0 if none
  
  bool isForwardProtoDecl; // declared with @protocol.
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
  
  ObjCProtocolDecl(SourceLocation L, IdentifierInfo *Id)
    : NamedDecl(ObjCProtocol, L, Id), 
      InstanceMethods(0), NumInstanceMethods(0), 
      ClassMethods(0), NumClassMethods(0),
      PropertyDecl(0), NumPropertyDecl(0),
      isForwardProtoDecl(true) {
  }
  
  virtual ~ObjCProtocolDecl();
  
public:
    
  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);
  
  static ObjCProtocolDecl *Create(ASTContext &C, SourceLocation L,
                                  IdentifierInfo *Id);

  void addMethods(ObjCMethodDecl **insMethods, unsigned numInsMembers,
                  ObjCMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
  
  const ObjCList<ObjCProtocolDecl> &getReferencedProtocols() const { 
    return ReferencedProtocols;
  }
  typedef ObjCList<ObjCProtocolDecl>::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  
  /// addReferencedProtocols - Set the list of protocols that this interface
  /// implements.
  void addReferencedProtocols(ObjCProtocolDecl *const*List, unsigned NumRPs) {
    ReferencedProtocols.set(List, NumRPs);
  }
  
  unsigned getNumInstanceMethods() const { return NumInstanceMethods; }
  unsigned getNumClassMethods() const { return NumClassMethods; }

  ObjCPropertyDecl *FindPropertyDeclaration(IdentifierInfo *PropertyId) const;
  
  unsigned getNumPropertyDecl() const { return NumPropertyDecl; }
  
  ObjCPropertyDecl * const * getPropertyDecl() const { return PropertyDecl; }
  ObjCPropertyDecl **getPropertyDecl() { return PropertyDecl; }
  
  void addProperties(ObjCPropertyDecl **Properties, unsigned NumProperties);
  
  typedef ObjCPropertyDecl * const * classprop_iterator;
  classprop_iterator classprop_begin() const { return PropertyDecl; }
  classprop_iterator classprop_end() const {
    return PropertyDecl+NumPropertyDecl;
  }

  typedef ObjCMethodDecl * const * instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods; }
  instmeth_iterator instmeth_end() const {
    return InstanceMethods+NumInstanceMethods;
  }
  
  typedef ObjCMethodDecl * const * classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods; }
  classmeth_iterator classmeth_end() const {
    return ClassMethods+NumClassMethods;
  }

  // Get the local instance method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector Sel) {
    for (instmeth_iterator I = instmeth_begin(), E = instmeth_end();
         I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
    return 0;
  }
  // Get the local class method declared in this interface.
  ObjCMethodDecl *getClassMethod(Selector Sel) {
    for (classmeth_iterator I = classmeth_begin(), E = classmeth_end(); 
         I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
    return 0;
  }
  
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
  
  // We also need to record the @end location.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }

  static bool classof(const Decl *D) { return D->getKind() == ObjCProtocol; }
  static bool classof(const ObjCProtocolDecl *D) { return true; }
};
  
/// ObjCClassDecl - Specifies a list of forward class declarations. For example:
///
/// @class NSCursor, NSImage, NSPasteboard, NSWindow;
///
class ObjCClassDecl : public Decl {
  ObjCInterfaceDecl **ForwardDecls;
  unsigned NumForwardDecls;
  
  ObjCClassDecl(SourceLocation L, ObjCInterfaceDecl **Elts, unsigned nElts)
    : Decl(ObjCClass, L) { 
    if (nElts) {
      ForwardDecls = new ObjCInterfaceDecl*[nElts];
      memcpy(ForwardDecls, Elts, nElts*sizeof(ObjCInterfaceDecl*));
    } else {
      ForwardDecls = 0;
    }
    NumForwardDecls = nElts;
  }
  
  virtual ~ObjCClassDecl();
  
public:
  
  /// Destroy - Call destructors and release memory.
  virtual void Destroy(ASTContext& C);
  
  static ObjCClassDecl *Create(ASTContext &C, SourceLocation L,
                               ObjCInterfaceDecl **Elts, unsigned nElts);
  
  void setInterfaceDecl(unsigned idx, ObjCInterfaceDecl *OID) {
    assert(idx < NumForwardDecls && "index out of range");
    ForwardDecls[idx] = OID;
  }
  ObjCInterfaceDecl** getForwardDecls() const { return ForwardDecls; }
  int getNumForwardDecls() const { return NumForwardDecls; }
  
  typedef ObjCInterfaceDecl * const * iterator;
  iterator begin() const { return ForwardDecls; }
  iterator end() const { return ForwardDecls+NumForwardDecls; }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCClass; }
  static bool classof(const ObjCClassDecl *D) { return true; }
};

/// ObjCForwardProtocolDecl - Specifies a list of forward protocol declarations.
/// For example:
/// 
/// @protocol NSTextInput, NSChangeSpelling, NSDraggingInfo;
/// 
class ObjCForwardProtocolDecl : public Decl {
  ObjCProtocolDecl **ReferencedProtocols;
  unsigned NumReferencedProtocols;
  
  ObjCForwardProtocolDecl(SourceLocation L,
                          ObjCProtocolDecl **Elts, unsigned nElts)
  : Decl(ObjCForwardProtocol, L) { 
    NumReferencedProtocols = nElts;
    if (nElts) {
      ReferencedProtocols = new ObjCProtocolDecl*[nElts];
      memcpy(ReferencedProtocols, Elts, nElts*sizeof(ObjCProtocolDecl*));
    } else {
      ReferencedProtocols = 0;
    }
  }
  
  virtual ~ObjCForwardProtocolDecl();
  
public:
  static ObjCForwardProtocolDecl *Create(ASTContext &C, SourceLocation L, 
                                         ObjCProtocolDecl **Elts, unsigned Num);

  
  void setForwardProtocolDecl(unsigned idx, ObjCProtocolDecl *OID) {
    assert(idx < NumReferencedProtocols && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  unsigned getNumForwardDecls() const { return NumReferencedProtocols; }
  
  ObjCProtocolDecl *getForwardProtocolDecl(unsigned idx) {
    assert(idx < NumReferencedProtocols && "index out of range");
    return ReferencedProtocols[idx];
  }
  const ObjCProtocolDecl *getForwardProtocolDecl(unsigned idx) const {
    assert(idx < NumReferencedProtocols && "index out of range");
    return ReferencedProtocols[idx];
  }
  
  typedef ObjCProtocolDecl * const * iterator;
  iterator begin() const { return ReferencedProtocols; }
  iterator end() const { return ReferencedProtocols+NumReferencedProtocols; }
  
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
class ObjCCategoryDecl : public NamedDecl {
  /// Interface belonging to this category
  ObjCInterfaceDecl *ClassInterface;
  
  /// referenced protocols in this category.
  ObjCList<ObjCProtocolDecl> ReferencedProtocols;
  
  /// category instance methods
  ObjCMethodDecl **InstanceMethods;  // Null if not defined
  unsigned NumInstanceMethods;  // 0 if none

  /// category class methods
  ObjCMethodDecl **ClassMethods;  // Null if not defined
  unsigned NumClassMethods;  // 0 if not defined
  
  /// Next category belonging to this class
  ObjCCategoryDecl *NextClassCategory;
  
  /// category properties
  ObjCPropertyDecl **PropertyDecl;  // Null if no property
  unsigned NumPropertyDecl;  // 0 if none  
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
  
  ObjCCategoryDecl(SourceLocation L, IdentifierInfo *Id)
    : NamedDecl(ObjCCategory, L, Id),
      ClassInterface(0),
      InstanceMethods(0), NumInstanceMethods(0),
      ClassMethods(0), NumClassMethods(0),
      NextClassCategory(0), PropertyDecl(0),  NumPropertyDecl(0) {
  }
public:
  
  static ObjCCategoryDecl *Create(ASTContext &C,
                                  SourceLocation L, IdentifierInfo *Id);
  
  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  void setClassInterface(ObjCInterfaceDecl *IDecl) { ClassInterface = IDecl; }
  
  /// addReferencedProtocols - Set the list of protocols that this interface
  /// implements.
  void addReferencedProtocols(ObjCProtocolDecl *const*List, unsigned NumRPs) {
    ReferencedProtocols.set(List, NumRPs);
  }
  
  const ObjCList<ObjCProtocolDecl> &getReferencedProtocols() const { 
    return ReferencedProtocols;
  }
  
  typedef ObjCProtocolDecl * const * protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  
  
  unsigned getNumInstanceMethods() const { return NumInstanceMethods; }
  unsigned getNumClassMethods() const { return NumClassMethods; }

  unsigned getNumPropertyDecl() const { return NumPropertyDecl; }
  
  ObjCPropertyDecl * const * getPropertyDecl() const { return PropertyDecl; }
  
  void addProperties(ObjCPropertyDecl **Properties, unsigned NumProperties);
  
  ObjCPropertyDecl *FindPropertyDeclaration(IdentifierInfo *PropertyId) const;
  
  typedef ObjCPropertyDecl * const * classprop_iterator;
  classprop_iterator classprop_begin() const { return PropertyDecl; }
  classprop_iterator classprop_end() const {
    return PropertyDecl+NumPropertyDecl;
  }
  
  typedef ObjCMethodDecl * const * instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods; }
  instmeth_iterator instmeth_end() const {
    return InstanceMethods+NumInstanceMethods;
  }
  
  typedef ObjCMethodDecl * const * classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods; }
  classmeth_iterator classmeth_end() const {
    return ClassMethods+NumClassMethods;
  }

  // Get the local instance method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector Sel) {
    for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); 
         I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
    return 0;
  }
  // Get the local class method declared in this interface.
  ObjCMethodDecl *getClassMethod(Selector Sel) {
    for (classmeth_iterator I = classmeth_begin(), E = classmeth_end(); 
         I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
    return 0;
  }
  
  void addMethods(ObjCMethodDecl **insMethods, unsigned numInsMembers,
                  ObjCMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
  
  ObjCCategoryDecl *getNextClassCategory() const { return NextClassCategory; }
  void insertNextClassCategory() {
    NextClassCategory = ClassInterface->getCategoryList();
    ClassInterface->setCategoryList(this);
  }
  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); } // '@'interface
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  // We also need to record the @end location.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjCCategory; }
  static bool classof(const ObjCCategoryDecl *D) { return true; }
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
class ObjCCategoryImplDecl : public NamedDecl {
  /// Class interface for this category implementation
  ObjCInterfaceDecl *ClassInterface;

  /// implemented instance methods
  llvm::SmallVector<ObjCMethodDecl*, 32> InstanceMethods;
  
  /// implemented class methods
  llvm::SmallVector<ObjCMethodDecl*, 32> ClassMethods;
  
  /// Property Implementations in this category
  llvm::SmallVector<ObjCPropertyImplDecl*, 8> PropertyImplementations;

  SourceLocation EndLoc;  

  ObjCCategoryImplDecl(SourceLocation L, IdentifierInfo *Id,
                       ObjCInterfaceDecl *classInterface)
    : NamedDecl(ObjCCategoryImpl, L, Id), ClassInterface(classInterface) {}
public:
  static ObjCCategoryImplDecl *Create(ASTContext &C,
                                      SourceLocation L, IdentifierInfo *Id,
                                      ObjCInterfaceDecl *classInterface);
        
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  
  unsigned getNumInstanceMethods() const { return InstanceMethods.size(); }
  unsigned getNumClassMethods() const { return ClassMethods.size(); }

  void addInstanceMethod(ObjCMethodDecl *method) {
    InstanceMethods.push_back(method);
  }
  void addClassMethod(ObjCMethodDecl *method) {
    ClassMethods.push_back(method);
  }   
  // Get the instance method definition for this implementation.
  ObjCMethodDecl *getInstanceMethod(Selector Sel);
  
  // Get the class method definition for this implementation.
  ObjCMethodDecl *getClassMethod(Selector Sel);
  
  void addPropertyImplementation(ObjCPropertyImplDecl *property) {
    PropertyImplementations.push_back(property);
  }

  unsigned getNumPropertyImplementations() const
  { return PropertyImplementations.size(); }
  
  
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
    
  static bool classof(const Decl *D) { return D->getKind() == ObjCCategoryImpl;}
  static bool classof(const ObjCCategoryImplDecl *D) { return true; }
};

/// ObjCImplementationDecl - Represents a class definition - this is where
/// method definitions are specified. For example:
///
/// @implementation MyClass
/// - (void)myMethod { /* do something */ }
/// @end
///
/// Typically, instance variables are specified in the class interface, 
/// *not* in the implemenentation. Nevertheless (for legacy reasons), we
/// allow instance variables to be specified in the implementation. When
/// specified, they need to be *identical* to the interface. Now that we
/// have support for non-fragile ivars in ObjC 2.0, we can consider removing
/// the legacy semantics and allow developers to move private ivar declarations
/// from the class interface to the class implementation (but I digress:-)
///
class ObjCImplementationDecl : public NamedDecl {
  /// Class interface for this implementation
  ObjCInterfaceDecl *ClassInterface;
  
  /// Implementation Class's super class.
  ObjCInterfaceDecl *SuperClass;
    
  /// Optional Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjCIvarDecl **Ivars;   // Null if not specified
  unsigned NumIvars;      // 0 if none.

  /// implemented instance methods
  llvm::SmallVector<ObjCMethodDecl*, 32> InstanceMethods;
  
  /// implemented class methods
  llvm::SmallVector<ObjCMethodDecl*, 32> ClassMethods;

  /// Propertys' being implemented
  llvm::SmallVector<ObjCPropertyImplDecl*, 8> PropertyImplementations;
  
  SourceLocation EndLoc;

  ObjCImplementationDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjCInterfaceDecl *classInterface,
                         ObjCInterfaceDecl *superDecl)
    : NamedDecl(ObjCImplementation, L, Id),
      ClassInterface(classInterface), SuperClass(superDecl),
      Ivars(0), NumIvars(0) {}
public:  
  static ObjCImplementationDecl *Create(ASTContext &C,
                                        SourceLocation L, IdentifierInfo *Id,
                                        ObjCInterfaceDecl *classInterface,
                                        ObjCInterfaceDecl *superDecl);
  
  
  void ObjCAddInstanceVariablesToClassImpl(ObjCIvarDecl **ivars, 
                                           unsigned numIvars);
    
  void addInstanceMethod(ObjCMethodDecl *method) {
    InstanceMethods.push_back(method);
  }
  void addClassMethod(ObjCMethodDecl *method) {
    ClassMethods.push_back(method);
  }    
  
  void addPropertyImplementation(ObjCPropertyImplDecl *property) {
    PropertyImplementations.push_back(property);
  } 
  typedef llvm::SmallVector<ObjCPropertyImplDecl*, 8>::const_iterator
  propimpl_iterator;
  propimpl_iterator propimpl_begin() const { 
    return PropertyImplementations.begin(); 
  }
  propimpl_iterator propimpl_end() const { 
    return PropertyImplementations.end(); 
  }
  
  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); }
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  const ObjCInterfaceDecl *getSuperClass() const { return SuperClass; }
  ObjCInterfaceDecl *getSuperClass() { return SuperClass; }
  
  void setSuperClass(ObjCInterfaceDecl * superCls) { SuperClass = superCls; }
  
  unsigned getNumInstanceMethods() const { return InstanceMethods.size(); }
  unsigned getNumClassMethods() const { return ClassMethods.size(); }
  
  unsigned getNumPropertyImplementations() const 
    { return PropertyImplementations.size(); }

  typedef llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator
       instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods.begin(); }
  instmeth_iterator instmeth_end() const { return InstanceMethods.end(); }

  typedef llvm::SmallVector<ObjCMethodDecl*, 32>::const_iterator
    classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods.begin(); }
  classmeth_iterator classmeth_end() const { return ClassMethods.end(); }
  
  // Get the instance method definition for this implementation.
  ObjCMethodDecl *getInstanceMethod(Selector Sel);
  
  // Get the class method definition for this implementation.
  ObjCMethodDecl *getClassMethod(Selector Sel);
  
  typedef ObjCIvarDecl * const *ivar_iterator;
  ivar_iterator ivar_begin() const { return Ivars; }
  ivar_iterator ivar_end() const { return Ivars+NumIvars; }
  unsigned ivar_size() const { return NumIvars; }
  bool ivar_empty() const { return NumIvars == 0; }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCImplementation;
  }
  static bool classof(const ObjCImplementationDecl *D) { return true; }
};

/// ObjCCompatibleAliasDecl - Represents alias of a class. This alias is 
/// declared as @compatibility_alias alias class.
class ObjCCompatibleAliasDecl : public NamedDecl {
  /// Class that this is an alias of.
  ObjCInterfaceDecl *AliasedClass;
  
  ObjCCompatibleAliasDecl(SourceLocation L, IdentifierInfo *Id,
                          ObjCInterfaceDecl* aliasedClass)
    : NamedDecl(ObjCCompatibleAlias, L, Id), AliasedClass(aliasedClass) {}
public:
  static ObjCCompatibleAliasDecl *Create(ASTContext &C,
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

  ObjCPropertyDecl(SourceLocation L, IdentifierInfo *Id, QualType T)
    : NamedDecl(ObjCProperty, L, Id), DeclType(T),
      PropertyAttributes(OBJC_PR_noattr), PropertyImplementation(None),
      GetterName(Selector()), 
      SetterName(Selector()),
      GetterMethodDecl(0), SetterMethodDecl(0) {}
public:
  static ObjCPropertyDecl *Create(ASTContext &C, SourceLocation L, 
                                  IdentifierInfo *Id, QualType T,
                                  PropertyControl propControl = None);
  QualType getType() const { return DeclType; }
  
  PropertyAttributeKind getPropertyAttributes() const {
    return PropertyAttributeKind(PropertyAttributes);
  }
  void setPropertyAttributes(PropertyAttributeKind PRVal) { 
    PropertyAttributes |= PRVal;
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
  enum PropertyImplKind {
    OBJC_PR_IMPL_SYNTHSIZE,
    OBJC_PR_IMPL_DYNAMIC
  };
private:
  SourceLocation AtLoc;   // location of @synthesize or @dynamic
  /// Property declaration being implemented
  ObjCPropertyDecl *PropertyDecl;

  /// Null for @dynamic. Required for @synthesize.
  ObjCIvarDecl *PropertyIvarDecl;

public:
  ObjCPropertyImplDecl(SourceLocation atLoc, SourceLocation L,
                       ObjCPropertyDecl *property, 
                       PropertyImplKind propertyKind, 
                       ObjCIvarDecl *ivarDecl)
  : Decl(ObjCPropertyImpl, L), AtLoc(atLoc), PropertyDecl(property), 
    PropertyIvarDecl(ivarDecl) {
      assert (propertyKind == OBJC_PR_IMPL_DYNAMIC || PropertyIvarDecl);
    }
  
  static ObjCPropertyImplDecl *Create(ASTContext &C, SourceLocation atLoc, 
                                      SourceLocation L, 
                                      ObjCPropertyDecl *property, 
                                      PropertyImplKind propertyKind, 
                                      ObjCIvarDecl *ivarDecl);

  ObjCPropertyDecl *getPropertyDecl() const {
    return PropertyDecl;
  }
  
  PropertyImplKind getPropertyImplementation() const {
    return PropertyDecl ? OBJC_PR_IMPL_SYNTHSIZE : OBJC_PR_IMPL_DYNAMIC;
  }
  
  ObjCIvarDecl *getPropertyIvarDecl() {
    return PropertyIvarDecl;
  }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCPropertyImpl;
  }
  static bool classof(const ObjCPropertyImplDecl *D) { return true; }  
};

}  // end namespace clang
#endif
