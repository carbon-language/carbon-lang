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
class ObjCMethodDecl : public Decl {
public:
  enum ImplementationControl { None, Required, Optional };
private:
  /// Bitfields must be first fields in this class so they pack with those
  /// declared in class Decl.
  /// instance (true) or class (false) method.
  bool IsInstance : 1;
  bool IsVariadic : 1;
  
  /// @required/@optional
  ImplementationControl DeclImplementation : 2;
  
  /// in, inout, etc.
  ObjCDeclQualifier objcDeclQualifier : 6;
  
  // Context this method is declared in.
  NamedDecl *MethodContext;
  
  // A unigue name for this method.
  Selector SelName;
  
  // Type of this method.
  QualType MethodDeclType;
  /// ParamInfo - new[]'d array of pointers to VarDecls for the formal
  /// parameters of this Method.  This is null if there are no formals.  
  ParmVarDecl **ParamInfo;
  int NumMethodParams;  // -1 if no parameters
  
  /// List of attributes for this method declaration.
  AttributeList *MethodAttrs;
  
  SourceLocation EndLoc; // the location of the ';' or '{'.
  
  // The following are only used for method definitions, null otherwise.
  // FIXME: space savings opportunity, consider a sub-class.
  Stmt *Body;
  ParmVarDecl *SelfDecl;
public:
  ObjCMethodDecl(SourceLocation beginLoc, SourceLocation endLoc,
                 Selector SelInfo, QualType T,
                 Decl *contextDecl,
                 ParmVarDecl **paramInfo = 0, int numParams=-1,
                 AttributeList *M = 0, bool isInstance = true,
                 bool isVariadic = false,
                 ImplementationControl impControl = None,
                 Decl *PrevDecl = 0)
  : Decl(ObjCMethod, beginLoc),
    IsInstance(isInstance), IsVariadic(isVariadic),
    DeclImplementation(impControl), objcDeclQualifier(OBJC_TQ_None),
    MethodContext(static_cast<NamedDecl*>(contextDecl)),
    SelName(SelInfo), MethodDeclType(T), 
    ParamInfo(paramInfo), NumMethodParams(numParams),
    MethodAttrs(M), EndLoc(endLoc), Body(0), SelfDecl(0) {}
  virtual ~ObjCMethodDecl();
  
  ObjCDeclQualifier getObjCDeclQualifier() const { return objcDeclQualifier; }
  void setObjCDeclQualifier(ObjCDeclQualifier QV) { objcDeclQualifier = QV; }
  
  // Location information, modeled after the Stmt API.
  SourceLocation getLocStart() const { return getLocation(); }
  SourceLocation getLocEnd() const { return EndLoc; }
  
  NamedDecl *getMethodContext() const { return MethodContext; }
  
  ObjCInterfaceDecl *const getClassInterface() const;
  
  Selector getSelector() const { return SelName; }
  QualType getResultType() const { return MethodDeclType; }
  
  // Iterator access to formal parameters.
  unsigned param_size() const {
    return NumMethodParams == -1 ? 0 : NumMethodParams;
  }
  typedef ParmVarDecl **param_iterator;
  typedef ParmVarDecl * const *param_const_iterator;
  param_iterator param_begin() { return ParamInfo; }
  param_iterator param_end() { return ParamInfo+param_size(); }
  param_const_iterator param_begin() const { return ParamInfo; }
  param_const_iterator param_end() const { return ParamInfo+param_size(); }
  
  int getNumParams() const { return NumMethodParams; }
  ParmVarDecl *getParamDecl(int i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }  
  void setMethodParams(ParmVarDecl **NewParamInfo, unsigned NumParams);
  
  AttributeList *getMethodAttrs() const {return MethodAttrs;}
  bool isInstance() const { return IsInstance; }
  bool isVariadic() const { return IsVariadic; }
  
  // Related to protocols declared in  @protocol
  void setDeclImplementation(ImplementationControl ic) { 
    DeclImplementation = ic; 
  }
  ImplementationControl getImplementationControl() const { 
    return DeclImplementation; 
  }
  Stmt *const getBody() const { return Body; }
  void setBody(Stmt *B) { Body = B; }

  ParmVarDecl *const getSelfDecl() const { return SelfDecl; }
  void setSelfDecl(ParmVarDecl *PVD) { SelfDecl = PVD; }
  
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
class ObjCInterfaceDecl : public TypeDecl {
  
  /// Class's super class.
  ObjCInterfaceDecl *SuperClass;
  
  /// Protocols referenced in interface header declaration
  ObjCProtocolDecl **ReferencedProtocols;  // Null if none
  unsigned NumReferencedProtocols;  // 0 if none
  
  /// Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjCIvarDecl **Ivars;   // Null if not defined.
  int NumIvars;   // -1 if not defined.
  
  /// instance methods
  ObjCMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined
  
  /// class methods
  ObjCMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined
  
  /// List of categories defined for this class.
  ObjCCategoryDecl *CategoryList;
    
  /// class properties
  ObjCPropertyDecl **PropertyDecl;  // Null if no property
  int NumPropertyDecl;  // -1 if no property
  
  bool ForwardDecl:1; // declared with @class.
  bool InternalInterface:1; // true - no @interface for @implementation
  
  SourceLocation EndLoc; // marks the '>', '}', or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
public:
  ObjCInterfaceDecl(SourceLocation atLoc, unsigned numRefProtos,
                    IdentifierInfo *Id, bool FD = false, 
                    bool isInternal = false)
    : TypeDecl(ObjCInterface, atLoc, Id, 0), SuperClass(0),
      ReferencedProtocols(0), NumReferencedProtocols(0), Ivars(0), 
      NumIvars(-1),
      InstanceMethods(0), NumInstanceMethods(-1), 
      ClassMethods(0), NumClassMethods(-1),
      CategoryList(0), PropertyDecl(0), NumPropertyDecl(-1),
      ForwardDecl(FD), InternalInterface(isInternal) {
        AllocIntfRefProtocols(numRefProtos);
      }
  
  // This is necessary when converting a forward declaration to a definition.
  void AllocIntfRefProtocols(unsigned numRefProtos) {
    if (numRefProtos) {
      ReferencedProtocols = new ObjCProtocolDecl*[numRefProtos];
      memset(ReferencedProtocols, '\0',
             numRefProtos*sizeof(ObjCProtocolDecl*));
      NumReferencedProtocols = numRefProtos;
    }
  }
  
  ObjCProtocolDecl **getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  unsigned getNumIntfRefProtocols() const { return NumReferencedProtocols; }
  
  int getNumInstanceVariables() const { return NumIvars; }
  
  typedef ObjCIvarDecl * const *ivar_iterator;
  unsigned ivar_size() const { return NumIvars == -1 ?0 : NumIvars; }
  ivar_iterator ivar_begin() const { return Ivars; }
  ivar_iterator ivar_end() const { return Ivars + ivar_size();}
  
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  int getNumClassMethods() const { return NumClassMethods; }
  
  typedef ObjCMethodDecl * const * instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods; }
  instmeth_iterator instmeth_end() const {
    return InstanceMethods+(NumInstanceMethods == -1 ? 0 : NumInstanceMethods);
  }
  
  typedef ObjCMethodDecl * const * classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods; }
  classmeth_iterator classmeth_end() const {
    return ClassMethods+(NumClassMethods == -1 ? 0 : NumClassMethods);
  }
  
  void addInstanceVariablesToClass(ObjCIvarDecl **ivars, unsigned numIvars,
                                   SourceLocation RBracLoc);

  void addMethods(ObjCMethodDecl **insMethods, unsigned numInsMembers,
                  ObjCMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEnd);
  
  bool isForwardDecl() const { return ForwardDecl; }
  void setForwardDecl(bool val) { ForwardDecl = val; }
  
  void setIntfRefProtocols(unsigned idx, ObjCProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjCInterfaceDecl *getSuperClass() const { return SuperClass; }
  void setSuperClass(ObjCInterfaceDecl * superCls) { SuperClass = superCls; }
  
  ObjCCategoryDecl* getCategoryList() const { return CategoryList; }
  void setCategoryList(ObjCCategoryDecl *category) { 
         CategoryList = category; 
  }
  ObjCIvarDecl *lookupInstanceVariable(IdentifierInfo *ivarName,
                                       ObjCInterfaceDecl *&clsDeclared);
									   
  // Get the local instance method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector &Sel) {
    for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); 
	     I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
	return 0;
  }
  // Get the local class method declared in this interface.
  ObjCMethodDecl *getClassMethod(Selector &Sel) {
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
  
  // We also need to record the @end location.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }
  
  int getNumPropertyDecl() const { return NumPropertyDecl; }
  void setNumPropertyDecl(int num) { NumPropertyDecl = num; }
  
  ObjCPropertyDecl **const getPropertyDecl() const { return PropertyDecl; }
  ObjCPropertyDecl **getPropertyDecl() { return PropertyDecl; }
  void setPropertyDecls(ObjCPropertyDecl **properties) { 
    PropertyDecl = properties; 
  }

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
///     id defaultToPrivate; // same as C++.
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
  ObjCIvarDecl(SourceLocation L, IdentifierInfo *Id, QualType T) 
    : FieldDecl(ObjCIvar, L, Id, T) {}
    
  enum AccessControl {
    None, Private, Protected, Public, Package
  };
  void setAccessControl(AccessControl ac) { DeclAccess = ac; }
  AccessControl getAccessControl() const { return DeclAccess; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjCIvar; }
  static bool classof(const ObjCIvarDecl *D) { return true; }
private:
  AccessControl DeclAccess : 3;
};


/// ObjCProtocolDecl - Represents a protocol declaration. ObjC protocols
/// declare a pure abstract type (i.e no instance variables are permitted). 
/// Protocols orginally drew inspiration from C++ pure virtual functions (a C++ 
/// feature with nice semantics and lousy syntax:-). Here is an example:
///
/// @protocol NSDraggingInfo
/// - (NSWindow *)draggingDestinationWindow;
/// - (NSImage *)draggedImage;
/// @end
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
  /// referenced protocols
  ObjCProtocolDecl **ReferencedProtocols;  // Null if none
  unsigned NumReferencedProtocols;  // 0 if none
  
  /// protocol instance methods
  ObjCMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined

  /// protocol class methods
  ObjCMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined

  bool isForwardProtoDecl; // declared with @protocol.
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
public:
  ObjCProtocolDecl(SourceLocation L, unsigned numRefProtos,
                   IdentifierInfo *Id, bool FD = false)
    : NamedDecl(ObjCProtocol, L, Id), 
      ReferencedProtocols(0), NumReferencedProtocols(0),
      InstanceMethods(0), NumInstanceMethods(-1), 
      ClassMethods(0), NumClassMethods(-1),
      isForwardProtoDecl(FD) {
        AllocReferencedProtocols(numRefProtos);
      }
  void AllocReferencedProtocols(unsigned numRefProtos) {
    if (numRefProtos) {
      ReferencedProtocols = new ObjCProtocolDecl*[numRefProtos];
      memset(ReferencedProtocols, '\0', 
             numRefProtos*sizeof(ObjCProtocolDecl*));
      NumReferencedProtocols = numRefProtos;
    }    
  }
  void addMethods(ObjCMethodDecl **insMethods, unsigned numInsMembers,
                  ObjCMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
  
  void setReferencedProtocols(unsigned idx, ObjCProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjCProtocolDecl** getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  unsigned getNumReferencedProtocols() const { return NumReferencedProtocols; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  int getNumClassMethods() const { return NumClassMethods; }

  typedef ObjCMethodDecl * const * instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods; }
  instmeth_iterator instmeth_end() const {
    return InstanceMethods+(NumInstanceMethods == -1 ? 0 : NumInstanceMethods);
  }
  
  typedef ObjCMethodDecl * const * classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods; }
  classmeth_iterator classmeth_end() const {
    return ClassMethods+(NumClassMethods == -1 ? 0 : NumClassMethods);
  }

  // Get the local instance method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector &Sel) {
    for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); 
	     I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
	return 0;
  }
  // Get the local class method declared in this interface.
  ObjCMethodDecl *getClassMethod(Selector &Sel) {
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
public:
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
  void setInterfaceDecl(unsigned idx, ObjCInterfaceDecl *OID) {
    assert(idx < NumForwardDecls && "index out of range");
    ForwardDecls[idx] = OID;
  }
  ObjCInterfaceDecl** getForwardDecls() const { return ForwardDecls; }
  int getNumForwardDecls() const { return NumForwardDecls; }
  
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
public:
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
/// Lisp and Smalltalk. More traditional class-based languages (C++, Java) 
/// don't support this level of dynamism, which is both powerful and dangerous.
///
class ObjCCategoryDecl : public NamedDecl {
  /// Interface belonging to this category
  ObjCInterfaceDecl *ClassInterface;
  
  /// referenced protocols in this category
  ObjCProtocolDecl **ReferencedProtocols;  // Null if none
  unsigned NumReferencedProtocols;  // 0 if none
  
  /// category instance methods
  ObjCMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined

  /// category class methods
  ObjCMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined
  
  /// Next category belonging to this class
  ObjCCategoryDecl *NextClassCategory;
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
public:
  ObjCCategoryDecl(SourceLocation L, unsigned numRefProtocol,IdentifierInfo *Id)
    : NamedDecl(ObjCCategory, L, Id),
      ClassInterface(0), ReferencedProtocols(0), NumReferencedProtocols(0),
      InstanceMethods(0), NumInstanceMethods(-1),
      ClassMethods(0), NumClassMethods(-1),
      NextClassCategory(0) {
        if (numRefProtocol) {
          ReferencedProtocols = new ObjCProtocolDecl*[numRefProtocol];
          memset(ReferencedProtocols, '\0', 
                 numRefProtocol*sizeof(ObjCProtocolDecl*));
          NumReferencedProtocols = numRefProtocol;
        }
      }

  ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  void setClassInterface(ObjCInterfaceDecl *IDecl) { ClassInterface = IDecl; }
  
  void setCatReferencedProtocols(unsigned idx, ObjCProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjCProtocolDecl **getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  unsigned getNumReferencedProtocols() const { return NumReferencedProtocols; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  int getNumClassMethods() const { return NumClassMethods; }

  typedef ObjCMethodDecl * const * instmeth_iterator;
  instmeth_iterator instmeth_begin() const { return InstanceMethods; }
  instmeth_iterator instmeth_end() const {
    return InstanceMethods+(NumInstanceMethods == -1 ? 0 : NumInstanceMethods);
  }
  
  typedef ObjCMethodDecl * const * classmeth_iterator;
  classmeth_iterator classmeth_begin() const { return ClassMethods; }
  classmeth_iterator classmeth_end() const {
    return ClassMethods+(NumClassMethods == -1 ? 0 : NumClassMethods);
  }

  // Get the local instance method declared in this interface.
  ObjCMethodDecl *getInstanceMethod(Selector &Sel) {
    for (instmeth_iterator I = instmeth_begin(), E = instmeth_end(); 
	     I != E; ++I) {
      if ((*I)->getSelector() == Sel)
        return *I;
    }
	return 0;
  }
  // Get the local class method declared in this interface.
  ObjCMethodDecl *getClassMethod(Selector &Sel) {
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
/// @implementation declaration.
class ObjCCategoryImplDecl : public NamedDecl {
  /// Class interface for this category implementation
  ObjCInterfaceDecl *ClassInterface;

  /// implemented instance methods
  llvm::SmallVector<ObjCMethodDecl*, 32> InstanceMethods;
  
  /// implemented class methods
  llvm::SmallVector<ObjCMethodDecl*, 32> ClassMethods;

  SourceLocation EndLoc;  
public:
  ObjCCategoryImplDecl(SourceLocation L, IdentifierInfo *Id,
                       ObjCInterfaceDecl *classInterface)
    : NamedDecl(ObjCCategoryImpl, L, Id), ClassInterface(classInterface) {}
        
  ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  
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
  /// Class interface for this category implementation
  ObjCInterfaceDecl *ClassInterface;
  
  /// Implementation Class's super class.
  ObjCInterfaceDecl *SuperClass;
    
  /// Optional Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjCIvarDecl **Ivars;   // Null if not specified
  int NumIvars;   // -1 if not defined.

  /// implemented instance methods
  llvm::SmallVector<ObjCMethodDecl*, 32> InstanceMethods;
  
  /// implemented class methods
  llvm::SmallVector<ObjCMethodDecl*, 32> ClassMethods;

  SourceLocation EndLoc;
public:
  ObjCImplementationDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjCInterfaceDecl *classInterface,
                         ObjCInterfaceDecl *superDecl)
    : NamedDecl(ObjCImplementation, L, Id),
      ClassInterface(classInterface), SuperClass(superDecl),
      Ivars(0), NumIvars(-1) {}
  
  void ObjCAddInstanceVariablesToClassImpl(ObjCIvarDecl **ivars, 
                                           unsigned numIvars);
    
  void addInstanceMethod(ObjCMethodDecl *method) {
    InstanceMethods.push_back(method);
  }
  void addClassMethod(ObjCMethodDecl *method) {
    ClassMethods.push_back(method);
  }    
  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); }
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  ObjCInterfaceDecl *getSuperClass() const { return SuperClass; }
  
  void setSuperClass(ObjCInterfaceDecl * superCls) 
         { SuperClass = superCls; }
  
  int getNumInstanceMethods() const { return InstanceMethods.size(); }
  int getNumClassMethods() const { return ClassMethods.size(); }

  int getImplDeclNumIvars() const { return NumIvars; }
  
  
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
  ivar_iterator ivar_end() const {return Ivars+(NumIvars == -1 ? 0 : NumIvars);}
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjCImplementation;
  }
  static bool classof(const ObjCImplementationDecl *D) { return true; }
};

/// ObjCCompatibleAliasDecl - Represents alias of a class. This alias is 
/// declared as @compatibility_alias alias class.
class ObjCCompatibleAliasDecl : public ScopedDecl {
  /// Class that this is an alias of.
  ObjCInterfaceDecl *AliasedClass;
  
public:
  ObjCCompatibleAliasDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjCInterfaceDecl* aliasedClass)
  : ScopedDecl(CompatibleAlias, L, Id, 0),
  AliasedClass(aliasedClass) {}
  
  ObjCInterfaceDecl *getClassInterface() const { return AliasedClass; }
  
  static bool classof(const Decl *D) {
    return D->getKind() == CompatibleAlias;
  }
  static bool classof(const ObjCCompatibleAliasDecl *D) { return true; }
  
};
  
class ObjCPropertyDecl : public Decl {
public:
  enum PropertyAttributeKind { OBJC_PR_noattr = 0x0, 
                       OBJC_PR_readonly = 0x01, 
                       OBJC_PR_getter = 0x02,
                       OBJC_PR_assign = 0x04, 
                       OBJC_PR_readwrite = 0x08, 
                       OBJC_PR_retain = 0x10,
                       OBJC_PR_copy = 0x20, 
                       OBJC_PR_nonatomic = 0x40,
                       OBJC_PR_setter = 0x80 };
private:
  // List of property name declarations
  // FIXME: Property is not an ivar.
  ObjCIvarDecl **PropertyDecls;
  int NumPropertyDecls;
  
  PropertyAttributeKind PropertyAttributes : 8;
  
  IdentifierInfo *GetterName;    // getter name of NULL if no getter
  IdentifierInfo *SetterName;    // setter name of NULL if no setter
  
public:
  ObjCPropertyDecl(SourceLocation L)
  : Decl(PropertyDecl, L),
  PropertyDecls(0), NumPropertyDecls(-1), PropertyAttributes(OBJC_PR_noattr),
  GetterName(0), SetterName(0) {}
  
  ObjCIvarDecl **const getPropertyDecls() const { return PropertyDecls; }
  void setPropertyDecls(ObjCIvarDecl **property) { PropertyDecls = property; }
  
  const int getNumPropertyDecls() const { return NumPropertyDecls; }
  void setNumPropertyDecls(int num) { NumPropertyDecls = num; }
  
  const PropertyAttributeKind getPropertyAttributes() const 
    { return PropertyAttributes; }
  void setPropertyAttributes(PropertyAttributeKind PRVal) { 
    PropertyAttributes = 
    (PropertyAttributeKind) (PropertyAttributes | PRVal);
  }
  
  const IdentifierInfo *getGetterName() const { return GetterName; }
  IdentifierInfo *getGetterName() { return GetterName; }
  void setGetterName(IdentifierInfo *Id) { GetterName = Id; }
  
  const IdentifierInfo *getSetterName() const { return SetterName; }
  IdentifierInfo *getSetterName() { return SetterName; }
  void setSetterName(IdentifierInfo *Id) { SetterName = Id; }
  
  static bool classof(const Decl *D) {
    return D->getKind() == PropertyDecl;
  }
  static bool classof(const ObjCPropertyDecl *D) { return true; }
};

}  // end namespace clang
#endif
