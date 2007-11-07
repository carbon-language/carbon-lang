//===--- DeclObjC.h - Classes for representing declarations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
class ObjcIvarDecl;
class ObjcMethodDecl;
class ObjcProtocolDecl;
class ObjcCategoryDecl;
class ObjcPropertyDecl;

/// ObjcInterfaceDecl - Represents an ObjC class declaration. For example:
///
///   // MostPrimitive declares no super class (not particularly useful).
///   @interface MostPrimitive 
///     // no instance variables or methods.
///   @end
///
///   // NSResponder inherits from NSObject & implements NSCoding (a protocol). 
///   @interface NSResponder : NSObject <NSCoding>
///   { // instance variables are represented by ObjcIvarDecl.
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
class ObjcInterfaceDecl : public TypeDecl {
  
  /// Class's super class.
  ObjcInterfaceDecl *SuperClass;
  
  /// Protocols referenced in interface header declaration
  ObjcProtocolDecl **ReferencedProtocols;  // Null if none
  int NumReferencedProtocols;  // -1 if none
  
  /// Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjcIvarDecl **Ivars;   // Null if not defined.
  int NumIvars;   // -1 if not defined.
  
  /// instance methods
  ObjcMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined
  
  /// class methods
  ObjcMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined
  
  /// List of categories defined for this class.
  ObjcCategoryDecl *CategoryList;
    
  /// class properties
  ObjcPropertyDecl **PropertyDecl;  // Null if no property
  int NumPropertyDecl;  // -1 if no property
  
  bool ForwardDecl:1; // declared with @class.
  bool InternalInterface:1; // true - no @interface for @implementation
  
  SourceLocation EndLoc; // marks the '>', '}', or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
public:
  ObjcInterfaceDecl(SourceLocation atLoc, unsigned numRefProtos,
                    IdentifierInfo *Id, bool FD = false, 
                    bool isInternal = false)
    : TypeDecl(ObjcInterface, atLoc, Id, 0), SuperClass(0),
      ReferencedProtocols(0), NumReferencedProtocols(-1), Ivars(0), 
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
      ReferencedProtocols = new ObjcProtocolDecl*[numRefProtos];
      memset(ReferencedProtocols, '\0',
             numRefProtos*sizeof(ObjcProtocolDecl*));
      NumReferencedProtocols = numRefProtos;
    }
  }
  
  ObjcProtocolDecl **getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  int getNumIntfRefProtocols() const { return NumReferencedProtocols; }
  
  ObjcIvarDecl **getIntfDeclIvars() const { return Ivars; }
  int getIntfDeclNumIvars() const { return NumIvars; }
  
  ObjcMethodDecl** getInstanceMethods() const { return InstanceMethods; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  
  ObjcMethodDecl** getClassMethods() const { return ClassMethods; }
  int getNumClassMethods() const { return NumClassMethods; }
  
  void addInstanceVariablesToClass(ObjcIvarDecl **ivars, unsigned numIvars,
                                   SourceLocation RBracLoc);

  void addMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                  ObjcMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEnd);
  
  bool isForwardDecl() const { return ForwardDecl; }
  void setForwardDecl(bool val) { ForwardDecl = val; }
  
  void setIntfRefProtocols(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjcInterfaceDecl *getSuperClass() const { return SuperClass; }
  void setSuperClass(ObjcInterfaceDecl * superCls) { SuperClass = superCls; }
  
  ObjcCategoryDecl* getCategoryList() const { return CategoryList; }
  void setCategoryList(ObjcCategoryDecl *category) { 
         CategoryList = category; 
  }
  ObjcMethodDecl *lookupInstanceMethod(Selector &Sel);
  ObjcMethodDecl *lookupClassMethod(Selector &Sel);

  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); } // '@'interface
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  // We also need to record the @end location.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }
  
  const int getNumPropertyDecl() const { return NumPropertyDecl; }
  int getNumPropertyDecl() { return NumPropertyDecl; }
  void setNumPropertyDecl(int num) { NumPropertyDecl = num; }
  
  ObjcPropertyDecl **const getPropertyDecl() const { return PropertyDecl; }
  ObjcPropertyDecl **getPropertyDecl() { return PropertyDecl; }
  void setPropertyDecls(ObjcPropertyDecl **properties) { 
    PropertyDecl = properties; 
  }

  /// ImplicitInterfaceDecl - check that this is an implicitely declared
  /// ObjcInterfaceDecl node. This is for legacy objective-c @implementation
  /// declaration without an @interface declaration.
  bool ImplicitInterfaceDecl() const { return InternalInterface; }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjcInterface; }
  static bool classof(const ObjcInterfaceDecl *D) { return true; }
};

/// ObjcIvarDecl - Represents an ObjC instance variable. In general, ObjC
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
class ObjcIvarDecl : public FieldDecl {
public:
  ObjcIvarDecl(SourceLocation L, IdentifierInfo *Id, QualType T) 
    : FieldDecl(ObjcIvar, L, Id, T) {}
    
  enum AccessControl {
    None, Private, Protected, Public, Package
  };
  void setAccessControl(AccessControl ac) { DeclAccess = ac; }
  AccessControl getAccessControl() const { return DeclAccess; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjcIvar; }
  static bool classof(const ObjcIvarDecl *D) { return true; }
private:
  AccessControl DeclAccess : 3;
};

/// ObjcMethodDecl - Represents an instance or class method declaration.
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
class ObjcMethodDecl : public Decl {
public:
  enum ImplementationControl { None, Required, Optional };
private:
  /// Bitfields must be first fields in this class so they pack with those
  /// declared in class Decl.
  /// instance (true) or class (false) method.
  bool IsInstance : 1;
  /// @required/@optional
  ImplementationControl DeclImplementation : 2;
  
  /// in, inout, etc.
  ObjcDeclQualifier objcDeclQualifier : 6;
  
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
public:
  ObjcMethodDecl(SourceLocation beginLoc, SourceLocation endLoc,
                 Selector SelInfo, QualType T,
                 ParmVarDecl **paramInfo = 0, int numParams=-1,
                 AttributeList *M = 0, bool isInstance = true,
                 ImplementationControl impControl = None,
                 Decl *PrevDecl = 0)
    : Decl(ObjcMethod, beginLoc),
      IsInstance(isInstance), DeclImplementation(impControl),
      objcDeclQualifier(OBJC_TQ_None),
      SelName(SelInfo), MethodDeclType(T), 
      ParamInfo(paramInfo), NumMethodParams(numParams),
      MethodAttrs(M), EndLoc(endLoc) {}
  virtual ~ObjcMethodDecl();
  
  ObjcDeclQualifier getObjcDeclQualifier() const { return objcDeclQualifier; }
  void setObjcDeclQualifier(ObjcDeclQualifier QV) { objcDeclQualifier = QV; }
  
  // Location information, modeled after the Stmt API.
  SourceLocation getLocStart() const { return getLocation(); }
  SourceLocation getLocEnd() const { return EndLoc; }
  
  Selector getSelector() const { return SelName; }
  QualType getResultType() const { return MethodDeclType; }

  int getNumParams() const { return NumMethodParams; }
  ParmVarDecl *getParamDecl(int i) const {
    assert(i < getNumParams() && "Illegal param #");
    return ParamInfo[i];
  }  
  void setMethodParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  AttributeList *getMethodAttrs() const {return MethodAttrs;}
  bool isInstance() const { return IsInstance; }
  // Related to protocols declared in  @protocol
  void setDeclImplementation(ImplementationControl ic)
         { DeclImplementation = ic; }
  ImplementationControl  getImplementationControl() const
                           { return DeclImplementation; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return D->getKind() == ObjcMethod; }
  static bool classof(const ObjcMethodDecl *D) { return true; }
};

/// ObjcProtocolDecl - Represents a protocol declaration. ObjC protocols
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
/// Objc protocols inspired Java interfaces. Unlike Java, ObjC classes and
/// protocols are in distinct namespaces. For example, Cocoa defines both
/// an NSObject protocol and class (which isn't allowed in Java). As a result, 
/// protocols are referenced using angle brackets as follows:
///
/// id <NSDraggingInfo> anyObjectThatImplementsNSDraggingInfo;
///
class ObjcProtocolDecl : public NamedDecl {
  /// referenced protocols
  ObjcProtocolDecl **ReferencedProtocols;  // Null if none
  int NumReferencedProtocols;  // -1 if none
  
  /// protocol instance methods
  ObjcMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined

  /// protocol class methods
  ObjcMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined

  bool isForwardProtoDecl; // declared with @protocol.
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
public:
  ObjcProtocolDecl(SourceLocation L, unsigned numRefProtos,
                   IdentifierInfo *Id, bool FD = false)
    : NamedDecl(ObjcProtocol, L, Id), 
      ReferencedProtocols(0), NumReferencedProtocols(-1),
      InstanceMethods(0), NumInstanceMethods(-1), 
      ClassMethods(0), NumClassMethods(-1),
      isForwardProtoDecl(FD) {
        AllocReferencedProtocols(numRefProtos);
      }
  void AllocReferencedProtocols(unsigned numRefProtos) {
    if (numRefProtos) {
      ReferencedProtocols = new ObjcProtocolDecl*[numRefProtos];
      memset(ReferencedProtocols, '\0', 
             numRefProtos*sizeof(ObjcProtocolDecl*));
      NumReferencedProtocols = numRefProtos;
    }    
  }
  void addMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                  ObjcMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
  
  void setReferencedProtocols(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjcProtocolDecl** getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  int getNumReferencedProtocols() const { return NumReferencedProtocols; }
  
  ObjcMethodDecl** getInstanceMethods() const { return InstanceMethods; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  
  ObjcMethodDecl** getClassMethods() const { return ClassMethods; }
  int getNumClassMethods() const { return NumClassMethods; }
  
  bool isForwardDecl() const { return isForwardProtoDecl; }
  void setForwardDecl(bool val) { isForwardProtoDecl = val; }

  // Location information, modeled after the Stmt API. 
  SourceLocation getLocStart() const { return getLocation(); } // '@'protocol
  SourceLocation getLocEnd() const { return EndLoc; }
  void setLocEnd(SourceLocation LE) { EndLoc = LE; };
  
  // We also need to record the @end location.
  SourceLocation getAtEndLoc() const { return AtEndLoc; }

  static bool classof(const Decl *D) { return D->getKind() == ObjcProtocol; }
  static bool classof(const ObjcProtocolDecl *D) { return true; }
};
  
/// ObjcClassDecl - Specifies a list of forward class declarations. For example:
///
/// @class NSCursor, NSImage, NSPasteboard, NSWindow;
///
class ObjcClassDecl : public Decl {
  ObjcInterfaceDecl **ForwardDecls;
  unsigned NumForwardDecls;
public:
  ObjcClassDecl(SourceLocation L, ObjcInterfaceDecl **Elts, unsigned nElts)
    : Decl(ObjcClass, L) { 
    if (nElts) {
      ForwardDecls = new ObjcInterfaceDecl*[nElts];
      memcpy(ForwardDecls, Elts, nElts*sizeof(ObjcInterfaceDecl*));
    } else {
      ForwardDecls = 0;
    }
    NumForwardDecls = nElts;
  }
  void setInterfaceDecl(unsigned idx, ObjcInterfaceDecl *OID) {
    assert(idx < NumForwardDecls && "index out of range");
    ForwardDecls[idx] = OID;
  }
  ObjcInterfaceDecl** getForwardDecls() const { return ForwardDecls; }
  int getNumForwardDecls() const { return NumForwardDecls; }
  
  static bool classof(const Decl *D) { return D->getKind() == ObjcClass; }
  static bool classof(const ObjcClassDecl *D) { return true; }
};

/// ObjcForwardProtocolDecl - Specifies a list of forward protocol declarations.
/// For example:
/// 
/// @protocol NSTextInput, NSChangeSpelling, NSDraggingInfo;
/// 
class ObjcForwardProtocolDecl : public Decl {
  ObjcProtocolDecl **ReferencedProtocols;
  unsigned NumReferencedProtocols;
public:
  ObjcForwardProtocolDecl(SourceLocation L, 
                          ObjcProtocolDecl **Elts, unsigned nElts)
  : Decl(ObjcForwardProtocol, L) { 
    NumReferencedProtocols = nElts;
    if (nElts) {
      ReferencedProtocols = new ObjcProtocolDecl*[nElts];
      memcpy(ReferencedProtocols, Elts, nElts*sizeof(ObjcProtocolDecl*));
    } else {
      ReferencedProtocols = 0;
    }
  }
  void setForwardProtocolDecl(unsigned idx, ObjcProtocolDecl *OID) {
    assert(idx < NumReferencedProtocols && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  unsigned getNumForwardDecls() const { return NumReferencedProtocols; }
  
  ObjcProtocolDecl *getForwardProtocolDecl(unsigned idx) {
    assert(idx < NumReferencedProtocols && "index out of range");
    return ReferencedProtocols[idx];
  }
  const ObjcProtocolDecl *getForwardProtocolDecl(unsigned idx) const {
    assert(idx < NumReferencedProtocols && "index out of range");
    return ReferencedProtocols[idx];
  }
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjcForwardProtocol;
  }
  static bool classof(const ObjcForwardProtocolDecl *D) { return true; }
};

/// ObjcCategoryDecl - Represents a category declaration. A category allows
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
class ObjcCategoryDecl : public NamedDecl {
  /// Interface belonging to this category
  ObjcInterfaceDecl *ClassInterface;
  
  /// referenced protocols in this category
  ObjcProtocolDecl **ReferencedProtocols;  // Null if none
  int NumReferencedProtocols;  // -1 if none
  
  /// category instance methods
  ObjcMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined

  /// category class methods
  ObjcMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined
  
  /// Next category belonging to this class
  ObjcCategoryDecl *NextClassCategory;
  
  SourceLocation EndLoc; // marks the '>' or identifier.
  SourceLocation AtEndLoc; // marks the end of the entire interface.
public:
  ObjcCategoryDecl(SourceLocation L, unsigned numRefProtocol,IdentifierInfo *Id)
    : NamedDecl(ObjcCategory, L, Id),
      ClassInterface(0), ReferencedProtocols(0), NumReferencedProtocols(-1),
      InstanceMethods(0), NumInstanceMethods(-1),
      ClassMethods(0), NumClassMethods(-1),
      NextClassCategory(0) {
        if (numRefProtocol) {
          ReferencedProtocols = new ObjcProtocolDecl*[numRefProtocol];
          memset(ReferencedProtocols, '\0', 
                 numRefProtocol*sizeof(ObjcProtocolDecl*));
          NumReferencedProtocols = numRefProtocol;
        }
      }

  ObjcInterfaceDecl *getClassInterface() const { return ClassInterface; }
  void setClassInterface(ObjcInterfaceDecl *IDecl) { ClassInterface = IDecl; }
  
  void setCatReferencedProtocols(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjcProtocolDecl **getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  int getNumReferencedProtocols() const { return NumReferencedProtocols; }
  
  ObjcMethodDecl **getInstanceMethods() const { return InstanceMethods; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  
  ObjcMethodDecl **getClassMethods() const { return ClassMethods; }
  int getNumClassMethods() const { return NumClassMethods; }
  
  void addMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                  ObjcMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
  
  ObjcCategoryDecl *getNextClassCategory() const { return NextClassCategory; }
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
  
  static bool classof(const Decl *D) { return D->getKind() == ObjcCategory; }
  static bool classof(const ObjcCategoryDecl *D) { return true; }
};

/// ObjcCategoryImplDecl - An object of this class encapsulates a category 
/// @implementation declaration.
class ObjcCategoryImplDecl : public NamedDecl {
  /// Class interface for this category implementation
  ObjcInterfaceDecl *ClassInterface;

  /// category instance methods being implemented
  ObjcMethodDecl **InstanceMethods; // Null if category is not implementing any
  int NumInstanceMethods;           // -1 if category is not implementing any
  
  /// category class methods being implemented
  ObjcMethodDecl **ClassMethods; // Null if category is not implementing any
  int NumClassMethods;  // -1 if category is not implementing any
  
  public:
    ObjcCategoryImplDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjcInterfaceDecl *classInterface)
    : NamedDecl(ObjcCategoryImpl, L, Id),
    ClassInterface(classInterface),
    InstanceMethods(0), NumInstanceMethods(-1),
    ClassMethods(0), NumClassMethods(-1) {}
        
    ObjcInterfaceDecl *getClassInterface() const { 
      return ClassInterface; 
    }
  
  ObjcMethodDecl **getInstanceMethods() const { return InstanceMethods; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  
  ObjcMethodDecl **getClassMethods() const { return ClassMethods; }
  int getNumClassMethods() const { return NumClassMethods; }
  
  void addMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                  ObjcMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
  
  static bool classof(const Decl *D) { return D->getKind() == ObjcCategoryImpl;}
  static bool classof(const ObjcCategoryImplDecl *D) { return true; }
};

/// ObjcImplementationDecl - Represents a class definition - this is where
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
class ObjcImplementationDecl : public NamedDecl {
  /// Class interface for this category implementation
  ObjcInterfaceDecl *ClassInterface;
  
  /// Implementation Class's super class.
  ObjcInterfaceDecl *SuperClass;
    
  /// Optional Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjcIvarDecl **Ivars;   // Null if not specified
  int NumIvars;   // -1 if not defined.
    
  /// implemented instance methods
  ObjcMethodDecl **InstanceMethods;  // Null if not defined
  int NumInstanceMethods;  // -1 if not defined
    
  /// implemented class methods
  ObjcMethodDecl **ClassMethods;  // Null if not defined
  int NumClassMethods;  // -1 if not defined
    
public:
  ObjcImplementationDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjcInterfaceDecl *classInterface,
                         ObjcInterfaceDecl *superDecl)
    : NamedDecl(ObjcImplementation, L, Id),
      ClassInterface(classInterface),
      SuperClass(superDecl),
      Ivars(0), NumIvars(-1),
      InstanceMethods(0), NumInstanceMethods(-1), 
      ClassMethods(0), NumClassMethods(-1) {}
  
  void ObjcAddInstanceVariablesToClassImpl(ObjcIvarDecl **ivars, 
                                           unsigned numIvars);
    
  void addMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                  ObjcMethodDecl **clsMethods, unsigned numClsMembers,
                  SourceLocation AtEndLoc);
    
  ObjcInterfaceDecl *getClassInterface() const { return ClassInterface; }
  ObjcInterfaceDecl *getSuperClass() const { return SuperClass; }
  
  void setSuperClass(ObjcInterfaceDecl * superCls) 
         { SuperClass = superCls; }
  
  ObjcMethodDecl **getInstanceMethods() const { return InstanceMethods; }
  int getNumInstanceMethods() const { return NumInstanceMethods; }
  
  ObjcMethodDecl **getClassMethods() const { return ClassMethods; }
  int getNumClassMethods() const { return NumClassMethods; }
  
  ObjcIvarDecl **getImplDeclIVars() const { return Ivars; }
  int getImplDeclNumIvars() const { return NumIvars; }
    
  static bool classof(const Decl *D) {
    return D->getKind() == ObjcImplementation;
  }
  static bool classof(const ObjcImplementationDecl *D) { return true; }
};

/// ObjcCompatibleAliasDecl - Represents alias of a class. This alias is 
/// declared as @compatibility_alias alias class.
class ObjcCompatibleAliasDecl : public ScopedDecl {
  /// Class that this is an alias of.
  ObjcInterfaceDecl *AliasedClass;
  
public:
  ObjcCompatibleAliasDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjcInterfaceDecl* aliasedClass)
  : ScopedDecl(CompatibleAlias, L, Id, 0),
  AliasedClass(aliasedClass) {}
  
  ObjcInterfaceDecl *getClassInterface() const { return AliasedClass; }
  
  static bool classof(const Decl *D) {
    return D->getKind() == CompatibleAlias;
  }
  static bool classof(const ObjcCompatibleAliasDecl *D) { return true; }
  
};
  
class ObjcPropertyDecl : public Decl {
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
  ObjcIvarDecl **PropertyDecls;
  int NumPropertyDecls;
  
  PropertyAttributeKind PropertyAttributes : 8;
  
  IdentifierInfo *GetterName;    // getter name of NULL if no getter
  IdentifierInfo *SetterName;    // setter name of NULL if no setter
  
public:
  ObjcPropertyDecl(SourceLocation L)
  : Decl(PropertyDecl, L),
  PropertyDecls(0), NumPropertyDecls(-1), PropertyAttributes(OBJC_PR_noattr),
  GetterName(0), SetterName(0) {}
  
  ObjcIvarDecl **const getPropertyDecls() const { return PropertyDecls; }
  void setPropertyDecls(ObjcIvarDecl **property) { PropertyDecls = property; }
  
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
  static bool classof(const ObjcPropertyDecl *D) { return true; }
};

}  // end namespace clang
#endif
