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

namespace clang {
class Expr;
class Stmt;
class FunctionDecl;
class AttributeList;
class ObjcIvarDecl;
class ObjcMethodDecl;
class ObjcProtocolDecl;
class ObjcCategoryDecl;

/// ObjcInterfaceDecl - Represents an ObjC class declaration. For example:
///
///   // MostPrimitive declares no super class (not particularly useful).
///   @interface MostPrimitive 
///     // no instance variables or methods.
///   @end
///
///   // NSResponder inherits from NSObject and implements NSCoding (a protocol). 
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
  ObjcProtocolDecl **IntfRefProtocols;  // Null if none
  int NumIntfRefProtocols;  // -1 if none
  
  /// Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjcIvarDecl **Ivars;   // Null if not defined.
  int NumIvars;   // -1 if not defined.
  
  /// instance methods
  ObjcMethodDecl **InsMethods;  // Null if not defined
  int NumInsMethods;  // -1 if not defined
  
  /// class methods
  ObjcMethodDecl **ClsMethods;  // Null if not defined
  int NumClsMethods;  // -1 if not defined
  
  /// List of categories defined for this class.
  ObjcCategoryDecl *ListCategories;
  
  bool isForwardDecl; // declared with @class.
public:
  ObjcInterfaceDecl(SourceLocation L, unsigned numRefProtos,
                    IdentifierInfo *Id, bool FD = false)
    : TypeDecl(ObjcInterface, L, Id, 0),
      SuperClass(0),
      IntfRefProtocols(0), NumIntfRefProtocols(-1),
      Ivars(0), NumIvars(-1),
      InsMethods(0), NumInsMethods(-1), ClsMethods(0), NumClsMethods(-1),
      ListCategories(0),
      isForwardDecl(FD) {
        AllocIntfRefProtocols(numRefProtos);
      }
    
  void AllocIntfRefProtocols(unsigned numRefProtos) {
    if (numRefProtos) {
      IntfRefProtocols = new ObjcProtocolDecl*[numRefProtos];
      memset(IntfRefProtocols, '\0',
             numRefProtos*sizeof(ObjcProtocolDecl*));
      NumIntfRefProtocols = numRefProtos;
    }
  }
  
  ObjcProtocolDecl **getIntfRefProtocols() const { return IntfRefProtocols; }
  int getNumIntfRefProtocols() const { return NumIntfRefProtocols; }
  
  ObjcIvarDecl **getIntfDeclIvars() const { return Ivars; }
  int getIntfDeclNumIvars() const { return NumIvars; }
  
  ObjcMethodDecl** getInsMethods() const { return InsMethods; }
  int getNumInsMethods() const { return NumInsMethods; }
  
  ObjcMethodDecl** getClsMethods() const { return ClsMethods; }
  int getNumClsMethods() const { return NumClsMethods; }
  
  void ObjcAddInstanceVariablesToClass(ObjcIvarDecl **ivars, 
				       unsigned numIvars);

  void ObjcAddMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                      ObjcMethodDecl **clsMethods, unsigned numClsMembers);
  
  bool getIsForwardDecl() const { return isForwardDecl; }
  void setIsForwardDecl(bool val) { isForwardDecl = val; }
  
  void setIntfRefProtocols(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumIntfRefProtocols) && "index out of range");
    IntfRefProtocols[idx] = OID;
  }
  
  ObjcInterfaceDecl *getSuperClass() const { return SuperClass; }
  void setSuperClass(ObjcInterfaceDecl * superCls) { SuperClass = superCls; }
  
  ObjcCategoryDecl* getListCategories() const { return ListCategories; }
  void setListCategories(ObjcCategoryDecl *category) { 
         ListCategories = category; 
  }
  ObjcMethodDecl *lookupInstanceMethod(Selector &Sel);
  ObjcMethodDecl *lookupClassMethod(Selector &Sel);
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjcInterface;
  }
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
  
  /// Loc - location of this declaration.
  SourceLocation Loc;

public:
  ObjcMethodDecl(SourceLocation L, Selector SelInfo, QualType T,
                 ParmVarDecl **paramInfo = 0, int numParams=-1,
                 AttributeList *M = 0, bool isInstance = true,
                 ImplementationControl impControl = None,
                 Decl *PrevDecl = 0)
    : Decl(ObjcMethod),
      IsInstance(isInstance), DeclImplementation(impControl),
      SelName(SelInfo), MethodDeclType(T), 
      ParamInfo(paramInfo), NumMethodParams(numParams),
      MethodAttrs(M), Loc(L) {}
  virtual ~ObjcMethodDecl();
  Selector getSelector() const { return SelName; }
  QualType getMethodType() const { return MethodDeclType; }
  unsigned getNumMethodParams() const { return NumMethodParams; }
  ParmVarDecl *getMethodParamDecl(unsigned i) {
    assert(i < getNumMethodParams() && "Illegal param #");
    return ParamInfo[i];
  }
  void setMethodParams(ParmVarDecl **NewParamInfo, unsigned NumParams);

  AttributeList *getMethodAttrs() const {return MethodAttrs;}
  SourceLocation getLocation() const { return Loc; }
  bool isInstance() const { return IsInstance; }
  // Related to protocols declared in  @protocol
  void setDeclImplementation(ImplementationControl ic)
         { DeclImplementation = ic; }
  ImplementationControl  getImplementationControl() const
                           { return DeclImplementation; }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { 
    return D->getKind() == ObjcMethod; 
  }
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
class ObjcProtocolDecl : public TypeDecl {
  /// referenced protocols
  ObjcProtocolDecl **ReferencedProtocols;  // Null if none
  int NumReferencedProtocols;  // -1 if none
  
  /// protocol instance methods
  ObjcMethodDecl **ProtoInsMethods;  // Null if not defined
  int NumProtoInsMethods;  // -1 if not defined

  /// protocol class methods
  ObjcMethodDecl **ProtoClsMethods;  // Null if not defined
  int NumProtoClsMethods;  // -1 if not defined

  bool isForwardProtoDecl; // declared with @protocol.
public:
  ObjcProtocolDecl(SourceLocation L, unsigned numRefProtos,
                   IdentifierInfo *Id, bool FD = false)
    : TypeDecl(ObjcProtocol, L, Id, 0), 
      ReferencedProtocols(0), NumReferencedProtocols(-1),
      ProtoInsMethods(0), NumProtoInsMethods(-1), 
      ProtoClsMethods(0), NumProtoClsMethods(-1),
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
  void ObjcAddProtoMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                           ObjcMethodDecl **clsMethods, unsigned numClsMembers);
  
  void setReferencedProtocols(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumReferencedProtocols) && "index out of range");
    ReferencedProtocols[idx] = OID;
  }
  
  ObjcProtocolDecl** getReferencedProtocols() const { 
    return ReferencedProtocols; 
  }
  int getNumReferencedProtocols() const { return NumReferencedProtocols; }
  
  ObjcMethodDecl** getInsMethods() const { return ProtoInsMethods; }
  int getNumInsMethods() const { return NumProtoInsMethods; }
  
  ObjcMethodDecl** getClsMethods() const { return ProtoClsMethods; }
  int getNumClsMethods() const { return NumProtoClsMethods; }
  
  bool getIsForwardProtoDecl() const { return isForwardProtoDecl; }
  void setIsForwardProtoDecl(bool val) { isForwardProtoDecl = val; }

  static bool classof(const Decl *D) {
    return D->getKind() == ObjcProtocol;
  }
  static bool classof(const ObjcProtocolDecl *D) { return true; }
};
  
/// ObjcClassDecl - Specifies a list of forward class declarations. For example:
///
/// @class NSCursor, NSImage, NSPasteboard, NSWindow;
///
class ObjcClassDecl : public TypeDecl {
  ObjcInterfaceDecl **ForwardDecls;   // Null if not defined.
  int NumForwardDecls;               // -1 if not defined.
public:
  ObjcClassDecl(SourceLocation L, unsigned nElts)
    : TypeDecl(ObjcClass, L, 0, 0) { 
    if (nElts) {
      ForwardDecls = new ObjcInterfaceDecl*[nElts];
      memset(ForwardDecls, '\0', nElts*sizeof(ObjcInterfaceDecl*));
    }
    NumForwardDecls = nElts;
  }
  void setInterfaceDecl(int idx, ObjcInterfaceDecl *OID) {
    assert((idx < NumForwardDecls) && "index out of range");
    ForwardDecls[idx] = OID;
  }
  static bool classof(const Decl *D) {
    return D->getKind() == ObjcClass;
  }
  static bool classof(const ObjcClassDecl *D) { return true; }
};

/// ObjcForwardProtocolDecl - Specifies a list of forward protocol declarations.
/// For example:
/// 
/// @protocol NSTextInput, NSChangeSpelling, NSDraggingInfo;
/// 
class ObjcForwardProtocolDecl : public TypeDecl {
  ObjcProtocolDecl **ForwardProtocolDecls;   // Null if not defined.
  int NumForwardProtocolDecls;               // -1 if not defined.
public:
  ObjcForwardProtocolDecl(SourceLocation L, unsigned nElts)
  : TypeDecl(ObjcForwardProtocol, L, 0, 0) { 
    if (nElts) {
      ForwardProtocolDecls = new ObjcProtocolDecl*[nElts];
      memset(ForwardProtocolDecls, '\0', nElts*sizeof(ObjcProtocolDecl*));
      NumForwardProtocolDecls = nElts;
    }
  }
  void setForwardProtocolDecl(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumForwardProtocolDecls) && "index out of range");
    ForwardProtocolDecls[idx] = OID;
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
class ObjcCategoryDecl : public Decl {
  /// Interface belonging to this category
  ObjcInterfaceDecl *ClassInterface;
  
  /// Category name
  IdentifierInfo *ObjcCatName;
  
  /// referenced protocols in this category
  ObjcProtocolDecl **CatReferencedProtocols;  // Null if none
  int NumCatReferencedProtocols;  // -1 if none
  
  /// category instance methods
  ObjcMethodDecl **CatInsMethods;  // Null if not defined
  int NumCatInsMethods;  // -1 if not defined

  /// category class methods
  ObjcMethodDecl **CatClsMethods;  // Null if not defined
  int NumCatClsMethods;  // -1 if not defined
  
  /// Next category belonging to this class
  ObjcCategoryDecl *NextClassCategory;

public:
  ObjcCategoryDecl(SourceLocation L, unsigned numRefProtocol)
    : Decl(ObjcCategory),
      ClassInterface(0), ObjcCatName(0),
      CatReferencedProtocols(0), NumCatReferencedProtocols(-1),
      CatInsMethods(0), NumCatInsMethods(-1),
      CatClsMethods(0), NumCatClsMethods(-1),
      NextClassCategory(0) {
        if (numRefProtocol) {
          CatReferencedProtocols = new ObjcProtocolDecl*[numRefProtocol];
          memset(CatReferencedProtocols, '\0', 
                 numRefProtocol*sizeof(ObjcProtocolDecl*));
          NumCatReferencedProtocols = numRefProtocol;
        }
      }

  ObjcInterfaceDecl *getClassInterface() const { return ClassInterface; }
  void setClassInterface(ObjcInterfaceDecl *IDecl) { ClassInterface = IDecl; }
  
  void setCatReferencedProtocols(int idx, ObjcProtocolDecl *OID) {
    assert((idx < NumCatReferencedProtocols) && "index out of range");
    CatReferencedProtocols[idx] = OID;
  }
  
  ObjcProtocolDecl **getCatReferencedProtocols() const { 
    return CatReferencedProtocols; 
  }
  int getNumCatReferencedProtocols() const { return NumCatReferencedProtocols; }
  
  ObjcMethodDecl **getCatInsMethods() const { return CatInsMethods; }
  int getNumCatInsMethods() const { return NumCatInsMethods; }
  
  ObjcMethodDecl **getCatClsMethods() const { return CatClsMethods; }
  int getNumCatClsMethods() const { return NumCatClsMethods; }
  
  void ObjcAddCatMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                         ObjcMethodDecl **clsMethods, unsigned numClsMembers);
  
  IdentifierInfo *getCatName() const { return ObjcCatName; }
  void setCatName(IdentifierInfo *catName) { ObjcCatName = catName; }
  
  ObjcCategoryDecl *getNextClassCategory() const { return NextClassCategory; }
  void insertNextClassCategory() {
    NextClassCategory = ClassInterface->getListCategories();
    ClassInterface->setListCategories(this);
  }

  static bool classof(const Decl *D) {
    return D->getKind() == ObjcCategory;
  }
  static bool classof(const ObjcCategoryDecl *D) { return true; }
};

/// ObjcCategoryImplDecl - An object of this class encapsulates a category 
/// @implementation declaration.
class ObjcCategoryImplDecl : public Decl {
  /// Class interface for this category implementation
  ObjcInterfaceDecl *ClassInterface;

  /// Category name
  IdentifierInfo *ObjcCatName;
      
  /// category instance methods being implemented
  ObjcMethodDecl **CatInsMethods; // Null if category is not implementing any
  int NumCatInsMethods;           // -1 if category is not implementing any
  
  /// category class methods being implemented
  ObjcMethodDecl **CatClsMethods; // Null if category is not implementing any
  int NumCatClsMethods;  // -1 if category is not implementing any
  
  public:
    ObjcCategoryImplDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjcInterfaceDecl *classInterface,
                         IdentifierInfo *catName)
    : Decl(ObjcCategoryImpl),
    ClassInterface(classInterface),
    ObjcCatName(catName),
    CatInsMethods(0), NumCatInsMethods(-1),
    CatClsMethods(0), NumCatClsMethods(-1) {}
        
    ObjcInterfaceDecl *getClassInterface() const { 
      return ClassInterface; 
    }
  
  IdentifierInfo *getObjcCatName() const { return ObjcCatName; }
  
  ObjcMethodDecl **getCatInsMethods() const { return CatInsMethods; }
  int getNumCatInsMethods() const { return NumCatInsMethods; }
  
  ObjcMethodDecl **getCatClsMethods() const { return CatClsMethods; }
  int getNumCatClsMethods() const { return NumCatClsMethods; }
  
  
  void ObjcAddCatImplMethods(
        ObjcMethodDecl **insMethods, unsigned numInsMembers,
        ObjcMethodDecl **clsMethods, unsigned numClsMembers);
  
  static bool classof(const Decl *D) {
    return D->getKind() == ObjcCategoryImpl;
  }
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
class ObjcImplementationDecl : public TypeDecl {
    
  /// Implementation Class's super class.
  ObjcInterfaceDecl *SuperClass;
    
  /// Optional Ivars/NumIvars - This is a new[]'d array of pointers to Decls.
  ObjcIvarDecl **Ivars;   // Null if not specified
  int NumIvars;   // -1 if not defined.
    
  /// implemented instance methods
  ObjcMethodDecl **InsMethods;  // Null if not defined
  int NumInsMethods;  // -1 if not defined
    
  /// implemented class methods
  ObjcMethodDecl **ClsMethods;  // Null if not defined
  int NumClsMethods;  // -1 if not defined
    
  public:
  ObjcImplementationDecl(SourceLocation L, IdentifierInfo *Id,
                         ObjcInterfaceDecl* superDecl)
    : TypeDecl(ObjcImplementation, L, Id, 0),
      SuperClass(superDecl),
      Ivars(0), NumIvars(-1),
      InsMethods(0), NumInsMethods(-1), ClsMethods(0), NumClsMethods(-1) {}
  
  void ObjcAddInstanceVariablesToClassImpl(ObjcIvarDecl **ivars, 
                                           unsigned numIvars);
    
  void ObjcAddImplMethods(ObjcMethodDecl **insMethods, unsigned numInsMembers,
                          ObjcMethodDecl **clsMethods, unsigned numClsMembers);
    
  ObjcInterfaceDecl *getImplSuperClass() const { return SuperClass; }
  
  void setImplSuperClass(ObjcInterfaceDecl * superCls) 
         { SuperClass = superCls; }
  
  ObjcMethodDecl **getInsMethods() const { return InsMethods; }
  int getNumInsMethods() const { return NumInsMethods; }
  
  ObjcMethodDecl **getClsMethods() const { return ClsMethods; }
  int getNumClsMethods() const { return NumClsMethods; }
    
  static bool classof(const Decl *D) {
    return D->getKind() == ObjcImplementation;
  }
  static bool classof(const ObjcImplementationDecl *D) { return true; }
};
  

}  // end namespace clang
#endif
