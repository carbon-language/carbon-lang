; This testcase used to fail due to a lack of this diff in Value.cpp:
; diff -r1.16 Value.cpp
; 11c11
; < #include "llvm/Type.h"
; ---
; > #include "llvm/DerivedTypes.h"
; 74c74,76
; <   assert(Ty.get() == (const Type*)OldTy &&"Can't refine anything but my type!");
; ---
; >   assert(Ty.get() == OldTy &&"Can't refine anything but my type!");
; >   if (OldTy == NewTy && !OldTy->isAbstract())
; >     Ty.removeUserFromConcrete();
;
; This was causing an assertion failure, due to the "foo" Method object never releasing
; it's reference to the opaque %bb value.
;
declare void "foo"(%bb)

%exception_descriptor = type opaque  

%bb = type int

%exception_descriptor = type int

implementation

